// luma.gl
// SPDX-License-Identifier: MIT AND BSD-2-Clause
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) 2016, Mapbox

import Supercluster from 'supercluster';
import {scaleOrdinal, scaleQuantile, scaleQuantize, scaleSqrt, scaleThreshold} from 'd3-scale';
import {getViewportBounds} from './viewport-bounds';

/** Native supercluster maxZoom in @kepler.gl/deckgl-layers cluster-utils. */
const CLUSTER_MAXIMUM_ZOOM = 20;

/** CPU copy of one source revision. Positions are interleaved longitude/latitude. */
export type KeplerClusterSource = {
  positions: Float32Array;
  /** Native point filters remove rows before clustering; omitted keeps every row. */
  selectionMask?: Uint32Array;
};

export type KeplerClusterView = {
  /** Deck viewport center and size; zoom is rounded exactly as native Math.round(mapState.zoom). */
  longitude: number;
  latitude: number;
  zoom: number;
  width: number;
  height: number;
};

export type KeplerClusterSettings = {
  clusterRadius: number;
  minClusterSize: number;
  /** Native clusterRadiusRange, in pixels. */
  radiusRange: readonly [number, number];
  /** Native default color measure is the member count. Otherwise supply the aggregate of rows. */
  getColorValue?: (memberRows: readonly number[]) => number;
  /** 'threshold' covers native jenks and custom, whose breaks come from getThresholds. */
  colorScaleType?: 'quantize' | 'quantile' | 'ordinal' | 'threshold';
  /** Receives the ascending finite aggregated values; returns the native jenks or custom breaks. */
  getThresholds?: (sortedValues: readonly number[]) => readonly number[];
  /** RGB or RGBA entries from the native color range. */
  colorRange: readonly (readonly number[])[];
};

export type KeplerCluster = {
  /** Source row indices in supercluster merge order. */
  memberRows: readonly number[];
  longitude: number;
  latitude: number;
  index: number;
};

export type KeplerClusterIndex = {
  readonly clusterer: Supercluster;
  readonly clusterRadius: number;
  readonly rowCount: number;
};

/** Persistent concatenation of member rows: merge order is preserved without quadratic copies. */
type MemberRope = number[] | {left: MemberRope; right: MemberRope};
type ClusterProperties = {points: MemberRope};

/**
 * Native getGeoJSON: filtered rows with finite positions become point features of row indices.
 * Features are slimmer than native (the unused points/point_count fields are omitted); the map and
 * reduce callbacks below reproduce the native member order.
 */
function getFeatures(source: KeplerClusterSource) {
  const rowCount = source.positions.length / 2;
  const features: Supercluster.PointFeature<{data: number}>[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (source.selectionMask && source.selectionMask[row] === 0) continue;
    const longitude = source.positions[row * 2];
    const latitude = source.positions[row * 2 + 1];
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) continue;
    features.push({
      type: 'Feature',
      properties: {data: row},
      geometry: {type: 'Point', coordinates: [longitude, latitude]}
    });
  }
  return features;
}

/**
 * Builds the index with the same supercluster options as native cluster-utils getClusterer
 * (maxZoom, radius, default extent and minPoints). Native accumulates member rows by array spread,
 * which is quadratic for large clusters; the rope reduce yields the identical member order.
 */
export function buildKeplerClusterIndex(
  source: KeplerClusterSource,
  clusterRadius: number
): KeplerClusterIndex {
  const rowCount = source.positions.length / 2;
  const clusterer = new Supercluster<{data: number}, ClusterProperties>({
    maxZoom: CLUSTER_MAXIMUM_ZOOM,
    radius: clusterRadius,
    reduce: (accumulated, properties) => {
      accumulated.points = {left: accumulated.points, right: properties.points};
    },
    map: properties => ({points: [properties.data]})
  }).load(getFeatures(source));
  return {clusterer: clusterer as unknown as Supercluster, clusterRadius, rowCount};
}

/** Supercluster output properties: a single point carries data, a cluster the member rope. */
type QueriedProperties = {
  cluster?: boolean;
  point_count: number;
  data: number;
  points: MemberRope;
};

/** Clusters of one view as flat typed arrays, ready to transfer between threads. */
export type KeplerClusterRanges = {
  clusterCount: number;
  /** Interleaved longitude/latitude in float64, as native reports them. */
  centers: Float64Array;
  /** clusterCount + 1 offsets into rows. */
  offsets: Uint32Array;
  /** Member rows of every cluster in supercluster merge order. */
  rows: Uint32Array;
};

/** Native clustersAtZoom: geo-viewport bounds at the rounded zoom, then the minimum-size cutoff. */
export function queryKeplerClusterRanges(
  index: KeplerClusterIndex,
  view: KeplerClusterView,
  minClusterSize: number
): KeplerClusterRanges {
  const zoom = Math.round(view.zoom);
  const boundingBox = getViewportBounds([view.longitude, view.latitude], zoom, [
    view.width,
    view.height
  ]);
  const cutoff = Number.isFinite(minClusterSize) ? minClusterSize : 1;
  const kept: {properties: QueriedProperties; coordinates: number[]; size: number}[] = [];
  let totalRows = 0;
  for (const feature of index.clusterer.getClusters(boundingBox, zoom)) {
    const properties = feature.properties as unknown as QueriedProperties;
    const size = properties.cluster ? properties.point_count : 1;
    if (size < cutoff) continue;
    kept.push({properties, coordinates: feature.geometry.coordinates, size});
    totalRows += size;
  }
  const centers = new Float64Array(kept.length * 2);
  const offsets = new Uint32Array(kept.length + 1);
  const rows = new Uint32Array(totalRows);
  let cursor = 0;
  const stack: MemberRope[] = [];
  for (const [clusterIndex, entry] of kept.entries()) {
    centers[clusterIndex * 2] = entry.coordinates[0];
    centers[clusterIndex * 2 + 1] = entry.coordinates[1];
    offsets[clusterIndex] = cursor;
    if (!entry.properties.cluster) {
      rows[cursor++] = entry.properties.data;
      continue;
    }
    // Depth-first, left before right: a left-deep rope of 100k merges must not recurse.
    stack.push(entry.properties.points);
    while (stack.length) {
      const node = stack.pop() as MemberRope;
      if (Array.isArray(node)) for (const row of node) rows[cursor++] = row;
      else stack.push(node.right, node.left);
    }
  }
  offsets[kept.length] = cursor;
  return {clusterCount: kept.length, centers, offsets, rows};
}

/** Native clustersAtZoom with member rows as arrays; prefer {@link queryKeplerClusterRanges}. */
export function queryKeplerClusters(
  index: KeplerClusterIndex,
  view: KeplerClusterView,
  minClusterSize: number
): KeplerCluster[] {
  const ranges = queryKeplerClusterRanges(index, view, minClusterSize);
  const clusters: KeplerCluster[] = [];
  for (let index_ = 0; index_ < ranges.clusterCount; index_++)
    clusters.push({
      memberRows: Array.from(
        ranges.rows.subarray(ranges.offsets[index_], ranges.offsets[index_ + 1])
      ),
      longitude: ranges.centers[index_ * 2],
      latitude: ranges.centers[index_ * 2 + 1],
      index: index_
    });
  return clusters;
}

export type KeplerClusterAppearance = {
  /** Interleaved longitude/latitude for the small GPU upload. */
  centers: Float32Array;
  counts: Uint32Array;
  /** Pixel radii after the native sqrt scale. */
  radii: Float32Array;
  /** Packed RGBA bytes. */
  colors: Uint8Array;
  colorValues: Float32Array;
};

/** One cluster reduced to the values appearance needs; no member rows are required. */
export type KeplerClusterSummary = {
  count: number;
  /** Native color measure: the member count, or the aggregated field value. */
  colorValue: number;
  longitude: number;
  latitude: number;
};

export type KeplerClusterAppearanceSettings = Omit<KeplerClusterSettings, 'getColorValue'>;

/**
 * Native appearance: radius is scaleSqrt over [0, maxCount]; color is a quantize/quantile scale
 * over the sorted aggregated values (domain from the min/max or all values) mapped onto the range.
 */
export function computeKeplerClusterAppearance(
  clusters: readonly KeplerCluster[],
  settings: KeplerClusterSettings
): KeplerClusterAppearance {
  const getColorValue = settings.getColorValue ?? ((rows: readonly number[]) => rows.length);
  return computeKeplerClusterSummaryAppearance(
    clusters.map(cluster => ({
      count: cluster.memberRows.length,
      colorValue: getColorValue(cluster.memberRows),
      longitude: cluster.longitude,
      latitude: cluster.latitude
    })),
    settings
  );
}

/** Same appearance as {@link computeKeplerClusterAppearance}, from precomputed cluster summaries. */
export function computeKeplerClusterSummaryAppearance(
  summaries: readonly KeplerClusterSummary[],
  settings: KeplerClusterAppearanceSettings
): KeplerClusterAppearance {
  const count = summaries.length;
  const counts = new Uint32Array(count);
  const colorValues = new Float32Array(count);
  let maximumCount = 0;
  for (const [index, summary] of summaries.entries()) {
    counts[index] = summary.count;
    maximumCount = Math.max(maximumCount, summary.count);
    colorValues[index] = summary.colorValue;
  }
  const radiusScale = scaleSqrt()
    .domain([0, maximumCount])
    .range([...settings.radiusRange]);
  // Native BinSorter ignores empty bins; non-finite aggregates cannot form a domain.
  const sorted = [...colorValues]
    .filter(value => Number.isFinite(value))
    .sort((left, right) => left - right);
  const palette = settings.colorRange.map(color => [...color]);
  let colorScale: (value: number) => number[] | undefined;
  switch (settings.colorScaleType) {
    case 'quantile':
      colorScale = scaleQuantile<number[]>().domain(sorted).range(palette);
      break;
    case 'ordinal':
      colorScale = scaleOrdinal<number, number[]>()
        .domain([...new Set(sorted)])
        .range(palette);
      break;
    case 'threshold':
      colorScale = scaleThreshold<number, number[]>()
        .domain([...(settings.getThresholds?.(sorted) ?? [])])
        .range(palette);
      break;
    default:
      colorScale = scaleQuantize<number[]>()
        .domain(sorted.length ? [sorted[0], sorted[sorted.length - 1]] : [0, 0])
        .range(palette);
  }
  const centers = new Float32Array(count * 2);
  const radii = new Float32Array(count);
  const colors = new Uint8Array(count * 4);
  for (const [index, summary] of summaries.entries()) {
    centers[index * 2] = summary.longitude;
    centers[index * 2 + 1] = summary.latitude;
    radii[index] = radiusScale(counts[index]);
    const color = colorScale(colorValues[index]) ?? [0, 0, 0];
    colors.set([color[0], color[1], color[2], color[3] ?? 255], index * 4);
  }
  return {centers, counts, radii, colors, colorValues};
}

/** Bounded picking: member rows are listed in merge order, truncated at the limit. */
export function listKeplerClusterMembers(
  cluster: KeplerCluster,
  limit: number
): {rows: readonly number[]; total: number; truncated: boolean} {
  const total = cluster.memberRows.length;
  return {rows: cluster.memberRows.slice(0, Math.max(0, limit)), total, truncated: total > limit};
}
