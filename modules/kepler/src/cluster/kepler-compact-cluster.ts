// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerKDTree} from './kepler-kd-tree';
import {getViewportBounds} from './viewport-bounds';
import type {KeplerClusterRanges, KeplerClusterSource, KeplerClusterView} from './kepler-cluster';

/**
 * Typed-array supercluster. It reproduces supercluster 7 (maxZoom 20, extent 512, minPoints 2)
 * exactly, including cluster order, merge order of member rows and Float32 centroids, but keeps
 * no per-point JavaScript objects, no property ropes and no per-level KD trees. Each zoom level is
 * a few typed arrays; levels where no cluster merged are shared with the level below, and member
 * rows are walked through the hierarchy only for the clusters a view actually shows.
 */
const EXTENT = 512;
const MAXIMUM_ZOOM = 20;

/** Thrown by the builder when the retained hierarchy would exceed `maximumBytes`. */
export const KEPLER_CLUSTER_BUDGET_MESSAGE = 'The cluster hierarchy exceeds its memory budget.';

type Level = {
  /** Zoom this level was created for. */
  zoom: number;
  count: number;
  /** Mercator [0, 1] coordinates, Float32 like the supercluster KD-tree. */
  x: Float32Array;
  y: Float32Array;
  /** Member counts; absent on the leaf level where every node is one row. */
  numPoints?: Uint32Array;
  /** Leaf level only: source row of each node. */
  rows?: Uint32Array;
  /** The level below, which this one clustered. */
  source?: Level;
  /** Node of `source` that started each cluster (the cluster's first member group). */
  origin?: Uint32Array;
  /** count + 1 offsets into `children`: the merged neighbors of each node, in merge order. */
  childOffsets?: Uint32Array;
  children?: Uint32Array;
};

export type KeplerCompactClusterIndex = {
  /**
   * levels[zoom] for zoom 0..maxZoom + 1; entries may be shared. Coarse levels are built on
   * demand (a view at zoom z needs only levels z..maxZoom + 1), so unbuilt entries are undefined.
   */
  readonly levels: (Level | undefined)[];
  readonly positions: Float32Array;
  readonly clusterRadius: number;
  readonly rowCount: number;
  /** Bytes of the distinct retained levels built so far. */
  byteLength: number;
  /** Bound on `byteLength`; building another level past it throws the budget message. */
  readonly maximumBytes: number;
  /** Continuation of the build: the finest unbuilt zoom is `nextZoom`, clustered from `current`. */
  builder?: {current: Level; tree: KeplerKDTree; nextZoom: number};
  queryTree?: {level: Level; tree: KeplerKDTree};
};

function getLevelByteLength(level: Level): number {
  return (
    level.x.byteLength +
    level.y.byteLength +
    (level.numPoints?.byteLength ?? 0) +
    (level.rows?.byteLength ?? 0) +
    (level.origin?.byteLength ?? 0) +
    (level.childOffsets?.byteLength ?? 0) +
    (level.children?.byteLength ?? 0)
  );
}

function getLongitudeX(longitude: number): number {
  return longitude / 360 + 0.5;
}

function getLatitudeY(latitude: number): number {
  const sine = Math.sin((latitude * Math.PI) / 180);
  const y = 0.5 - (0.25 * Math.log((1 + sine) / (1 - sine))) / Math.PI;
  return y < 0 ? 0 : y > 1 ? 1 : y;
}

function getXLongitude(x: number): number {
  return (x - 0.5) * 360;
}

function getYLatitude(y: number): number {
  const angle = ((180 - y * 360) * Math.PI) / 180;
  return (360 * Math.atan(Math.exp(angle))) / Math.PI - 90;
}

/** Native getGeoJSON filtering: masked-out rows and non-finite positions are not clustered. */
function createLeafLevel(source: KeplerClusterSource): Level {
  const rowCount = source.positions.length / 2;
  let count = 0;
  const keep = new Uint8Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    if (source.selectionMask && source.selectionMask[row] === 0) continue;
    if (
      !Number.isFinite(source.positions[row * 2]) ||
      !Number.isFinite(source.positions[row * 2 + 1])
    )
      continue;
    keep[row] = 1;
    count++;
  }
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const rows = new Uint32Array(count);
  let node = 0;
  for (let row = 0; row < rowCount; row++) {
    if (!keep[row]) continue;
    x[node] = getLongitudeX(source.positions[row * 2]);
    y[node] = getLatitudeY(source.positions[row * 2 + 1]);
    rows[node++] = row;
  }
  return {zoom: MAXIMUM_ZOOM + 1, count, x, y, rows};
}

/** One supercluster `_cluster` pass; returns undefined when nothing merged (level is shared). */
function clusterLevel(
  source: Level,
  tree: KeplerKDTree,
  zoom: number,
  clusterRadius: number
): Level | undefined {
  const radius = clusterRadius / (EXTENT * Math.pow(2, zoom));
  const {count} = source;
  const visited = new Uint8Array(count);
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const numPoints = new Uint32Array(count);
  const origin = new Uint32Array(count);
  const childOffsets = new Uint32Array(count + 1);
  const children = new Uint32Array(count);
  const neighbors: number[] = [];
  let outputCount = 0;
  let childTotal = 0;
  let merged = false;
  for (let index = 0; index < count; index++) {
    if (visited[index]) continue;
    visited[index] = 1;
    tree.within(source.x[index], source.y[index], radius, neighbors);
    const originPoints = source.numPoints ? source.numPoints[index] : 1;
    let totalPoints = originPoints;
    for (const neighbor of neighbors)
      if (!visited[neighbor]) totalPoints += source.numPoints ? source.numPoints[neighbor] : 1;
    childOffsets[outputCount] = childTotal;
    origin[outputCount] = index;
    if (totalPoints > originPoints) {
      let weightedX = source.x[index] * originPoints;
      let weightedY = source.y[index] * originPoints;
      for (const neighbor of neighbors) {
        if (visited[neighbor]) continue;
        visited[neighbor] = 1;
        const weight = source.numPoints ? source.numPoints[neighbor] : 1;
        weightedX += source.x[neighbor] * weight;
        weightedY += source.y[neighbor] * weight;
        children[childTotal++] = neighbor;
      }
      x[outputCount] = weightedX / totalPoints;
      y[outputCount] = weightedY / totalPoints;
      merged = true;
    } else {
      x[outputCount] = source.x[index];
      y[outputCount] = source.y[index];
    }
    numPoints[outputCount] = totalPoints;
    outputCount++;
  }
  if (!merged) return undefined;
  childOffsets[outputCount] = childTotal;
  return {
    zoom,
    count: outputCount,
    x: x.slice(0, outputCount),
    y: y.slice(0, outputCount),
    numPoints: numPoints.slice(0, outputCount),
    source,
    origin: origin.slice(0, outputCount),
    childOffsets: childOffsets.slice(0, outputCount + 1),
    children: children.slice(0, childTotal)
  };
}

/**
 * Largest zoom whose clustering radius reaches a second node, or -1. A pair within the radius at
 * zoom z is within it at every coarser zoom, so a binary search finds where merging starts and
 * every finer level is shared with the leaf level exactly as an unmerged `clusterLevel` would be.
 */
function getFirstMergeZoom(leaf: Level, tree: KeplerKDTree, clusterRadius: number): number {
  const hasPair = (zoom: number): boolean => {
    const radius = clusterRadius / (EXTENT * Math.pow(2, zoom));
    for (let index = 0; index < leaf.count; index++)
      if (tree.hasOtherWithin(leaf.x[index], leaf.y[index], radius, index)) return true;
    return false;
  };
  if (!hasPair(0)) return -1;
  let low = 0;
  let high = MAXIMUM_ZOOM;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (hasPair(middle)) low = middle;
    else high = middle - 1;
  }
  return low;
}

/**
 * Builds the hierarchy down to `lowestZoom` (default every zoom). Coarser levels are added later by
 * {@link extendKeplerCompactClusterIndex}, which queries call automatically. `maximumBytes` bounds
 * the retained levels (the builder throws {@link KEPLER_CLUSTER_BUDGET_MESSAGE} when exceeded) so
 * admission is explicit, not estimated.
 */
export function buildKeplerCompactClusterIndex(
  source: KeplerClusterSource,
  clusterRadius: number,
  maximumBytes = Infinity,
  lowestZoom = 0
): KeplerCompactClusterIndex {
  const leaf = createLeafLevel(source);
  const levels: (Level | undefined)[] = new Array(MAXIMUM_ZOOM + 2);
  levels[MAXIMUM_ZOOM + 1] = leaf;
  const retainedBytes = getLevelByteLength(leaf);
  if (retainedBytes > maximumBytes) throw new Error(KEPLER_CLUSTER_BUDGET_MESSAGE);
  const tree = new KeplerKDTree(leaf.x, leaf.y, leaf.count);
  // Levels finer than the first merge zoom are the leaf level itself.
  const firstMergeZoom = getFirstMergeZoom(leaf, tree, clusterRadius);
  for (let zoom = MAXIMUM_ZOOM; zoom > firstMergeZoom; zoom--) levels[zoom] = leaf;
  const index: KeplerCompactClusterIndex = {
    levels,
    positions: source.positions,
    clusterRadius,
    rowCount: source.positions.length / 2,
    byteLength: retainedBytes,
    maximumBytes,
    builder: firstMergeZoom >= 0 ? {current: leaf, tree, nextZoom: firstMergeZoom} : undefined
  };
  extendKeplerCompactClusterIndex(index, lowestZoom);
  return index;
}

/** Builds the levels down to `lowestZoom` that are still missing; throws if the byte budget is hit. */
export function extendKeplerCompactClusterIndex(
  index: KeplerCompactClusterIndex,
  lowestZoom: number
): void {
  const builder = index.builder;
  if (!builder) return;
  while (builder.nextZoom >= Math.max(0, lowestZoom)) {
    const zoom = builder.nextZoom;
    const next = clusterLevel(builder.current, builder.tree, zoom, index.clusterRadius);
    if (!next) {
      index.levels[zoom] = builder.current;
    } else {
      const bytes = index.byteLength + getLevelByteLength(next);
      if (bytes > index.maximumBytes) throw new Error(KEPLER_CLUSTER_BUDGET_MESSAGE);
      index.byteLength = bytes;
      index.levels[zoom] = next;
      builder.current = next;
      builder.tree = new KeplerKDTree(next.x, next.y, next.count);
    }
    builder.nextZoom--;
  }
  if (builder.nextZoom < 0) index.builder = undefined;
}

/** Walks member rows of one node depth first: origin group, then merged neighbors in order. */
function appendMembers(
  rows: Uint32Array,
  cursorStart: number,
  startLevel: Level,
  startNode: number,
  stackLevels: Level[],
  stackNodes: number[]
): number {
  let cursor = cursorStart;
  stackLevels.length = 0;
  stackNodes.length = 0;
  stackLevels.push(startLevel);
  stackNodes.push(startNode);
  while (stackLevels.length) {
    let level = stackLevels.pop() as Level;
    let node = stackNodes.pop() as number;
    // Unmerged nodes simply continue into the level below.
    while (level.source && level.childOffsets![node] === level.childOffsets![node + 1]) {
      node = level.origin![node];
      level = level.source;
    }
    if (!level.source) {
      rows[cursor++] = (level.rows as Uint32Array)[node];
      continue;
    }
    const sourceLevel = level.source;
    const children = level.children as Uint32Array;
    for (
      let child = level.childOffsets![node + 1] - 1;
      child >= level.childOffsets![node];
      child--
    ) {
      stackLevels.push(sourceLevel);
      stackNodes.push(children[child]);
    }
    stackLevels.push(sourceLevel);
    stackNodes.push(level.origin![node]);
  }
  return cursor;
}

function getClustersInBounds(
  index: KeplerCompactClusterIndex,
  boundingBox: readonly number[],
  zoom: number,
  result: {level: Level; ids: number[]}[]
): void {
  let minLongitude = ((((boundingBox[0] + 180) % 360) + 360) % 360) - 180;
  const minLatitude = Math.max(-90, Math.min(90, boundingBox[1]));
  let maxLongitude =
    boundingBox[2] === 180 ? 180 : ((((boundingBox[2] + 180) % 360) + 360) % 360) - 180;
  const maxLatitude = Math.max(-90, Math.min(90, boundingBox[3]));
  if (boundingBox[2] - boundingBox[0] >= 360) {
    minLongitude = -180;
    maxLongitude = 180;
  } else if (minLongitude > maxLongitude) {
    getClustersInBounds(index, [minLongitude, minLatitude, 180, maxLatitude], zoom, result);
    getClustersInBounds(index, [-180, minLatitude, maxLongitude, maxLatitude], zoom, result);
    return;
  }
  const levelZoom = Math.max(0, Math.min(Math.floor(+zoom), MAXIMUM_ZOOM + 1));
  extendKeplerCompactClusterIndex(index, levelZoom);
  const level = index.levels[levelZoom] as Level;
  if (index.queryTree?.level !== level)
    index.queryTree = {level, tree: new KeplerKDTree(level.x, level.y, level.count)};
  const ids: number[] = [];
  index.queryTree.tree.range(
    getLongitudeX(minLongitude),
    getLatitudeY(maxLatitude),
    getLongitudeX(maxLongitude),
    getLatitudeY(minLatitude),
    ids
  );
  result.push({level, ids});
}

/** Native clustersAtZoom on the compact hierarchy; output matches the supercluster-backed query. */
export function queryKeplerCompactClusterRanges(
  index: KeplerCompactClusterIndex,
  view: KeplerClusterView,
  minClusterSize: number
): KeplerClusterRanges {
  const zoom = Math.round(view.zoom);
  const cutoff = Number.isFinite(minClusterSize) ? minClusterSize : 1;
  const groups: {level: Level; ids: number[]}[] = [];
  getClustersInBounds(
    index,
    getViewportBounds([view.longitude, view.latitude], zoom, [view.width, view.height]),
    zoom,
    groups
  );
  let clusterCount = 0;
  let totalRows = 0;
  for (const {level, ids} of groups)
    for (const id of ids) {
      const size = level.numPoints ? level.numPoints[id] : 1;
      if (size < cutoff) continue;
      clusterCount++;
      totalRows += size;
    }
  const centers = new Float64Array(clusterCount * 2);
  const offsets = new Uint32Array(clusterCount + 1);
  const rows = new Uint32Array(totalRows);
  const stackLevels: Level[] = [];
  const stackNodes: number[] = [];
  let cursor = 0;
  let output = 0;
  for (const {level, ids} of groups)
    for (const id of ids) {
      const size = level.numPoints ? level.numPoints[id] : 1;
      if (size < cutoff) continue;
      offsets[output] = cursor;
      const nextCursor = appendMembers(rows, cursor, level, id, stackLevels, stackNodes);
      if (size > 1) {
        centers[output * 2] = getXLongitude(level.x[id]);
        centers[output * 2 + 1] = getYLatitude(level.y[id]);
      } else {
        // Single points report their source coordinates, as supercluster does.
        const row = rows[cursor];
        centers[output * 2] = index.positions[row * 2];
        centers[output * 2 + 1] = index.positions[row * 2 + 1];
      }
      cursor = nextCursor;
      output++;
    }
  offsets[clusterCount] = cursor;
  return {clusterCount, centers, offsets, rows};
}
