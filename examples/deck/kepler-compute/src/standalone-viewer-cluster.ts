// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScatterplotLayer} from '@deck.gl/layers';
import {
  buildKeplerClusterIndex,
  computeKeplerClusterSummaryAppearance,
  queryKeplerClusters,
  type KeplerClusterIndex
} from '@luma-contributors/kepler';
import {readColorMap} from './standalone-viewer-color-map';

/** CPU supercluster layer using the same index/appearance code as the resident cluster path. */
export type ViewerClusterView = {
  longitude: number;
  latitude: number;
  zoom: number;
  width: number;
  height: number;
};

const indexCache = new Map<string, KeplerClusterIndex>();

function aggregateValues(aggregation: string, values: number[]): number {
  if (!values.length) return NaN;
  switch (aggregation) {
    case 'sum':
      return values.reduce((total, value) => total + value, 0);
    case 'minimum':
    case 'min':
      return Math.min(...values);
    case 'maximum':
    case 'max':
      return Math.max(...values);
    default:
      return values.reduce((total, value) => total + value, 0) / values.length;
  }
}

export function buildClusterLayer(options: {
  id: string;
  opacity: number;
  rows: Uint32Array;
  longitudes: ArrayLike<number>;
  latitudes: ArrayLike<number>;
  colorValues: ArrayLike<number> | null;
  colorAggregation: string;
  visConfig: Record<string, any>;
  scaleType: string | null;
  colors: [number, number, number][];
  view: ViewerClusterView;
}): {layer: ScatterplotLayer; clusterCount: number} {
  const {rows, visConfig} = options;
  const rowCount = options.longitudes.length;
  const clusterRadius = Number(visConfig['clusterRadius'] ?? 40);
  const cacheKey = `${options.id}:${rows.length}:${rows[0]}:${rows[rows.length - 1]}:${clusterRadius}`;
  let index = indexCache.get(cacheKey);
  if (!index) {
    const positions = new Float32Array(rowCount * 2);
    for (let row = 0; row < rowCount; row++) {
      positions[row * 2] = options.longitudes[row];
      positions[row * 2 + 1] = options.latitudes[row];
    }
    const selectionMask = new Uint32Array(rowCount);
    for (const row of rows) selectionMask[row] = 1;
    index = buildKeplerClusterIndex({positions, selectionMask}, clusterRadius);
    for (const key of indexCache.keys())
      if (key.startsWith(`${options.id}:`)) indexCache.delete(key);
    indexCache.set(cacheKey, index);
  }
  const clusters = queryKeplerClusters(
    index,
    options.view,
    Number(visConfig['minClusterSize'] ?? 1)
  );
  const colorMap = readColorMap(visConfig['colorRange']);
  const summaries = clusters.map(cluster => ({
    count: cluster.memberRows.length,
    colorValue: options.colorValues
      ? aggregateValues(
          options.colorAggregation,
          cluster.memberRows.map(row => options.colorValues![row]).filter(Number.isFinite)
        )
      : cluster.memberRows.length,
    longitude: cluster.longitude,
    latitude: cluster.latitude
  }));
  const appearance = computeKeplerClusterSummaryAppearance(summaries, {
    clusterRadius,
    minClusterSize: 1,
    radiusRange: Array.isArray(visConfig['radiusRange'])
      ? [visConfig['radiusRange'][0], visConfig['radiusRange'][1]]
      : [1, 40],
    colorRange: options.colors,
    colorScaleType:
      options.scaleType === 'quantile'
        ? 'quantile'
        : options.scaleType === 'ordinal'
          ? 'ordinal'
          : colorMap && options.scaleType === 'custom'
            ? 'threshold'
            : 'quantize',
    ...(colorMap && options.scaleType === 'custom'
      ? {getThresholds: () => colorMap.slice(0, -1).map(entry => Number(entry[0]))}
      : {})
  });
  const data = clusters.map((cluster, clusterIndex) => ({
    position: [cluster.longitude, cluster.latitude] as [number, number],
    radius: appearance.radii[clusterIndex],
    color: [
      appearance.colors[clusterIndex * 4],
      appearance.colors[clusterIndex * 4 + 1],
      appearance.colors[clusterIndex * 4 + 2],
      appearance.colors[clusterIndex * 4 + 3]
    ] as [number, number, number, number],
    count: summaries[clusterIndex].count,
    value: summaries[clusterIndex].colorValue,
    tooltipText: `Points: ${summaries[clusterIndex].count}${options.colorValues ? `\nValue: ${summaries[clusterIndex].colorValue}` : ''}`
  }));
  const layer = new ScatterplotLayer<(typeof data)[number]>({
    id: options.id,
    data,
    pickable: true,
    opacity: options.opacity,
    radiusUnits: 'pixels',
    getPosition: item => item.position,
    getRadius: item => item.radius,
    getFillColor: item => item.color
  });
  return {layer, clusterCount: clusters.length};
}
