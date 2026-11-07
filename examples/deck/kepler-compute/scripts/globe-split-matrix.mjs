// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Layer configurations and screen regions shared by scripts/globe-split-aggregates.mjs and the
// node spec that keeps the matrix well formed.

/** Pixel columns (inclusive start, exclusive end) the sidebar and edges excluded. */
export const GLOBE_SPLIT_REGIONS = {
  full: [340, 1400],
  left: [340, 700],
  right: [740, 1400]
};

const aggregateStyle = {
  worldUnitSize: 600,
  coverage: 1,
  enable3d: false,
  percentile: [0, 100],
  opacity: 1,
  colorAggregation: 'count',
  colorRange: {colors: ['#ff14aa', '#ff14aa']}
};

export const GLOBE_SPLIT_AGGREGATES = {
  grid: {layerType: 'grid', zoom: 2.5, minPixels: 400, visConfig: aggregateStyle},
  hexagon: {layerType: 'hexagon', zoom: 2.5, minPixels: 400, visConfig: aggregateStyle},
  heatmap: {
    layerType: 'heatmap',
    zoom: 2.5,
    minPixels: 400,
    visConfig: {opacity: 1, radius: 400, intensity: 3, threshold: 0.01, aggregation: 'SUM', colorRange: {colors: ['#ff14aa', '#ff14aa']}}
  }
};
