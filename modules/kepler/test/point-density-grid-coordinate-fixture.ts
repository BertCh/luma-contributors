// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Sample, GridSpec} from '../src/analysis/catalog-platform3-b-common';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

// Published binary32 definition: 0.99999999 uploads as 1, then (1 - 0) / 4 * 4 = 1.
// Both first rows occupy cell 1. The actual inclusive corner (4,3) occupies row 2, column 3.
// Using interior y=2 would permit row 1 under WGSL division rounding and would not test the corner.
// Outside and NaN coordinates are rejected by the GPU builder rather than clamped into a cell.
// Weights are exact binary32 integers: cell 1 has count 2, sum 8, mean 4; cell 11 has 1,10,10.
export const POINT_DENSITY_BOUNDARY_GRID: GridSpec = {
  minX: 0,
  minY: 0,
  cellSize: 1,
  columns: 4,
  rows: 3
};
export const POINT_DENSITY_BOUNDARY_COUNTS = [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
export const POINT_DENSITY_BOUNDARY_VALUES = {
  count: POINT_DENSITY_BOUNDARY_COUNTS,
  sum: [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
  mean: [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
};

export function createPointDensityBoundarySample(): Sample {
  return {
    x: Float64Array.of(0.99999999, 1, 4, 5, -1, NaN),
    y: Float64Array.of(0, 0, 3, 0, 0, 0),
    value: Float64Array.of(2, 6, 10, 100, 100, 100),
    sourceIndex: Int32Array.of(0, 1, 2, 3, 4, 5),
    partIndex: new Int32Array(6),
    count: 6,
    droppedCount: 0
  };
}

export function createPointDensityCatalogPoints(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [0, 0.99999999, 1].map((horizontal, row) => ({
      geometry: {type: 'Point', coordinates: [horizontal, 0]},
      properties: {weight: [1, 2, 6][row]}
    }))
  };
}

// Local-frame origin is 0. Cell 0 holds weight 1, cell 1 holds weights 2+6 after upload.
export const POINT_DENSITY_CATALOG_VALUES = {count: [1, 2], sum: [1, 8], mean: [1, 4]};

// Independent NumPy round-to-nearest binary32: f32(f32(2 / 41) * 41) = 1.9999998807907104.
// The old physical-cell floor gives column 2, while this nominal CPU evaluation gives column 1.
// WGSL division permits 2.5 ULP error and does not specify a rounding mode:
// https://www.w3.org/TR/WGSL/#floating-point-accuracy
// grid-coordinate-bump.oracle.py enumerates the quotient error bounds, independently of GPU output.
export const POINT_DENSITY_NORMALIZED_CPU_COLUMNS = [0, 1, 40];
export const POINT_DENSITY_NORMALIZED_GPU_COLUMNS = [
  [0, 1, 39],
  [0, 1, 40],
  [0, 2, 39],
  [0, 2, 40]
];

export function createPointDensityNormalizedCatalogPoints(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [0, 2, 40].map((horizontal, row) => ({
      geometry: {type: 'Point', coordinates: [horizontal, 0]},
      properties: {weight: [1, 5, 11][row]}
    }))
  };
}
