// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import canyon from './rasters/grand-canyon.bin?url&inline';
import {decodeRaster, sampleGrid} from './sample-grid';

const WIDTH = 128;
const HEIGHT = 80;
const METERS_PER_CELL = 70;

let elevation: Uint16Array | undefined;

/** USGS 3DEP Grand Canyon crop, elevations and square cell spacing in meters. */
export function sampleElevation(columns: number, rows: number) {
  elevation ??= new Uint16Array(decodeRaster(canyon).buffer);
  return {
    elevation: sampleGrid(elevation, WIDTH, HEIGHT, columns, rows),
    cellSize: Math.min(WIDTH / columns, HEIGHT / rows) * METERS_PER_CELL
  };
}

/** Elevation mapped to the [0,1] input range of scalar raster operations. */
export function normalizeElevation(elevation: Float32Array): Float32Array {
  let low = Infinity;
  let high = -Infinity;
  for (const value of elevation) {
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  const range = Math.max(1, high - low);
  return elevation.map(value => (value - low) / range);
}
