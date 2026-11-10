// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import helene from './rasters/helene-wind.bin?url&inline';
import {decodeRaster, sampleGrid} from './sample-grid';

let components: {east: Float32Array; north: Float32Array} | undefined;

/** NOAA GFS 10 m wind, Gulf of Mexico, 2024-09-27 02 UTC. Output is north-up in m/s. */
export function sampleHeleneWind(columns: number, rows: number) {
  if (!components) {
    const bytes = decodeRaster(helene);
    components = {
      east: Float32Array.from({length: 64 * 40}, (_, cell) => -36 + (bytes[cell * 2] * 81) / 255),
      north: Float32Array.from(
        {length: 64 * 40},
        (_, cell) => -36 + (bytes[cell * 2 + 1] * 77) / 255
      )
    };
  }
  return {
    east: sampleGrid(components.east, 64, 40, columns, rows),
    north: sampleGrid(components.north, 64, 40, columns, rows)
  };
}
