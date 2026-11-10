// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Greyscale relief drawing for illumination and hillshade buffers.

import type {RasterRunLayer, RunBuffer} from '../../../define-run';

/** A greyscale relief layer: shadow to lit across `range`. */
export function drawRelief(
  terrain: {columns: number; rows: number},
  values: RunBuffer,
  range: readonly [number, number] = [0, 1]
): RasterRunLayer {
  return {
    layer: 'raster',
    values,
    columns: terrain.columns,
    rows: terrain.rows,
    ramp: 'grey',
    range,
    smoothing: true
  };
}
