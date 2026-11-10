// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';

/** WGSL `readSeriesValue(cell, slice)` over `values` in `layout`; needs `CELL_COUNT` and `SLICE_COUNT`. */
export function getChangePointSeriesReaderWGSL(layout: GPUSpaceTimeWeightsLayout): string {
  const row = layout === 'cell-major' ? 'cell * SLICE_COUNT + slice' : 'slice * CELL_COUNT + cell';
  return /* wgsl */ `fn readSeriesValue(cell: u32, slice: u32) -> f32 {
  return values[valuesOffset + ${row}];
}`;
}
