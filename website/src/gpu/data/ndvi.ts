// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import dixieNdvi from './rasters/dixie-ndvi.bin?url&inline';
import {decodeRaster} from './sample-grid';

let series: Uint8Array | undefined;

/** Cell-major NDVI observations. Nearest-neighbor lookup preserves the QA no-data sentinel. */
export function sampleNdviSeries(columns: number, rows: number, dateIndices: readonly number[]) {
  series ??= decodeRaster(dixieNdvi);
  const bytes = series;
  const output = new Float32Array(columns * rows * dateIndices.length);
  const step = Math.min(64 / columns, 40 / rows);
  for (let row = 0; row < rows; row++) {
    const sourceRow = Math.floor((40 - rows * step) / 2 + (row + 0.5) * step);
    for (let column = 0; column < columns; column++) {
      const sourceColumn = Math.floor((64 - columns * step) / 2 + (column + 0.5) * step);
      const cell = row * columns + column;
      dateIndices.forEach((dateIndex, slice) => {
        const value = bytes[dateIndex * 64 * 40 + sourceRow * 64 + sourceColumn];
        output[cell * dateIndices.length + slice] = value === 0 ? Number.NaN : value / 127.5 - 1;
      });
    }
  }
  return output;
}
