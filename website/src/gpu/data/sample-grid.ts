// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** The bytes travel in the lazy operation module, so a run can prepare its inputs synchronously. */
export function decodeRaster(dataUrl: string): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(atob(dataUrl.slice(dataUrl.indexOf(',') + 1)), char => char.charCodeAt(0));
}

/** A centered crop with square source cells, row zero at the north edge. */
export function sampleGrid(
  source: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  columns: number,
  rows: number,
  offset = 0
): Float32Array {
  const step = Math.min(sourceWidth / columns, sourceHeight / rows);
  const left = (sourceWidth - step * columns) / 2;
  const top = (sourceHeight - step * rows) / 2;
  const values = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    const y = Math.max(0, top + (row + 0.5) * step - 0.5);
    const y0 = Math.floor(y);
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    for (let column = 0; column < columns; column++) {
      const x = Math.max(0, left + (column + 0.5) * step - 0.5);
      const x0 = Math.floor(x);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const north =
        source[offset + y0 * sourceWidth + x0] * (1 - (x - x0)) +
        source[offset + y0 * sourceWidth + x1] * (x - x0);
      const south =
        source[offset + y1 * sourceWidth + x0] * (1 - (x - x0)) +
        source[offset + y1 * sourceWidth + x1] * (x - x0);
      values[row * columns + column] = north * (1 - (y - y0)) + south * (y - y0);
    }
  }
  return values;
}
