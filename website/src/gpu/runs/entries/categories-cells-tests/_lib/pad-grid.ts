// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A raster laid on the left of a wider display grid (NaN to the right), so a chart fits beside it.

export function padToGrid(
  values: ArrayLike<number>,
  columns: number,
  rows: number,
  displayColumns: number
): Float32Array {
  const display = new Float32Array(displayColumns * rows).fill(Number.NaN);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      display[row * displayColumns + column] = values[row * columns + column];
    }
  }
  return display;
}
