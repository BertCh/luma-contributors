// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Local minima of a raster, lowest first: the usual basin markers for a watershed.

import type {RasterInput} from '../../../inputs';

/** Row-major indices of the strict 8-neighbor local minima, lowest elevation first. */
export function findLocalMinima(raster: RasterInput): number[] {
  const {columns, rows, values} = raster;
  const minima: number[] = [];
  for (let row = 1; row < rows - 1; row++) {
    for (let column = 1; column < columns - 1; column++) {
      const index = row * columns + column;
      let isMinimum = true;
      for (let rowOffset = -1; rowOffset <= 1 && isMinimum; rowOffset++) {
        for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
          if (
            (rowOffset !== 0 || columnOffset !== 0) &&
            values[(row + rowOffset) * columns + column + columnOffset] <= values[index]
          ) {
            isMinimum = false;
            break;
          }
        }
      }
      if (isMinimum) minima.push(index);
    }
  }
  return minima.sort((first, second) => values[first] - values[second]);
}
