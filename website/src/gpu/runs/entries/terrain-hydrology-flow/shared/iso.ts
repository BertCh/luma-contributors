// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs for the contouring runs. The operations place row 0 at the smallest y of the extent,
// while the raster layer draws row 0 at the top, so the field exists twice: bottom-up for the
// operation and top-down for the context raster.

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {RunInputContext} from '../../../define-run';
import {createSmoothField} from '../../../inputs';

type RasterGrid = ReturnType<typeof makeGPURasterMetadata>;

export const LEVEL_CAPACITY = 8;
/** Values span 0 to this, so levels are readable numbers. */
export const FIELD_MAXIMUM = 100;

export type IsoInputs = {
  columns: number;
  rows: number;
  cellCount: number;
  grid: RasterGrid;
  /** World extent `[minX, minY, maxX, maxY]`; row 0 is at minY. */
  extent: readonly [number, number, number, number];
  /** Row 0 at the bottom: what the operations read. */
  valuesBottomUp: Float32Array;
  /** Row 0 at the top: what the raster layer draws. */
  valuesTopDown: Float32Array;
};

export function createIsoInputs(context: RunInputContext, rows = 40): IsoInputs {
  const {random, world} = context;
  const columns = Math.max(24, Math.round(rows * world.width));
  const field = createSmoothField(random, 4, 2);
  const cellCount = columns * rows;
  const valuesBottomUp = new Float32Array(cellCount);
  const valuesTopDown = new Float32Array(cellCount);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const value = FIELD_MAXIMUM * field((column + 0.5) / columns, (row + 0.5) / rows);
      valuesBottomUp[row * columns + column] = value;
      valuesTopDown[(rows - 1 - row) * columns + column] = value;
    }
  }
  return {
    columns,
    rows,
    cellCount,
    grid: makeGPURasterMetadata({width: columns, height: rows}),
    extent: [0, 0, world.width, world.height],
    valuesBottomUp,
    valuesTopDown
  };
}

/** Evenly spaced levels or breaks: `first + index * interval`, padded to the buffer capacity. */
export function getLevels(first: number, interval: number): Float32Array {
  return Float32Array.from({length: LEVEL_CAPACITY}, (_, index) => first + index * interval);
}
