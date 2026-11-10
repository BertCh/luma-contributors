// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs for the contouring runs. The operations place row 0 at the smallest y of the extent,
// while the raster layer draws row 0 at the top, so the field exists twice: bottom-up for the
// operation and top-down for the context raster.

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {RunInputContext} from '../../../define-run';
import {getRasterCell} from '../../../pick';
import {sampleElevation, normalizeElevation} from '../../../../data/elevation';

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
  const {world} = context;
  const columns = Math.max(24, Math.round(rows * world.width));
  const elevation = normalizeElevation(sampleElevation(columns, rows).elevation);
  const cellCount = columns * rows;
  const valuesBottomUp = new Float32Array(cellCount);
  const valuesTopDown = new Float32Array(cellCount);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const value = FIELD_MAXIMUM * elevation[(rows - 1 - row) * columns + column];
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

/** The field value under a drawing-world position (nearest cell). */
export function sampleValue(inputs: IsoInputs, x: number, y: number): number {
  const cell = getRasterCell(inputs.columns, inputs.rows, inputs.extent, x, y);
  return inputs.valuesTopDown[cell.index];
}

/** Levels that contain `anchor` first, then alternate above and below it: anchor, +1, -1, +2, ... */
export function getLevelsAround(anchor: number, interval: number): Float32Array {
  return Float32Array.from({length: LEVEL_CAPACITY}, (_, index) => {
    const step = Math.ceil(index / 2);
    return anchor + (index % 2 === 1 ? step : -step) * interval;
  });
}

/** Ascending breaks of which the middle one equals `value`, so a band boundary crosses it. */
export function getBreaksThrough(
  value: number,
  interval: number,
  breakCount: number
): Float32Array {
  const below = Math.floor(breakCount / 2);
  return getLevels(value - below * interval, interval);
}
