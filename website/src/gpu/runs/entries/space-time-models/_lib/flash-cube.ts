// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs of the space-time local statistics runs: the GOES lightning flashes over the
// Midwest counted into a lattice of `columns x rows` cells and `periodCount` equal time periods
// (a period-major cube, `period * cellCount + cell`), and queen-contiguity spatial weights of the
// lattice. Cell `row * columns + column` has row 0 at the top, as the raster layer draws it.

import {sampleFlashes} from '../../../../data/lightning';
import {buildRows, type CsrWeights} from '../../../inputs';
import {getRasterCell, getRasterCellCenter} from '../../../pick';
import type {Point} from '../../../interaction';

/** Length of the lightning sample, hours. */
export const FLASH_HOURS = 18;

export type FlashCube = {
  columns: number;
  rows: number;
  cellCount: number;
  periodCount: number;
  /** Hours per period. */
  periodHours: number;
  width: number;
  /** Flash count of cell `cell` in period `period`, at `period * cellCount + cell`. */
  counts: Float32Array;
  /**
   * Mean flash area per cell and period, at `period * cellCount + cell`. A period without flashes
   * takes the cell's mean over all its flashes (a cell with none is 0), so the series stay finite.
   */
  meanArea: Float32Array;
  /** Queen contiguity, row-standardized. */
  weights: CsrWeights;
};

export function createFlashCube(
  world: {width: number; height: number},
  rows: number,
  periodCount: number,
  flashCount = 4096
): FlashCube {
  const columns = Math.max(4, Math.round(rows * world.width));
  const cellCount = columns * rows;
  const flashes = sampleFlashes({width: 1, height: 1}, flashCount);
  const counts = new Float32Array(cellCount * periodCount);
  const areaSums = new Float32Array(cellCount * periodCount);
  for (let flash = 0; flash < flashes.count; flash++) {
    const column = Math.min(columns - 1, Math.floor(flashes.positions[flash * 2] * columns));
    const row = Math.min(rows - 1, Math.floor((1 - flashes.positions[flash * 2 + 1]) * rows));
    const period = Math.min(
      periodCount - 1,
      Math.floor((flashes.hours[flash] / FLASH_HOURS) * periodCount)
    );
    counts[period * cellCount + row * columns + column]++;
    areaSums[period * cellCount + row * columns + column] += flashes.area[flash];
  }
  const meanArea = new Float32Array(cellCount * periodCount);
  for (let cell = 0; cell < cellCount; cell++) {
    let total = 0;
    let flashTotal = 0;
    for (let period = 0; period < periodCount; period++) {
      total += areaSums[period * cellCount + cell];
      flashTotal += counts[period * cellCount + cell];
    }
    const cellMean = flashTotal > 0 ? total / flashTotal : 0;
    for (let period = 0; period < periodCount; period++) {
      const index = period * cellCount + cell;
      meanArea[index] = counts[index] > 0 ? areaSums[index] / counts[index] : cellMean;
    }
  }
  const weights = buildRows(cellCount, cell => {
    const column = cell % columns;
    const row = Math.floor(cell / columns);
    const found: [number, number][] = [];
    for (const neighbor of getQueenNeighbors(column, row, columns, rows)) {
      found.push([neighbor, 1]);
    }
    return found;
  });
  for (let cell = 0; cell < cellCount; cell++) {
    const first = weights.offsets[cell];
    const last = weights.offsets[cell + 1];
    for (let slot = first; slot < last; slot++) weights.weights[slot] = 1 / (last - first);
  }
  return {
    columns,
    rows,
    cellCount,
    periodCount,
    periodHours: FLASH_HOURS / periodCount,
    width: world.width,
    counts,
    meanArea,
    weights
  };
}

function getQueenNeighbors(column: number, row: number, columns: number, rows: number): number[] {
  const neighbors: number[] = [];
  for (let rowStep = -1; rowStep <= 1; rowStep++) {
    for (let columnStep = -1; columnStep <= 1; columnStep++) {
      const otherColumn = column + columnStep;
      const otherRow = row + rowStep;
      if ((columnStep === 0 && rowStep === 0) || otherColumn < 0 || otherRow < 0) continue;
      if (otherColumn >= columns || otherRow >= rows) continue;
      neighbors.push(otherRow * columns + otherColumn);
    }
  }
  return neighbors;
}

/** The cell under a world position. */
export function getCellAt(cube: FlashCube, x: number, y: number) {
  return getRasterCell(cube.columns, cube.rows, [0, 0, cube.width, 1], x, y);
}

export function getCellCenter(cube: FlashCube, cell: number): Point {
  return getRasterCellCenter(
    cube.columns,
    cube.rows,
    [0, 0, cube.width, 1],
    cell % cube.columns,
    Math.floor(cell / cube.columns)
  );
}

/** World rectangle (minimum, maximum) of a cell. */
export function getCellBox(cube: FlashCube, cell: number): {minimum: Point; maximum: Point} {
  const [x, y] = getCellCenter(cube, cell);
  const halfWidth = cube.width / cube.columns / 2;
  const halfHeight = 1 / cube.rows / 2;
  return {
    minimum: [x - halfWidth, y - halfHeight],
    maximum: [x + halfWidth, y + halfHeight]
  };
}

/** One period of the cube as a standalone array. */
export function sliceFlashPeriod(cube: FlashCube, period: number): Float32Array {
  return cube.counts.slice(period * cube.cellCount, (period + 1) * cube.cellCount);
}

/** Where a flash count sits in the unit interval, for glyph sizes. */
export function getCountRadius(cube: FlashCube, count: number): number {
  const cellSize = Math.min(cube.width / cube.columns, 1 / cube.rows);
  return cellSize * Math.min(0.46, 0.08 + 0.1 * Math.sqrt(count));
}

/** Reads a whole buffer back as `float32` or `uint32` rows. */
export async function readRunBuffer(buffer: {
  readonly buffer: {readAsync(): Promise<Uint8Array>};
  readonly format: string;
}): Promise<Float32Array | Uint32Array> {
  const bytes = await buffer.buffer.readAsync();
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return buffer.format === 'uint32' ? new Uint32Array(copy) : new Float32Array(copy);
}
