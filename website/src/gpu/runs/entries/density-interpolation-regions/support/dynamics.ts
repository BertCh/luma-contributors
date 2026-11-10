// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inputs and drawing helpers shared by the distribution-dynamics runs: tracts observed over
// several periods (period-major values, as the operations expect), quantile breaks, and a matrix
// drawn as a grid of colored cells through the polygon layer.

import {createTracts, type CsrWeights, type TractsInput} from '../../../inputs';
import workplaceJobs from '../../../../data/chicago-workplace-jobs.json';
import type {RunBuilder} from '../../../define-run';

export const PERIOD_COUNT = 6;

export type DynamicsInput = {
  tracts: TractsInput;
  /** Period-major values: period t of place i is at t * count + i. */
  values: Float32Array;
};

const jobsByTract: Record<string, number[]> = workplaceJobs;
const maximumLogJobs = Math.log1p(Math.max(...Object.values(jobsByTract).flat()));

/** Workplace jobs in 2018–23, log-scaled on a single range shared by all six years. */
export function createDynamics(
  random: () => number,
  world: {width: number; height: number},
  rows = 10,
  share = 0.5,
  bottomMargin = 0
): DynamicsInput {
  const tracts = createTracts(random, {width: world.width * share, height: 1 - bottomMargin}, rows);
  if (bottomMargin > 0) {
    // Lift the map so that a strip fits under it.
    for (const array of [tracts.vertices, tracts.centers, tracts.outline]) {
      for (let index = 1; index < array.length; index += 2) array[index] += bottomMargin;
    }
  }
  const values = new Float32Array(tracts.count * PERIOD_COUNT);
  for (let place = 0; place < tracts.count; place++) {
    const years = jobsByTract[tracts.geoids[place]];
    for (let period = 0; period < PERIOD_COUNT; period++) {
      values[period * tracts.count + place] = Math.log1p(years[period]) / maximumLogJobs;
    }
  }
  return {tracts, values};
}

/** Breaks (classCount + 1) from equal intervals or quantiles of all values, padded to `length`. */
export function createBreaks(
  values: Float32Array,
  classCount: number,
  quantile: boolean,
  length = 10
): Float32Array {
  const breaks = new Float32Array(length);
  const sorted = Float32Array.from(values).sort();
  for (let index = 0; index <= classCount; index++) {
    breaks[index] = quantile
      ? sorted[Math.min(sorted.length - 1, Math.floor((index / classCount) * (sorted.length - 1)))]
      : (index / classCount) * 1.0001;
  }
  breaks[0] = -0.0001;
  breaks[classCount] = 1.0001;
  return breaks;
}

/** Owners array that shows period `period` of the period-major values on the tract mesh. */
export function getPeriodOwners(
  triangleOwners: Uint32Array,
  placeCount: number,
  period: number
): Uint32Array {
  return Uint32Array.from(triangleOwners, owner => owner + period * placeCount);
}

export type MatrixMesh = {
  vertices: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  triangleCount: number;
  outline: Float32Array;
  outlineCount: number;
};

/**
 * A `rows` x `columns` grid of quads inside `rectangle` (x0, y0, x1, y1 in world units), row 0 at
 * the top, one owner per cell so a value per cell colors it. Several matrices can be stacked by
 * calling it once per rectangle and offsetting owners.
 */
export function createMatrixMesh(
  rows: number,
  columns: number,
  rectangle: readonly [number, number, number, number],
  ownerOffset = 0
): MatrixMesh {
  const [x0, y0, x1, y1] = rectangle;
  const cellWidth = (x1 - x0) / columns;
  const cellHeight = (y1 - y0) / rows;
  const vertices = new Float32Array(rows * columns * 8);
  const indices = new Uint32Array(rows * columns * 6);
  const owners = new Uint32Array(rows * columns * 2);
  const outline: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      const left = x0 + column * cellWidth;
      const top = y1 - row * cellHeight;
      vertices.set(
        [
          left,
          top,
          left + cellWidth,
          top,
          left + cellWidth,
          top - cellHeight,
          left,
          top - cellHeight
        ],
        cell * 8
      );
      indices.set(
        [0, 1, 2, 0, 2, 3].map(corner => cell * 4 + corner),
        cell * 6
      );
      owners[cell * 2] = ownerOffset + cell;
      owners[cell * 2 + 1] = ownerOffset + cell;
      outline.push(
        left,
        top,
        left + cellWidth,
        top,
        left + cellWidth,
        top,
        left + cellWidth,
        top - cellHeight
      );
    }
  }
  return {
    vertices,
    indices,
    owners,
    triangleCount: owners.length,
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4
  };
}

/** Uploads a matrix mesh and returns the buffers a `polygons` layer needs. */
export function uploadMatrixMesh(builder: RunBuilder, mesh: MatrixMesh, id: string) {
  return {
    vertices: builder.upload(`${id}-vertices`, 'float32x2', mesh.vertices),
    indices: builder.upload(`${id}-indices`, 'uint32', mesh.indices),
    owners: builder.upload(`${id}-owners`, 'uint32', mesh.owners),
    triangleCount: mesh.triangleCount
  };
}

/** Several `size` x `size` matrices on a grid inside `rectangle`, matrix m owning cells m * size * size and up. */
export function createMatrixGrid(
  matrixCount: number,
  size: number,
  rectangle: readonly [number, number, number, number],
  columns = 2
): MatrixMesh {
  const [x0, y0, x1, y1] = rectangle;
  const rows = Math.ceil(matrixCount / columns);
  const gap = 0.012;
  const width = (x1 - x0 - gap * (columns - 1)) / columns;
  const height = (y1 - y0 - gap * (rows - 1)) / rows;
  const parts: MatrixMesh[] = [];
  for (let matrix = 0; matrix < matrixCount; matrix++) {
    const left = x0 + (matrix % columns) * (width + gap);
    const top = y1 - Math.floor(matrix / columns) * (height + gap);
    parts.push(
      createMatrixMesh(size, size, [left, top - height, left + width, top], matrix * size * size)
    );
  }
  let vertexOffset = 0;
  const merged = {
    vertices: [] as number[],
    indices: [] as number[],
    owners: [] as number[],
    outline: [] as number[]
  };
  for (const part of parts) {
    merged.vertices.push(...part.vertices);
    merged.indices.push(...part.indices.map(index => index + vertexOffset));
    merged.owners.push(...part.owners);
    merged.outline.push(...part.outline);
    vertexOffset += part.vertices.length / 2;
  }
  return {
    vertices: Float32Array.from(merged.vertices),
    indices: Uint32Array.from(merged.indices),
    owners: Uint32Array.from(merged.owners),
    triangleCount: merged.owners.length,
    outline: Float32Array.from(merged.outline),
    outlineCount: merged.outline.length / 4
  };
}

/** Row-standardized copy of rook weights (each row sums to one). */
export function rowStandardize(csr: CsrWeights): CsrWeights {
  const weights = new Float32Array(csr.weights.length);
  for (let row = 0; row < csr.rowCount; row++) {
    const start = csr.offsets[row];
    const end = csr.offsets[row + 1];
    for (let slot = start; slot < end; slot++) weights[slot] = 1 / (end - start);
  }
  return {...csr, weights};
}

export type MatrixRectangle = readonly [number, number, number, number];

/** The (row, column) of the cell of a `rows` x `columns` matrix under a world position, or null. */
export function getMatrixCell(
  rectangle: MatrixRectangle,
  rows: number,
  columns: number,
  x: number,
  y: number
): {row: number; column: number} | null {
  const [x0, y0, x1, y1] = rectangle;
  if (x < x0 || x > x1 || y < y0 || y > y1) return null;
  return {
    column: Math.min(columns - 1, Math.floor(((x - x0) / (x1 - x0)) * columns)),
    row: Math.min(rows - 1, Math.floor(((y1 - y) / (y1 - y0)) * rows))
  };
}

/** The world rectangle of one cell of a matrix, as a box glyph's corners. */
export function getMatrixCellBox(
  rectangle: MatrixRectangle,
  rows: number,
  columns: number,
  cell: {row: number; column: number}
): {minimum: [number, number]; maximum: [number, number]} {
  const [x0, y0, x1, y1] = rectangle;
  const width = (x1 - x0) / columns;
  const height = (y1 - y0) / rows;
  return {
    minimum: [x0 + cell.column * width, y1 - (cell.row + 1) * height],
    maximum: [x0 + (cell.column + 1) * width, y1 - cell.row * height]
  };
}

/** The (from, to) classes of one tract between `period` and the period `lag` later (earlier past the last period). */
export function getTractMove(
  classes: ArrayLike<number>,
  tractCount: number,
  period: number,
  lag: number,
  tract: number
): {row: number; column: number} {
  const start = period + lag < PERIOD_COUNT ? period : Math.max(0, PERIOD_COUNT - 1 - lag);
  const end = Math.min(PERIOD_COUNT - 1, start + lag);
  return {row: classes[start * tractCount + tract], column: classes[end * tractCount + tract]};
}

/**
 * 1 for every tract that moved from class `from` to class `to` between `period` and the period
 * `lag` later (earlier when that runs past the last period), 0 for the others; also the count.
 */
export function createMoveLabels(
  classes: ArrayLike<number>,
  tractCount: number,
  period: number,
  lag: number,
  from: number,
  to: number
): {labels: Uint32Array; moveCount: number} {
  const start = period + lag < PERIOD_COUNT ? period : Math.max(0, PERIOD_COUNT - 1 - lag);
  const end = Math.min(PERIOD_COUNT - 1, start + lag);
  const labels = new Uint32Array(tractCount);
  let moveCount = 0;
  for (let tract = 0; tract < tractCount; tract++) {
    if (classes[start * tractCount + tract] === from && classes[end * tractCount + tract] === to) {
      labels[tract] = 1;
      moveCount++;
    }
  }
  return {labels, moveCount};
}

/** The rectangles `createMatrixGrid` lays its matrices out in, in matrix order. */
export function getMatrixGridRectangles(
  matrixCount: number,
  rectangle: MatrixRectangle,
  columns = 2
): MatrixRectangle[] {
  const [x0, y0, x1, y1] = rectangle;
  const rows = Math.ceil(matrixCount / columns);
  const gap = 0.012;
  const width = (x1 - x0 - gap * (columns - 1)) / columns;
  const height = (y1 - y0 - gap * (rows - 1)) / rows;
  return Array.from({length: matrixCount}, (_, matrix) => {
    const left = x0 + (matrix % columns) * (width + gap);
    const top = y1 - Math.floor(matrix / columns) * (height + gap);
    return [left, top - height, left + width, top] as const;
  });
}
