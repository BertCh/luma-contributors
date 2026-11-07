// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inputs and drawing helpers shared by the distribution-dynamics runs: tracts observed over
// several periods (period-major values, as the operations expect), quantile breaks, and a matrix
// drawn as a grid of colored cells through the polygon layer.

import {createTracts, type CsrWeights, type TractsInput} from '../../../inputs';
import type {RunBuilder} from '../../../define-run';

export const PERIOD_COUNT = 6;

export type DynamicsInput = {
  tracts: TractsInput;
  /** Period-major values: period t of place i is at t * count + i. */
  values: Float32Array;
};

/** Tracts on the left `share` of the world with a smooth field that drifts and regresses over time. */
export function createDynamics(
  random: () => number,
  world: {width: number; height: number},
  rows = 10,
  share = 0.5
): DynamicsInput {
  const noise = 0.1;
  const tracts = createTracts(random, {width: world.width * share, height: 1}, rows, 0.2);
  const values = new Float32Array(tracts.count * PERIOD_COUNT);
  for (let period = 0; period < PERIOD_COUNT; period++) {
    for (let place = 0; place < tracts.count; place++) {
      const x = tracts.centers[place * 2];
      const y = tracts.centers[place * 2 + 1];
      const drift = 0.25 * Math.sin(period * 0.9 + x * 7 + y * 3);
      const regression = 0.5 + (tracts.values[place] - 0.5) * (1 - 0.08 * period);
      values[period * tracts.count + place] = Math.min(
        1,
        Math.max(0, regression + drift * 0.6 + (random() - 0.5) * noise)
      );
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
