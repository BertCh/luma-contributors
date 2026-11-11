// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {
  getGPUKrigingParameterValues,
  GPUKriging,
  GPU_KRIGING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from './extent-grid';

// Math correctness review, interpolation-density-08. `GPUKriging` evaluated the variogram shape
// `1 - correlation` directly in float32. For a lag that is a small fraction of the range the
// correlation is within a few float32 steps of 1, so the difference keeps only one or two
// digits (at a lag of 1e-4 of the range the exponential shape is 3e-4 and one float32 step of
// the correlation is 6e-8, an error of 2e-4 per entry before the solve amplifies it). The
// expected values are a float64 solve of the ordinary kriging system with the float64 shapes
// (`Math.expm1` for the exponential forms), not a copy of the kernel.

type ModelName = 'exponential' | 'gaussian' | 'matern-3/2' | 'matern-5/2';

/** Variogram shape `1 - correlation` in float64, without cancellation. */
function getShape(model: ModelName, scaled: number): number {
  switch (model) {
    case 'exponential':
      return -Math.expm1(-3 * scaled);
    case 'gaussian':
      return -Math.expm1(-3 * scaled * scaled);
    case 'matern-3/2': {
      const argument = MATERN_3_2_RANGE_FACTOR * scaled;
      // 1 - (1 + a) exp(-a) = -expm1(-a) - a exp(-a).
      return -Math.expm1(-argument) - argument * Math.exp(-argument);
    }
    default: {
      const argument = MATERN_5_2_RANGE_FACTOR * scaled;
      return -Math.expm1(-argument) - (argument + (argument * argument) / 3) * Math.exp(-argument);
    }
  }
}

/** Solves `matrix * x = right` in float64 by Gaussian elimination with partial pivoting. */
function solveLinearSystem(matrix: number[][], right: number[]): number[] {
  const size = right.length;
  const rows = matrix.map((row, index) => [...row, right[index]]);
  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivotRow][column])) {
        pivotRow = row;
      }
    }
    [rows[column], rows[pivotRow]] = [rows[pivotRow], rows[column]];
    for (let row = column + 1; row < size; row++) {
      const factor = rows[row][column] / rows[column][column];
      for (let entry = column; entry <= size; entry++) {
        rows[row][entry] -= factor * rows[column][entry];
      }
    }
  }
  const solution = new Array<number>(size).fill(0);
  for (let row = size - 1; row >= 0; row--) {
    let value = rows[row][size];
    for (let column = row + 1; column < size; column++) {
      value -= rows[row][column] * solution[column];
    }
    solution[row] = value / rows[row][row];
  }
  return solution;
}

// Eight samples within about two units of each other; float32-exact coordinates.
const SAMPLE_POSITIONS = [
  [0.25, 0.5],
  [1.75, 0.25],
  [0.5, 1.75],
  [1.5, 1.5],
  [1, 0.75],
  [0.125, 1.125],
  [1.875, 1],
  [0.875, 1.875]
];
const SAMPLE_VALUES = [10, -40, 75, 20, -15, 60, 5, 90];
const COLUMNS = 4;
const ROWS = 4;
const EXTENT: [number, number, number, number] = [0, 0, 2, 2];

/** Ordinary kriging prediction at `query` in float64 (zero nugget, unit sill). */
function getReferencePrediction(model: ModelName, range: number, query: number[]): number {
  const count = SAMPLE_POSITIONS.length;
  const getGamma = (first: number[], second: number[]) =>
    getShape(model, Math.hypot(first[0] - second[0], first[1] - second[1]) / range);
  const matrix = SAMPLE_POSITIONS.map(first => [
    ...SAMPLE_POSITIONS.map(second => getGamma(first, second)),
    1
  ]);
  matrix.push([...new Array<number>(count).fill(1), 0]);
  const solution = solveLinearSystem(matrix, [
    ...SAMPLE_POSITIONS.map(sample => getGamma(sample, query)),
    1
  ]);
  return SAMPLE_VALUES.reduce((sum, value, index) => sum + solution[index] * value, 0);
}

async function runKriging(model: ModelName, range: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cellCount = COLUMNS * ROWS;
  const sampleCount = SAMPLE_VALUES.length;
  const grid = makeExtentGrid(COLUMNS, ROWS, EXTENT);
  const positionsBuffer = createInputBuffer(device, Float32Array.from(SAMPLE_POSITIONS.flat()));
  const valuesBuffer = createInputBuffer(device, Float32Array.from(SAMPLE_VALUES));
  const valuesOut = createOutputBuffer(device, cellCount);
  const varianceOut = createOutputBuffer(device, cellCount);
  const invalidOut = createOutputBuffer(device, 1);
  const uncertainOut = createOutputBuffer(device, 1);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kriging-small-lag-parameters',
    format: 'float32',
    length: GPU_KRIGING_PARAMETER_LENGTH,
    values: getGPUKrigingParameterValues(
      {
        searchRadius: Infinity,
        neighborCount: sampleCount,
        variogram: {model, nugget: 0, sill: 1, range}
      },
      grid
    )
  });
  const graph = new GPUCommandGraph(device, {id: 'kriging-small-lag'});
  graph.add(
    new GPUKriging({
      id: 'kriging',
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', sampleCount),
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', sampleCount),
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: [2, 2],
      indexBounds: EXTENT,
      maximumNeighborCount: sampleCount,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        variance: importGraphBuffer(graph, 'variance-out', varianceOut, 'float32', cellCount),
        status: {
          invalidCount: importGraphBuffer(graph, 'invalid-out', invalidOut, 'uint32', 1),
          uncertainCount: importGraphBuffer(graph, 'uncertain-out', uncertainOut, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: await readFloat32(valuesOut, cellCount),
    invalidCount: (await readUint32(invalidOut, 1))[0],
    uncertainCount: (await readUint32(uncertainOut, 1))[0]
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [
    positionsBuffer,
    valuesBuffer,
    valuesOut,
    varianceOut,
    invalidOut,
    uncertainOut
  ]) {
    buffer.destroy();
  }
  return result;
}

// The smooth models (Gaussian, Matern) are not here: at these lags their systems are below the
// pivot-ratio limit and every cell is reported uncertain, with the direct form and with the
// series alike. Their shapes are checked directly in `variogram-model-shape.spec.ts`.
for (const range of [300, 1000, 1500]) {
  it(`GPUKriging keeps float32 accuracy for lags near ${(1 / range).toExponential(0)} of the exponential range`, async () => {
    const {values, invalidCount, uncertainCount} = await runKriging('exponential', range);
    expect(invalidCount).toBe(0);
    expect(uncertainCount).toBe(0);
    let worst = 0;
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        const query = [(column + 0.5) * (2 / COLUMNS), (row + 0.5) * (2 / ROWS)];
        const expected = getReferencePrediction('exponential', range, query);
        worst = Math.max(worst, Math.abs(values[row * COLUMNS + column] - expected));
      }
    }
    // The sample values span 130. Measured on the test GPU: 6e-4 (range 300) to 3.7e-3 (range
    // 1500) with the direct `1 - exp(-x)`, and 1.4e-5 to 2.1e-5 with the series.
    expect(worst).toBeLessThan(1e-4);
  });
}
