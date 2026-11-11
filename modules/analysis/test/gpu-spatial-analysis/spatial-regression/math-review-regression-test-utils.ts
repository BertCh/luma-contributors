// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** One command graph on the headless device with input and output buffer helpers. */
export async function createRegressionTestContext() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(format: Format, data: ArrayLike<number>) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${serial++}`, buffer, format, data.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    device,
    graph,
    input,
    output,
    run: () => {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

/** Deterministic uniform stream in [0, 1) (mulberry32). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal deviates (Box-Muller) from a uniform stream. */
export function createNormalRandom(seed: number): () => number {
  const random = createRandom(seed);
  return () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

/** Float64 Gaussian elimination with partial pivoting of a small dense system. */
export function solveDense(matrix: number[][], right: number[]): number[] {
  const size = right.length;
  const augmented = matrix.map((row, index) => [...row, right[index]]);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++)
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    for (let row = column + 1; row < size; row++) {
      const factor = augmented[row][column] / augmented[column][column];
      for (let other = column; other <= size; other++)
        augmented[row][other] -= factor * augmented[column][other];
    }
  }
  const solution = new Array<number>(size).fill(0);
  for (let row = size - 1; row >= 0; row--) {
    let value = augmented[row][size];
    for (let other = row + 1; other < size; other++)
      value -= augmented[row][other] * solution[other];
    solution[row] = value / augmented[row][row];
  }
  return solution;
}

/** Float64 inverse of a small dense matrix, column by column. */
export function invertDense(matrix: number[][]): number[][] {
  const size = matrix.length;
  const columns = Array.from({length: size}, (_, column) =>
    solveDense(
      matrix,
      Array.from({length: size}, (_, row) => (row === column ? 1 : 0))
    )
  );
  return Array.from({length: size}, (_, row) => columns.map(column => column[row]));
}

/**
 * Float64 least squares of `response` on the rows of the raw `design` (normal equations, partial
 * pivoting): estimate, inverse normal matrix and residual sum of squares. Float64 holds a Gram
 * condition number of `1e6` (a predictor whose mean is 1000 spreads) to about `1e-10`.
 */
export function fitLeastSquares(design: number[][], response: number[]) {
  const size = design[0].length;
  const normal = Array.from({length: size}, () => new Array<number>(size).fill(0));
  const right = new Array<number>(size).fill(0);
  design.forEach((values, row) => {
    for (let first = 0; first < size; first++) {
      right[first] += values[first] * response[row];
      for (let second = 0; second < size; second++)
        normal[first][second] += values[first] * values[second];
    }
  });
  const estimate = solveDense(normal, right);
  const inverse = invertDense(normal);
  let squares = 0;
  design.forEach((values, row) => {
    const residual =
      response[row] - values.reduce((sum, value, column) => sum + value * estimate[column], 0);
    squares += residual * residual;
  });
  return {estimate, inverse, squares};
}

/** Abramowitz-Stegun 26.2.17 normal distribution function in float64 (|error| < 7.5e-8). */
export function getNormalDistribution(value: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(value));
  const polynomial =
    t *
    (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const upper = (Math.exp(-0.5 * value * value) / Math.sqrt(2 * Math.PI)) * polynomial;
  return value >= 0 ? 1 - upper : upper;
}
