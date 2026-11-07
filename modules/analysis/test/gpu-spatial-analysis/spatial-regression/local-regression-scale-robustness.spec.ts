// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    data: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
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

function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

/** Singular value ratio of a design matrix with float64 Jacobi sweeps on its Gram matrix. */
function getReferenceCondition(design: number[][]): number {
  const size = design[0].length;
  const gram = Array.from({length: size}, (_, first) =>
    Array.from({length: size}, (_, second) =>
      design.reduce((total, row) => total + row[first] * row[second], 0)
    )
  );
  for (let sweep = 0; sweep < 100; sweep++) {
    for (let first = 0; first < size; first++) {
      for (let second = first + 1; second < size; second++) {
        const cross = gram[first][second];
        if (Math.abs(cross) < 1e-300) continue;
        const theta = (gram[second][second] - gram[first][first]) / (2 * cross);
        const tangent = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
        const cosine = 1 / Math.sqrt(1 + tangent * tangent);
        const sine = tangent * cosine;
        for (let row = 0; row < size; row++) {
          const left = gram[row][first];
          const right = gram[row][second];
          gram[row][first] = cosine * left - sine * right;
          gram[row][second] = sine * left + cosine * right;
        }
        for (let column = 0; column < size; column++) {
          const left = gram[first][column];
          const right = gram[second][column];
          gram[first][column] = cosine * left - sine * right;
          gram[second][column] = sine * left + cosine * right;
        }
      }
    }
  }
  const eigenvalues = gram.map((row, index) => row[index]);
  return Math.sqrt(Math.max(...eigenvalues) / Math.min(...eigenvalues));
}

it('local collinearity reports design and correlation condition numbers above 3e3', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const rowCount = 80;
  const random = createRandom(7);
  // x2 is x1 plus a small perturbation: the scaled design has a condition number of order 1e4.
  const first = Array.from({length: rowCount}, () => random());
  const second = first.map(value => Math.fround(value + 2e-4 * (random() - 0.5)));
  const predictors = first.flatMap((value, row) => [Math.fround(value), second[row]]);
  try {
    const inflation = output('float32', rowCount * 2);
    const correlationCondition = output('float32', rowCount);
    const designCondition = output('float32', rowCount);
    const decomposition = output('float32', rowCount * 3);
    const status = output('uint32', rowCount);
    graph.add(
      new GPULocalCollinearity({
        predictors: input('float32', predictors),
        predictorCount: 2,
        weights: {
          offsets: input(
            'uint32',
            Array.from({length: rowCount + 1}, (_, row) => row * rowCount)
          ),
          neighbors: input(
            'uint32',
            Array.from({length: rowCount * rowCount}, (_, slot) => slot % rowCount)
          ),
          weights: input('float32', Array(rowCount * rowCount).fill(1))
        },
        output: {
          varianceInflationFactors: inflation.view,
          conditionNumbers: correlationCondition.view,
          designConditionNumbers: designCondition.view,
          varianceDecomposition: decomposition.view,
          status: status.view
        }
      })
    );
    context.run();
    const statuses = Array.from(await readUint32(status.buffer, rowCount));
    // Design condition: columns of [1, x1, x2] scaled to unit norm.
    const rows = predictors.length / 2;
    const columns = [0, 1, 2].map(column =>
      Array.from({length: rows}, (_, row) => (column === 0 ? 1 : predictors[row * 2 + column - 1]))
    );
    const scaled = Array.from({length: rows}, (_, row) =>
      columns.map(values => values[row] / Math.sqrt(values.reduce((sum, v) => sum + v * v, 0)))
    );
    const expectedDesign = getReferenceCondition(scaled);
    const meanFirst = first.reduce((sum, v) => sum + Math.fround(v), 0) / rows;
    const meanSecond = second.reduce((sum, v) => sum + v, 0) / rows;
    const centered = Array.from({length: rows}, (_, row) => [
      Math.fround(first[row]) - meanFirst,
      second[row] - meanSecond
    ]);
    const norms = [0, 1].map(column =>
      Math.sqrt(centered.reduce((sum, row) => sum + row[column] ** 2, 0))
    );
    const expectedCorrelation = getReferenceCondition(
      centered.map(row => row.map((value, column) => value / norms[column]))
    );
    expect(expectedDesign).toBeGreaterThan(3e3);
    expect(expectedCorrelation).toBeGreaterThan(3e3);
    expect(statuses[0]).toBe(0);
    const design = await readFloat32(designCondition.buffer, rowCount);
    const correlation = await readFloat32(correlationCondition.buffer, rowCount);
    expect(Math.abs(design[0] / expectedDesign - 1)).toBeLessThan(0.1);
    expect(Math.abs(correlation[0] / expectedCorrelation - 1)).toBeLessThan(0.1);
  } finally {
    context.destroy();
  }
});

it('local collinearity reports a capped condition number for exactly collinear designs', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const rowCount = 12;
  const predictors = Array.from({length: rowCount}, (_, row) => [row, 2 * row]).flat();
  try {
    const inflation = output('float32', rowCount * 2);
    const correlationCondition = output('float32', rowCount);
    const status = output('uint32', rowCount);
    graph.add(
      new GPULocalCollinearity({
        predictors: input('float32', predictors),
        predictorCount: 2,
        weights: {
          offsets: input(
            'uint32',
            Array.from({length: rowCount + 1}, (_, row) => row * rowCount)
          ),
          neighbors: input(
            'uint32',
            Array.from({length: rowCount * rowCount}, (_, slot) => slot % rowCount)
          ),
          weights: input('float32', Array(rowCount * rowCount).fill(1))
        },
        output: {
          varianceInflationFactors: inflation.view,
          conditionNumbers: correlationCondition.view,
          status: status.view
        }
      })
    );
    context.run();
    expect((await readUint32(status.buffer, rowCount))[0]).toBe(2);
    expect((await readFloat32(correlationCondition.buffer, rowCount))[0]).toBeGreaterThan(1e30);
  } finally {
    context.destroy();
  }
});

it('generalized GWR is invariant to a large predictor mean-to-spread ratio', async () => {
  const random = createRandom(31);
  const side = 7;
  const positions: number[] = [];
  const centeredYear: number[] = [];
  const response: number[] = [];
  for (let row = 0; row < side; row++) {
    for (let column = 0; column < side; column++) {
      positions.push(column, row);
      // Integer offsets keep year = 2000 + offset exactly representable in float32.
      const offset = Math.floor(random() * 11) - 5;
      centeredYear.push(offset);
      response.push(Math.round(Math.exp(0.4 + 0.12 * offset + 0.3 * (random() - 0.5)) * 3));
    }
  }
  async function fit(predictors: number[]) {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const coefficients = output('float32', side * side * 2);
      const fitted = output('float32', side * side);
      const codes = output('uint32', side * side);
      graph.add(
        new GPUGeneralizedGeographicallyWeightedRegression({
          family: 'poisson',
          kernel: 'gaussian',
          positions: input('float32x2', positions),
          predictors: input('float32', predictors),
          response: input('float32', response),
          predictorCount: 1,
          parameters: input('float32', [4, 1e-5]),
          maximumIterations: 64,
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: output('uint32', side * side).view}
          }
        })
      );
      context.run();
      return {
        coefficients: await readFloat32(coefficients.buffer, side * side * 2),
        fitted: await readFloat32(fitted.buffer, side * side),
        codes: Array.from(await readUint32(codes.buffer, side * side))
      };
    } finally {
      context.destroy();
    }
  }
  const baseline = await fit(centeredYear);
  const shifted = await fit(centeredYear.map(value => 2000 + value));
  expect(baseline.codes.every(code => code === 0)).toBe(true);
  expect(shifted.codes.every(code => code === 0)).toBe(true);
  for (let row = 0; row < side * side; row++) {
    expect(shifted.coefficients[row * 2 + 1]).toBeCloseTo(baseline.coefficients[row * 2 + 1], 3);
    expect(shifted.fitted[row] / baseline.fitted[row]).toBeCloseTo(1, 3);
    // The raw intercept absorbs the shift of the predictor origin.
    const expectedIntercept =
      baseline.coefficients[row * 2] - 2000 * baseline.coefficients[row * 2 + 1];
    expect(Math.abs(shifted.coefficients[row * 2] - expectedIntercept)).toBeLessThan(
      0.02 + 1e-3 * Math.abs(expectedIntercept)
    );
  }
});
