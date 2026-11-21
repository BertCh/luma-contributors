// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';

/** Above the former 65,536-row cap of the four single-thread regression fits. */
const ROW_COUNT = 70001;

async function setup() {
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
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Float64 Gaussian elimination with partial pivoting of a small dense system. */
function solveDense(matrix: number[][], right: number[]): number[] {
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

/** Float64 OLS with intercept over the given rows of a row-major design (float32-rounded data). */
function fitOrdinaryLeastSquares(
  predictors: Float32Array,
  response: Float32Array,
  predictorCount: number,
  rows: number[]
): number[] {
  const size = predictorCount + 1;
  const normal = Array.from({length: size}, () => new Array<number>(size).fill(0));
  const right = new Array<number>(size).fill(0);
  for (const row of rows) {
    const design = [
      1,
      ...Array.from(predictors.subarray(row * predictorCount, (row + 1) * predictorCount))
    ];
    for (let first = 0; first < size; first++) {
      right[first] += design[first] * response[row];
      for (let second = 0; second < size; second++)
        normal[first][second] += design[first] * design[second];
    }
  }
  return solveDense(normal, right);
}

function expectClose(actual: ArrayLike<number>, expected: number[], tolerance: number) {
  expected.forEach((value, index) =>
    expect(
      Math.abs(actual[index] - value),
      `element ${index}: ${actual[index]} vs ${value}`
    ).toBeLessThan(tolerance * Math.max(1, Math.abs(value)))
  );
}

it('spatial regime regression above the former row cap matches a float64 reference', async () => {
  const predictorCount = 2;
  const regimeCount = 5;
  const random = createRandom(20261109);
  const predictors = new Float32Array(ROW_COUNT * predictorCount);
  const response = new Float32Array(ROW_COUNT);
  const regimes = new Uint32Array(ROW_COUNT);
  const mask = new Uint32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    // Regime 6 is out of range and must be excluded; one row in 50 is masked out.
    const regime = Math.floor(random() * (regimeCount + 1));
    regimes[row] = regime === regimeCount ? 6 : regime;
    mask[row] = random() < 0.02 ? 0 : 1;
    const first = 100 + 10 * random();
    const second = random() * 2 - 1;
    predictors[row * 2] = first;
    predictors[row * 2 + 1] = second;
    response[row] =
      regime + (regime - 2) * 0.5 * first + (1 + regime) * second + (random() - 0.5) * 0.2;
  }
  const expected: number[] = [];
  const expectedCounts: number[] = [];
  for (let regime = 0; regime < regimeCount; regime++) {
    const rows = Array.from({length: ROW_COUNT}, (_, row) => row).filter(
      row => regimes[row] === regime && mask[row] !== 0
    );
    expectedCounts.push(rows.length);
    expected.push(...fitOrdinaryLeastSquares(predictors, response, predictorCount, rows));
  }

  const context = await setup();
  const {graph, input, output} = context;
  try {
    const coefficients = output('float32', regimeCount * 3);
    const fitted = output('float32', ROW_COUNT);
    const status = output('uint32', regimeCount);
    const counts = output('uint32', regimeCount);
    graph.add(
      new GPUSpatialRegimeRegression({
        predictors: input('float32', predictors),
        predictorCount,
        response: input('float32', response),
        regimes: input('uint32', regimes),
        regimeCount,
        mask: input('uint32', mask),
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: status.view,
          counts: counts.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, regimeCount))).toEqual([0, 0, 0, 0, 0]);
    expect(Array.from(await readUint32(counts.buffer, regimeCount))).toEqual(expectedCounts);
    // Float32 centered moments over about 14,000 rows per regime: slopes to about 1e-3.
    expectClose(await readFloat32(coefficients.buffer, regimeCount * 3), expected, 2e-3);
    const fittedValues = await readFloat32(fitted.buffer, ROW_COUNT);
    expect(Number.isNaN(fittedValues[regimes.findIndex(regime => regime === 6)])).toBe(true);
  } finally {
    context.destroy();
  }
});

/** Float64 least squares of `response` on the rows of `design`: estimate, inverse normal and SSE. */
function fitDesign(design: number[][], response: number[]) {
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
  const inverse = Array.from({length: size}, (_, column) =>
    solveDense(
      normal,
      Array.from({length: size}, (_, row) => (row === column ? 1 : 0))
    )
  );
  let squares = 0;
  design.forEach((values, row) => {
    const residual =
      response[row] - values.reduce((sum, value, column) => sum + value * estimate[column], 0);
    squares += residual * residual;
  });
  return {estimate, inverse, squares};
}

it('restricted regime regression above the former row cap matches a float64 reference', async () => {
  const predictorCount = 2;
  const regimeCount = 3;
  const common = [false, true, false];
  const random = createRandom(7);
  const predictors = new Float32Array(ROW_COUNT * predictorCount);
  const response = new Float32Array(ROW_COUNT);
  const regimes = new Uint32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    const regime = Math.floor(random() * regimeCount);
    regimes[row] = regime;
    predictors[row * 2] = random() * 4;
    predictors[row * 2 + 1] = random() * 2 - 1;
    // The shared slope differs a little by regime, so the Chow statistic is positive.
    response[row] =
      1 +
      regime +
      (0.5 + 0.01 * regime) * predictors[row * 2] -
      regime * predictors[row * 2 + 1] +
      (random() - 0.5);
  }
  // Restricted design: varying intercept (3), shared first slope (1), varying second slope (3).
  const offsets = [0, 3, 4];
  const restrictedDesign: number[][] = [];
  const unrestrictedDesign: number[][] = [];
  for (let row = 0; row < ROW_COUNT; row++) {
    const values = [1, predictors[row * 2], predictors[row * 2 + 1]];
    const restrictedRow = new Array<number>(7).fill(0);
    const unrestrictedRow = new Array<number>(9).fill(0);
    values.forEach((value, column) => {
      restrictedRow[offsets[column] + (common[column] ? 0 : regimes[row])] = value;
      unrestrictedRow[column * regimeCount + regimes[row]] = value;
    });
    restrictedDesign.push(restrictedRow);
    unrestrictedDesign.push(unrestrictedRow);
  }
  const responseValues = Array.from(response);
  const restrictedFit = fitDesign(restrictedDesign, responseValues);
  const unrestrictedFit = fitDesign(unrestrictedDesign, responseValues);
  const variance = restrictedFit.squares / (ROW_COUNT - 7);
  const expectedCovariance = restrictedFit.inverse.flatMap(row =>
    row.map(value => value * variance)
  );
  const expectedChow =
    (restrictedFit.squares - unrestrictedFit.squares) /
    2 /
    (unrestrictedFit.squares / (ROW_COUNT - 9));

  const context = await setup();
  const {graph, input, output} = context;
  try {
    const coefficients = output('float32', 7);
    const covariance = output('float32', 49);
    const chow = output('float32', 3);
    const status = output('uint32', 1);
    graph.add(
      new GPURestrictedRegimeRegression({
        predictors: input('float32', predictors),
        predictorCount,
        response: input('float32', response),
        regimes: input('uint32', regimes),
        regimeCount,
        common,
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          chow: chow.view,
          fitted: output('float32', ROW_COUNT).view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    expectClose(await readFloat32(coefficients.buffer, 7), restrictedFit.estimate, 2e-3);
    const covarianceValues = await readFloat32(covariance.buffer, 49);
    expectedCovariance.forEach((value, index) =>
      expect(Math.abs(covarianceValues[index] - value)).toBeLessThan(
        2e-2 * Math.abs(expectedCovariance[(index % 7) * 8]) + 1e-9
      )
    );
    const chowValues = await readFloat32(chow.buffer, 3);
    expect(Math.abs(chowValues[0] / expectedChow - 1)).toBeLessThan(2e-2);
    expect(Array.from(chowValues.slice(1))).toEqual([2, ROW_COUNT - 9]);
  } finally {
    context.destroy();
  }
});

/** Float64 inverse of a small dense matrix, column by column. */
function invertDense(matrix: number[][]): number[][] {
  const size = matrix.length;
  const columns = Array.from({length: size}, (_, column) =>
    solveDense(
      matrix,
      Array.from({length: size}, (_, row) => (row === column ? 1 : 0))
    )
  );
  return Array.from({length: size}, (_, row) => columns.map(column => column[row]));
}

function multiply(left: number[][], right: number[][]): number[][] {
  return left.map(row =>
    right[0].map((_, column) =>
      row.reduce((sum, value, inner) => sum + value * right[inner][column], 0)
    )
  );
}

function transpose(matrix: number[][]): number[][] {
  return matrix[0].map((_, column) => matrix.map(row => row[column]));
}

for (const method of ['HC1', 'HAC'] as const)
  it(`IV ${method} covariance above the former row cap matches a float64 reference`, async () => {
    const columns = 2;
    const instrumentCount = 3;
    const coefficients = columns + 1;
    const random = createRandom(method === 'HC1' ? 11 : 12);
    const design = new Float32Array(ROW_COUNT * columns);
    const instruments = new Float32Array(ROW_COUNT * instrumentCount);
    const residuals = new Float32Array(ROW_COUNT);
    for (let row = 0; row < ROW_COUNT; row++) {
      const exogenous = random() * 2;
      const first = random() - 0.5;
      const second = random() - 0.5;
      instruments[row * 3] = exogenous;
      instruments[row * 3 + 1] = first;
      instruments[row * 3 + 2] = second;
      design[row * 2] = exogenous;
      design[row * 2 + 1] = first + 0.5 * second + 0.3 * (random() - 0.5);
      residuals[row] = (random() - 0.5) * (1 + exogenous);
    }
    // HAC kernel: tridiagonal, unit diagonal, 0.5 to each adjacent row (symmetric, PSD).
    const offsets = [0];
    const neighbors: number[] = [];
    const weights: number[] = [];
    for (let row = 0; row < ROW_COUNT; row++) {
      for (const neighbor of [row - 1, row, row + 1])
        if (neighbor >= 0 && neighbor < ROW_COUNT) {
          neighbors.push(neighbor);
          weights.push(neighbor === row ? 1 : 0.5);
        }
      offsets.push(neighbors.length);
    }

    // Float64 reference on the float32-rounded inputs.
    const mean = (values: Float32Array, stride: number, column: number) => {
      let sum = 0;
      for (let row = 0; row < ROW_COUNT; row++) sum += values[row * stride + column];
      return sum / ROW_COUNT;
    };
    const designMeans = [0, 1].map(column => mean(design, columns, column));
    const instrumentMeans = [0, 1, 2].map(column => mean(instruments, instrumentCount, column));
    const centeredDesign = Array.from({length: ROW_COUNT}, (_, row) =>
      designMeans.map((value, column) => design[row * columns + column] - value)
    );
    const centeredInstruments = Array.from({length: ROW_COUNT}, (_, row) =>
      instrumentMeans.map((value, column) => instruments[row * instrumentCount + column] - value)
    );
    const instrumentGram = multiply(transpose(centeredInstruments), centeredInstruments);
    const crossGram = multiply(transpose(centeredDesign), centeredInstruments);
    const instrumentInverse = invertDense(instrumentGram);
    const projected = multiply(crossGram, instrumentInverse);
    const projection = multiply(invertDense(multiply(projected, transpose(crossGram))), projected);
    const scores = centeredInstruments.map((row, index) => {
      const influences = projection.map(weightsRow =>
        weightsRow.reduce((sum, value, instrument) => sum + value * row[instrument], 0)
      );
      const intercept =
        1 / ROW_COUNT -
        influences.reduce((sum, value, column) => sum + designMeans[column] * value, 0);
      return [intercept, ...influences].map(value => value * residuals[index]);
    });
    const smoothed =
      method === 'HAC'
        ? scores.map((_, row) => {
            const result = new Array<number>(coefficients).fill(0);
            for (let slot = offsets[row]; slot < offsets[row + 1]; slot++)
              for (let column = 0; column < coefficients; column++)
                result[column] += weights[slot] * scores[neighbors[slot]][column];
            return result;
          })
        : scores;
    const expected = new Array<number>(coefficients ** 2).fill(0);
    scores.forEach((row, index) => {
      for (let first = 0; first < coefficients; first++)
        for (let second = 0; second < coefficients; second++)
          expected[first * coefficients + second] += row[first] * smoothed[index][second];
    });
    if (method === 'HC1')
      for (let slot = 0; slot < expected.length; slot++)
        expected[slot] *= ROW_COUNT / (ROW_COUNT - coefficients);

    const context = await setup();
    const {graph, input, output} = context;
    try {
      const covariance = output('float32', coefficients ** 2);
      const status = output('uint32', 1);
      graph.add(
        new GPUInstrumentalVariablesCovariance({
          design: input('float32', design),
          coefficientCount: columns,
          instruments: input('float32', instruments),
          instrumentCount,
          residuals: input('float32', residuals),
          method,
          ...(method === 'HAC'
            ? {
                kernel: {
                  offsets: input('uint32', offsets),
                  neighbors: input('uint32', neighbors),
                  weights: input('float32', weights)
                }
              }
            : {}),
          output: {covariance: covariance.view, status: {code: status.view}}
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      const actual = await readFloat32(covariance.buffer, coefficients ** 2);
      expected.forEach((value, index) => {
        const row = Math.floor(index / coefficients);
        const column = index % coefficients;
        const scale = Math.sqrt(
          expected[row * (coefficients + 1)] * expected[column * (coefficients + 1)]
        );
        expect(
          Math.abs(actual[index] - value),
          `entry ${index}: ${actual[index]} vs ${value}`
        ).toBeLessThan(2e-3 * scale);
      });
    } finally {
      context.destroy();
    }
  });

function getNormalCdf(value: number): number {
  // Abramowitz-Stegun 7.1.26 erf in float64 (|error| < 1.5e-7).
  const t = 1 / (1 + 0.3275911 * Math.abs(value) * Math.SQRT1_2);
  const polynomial =
    t *
    (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - polynomial * Math.exp(-0.5 * value * value);
  return 0.5 * (1 + Math.sign(value) * erf);
}

function getNormalDensity(value: number): number {
  return Math.exp(-0.5 * value * value) / Math.sqrt(2 * Math.PI);
}

/** Float64 Fisher-scoring probit with intercept on a row-major design. */
function fitProbit(predictors: Float32Array, response: Float32Array, predictorCount: number) {
  const size = predictorCount + 1;
  const rows = response.length;
  let estimate = new Array<number>(size).fill(0);
  let logLikelihood = 0;
  let standardErrors: number[] = [];
  const getLinear = (row: number) =>
    estimate[0] +
    Array.from(
      {length: predictorCount},
      (_, column) => predictors[row * predictorCount + column]
    ).reduce((sum, value, column) => sum + value * estimate[column + 1], 0);
  for (let iteration = 0; iteration < 30; iteration++) {
    const gradient = new Array<number>(size).fill(0);
    const information = Array.from({length: size}, () => new Array<number>(size).fill(0));
    logLikelihood = 0;
    for (let row = 0; row < rows; row++) {
      const linear = getLinear(row);
      const fitted = Math.min(Math.max(getNormalCdf(linear), 1e-12), 1 - 1e-12);
      const density = getNormalDensity(linear);
      const outcome = response[row];
      logLikelihood += outcome * Math.log(fitted) + (1 - outcome) * Math.log(1 - fitted);
      const score = ((outcome - fitted) * density) / (fitted * (1 - fitted));
      const weight = (density * density) / (fitted * (1 - fitted));
      const design = [
        1,
        ...Array.from(predictors.subarray(row * predictorCount, (row + 1) * predictorCount))
      ];
      for (let first = 0; first < size; first++) {
        gradient[first] += design[first] * score;
        for (let second = 0; second < size; second++)
          information[first][second] += design[first] * design[second] * weight;
      }
    }
    standardErrors = invertDense(information).map((row, column) => Math.sqrt(row[column]));
    const step = solveDense(information, gradient);
    estimate = estimate.map((value, column) => value + step[column]);
  }
  let densitySum = 0;
  for (let row = 0; row < rows; row++) densitySum += getNormalDensity(getLinear(row));
  return {
    coefficients: estimate,
    logLikelihood,
    standardErrors,
    effects: estimate.slice(1).map(value => (densitySum / rows) * value)
  };
}

it('probit regression above the former row cap matches a float64 Fisher-scoring reference', async () => {
  const predictorCount = 2;
  const random = createRandom(4242);
  const predictors = new Float32Array(ROW_COUNT * predictorCount);
  const response = new Float32Array(ROW_COUNT);
  // A spatially smooth latent term along the row chain gives the residual diagnostics a signal.
  let smooth = 0;
  for (let row = 0; row < ROW_COUNT; row++) {
    const first = random() * 2 - 1;
    const second = random() * 4;
    smooth = 0.9 * smooth + 0.3 * (random() - 0.5);
    predictors[row * 2] = first;
    predictors[row * 2 + 1] = second;
    // Box-Muller normal noise.
    const noise = Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    response[row] = -0.4 + 0.8 * first + 0.3 * second + smooth + noise > 0 ? 1 : 0;
  }
  const expected = fitProbit(predictors, response, predictorCount);
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < ROW_COUNT; row++) {
    const adjacent = [row - 1, row + 1].filter(neighbor => neighbor >= 0 && neighbor < ROW_COUNT);
    for (const neighbor of adjacent) {
      neighbors.push(neighbor);
      weights.push(1 / adjacent.length);
    }
    offsets.push(neighbors.length);
  }

  const context = await setup();
  const {graph, input, output} = context;
  try {
    const coefficients = output('float32', 3);
    const errors = output('float32', 3);
    const effects = output('float32', 2);
    const summary = output('float32', 4);
    const status = output('uint32', 1);
    const tests = output('float32', 6);
    const diagnosticStatus = output('uint32', 1);
    graph.add(
      new GPUProbitRegression({
        predictors: input('float32', predictors),
        predictorCount,
        response: input('float32', response),
        spatialDiagnostics: {
          weights: {
            offsets: input('uint32', offsets),
            neighbors: input('uint32', neighbors),
            weights: input('float32', weights)
          },
          output: {tests: tests.view, status: diagnosticStatus.view}
        },
        output: {
          coefficients: coefficients.view,
          standardErrors: errors.view,
          fitted: output('float32', ROW_COUNT).view,
          marginalEffects: effects.view,
          summary: summary.view,
          status: status.view
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    expectClose(await readFloat32(coefficients.buffer, 3), expected.coefficients, 2e-3);
    expectClose(await readFloat32(effects.buffer, 2), expected.effects, 2e-3);
    const summaryValues = await readFloat32(summary.buffer, 4);
    expect(summaryValues[0]).toBe(ROW_COUNT);
    expect(Math.abs(summaryValues[1] / expected.logLikelihood - 1)).toBeLessThan(1e-4);
    // Expected-information standard errors.
    const errorValues = await readFloat32(errors.buffer, 3);
    expected.standardErrors.forEach((value, index) =>
      expect(Math.abs(errorValues[index] / value - 1), `error ${index}`).toBeLessThan(1e-2)
    );
    // The smooth latent term makes the residuals spatially correlated along the chain.
    expect(Array.from(await readUint32(diagnosticStatus.buffer, 1))).toEqual([0]);
    const testValues = await readFloat32(tests.buffer, 6);
    expect(testValues[2]).toBeGreaterThan(3);
    expect(testValues[3]).toBeLessThan(1e-2);
  } finally {
    context.destroy();
  }
});
