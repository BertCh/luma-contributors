// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSpatialLogDeterminantDerivativeFromTraces,
  getSpatialMaximumLikelihoodTraceOne,
  getSpatialMaximumLikelihoodTraceTwo
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  SPATIAL_SERIES_TAIL_TERM_LIMIT,
  getSpatialSeriesTailModel,
  getSpatialSeriesTailWGSL
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-series-tail';
import {
  getSpatialLogDeterminantErrorBound,
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantWGSL
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {getSpatialWeightsTracesOnCPU} from '../../reference/spatial-weights-traces-cpu';
import {createLatticeWeights, toDenseWeights} from './spatial-regression-diagnostics-oracle';

/** Exact `ln|det(I - rho W)|` by Gaussian elimination with partial pivoting (float64). */
function getExactLogDeterminant(dense: number[][], rho: number): number {
  const size = dense.length;
  const matrix = dense.map((row, i) => row.map((value, j) => (i === j ? 1 : 0) - rho * value));
  let logDeterminant = 0;
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    logDeterminant += Math.log(Math.abs(matrix[column][column]));
    for (let row = column + 1; row < size; row++) {
      const factor = matrix[row][column] / matrix[column][column];
      for (let j = column; j < size; j++) {
        matrix[row][j] -= factor * matrix[column][j];
      }
    }
  }
  return logDeterminant;
}

const weights = createLatticeWeights(12, true);
const dense = toDenseWeights(weights) as number[][];
const maximumPower = 128;
const traces = getSpatialWeightsTracesOnCPU(weights, maximumPower).powerTraces;

it('tail extrapolation keeps the log-determinant accurate near the stationarity bound', () => {
  for (const rho of [0.9, 0.97, 0.99, -0.97]) {
    const exact = getExactLogDeterminant(dense, rho);
    const plain = getSpatialLogDeterminantFromTraces(rho, traces, 1);
    const extrapolated = getSpatialLogDeterminantFromTraces(rho, traces, 1, true);
    const plainError = Math.abs(plain - exact);
    const extrapolatedError = Math.abs(extrapolated - exact);
    expect(extrapolatedError, `rho ${rho}`).toBeLessThanOrEqual(plainError + 1e-9);
    if (rho >= 0.97) {
      // The plain series is off by 1e-2 or more here; the tail removes most of that.
      expect(plainError).toBeGreaterThan(1e-3);
      expect(extrapolatedError).toBeLessThan(plainError / 4);
    }
  }
});

it('tail extrapolation matches the derivative of the extrapolated log-determinant', () => {
  const step = 1e-5;
  for (const rho of [0.5, 0.95, 0.99]) {
    const numeric =
      (getSpatialLogDeterminantFromTraces(rho + step, traces, 1, true) -
        getSpatialLogDeterminantFromTraces(rho - step, traces, 1, true)) /
      (2 * step);
    const analytic = getSpatialLogDeterminantDerivativeFromTraces(rho, traces, 1, true);
    // The two tails use different closed forms of one geometric model, so they agree to the
    // model's bounded term count, not to float64 round-off.
    expect(Math.abs(numeric - analytic)).toBeLessThan(1e-3 * Math.max(1, Math.abs(analytic)));
  }
});

it('tail extrapolation is a no-op at small rho and keeps the trace series consistent', () => {
  const plain = getSpatialMaximumLikelihoodTraceOne(0.3, traces, 1);
  expect(getSpatialMaximumLikelihoodTraceOne(0.3, traces, 1, true)).toBeCloseTo(plain, 12);
  const plainTwo = getSpatialMaximumLikelihoodTraceTwo(0.3, traces, 1);
  expect(getSpatialMaximumLikelihoodTraceTwo(0.3, traces, 1, true)).toBeCloseTo(plainTwo, 12);
  const exactTraceOne = -getSpatialLogDeterminantDerivativeFromTraces(0.98, traces, 1, true);
  const step = 1e-6;
  const exactNumeric =
    -(getExactLogDeterminant(dense, 0.98 + step) - getExactLogDeterminant(dense, 0.98 - step)) /
    (2 * step);
  expect(
    Math.abs(exactTraceOne - exactNumeric),
    'extrapolated tr1 versus the exact derivative'
  ).toBeLessThan(Math.abs(getSpatialMaximumLikelihoodTraceOne(0.98, traces, 1) - exactNumeric) / 2);
});

it('estimates the tail model and generates WGSL for it', () => {
  const model = getSpatialSeriesTailModel(traces);
  expect(model.ratio).toBeGreaterThan(0.9);
  expect(model.ratio).toBeLessThanOrEqual(1);
  expect(model.level).toBeGreaterThan(0.9);
  expect(SPATIAL_SERIES_TAIL_TERM_LIMIT).toBeGreaterThan(0);
  const tail = getSpatialSeriesTailWGSL(maximumPower, 'powerTraces');
  expect(tail).toContain('fn getSeriesTailLogDeterminant');
  expect(getSpatialLogDeterminantWGSL(maximumPower, 'powerTraces', '1.0', true)).toContain(
    'getSeriesTailLogDeterminant'
  );
  expect(getSpatialLogDeterminantWGSL(maximumPower, 'powerTraces', '1.0')).not.toContain(
    'getSeriesTail'
  );
  expect(() => getSpatialSeriesTailWGSL(2, 'powerTraces')).toThrow();
});

it('bounds the stochastic log-determinant error from the trace standard errors', () => {
  const stochastic = getSpatialWeightsTracesOnCPU(weights, 32, {
    mode: 'stochastic',
    probeCount: 64,
    seed: 3
  });
  const exact = getSpatialWeightsTracesOnCPU(weights, 32);
  expect(getSpatialLogDeterminantErrorBound(0.5, exact.powerTraceStandardErrors, 1)).toBe(0);
  for (const rho of [0.3, 0.7]) {
    const error = Math.abs(
      getSpatialLogDeterminantFromTraces(rho, stochastic.powerTraces, 1) -
        getSpatialLogDeterminantFromTraces(rho, exact.powerTraces, 1)
    );
    const bound = getSpatialLogDeterminantErrorBound(rho, stochastic.powerTraceStandardErrors, 1);
    expect(bound).toBeGreaterThan(0);
    // A bound on the sampling error at one standard error per trace: the observed error is of the
    // same order, well inside three times the bound.
    expect(error).toBeLessThan(3 * bound);
  }
});
