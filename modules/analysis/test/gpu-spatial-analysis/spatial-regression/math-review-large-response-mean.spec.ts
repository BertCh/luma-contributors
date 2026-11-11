// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {
  createNormalRandom,
  createRandom,
  createRegressionTestContext,
  fitLeastSquares
} from './math-review-regression-test-utils';
import fixture from '../../oracles/fixtures/statistics-regression-large-mean.json';

/**
 * A response whose mean is far above its residuals: `y = 10000 + 2 x + 0.002 e`. The float32 step
 * of 10,000 is 0.00098, half the residual deviation.
 *
 * Two things can go wrong here and they look different. A residual chain regrouped by the shader
 * compiler as `y - (mean + fit)` rounds every residual separately to a float32 step of the
 * response (seen in the regime and error-ML kernels before `getRoundedDifference`). The float32
 * mean of the response shifts every residual by one common offset of up to a float32 step of
 * that mean, which inflated the residual sum of squares by `n` times its square: 4% on the
 * first scene and 19.6% on the second before the closing pass of 2026-11-11.
 * `GPUOrdinaryLeastSquares` now carries the part of each mean that float32 cannot hold (the sums
 * of the deviations from the stored means) through the moments and the residuals, so both
 * scenes must match float64.
 */
function createScene() {
  const random = createRandom(91);
  const normal = createNormalRandom(92);
  const rows = 600;
  const predictor = new Float32Array(rows);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    predictor[row] = 2 * random() - 0.5;
    response[row] = 10000 + 2 * predictor[row] + 0.002 * normal();
  }
  return {rows, predictor, response};
}

async function runOrdinaryLeastSquares(predictor: Float32Array, response: Float32Array) {
  const rows = response.length;
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const summary = output('float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH);
    const status = output('uint32', 1);
    const residuals = output('float32', rows);
    const fitted = output('float32', rows);
    const coefficients = output('float32', 2);
    const standardErrors = output('float32', 2);
    graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: input('float32', predictor),
        response: input('float32', response),
        predictorCount: 1,
        output: {
          coefficients: coefficients.view,
          standardErrors: standardErrors.view,
          tStatistics: output('float32', 2).view,
          summary: summary.view,
          status: {code: status.view},
          residuals: residuals.view,
          fitted: fitted.view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      summary: await readFloat32(summary.buffer, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
      residuals: await readFloat32(residuals.buffer, rows),
      fitted: await readFloat32(fitted.buffer, rows),
      coefficients: await readFloat32(coefficients.buffer, 2),
      standardErrors: await readFloat32(standardErrors.buffer, 2)
    };
  } finally {
    context.destroy();
  }
}

/**
 * Largest error of the residuals against float64 residuals at the reference coefficients, and
 * their mean error (the common offset), both over the residual standard deviation.
 */
function getResidualErrors(
  residuals: ArrayLike<number>,
  predictor: Float32Array,
  response: Float32Array,
  estimate: number[],
  squares: number
) {
  const errors = Array.from(
    residuals,
    (value, row) => value - (response[row] - estimate[0] - estimate[1] * predictor[row])
  );
  const deviation = Math.sqrt(squares / residuals.length);
  return {
    offset: errors.reduce((sum, value) => sum + value, 0) / residuals.length / deviation,
    largest: Math.max(...errors.map(Math.abs)) / deviation
  };
}

it('OLS of a response with a large mean matches float64 least squares: residuals, their sum of squares and the report', async () => {
  const scene = createScene();
  const reference = fitLeastSquares(
    Array.from(scene.predictor, value => [1, value]),
    Array.from(scene.response)
  );
  const result = await runOrdinaryLeastSquares(scene.predictor, scene.response);
  expect(result.status).toBe(0);
  expect(Math.abs(result.coefficients[1] / reference.estimate[1] - 1)).toBeLessThan(2e-6);
  // The float32 intercept is within a step of 10,000.
  expect(Math.abs(result.coefficients[0] - reference.estimate[0])).toBeLessThan(1e-3);
  // Row by row: no common offset (it was up to a float32 step, half the residual deviation)
  // and no per-row rounding to a step of the response.
  const errors = getResidualErrors(
    result.residuals,
    scene.predictor,
    scene.response,
    reference.estimate,
    reference.squares
  );
  expect(Math.abs(errors.offset), `offset ${errors.offset}`).toBeLessThan(1e-3);
  expect(errors.largest, `largest residual error ${errors.largest}`).toBeLessThan(5e-3);
  // The residual sum of squares was 4% high on this scene.
  const squares = result.summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES];
  expect(
    Math.abs(squares / reference.squares - 1),
    `RSS ${squares} against ${reference.squares}`
  ).toBeLessThan(1e-4);
  expect(
    Math.abs(
      result.summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED] /
        (reference.squares / (scene.rows - 2)) -
        1
    )
  ).toBeLessThan(1e-4);
  // Standard errors: sqrt(sigma2 diag((X'X)^-1)) in float64.
  const sigmaSquared = reference.squares / (scene.rows - 2);
  for (let column = 0; column < 2; column++) {
    expect(
      Math.abs(
        result.standardErrors[column] /
          Math.sqrt(sigmaSquared * reference.inverse[column][column]) -
          1
      ),
      `standard error ${column}`
    ).toBeLessThan(1e-4);
  }
  // Fitted values plus residuals give back the response to a float32 step of the response.
  for (let row = 0; row < scene.rows; row += 37) {
    expect(Math.abs(result.fitted[row] + result.residuals[row] - scene.response[row])).toBeLessThan(
      1e-3
    );
  }
});

/**
 * `ordinaryLeastSquares` of `test/oracles/fixtures/statistics-regression-large-mean.json`, written
 * by `test/oracles/statistics-regression-large-mean.py`:
 * spreg 1.9.1 `OLS(y, x, nonspat_diag=True)` (libpysal 4.15.0, NumPy 2.5.4, SciPy 1.16.3,
 * CPython 3.14) in float64 on the float32 inputs of the `plain` scene of
 * `math-review-closing-residuals.spec.ts`: `y = 10000 + 2 x + 0.002 e`, 600 rows, from a 32-bit
 * linear congruential stream that the Python generator repeats bit for bit (`responseSum`).
 * Here the float32 sum of the responses loses most of a step, so the stored mean missed the true
 * one by 0.9 of a float32 step and the residual sum of squares was 19.6% high.
 *
 * The two heteroskedasticity tests are not spreg's: spreg regresses the squared residuals on the
 * squared regressors, this class on `[1, X]` (its documented definition). They are computed from
 * that definition with NumPy 2.5.4 least squares in float64 on the same inputs: Koenker-Bassett
 * is `n R^2` of `e^2` on `[1, x]`, Breusch-Pagan half the explained sum of squares of
 * `e^2 / (e'e / n) - 1` on `[1, x]`.
 */
const SPREG_REFERENCE = fixture.ordinaryLeastSquares;

function createSpregScene() {
  const createStream = (seed: number) => {
    let state = seed >>> 0;
    return () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state >>> 8) / 16777216;
    };
  };
  const uniformPredictor = createStream(20261111);
  const uniformNoise = createStream(77003);
  const rows = 600;
  const predictor = new Float32Array(rows);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) predictor[row] = 4 * uniformPredictor() - 2;
  for (let row = 0; row < rows; row++) {
    const total = uniformNoise() + uniformNoise() + uniformNoise() + uniformNoise();
    response[row] = 10000 + 2 * predictor[row] + 0.002 * ((total - 2) * 1.7320508075688772);
  }
  return {rows, predictor, response};
}

it('OLS of a response with a large mean matches spreg OLS in float64, with its diagnostics', async () => {
  const scene = createSpregScene();
  expect(Array.from(scene.response).reduce((sum, value) => sum + value, 0)).toBe(
    SPREG_REFERENCE.responseSum
  );
  const result = await runOrdinaryLeastSquares(scene.predictor, scene.response);
  expect(result.status).toBe(0);
  const getRelativeError = (value: number, reference: number) => Math.abs(value / reference - 1);
  const summary = result.summary;
  const message = `RSS ${summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES]} sigma2 ${summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED]}`;
  expect(
    getRelativeError(
      summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES],
      SPREG_REFERENCE.residualSumOfSquares
    ),
    message
  ).toBeLessThan(1e-4);
  expect(
    getRelativeError(
      summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED],
      SPREG_REFERENCE.sigmaSquared
    ),
    message
  ).toBeLessThan(1e-4);
  expect(getRelativeError(result.coefficients[1], SPREG_REFERENCE.coefficients[1])).toBeLessThan(
    2e-6
  );
  expect(Math.abs(result.coefficients[0] - SPREG_REFERENCE.coefficients[0])).toBeLessThan(1e-3);
  for (let column = 0; column < 2; column++) {
    expect(
      getRelativeError(result.standardErrors[column], SPREG_REFERENCE.standardErrors[column])
    ).toBeLessThan(1e-4);
  }
  // 1 - R^2 is 7.6e-7, a dozen float32 steps of 1: R^2 itself is good to one step.
  expect(
    Math.abs(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED] - SPREG_REFERENCE.rSquared)
  ).toBeLessThan(1.2e-7);
  // The log-likelihood was 53 units low (n / 2 ln 1.196).
  expect(
    Math.abs(
      summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD] - SPREG_REFERENCE.logLikelihood
    )
  ).toBeLessThan(0.02);
  // Residual-moment diagnostics: a common offset in the residuals moved all three.
  expect(
    getRelativeError(
      summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA],
      SPREG_REFERENCE.jarqueBera
    )
  ).toBeLessThan(2e-3);
  expect(
    getRelativeError(
      summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT],
      SPREG_REFERENCE.koenkerBassett
    )
  ).toBeLessThan(2e-3);
  expect(
    getRelativeError(
      summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN],
      SPREG_REFERENCE.breuschPagan
    )
  ).toBeLessThan(2e-3);
  const errors = getResidualErrors(
    result.residuals,
    scene.predictor,
    scene.response,
    SPREG_REFERENCE.coefficients,
    SPREG_REFERENCE.residualSumOfSquares
  );
  expect(Math.abs(errors.offset), `offset ${errors.offset}`).toBeLessThan(1e-3);
  expect(errors.largest, `largest residual error ${errors.largest}`).toBeLessThan(5e-3);
});
