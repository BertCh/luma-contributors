// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-parameters';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import fixture from '../../oracles/fixtures/statistics-regression-large-mean.json';

/**
 * Residual passes of the lag maximum-likelihood, generalized-moments and two-stage least squares
 * estimators on a response whose mean (10,000) is far above its residuals (0.002 to 0.004, two to
 * four float32 steps of the response): the shape that exposed the regime and error-ML residual
 * faults of the 2026-11-10 review.
 *
 * Reference numbers: `test/oracles/fixtures/statistics-regression-large-mean.json`, written by
 * `test/oracles/statistics-regression-large-mean.py` (its docstring has the full recipe).
 * In short: spreg 1.9.1 (`ML_Lag(method='full')`, `GM_Lag(w_lags=1, spat_diag=True)`,
 * `GM_Error_Hom(A1='hom_sc', max_iter=1)`) with libpysal 4.15.0, NumPy 2.5.4, SciPy 1.16.3 on
 * CPython 3.14, run in float64 on the same float32 inputs with 10,000 subtracted from the
 * response. On a row-standardized `W` that shift changes only the intercept, and it removes the
 * conditioning error of spreg's own raw normal equations (its unshifted `GM_Lag` residuals have
 * a mean of 2e-5). For `GM_Error` the reference is the exact minimum of spreg's moment objective
 * (`_momentsGM_Error`, golden section) followed by spreg's filtered regression, because spreg's
 * L-BFGS-B search stops within 2e-4 of it and ran to -0.99 on the shifted plain scene.
 *
 * The scenes use a 32-bit linear congruential stream and IEEE double arithmetic only, so the
 * Python generator rebuilds bit-identical float32 inputs (checked through `responseSum`).
 *
 * Before the closing pass (2026-11-11), on the test GPU:
 * - lag ML: `sigma2` 19.6% high (plain) and the log-likelihood 53 units low, from the float32
 *   response mean of the composed OLS fits; the standard errors of `rho` and of the intercept
 *   half (plain) and a third (lag) of the reference, from the uncentered `W A X b` column;
 * - two-stage least squares: `sigma2` 19.6% high (plain), Anselin-Kelejian 16.25 against 1.98;
 * - GM (Kelejian-Prucha): `lambda` 0.192 against 0.060, `sigma2` 5.4% high (plain);
 * - GM (homoskedastic): `sigma2` 1.5% high (plain), 0.8% high (error).
 */
const ROWS = 600;
const RESPONSE_CENTER = 10000;

/** Uniform stream in [0, 1): a 32-bit linear congruential state, top 24 bits. */
function createStream(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) / 16777216;
  };
}

/** Ring lattice spatial lag: the mean of the two neighbours. */
function getRingLag(values: ArrayLike<number>): number[] {
  return Array.from(
    {length: ROWS},
    (_, row) => 0.5 * values[(row + ROWS - 1) % ROWS] + 0.5 * values[(row + 1) % ROWS]
  );
}

/** `sum_k parameter^k W^k values`, 80 terms in a fixed order. */
function getInverseSeries(values: number[], parameter: number): number[] {
  const total = [...values];
  let term = [...values];
  for (let power = 1; power < 80; power++) {
    term = getRingLag(term).map(value => parameter * value);
    for (let row = 0; row < ROWS; row++) total[row] += term[row];
  }
  return total;
}

type SceneKind = 'plain' | 'lag' | 'lag-unrounded' | 'error';

/**
 * `plain`: `y = 10000 + 2 x + 0.002 e`. `lag`: `y = (I - 0.4 W)^-1 (6000 + 2 x + 0.004 e)`,
 * rounded to multiples of `2^-9` so that the float32 ring lag `0.5 a + 0.5 b` is exact near
 * 10,000 (`lag-unrounded` skips that rounding). `error`: `y = 10000 + 2 x + (I - 0.5 W)^-1 0.004 e`.
 */
function createScene(kind: SceneKind) {
  const uniformPredictor = createStream(20261111);
  const uniformNoise = createStream(77003);
  const predictor = new Float32Array(ROWS);
  const noise: number[] = [];
  for (let row = 0; row < ROWS; row++) predictor[row] = 4 * uniformPredictor() - 2;
  for (let row = 0; row < ROWS; row++) {
    const total = uniformNoise() + uniformNoise() + uniformNoise() + uniformNoise();
    noise.push((total - 2) * 1.7320508075688772);
  }
  const response = new Float32Array(ROWS);
  if (kind === 'plain') {
    for (let row = 0; row < ROWS; row++)
      response[row] = RESPONSE_CENTER + 2 * predictor[row] + 0.002 * noise[row];
  } else if (kind === 'error') {
    const error = getInverseSeries(
      noise.map(value => 0.004 * value),
      0.5
    );
    for (let row = 0; row < ROWS; row++)
      response[row] = RESPONSE_CENTER + 2 * predictor[row] + error[row];
  } else {
    const reduced = noise.map(
      (value, row) => 0.6 * RESPONSE_CENTER + 2 * predictor[row] + 0.004 * value
    );
    const series = getInverseSeries(reduced, 0.4);
    for (let row = 0; row < ROWS; row++)
      response[row] = kind === 'lag' ? Math.floor(series[row] * 512 + 0.5) / 512 : series[row];
  }
  return {predictor, response};
}

function getRingWeights() {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < ROWS; row++) {
    const pair = [(row + ROWS - 1) % ROWS, (row + 1) % ROWS].sort(
      (first, second) => first - second
    );
    neighbors.push(...pair);
    weights.push(0.5, 0.5);
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors, weights};
}

type Estimator = 'lag-maximum-likelihood' | 'two-stage' | 'kelejian-prucha' | 'homoskedastic';

async function runEstimator(estimator: Estimator, scene: ReturnType<typeof createScene>) {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const ring = getRingWeights();
    const weights = {
      offsets: input('uint32', ring.offsets),
      neighbors: input('uint32', ring.neighbors),
      weights: input('float32', ring.weights)
    };
    const table = output('float32', 12);
    const summaryLength =
      estimator === 'lag-maximum-likelihood'
        ? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
        : estimator === 'two-stage'
          ? GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
          : GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH;
    const summary = output('float32', summaryLength);
    const status = output('uint32', 1);
    const residuals = output('float32', ROWS);
    const common = {
      weights,
      predictors: input('float32', scene.predictor),
      predictorCount: 1,
      response: input('float32', scene.response),
      output: {
        table: table.view,
        summary: summary.view,
        status: {code: status.view},
        residuals: residuals.view
      }
    };
    if (estimator === 'lag-maximum-likelihood')
      graph.add(new GPUSpatialLagMaximumLikelihood(common));
    else if (estimator === 'two-stage') graph.add(new GPUSpatialTwoStageLeastSquares(common));
    else graph.add(new GPUSpatialErrorGM({...common, estimator}));
    context.run();
    const tableValues = await readFloat32(table.buffer, 12);
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: [tableValues[0], tableValues[4], tableValues[8]],
      standardErrors: [tableValues[1], tableValues[5], tableValues[9]],
      summary: Array.from(await readFloat32(summary.buffer, summaryLength)),
      residuals: Array.from(await readFloat32(residuals.buffer, ROWS))
    };
  } finally {
    context.destroy();
  }
}

type Reference = {
  estimator: Estimator;
  scene: SceneKind;
  /** Sum of the float32 responses in float64: the generator and the spec built the same inputs. */
  responseSum: number;
  /** Intercept of the fit of `y - 10000`, slope, spatial parameter. */
  shiftedIntercept: number;
  slope: number;
  parameter: number;
  /**
   * Standard errors of the intercept of the unshifted model, the slope and the parameter; null
   * where the estimator reports none (`lambda` of the Kelejian-Prucha estimator).
   */
  standardErrors: (number | null)[];
  sigmaSquared: number;
  residualSumOfSquares: number;
  logLikelihood?: number;
  anselinKelejian?: number;
  /** Kelejian-Prucha only: `lambda` of spreg's own `GM_Error` search (see `REFERENCE_NAMES`). */
  spregOptimizerLambda?: number;
};

/** Eight fits: each estimator on two scenes (see the generator's docstring). */
const REFERENCES = fixture.references as Reference[];

/**
 * What each estimator is held to. Three are spreg estimates. The Kelejian-Prucha reference is
 * not spreg's `GM_Error` estimate: it is the exact minimum of spreg's moment objective
 * (`_momentsGM_Error`, grid and golden section in the generator) followed by spreg's filtered
 * regression, because spreg's own L-BFGS-B search stops within 2e-4 of that minimum and ran to
 * -0.99 on the shifted plain scene. spreg's own `lambda` is asserted separately, at the
 * distance its search leaves, on the scene where it is near the minimum.
 */
const REFERENCE_NAMES: Record<Estimator, string> = {
  'lag-maximum-likelihood': 'matches spreg ML_Lag in float64',
  'two-stage': 'matches spreg GM_Lag in float64',
  homoskedastic: 'matches spreg GM_Error_Hom in float64',
  'kelejian-prucha':
    "matches the exact minimum of spreg's GM_Error moment objective in float64 and spreg's own lambda to its search tolerance"
};

const getRelativeError = (value: number, reference: number) => Math.abs(value / reference - 1);

for (const reference of REFERENCES)
  it(`${reference.estimator} on the ${reference.scene} scene with a response mean of 10000 ${REFERENCE_NAMES[reference.estimator]}`, async () => {
    const scene = createScene(reference.scene);
    expect(Array.from(scene.response).reduce((sum, value) => sum + value, 0)).toBe(
      reference.responseSum
    );
    const result = await runEstimator(reference.estimator, scene);
    const isLagModel =
      reference.estimator === 'lag-maximum-likelihood' || reference.estimator === 'two-stage';
    const [sigmaSlot, squaresSlot] =
      reference.estimator === 'lag-maximum-likelihood'
        ? [
            GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
            GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES
          ]
        : reference.estimator === 'two-stage'
          ? [
              GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
              GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES
            ]
          : [
              GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED,
              GPU_SPATIAL_ERROR_GM_SUMMARY_RESIDUAL_SUM_OF_SQUARES
            ];
    const message = `GPU ${JSON.stringify({
      coefficients: result.coefficients,
      standardErrors: result.standardErrors,
      sigmaSquared: result.summary[sigmaSlot],
      residualSumOfSquares: result.summary[squaresSlot]
    })}`;
    expect(result.status, message).toBe(0);

    // Coefficients. The spatial parameter is held to a hundredth of its standard error for the
    // lag models; lambda of the moment estimators sits on a flat objective and is held to 1e-4.
    const parameterError = Math.abs(result.coefficients[2] - reference.parameter);
    const parameterStandardError = reference.standardErrors[2];
    expect(parameterError, message).toBeLessThan(
      isLagModel && parameterStandardError !== null ? 0.01 * parameterStandardError : 1e-4
    );
    // The library's own Kelejian-Prucha lambda. Its search stops 1.8e-4 from the minimum on the
    // error scene (0.53285 against 0.53267), so it bounds the kernel at 5e-4; on the plain scene
    // it ran to the edge of its interval (-0.99) and bounds nothing.
    if (reference.spregOptimizerLambda !== undefined && reference.scene === 'error') {
      expect(Math.abs(reference.spregOptimizerLambda - reference.parameter)).toBeGreaterThan(1e-4);
      expect(
        Math.abs(result.coefficients[2] - reference.spregOptimizerLambda),
        message
      ).toBeLessThan(5e-4);
    }
    expect(getRelativeError(result.coefficients[1], reference.slope), message).toBeLessThan(2e-6);
    // The intercept of the unshifted model: a lag model moves it by 10000 (1 - rho).
    const intercept =
      reference.shiftedIntercept + RESPONSE_CENTER * (isLagModel ? 1 - reference.parameter : 1);
    expect(Math.abs(result.coefficients[0] - intercept), message).toBeLessThan(5e-3);

    // Residual variance and sum of squares: 19.6% and 0.2% to 6% off before the closing pass.
    expect(
      getRelativeError(result.summary[sigmaSlot], reference.sigmaSquared),
      message
    ).toBeLessThan(1e-4);
    expect(
      getRelativeError(result.summary[squaresSlot], reference.residualSumOfSquares),
      message
    ).toBeLessThan(1e-4);

    // Standard errors (the intercept and rho of the lag ML fit were a half to a third of these).
    for (let row = 0; row < 3; row++) {
      const standardError = reference.standardErrors[row];
      if (standardError === null) {
        expect(Number.isNaN(result.standardErrors[row]), message).toBe(true);
      } else {
        expect(
          getRelativeError(result.standardErrors[row], standardError),
          `${message} standard error ${row}`
        ).toBeLessThan(1e-3);
      }
    }
    if (reference.logLikelihood !== undefined) {
      expect(
        Math.abs(
          result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD] -
            reference.logLikelihood
        ),
        message
      ).toBeLessThan(0.02);
    }
    if (reference.anselinKelejian !== undefined) {
      expect(
        getRelativeError(
          result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN],
          reference.anselinKelejian
        ),
        message
      ).toBeLessThan(1e-3);
    }

    // Row by row against the float64 residuals at the reference coefficients: no common offset
    // (the float32 mean) and no per-row rounding to a float32 step of the response (a regrouped
    // residual chain). Both were up to half a step, 0.0005, a quarter of the residual deviation.
    const shifted = Array.from(scene.response, value => value - RESPONSE_CENTER);
    const lagged = getRingLag(shifted);
    const errors = result.residuals.map(
      (value, row) =>
        value -
        (shifted[row] -
          reference.shiftedIntercept -
          reference.slope * scene.predictor[row] -
          (isLagModel ? reference.parameter * lagged[row] : 0))
    );
    const deviation = Math.sqrt(reference.residualSumOfSquares / ROWS);
    expect(
      Math.max(...errors.map(Math.abs)) / deviation,
      `${message} largest residual error over the residual deviation`
    ).toBeLessThan(0.01);
  });

/**
 * The lag of a response that is not exact in float32. A sum of `w * y` over the raw response
 * near 10,000 is rounded to a float32 step of 10,000 (0.001) unless it happens to be exact, as
 * in the `lag` scene above; with residuals of 0.004 and `rho = 0.4` that rounding added 0.345%
 * to the residual variance of both estimators, and up to 4.2% on other weights
 * (frc-regression-02 of the fix review). Both kernels now lag the response about its mean and
 * carry the row sums of `W` exactly, so this scene meets the same bound as its rounded twin.
 * Reference: spreg 1.9.1 `sig2` of `ML_Lag` and `GM_Lag` on the same unrounded inputs
 * (`unroundedLag` of the fixture). The bound over other weights, `rho` and residual scales is
 * pinned by `math-review-lag-conditioning.spec.ts`.
 */
for (const [estimator, slot, sigmaSquared] of [
  [
    'lag-maximum-likelihood',
    GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
    fixture.unroundedLag.lagMaximumLikelihoodSigmaSquared
  ],
  [
    'two-stage',
    GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
    fixture.unroundedLag.twoStageSigmaSquared
  ]
] as const)
  it(`${estimator} with a spatial lag of a response near 10000 that is inexact in float32 matches the spreg residual variance to 4e-5`, async () => {
    const scene = createScene('lag-unrounded');
    expect(Array.from(scene.response).reduce((sum, value) => sum + value, 0)).toBe(
      fixture.unroundedLag.responseSum
    );
    const result = await runEstimator(estimator, scene);
    expect(result.status).toBe(0);
    expect(
      getRelativeError(result.summary[slot], sigmaSquared),
      `sigma2 ${result.summary[slot]} rho ${result.coefficients[2]}`
    ).toBeLessThan(4e-5);
  });
