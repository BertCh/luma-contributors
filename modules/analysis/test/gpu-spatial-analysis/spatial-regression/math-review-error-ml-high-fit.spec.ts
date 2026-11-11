// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  createNormalRandom,
  createRegressionTestContext,
  fitLeastSquares
} from './math-review-regression-test-utils';

const ROW_COUNT = 2000;

/** Ring lattice spatial lag: the mean of the two neighbours. */
function getRingLag(values: ArrayLike<number>): number[] {
  return Array.from(
    {length: ROW_COUNT},
    (_, row) =>
      0.5 * values[(row + ROW_COUNT - 1) % ROW_COUNT] + 0.5 * values[(row + 1) % ROW_COUNT]
  );
}

/**
 * `y = intercept + 2 x + u`, `u = (I - 0.5 W)^-1 e` on a ring of 2000 rows, with the innovation
 * deviation chosen so that `1 - R^2` is about `unexplained` (regression-03).
 */
function createScene(center: number, unexplained: number, seed: number, responseCenter = 1) {
  const normal = createNormalRandom(seed);
  const predictor = new Float32Array(ROW_COUNT);
  const innovation: number[] = [];
  for (let row = 0; row < ROW_COUNT; row++) {
    predictor[row] = center + normal();
    // Var(2 x) = 4 and Var(u) is about 1.15 sigma^2 at lambda 0.5.
    innovation.push(Math.sqrt((4 * unexplained) / 1.15) * normal());
  }
  // u = sum_k (0.5 W)^k e, truncated where 0.5^k is below float64 rounding.
  let term = innovation;
  const error = [...innovation];
  for (let power = 1; power < 70; power++) {
    term = getRingLag(term).map(value => 0.5 * value);
    for (let row = 0; row < ROW_COUNT; row++) error[row] += term[row];
  }
  const response = new Float32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++)
    response[row] = responseCenter + 2 * (predictor[row] - center) + error[row];
  return {predictor, response};
}

/**
 * Float64 spatial error maximum likelihood (Anselin 1988; spreg `ML_Error`): for each lambda the
 * filtered regression of `y - lambda W y` on `[1 - lambda, x - lambda W x]` gives `ee(lambda)`
 * from its per-row residuals, and the concentrated likelihood
 * `-n/2 ln(ee / n) + ln|I - lambda W|` is maximized by golden section. The log-determinant uses
 * the ring's eigenvalues `cos(2 pi k / n)`.
 */
function fitReference(predictor: Float32Array, response: Float32Array) {
  const lagPredictor = getRingLag(predictor);
  const lagResponse = getRingLag(response);
  const fit = (lambda: number) => {
    const design: number[][] = [];
    const target: number[] = [];
    for (let row = 0; row < ROW_COUNT; row++) {
      design.push([1 - lambda, predictor[row] - lambda * lagPredictor[row]]);
      target.push(response[row] - lambda * lagResponse[row]);
    }
    const result = fitLeastSquares(design, target);
    let logDeterminant = 0;
    for (let index = 0; index < ROW_COUNT; index++)
      logDeterminant += Math.log(1 - lambda * Math.cos((2 * Math.PI * index) / ROW_COUNT));
    const variance = result.squares / ROW_COUNT;
    return {
      estimate: result.estimate,
      variance,
      concentrated: (-ROW_COUNT / 2) * Math.log(variance) + logDeterminant
    };
  };
  const ratio = (Math.sqrt(5) - 1) / 2;
  let low = -0.9;
  let high = 0.9;
  for (let step = 0; step < 80; step++) {
    const left = high - ratio * (high - low);
    const right = low + ratio * (high - low);
    if (fit(left).concentrated >= fit(right).concentrated) high = right;
    else low = left;
  }
  const lambda = (low + high) / 2;
  const best = fit(lambda);
  return {
    lambda,
    intercept: best.estimate[0],
    slope: best.estimate[1],
    variance: best.variance,
    logLikelihood: best.concentrated - (ROW_COUNT / 2) * (Math.log(2 * Math.PI) + 1),
    getConcentrated: (value: number) => fit(value).concentrated
  };
}

async function runErrorMaximumLikelihood(predictor: Float32Array, response: Float32Array) {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const offsets = [0];
    const neighbors: number[] = [];
    const weights: number[] = [];
    for (let row = 0; row < ROW_COUNT; row++) {
      const pair = [(row + ROW_COUNT - 1) % ROW_COUNT, (row + 1) % ROW_COUNT].sort(
        (first, second) => first - second
      );
      neighbors.push(...pair);
      weights.push(0.5, 0.5);
      offsets.push(neighbors.length);
    }
    const table = output('float32', 12);
    const summary = output('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH);
    const status = output('uint32', 1);
    graph.add(
      new GPUSpatialErrorMaximumLikelihood({
        weights: {
          offsets: input('uint32', offsets),
          neighbors: input('uint32', neighbors),
          weights: input('float32', weights)
        },
        predictors: input('float32', predictor),
        predictorCount: 1,
        response: input('float32', response),
        output: {table: table.view, summary: summary.view, status: {code: status.view}}
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      table: Array.from(await readFloat32(table.buffer, 12)),
      summary: Array.from(
        await readFloat32(summary.buffer, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
      )
    };
  } finally {
    context.destroy();
  }
}

for (const [name, center, unexplained, responseCenter] of [
  ['R^2 = 0.9 (control)', 0, 0.1, 1],
  ['R^2 = 0.9999', 0, 1e-4, 1],
  ['1 - R^2 = 6e-6', 0, 6e-6, 1],
  ['R^2 = 0.9999 and a predictor mean of 1500', 1500, 1e-4, 1],
  // Residuals of about two float32 steps of the response: the first-stage residual must be
  // formed as (y - mean) - fit, not regrouped by the compiler as y - (mean + fit).
  ['1 - R^2 = 1e-6 and a response mean of 10000', 0, 1e-6, 10000]
] as const)
  it(`spatial error ML at ${name} matches the float64 likelihood maximum (regression-03)`, async () => {
    const scene = createScene(center, unexplained, 303, responseCenter);
    const reference = fitReference(scene.predictor, scene.response);
    const result = await runErrorMaximumLikelihood(scene.predictor, scene.response);
    expect(result.status).toBe(0);
    const lambda = result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER];
    const variance = result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED];
    const logLikelihood = result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD];
    const message = `GPU lambda ${lambda} s2 ${variance} LL ${logLikelihood} slope ${result.table[4]}; reference lambda ${reference.lambda} s2 ${reference.variance} LL ${reference.logLikelihood} slope ${reference.slope}`;
    // The former `yy - 2 b'r + b'Ab` was 0.2% off in ee at R^2 = 0.9999 and 4% off at
    // 1 - R^2 = 6e-6 (2 and 40 log-likelihood units of noise in the lambda search).
    expect(Math.abs(variance / reference.variance - 1), message).toBeLessThan(5e-4);
    expect(Math.abs(logLikelihood - reference.logLikelihood), message).toBeLessThan(0.5);
    // The estimated lambda is within the float32 flat top of the likelihood: its float64
    // concentrated likelihood is within 0.01 of the maximum.
    expect(
      reference.getConcentrated(reference.lambda) - reference.getConcentrated(lambda),
      message
    ).toBeLessThan(0.01);
    expect(Math.abs(lambda - reference.lambda), message).toBeLessThan(5e-3);
    expect(Math.abs(result.table[4] / reference.slope - 1), message).toBeLessThan(2e-5);
    expect(Math.abs(result.table[0] - reference.intercept), message).toBeLessThan(
      1e-4 * Math.max(1, Math.abs(reference.intercept))
    );
  });
