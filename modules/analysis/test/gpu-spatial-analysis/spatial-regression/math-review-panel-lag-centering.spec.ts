// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  createNormalRandom,
  createRandom,
  createRegressionTestContext,
  fitLeastSquares,
  solveDense
} from './math-review-regression-test-utils';

const ENTITY_COUNT = 16;
const PERIOD_COUNT = 16;
const ROW_COUNT = ENTITY_COUNT * PERIOD_COUNT;

/** Ring lattice: each entity has its two neighbours with weight 1/2. */
function createRingWeights() {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let entity = 0; entity < ENTITY_COUNT; entity++) {
    const pair = [(entity + ENTITY_COUNT - 1) % ENTITY_COUNT, (entity + 1) % ENTITY_COUNT].sort(
      (first, second) => first - second
    );
    neighbors.push(...pair);
    weights.push(0.5, 0.5);
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors, weights};
}

/** Spatial lag of a period-major panel on the ring. */
function getRingLag(values: ArrayLike<number>): number[] {
  return Array.from({length: ROW_COUNT}, (_, row) => {
    const entity = row % ENTITY_COUNT;
    const start = row - entity;
    return (
      0.5 * values[start + ((entity + ENTITY_COUNT - 1) % ENTITY_COUNT)] +
      0.5 * values[start + ((entity + 1) % ENTITY_COUNT)]
    );
  });
}

/**
 * Random-effects spatial lag panel with rho 0.3, phi 0.3 and slope 0.5 on one predictor centered
 * at `center` (regression-04). Rows are period-major: row = period * entities + entity.
 */
function createPanelScene(center: number, halfRange: number, integerValues: boolean) {
  const random = createRandom(416);
  const normal = createNormalRandom(417);
  const rho = 0.3;
  // phi = sigma_e / sqrt(T sigma_mu^2 + sigma_e^2) = 0.3 with sigma_e = 1.
  const effectDeviation = Math.sqrt((1 / 0.09 - 1) / PERIOD_COUNT);
  const effects = Array.from({length: ENTITY_COUNT}, () => effectDeviation * normal());
  const predictor = new Float32Array(ROW_COUNT);
  const response = new Float32Array(ROW_COUNT);
  const scale = 4 / halfRange;
  for (let period = 0; period < PERIOD_COUNT; period++) {
    const signal: number[] = [];
    for (let entity = 0; entity < ENTITY_COUNT; entity++) {
      const row = period * ENTITY_COUNT + entity;
      const offset = (2 * random() - 1) * halfRange;
      predictor[row] = center + (integerValues ? Math.round(offset) : offset);
      // A slope of 0.5 per unit of the scaled spread keeps the response of order 1 to 10.
      signal.push(2 + 0.5 * scale * (predictor[row] - center) + effects[entity] + normal());
    }
    // y = (I - rho W)^-1 signal for this period.
    const matrix = Array.from({length: ENTITY_COUNT}, (_, entity) =>
      Array.from({length: ENTITY_COUNT}, (_, other) => {
        const isNeighbor =
          other === (entity + 1) % ENTITY_COUNT ||
          other === (entity + ENTITY_COUNT - 1) % ENTITY_COUNT;
        return (entity === other ? 1 : 0) - (isNeighbor ? rho * 0.5 : 0);
      })
    );
    const solved = solveDense(matrix, signal);
    for (let entity = 0; entity < ENTITY_COUNT; entity++)
      response[period * ENTITY_COUNT + entity] = solved[entity];
  }
  return {predictor, response, slope: 0.5 * scale};
}

/**
 * Float64 maximum of the random-effects spatial lag likelihood (Elhorst; spreg `Panel_RE_Lag`)
 * `-NT/2 ln(sigma2) + T ln|I - rho W| + N ln(phi)`, with `sigma2` the mean squared residual of
 * the quasi-demeaned (theta = 1 - phi) regression of `y - rho W y` on `[1, x]`. The maximum is
 * found by nested golden-section searches over rho and phi, not by the kernel's alternation, and
 * the log-determinant uses the ring's eigenvalues `cos(2 pi k / N)`.
 */
function fitPanelReference(predictor: Float32Array, response: Float32Array) {
  const lag = getRingLag(response);
  const getEntityMeans = (values: ArrayLike<number>) => {
    const means = new Array<number>(ENTITY_COUNT).fill(0);
    for (let row = 0; row < ROW_COUNT; row++)
      means[row % ENTITY_COUNT] += values[row] / PERIOD_COUNT;
    return means;
  };
  const predictorMeans = getEntityMeans(predictor);
  const responseMeans = getEntityMeans(response);
  const lagMeans = getEntityMeans(lag);
  const fit = (rho: number, phi: number) => {
    const design: number[][] = [];
    const target: number[] = [];
    for (let row = 0; row < ROW_COUNT; row++) {
      const entity = row % ENTITY_COUNT;
      design.push([phi, predictor[row] - (1 - phi) * predictorMeans[entity]]);
      target.push(
        response[row] -
          (1 - phi) * responseMeans[entity] -
          rho * (lag[row] - (1 - phi) * lagMeans[entity])
      );
    }
    const result = fitLeastSquares(design, target);
    let logDeterminant = 0;
    for (let index = 0; index < ENTITY_COUNT; index++)
      logDeterminant += Math.log(1 - rho * Math.cos((2 * Math.PI * index) / ENTITY_COUNT));
    const variance = result.squares / ROW_COUNT;
    return {
      estimate: result.estimate,
      variance,
      logLikelihood:
        (-ROW_COUNT / 2) * Math.log(variance) +
        PERIOD_COUNT * logDeterminant +
        ENTITY_COUNT * Math.log(phi)
    };
  };
  const maximize = (lower: number, upper: number, score: (value: number) => number) => {
    const ratio = (Math.sqrt(5) - 1) / 2;
    let low = lower;
    let high = upper;
    for (let step = 0; step < 80; step++) {
      const left = high - ratio * (high - low);
      const right = low + ratio * (high - low);
      if (score(left) >= score(right)) high = right;
      else low = left;
    }
    return (low + high) / 2;
  };
  const getBestPhi = (rho: number) => maximize(1e-4, 1, phi => fit(rho, phi).logLikelihood);
  const rho = maximize(-0.9, 0.9, value => fit(value, getBestPhi(value)).logLikelihood);
  const phi = getBestPhi(rho);
  const best = fit(rho, phi);
  return {intercept: best.estimate[0], slope: best.estimate[1], rho, phi, variance: best.variance};
}

async function runPanel(predictor: Float32Array, response: Float32Array) {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const ring = createRingWeights();
    const coefficients = output('float32', 4);
    const covariance = output('float32', 9);
    const summary = output('float32', 6);
    const status = output('uint32', 1);
    const fitted = output('float32', ROW_COUNT);
    const residuals = output('float32', ROW_COUNT);
    graph.add(
      new GPUSpatialPanelLagMaximumLikelihood({
        weights: {
          offsets: input('uint32', ring.offsets),
          neighbors: input('uint32', ring.neighbors),
          weights: input('float32', ring.weights)
        },
        periodCount: PERIOD_COUNT,
        predictorCount: 1,
        predictors: input('float32', predictor),
        response: input('float32', response),
        effects: 'random',
        maximumIterations: 128,
        tolerance: 2e-5,
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          summary: summary.view,
          status: status.view,
          fitted: fitted.view,
          residuals: residuals.view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from(await readFloat32(coefficients.buffer, 4)),
      covariance: Array.from(await readFloat32(covariance.buffer, 9)),
      summary: Array.from(await readFloat32(summary.buffer, 6)),
      fitted: Array.from(await readFloat32(fitted.buffer, ROW_COUNT)),
      residuals: Array.from(await readFloat32(residuals.buffer, ROW_COUNT))
    };
  } finally {
    context.destroy();
  }
}

for (const [name, center, halfRange, integerValues] of [
  ['an order-1 predictor (control)', 0, 2, false],
  ['a year (2010 +- 10)', 2010, 10, true],
  ['an elevation (1500 +- 3)', 1500, 3, false],
  ['an income (50000 +- 500)', 50000, 500, false]
] as const)
  it(`random-effects panel lag on ${name} matches the float64 likelihood maximum (regression-04)`, async () => {
    const scene = createPanelScene(center, halfRange, integerValues);
    const reference = fitPanelReference(scene.predictor, scene.response);
    expect(Math.abs(reference.slope / scene.slope - 1)).toBeLessThan(0.2);
    const result = await runPanel(scene.predictor, scene.response);
    expect(result.status).toBe(0);
    const [intercept, slope, rho, phi] = result.coefficients;
    const message = `GPU ${result.coefficients} reference ${[reference.intercept, reference.slope, reference.rho, reference.phi]}`;
    // The former raw Gram was 1.6%, 6.7% and 0.6% off in the slope on the three large-mean scenes.
    expect(Math.abs(slope / reference.slope - 1), message).toBeLessThan(2e-3);
    expect(Math.abs(rho - reference.rho), message).toBeLessThan(3e-3);
    expect(Math.abs(phi - reference.phi), message).toBeLessThan(3e-3);
    expect(Math.abs(intercept - reference.intercept), message).toBeLessThan(
      3e-3 * Math.max(1, Math.abs(reference.intercept))
    );
    expect(Math.abs(result.summary[0] / reference.variance - 1), message).toBeLessThan(2e-3);

    // Fitted values and innovations are those of the quasi-demeaned, uncentered data.
    const lag = getRingLag(scene.response);
    let largestError = 0;
    for (let row = 0; row < ROW_COUNT; row++) {
      const entity = row % ENTITY_COUNT;
      let quasiResponse = scene.response[row];
      let quasiLag = lag[row];
      let quasiPredictor = scene.predictor[row];
      for (let period = 0; period < PERIOD_COUNT; period++) {
        const other = period * ENTITY_COUNT + entity;
        quasiResponse -= ((1 - reference.phi) * scene.response[other]) / PERIOD_COUNT;
        quasiLag -= ((1 - reference.phi) * lag[other]) / PERIOD_COUNT;
        quasiPredictor -= ((1 - reference.phi) * scene.predictor[other]) / PERIOD_COUNT;
      }
      const expectedFitted =
        reference.rho * quasiLag +
        reference.intercept * reference.phi +
        reference.slope * quasiPredictor;
      largestError = Math.max(
        largestError,
        Math.abs(result.fitted[row] - expectedFitted),
        Math.abs(result.residuals[row] - (quasiResponse - expectedFitted))
      );
    }
    expect(largestError, 'fitted and residuals').toBeLessThan(3e-2);
  });

it('panel lag covariance of a shifted predictor is the congruence transform of the unshifted one', async () => {
  // Shifting the predictor by c leaves slope, rho and their covariance unchanged and moves the
  // intercept to `intercept - slope c`, so its covariance row is [1, -c, 0] V. The base scene is
  // integer-valued so that the shifted float32 predictor is exactly the base plus c.
  const base = createPanelScene(0, 10, true);
  const shift = 2010;
  const shifted = base.predictor.map(value => value + shift);
  const baseResult = await runPanel(base.predictor, base.response);
  const shiftedResult = await runPanel(shifted, base.response);
  expect(baseResult.status).toBe(0);
  expect(shiftedResult.status).toBe(0);
  const transform = [
    [1, -shift, 0],
    [0, 1, 0],
    [0, 0, 1]
  ];
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++) {
      let expected = 0;
      for (let first = 0; first < 3; first++)
        for (let second = 0; second < 3; second++)
          expected +=
            transform[row][first] *
            baseResult.covariance[first * 3 + second] *
            transform[column][second];
      const scale = Math.sqrt(
        Math.abs(shiftedResult.covariance[row * 4] * shiftedResult.covariance[column * 4])
      );
      expect(
        Math.abs(shiftedResult.covariance[row * 3 + column] - expected),
        `covariance ${row},${column}: ${shiftedResult.covariance[row * 3 + column]} vs ${expected}`
      ).toBeLessThan(5e-3 * scale);
    }
});
