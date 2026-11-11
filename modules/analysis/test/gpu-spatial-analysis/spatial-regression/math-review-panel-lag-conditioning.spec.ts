// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import fixture from '../../oracles/fixtures/statistics-regression-panel-lag-conditioning.json';

/**
 * The panel spatial lag estimator on fits whose residual sum of squares is a small fraction of
 * the response's: frc-regression-09 of the re-review in
 * `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-regression.md`.
 *
 * The kernel published `sigma2 = (e0'e0 - 2 rho e0'e1 + rho^2 e1'e1) / n` from three float32
 * sums, the form frc-regression-01 removed from the cross-section kernel, and searched `rho` on
 * the same difference. Before the fix, on the test GPU (unit effects, random effects):
 * `slope-2` (`u'u` 3e-5 of `e0'e0`) `sigma2` 1% to 2% off; `slope-20` (3e-7) -36% and +96%;
 * the eight `slope-200` scenes (3e-9) status 0 with `sigma2` up to hundreds of times the
 * reference, or status 2 and 3.
 *
 * Reference numbers: `test/oracles/fixtures/statistics-regression-panel-lag-conditioning.json`,
 * written by `test/oracles/statistics-regression-panel-lag-conditioning.py`: spreg 1.9.1
 * `Panel_FE_Lag` and `Panel_RE_Lag` (`epsilon=1e-9`) in float64 on the float32 inputs, which the
 * generator holds to float64 maxima of the two likelihoods found another way.
 *
 * The scenes are rebuilt here from the parameters stored in the fixture, with a 32-bit linear
 * congruential stream and IEEE double arithmetic in the generator's order (`responseSum` checks
 * that the inputs are the same bits).
 */
const ENTITY_COUNT = fixture.entityCount;
const PERIOD_COUNT = fixture.periodCount;
const ROW_COUNT = ENTITY_COUNT * PERIOD_COUNT;

type Scene = (typeof fixture.scenes)[number];
/** What `createScene` reads; `opposingEffect` is the scene of the third round (see below). */
type SceneParameters = Pick<Scene, 'seed' | 'center' | 'slope' | 'deviation' | 'responseSum'> & {
  opposingEffect?: number;
  /** Entities per run of one sign of `opposingEffect` (1: alternating). */
  effectRunLength?: number;
};

/** Uniform stream in [0, 1): a 32-bit linear congruential state, top 24 bits. */
function createStream(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) / 16777216;
  };
}

/** Lag of one period on the ring: `0.5 previous + 0.5 next`. */
function getRingLag(values: number[]): number[] {
  return values.map(
    (_, entity) =>
      0.5 * values[(entity + ENTITY_COUNT - 1) % ENTITY_COUNT] +
      0.5 * values[(entity + 1) % ENTITY_COUNT]
  );
}

/** `y_t = (I - 0.4 W)^-1 (0.6 center + slope x_t + effects + deviation e_t)`, period-major. */
function createScene(scene: SceneParameters) {
  const getNoiseUniform = createStream(scene.seed);
  const getPredictorUniform = createStream(scene.seed + 5000);
  const getDeviate = () => {
    const total = getNoiseUniform() + getNoiseUniform() + getNoiseUniform() + getNoiseUniform();
    return (total - 2) * 1.7320508075688772;
  };
  let effects = Array.from({length: ENTITY_COUNT}, () => fixture.effectDeviation * getDeviate());
  const opposingEffect = scene.opposingEffect;
  if (opposingEffect !== undefined) {
    // Entity effects of alternating sign, an eigenvector of the ring.
    const runLength = scene.effectRunLength ?? 1;
    effects = effects.map(
      (effect, entity) =>
        effect + (Math.floor(entity / runLength) % 2 === 0 ? opposingEffect : -opposingEffect)
    );
  }
  const predictor = Float32Array.from({length: ROW_COUNT}, () => 4 * getPredictorUniform() - 2);
  const response = new Float32Array(ROW_COUNT);
  for (let period = 0; period < PERIOD_COUNT; period++) {
    const signal = Array.from(
      {length: ENTITY_COUNT},
      (_, entity) =>
        0.6 * scene.center +
        scene.slope * predictor[period * ENTITY_COUNT + entity] +
        effects[entity] +
        scene.deviation * getDeviate()
    );
    const total = [...signal];
    let term = [...signal];
    for (let power = 1; power < fixture.seriesTermCount; power++) {
      term = getRingLag(term).map(value => fixture.parameter * value);
      for (let entity = 0; entity < ENTITY_COUNT; entity++) total[entity] += term[entity];
    }
    response.set(total, period * ENTITY_COUNT);
  }
  expect(Array.from(response).reduce((sum, value) => sum + value, 0)).toBe(scene.responseSum);
  return {predictor, response};
}

function createRingWeights() {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let entity = 0; entity < ENTITY_COUNT; entity++) {
    neighbors.push(
      ...[(entity + ENTITY_COUNT - 1) % ENTITY_COUNT, (entity + 1) % ENTITY_COUNT].sort(
        (first, second) => first - second
      )
    );
    weights.push(0.5, 0.5);
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors, weights};
}

async function runPanel(
  effects: 'unit' | 'random',
  predictor: Float32Array,
  response: Float32Array
) {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const ring = createRingWeights();
    const coefficients = output('float32', 4);
    const covariance = output('float32', 9);
    const summary = output('float32', 6);
    const status = output('uint32', 1);
    const residuals = output('float32', ROW_COUNT);
    const fitted = output('float32', ROW_COUNT);
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
        effects,
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
    const covarianceValues = await readFloat32(covariance.buffer, 9);
    const residualValues = await readFloat32(residuals.buffer, ROW_COUNT);
    return {
      status: (await readUint32(status.buffer, 1))[0],
      // Intercept, slope, rho, phi.
      coefficients: Array.from(await readFloat32(coefficients.buffer, 4)),
      // Of the intercept, the slope and rho.
      standardErrors: [0, 4, 8].map(slot => Math.sqrt(covarianceValues[slot])),
      sigmaSquared: (await readFloat32(summary.buffer, 6))[0],
      logLikelihood: (await readFloat32(summary.buffer, 6))[1],
      residuals: Array.from(residualValues),
      fitted: Array.from(await readFloat32(fitted.buffer, ROW_COUNT)),
      /** Boundary bits: 1 rho near an end, 2 phi at an end, 4 phi below its lowest value. */
      boundary: (await readFloat32(summary.buffer, 6))[5],
      residualMeanSquare:
        Array.from(residualValues).reduce((sum, value) => sum + value * value, 0) / ROW_COUNT
    };
  } finally {
    context.destroy();
  }
}

const getRelativeError = (value: number, reference: number) => Math.abs(value / reference - 1);

/**
 * Bounds by fit: about three times the largest error measured on the test GPU (the measured
 * values are in the review report, regression.md, "Closing pass, review fixes", second round).
 * On the `slope-200` scenes what remains is the float32 rounding of the fitted values (200
 * times a predictor of 2 against residuals of 0.004: 0.6% of a residual per row, over 256
 * rows). `rho` is found by a golden section of 32 steps on [-0.999, 0.999], which resolves
 * 4e-7, a third of its standard error there (1.3e-6). On the ordinary scene the slope and
 * `sigma2` carry the float32 rounding of a serial fit of 256 rows.
 */
function getBounds(scene: Scene) {
  if (scene.slope === 200) {
    return {
      parameter: 0.15,
      slope: 4e-6,
      sigmaSquared: 5e-3,
      logLikelihood: 0.7,
      standardError: 3e-3
    };
  }
  if (scene.slope === 20) {
    return {
      parameter: 0.012,
      slope: 4e-6,
      sigmaSquared: 2e-4,
      logLikelihood: 0.03,
      standardError: 1e-4
    };
  }
  return {
    parameter: 0.04,
    slope: scene.name === 'ordinary' ? 1e-4 : 4e-6,
    sigmaSquared: 2.5e-4,
    logLikelihood: 3e-3,
    standardError: 5e-4
  };
}

for (const scene of fixture.scenes)
  for (const effects of ['unit', 'random'] as const)
    it(`panel lag maximum likelihood with ${effects} effects on ${scene.name} matches spreg in float64 (frc-regression-09)`, async () => {
      const {predictor, response} = createScene(scene);
      const result = await runPanel(effects, predictor, response);
      const reference = scene[effects];
      const parameterStandardError = reference.standardErrors[reference.standardErrors.length - 1];
      const errors = {
        status: result.status,
        parameterErrorOverStandardError:
          Math.abs(result.coefficients[2] - reference.parameter) / parameterStandardError,
        slopeError: getRelativeError(result.coefficients[1], reference.slope),
        sigmaSquaredError: getRelativeError(result.sigmaSquared, reference.sigmaSquared),
        logLikelihoodError: result.logLikelihood - reference.logLikelihood,
        // Unit effects have no intercept: slope and rho only.
        standardErrorError: Math.max(
          ...reference.standardErrors.map((standardError, index) =>
            getRelativeError(
              result.standardErrors[index + 3 - reference.standardErrors.length],
              standardError
            )
          )
        ),
        residualMeanSquareError: getRelativeError(result.residualMeanSquare, reference.sigmaSquared)
      };
      const message = `GPU ${JSON.stringify({...errors, sigmaSquared: result.sigmaSquared})}`;
      const bounds = getBounds(scene);
      expect(result.status, message).toBe(0);
      expect(errors.parameterErrorOverStandardError, message).toBeLessThan(bounds.parameter);
      expect(errors.slopeError, message).toBeLessThan(bounds.slope);
      expect(errors.sigmaSquaredError, message).toBeLessThan(bounds.sigmaSquared);
      expect(Math.abs(errors.logLikelihoodError), message).toBeLessThan(bounds.logLikelihood);
      expect(errors.standardErrorError, message).toBeLessThan(bounds.standardError);
      // The published variance is that of the published residuals.
      expect(errors.residualMeanSquareError, message).toBeLessThan(bounds.sigmaSquared);
      if (effects === 'random') {
        expect(Math.abs(result.coefficients[3] - scene.random.phi), message).toBeLessThan(
          2e-3 * scene.random.phi + 1e-5
        );
      }
    });

it('panel lag maximum likelihood with unit effects forms the quasi-demeaned difference before the fitted part when entity effects are large and of opposing sign (frc2-regression-05)', async () => {
  // Third round of review fixes: the response of an entity is within 5 of +3000 or -3000 (a
  // float32 step of 2^-12), the pooled mean is near zero, the residual deviation is 0.004.
  // The fit kernel formed value - (1 - phi) mean - fitted in one chain, which the shader
  // compiler of the test GPU regrouped as value - (mean + fitted): before the fix all 256
  // published residuals were multiples of 2^-12, rho was 0.026 standard errors from spreg and
  // the within-entity part of the residuals had a root mean square error of 7.7e-5.
  // Reference: spreg 1.9.1 Panel_FE_Lag (the generator's `opposingEffectScene`).
  const scene = fixture.opposingEffectScene;
  expect(scene.largestEntityMean).toBeGreaterThan(2990);
  expect(Math.abs(scene.pooledMean)).toBeLessThan(1);
  const {predictor, response} = createScene(scene);
  const result = await runPanel('unit', predictor, response);
  const reference = scene.unit;
  // Float64 residuals at the reference coefficients, within entities.
  const getWithin = (values: ArrayLike<number>) => {
    const means = new Array<number>(ENTITY_COUNT).fill(0);
    for (let row = 0; row < ROW_COUNT; row++)
      means[row % ENTITY_COUNT] += values[row] / PERIOD_COUNT;
    return Array.from({length: ROW_COUNT}, (_, row) => values[row] - means[row % ENTITY_COUNT]);
  };
  const lagged = new Array<number>(ROW_COUNT);
  for (let period = 0; period < PERIOD_COUNT; period++) {
    const start = period * ENTITY_COUNT;
    getRingLag(Array.from(response.subarray(start, start + ENTITY_COUNT))).forEach(
      (value, entity) => {
        lagged[start + entity] = value;
      }
    );
  }
  const withinResponse = getWithin(response);
  const withinLag = getWithin(lagged);
  const withinPredictor = getWithin(predictor);
  const withinResiduals = getWithin(result.residuals);
  let squaredError = 0;
  let largestError = 0;
  for (let row = 0; row < ROW_COUNT; row++) {
    const expected =
      withinResponse[row] -
      reference.parameter * withinLag[row] -
      reference.slope * withinPredictor[row];
    const error = Math.abs(withinResiduals[row] - expected);
    squaredError += error * error;
    largestError = Math.max(largestError, error);
  }
  const errors = {
    status: result.status,
    parameterErrorOverStandardError:
      Math.abs(result.coefficients[2] - reference.parameter) / reference.standardErrors[1],
    slopeError: getRelativeError(result.coefficients[1], reference.slope),
    sigmaSquaredError: getRelativeError(result.sigmaSquared, reference.sigmaSquared),
    residualMeanSquareError: getRelativeError(result.residualMeanSquare, reference.sigmaSquared),
    standardErrorError: Math.max(
      getRelativeError(result.standardErrors[1], reference.standardErrors[0]),
      getRelativeError(result.standardErrors[2], reference.standardErrors[1])
    ),
    // Residuals that are a multiple of the float32 step of the response (2^-12 near 3000).
    residualsOnResponseStep: result.residuals.filter(value => Number.isInteger(value * 4096))
      .length,
    withinResidualRootMeanSquareError: Math.sqrt(squaredError / ROW_COUNT),
    largestWithinResidualError: largestError
  };
  const message = `GPU ${JSON.stringify(errors)}`;
  expect(result.status, message).toBe(0);
  // Before: 256 of 256.
  expect(errors.residualsOnResponseStep, message).toBeLessThan(16);
  // Before: 0.026 standard errors and 1.0e-6; measured after: 0.0022 and 5.3e-8.
  expect(errors.parameterErrorOverStandardError, message).toBeLessThan(0.008);
  expect(errors.slopeError, message).toBeLessThan(2.5e-7);
  // Before: 7.7e-5 and 1.8e-4. What remains (3.3e-5 and 6.9e-5) is the float32 lag of the
  // centered response, rounded to a step of 2^-12.
  expect(errors.withinResidualRootMeanSquareError, message).toBeLessThan(5.5e-5);
  expect(errors.largestWithinResidualError, message).toBeLessThan(1.3e-4);
  // The entity means are sequential float32 sums, off by up to 6e-4: a constant per entity in
  // the residuals, which leaves sigma2 0.58% high (measured; 1.1% before).
  expect(errors.sigmaSquaredError, message).toBeLessThan(9e-3);
  expect(errors.residualMeanSquareError, message).toBeLessThan(9e-3);
  expect(errors.standardErrorError, message).toBeLessThan(5e-3);
});

/**
 * Fourth round (2026-11-11): frc-regression-13 and -15 of the re-review of round three in
 * `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-regression.md`.
 */
/**
 * Bounds of the fit with a `phi` of 2.5e-7: three times what the test GPU measures (`phi`
 * 1.9e-3 relative, the intercept 0.20 standard errors, the slope 3.1e-7, `rho` 0.009 standard
 * errors, `sigma2` 1.1%, the log-likelihood 1.5, standard errors 5.7e-3). What remains is the
 * float32 entity means of the pack node (off by up to 6e-4 for an effect of 3000), as for unit
 * effects on the same data (`sigma2` 0.58%). Before the fix `sigma2` was 64 times spreg's.
 */
const BOUNDS_SMALL_PHI = {
  phi: 6e-3,
  intercept: 0.6,
  slope: 1e-6,
  parameter: 0.03,
  sigmaSquared: 0.035,
  logLikelihood: 4.5,
  standardError: 0.017
};
/** Measured 2.1e-5 (the float32 rounding of two outputs near 16 and 400). */
const FITTED_IDENTITY_BOUND = 1e-4;
it('panel lag maximum likelihood with random effects follows a phi of 2.5e-7 and matches spreg Panel_RE_Lag (frc-regression-13)', async () => {
  // Entity effects of +3000 and -3000 with residuals of 0.004. The kernel held phi at 1e-5 and
  // returned the fit of that constrained problem with status 0: on the reviewer's scenes sigma2
  // 28 to 119 times spreg's and rho 2.3 to 69 standard errors off.
  const scene = fixture.opposingEffectScene;
  const reference = scene.random;
  expect(reference.phi).toBeLessThan(1e-6);
  const {predictor, response} = createScene(scene);
  const result = await runPanel('random', predictor, response);
  const errors = {
    status: result.status,
    boundary: result.boundary,
    phi: result.coefficients[3],
    phiError: getRelativeError(result.coefficients[3], reference.phi),
    interceptErrorOverStandardError:
      Math.abs(result.coefficients[0] - reference.intercept) / reference.standardErrors[0],
    slopeError: getRelativeError(result.coefficients[1], reference.slope),
    parameterErrorOverStandardError:
      Math.abs(result.coefficients[2] - reference.parameter) / reference.standardErrors[2],
    sigmaSquaredError: getRelativeError(result.sigmaSquared, reference.sigmaSquared),
    residualMeanSquareError: getRelativeError(result.residualMeanSquare, reference.sigmaSquared),
    logLikelihoodError: result.logLikelihood - reference.logLikelihood,
    standardErrorError: Math.max(
      ...reference.standardErrors.map((standardError, index) =>
        getRelativeError(result.standardErrors[index], standardError)
      )
    )
  };
  const message = `GPU ${JSON.stringify(errors)}`;
  expect(result.status, message).toBe(0);
  // phi is at neither end.
  expect(result.boundary, message).toBe(0);
  expect(errors.phiError, message).toBeLessThan(BOUNDS_SMALL_PHI.phi);
  expect(errors.interceptErrorOverStandardError, message).toBeLessThan(BOUNDS_SMALL_PHI.intercept);
  expect(errors.slopeError, message).toBeLessThan(BOUNDS_SMALL_PHI.slope);
  expect(errors.parameterErrorOverStandardError, message).toBeLessThan(BOUNDS_SMALL_PHI.parameter);
  expect(errors.sigmaSquaredError, message).toBeLessThan(BOUNDS_SMALL_PHI.sigmaSquared);
  expect(errors.residualMeanSquareError, message).toBeLessThan(BOUNDS_SMALL_PHI.sigmaSquared);
  expect(Math.abs(errors.logLikelihoodError), message).toBeLessThan(BOUNDS_SMALL_PHI.logLikelihood);
  expect(errors.standardErrorError, message).toBeLessThan(BOUNDS_SMALL_PHI.standardError);
});

it('panel lag maximum likelihood with random effects reports a phi below its lowest value as status 3 with its own boundary bit (frc-regression-13)', async () => {
  // Entity effects of +16,000,000 and -16,000,000 (a float32 step of 1) and residuals of 0.5:
  // spreg's phi is 9.1e-9, below the 1e-8 the kernel follows. The estimate is held at 1e-8 and
  // the fit is that of the constrained problem: not converged, bits 2 (phi at an end) and 4
  // (the lower end), where a phi of 1 sets bit 2 alone.
  const scene = fixture.phiBelowLimitScene;
  expect(scene.random.phi).toBeLessThan(1e-8);
  const {predictor, response} = createScene(scene);
  const result = await runPanel('random', predictor, response);
  const message = `GPU ${JSON.stringify({status: result.status, boundary: result.boundary, coefficients: result.coefficients})}`;
  expect(result.status, message).toBe(3);
  expect(result.boundary, message).toBe(6);
  expect(result.coefficients[3], message).toBe(Math.fround(1e-8));
  // No covariance for a fit that is not the maximum.
  expect(result.standardErrors.every(Number.isNaN), message).toBe(true);
});

it('panel lag maximum likelihood with random effects publishes a fitted value that is the quasi-demeaned response at the published phi less the residual (frc-regression-15)', async () => {
  // fitted + residual = y - (1 - phi) mean_i(y) at the kernel's own phi. Against spreg the
  // fitted values of this scene (response mean 10,000, residuals of 0.004) are up to 1.2
  // residual deviations off while the residuals are within 0.05: phi agrees with spreg to
  // 4.5e-7, inside the tolerance of the alternation, and the quasi-demeaned response itself
  // moves by that times the entity mean. The identity below is what the two outputs owe each
  // other.
  const scene = fixture.scenes.find(
    candidate => candidate.name === 'slope-200-center-10000-stream-21'
  )!;
  const {predictor, response} = createScene(scene);
  const result = await runPanel('random', predictor, response);
  expect(result.status).toBe(0);
  const phi = result.coefficients[3];
  const means = new Array<number>(ENTITY_COUNT).fill(0);
  for (let row = 0; row < ROW_COUNT; row++)
    means[row % ENTITY_COUNT] += response[row] / PERIOD_COUNT;
  let largest = 0;
  for (let row = 0; row < ROW_COUNT; row++) {
    const quasiDemeaned = response[row] - (1 - phi) * means[row % ENTITY_COUNT];
    largest = Math.max(
      largest,
      Math.abs(result.fitted[row] + result.residuals[row] - quasiDemeaned)
    );
  }
  expect(largest, `phi ${phi}`).toBeLessThan(FITTED_IDENTITY_BOUND);
});
