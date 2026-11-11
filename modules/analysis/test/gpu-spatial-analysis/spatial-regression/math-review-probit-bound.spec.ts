// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import fixture from '../../oracles/fixtures/statistics-probit-tails.json';

/**
 * Probit fits whose linear predictor leaves the range the kernel used to bound (|eta| <= 12),
 * and separated data (regression-01 of the 2026-11-10 review and its closing pass).
 *
 * History. Before the review a fit whose maximum put a row beyond 12 stopped on the bound and
 * was published as converged (a slope of 1.32 against 2.07). The review made that status 4. The
 * closing pass (2026-11-11) evaluates the likelihood, score and information through the tail
 * probabilities, bounds only rows on the wrong side of the fit, and so returns the maximum when
 * the rows beyond 12 are correctly classified; separated data still report status 4.
 *
 * Reference numbers: `test/oracles/fixtures/statistics-probit-tails.json`, written by
 * `test/oracles/statistics-probit-tails.py` (its docstring has the full recipe). In short,
 * float64 on the float32 predictors, CPython 3.14, NumPy 2.5.4, SciPy 1.16.3:
 * - coefficients and log-likelihood: a Newton solve of the tail-exact likelihood
 *   `sum log_ndtr((2 y - 1) eta)` (`scipy.special.log_ndtr`), which statsmodels 0.15.0
 *   `Probit(...).fit(method='newton')` matches to 1e-9 on every scene but the gross outlier (the
 *   generator asserts it);
 * - standard errors: the expected information `X' diag(phi^2 / (Phi (1 - Phi))) X` at that
 *   maximum (statsmodels reports the observed-information ones, which differ);
 * - the gross outlier: the tail-exact Newton solve only. statsmodels clips probabilities at
 *   2.2e-16 and returns the fit without the outlier there, which is not the maximum.
 *
 * The scenes use a 32-bit linear congruential stream and IEEE double arithmetic only, so the
 * Python generator rebuilds the same inputs (checked through `predictorSum`).
 */
function createProbitSample(
  rows: number,
  intercept: number,
  slope: number,
  seed: number,
  noisy = true
) {
  let state = seed >>> 0;
  const getUniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
  const getDeviate = () => {
    let sum = 0;
    for (let term = 0; term < 12; term++) sum += getUniform();
    return sum - 6;
  };
  const predictor: number[] = [];
  const response: number[] = [];
  for (let row = 0; row < rows; row++) {
    const value = getDeviate();
    predictor.push(value);
    response.push(intercept + slope * value + (noisy ? getDeviate() : 0) > 0 ? 1 : 0);
  }
  return {predictor, response};
}

/** Sum of the predictors as the GPU receives them (float32), in float64. */
function getPredictorSum(predictor: number[]): number {
  return predictor.reduce((sum, value) => sum + Math.fround(value), 0);
}

type ProbitReference = {
  /** Intercept and slope of the tail-exact maximum. */
  coefficients: number[];
  /** Expected-information standard errors. */
  standardErrors: number[];
  logLikelihood: number;
  predictorSum: number;
  /** 0: the linear program of the generator found no separating direction. */
  separationObjective: number;
  statsmodels: {
    /** statsmodels `GLM(Binomial(Probit))` standard errors, where it is at the maximum. */
    expectedInformationStandardErrors?: number[];
  };
};

/**
 * Runs the probit fit. `predictor` is one column, or row-major values of `columns` predictors;
 * `options` are the `maximumIterations` and `tolerance` props (the defaults when omitted), and
 * `ringDiagnostics` asks for the spatial diagnostics on a ring of weights 0.5 over the row
 * order (then `fitted` and `tests` are read back too).
 */
async function runProbit(
  predictor: ArrayLike<number>,
  response: ArrayLike<number>,
  columns = 1,
  options: {maximumIterations?: number; tolerance?: number; ringDiagnostics?: boolean} = {}
) {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const rowCount = response.length;
    const coefficients = output('float32', columns + 1);
    const errors = output('float32', columns + 1);
    const summary = output('float32', 4);
    const status = output('uint32', 1);
    const fitted = output('float32', rowCount);
    const effects = output('float32', columns);
    const tests = output('float32', 6);
    const diagnosticStatus = output('uint32', 1);
    const {ringDiagnostics, ...fitOptions} = options;
    graph.add(
      new GPUProbitRegression({
        predictors: input('float32', predictor),
        predictorCount: columns,
        response: input('float32', response),
        ...fitOptions,
        ...(ringDiagnostics
          ? {
              spatialDiagnostics: {
                weights: {
                  offsets: input(
                    'uint32',
                    Array.from({length: rowCount + 1}, (_, row) => 2 * row)
                  ),
                  neighbors: input(
                    'uint32',
                    Array.from({length: rowCount}, (_, row) =>
                      [(row + rowCount - 1) % rowCount, (row + 1) % rowCount].sort(
                        (first, second) => first - second
                      )
                    ).flat()
                  ),
                  weights: input('float32', new Array<number>(2 * rowCount).fill(0.5))
                },
                output: {tests: tests.view, status: diagnosticStatus.view}
              }
            }
          : {}),
        output: {
          coefficients: coefficients.view,
          standardErrors: errors.view,
          fitted: fitted.view,
          marginalEffects: effects.view,
          summary: summary.view,
          status: status.view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from(await readFloat32(coefficients.buffer, columns + 1)),
      errors: Array.from(await readFloat32(errors.buffer, columns + 1)),
      summary: Array.from(await readFloat32(summary.buffer, 4)),
      marginalEffects: Array.from(await readFloat32(effects.buffer, columns)),
      fitted: ringDiagnostics ? Array.from(await readFloat32(fitted.buffer, rowCount)) : [],
      tests: ringDiagnostics ? Array.from(await readFloat32(tests.buffer, 6)) : [],
      diagnosticStatus: ringDiagnostics ? (await readUint32(diagnosticStatus.buffer, 1))[0] : -1
    };
  } finally {
    context.destroy();
  }
}

/** 2000 rows: `y = 1` when `0.2 + 2 x` plus a unit deviate of the same stream is positive. */
const createBaseSample = () => createProbitSample(2000, 0.2, 2.0, 777);

const SCENES = fixture.scenes;

async function expectMaximum(predictor: number[], response: number[], reference: ProbitReference) {
  expect(getPredictorSum(predictor)).toBe(reference.predictorSum);
  const result = await runProbit(predictor, response);
  const message = `GPU ${JSON.stringify(result)}`;
  expect(result.status, message).toBe(0);
  for (let column = 0; column < 2; column++) {
    // A hundredth of a standard error.
    expect(
      Math.abs(result.coefficients[column] - reference.coefficients[column]),
      message
    ).toBeLessThan(0.01 * reference.standardErrors[column]);
    expect(
      Math.abs(result.errors[column] / reference.standardErrors[column] - 1),
      message
    ).toBeLessThan(2e-3);
    // The library's own expected-information standard errors (statsmodels GLM, IRLS).
    const libraryErrors = reference.statsmodels.expectedInformationStandardErrors;
    if (libraryErrors) {
      expect(Math.abs(result.errors[column] / libraryErrors[column] - 1), message).toBeLessThan(
        2e-3
      );
    }
  }
  expect(Math.abs(result.summary[1] / reference.logLikelihood - 1), message).toBeLessThan(2e-5);
  // Not separated: the generator's linear program found no direction that moves every row to
  // the right side.
  expect(reference.separationObjective).toBe(0);
}

async function expectSeparated(
  predictor: ArrayLike<number>,
  response: ArrayLike<number>,
  columns = 1,
  options: {maximumIterations?: number; tolerance?: number} = {}
) {
  const result = await runProbit(predictor, response, columns, options);
  expect(result.status, `GPU ${JSON.stringify(result)}`).toBe(4);
  expect(result.coefficients.every(Number.isNaN)).toBe(true);
  expect(result.errors.every(Number.isNaN)).toBe(true);
}

it('probit without a row beyond a linear predictor of 12 converges to the statsmodels maximum', async () => {
  const {predictor, response} = createBaseSample();
  await expectMaximum(predictor, response, SCENES.control);
});

it('probit with a correctly classified row far beyond a linear predictor of 12 returns the maximum (regression-01)', async () => {
  // One correctly classified, high-leverage row: at the maximum its linear predictor is 18.9.
  // It adds log(Phi(18.9)) = 0 to the likelihood, so the maximum is that of the other rows.
  const {predictor, response} = createBaseSample();
  expect(SCENES.correctOutlier.largestLinear).toBeGreaterThan(18);
  await expectMaximum([...predictor, 9], [...response, 1], SCENES.correctOutlier);
});

it('probit whose maximum lies just beyond a linear predictor of 12 returns that maximum', async () => {
  // 400 rows, slope 3.5: the maximum has max |eta| = 12.06 on a correctly classified row. The
  // review's kernel reported status 4 here; the one before it a slope 0.5% low as converged.
  const {predictor, response} = createProbitSample(400, 0.2, 3.5, 12345);
  expect(SCENES.justBeyondBound.largestLinear).toBeGreaterThan(12);
  expect(SCENES.justBeyondBound.largestLinear).toBeLessThan(12.2);
  await expectMaximum(predictor, response, SCENES.justBeyondBound);
});

it('probit with a misclassified row beyond the float32 rounding of its probability is pulled by it, as the likelihood requires', async () => {
  // A response of 0 at x = 4: at the maximum its linear predictor is 7.69, where 1 - Phi is
  // 7e-15. A probability floored at 1e-7 gives that row a flat likelihood and no score; the true
  // likelihood costs it -eta^2 / 2 and lowers the slope from 2.07 to 1.86.
  const {predictor, response} = createBaseSample();
  expect(SCENES.misclassifiedOutlier.largestMisclassifiedLinear).toBeGreaterThan(7.5);
  await expectMaximum([...predictor, 4], [...response, 0], SCENES.misclassifiedOutlier);
});

it('probit whose maximum puts a misclassified row beyond a linear predictor of 12 is status 3 with the pinned iterate, never status 4', async () => {
  // A response of 0 at x = 9: the tail-exact float64 maximum is (0.1952, 1.4466) with a linear
  // predictor of 13.2 on that row, where 1 - Phi (4e-40) is below the float32 range. The kernel
  // bounds rows on the wrong side at 12, so the fit is pinned short of its maximum: not
  // converged. These data are not separated (the linear program of the generator is 0), so the
  // status is 3 with the iterate, not 4 (it was 4 with NaN until the third round of review
  // fixes, frc2-regression-03: status 4 is kept for a certificate of separation).
  const {predictor, response} = createBaseSample();
  const reference = SCENES.grossMisclassifiedOutlier;
  expect(getPredictorSum([...predictor, 9])).toBe(reference.predictorSum);
  expect(reference.largestMisclassifiedLinear).toBeGreaterThan(13);
  expect(reference.separationObjective).toBe(0);
  // statsmodels returns the fit without the outlier here, not this maximum.
  expect(reference.statsmodels.coefficients[1] - reference.coefficients[1]).toBeGreaterThan(0.5);
  const result = await runProbit([...predictor, 9], [...response, 0]);
  const message = `GPU ${JSON.stringify(result)}`;
  expect(result.status, message).toBe(3);
  expect(result.errors.every(Number.isNaN), message).toBe(true);
  // The iterate is pinned where the outlier sits on the bound.
  const outlierLinear = result.coefficients[0] + 9 * result.coefficients[1];
  expect(outlierLinear, message).toBeGreaterThan(11.9);
  expect(outlierLinear, message).toBeLessThan(12.001);
  // Below the maximum, which has the outlier at 13.2.
  expect(result.coefficients[1], message).toBeLessThan(reference.coefficients[1]);
  expect(result.summary[1], message).toBeLessThan(reference.logLikelihood);
});

it('probit on completely separated data is status 4', async () => {
  // y = 1 exactly when 0.2 + 3.5 x > 0: no finite maximum (statsmodels runs to a slope of 412
  // and reports no convergence).
  const {predictor, response} = createProbitSample(400, 0.2, 3.5, 4242, false);
  expect(getPredictorSum(predictor)).toBe(SCENES.completeSeparation.predictorSum);
  expect(SCENES.completeSeparation.statsmodels.converged).toBe(false);
  await expectSeparated(predictor, response);
});

it('probit on quasi-completely separated data is status 4', async () => {
  // The same data plus six rows on the separating value with both outcomes: the likelihood has
  // a finite supremum (6 ln 0.5) that no finite coefficients reach.
  const {predictor, response} = createProbitSample(400, 0.2, 3.5, 4242, false);
  const boundary = -0.2 / 3.5;
  const separatedPredictor = [...predictor, ...new Array<number>(6).fill(boundary)];
  expect(getPredictorSum(separatedPredictor)).toBe(SCENES.quasiCompleteSeparation.predictorSum);
  expect(SCENES.quasiCompleteSeparation.statsmodels.converged).toBe(false);
  await expectSeparated(separatedPredictor, [...response, 1, 0, 1, 0, 1, 0]);
});

it('probit on data separated by two predictors is status 4', async () => {
  const first = createProbitSample(500, 0, 1, 99, false).predictor;
  const second = createProbitSample(500, 0, 1, 4711, false).predictor;
  const predictors = first.flatMap((value, row) => [value, second[row]]);
  const response = first.map((value, row) => (value + 0.5 * second[row] > 0.1 ? 1 : 0));
  await expectSeparated(predictors, response, 2);
});

/**
 * Review fixes of the closing pass (2026-11-11): frc-regression-03 and -05 of
 * `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-regression.md`.
 *
 * Separation was decided by a threshold: a fit that ended without converging was "separated"
 * when some |linear predictor| had passed 8. That is wrong both ways. Quasi-separation by an
 * indicator of a few rows loses the indicator direction of the information matrix near 6.5, or
 * runs out of iterations near 7.5 (status 2, or status 3 with the runaway coefficient written);
 * and a legitimate fit stopped early with a correctly classified row beyond 8 was status 4 with
 * NaN coefficients. The kernel now asks what separation is: whether a Fisher direction of the
 * fit moves some row to the right side and no row towards the wrong side.
 *
 * Oracle for "separated": `separationObjective` of the fixture, the optimum of a linear program
 * over directions (SciPy `linprog`), positive exactly when a separating direction exists; and
 * statsmodels `Probit`, which does not converge on these scenes.
 */
function createIndicatorScene(name: keyof typeof SCENES) {
  const {predictor, response} = createProbitSample(600, 0.2, 1.0, 31337);
  const getIndicator = (isMember: (row: number) => boolean) =>
    predictor.map((_, row) => (isMember(row) ? 1 : 0));
  const three = getIndicator(row => row === 17 || row === 301 || row === 555);
  const getNoise = (seed: number) => createProbitSample(600, 0, 1, seed).predictor;
  let indicator = three;
  let columns: number[][];
  if (name === 'indicatorOfOneRow') {
    indicator = getIndicator(row => row === 17);
    columns = [predictor, indicator];
  } else if (name === 'indicatorAmongElevenPredictors') {
    columns = [
      predictor,
      ...Array.from({length: 9}, (_, column) => getNoise(1000 + column)),
      three
    ];
  } else if (name === 'indicatorOfEveryTwentiethRow') {
    indicator = getIndicator(row => row % 20 === 0);
    columns = [predictor, indicator];
  } else if (name === 'indicatorOfEverySecondRow') {
    indicator = getIndicator(row => row % 2 === 0);
    columns = [predictor, indicator];
  } else if (name === 'indicatorAsMiddleOfThree') {
    columns = [predictor, three, getNoise(2024)];
  } else {
    columns = [predictor, three];
  }
  const reference = SCENES[name] as {predictorSums?: number[]; separationObjective: number};
  expect(columns.map(getPredictorSum)).toEqual(reference.predictorSums);
  return {
    // Row-major predictors.
    predictors: predictor.flatMap((_, row) => columns.map(column => column[row])),
    response: response.map((value, row) => (indicator[row] === 1 ? 1 : value)),
    columnCount: columns.length,
    reference
  };
}

for (const name of [
  'indicatorOfThreeRows',
  'indicatorOfOneRow',
  'indicatorAmongElevenPredictors',
  'indicatorOfEveryTwentiethRow',
  'indicatorOfEverySecondRow',
  'indicatorAsMiddleOfThree'
] as const)
  it(`probit on data quasi-separated by an indicator (${name}) is status 4 (frc-regression-03)`, async () => {
    // Before the fix the first three were status 2, status 3 with coefficients (0.169, 1.028,
    // 7.22) written, and status 3 with an indicator coefficient of 7.57.
    const scene = createIndicatorScene(name);
    expect(scene.reference.separationObjective).toBeGreaterThan(0.5);
    expect(SCENES[name].statsmodels.converged).toBe(false);
    await expectSeparated(scene.predictors, scene.response, scene.columnCount);
  });

it('probit on data quasi-separated by an indicator is status 4 with a larger iteration budget too', async () => {
  const scene = createIndicatorScene('indicatorOfThreeRows');
  await expectSeparated(scene.predictors, scene.response, scene.columnCount, {
    maximumIterations: 128
  });
});

it('probit on a steep fit that is not separated returns the statsmodels maximum', async () => {
  // Latent index 0.2 + 8 x: the largest linear predictor at the maximum is 29.5.
  const {predictor, response} = createProbitSample(2000, 0.2, 8.0, 5);
  expect(SCENES.steepSlope.largestLinear).toBeGreaterThan(29);
  await expectMaximum(predictor, response, SCENES.steepSlope);
});

/**
 * A fit that runs out of iterations is status 3 and publishes its iterate
 * (frc-regression-05). The iterate has no library value (statsmodels starts its iterations
 * elsewhere), so it is bounded by what any ascent from the zero start must satisfy: a
 * log-likelihood above that of the zero start (`n ln 0.5`) and not above the maximum of the
 * fixture, and coefficients nearer to the maximum than the start is.
 */
async function expectUnfinished(
  predictor: number[],
  response: number[],
  reference: ProbitReference,
  maximumIterations: number
) {
  expect(getPredictorSum(predictor)).toBe(reference.predictorSum);
  expect(reference.separationObjective).toBe(0);
  const result = await runProbit(predictor, response, 1, {maximumIterations});
  const message = `GPU ${JSON.stringify(result)}`;
  expect(result.status, message).toBe(3);
  expect(result.coefficients.every(Number.isFinite), message).toBe(true);
  expect(result.errors.every(Number.isNaN), message).toBe(true);
  expect(result.summary[3], message).toBe(maximumIterations);
  expect(result.summary[1], message).toBeGreaterThan(response.length * Math.log(0.5));
  expect(result.summary[1], message).toBeLessThan(reference.logLikelihood);
  expect(Math.abs(result.coefficients[1] - reference.coefficients[1]), message).toBeLessThan(
    Math.abs(reference.coefficients[1])
  );
}

it('probit that runs out of iterations with a correctly classified row beyond a linear predictor of 8 is status 3 with its iterate (frc-regression-05)', async () => {
  // The correct-outlier scene converges in 7 steps; after 3 the outlier is beyond 16. Before the
  // fix this was status 4 with NaN coefficients.
  const {predictor, response} = createBaseSample();
  await expectUnfinished([...predictor, 9], [...response, 1], SCENES.correctOutlier, 3);
});

it('probit on a steep fit that runs out of iterations is status 3 with its iterate (frc-regression-05)', async () => {
  const {predictor, response} = createProbitSample(2000, 0.2, 8.0, 5);
  await expectUnfinished(predictor, response, SCENES.steepSlope, 4);
});

it('probit without a row beyond a linear predictor of 8 that runs out of iterations is status 3 with its iterate', async () => {
  const {predictor, response} = createBaseSample();
  await expectUnfinished(predictor, response, SCENES.control, 3);
});

it('probit on completely separated data is status 4 with a budget of eight iterations', async () => {
  // The first directions from the zero start still misplace rows next to the boundary; from the
  // fourth on every row of completely separated data moves to the right side (measured: status
  // 4 from a budget of four iterations), long before a linear predictor is large.
  const {predictor, response} = createProbitSample(400, 0.2, 3.5, 4242, false);
  expect(SCENES.completeSeparation.separationObjective).toBeGreaterThan(0.5);
  await expectSeparated(predictor, response, 1, {maximumIterations: 8});
});

it('probit on data quasi-separated by an indicator of one row is status 4 with a tolerance of 1e-3 and of 1e-4, not a converged step', async () => {
  // With these tolerances the fit used to report status 0 at an indicator coefficient of 7.30
  // with a standard error of 5,511: near a linear predictor of 7 the Fisher direction of the
  // separated row is rounding, and a small rounding step passed for convergence. A separating
  // direction found earlier now forbids status 0.
  const scene = createIndicatorScene('indicatorOfOneRow');
  for (const tolerance of [1e-3, 1e-4]) {
    await expectSeparated(scene.predictors, scene.response, scene.columnCount, {tolerance});
  }
});

it('probit on data quasi-separated by an indicator is status 4 from a budget of ten iterations', async () => {
  // The other coefficients have to settle before the direction is the separating one: six
  // iterations for the indicators of one and of three rows, ten for every second row (measured);
  // a smaller budget is status 3, which the class documentation states.
  for (const name of [
    'indicatorOfThreeRows',
    'indicatorOfOneRow',
    'indicatorOfEverySecondRow'
  ] as const) {
    const scene = createIndicatorScene(name);
    await expectSeparated(scene.predictors, scene.response, scene.columnCount, {
      maximumIterations: 10
    });
  }
});

/**
 * Large samples: frc-regression-10 of the re-review. When the separated group is 2e-5 of the
 * rows or less, the rows outside it have a move that is a difference of two equal terms, each
 * as small as the group's share of the rows, and the rounding of the settled coefficients'
 * direction is no longer small against them: no direction was certified, and at the default
 * settings these five scenes were status 0 (one row of 100,000 with an indicator coefficient
 * of 7.59 and a standard error of 17,060; three of 200,000; twenty of 1,000,000), status 2
 * (five of 400,000) and status 3 (five of 1,000,000). The kernel now takes the rounding out of
 * the direction before it tests it.
 *
 * Oracle as above: the linear program of the generator (`separationObjective`) and statsmodels
 * `Probit`, which does not converge; for the twin whose group has both outcomes, the
 * statsmodels maximum.
 */
const LARGE_SAMPLE_ROWS = [17, 3001, 15555, 40000, 77777];
const LARGE_SAMPLE_TIMEOUT = 120000;

function createLargeIndicatorScene(
  name: keyof typeof SCENES,
  rowCount: number,
  rows: number[],
  outcomes: number[]
) {
  const {predictor, response} = createProbitSample(rowCount, 0.2, 1.0, 31337);
  const predictors = new Float32Array(2 * rowCount);
  let predictorSum = 0;
  for (let row = 0; row < rowCount; row++) {
    predictors[2 * row] = predictor[row];
    predictorSum += predictors[2 * row];
  }
  rows.forEach((row, index) => {
    predictors[2 * row + 1] = 1;
    response[row] = outcomes[index];
  });
  const reference = SCENES[name] as {predictorSums?: number[]; separationObjective: number};
  expect([predictorSum, rows.length]).toEqual(reference.predictorSums);
  return {predictors, response, reference};
}

for (const [name, rowCount, rows] of [
  ['indicatorOfOneRowIn100000', 100000, LARGE_SAMPLE_ROWS.slice(0, 1)],
  ['indicatorOfThreeRowsIn200000', 200000, LARGE_SAMPLE_ROWS.slice(0, 3)],
  ['indicatorOfFiveRowsIn400000', 400000, LARGE_SAMPLE_ROWS],
  ['indicatorOfFiveRowsIn1000000', 1000000, LARGE_SAMPLE_ROWS],
  [
    'indicatorOfTwentyRowsIn1000000',
    1000000,
    Array.from({length: 20}, (_, index) => 13 + 9973 * index)
  ]
] as const)
  it(
    `probit on data quasi-separated by a group of ${rows.length} of ${rowCount} rows (${name}) is status 4 (frc-regression-10)`,
    async () => {
      const scene = createLargeIndicatorScene(
        name,
        rowCount,
        [...rows],
        rows.map(() => 1)
      );
      expect(scene.reference.separationObjective).toBeGreaterThan(0.5);
      expect(SCENES[name].statsmodels.converged).toBe(false);
      await expectSeparated(scene.predictors, scene.response, 2);
    },
    LARGE_SAMPLE_TIMEOUT
  );

it(
  'probit on 100000 rows with an indicator group of three rows that has both outcomes returns the statsmodels maximum',
  async () => {
    // The twin of the scenes above that is not separated: no certificate, status 0.
    const scene = createLargeIndicatorScene(
      'indicatorWithBothOutcomesIn100000',
      100000,
      LARGE_SAMPLE_ROWS.slice(0, 3),
      [1, 1, 0]
    );
    const reference = SCENES.indicatorWithBothOutcomesIn100000;
    expect(reference.separationObjective).toBe(0);
    const result = await runProbit(scene.predictors, scene.response, 2);
    const message = `GPU ${JSON.stringify(result)}`;
    expect(result.status, message).toBe(0);
    for (let column = 0; column < 3; column++) {
      expect(
        Math.abs(result.coefficients[column] - reference.coefficients[column]),
        message
      ).toBeLessThan(0.01 * reference.standardErrors[column]);
      expect(
        Math.abs(
          result.errors[column] / reference.statsmodels.expectedInformationStandardErrors[column] -
            1
        ),
        message
      ).toBeLessThan(2e-3);
    }
  },
  LARGE_SAMPLE_TIMEOUT
);

/**
 * Third round of review fixes (2026-11-11): frc2-regression-03, -04 and -06 of
 * `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-regression-codex.md`.
 *
 * Status 4 now needs a certificate with no tolerance: a direction whose every row provably
 * moves towards the right side or stands still. The inputs of these scenes are written out in
 * the generator's docstring (no stream).
 */
function createNearlySeparatedSample() {
  const predictor = [
    ...new Array<number>(1000).fill(-1),
    ...new Array<number>(2000).fill(1),
    -1e-5,
    1e-5
  ];
  const response = [...new Array<number>(1000).fill(0), ...new Array<number>(2000).fill(1), 1, 0];
  return {predictor, response};
}

it('probit on data within 1.5e-5 of separation converges to the statsmodels maximum (frc2-regression-03)', async () => {
  // The rows at -1e-5 (response 1) and 1e-5 (response 0) exclude every separating direction.
  // The first Fisher direction moves them towards the wrong side by 1.5e-5 of their own terms;
  // the certificate accepted moves up to 1e-3 and this was status 4 at every iteration budget.
  const {predictor, response} = createNearlySeparatedSample();
  const reference = SCENES.nearlySeparated;
  expect(reference.largestMisclassifiedLinear).toBeLessThan(1e-4);
  await expectMaximum(predictor, response, reference);
});

it('probit on data within 1.5e-5 of separation is status 3 with its iterate while the iterations run out, never status 4 (frc2-regression-03)', async () => {
  const {predictor, response} = createNearlySeparatedSample();
  const reference = SCENES.nearlySeparated;
  for (const maximumIterations of [1, 2, 4, 8, 16]) {
    const result = await runProbit(predictor, response, 1, {maximumIterations});
    const message = `budget ${maximumIterations} GPU ${JSON.stringify(result)}`;
    expect(result.status, message).toBe(3);
    expect(result.coefficients.every(Number.isFinite), message).toBe(true);
    expect(result.summary[3], message).toBe(maximumIterations);
    // An ascent from the zero start that has not passed the maximum (to the 2e-5 of the
    // float32 log-likelihood that `expectMaximum` allows).
    expect(result.summary[1], message).toBeGreaterThan(response.length * Math.log(0.5));
    expect(result.summary[1], message).toBeLessThan(reference.logLikelihood * (1 - 2e-5));
    expect(result.coefficients[1], message).toBeLessThan(reference.coefficients[1]);
  }
});

it('probit fitted probabilities, marginal effects and spatial diagnostics are those of the centered fit for a predictor near 2^24 (frc2-regression-04)', async () => {
  // Five groups of 1,000 rows at x = 2^24 + 2 z, z = -2..2. The raw intercept is a - mean b, a
  // float32 near -5.86e6 with a step of 0.5: rebuilt from the published raw coefficients the
  // fitted probabilities were 0.185, 0.421, 0.691, 0.885, 0.971 and the marginal effect 0.0888.
  const reference = SCENES.largeOffsetPredictor;
  const predictor: number[] = [];
  const response: number[] = [];
  [136, 345, 618, 841, 955].forEach((positiveCount, group) => {
    for (let row = 0; row < 1000; row++) {
      predictor.push(reference.predictorOffset + 2 * (group - 2));
      response.push((row * 37) % 1000 < positiveCount ? 1 : 0);
    }
  });
  expect(getPredictorSum(predictor)).toBe(reference.predictorSum);
  expect(reference.separationObjective).toBe(0);
  const result = await runProbit(predictor, response, 1, {ringDiagnostics: true});
  const message = `GPU ${JSON.stringify({...result, fitted: [0, 1, 2, 3, 4].map(group => result.fitted[1000 * group])})}`;
  expect(result.status, message).toBe(0);
  // The slope, in standard errors; the raw intercept to its float32 resolution (two steps).
  expect(Math.abs(result.coefficients[1] - reference.coefficients[1]), message).toBeLessThan(
    0.01 * reference.standardErrors[1]
  );
  const rawIntercept =
    reference.coefficients[0] - reference.predictorOffset * reference.coefficients[1];
  expect(Math.abs(result.coefficients[0] - rawIntercept), message).toBeLessThan(1.5);
  expect(Math.abs(result.errors[1] / reference.standardErrors[1] - 1), message).toBeLessThan(2e-3);
  expect(Math.abs(result.summary[1] / reference.logLikelihood - 1), message).toBeLessThan(2e-5);
  // statsmodels Probit.predict: one value per group, every row of a group alike.
  reference.statsmodels.groupFitted.forEach((probability, group) => {
    for (const row of [0, 499, 999]) {
      expect(Math.abs(result.fitted[1000 * group + row] - probability), message).toBeLessThan(1e-6);
    }
  });
  // statsmodels get_margeff(at='overall').
  expect(
    Math.abs(result.marginalEffects[0] / reference.statsmodels.averageMarginalEffect - 1),
    message
  ).toBeLessThan(1e-5);
  // spreg Probit(spat_diag=True): Pinkse, Kelejian-Prucha and Pinkse-Slade statistics.
  expect(result.diagnosticStatus, message).toBe(0);
  for (const slot of [0, 2, 4]) {
    expect(
      Math.abs(result.tests[slot] / reference.spreg.diagnostics[slot] - 1),
      message
    ).toBeLessThan(1e-4);
  }
});

function createThreeValuedSample() {
  // 40,000 rows (0, 0), 60,000 rows (0, 1), then (-1, 0) and (1, 1).
  const predictor = new Float32Array(100002);
  const response = new Float32Array(100002);
  response.fill(1, 40000, 100000);
  predictor[100000] = -1;
  predictor[100001] = 1;
  response[100001] = 1;
  return {predictor, response};
}

for (const options of [
  {},
  {maximumIterations: 128},
  {maximumIterations: 8},
  {tolerance: 1e-3}
] as const)
  it(
    `probit on a three-valued predictor whose rows with both responses are at its middle value is status 4 (${JSON.stringify(options)}, frc2-regression-06)`,
    async () => {
      // Quasi-separated by the direction (0, 1): the rows at 0 stand still, the rows at -1 and 1
      // move to the right side. No end of the predictor is free of mixed rows, so the certificate
      // is the threshold at the data value next to where the Fisher direction changes sign.
      // Before the fix: status 3 at the defaults, status 2 with 128 iterations, and status 0
      // (slope 3.93, standard error 21.9) with a tolerance of 1e-3.
      const reference = SCENES.threeValuedMiddle;
      expect(reference.separationObjective).toBeGreaterThan(0.5);
      expect(reference.statsmodels.converged).toBe(false);
      const {predictor, response} = createThreeValuedSample();
      await expectSeparated(predictor, response, 1, options);
    },
    LARGE_SAMPLE_TIMEOUT
  );

it('probit on data quasi-separated by an indicator is status 4 at any iteration budget and tolerance (frc2-regression-06)', async () => {
  // A column that separates at its lowest value is certified by comparing raw values, before
  // any iteration: an indicator of one row of 600 was status 3 below six iterations and status
  // 0 with a tolerance of 1e-2 (coefficient 3.94, standard error 5.39).
  const scene = createIndicatorScene('indicatorOfOneRow');
  for (const options of [{maximumIterations: 1}, {maximumIterations: 3}, {tolerance: 1e-2}]) {
    await expectSeparated(scene.predictors, scene.response, scene.columnCount, options);
  }
});
