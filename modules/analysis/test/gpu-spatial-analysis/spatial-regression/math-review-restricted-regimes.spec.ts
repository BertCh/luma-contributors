// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {
  createNormalRandom,
  createRandom,
  createRegressionTestContext,
  fitLeastSquares,
  getNormalDistribution,
  solveDense
} from './math-review-regression-test-utils';

type RegimeScene = {
  predictorCount: number;
  regimeCount: number;
  /** Intercept first. */
  common: boolean[];
  predictors: Float32Array;
  response: Float32Array;
  regimes: Uint32Array;
};

/** Float64 reference of the restricted model, the Chow F and the Wald stability tests. */
function fitReference(scene: RegimeScene) {
  const {predictorCount, regimeCount, common, predictors, response, regimes} = scene;
  const rows = response.length;
  const offsets: number[] = [];
  let restrictedCount = 0;
  for (const shared of common) {
    offsets.push(restrictedCount);
    restrictedCount += shared ? 1 : regimeCount;
  }
  const unrestrictedCount = (predictorCount + 1) * regimeCount;
  const restrictedDesign: number[][] = [];
  const unrestrictedDesign: number[][] = [];
  for (let row = 0; row < rows; row++) {
    const values = [
      1,
      ...Array.from(predictors.subarray(row * predictorCount, (row + 1) * predictorCount))
    ];
    const restrictedRow = new Array<number>(restrictedCount).fill(0);
    const unrestrictedRow = new Array<number>(unrestrictedCount).fill(0);
    values.forEach((value, column) => {
      restrictedRow[offsets[column] + (common[column] ? 0 : regimes[row])] = value;
      unrestrictedRow[column * regimeCount + regimes[row]] = value;
    });
    restrictedDesign.push(restrictedRow);
    unrestrictedDesign.push(unrestrictedRow);
  }
  const responseValues = Array.from(response);
  const restricted = fitLeastSquares(restrictedDesign, responseValues);
  const unrestricted = fitLeastSquares(unrestrictedDesign, responseValues);
  const variance = restricted.squares / (rows - restrictedCount);
  const covariance = restricted.inverse.map(row => row.map(value => value * variance));
  const restrictionCount = unrestrictedCount - restrictedCount;
  const chow =
    (restricted.squares - unrestricted.squares) /
    restrictionCount /
    (unrestricted.squares / (rows - unrestrictedCount));
  // Wald tests of equal varying coefficients: one row per source column, then the joint row.
  const getWald = (columns: number[]) => {
    const contrasts: [number, number][] = [];
    for (const column of columns)
      if (!common[column])
        for (let regime = 1; regime < regimeCount; regime++)
          contrasts.push([offsets[column] + regime, offsets[column]]);
    if (contrasts.length === 0) return {statistic: Number.NaN, degrees: 0};
    const difference = contrasts.map(
      ([positive, negative]) => restricted.estimate[positive] - restricted.estimate[negative]
    );
    const matrix = contrasts.map(([positive, negative]) =>
      contrasts.map(
        ([otherPositive, otherNegative]) =>
          covariance[positive][otherPositive] -
          covariance[positive][otherNegative] -
          covariance[negative][otherPositive] +
          covariance[negative][otherNegative]
      )
    );
    const solved = solveDense(matrix, difference);
    return {
      statistic: difference.reduce((sum, value, index) => sum + value * solved[index], 0),
      degrees: contrasts.length
    };
  };
  const allColumns = Array.from({length: predictorCount + 1}, (_, column) => column);
  const stability = [...allColumns.map(column => getWald([column])), getWald(allColumns)];
  return {
    restrictedCount,
    unrestrictedCount,
    estimate: restricted.estimate,
    covariance,
    chow,
    restrictionCount,
    stability
  };
}

async function runRestricted(scene: RegimeScene, restrictedCount: number) {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  const rows = scene.response.length;
  try {
    const coefficients = output('float32', restrictedCount);
    const covariance = output('float32', restrictedCount * restrictedCount);
    const chow = output('float32', 3);
    const stability = output('float32', (scene.predictorCount + 2) * 3);
    const status = output('uint32', 1);
    graph.add(
      new GPURestrictedRegimeRegression({
        predictors: input('float32', scene.predictors),
        predictorCount: scene.predictorCount,
        response: input('float32', scene.response),
        regimes: input('uint32', scene.regimes),
        regimeCount: scene.regimeCount,
        common: scene.common,
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          chow: chow.view,
          stability: stability.view,
          fitted: output('float32', rows).view,
          status: status.view
        }
      })
    );
    context.run();
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from(await readFloat32(coefficients.buffer, restrictedCount)),
      covariance: Array.from(
        await readFloat32(covariance.buffer, restrictedCount * restrictedCount)
      ),
      chow: Array.from(await readFloat32(chow.buffer, 3)),
      stability: Array.from(await readFloat32(stability.buffer, (scene.predictorCount + 2) * 3))
    };
  } finally {
    context.destroy();
  }
}

function expectMatchesReference(
  scene: RegimeScene,
  reference: ReturnType<typeof fitReference>,
  result: Awaited<ReturnType<typeof runRestricted>>,
  tolerance: number
) {
  const count = reference.restrictedCount;
  expect(result.status).toBe(0);
  reference.estimate.forEach((value, index) =>
    expect(
      Math.abs(result.coefficients[index] - value),
      `coefficient ${index}: ${result.coefficients[index]} vs ${value}`
    ).toBeLessThan(tolerance * Math.max(Math.abs(value), 1e-3))
  );
  for (let row = 0; row < count; row++)
    for (let column = 0; column < count; column++) {
      const scale = Math.sqrt(
        reference.covariance[row][row] * reference.covariance[column][column]
      );
      expect(
        Math.abs(result.covariance[row * count + column] - reference.covariance[row][column]),
        `covariance ${row},${column}`
      ).toBeLessThan(20 * tolerance * scale);
    }
  expect(result.chow[1]).toBe(reference.restrictionCount);
  expect(result.chow[2]).toBe(scene.response.length - reference.unrestrictedCount);
  expect(
    Math.abs(result.chow[0] - reference.chow),
    `Chow F ${result.chow[0]} vs ${reference.chow}`
  ).toBeLessThan(20 * tolerance * Math.max(reference.chow, 1));
  reference.stability.forEach((expected, row) => {
    expect(result.stability[row * 3 + 1]).toBe(expected.degrees);
    if (expected.degrees === 0) {
      expect(Number.isNaN(result.stability[row * 3])).toBe(true);
      return;
    }
    expect(
      Math.abs(result.stability[row * 3] - expected.statistic),
      `Wald ${row}: ${result.stability[row * 3]} vs ${expected.statistic}`
    ).toBeLessThan(20 * tolerance * Math.max(expected.statistic, 1));
    if (expected.degrees === 1) {
      // Chi-square survival with one degree of freedom is the two-sided normal tail.
      const probability = 2 * (1 - getNormalDistribution(Math.sqrt(expected.statistic)));
      expect(Math.abs(result.stability[row * 3 + 2] - probability)).toBeLessThan(
        2e-3 * Math.max(probability, 1e-3) + 20 * tolerance
      );
    }
  });
}

/**
 * The three large-mean predictors of regression-02: two regimes of 200 rows, a slope of 0.5
 * common to both, intercepts that differ by regime.
 */
function createLargeMeanScene(center: number, halfRange: number, seed: number): RegimeScene {
  const random = createRandom(seed);
  const normal = createNormalRandom(seed + 1);
  const rows = 400;
  const predictors = new Float32Array(rows);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const regime = row % 2;
    regimes[row] = regime;
    predictors[row] = center + (2 * random() - 1) * halfRange;
    // The response is of the order of the spread, so its own float32 rounding is not the test.
    response[row] =
      3 + 0.4 * halfRange * regime + 0.5 * (predictors[row] - center) + 0.1 * halfRange * normal();
  }
  return {predictorCount: 1, regimeCount: 2, common: [false, true], predictors, response, regimes};
}

for (const [name, center, halfRange] of [
  ['a year (2010 +- 10)', 2010, 10],
  ['an elevation (1500 +- 3)', 1500, 3],
  ['a UTM northing (5.2e6 +- 5000)', 5.2e6, 5000]
] as const)
  it(`restricted regimes with a common slope on ${name} match float64 least squares (regression-02)`, async () => {
    const scene = createLargeMeanScene(center, halfRange, 20261110);
    const reference = fitReference(scene);
    expect(Math.abs(reference.estimate[2] - 0.5)).toBeLessThan(0.05);
    const result = await runRestricted(scene, reference.restrictedCount);
    // The former raw normal equations were 0.8%, 11% and 31% off in the slope on these scenes.
    expectMatchesReference(scene, reference, result, 2e-4);
  });

it('restricted regimes with varying slopes on a large-mean predictor match float64 least squares', async () => {
  // Three regimes, two predictors: the elevation slope varies, the second slope is common.
  const random = createRandom(5);
  const normal = createNormalRandom(6);
  const rows = 900;
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const regime = row % 3;
    regimes[row] = regime;
    predictors[row * 2] = 1500 + regime + (2 * random() - 1) * 3;
    predictors[row * 2 + 1] = 40000 + 500 * normal();
    response[row] =
      1 +
      regime +
      (0.5 + 0.2 * regime) * (predictors[row * 2] - 1500) +
      0.002 * (predictors[row * 2 + 1] - 40000) +
      0.3 * normal();
  }
  const scene: RegimeScene = {
    predictorCount: 2,
    regimeCount: 3,
    common: [false, false, true],
    predictors,
    response,
    regimes
  };
  const reference = fitReference(scene);
  const result = await runRestricted(scene, reference.restrictedCount);
  expectMatchesReference(scene, reference, result, 5e-4);
});

it('restricted regimes with a common intercept match float64 least squares', async () => {
  // Common intercept, a varying slope on an order-1 predictor and a common slope on a predictor
  // with a large mean (income 50000 +- 500): the pooled shift serves the common column.
  const random = createRandom(11);
  const normal = createNormalRandom(12);
  const rows = 600;
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const regime = row % 2;
    regimes[row] = regime;
    predictors[row * 2] = 2 * random() - 0.5;
    predictors[row * 2 + 1] = 50000 + 500 * normal() + 100 * regime;
    response[row] =
      2 +
      (1 + regime) * predictors[row * 2] +
      0.004 * (predictors[row * 2 + 1] - 50000) +
      0.2 * normal();
  }
  const scene: RegimeScene = {
    predictorCount: 2,
    regimeCount: 2,
    common: [true, false, true],
    predictors,
    response,
    regimes
  };
  const reference = fitReference(scene);
  const result = await runRestricted(scene, reference.restrictedCount);
  expectMatchesReference(scene, reference, result, 5e-4);
});

/**
 * A varying slope on a large-mean predictor under a COMMON intercept (fr-statistics-01): 600 rows,
 * two regimes. The raw columns `x 1[g]` are then almost parallel to the constant, and the kernel
 * solves in the basis `{x - mean, x 1[g]}` instead.
 */
function createCommonInterceptScene(center: number, halfRange: number): RegimeScene {
  const random = createRandom(41);
  const normal = createNormalRandom(42);
  const rows = 600;
  const predictors = new Float32Array(rows);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const regime = row % 2;
    regimes[row] = regime;
    predictors[row] = center + (2 * random() - 1) * halfRange;
    response[row] =
      2 + ((0.5 + 0.1 * regime) * (predictors[row] - center)) / halfRange + 0.2 * normal();
  }
  return {predictorCount: 1, regimeCount: 2, common: [true, false], predictors, response, regimes};
}

for (const [name, center, halfRange] of [
  ['a year (2010 +- 10)', 2010, 10],
  ['an elevation (1500 +- 3)', 1500, 3],
  ['a UTM northing (5.2e6 +- 5000)', 5.2e6, 5000]
] as const)
  it(`restricted regimes with a common intercept and a varying slope on ${name} match float64 least squares (fr-statistics-01)`, async () => {
    const scene = createCommonInterceptScene(center, halfRange);
    const reference = fitReference(scene);
    const result = await runRestricted(scene, reference.restrictedCount);
    // The raw basis was 0.28% and 3.4% off in both slopes on the first two scenes with status 0,
    // and status 2 on the third although the model is identified.
    expectMatchesReference(scene, reference, result, 2e-4);
  });

it('restricted regimes with a common intercept, two varying slopes and a common slope, all with large means, match float64 least squares', async () => {
  // Three regimes. The elevation is the pivot (largest mean against its spread), the temperature
  // (285 +- 4) is a second varying slope and the income (50000 +- 500) a common slope.
  const random = createRandom(51);
  const normal = createNormalRandom(52);
  const rows = 900;
  const predictors = new Float32Array(rows * 3);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const regime = row % 3;
    regimes[row] = regime;
    predictors[row * 3] = 1500 + 0.5 * regime + (2 * random() - 1) * 3;
    predictors[row * 3 + 1] = 285 + 4 * normal();
    predictors[row * 3 + 2] = 50000 + 500 * normal();
    response[row] =
      4 +
      0.3 * regime +
      (0.5 + 0.2 * regime) * (predictors[row * 3] - 1500) +
      (0.1 - 0.05 * regime) * (predictors[row * 3 + 1] - 285) +
      0.002 * (predictors[row * 3 + 2] - 50000) +
      0.3 * normal();
  }
  const scene: RegimeScene = {
    predictorCount: 3,
    regimeCount: 3,
    common: [true, false, false, true],
    predictors,
    response,
    regimes
  };
  const reference = fitReference(scene);
  const result = await runRestricted(scene, reference.restrictedCount);
  expectMatchesReference(scene, reference, result, 5e-4);
});

it('restricted regimes with a common intercept and a response mean far above its residuals match float64 least squares', async () => {
  // Response near 10,000 with residuals of 0.05: a residual formed as response - (shift +
  // intercept) instead of (response - shift) - intercept would round by half a float32 step of
  // the response mean (0.0005), which is the resolution of the response itself. The order-1
  // varying predictor keeps the raw basis (no pivot), where the shifted intercept is not zero.
  const random = createRandom(61);
  const normal = createNormalRandom(62);
  const rows = 600;
  const predictors = new Float32Array(rows);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const regime = row % 2;
    regimes[row] = regime;
    predictors[row] = 2 * random() - 0.5;
    response[row] = 10000 + (1 + regime) * predictors[row] + 0.05 * normal();
  }
  const scene: RegimeScene = {
    predictorCount: 1,
    regimeCount: 2,
    common: [true, false],
    predictors,
    response,
    regimes
  };
  const reference = fitReference(scene);
  const result = await runRestricted(scene, reference.restrictedCount);
  expectMatchesReference(scene, reference, result, 5e-4);
});

/** Two predictors that agree to `spread` (relative), in two regimes. */
function createCollinearScene(spread: number): RegimeScene {
  const normal = createNormalRandom(31);
  const rows = 400;
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  const regimes = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    regimes[row] = row % 2;
    const value = normal();
    predictors[row * 2] = value;
    predictors[row * 2 + 1] = value + spread * normal();
    response[row] = 1 + value + 0.1 * normal();
  }
  return {
    predictorCount: 2,
    regimeCount: 2,
    common: [false, true, true],
    predictors,
    response,
    regimes
  };
}

// The second predictor is the first plus `spread` of independent noise, so the squared cosine of
// the two centered columns is 1 - spread^2: 1e-5 is far beyond the OLS pivot limit of 1 - 1e-6 and
// 7e-4 (1 - 4.9e-7) just beyond it, where the old positivity test still had a positive pivot.
for (const spread of [1e-5, 7e-4])
  it(`restricted and fully varying regimes report predictors with a spread of ${spread} as singular (regression-10)`, async () => {
    const scene = createCollinearScene(spread);
    const restricted = await runRestricted(scene, 4);
    expect(restricted.status).toBe(2);
    expect(restricted.coefficients.every(value => Number.isNaN(value))).toBe(true);

    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const status = output('uint32', 2);
      const coefficients = output('float32', 6);
      graph.add(
        new GPUSpatialRegimeRegression({
          predictors: input('float32', scene.predictors),
          predictorCount: 2,
          response: input('float32', scene.response),
          regimes: input('uint32', scene.regimes),
          regimeCount: 2,
          output: {
            coefficients: coefficients.view,
            fitted: output('float32', scene.response.length).view,
            status: status.view,
            counts: output('uint32', 2).view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 2))).toEqual([2, 2]);
      expect(Array.from(await readFloat32(coefficients.buffer, 6)).every(Number.isNaN)).toBe(true);
    } finally {
      context.destroy();
    }
  });

it('regimes with moderately correlated predictors still fit (pivot limit control)', async () => {
  // A correlation of about 0.995 (spread 0.1) is well inside the limit.
  const scene = createCollinearScene(0.1);
  const reference = fitReference(scene);
  const result = await runRestricted(scene, reference.restrictedCount);
  expectMatchesReference(scene, reference, result, 2e-3);
});

it('regimes with a squared cosine of 1 - 4e-6 between predictors are inside the pivot limit', async () => {
  // Spread 2e-3: just inside the limit of 1 - 1e-6. The float32 solution of so ill-conditioned a
  // system is not compared with float64; the status is what pins the limit from this side.
  const scene = createCollinearScene(2e-3);
  const result = await runRestricted(scene, 4);
  expect(result.status).toBe(0);
  expect(result.coefficients.every(Number.isFinite)).toBe(true);
});
