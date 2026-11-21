// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import type {GPUGeographicallyWeightedRegressionSettings} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  getSwapOrNotPermutationIndex,
  getSwapOrNotRounds
} from '../../../src/analysis-core/kernels/permutation-shuffle';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {validateCapacity} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

/** Scene of the non-stationarity specs. */
export type NonstationarityScene = {
  positions: Float32Array;
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
  mask?: Uint32Array;
};

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/**
 * Jittered `side x side` grid on `[0, 10]^2` with two predictors and a response whose first slope
 * varies strongly west to east, whose second slope is constant and whose intercept varies mildly,
 * so the first slope must test as non-stationary and the second must not.
 */
export function createNonstationarityScene(side: number, seed: number): NonstationarityScene {
  const random = createRandom(seed);
  const rows = side * side;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const x = ((row % side) + 0.2 + 0.6 * random()) * (10 / side);
    const y = (Math.floor(row / side) + 0.2 + 0.6 * random()) * (10 / side);
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    const x1 = random() * 4 - 2 + 1;
    const x2 = random() * 4 - 2;
    predictors[2 * row] = x1;
    predictors[2 * row + 1] = x2;
    const slope1 = -2 + 0.8 * Math.fround(x);
    const intercept = 1 + 0.1 * Math.fround(y);
    response[row] =
      intercept + slope1 * Math.fround(x1) + 0.5 * Math.fround(x2) + 0.6 * (random() - 0.5);
  }
  return {positions, predictors, predictorCount: 2, response};
}

/**
 * The relabelling of permutation `permutation` (1-based): location `j` receives the observation of
 * row `result[j]`, for `count` included rows (the keyed swap-or-not shuffle of the permutation
 * contributors).
 */
export function getNonstationarityAssignment(
  count: number,
  seed: number,
  permutation: number
): number[] {
  const rounds = getSwapOrNotRounds(
    getPermutationSeedKey(seed),
    permutation,
    count,
    PHILOX_PURPOSES.geographicallyWeightedRegressionNonstationarityShuffle.id
  );
  return Array.from({length: count}, (_, location) =>
    getSwapOrNotPermutationIndex(location, count, rounds)
  );
}

/** Population standard deviation of each coefficient column over rows whose fit succeeded. */
export function getLocalEstimateStandardDeviations(
  coefficients: readonly number[],
  coefficientCount: number
): number[] {
  const rows = coefficients.length / coefficientCount;
  const retained: number[][] = [];
  for (let row = 0; row < rows; row++) {
    const values = coefficients.slice(row * coefficientCount, (row + 1) * coefficientCount);
    if (!values.every(Number.isFinite)) {
      continue;
    }
    retained.push(values);
  }
  return Array.from({length: coefficientCount}, (_, column) => {
    if (!retained.length) return NaN;
    const anchor = retained[0][column];
    const differences = retained.map(values => values[column] - anchor);
    const mean = differences.reduce((sum, value) => sum + value, 0) / differences.length;
    return Math.sqrt(
      differences.reduce((sum, value) => sum + (value - mean) ** 2, 0) / differences.length
    );
  });
}

/** Result of {@link computeNonstationarityOnCPU}. */
export type NonstationarityOracleResult = {
  /** Row 0 is the observed run, rows `1..P` the permutations; each holds one value per coefficient. */
  standardDeviations: number[][];
  /** Per coefficient: `(g + 1) / (P + 1)` with `g = #{sd_p >= sd_observed}`; NaN unless every spread is defined. */
  pseudoPValues: number[];
  exceedances: number[];
};

/**
 * Float64 reference of `GPUGeographicallyWeightedRegressionNonstationarityTest`: every permutation
 * physically rearranges `(X, y)` among the locations by the swap-or-not assignment and refits with the
 * float64 `GPUGeographicallyWeightedRegression` oracle at the same fixed bandwidth, then compares
 * the standard deviations of the local estimates. Throws when permutations is not a positive
 * integer. Undefined observed or requested permuted spreads make the corresponding exceedance
 * count and pseudo p-value NaN.
 */
export function computeNonstationarityOnCPU(
  scene: NonstationarityScene,
  settings: GPUGeographicallyWeightedRegressionSettings,
  permutations: number,
  seed: number
): NonstationarityOracleResult {
  validateCapacity('computeNonstationarityOnCPU', 'permutations', permutations, 1);
  const {predictorCount} = scene;
  const retained = Array.from({length: scene.response.length}, (_, row) => row).filter(
    row =>
      (!scene.mask || scene.mask[row] !== 0) &&
      Number.isFinite(scene.response[row]) &&
      Number.isFinite(scene.positions[2 * row]) &&
      Number.isFinite(scene.positions[2 * row + 1]) &&
      Array.from(
        {length: predictorCount},
        (_, column) => scene.predictors[row * predictorCount + column]
      ).every(Number.isFinite)
  );
  const positions = Float32Array.from(
    retained.flatMap(row => [scene.positions[2 * row], scene.positions[2 * row + 1]])
  );
  const predictors = Float32Array.from(
    retained.flatMap(row =>
      Array.from(
        {length: predictorCount},
        (_, column) => scene.predictors[row * predictorCount + column]
      )
    )
  );
  const response = Float32Array.from(retained.map(row => scene.response[row]));
  const rows = response.length;
  const coefficientCount = predictorCount + 1;
  const observed = computeGeographicallyWeightedRegressionOnCPU({
    positions,
    predictors,
    predictorCount,
    response,
    settings
  });
  const fixedSettings = {...settings, bandwidths: [observed.selectedValue]};
  const run = (assignment: readonly number[]) => {
    const permutedPredictors = new Float32Array(predictors.length);
    const permutedResponse = new Float32Array(rows);
    for (let location = 0; location < rows; location++) {
      const source = assignment[location];
      for (let column = 0; column < predictorCount; column++) {
        permutedPredictors[location * predictorCount + column] =
          predictors[source * predictorCount + column];
      }
      permutedResponse[location] = response[source];
    }
    const result = computeGeographicallyWeightedRegressionOnCPU({
      positions,
      predictors: permutedPredictors,
      predictorCount,
      response: permutedResponse,
      settings: fixedSettings
    });
    return getLocalEstimateStandardDeviations(result.coefficients, coefficientCount);
  };
  const standardDeviations = [
    getLocalEstimateStandardDeviations(observed.coefficients, coefficientCount)
  ];
  for (let permutation = 1; permutation <= permutations; permutation++) {
    standardDeviations.push(run(getNonstationarityAssignment(rows, seed, permutation)));
  }
  const exceedances = standardDeviations[0].map((observed, column) =>
    Number.isFinite(observed) &&
    standardDeviations.slice(1).every(row => Number.isFinite(row[column]))
      ? standardDeviations.slice(1).filter(row => row[column] >= observed).length
      : NaN
  );
  return {
    standardDeviations,
    exceedances,
    pseudoPValues: exceedances.map(count => (count + 1) / (permutations + 1))
  };
}
