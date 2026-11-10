// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPredictionCovariance} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';

/** Double-precision correlation of a prediction covariance at a nonzero separation. */
export function getReferenceCorrelation(
  covariance: GPUPredictionCovariance,
  separation: number
): number {
  const scaled = separation / covariance.range;
  switch (covariance.model) {
    case 'spherical':
      return scaled >= 1 ? 0 : 1 - 1.5 * scaled + 0.5 * scaled ** 3;
    case 'exponential':
      return Math.exp(-3 * scaled);
    case 'gaussian':
      return Math.exp(-3 * scaled * scaled);
    case 'matern': {
      if (covariance.smoothness === 0.5) {
        return Math.exp(-3 * scaled);
      }
      if (covariance.smoothness === 1.5) {
        const argument = MATERN_3_2_RANGE_FACTOR * scaled;
        return (1 + argument) * Math.exp(-argument);
      }
      const argument = MATERN_5_2_RANGE_FACTOR * scaled;
      return (1 + argument + (argument * argument) / 3) * Math.exp(-argument);
    }
    default:
      throw new Error('Unknown model');
  }
}

/** Double-precision covariance at a pair of coordinates. */
export function getReferenceCovariance(
  covariance: GPUPredictionCovariance,
  first: readonly [number, number],
  second: readonly [number, number]
): number {
  const separation = Math.hypot(first[0] - second[0], first[1] - second[1]);
  return (
    covariance.sill * getReferenceCorrelation(covariance, separation) +
    (separation === 0 ? (covariance.nugget ?? 0) : 0)
  );
}

/** Solves `matrix x = right` by Gaussian elimination with partial pivoting, in binary64. */
function solveDense(matrix: Float64Array[], right: Float64Array): {solution: Float64Array} {
  const size = right.length;
  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivotRow][column])) pivotRow = row;
    }
    [matrix[column], matrix[pivotRow]] = [matrix[pivotRow], matrix[column]];
    [right[column], right[pivotRow]] = [right[pivotRow], right[column]];
    for (let row = column + 1; row < size; row++) {
      const factor = matrix[row][column] / matrix[column][column];
      for (let entry = column; entry < size; entry++)
        matrix[row][entry] -= factor * matrix[column][entry];
      right[row] -= factor * right[column];
    }
  }
  const solution = new Float64Array(size);
  for (let row = size - 1; row >= 0; row--) {
    let value = right[row];
    for (let column = row + 1; column < size; column++)
      value -= matrix[row][column] * solution[column];
    solution[row] = value / matrix[row][row];
  }
  return {solution};
}

/** Independent binary64 local kriging of one query, nearest `neighborCount` samples. */
export function solveReferenceKriging(props: {
  mode: 'simple' | 'ordinary';
  positions: readonly (readonly [number, number])[];
  /** Rows `[value, observationErrorVariance]`. */
  observations: readonly (readonly [number, number])[];
  query: readonly [number, number];
  covariance: GPUPredictionCovariance;
  neighborCount: number;
  mean?: number;
}): {prediction: number; variance: number; weights: number[]; order: number[]} {
  const {positions, observations, query, covariance, neighborCount, mode} = props;
  const order = positions
    .map(
      (position, index) =>
        [Math.hypot(position[0] - query[0], position[1] - query[1]), index] as const
    )
    .sort((first, second) => first[0] - second[0] || first[1] - second[1])
    .slice(0, neighborCount)
    .map(([, index]) => index);
  const mean = props.mean ?? 0;
  const count = order.length;
  const size = count + (mode === 'ordinary' ? 1 : 0);
  const matrix = Array.from({length: size}, () => new Float64Array(size));
  const right = new Float64Array(size);
  const cross = new Float64Array(count);
  for (let row = 0; row < count; row++) {
    for (let column = 0; column < count; column++) {
      matrix[row][column] = getReferenceCovariance(
        covariance,
        positions[order[row]],
        positions[order[column]]
      );
    }
    matrix[row][row] += observations[order[row]][1];
    cross[row] = getReferenceCovariance(covariance, positions[order[row]], query);
    right[row] = cross[row];
    if (mode === 'ordinary') {
      matrix[row][count] = 1;
      matrix[count][row] = 1;
    }
  }
  if (mode === 'ordinary') {
    right[count] = 1;
  }
  const {solution} = solveDense(matrix, right);
  const weights = Array.from(solution.subarray(0, count));
  let prediction = mode === 'ordinary' ? 0 : mean;
  let variance = getReferenceCovariance(covariance, query, query);
  for (let row = 0; row < count; row++) {
    prediction += weights[row] * (observations[order[row]][0] - (mode === 'ordinary' ? 0 : mean));
    variance -= weights[row] * cross[row];
  }
  if (mode === 'ordinary') {
    variance -= solution[count];
  }
  return {prediction, variance, weights, order};
}
