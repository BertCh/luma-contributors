// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createConstantFittedReviewInput,
  createReviewInput
} from './spatial-regression-review-fixtures';

/**
 * W5-31: the 64-row constant-fitted fixture (x, y and residuals unchanged) on a weighted graph.
 * Each even row has one unit link to the next odd row; each odd row has 32 links of weight 1/32
 * to the even rows. Every row sums to 1, so the lag of the constant fit is exactly constant and
 * J = T = tr(W'W) + tr(W^2) = (32 + 32 * 32 / 32^2) + 2 * 32 / 32 = 35.
 */
export function createWeightedConstantFittedInput() {
  const input = createConstantFittedReviewInput();
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < 64; row++) {
    if (row % 2 === 0) {
      neighbors.push(row + 1);
      weights.push(1);
    } else
      for (let even = 0; even < 64; even += 2) {
        neighbors.push(even);
        weights.push(1 / 32);
      }
    offsets.push(neighbors.length);
  }
  return {
    ...input,
    weights: {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: Float32Array.from(weights)
    }
  };
}

/**
 * W5-30: K(8,8) with the partition indicator as predictor and residuals that sum to zero within
 * each partition. W = a b' + b a' for the partition indicators a, b, which lie in the design span,
 * so M W M = 0: the Moran variance is exactly zero and its z and p are undefined. A positive
 * increment on the first link makes the operator nonzero again (nearly complete controls).
 */
export function createBipartiteReviewInput(increment = 0) {
  const count = 16;
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < count; row++) {
    for (let neighbor = 0; neighbor < count; neighbor++)
      if (row < 8 !== neighbor < 8) {
        neighbors.push(neighbor);
        weights.push(neighbors.length === 1 ? 1 + increment : 1);
      }
    offsets.push(neighbors.length);
  }
  const predictors = Float32Array.from({length: count}, (_, row) => (row < 8 ? 0 : 1));
  const residuals = Float32Array.from({length: count}, (_, row) => (row % 2 ? -1 : 1));
  return {
    predictors,
    residuals,
    response: Float32Array.from(predictors, (value, row) => 2 + value + residuals[row]),
    weights: {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: Float32Array.from(weights)
    }
  };
}

/**
 * W5-32: x = 7t - 2 and y = 2396745 + t for the integer t of the rational review fixture, so
 * 7y = x + 16777217 exactly; 16777217 = 2^24 + 1 is not a float32. The departure moves y[0] by one
 * ULP (2^-2 at 2396746), which raises the augmented rational rank from 2 to 3.
 */
export function createTranslatedRationalAffineInput(departure = false) {
  const input = createReviewInput();
  const steps = [1, 3, -1, -3, 2, -2, 7, -7, 5, -5, 6, -6];
  const predictors = Float32Array.from(steps, step => 7 * step - 2);
  const response = Float32Array.from(steps, step => 2396745 + step);
  if (departure) response[0] += 0.25;
  return {...input, predictors, response};
}

/**
 * W5-33: 524,288 rows, one outlier of zero and every other value float32(1.9793816). The centered
 * sum of squares is v^2 (n - 1) / n = 3.9179439 (float64 from the float32 value), positive however
 * the rows are ordered. With one unit residual at the outlier and identical design/instruments,
 * NumPy's float64 HC0 sandwich (X'X)^-1 X' diag(u^2) X (X'X)^-1, X = [1, x], is
 * [[1, -0.5052083], [-0.5052083, 0.2552354]] for the outlier at row 0 or at the last row
 * (`.typecheck/r19/scratch/spatial-stats-w5-regression-weights-and-diagnostics/iv_outlier_reference.py`).
 */
export const IV_OUTLIER_HC0 = [1, -0.5052083, -0.5052083, 0.2552354];
export const IV_OUTLIER_ROW_COUNT = 524288;
export function createInstrumentalVariablesOutlierColumn(outlierRow: number) {
  const column = new Float32Array(IV_OUTLIER_ROW_COUNT).fill(Math.fround(1.9793816));
  column[outlierRow] = 0;
  return column;
}
