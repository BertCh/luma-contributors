// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createConstantFittedReviewInput,
  createRationalAffineReviewInput
} from './spatial-regression-review-fixtures';

export function createBipartiteDiagnosticsInput() {
  // W has only cross-partition links: its range is span(1, partition), so MWM=0.
  const predictors = Float32Array.from({length: 16}, (_, row) => Number(row >= 8));
  const residuals = Float32Array.from({length: 16}, (_, row) => (row % 4 < 2 ? 1 : -1));
  return {
    predictors,
    residuals,
    response: Float32Array.from(predictors, (value, row) => 2 + value + residuals[row]),
    weights: {
      offsets: Uint32Array.from({length: 17}, (_, row) => 8 * row),
      neighbors: Uint32Array.from({length: 128}, (_, slot) => (slot < 64 ? 8 : 0) + (slot % 8)),
      weights: new Float32Array(128).fill(1)
    }
  };
}

export function createWeightedConstantLagInput() {
  const input = createConstantFittedReviewInput();
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < 64; row++) {
    if (row % 2 === 0) {
      neighbors.push(row + 1);
      weights.push(1);
    } else
      for (let neighbor = 0; neighbor < 64; neighbor += 2) {
        neighbors.push(neighbor);
        weights.push(1 / 32);
      }
    offsets.push(neighbors.length);
  }
  // Each row sums to 1. ||W||²=33 and tr(W²)=2, hence T=J=35 for constant fitted values.
  return {
    ...input,
    weights: {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: Float32Array.from(weights)
    }
  };
}

export function createTranslatedAffineInput(departure = false) {
  const input = createRationalAffineReviewInput();
  const predictors = Float32Array.from(input.response, value => 7 * value - 2);
  const response = Float32Array.from(input.response, value => 2396745 + value);
  // 7y=x+16777217 exactly; the next response float differs by 1/4, breaking affine rank.
  if (departure) response[0] += 1 / 4;
  return {...input, predictors, response};
}
