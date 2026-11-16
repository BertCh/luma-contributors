// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUForestInferenceModelShape} from '../../../src/dataframe-analysis/forest-inference/index';

/** Closed-form output after zeroing every leaf and setting raw base scores to [1, 0, ...]. */
export function getZeroLeafReplayExpectation(model: GPUForestInferenceModelShape): {
  baseScores: number[];
  scores: number[];
  prediction: number;
} {
  const baseScores = new Array<number>(model.outputCount).fill(0);
  baseScores[0] = 1;
  let scores = baseScores.slice();
  if (model.transform === 'sigmoid') {
    // Binary sigmoid consumes channel 1's zero margin, irrespective of the positive multiplier.
    scores = [0.5, 0.5];
  } else if (model.transform === 'softmax') {
    const denominator = Math.E + model.outputCount - 1;
    scores = baseScores.map((_, channel) => (channel === 0 ? Math.E : 1) / denominator);
  }
  return {baseScores, scores, prediction: model.task === 'regression' ? 1 : 0};
}
