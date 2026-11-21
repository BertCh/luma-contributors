// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGaussianScoreErrorFunction,
  getGaussianScoreSmallTail,
  resolveGaussianPredictiveScoreOptions,
  type GaussianPredictiveScoreOptions
} from './gaussian-predictive-score-math';

/** Input rows of the Gaussian score CPU twin. */
export type CPUGaussianPredictiveScoresInput = {
  means: Float32Array;
  variances: Float32Array;
  observations: Float32Array;
};
/** Requested optional scores; CRPS is always returned, matching the contributor. */
export type CPUGaussianPredictiveScoresOptions = GaussianPredictiveScoreOptions & {
  scores?: readonly ('logScore' | 'brierScore' | 'intervalScore')[];
};
/** One float32 value per predictive mean on every requested port. */
export type CPUGaussianPredictiveScoresResult = {
  crps: Float32Array;
  logScore?: Float32Array;
  brierScore?: Float32Array;
  intervalScore?: Float32Array;
};

/**
 * Binary64 Gaussian score evaluation with one float32 output rounding. Compile-time
 * alpha and threshold use the contributor's float32 settings. Subnormal input values
 * may flush on GPU; the CPU retains them. No GPU dispatch-size cap applies on CPU.
 */
export function computeGaussianPredictiveScoresOnCPU(
  input: CPUGaussianPredictiveScoresInput,
  options: CPUGaussianPredictiveScoresOptions = {}
): CPUGaussianPredictiveScoresResult {
  const id = 'computeGaussianPredictiveScoresOnCPU';
  const count = input.means.length;
  validateCapacity(id, 'means.length', count, 1);
  validateRequirement(
    id,
    input.variances.length >= count && input.observations.length >= count,
    'variances and observations need one row per mean'
  );
  const scores = options.scores ?? [];
  validateRequirement(
    id,
    scores.every(score => ['logScore', 'brierScore', 'intervalScore'].includes(score)),
    'scores must name logScore, brierScore or intervalScore'
  );
  const settings = resolveGaussianPredictiveScoreOptions(
    id,
    options,
    scores.includes('brierScore'),
    scores.includes('intervalScore')
  );
  const result: CPUGaussianPredictiveScoresResult = {
    crps: new Float32Array(count),
    ...Object.fromEntries(scores.map(score => [score, new Float32Array(count)]))
  };
  for (let row = 0; row < count; row++) {
    const mean = input.means[row];
    const variance = input.variances[row];
    const observation = input.observations[row];
    const error = observation - mean;
    const deviation = Math.sqrt(variance);
    const standardized = Math.abs(error) / deviation;
    const finiteMoments =
      Number.isFinite(mean) &&
      Number.isFinite(observation) &&
      Number.isFinite(variance) &&
      variance >= 0;
    result.crps[row] =
      variance === 0
        ? Math.abs(error)
        : variance > 0
          ? Math.abs(error) * getGaussianScoreErrorFunction(standardized / Math.SQRT2) +
            deviation *
              (Math.sqrt(2 / Math.PI) * Math.exp(-0.5 * standardized * standardized) -
                1 / Math.sqrt(Math.PI))
          : NaN;
    if (result.logScore)
      result.logScore[row] =
        finiteMoments && variance > 0
          ? 0.5 * Math.log(2 * Math.PI) + Math.log(deviation) + 0.5 * (error / deviation) ** 2
          : NaN;
    if (result.brierScore) {
      const event = observation > settings.brierThreshold;
      const centered = event ? settings.brierThreshold - mean : mean - settings.brierThreshold;
      const smallTail = getGaussianScoreSmallTail(Math.abs(centered) / deviation);
      const residual =
        variance === 0
          ? Number(mean > settings.brierThreshold !== event)
          : centered <= 0
            ? smallTail
            : 1 - smallTail;
      result.brierScore[row] = finiteMoments ? residual ** 2 : NaN;
    }
    if (result.intervalScore) {
      const halfWidth = settings.intervalMultiplier * deviation;
      result.intervalScore[row] = finiteMoments
        ? 2 * halfWidth + (2 / settings.intervalAlpha) * Math.max(Math.abs(error) - halfWidth, 0)
        : NaN;
    }
  }
  return result;
}
