// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Slots of the float32 state buffer of {@link GPUSinkhornDivergence}. */
export const SINKHORN_DIVERGENCE_STATE_SLOTS = {
  /** Regularization the next blur pass uses. */
  epsilon: 0,
  done: 1,
  iterationCount: 2,
  converged: 3,
  iterationLimitReached: 4,
  /** L1 error of the target marginal, summed over cells by the last iteration. */
  marginalError: 5,
  /** One when the masses and parameters allow a fit. */
  valid: 6
} as const;

/** Length of the float32 state buffer. */
export const SINKHORN_DIVERGENCE_STATE_LENGTH = 8;

/**
 * Log of a mass that is zero. A finite stand-in for minus infinity (WGSL has no portable
 * infinity); terms at or below half of it are skipped, never summed.
 */
export const SINKHORN_DIVERGENCE_MASKED_LOG = -1e30;

/** Potential slots in the `potentials` buffer: each is one block of `width * height` floats. */
export const SINKHORN_POTENTIAL_SLOTS = {
  /** Source-side potential of the (source, target) problem. */
  sourceOfCross: 0,
  /** Target-side potential of the (source, target) problem. */
  targetOfCross: 1,
  /** Potential of the (source, source) problem. */
  sourceOfSelf: 2,
  /** Potential of the (target, target) problem. */
  targetOfSelf: 3
} as const;

/**
 * WGSL helpers of the log-domain blur: `accumulateLogSum` adds `value` to the running
 * `(maximum, sum)` pair (`sum` is the sum of `exp(term - maximum)`), skipping masked terms.
 */
export const SINKHORN_LOG_SUM_WGSL = `const MASKED: f32 = ${SINKHORN_DIVERGENCE_MASKED_LOG};
const MASKED_LIMIT: f32 = ${SINKHORN_DIVERGENCE_MASKED_LOG / 2};
fn accumulateLogSum(value: f32, maximum: ptr<function, f32>, sum: ptr<function, f32>) {
  if (value <= MASKED_LIMIT) {
    return;
  }
  if (value > *maximum) {
    *sum = *sum * exp(*maximum - value) + 1.0;
    *maximum = value;
  } else {
    *sum += exp(value - *maximum);
  }
}`;
