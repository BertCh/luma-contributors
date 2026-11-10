// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUSinkhornDivergence} parameter view. */
export const GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH = 6;

/** Default largest L1 error of the target marginal that counts as converged. */
export const GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE = 1e-4;

/** Per-frame settings of {@link GPUSinkhornDivergence}. */
export type GPUSinkhornDivergenceSettings = {
  /** Cell width in position units. Positive and finite. */
  cellWidth: number;
  /** Cell height in position units. Positive and finite. */
  cellHeight: number;
  /**
   * Entropic regularization, in squared position units (the cost is the squared distance between
   * cell centers). Positive. In float32 the log-domain exponents reach `maximum cost / epsilon`,
   * and each carries an absolute rounding error of about `6e-8` times that, so keep `epsilon` at
   * least about `1e-3` times the squared grid diagonal.
   */
  epsilon: number;
  /**
   * Largest L1 distance between the target marginal of the plan and the target masses that counts
   * as converged. Defaults to {@link GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE}.
   */
  tolerance?: number;
  /** First regularization of the epsilon schedule, at least `epsilon`. Defaults to `epsilon` (no scaling). */
  epsilonStart?: number;
  /**
   * Factor in `(0, 1]` applied to the regularization each iteration until it reaches `epsilon`:
   * iteration `k` uses `max(epsilon, epsilonStart * scalingFactor^k)`. Defaults to 1.
   */
  scalingFactor?: number;
};

/**
 * Packs per-frame {@link GPUSinkhornDivergence} parameters.
 *
 * Layout (float32): `[cellWidth, cellHeight, epsilon, tolerance, epsilonStart, scalingFactor]`.
 *
 * @param settings Cell size, regularization, tolerance and schedule.
 * @param target Optional destination of at least {@link GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH} elements.
 * @throws If a value is not positive and finite, `epsilonStart` is below `epsilon`, or
 * `scalingFactor` is outside `(0, 1]`.
 */
export function getGPUSinkhornDivergenceParameterValues(
  settings: GPUSinkhornDivergenceSettings,
  target: Float32Array = new Float32Array(GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUSinkhornDivergenceParameterValues';
  if (target.length < GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH) {
    throw new Error(
      `${name} target must hold ${GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH} elements`
    );
  }
  const values = [
    ['cellWidth', settings.cellWidth],
    ['cellHeight', settings.cellHeight],
    ['epsilon', settings.epsilon],
    ['tolerance', settings.tolerance ?? GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE],
    ['epsilonStart', settings.epsilonStart ?? settings.epsilon],
    ['scalingFactor', settings.scalingFactor ?? 1]
  ] as const;
  for (const [label, value] of values) {
    if (!(Math.fround(value) > 0) || !Number.isFinite(Math.fround(value))) {
      throw new Error(`${name} ${label} must be positive and finite`);
    }
  }
  if (!(Math.fround(values[4][1]) >= Math.fround(settings.epsilon))) {
    throw new Error(`${name} epsilonStart must be at least epsilon`);
  }
  if (!(Math.fround(values[5][1]) <= 1)) {
    throw new Error(`${name} scalingFactor must be in (0, 1]`);
  }
  target.set(values.map(([, value]) => value));
  return target;
}
