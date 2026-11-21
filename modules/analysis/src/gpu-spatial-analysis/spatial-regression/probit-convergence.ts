// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Largest accepted value of the ordinary probit tolerance prop. */
export const MAXIMUM_PROBIT_TOLERANCE = 1e-2;

/** Largest relative per-coefficient step threshold that may certify an ordinary probit fit. */
export const MAXIMUM_PROBIT_CONVERGENCE_TOLERANCE = 1e-5;

/** Smallest positive normal float32 (2^-126): the floor of the threshold the GPU kernel receives. */
const SMALLEST_NORMAL_FLOAT32 = 2 ** -126;

/**
 * Validate the requested tolerance before float32 rounding and cap the success certificate.
 */
export function resolveProbitConvergenceTolerance(id: string, tolerance?: number): number {
  const requested = tolerance ?? MAXIMUM_PROBIT_CONVERGENCE_TOLERANCE;
  validateRequirement(
    id,
    Number.isFinite(requested) && requested > 0 && requested <= MAXIMUM_PROBIT_TOLERANCE,
    `tolerance must be a positive finite number at most ${MAXIMUM_PROBIT_TOLERANCE}; received ${String(requested)}`
  );
  return Math.min(requested, MAXIMUM_PROBIT_CONVERGENCE_TOLERANCE);
}

/**
 * The threshold the GPU kernel compares in float32: the resolved tolerance rounded to float32,
 * floored at the smallest normal float32 (2^-126) so that a very tight request never becomes a
 * zero threshold (Apple Metal flushes subnormals). Requests below the float32 relative precision
 * (about 1.2e-7) ask for more than a float32 step can show and usually end at status 3.
 */
export function getGPUProbitConvergenceThreshold(id: string, tolerance?: number): number {
  return Math.max(
    Math.fround(resolveProbitConvergenceTolerance(id, tolerance)),
    SMALLEST_NORMAL_FLOAT32
  );
}

/**
 * Per-coefficient convergence test: every |step[c]| <= tolerance * (1 + |estimate[c]|). A test
 * against the largest coefficient would let one large coefficient certify an unrelated small one.
 */
export function isProbitStepWithinTolerance(
  steps: readonly number[],
  estimates: readonly number[],
  tolerance: number
): boolean {
  return steps.every(
    (step, column) => Math.abs(step) <= tolerance * (1 + Math.abs(estimates[column]))
  );
}

/**
 * WGSL verdict of one coefficient's Fisher-scoring step, the kernel form of
 * `isProbitStepWithinTolerance`: 0 when the accepted step `scale * direction` is above
 * `tolerance * (1 + |candidate|)`; 1 when the accepted step is within it but the full unshortened
 * direction is not (a step shortened by the bound or by likelihood backtracking: not converged);
 * 2 when both are within it. The fit's verdict is the minimum over its coefficients.
 */
export const PROBIT_STEP_VERDICT_WGSL = /* wgsl */ `fn getProbitStepVerdict(scale: f32, direction: f32, candidate: f32, tolerance: f32) -> u32 {
 let threshold = tolerance * (1.0 + abs(candidate));
 if (!(abs(scale * direction) <= threshold)) { return 0u; }
 if (!(abs(direction) <= threshold)) { return 1u; }
 return 2u;
}`;
