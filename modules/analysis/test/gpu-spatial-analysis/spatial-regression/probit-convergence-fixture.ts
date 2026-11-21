// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Each predictor group has both outcomes, so no separating direction exists. */
export function createProbitConvergenceFixture() {
  return {
    predictors: Float32Array.from({length: 40}, (_, row) => (row < 20 ? -1 : 1)),
    response: Float32Array.from({length: 40}, (_, row) => (row % 20 < (row < 20 ? 2 : 18) ? 1 : 0))
  };
}

// Closed form: independent binomial groups have MLE probabilities 2/20 and 18/20.
// Thus intercept = 0, slope = scipy.special.ndtri(.9) = 1.2815515655446004;
// log L = 36 log(.9) + 4 log(.1) = -13.003318935657928.
export const PROBIT_CONVERGENCE_SLOPE = 1.2815515655446004;
export const PROBIT_CONVERGENCE_LOG_LIKELIHOOD = -13.003318935657928;

// Expected-information standard errors of both coefficients at the closed-form maximum: each group
// has weight phi(z)^2 / (.9 * .1) with z = ndtri(.9), and the centered design is orthogonal, so
// SE = sqrt(.09 / (40 * phi(z)^2)) =
// math.sqrt(.09 / (40 * scipy.stats.norm.pdf(scipy.special.ndtri(.9)) ** 2)) = 0.2702827108395157.
export const PROBIT_CONVERGENCE_STANDARD_ERROR = 0.2702827108395157;

/**
 * Review SSR-109 B1: one large coefficient must not certify an unrelated one. Columns 1 and 2
 * are nearly collinear (they differ by +-2^-16), so their coefficients are about +-44203; column 3
 * is unrelated. 264 rows: for a, b in {-1, 1}, 16 rows [a, a + b 2^-16, 0] with 12 successes for
 * b = 1 and 4 for b = -1; 100 rows [0, 0, 1] with 99 successes; 100 rows [0, 0, -1] with 1.
 * Both outcomes occur in every cell, so the data are not separated.
 */
export function createProbitLargeCoefficientFixture() {
  const predictors: number[] = [];
  const response: number[] = [];
  for (const a of [-1, 1]) {
    for (const b of [-1, 1]) {
      for (let row = 0; row < 16; row++) {
        predictors.push(a, a + b * 2 ** -16, 0);
        response.push(row < (b === 1 ? 12 : 4) ? 1 : 0);
      }
    }
  }
  for (const side of [1, -1]) {
    for (let row = 0; row < 100; row++) {
      predictors.push(0, 0, side);
      response.push(row < (side === 1 ? 99 : 1) ? 1 : 0);
    }
  }
  return {predictors: Float32Array.from(predictors), response: Float32Array.from(response)};
}

// The model is saturated over its six cells: intercept 0, coefficient 1 = -coefficient 2 =
// ndtri(.75) * 2^16 = 44203.36026885041, and coefficient 3 = scipy.special.ndtri(.99) =
// 2.3263478740408408 (rows with x3 = +-1 have probabilities .99 and .01).
export const PROBIT_LARGE_COEFFICIENT_SLOPE = 2.3263478740408408;
