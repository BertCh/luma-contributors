// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Complementary error function, Numerical Recipes Chebyshev fit (fractional error below 1.2e-7).
 * The TypeScript reference of the WGSL `erfcValue` in `utils/wgsl-common.ts` (`ERFC_VALUE_WGSL`).
 *
 * @internal
 */
export function getComplementaryErrorFunction(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const polynomial = [
    0.17087277, -0.82215223, 1.48851587, -1.13520398, 0.27886807, -0.18628806, 0.09678418,
    0.37409196, 1.00002368, -1.26551223
  ].reduce((accumulator, coefficient) => coefficient + t * accumulator, 0);
  const r = t * Math.exp(-z * z + polynomial);
  return x >= 0 ? r : 2 - r;
}
