// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Complementary error function, Numerical Recipes Chebyshev fit (fractional error below 1.2e-7).
 * The TypeScript reference of the WGSL `erfcValue` in `analysis-core/kernels/wgsl-common.ts` (`ERFC_VALUE_WGSL`).
 *
 * @internal
 */
export function getComplementaryErrorFunction(value: number): number {
  const magnitude = Math.abs(value);
  const transform = 1 / (1 + 0.5 * magnitude);
  const polynomial = [
    0.17087277, -0.82215223, 1.48851587, -1.13520398, 0.27886807, -0.18628806, 0.09678418,
    0.37409196, 1.00002368, -1.26551223
  ].reduce((accumulator, coefficient) => coefficient + transform * accumulator, 0);
  const result = transform * Math.exp(-magnitude * magnitude + polynomial);
  return value >= 0 ? result : 2 - result;
}
