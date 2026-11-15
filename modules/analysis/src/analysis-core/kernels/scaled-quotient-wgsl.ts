// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Computes `value * 2^power / (first * second)` without forming the denominator product.
 * Only bounded mantissas are multiplied/divided; exponents are combined as integers. Thus normal
 * float32 results survive even if the denominator product would overflow or underflow, and
 * reassociation cannot move its scale back into a floating-point multiplication. Inputs must be
 * finite with nonzero denominators. Subnormal inputs/results can be flushed by the GPU.
 * @internal
 */
export const SCALED_QUOTIENT_WGSL = /* wgsl */ `
fn getScaledQuotient(value: f32, first: f32, second: f32, power: i32) -> f32 {
  if (value == 0.0) { return value; }
  let numerator = frexp(value);
  let firstDivisor = frexp(first);
  let secondDivisor = frexp(second);
  let quotient = frexp(numerator.fract / (firstDivisor.fract * secondDivisor.fract));
  return ldexp(quotient.fract,
    numerator.exp - firstDivisor.exp - secondDivisor.exp + power + quotient.exp);
}
`;
