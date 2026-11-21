// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Exceptional weights arithmetic keeps large scales in integer exponents under Metal fast math. @internal */
export const SPATIAL_WEIGHTS_TRANSFORM_ARITHMETIC_WGSL = /* wgsl */ `
${SCALED_QUOTIENT_WGSL}
struct SpatialWeightsDoubleScale {
  value: f32,
  useScaledQuotient: u32,
}
fn getSpatialWeightsDoubleScale(total: f32, desiredTotal: f32) -> SpatialWeightsDoubleScale {
  if (!(total > 0.0) || !isFiniteFloat(total)) {
    return SpatialWeightsDoubleScale(0.0, 0u);
  }
  let denominator = frexp(total);
  let numerator = frexp(desiredTotal);
  let exponent = numerator.exp - denominator.exp;
  // Decide before forming a quotient that could overflow. At an exponent gap
  // of 128 the quotient fits exactly when numerator.fract < denominator.fract.
  let useScaledQuotient = exponent > 128i ||
    (exponent == 128i && numerator.fract >= denominator.fract);
  var scale = 0.0;
  if (!useScaledQuotient) { scale = desiredTotal / total; }
  return SpatialWeightsDoubleScale(scale, u32(useScaledQuotient));
}
fn halveSpatialWeight(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  // Exact halving of a positive normal word. Integer exponent subtraction
  // cannot be factored back into the overflowing original float sum.
  if (bits >= 0x01000000u && bits < 0x7f800000u) {
    return bitcast<f32>(bits - 0x00800000u);
  }
  return 0.5 * value;
}
fn averageSpatialWeights(forward: f32, reverse: f32) -> f32 {
  // Any overflowing finite nonnegative sum has an operand at least 2^127.
  // Inspect source words before adding; overflow is not a portable branch predicate.
  if (max(bitcast<u32>(forward), bitcast<u32>(reverse)) >= 0x7f000000u) {
    return halveSpatialWeight(forward) + halveSpatialWeight(reverse);
  }
  return 0.5 * (forward + reverse);
}
`;
