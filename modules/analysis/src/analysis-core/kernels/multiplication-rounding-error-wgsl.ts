// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `getMultiplicationRoundingError(first, second, product) -> f32`: the exact rounding error
 * `first * second - product` of the f32 multiplication `product = first * second`, for finite
 * operands of either sign. It is the companion of `getAdditionRoundingError`
 * (`addition-rounding-error-wgsl.ts`): together they carry a residual
 * `y - sum(x * b)` as a float32 value plus a float32 correction, which is what a fit needs when
 * the products are thousands of times the residual.
 *
 * Why integers (contributor contract section 11). The float forms of this term (a Veltkamp
 * split, an `fma` residual) are identities a shader compiler may simplify or may not fuse. Here
 * the 48-bit product of the two 24-bit significands is formed in 12-bit limbs with `u32`
 * arithmetic, the significand of `product` is subtracted at its place, and one `ldexp` scales
 * the difference: integer operations give the compiler nothing to fold.
 *
 * What is exact. `product` must be the f32 the caller computed for `first * second`.
 * - A correctly rounded `product` (round to nearest): the difference is at most `2^24` units of
 *   the last place of the exact product, and the rounding error of a correctly rounded
 *   multiplication is itself an f32, so the conversion and the scaling are exact. The result is
 *   the exact error.
 * - Any other f32 within two units in the last place of the exact product (a multiplication
 *   that is not correctly rounded): the integer difference is still exact, but it can need 25
 *   significant bits, and the result is that difference rounded to float32 (a relative error
 *   of at most `2^-24`). Example: `first = second = 2 - 2^-23` with `product = 4 - 3 * 2^-22`,
 *   one step below the correctly rounded product; the distance is `2^-22 + 2^-46` and the
 *   result is `2^-22`.
 *
 * What is reported. Returns 0 when an operand is zero or not finite, when the product is zero,
 * subnormal or not finite, when an operand is subnormal, and when the binary exponents of the
 * two operands sum to less than -80 (the biased exponents to less than 174), which keeps the
 * `ldexp` exponent at -126 or above. In terms of the exact product: every product below
 * `2^-80` reports no error, every product of `2^-79` or more is reported, and a product in
 * `[2^-80, 2^-79)` is reported only when its operands' significands multiply to less than 2
 * (`(1.5 + 2^-23) 2^-41` times `(1.5 + 2^-23) 2^-40` is `1.125 * 2^-80` and reports 0, although
 * its error is `-4.93e-32`). An unreported error is at most `2^-104` in magnitude.
 *
 * Checked on the GPU against float64 in
 * `test/utils/multiplication-rounding-error-wgsl.spec.ts`.
 *
 * @internal
 */
export const MULTIPLICATION_ROUNDING_ERROR_WGSL = /* wgsl */ `
fn getMultiplicationRoundingError(first: f32, second: f32, product: f32) -> f32 {
  let firstBits = bitcast<u32>(first);
  let secondBits = bitcast<u32>(second);
  let productBits = bitcast<u32>(product);
  // Biased exponents: |value| = significand * 2^(exponent - 150) for a normal value.
  let firstExponent = (firstBits >> 23u) & 0xffu;
  let secondExponent = (secondBits >> 23u) & 0xffu;
  let productExponent = (productBits >> 23u) & 0xffu;
  if (firstExponent == 0u || secondExponent == 0u || productExponent == 0u ||
      firstExponent == 255u || secondExponent == 255u || productExponent == 255u) {
    return 0.0;
  }
  // The exact product is exactSignificand * 2^(firstExponent + secondExponent - 300), with a
  // significand of 47 or 48 bits. The error is scaled by one ldexp whose exponent is kept at
  // -126 or above, where ldexp is dependable: operands whose binary exponents sum to less than
  // -80 report no error (every product below 2^-80, and those below 2^-79 whose significands
  // multiply to 2 or more).
  let exponentSum = firstExponent + secondExponent;
  if (exponentSum < 174u) {
    return 0.0;
  }
  let firstSignificand = (firstBits & 0x7fffffu) | 0x800000u;
  let secondSignificand = (secondBits & 0x7fffffu) | 0x800000u;
  // 48-bit product of the two 24-bit significands, as two 24-bit limbs.
  let firstHigh = firstSignificand >> 12u;
  let firstLow = firstSignificand & 0xfffu;
  let secondHigh = secondSignificand >> 12u;
  let secondLow = secondSignificand & 0xfffu;
  let middle = firstHigh * secondLow + firstLow * secondHigh;
  let lowSum = firstLow * secondLow + ((middle & 0xfffu) << 12u);
  let exactLow = lowSum & 0xffffffu;
  let exactHigh = firstHigh * secondHigh + (middle >> 12u) + (lowSum >> 24u);
  // The significand of the rounded product sits 23, 24 or 25 bits above the last place of the
  // exact product (25 when the rounding carried into the next binade).
  let shift = productExponent + 150u - exponentSum;
  if (shift < 23u || shift > 25u) {
    return 0.0;
  }
  let productSignificand = (productBits & 0x7fffffu) | 0x800000u;
  var roundedHigh = productSignificand;
  var roundedLow = 0u;
  if (shift == 23u) {
    roundedHigh = productSignificand >> 1u;
    roundedLow = (productSignificand & 1u) << 23u;
  } else if (shift == 25u) {
    roundedHigh = productSignificand << 1u;
  }
  // Exact as an integer. Its conversion to f32 is exact for a correctly rounded product (at
  // most 2^24 units); for a product one or two steps off it can need 25 bits and is rounded.
  let errorUnits = (i32(exactHigh) - i32(roundedHigh)) * 16777216 + (i32(exactLow) - i32(roundedLow));
  let magnitude = ldexp(f32(errorUnits), i32(exponentSum) - 300);
  // The exact product has the sign of the operands' signs combined; so has its error's scale.
  return select(magnitude, -magnitude, ((firstBits ^ secondBits) & 0x80000000u) != 0u);
}
`;
