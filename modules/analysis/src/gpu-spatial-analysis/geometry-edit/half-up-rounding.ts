// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Fractional-remainder half-up rounding, preserving integral float32 values. @internal */
export const HALF_UP_ROUNDING_WGSL = /* wgsl */ `
fn roundHalfUp(value: f32) -> f32 {
  let lower = floor(value);
  return lower + step(0.5, value - lower);
}
`;

/**
 * Certifies a grid cell against original f32 operands, independent of reciprocal division.
 * The quotient supplies only a nearby candidate. Integer dyadic sums compare the original
 * coordinate against each half-cell boundary, preserving ties even at large integral quotients.
 * Ten 32-bit words cover f32 significands times the bounded 26-bit boundary coefficient.
 * @internal
 */
export const GRID_HALF_UP_ROUNDING_WGSL = /* wgsl */ `
${HALF_UP_ROUNDING_WGSL}
fn gridRoundMultiply(first: u32, second: u32) -> vec2u {
  let low = (first & 65535u) * (second & 65535u);
  let middleFirst = (first >> 16u) * (second & 65535u) + (low >> 16u);
  let middleSecond = (first & 65535u) * (second >> 16u) + (middleFirst & 65535u);
  return vec2u((middleSecond << 16u) | (low & 65535u),
    (first >> 16u) * (second >> 16u) + (middleFirst >> 16u) + (middleSecond >> 16u));
}
fn gridRoundAdd(words: ptr<function, array<u32, 10>>, index: u32, value: u32) {
  var carry = value;
  for (var slot = index; slot < 10u && carry != 0u; slot++) {
    let previous = (*words)[slot];
    let sum = previous + carry;
    (*words)[slot] = sum;
    carry = select(0u, 1u, sum < previous);
  }
}
fn gridRoundTerm(positive: ptr<function, array<u32, 10>>, negative: ptr<function, array<u32, 10>>, value: f32, multiplier: i32) {
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 8388607u) | select(8388608u, 0u, exponent == 0u);
  let product = gridRoundMultiply(significand, u32(abs(multiplier)));
  // All terms are integer multiples of 2^-149, including subnormal coordinates/spacing.
  let shift = select(exponent - 1u, 0u, exponent == 0u);
  let word = shift / 32u;
  let remainder = shift % 32u;
  var parts = vec3u(product, 0u);
  if (remainder != 0u) {
    parts = vec3u(product.x << remainder,
      (product.y << remainder) | (product.x >> (32u - remainder)),
      product.y >> (32u - remainder));
  }
  let subtract = ((bits & 0x80000000u) != 0u) != (multiplier < 0);
  for (var part = 0u; part < 3u; part++) {
    if (subtract) { gridRoundAdd(negative, word + part, parts[part]); }
    else { gridRoundAdd(positive, word + part, parts[part]); }
  }
}
fn gridRoundCompare(value: f32, origin: f32, spacing: f32, twiceBoundary: i32) -> i32 {
  var positive: array<u32, 10>;
  var negative: array<u32, 10>;
  // Sign of 2*(value-origin) - twiceBoundary*spacing, without rounded products/differences.
  gridRoundTerm(&positive, &negative, value, 2);
  gridRoundTerm(&positive, &negative, origin, -2);
  gridRoundTerm(&positive, &negative, spacing, -twiceBoundary);
  for (var word = 9; word >= 0; word--) {
    if (positive[u32(word)] > negative[u32(word)]) { return 1; }
    if (positive[u32(word)] < negative[u32(word)]) { return -1; }
  }
  return 0;
}
fn roundGridHalfUp(value: f32, origin: f32, spacing: f32) -> f32 {
  let steps = (value - origin) / spacing;
  let candidate = roundHalfUp(steps);
  // Beyond the documented exact cell range retain the old bounded/uncertain route.
  if ((bitcast<u32>(steps) & 0x7fffffffu) >= 0x4b800000u ||
      (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u ||
      (bitcast<u32>(origin) & 0x7f800000u) == 0x7f800000u) { return candidate; }
  var cell = i32(candidate);
  // A rounded difference and reciprocal product put the candidate within four cells for
  // |steps| < 2^24. Exact boundary tests decide half-up ties, including negative coordinates.
  for (var correction = 0u; correction < 4u; correction++) {
    if (gridRoundCompare(value, origin, spacing, 2 * cell - 1) < 0) { cell--; }
    else if (gridRoundCompare(value, origin, spacing, 2 * cell + 1) >= 0) { cell++; }
    else { break; }
  }
  return f32(cell);
}
`;
