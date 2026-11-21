// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Directly rounds a signed 96-bit fixed-point total divided by an exact uint32 count.
 * Requires FIXED_POINT_WGSL and getNaN. Integer long division avoids both rounded counts and
 * reciprocal-multiply overflow near the largest finite float32. No floating division is used.
 * @internal
 */
export const FINITE_MEAN_WGSL = /* wgsl */ `
const MEAN_DIVISION_WORD_BITS: u32 = 32u;
const MEAN_DIVISION_TOTAL_BITS: i32 = 96i;
fn getMeanLeading(value: vec3<u32>) -> u32 {
  if (value.z != 0u) { return 64u + firstLeadingBit(value.z); }
  if (value.y != 0u) { return 32u + firstLeadingBit(value.y); }
  return firstLeadingBit(value.x);
}
fn shiftMeanLeft(value: vec3<u32>, shift: u32) -> vec3<u32> {
  var source = array<u32, 3>(value.x, value.y, value.z);
  var result = array<u32, 3>(0u, 0u, 0u);
  let words = shift / MEAN_DIVISION_WORD_BITS;
  let bits = shift % MEAN_DIVISION_WORD_BITS;
  for (var index = 0u; index < 3u; index++) {
    let destination = index + words;
    if (destination < 3u) {
      result[destination] = result[destination] | (source[index] << bits);
      if (bits != 0u && destination + 1u < 3u) {
        result[destination + 1u] = result[destination + 1u] | (source[index] >> (32u - bits));
      }
    }
  }
  return vec3<u32>(result[0], result[1], result[2]);
}
fn getFiniteFixedPointMean(total: vec3<u32>, unitExponent: i32, count: u32) -> f32 {
  if (count == 0u) { return getNaN(); }
  let negative = (total.z >> 31u) != 0u;
  var magnitude = select(total, negateFixedPoint(total), negative);
  if (all(magnitude == vec3<u32>(0u))) { return 0.0; }
  // If cancellation leaves a small integer, retain enough fractional quotient bits too.
  let extra = u32(max(0i, i32(firstLeadingBit(count)) + 24i - i32(getMeanLeading(magnitude))));
  magnitude = shiftMeanLeft(magnitude, extra);
  var quotient = array<u32, 3>(0u, 0u, 0u);
  var remainder = 0u;
  for (var bit = MEAN_DIVISION_TOTAL_BITS - 1i; bit >= 0i; bit--) {
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | getFixedPointBit(magnitude, u32(bit));
    if (carry != 0u || remainder >= count) {
      remainder = remainder - count;
      let word = u32(bit) / 32u;
      quotient[word] = quotient[word] | (1u << (u32(bit) % 32u));
    }
  }
  let words = vec3<u32>(quotient[0], quotient[1], quotient[2]);
  // Round once at the final binary32 spacing, including subnormal spacing 2^-149.
  let quotientExponent = unitExponent - i32(extra);
  let shift = u32(max(max(0i, i32(getMeanLeading(words)) - 23i), -149i - quotientExponent));
  if (shift >= 96u) { return bitcast<f32>(select(0u, 0x80000000u, negative)); }
  var significand = shiftFixedPointRight(words, shift);
  if (shift > 0u) {
    let roundBit = getFixedPointBit(words, shift - 1u);
    let sticky = remainder != 0u || hasFixedPointBitsBelow(words, shift - 1u);
    if (roundBit != 0u && (sticky || (significand & 1u) != 0u)) {
      significand = significand + 1u;
    }
  } else {
    // Quotient has exactly 24 bits. Compare the rational remainder to one half without overflow.
    let complement = count - remainder;
    if (remainder > complement || (remainder == complement && (significand & 1u) != 0u)) {
      significand = significand + 1u;
    }
  }
  let resultExponent = quotientExponent + i32(shift);
  if (resultExponent == -149i && significand <= 0x800000u) {
    return bitcast<f32>(significand | select(0u, 0x80000000u, negative));
  }
  if (significand == 0u) { return bitcast<f32>(select(0u, 0x80000000u, negative)); }
  let leading = i32(firstLeadingBit(significand));
  let value = ldexp(ldexp(f32(significand), -leading), unitExponent - i32(extra) + i32(shift) + leading);
  return select(value, -value, negative);
}
`;
