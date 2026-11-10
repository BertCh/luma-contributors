// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL for the exact bucket division of word-mode temporal reduction. Needs `TIME_WORDS_WGSL`.
 *
 * `divideTimeDifference(difference, width)` divides an unsigned 64-bit difference by an unsigned
 * 64-bit width. `overflow` is set when the quotient does not fit 32 bits (only possible with a
 * width below 2^32); otherwise `quotient` is exact and `remainder` is the exact 64-bit offset
 * inside the bucket. The narrow path (width below 2^32) is the original word-by-word division.
 *
 * @internal
 */
export const TEMPORAL_REDUCTION_WORD_DIVISION_WGSL = /* wgsl */ `
struct TimeDivision {
  quotient: u32,
  overflow: bool,
  remainder: vec2<u32>
}

fn divideTimeDifference(difference: vec2<u32>, width: vec2<u32>) -> TimeDivision {
  var quotientLow = 0u;
  if (width.y == 0u) {
    // High word first, then the low word bit by bit.
    let quotientHigh = difference.y / width.x;
    var remainder = difference.y % width.x;
    for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
      let carry = remainder >> 31u;
      remainder = (remainder << 1u) | ((difference.x >> u32(bit)) & 1u);
      if (carry != 0u || remainder >= width.x) {
        remainder = remainder - width.x;
        quotientLow = quotientLow | (1u << u32(bit));
      }
    }
    return TimeDivision(quotientLow, quotientHigh != 0u, vec2<u32>(remainder, 0u));
  }
  // The width is at least 2^32, so the quotient is below 2^32 and the high word of the
  // difference is already a remainder below the width. Shift in the low word bit by bit.
  var remainder = vec2<u32>(difference.y, 0u);
  for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
    let carry = remainder.y >> 31u;
    remainder = vec2<u32>(
      (remainder.x << 1u) | ((difference.x >> u32(bit)) & 1u),
      (remainder.y << 1u) | (remainder.x >> 31u));
    let isAtLeastWidth = remainder.y > width.y || (remainder.y == width.y && remainder.x >= width.x);
    if (carry != 0u || isAtLeastWidth) {
      remainder = timeWordsSubtract(remainder, width);
      quotientLow = quotientLow | (1u << u32(bit));
    }
  }
  return TimeDivision(quotientLow, false, remainder);
}
`;
