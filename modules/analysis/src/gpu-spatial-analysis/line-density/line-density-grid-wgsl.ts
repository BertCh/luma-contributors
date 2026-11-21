// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Certify stored-coordinate cell ownership before any raster permutation. @internal */
export const LINE_DENSITY_GRID_CLASSIFICATION_WGSL = /* wgsl */ `
// Sign of value - origin - boundary * step, from the original float32 bits. Cell indices
// are below 2^24. Ten uint32 limbs cover all finite float32 values times those indices
// in units of 2^-149 (at most 302 bits), including subnormals on flush-to-zero devices.
fn compareGridBoundary(value: f32, origin: f32, step: f32, boundary: u32) -> i32 {
  var operands = array<u32, 3>(bitcast<u32>(value), bitcast<u32>(origin) ^ 0x80000000u, bitcast<u32>(step) ^ 0x80000000u);
  var positive: array<u32, 10>;
  var negative: array<u32, 10>;
  for (var term = 0u; term < 3u; term++) {
    let bits = operands[term];
    let exponentBits = (bits >> 23u) & 255u;
    let significand = select((bits & 0x7fffffu) | 0x800000u, bits & 0x7fffffu, exponentBits == 0u);
    let multiplier = select(1u, boundary, term == 2u);
    let lowProduct = (significand & 65535u) * (multiplier & 65535u);
    let middle = (significand >> 16u) * (multiplier & 65535u) + (significand & 65535u) * (multiplier >> 16u);
    let low = lowProduct + (middle << 16u);
    let high = (significand >> 16u) * (multiplier >> 16u) + (middle >> 16u) + u32(low < lowProduct);
    let shift = select(exponentBits - 1u, 0u, exponentBits == 0u);
    let firstWord = shift / 32u;
    let bit = shift % 32u;
    var words = array<u32, 3>(low << bit, high << bit, 0u);
    if (bit > 0u) {
      words[1] |= low >> (32u - bit);
      words[2] = high >> (32u - bit);
    }
    var carry = 0u;
    for (var slot = firstWord; slot < 10u; slot++) {
      let relative = slot - firstWord;
      var addend = 0u;
      if (relative < 3u) { addend = words[relative]; }
      else if (carry == 0u) { break; }
      var previous = positive[slot];
      if ((bits >> 31u) != 0u) { previous = negative[slot]; }
      let partial = previous + addend;
      let total = partial + carry;
      carry = u32(partial < previous || total < partial);
      if ((bits >> 31u) != 0u) { negative[slot] = total; }
      else { positive[slot] = total; }
    }
  }
  for (var slot = 10u; slot > 0u; slot--) {
    if (positive[slot - 1u] != negative[slot - 1u]) {
      return select(-1, 1, positive[slot - 1u] > negative[slot - 1u]);
    }
  }
  return 0;
}

fn getStoredCoordinateCell(value: f32, origin: f32, step: f32, count: i32, approximate: f32) -> i32 {
  let candidate = clamp(i32(floor(approximate)), 0, count - 1);
  // The ordinary path needs two certificates. If rounding crossed either boundary, find
  // the exact floor by an integer binary search (at most 24 comparisons for this route).
  if (compareGridBoundary(value, origin, step, u32(candidate)) >= 0 &&
      compareGridBoundary(value, origin, step, u32(candidate + 1)) < 0) { return candidate; }
  var lower = 0;
  var upper = count;
  loop {
    if (upper - lower <= 1) { break; }
    let middle = lower + (upper - lower) / 2;
    if (compareGridBoundary(value, origin, step, u32(middle)) >= 0) { lower = middle; }
    else { upper = middle; }
  }
  return lower;
}
`;
