// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Exact comparison of positive float32 coordinates times <=65536 quadrat indices. @internal */
export const QUADRAT_INDEX_WGSL = /* wgsl */ `
// QUADRAT_PRODUCT_HELPERS
// Normalize an exact <=40-bit significand product into (high, low, biased top exponent).
fn getQuadratProduct(value: f32, multiplier: u32) -> vec3u {
  let bits = bitcast<u32>(value) & 0x7fffffffu;
  if (bits == 0u || multiplier == 0u) { return vec3u(0u); }
  let exponentBits = bits >> 23u;
  let significand = select((bits & 0x7fffffu) | 0x800000u, bits & 0x7fffffu, exponentBits == 0u);
  let exponent = select(i32(exponentBits) - 150, -149, exponentBits == 0u);
  let lowProduct = (significand & 0xffffu) * multiplier;
  let highProduct = (significand >> 16u) * multiplier;
  let low = lowProduct + (highProduct << 16u);
  let high = (highProduct >> 16u) + select(0u, 1u, low < lowProduct);
  if (high == 0u) {
    let highest = 31u - countLeadingZeros(low);
    if (highest <= 23u) { return vec3u(low << (23u - highest), 0u, u32(exponent + i32(highest) + 512)); }
    let shift = highest - 23u;
    return vec3u(low >> shift, low << (32u - shift), u32(exponent + i32(highest) + 512));
  }
  let highest = 31u - countLeadingZeros(high);
  let shift = highest + 9u;
  return vec3u((high << (32u - shift)) | (low >> shift), low << (32u - shift),
    u32(exponent + i32(highest) + 544));
}

// Exact product comparisons avoid quotient reciprocals, including subnormal flushing.
fn isQuadratProductBelow(coordinate: vec3u, edge: vec3u) -> bool {
  if (coordinate.x == 0u) { return edge.x > 0u; }
  if (edge.x == 0u) { return false; }
  if (coordinate.z != edge.z) { return coordinate.z < edge.z; }
  if (coordinate.x != edge.x) { return coordinate.x < edge.x; }
  return coordinate.y < edge.y;
}
fn getQuadratIndex(offset: f32, extent: f32, count: u32) -> u32 {
  let coordinate = getQuadratProduct(offset, count);
  var lower = 0u;
  var upper = count;
  // count <= GPU_QUADRAT_MAXIMUM_COUNT = 65536, at most 17 comparisons.
  loop {
    if (upper - lower <= 1u) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (isQuadratProductBelow(coordinate, getQuadratProduct(extent, middle))) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  return lower;
}`;
