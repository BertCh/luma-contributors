// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Coordinate-only boundary refinement; requires published integer fp64 arithmetic. @internal */
export const POINT_CLOUD_NORM_COMPARISON_WGSL = `
// For normal values, subtraction/square/dot roundoff is bounded by gamma(5) per norm.
// 64 positive-float32 key steps conservatively cover both norms (including exponent transitions),
// the rounded cutoff, and fused/reassociated dot order. Subnormal keys always use the refinement
// because a device may flush subnormal products to zero.
fn pointCloudNormKeysSeparated(first: f32, second: f32) -> bool {
  let firstKey = bitcast<u32>(first) & 0x7fffffffu;
  let secondKey = bitcast<u32>(second) & 0x7fffffffu;
  if (min(firstKey, secondKey) < 0x00800000u) { return false; }
  return max(firstKey, secondKey) - min(firstKey, secondKey) > 64u;
}
fn pointCloudSquaredNorm(candidate: vec3f, query: vec3f, scale: f32) -> vec2f {
  let differenceX = mul_fp64(sub_fp64(vec2f(candidate.x, 0.0), vec2f(query.x, 0.0)), vec2f(scale, 0.0));
  let differenceY = mul_fp64(sub_fp64(vec2f(candidate.y, 0.0), vec2f(query.y, 0.0)), vec2f(scale, 0.0));
  let differenceZ = mul_fp64(sub_fp64(vec2f(candidate.z, 0.0), vec2f(query.z, 0.0)), vec2f(scale, 0.0));
  return sum_fp64(sum_fp64(mul_fp64(differenceX, differenceX), mul_fp64(differenceY, differenceY)), mul_fp64(differenceZ, differenceZ));
}
fn comparePointCloudNorms(first: vec3f, firstQuery: vec3f, second: vec3f, secondQuery: vec3f) -> i32 {
  let maximumDifference = max(abs(first - firstQuery), abs(second - secondQuery));
  let magnitude = max(max(maximumDifference.x, maximumDifference.y), maximumDifference.z);
  let exponent = max((bitcast<u32>(magnitude) >> 23u) & 255u, 1u);
  let scale = bitcast<f32>(max(254u - exponent, 1u) << 23u);
  let firstNorm = pointCloudSquaredNorm(first, firstQuery, scale);
  let secondNorm = pointCloudSquaredNorm(second, secondQuery, scale);
  if (firstNorm.x < secondNorm.x || (firstNorm.x == secondNorm.x && firstNorm.y < secondNorm.y)) { return -1; }
  if (firstNorm.x > secondNorm.x || (firstNorm.x == secondNorm.x && firstNorm.y > secondNorm.y)) { return 1; }
  return 0;
}
`;
