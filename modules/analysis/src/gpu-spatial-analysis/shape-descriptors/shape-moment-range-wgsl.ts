// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Central polygon edge integrals of x², y² and xy. Normalize both endpoints by one power of two
 * before their fourth-degree products, then restore each final integral's exponent. Dividing the
 * unscaled cross by 12/24 first is insufficient: Metal may reassociate it after the product and
 * overflow an intermediate even when the integral is finite. Mantissa/exponent decomposition
 * prevents that rearrangement. Inputs are finite coordinates relative to the feature centroid.
 * @internal
 */
export const SHAPE_MOMENT_RANGE_WGSL = /* wgsl */ `
// A normalized f32 fraction has magnitude below one; its largest finite exponent is 128.
const SHAPE_MOMENT_MAXIMUM_EXPONENT: i32 = 128;
fn restoreShapeMoment(value: f32, power: i32) -> f32 {
  if (value == 0.0) { return value; }
  let parts = frexp(value);
  let exponent = parts.exp + power;
  if (exponent > SHAPE_MOMENT_MAXIMUM_EXPONENT) {
    return bitcast<f32>(0x7f800000u | (bitcast<u32>(value) & 0x80000000u));
  }
  return ldexp(parts.fract, exponent);
}
fn getShapeMomentTerms(previous: vec2<f32>, current: vec2<f32>) -> vec3<f32> {
  let magnitude = max(max(abs(previous.x), abs(previous.y)), max(abs(current.x), abs(current.y)));
  if (magnitude == 0.0) { return vec3<f32>(0.0); }
  let exponent = frexp(magnitude).exp;
  let first = vec2<f32>(ldexp(previous.x, -exponent), ldexp(previous.y, -exponent));
  let second = vec2<f32>(ldexp(current.x, -exponent), ldexp(current.y, -exponent));
  let cross = first.x * second.y - second.x * first.y;
  let xx = (cross / 12.0) * (first.x * first.x + first.x * second.x + second.x * second.x);
  let yy = (cross / 12.0) * (first.y * first.y + first.y * second.y + second.y * second.y);
  let xy = (cross / 24.0) * (first.x * second.y + 2.0 * first.x * first.y + 2.0 * second.x * second.y + second.x * first.y);
  return vec3<f32>(restoreShapeMoment(xx, 4 * exponent),
    restoreShapeMoment(yy, 4 * exponent), restoreShapeMoment(xy, 4 * exponent));
}
`;
