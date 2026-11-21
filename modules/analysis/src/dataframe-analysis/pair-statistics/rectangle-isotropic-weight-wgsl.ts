// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** @internal Rectangle arc fractions shared by Ripley and space-time K. Needs PI, HALF_PI, WEIGHT_CAP. */
export const RECTANGLE_ISOTROPIC_WEIGHT_WGSL = /* wgsl */ `
// Angle of the arc of a circle of radius d (centered at distance h from a side line) outside it.
fn getSideAngle(h: f32, d: f32) -> f32 {
  if (d > h) {
    // A point on the side line: a quarter turn. h can be -0.0 (a coordinate of -0.0 against a
    // bound of 0), and atan2(y, -0.0) is -pi / 2 on some adapters.
    if (h == 0.0) {
      return HALF_PI;
    }
    return atan2(sqrt((d - h) * (d + h)), h);
  }
  return 0.0;
}

fn getArcOverlap(first: f32, second: f32) -> f32 {
  return max(0.0, first + second - HALF_PI);
}

// Ripley (1977) weight 1 / (fraction of the circle inside the rectangle), capped.
fn getIsotropicWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  let a = getSideAngle(left, d);
  let b = getSideAngle(bottom, d);
  let c = getSideAngle(right, d);
  let e = getSideAngle(top, d);
  let overlap = getArcOverlap(a, b) + getArcOverlap(b, c) + getArcOverlap(c, e) + getArcOverlap(e, a);
  let inside = 1.0 - (a + b + c + e) / PI + overlap / (2.0 * PI);
  return min(1.0 / max(inside, 1.0 / WEIGHT_CAP), WEIGHT_CAP);
}`;
