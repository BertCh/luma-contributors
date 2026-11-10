// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Fixed-point scale of one pair weight in the space-time K accumulators, `2^16`: a weight is at
 * most `2 * GPU_SPACE_TIME_K_WEIGHT_CAP` (spatial cap times the temporal weight 2), so a term
 * stays below `2^24` units and is exact to half a unit; larger terms (a small intensity) take the
 * large channel of `accumulateScaled`.
 * @internal
 */
export const SPACE_TIME_K_WEIGHT_SCALE = 65536;

/**
 * Largest spatial isotropic edge-correction weight, as in `GPURipley` (the `maxweight = 100`
 * default of spatstat `edge.Ripley`): the circle fraction inside the window is floored at 1/100.
 */
export const GPU_SPACE_TIME_K_WEIGHT_CAP = 100;

/**
 * WGSL of the Ripley (1977) spatial isotropic weight for a rectangular window, and the temporal
 * counterpart on an interval. Declares `getSpaceTimeSpatialWeight(left, bottom, right, top, d)`
 * and `getSpaceTimeTemporalWeight(lag, radius)`. The spatial function is the arc formula of
 * `GPURipley` (the angle of each side's cut-off arc minus the overlap of adjacent arcs); it is
 * stated here again because `GPURipley` keeps it inside one kernel declaration. The temporal weight
 * is the one-dimensional analogue: the "circle" of radius `lag` about `t_i` is the pair
 * `t_i - lag`, `t_i + lag`; both lie in the window when `lag <= radius`, with `radius = min(t_i -
 * start, end - t_i)`, and only one does otherwise (the other end is `t_j`), so the weight is 1 or 2.
 * Requires the constant `WEIGHT_CAP`.
 * @internal
 */
export const SPACE_TIME_K_EDGE_WEIGHT_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;

// Angle of the arc of a circle of radius d (centered at distance h from a side line) outside it.
fn getSideAngle(h: f32, d: f32) -> f32 {
  if (d > h) {
    return atan2(sqrt((d - h) * (d + h)), h);
  }
  return 0.0;
}

fn getArcOverlap(first: f32, second: f32) -> f32 {
  return max(0.0, first + second - HALF_PI);
}

fn getSpaceTimeSpatialWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  let a = getSideAngle(left, d);
  let b = getSideAngle(bottom, d);
  let c = getSideAngle(right, d);
  let e = getSideAngle(top, d);
  let overlap = getArcOverlap(a, b) + getArcOverlap(b, c) + getArcOverlap(c, e) + getArcOverlap(e, a);
  let inside = 1.0 - (a + b + c + e) / PI + overlap / (2.0 * PI);
  return min(1.0 / max(inside, 1.0 / WEIGHT_CAP), WEIGHT_CAP);
}

fn getSpaceTimeTemporalWeight(lag: f32, radius: f32) -> f32 {
  return select(2.0, 1.0, lag <= radius);
}
`;
