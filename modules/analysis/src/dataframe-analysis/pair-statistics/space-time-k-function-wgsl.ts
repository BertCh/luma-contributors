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
 * The window is closed: a lag that exactly reaches a limit keeps weight 1. This is the rule of
 * stpp 2.0-8 (`stikfunction.f`: `vij = 1` when `bsup .le. bsupt` and `binf .ge. binft`), the
 * reference of `GPUSpaceTimeKFunction`. splancs differs, see
 * {@link SPACE_TIME_K_SPLANCS_TIE_WGSL}.
 * Requires the constant `WEIGHT_CAP`.
 * @internal
 */
export const SPACE_TIME_K_EDGE_WEIGHT_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;

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

/**
 * Which package's rule decides the two exact ties of the space-time K estimators. They matter for
 * times recorded as whole days and a whole-day `maximumTimeLag`.
 * - `'stpp'` (stpp 2.0-8 `stikfunction.f`, the rule of `GPUSpaceTimeKFunction`): a lag that
 *   exactly reaches a limit of the time window has temporal weight 1, and a pair at exactly
 *   `maximumDistance` or `maximumTimeLag` is counted in the last class.
 * - `'splancs'` (splancs 2.01-45 `stkhat.f`, the rule of `GPUSpaceTimeDiggleTest`): that lag has
 *   weight 2 (`(ti - tlow) .le. tj`), and a pair at exactly `maximumDistance` or
 *   `maximumTimeLag` is dropped (`sj .ge. smax`, `tj .ge. tmax`).
 * Interior class breaks are closed above in both. The `'stpp'` path finds the class from the
 * scaled value `value / maximum * count`, whose float32 rounding can put a value that lies
 * exactly on a break one class up (in IEEE float32, `7 / 13 * 13` is above 7); the `'splancs'`
 * path decides breaks with products, which whole numbers pass exactly.
 * @internal
 */
export type SpaceTimeKTieRule = 'stpp' | 'splancs';

/**
 * WGSL of the splancs rules at exact ties.
 * - `getSpaceTimeSplancsTemporalWeight(lag, radius)`: 2 as soon as the lag reaches a limit of the
 *   time window (`radius <= lag`), 1 otherwise. It differs from `getSpaceTimeTemporalWeight` only
 *   when `lag == radius`.
 * - `getSpaceTimeSplancsClass(value, maximum, count)`: the class of a distance or lag on `count`
 *   equal steps up to `maximum`, as `iplace.f` finds it (the first break at or above the value),
 *   or `count` when the pair is dropped (`value >= maximum`, or not a number). The break test is
 *   `value * count <= maximum * (k + 1)`, two float32 products and no division, so a whole-number
 *   lag on a whole-number break stays in the class the break closes on every adapter.
 * @internal
 */
export const SPACE_TIME_K_SPLANCS_TIE_WGSL = /* wgsl */ `
fn getSpaceTimeSplancsTemporalWeight(lag: f32, radius: f32) -> f32 {
  return select(1.0, 2.0, radius <= lag);
}

fn getSpaceTimeSplancsClass(value: f32, maximum: f32, count: u32) -> u32 {
  if (!(value < maximum)) {
    return count;
  }
  // A first guess from the scaled value, then one step either way by the product test.
  var classIndex = min(u32(max(ceil(value / maximum * f32(count)) - 1.0, 0.0)), count - 1u);
  let scaled = value * f32(count);
  if (classIndex > 0u && scaled <= maximum * f32(classIndex)) {
    classIndex = classIndex - 1u;
  } else if (classIndex + 1u < count && scaled > maximum * f32(classIndex + 1u)) {
    classIndex = classIndex + 1u;
  }
  return classIndex;
}
`;
