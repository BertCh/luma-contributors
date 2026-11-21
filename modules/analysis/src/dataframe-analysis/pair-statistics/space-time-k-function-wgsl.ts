// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RECTANGLE_ISOTROPIC_WEIGHT_WGSL} from './rectangle-isotropic-weight-wgsl';
import {QUADRAT_INDEX_WGSL} from './quadrat-index-wgsl';

/**
 * Fixed-point scale of one pair weight in the space-time K accumulators, `2^16`: a weight is at
 * most `2 * GPU_SPACE_TIME_K_WEIGHT_CAP` (spatial cap times the temporal weight 2), so a term
 * stays below `2^24` units and is exact to half a unit. It is the scale of the homogeneous terms
 * and of the spatial marginal. The intensity-weighted terms of the inhomogeneous estimator use the
 * per-frame power of two `2^16 / m^2` (to a factor of 4), `m` the smallest included
 * `1 / lambda'`, so the smallest term keeps at least `2^16` units (rounding at most `2^-17` of a
 * K term) while the required scale exponent is in [-100,100]. Unsupported intensity scales
 * report loss and suppress estimates. PCF has an additional 2^24 scale, with half-unit absolute
 * rounding error; a positive term smaller than half a unit reports loss instead of disappearing.
 * Larger terms take the large channel of `accumulateScaled`.
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
 * `GPURipley` (the angle of each side's cut-off arc minus the overlap of adjacent arcs), shared
 * through `RECTANGLE_ISOTROPIC_WEIGHT_WGSL`. The temporal weight
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

${RECTANGLE_ISOTROPIC_WEIGHT_WGSL}

fn getSpaceTimeSpatialWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  return getIsotropicWeight(left, bottom, right, top, d);
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
 * Interior class breaks are closed above in both. Both paths compare exact float32/integer
 * products, so reciprocal lowering and reassociation cannot move a tie or drop a maximum.
 * @internal
 */
export type SpaceTimeKTieRule = 'stpp' | 'splancs';

/** @internal Exact right-closed class predicates shared by the pair pass and temporal marginal. */
export const SPACE_TIME_K_CLASS_WGSL = /* wgsl */ `
${QUADRAT_INDEX_WGSL}
fn isSpaceTimeKClassIncluded(value: f32, maximum: f32, count: u32, limit: u32) -> bool {
  return !isQuadratProductBelow(getQuadratProduct(maximum, limit), getQuadratProduct(value, count));
}
fn getSpaceTimeKClass(value: f32, maximum: f32, count: u32) -> u32 {
  var lower = 0u;
  var upper = count - 1u;
  // count <= GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT = 64, at most 6 comparisons.
  for (var step = 0u; step < 6u && lower < upper; step++) {
    let middle = lower + (upper - lower) / 2u;
    if (isSpaceTimeKClassIncluded(value, maximum, count, middle + 1u)) {
      upper = middle;
    } else {
      lower = middle + 1u;
    }
  }
  return lower;
}`;

/**
 * WGSL of the splancs rules at exact ties.
 * - `getSpaceTimeSplancsTemporalWeight(lag, radius)`: 2 as soon as the lag reaches a limit of the
 *   time window (`radius <= lag`), 1 otherwise. It differs from `getSpaceTimeTemporalWeight` only
 *   when `lag == radius`.
 * - `getSpaceTimeSplancsClass(value, maximum, count)`: the class of a distance or lag on `count`
 *   equal steps up to `maximum`, as `iplace.f` finds it (the first break at or above the value),
 *   or `count` when the pair is dropped (`value >= maximum`, or not a number). The break test is
 *   the exact significand-product comparison `value * count <= maximum * (k + 1)`, avoiding
 *   overflowing products and division, so a tie stays in the class it closes on every adapter.
 * @internal
 */
export const SPACE_TIME_K_SPLANCS_TIE_WGSL = /* wgsl */ `
${SPACE_TIME_K_CLASS_WGSL}
fn getSpaceTimeSplancsTemporalWeight(lag: f32, radius: f32) -> f32 {
  return select(1.0, 2.0, radius <= lag);
}

fn getSpaceTimeSplancsClass(value: f32, maximum: f32, count: u32) -> u32 {
  if (!(value < maximum)) {
    return count;
  }
  return getSpaceTimeKClass(value, maximum, count);
}
`;
