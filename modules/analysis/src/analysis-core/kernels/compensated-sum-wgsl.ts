// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ADDITION_ROUNDING_ERROR_WGSL} from './addition-rounding-error-wgsl';

/**
 * Scalar compensated accumulation in `(sum, correction)` order. Includes the integer rounding
 * error helper; do not also include ADDITION_ROUNDING_ERROR_WGSL in the same module.
 *
 * `addCompensatedValue` returns the next pair; `addCompensated` updates a function-local pair;
 * `getCompensated` combines it. The rounded sum is computed first, its integer-recovered error
 * is added to the old correction second, and only then is the pair replaced. This preserves the
 * evaluation order of the geometry measures, moments and line-measure accumulators.
 *
 * The finite/subnormal behavior is exactly that of getAdditionRoundingError. This is for bounded
 * serial walks; data-sized reductions use the parallel/fixed-point reduction helpers instead.
 * @internal
 */
export const COMPENSATED_SUM_WGSL = /* wgsl */ `${ADDITION_ROUNDING_ERROR_WGSL}
fn addCompensatedValue(accumulator: vec2<f32>, value: f32) -> vec2<f32> {
  let previous = accumulator.x;
  let total = previous + value;
  let correction = accumulator.y + getAdditionRoundingError(previous, value, total);
  return vec2<f32>(total, correction);
}

fn addCompensated(accumulator: ptr<function, vec2<f32>>, value: f32) {
  *accumulator = addCompensatedValue(*accumulator, value);
}

fn getCompensated(accumulator: vec2<f32>) -> f32 {
  return accumulator.x + accumulator.y;
}
`;
