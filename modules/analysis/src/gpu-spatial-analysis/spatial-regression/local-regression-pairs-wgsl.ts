// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

/** Two-word residual/component arithmetic; integer rounding helpers resist reassociation. @internal */
export const LOCAL_REGRESSION_PAIRS_WGSL = /* wgsl */ `
${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
${getPairArithmeticWGSL()}`;

/** For shaders that already declare the two integer rounding-error helpers. @internal */
export function getPairArithmeticWGSL(): string {
  return /* wgsl */ `
fn addRegressionPairs(left: vec2f, right: vec2f) -> vec2f {
  let high = left.x + right.x;
  let low = getAdditionRoundingError(left.x, right.x, high) + left.y + right.y;
  let combined = high + low;
  return vec2f(combined, getAdditionRoundingError(high, low, combined));
}
fn multiplyRegressionPair(value: vec2f, factor: f32) -> vec2f {
  let high = value.x * factor;
  let low = getMultiplicationRoundingError(value.x, factor, high) + value.y * factor;
  return addRegressionPairs(vec2f(high, 0.0), vec2f(low, 0.0));
}
fn multiplyRegressionPairs(first: vec2f, second: vec2f) -> vec2f {
  return addRegressionPairs(multiplyRegressionPair(first, second.x), multiplyRegressionPair(first, second.y));
}
fn divideRegressionPair(value: vec2f, denominator: vec2f) -> vec2f {
  let high = value.x / denominator.x;
  let remainder = addRegressionPairs(value, -multiplyRegressionPair(denominator, high));
  return addRegressionPairs(vec2f(high, 0.0), vec2f((remainder.x + remainder.y) / denominator.x, 0.0));
}
fn getRegressionResponseOrigin(minimum: f32, maximum: f32) -> f32 {
  // Keep zero for mixed signs; otherwise use the endpoint closest to zero. A negligible
  // positive extreme must not move a small positive neighborhood's origin toward that extreme.
  if ((minimum > 0.0 && minimum >= maximum * 0.5) ||
      (maximum < 0.0 && minimum * 0.5 >= maximum)) {
    return minimum * 0.5 + maximum * 0.5;
  }
  return max(minimum, min(0.0, maximum));
}
`;
}
