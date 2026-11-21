// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

/** Original-space intercept as a two-component sum. The centered correction stays separate
 * until response/predictor origins cancel. Integer rounding-error helpers resist fast math.
 * @internal
 */
export const LOCAL_REGRESSION_INTERCEPT_WGSL = /* wgsl */ `
${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
fn subtractLocalRegressionInterceptProduct(state: vec2f, slope: f32, origin: f32) -> vec2f {
  let product = slope * origin;
  let next = state.x - product;
  let correction = state.y + getAdditionRoundingError(state.x, -product, next)
    - getMultiplicationRoundingError(slope, origin, product);
  return vec2f(next, correction);
}
`;
