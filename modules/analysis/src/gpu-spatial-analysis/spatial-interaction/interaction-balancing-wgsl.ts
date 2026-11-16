// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Multiply/divide positive masses without overflowing or underflowing an intermediate. */
export const INTERACTION_BALANCING_WGSL = /* wgsl */ `
${SCALED_QUOTIENT_WGSL}
fn getBalancedPrediction(prediction: f32, total: f32, sum: f32) -> f32 {
  if (!(sum > 0.0) || !(prediction > 0.0) || !(total > 0.0)) { return 0.0; }
  if (prediction == sum) { return total; }
  let predictionParts = frexp(prediction);
  let totalParts = frexp(total);
  let sumParts = frexp(sum);
  // Each fraction is in [0.5, 1), so this intermediate stays in [0.25, 2).
  let productParts = frexp(getScaledQuotient(predictionParts.fract * totalParts.fract, sumParts.fract, 1.0, 0));
  let exponent = predictionParts.exp + totalParts.exp - sumParts.exp + productParts.exp;
  // ldexp requires an exponent <= 128. Only an unrepresentable final result overflows.
  if (exponent > 128) { var overflowBits = 0x7f800000u; return bitcast<f32>(overflowBits); }
  if (exponent < -149) { return 0.0; }
  return ldexp(productParts.fract, exponent);
}`;
