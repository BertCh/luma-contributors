// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_UPPER_INCOMPLETE_GAMMA_WGSL,
  GPU_CENTERED_INCOMPLETE_GAMMA_WGSL
} from '../../analysis-core/kernels/incomplete-gamma-wgsl';

/**
 * Convergence-checked gamma tails `getConvergedRegularizedGammaTails(a,x) -> vec2(P,Q)`.
 * Uses the centered prefactor and bounded series (2048 terms) / continued fraction (499 terms)
 * of the shared exact gamma route. Exhaustion returns NaN in both tails, so evaluation publishes
 * unavailable status rather than a partial CDF. The directly evaluated small tail keeps relative
 * precision; its complement can lose relative precision near zero. No large-shape normal shortcut.
 * @internal
 */
export const RASTER_REGULARIZED_GAMMA_TAILS_WGSL = `${GET_NAN_WGSL}
${IS_NAN_BITS_WGSL}
${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}
${GPU_CENTERED_INCOMPLETE_GAMMA_WGSL}
fn getConvergedRegularizedGammaTails(a: f32, x: f32) -> vec2<f32> {
  return getCenteredGammaTails(a, x);
}
`;

/** Original-value atom predicates use bit identity so FTZ cannot move a negative subnormal across the jump. @internal */
export const RASTER_CLIMATE_ZERO_ATOM_WGSL = `
fn isRasterClimateOriginalZero(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) == 0u;
}
fn isRasterClimateOriginalNegative(value: f32) -> bool {
  let bits = bitcast<u32>(value);
  return (bits & 0x80000000u) != 0u && (bits & 0x7fffffffu) != 0u;
}
`;

/** Requires getScaledQuotient; subtract bounded significands before restoring the common exponent. @internal */
export const RASTER_CLIMATE_SHIFTED_RATIO_WGSL = `
fn getRasterClimateShiftedRatio(value: f32, location: f32, scale: f32) -> f32 {
  let valueParts = frexp(value);
  let locationParts = frexp(location);
  let exponent = max(valueParts.exp, locationParts.exp);
  let difference = ldexp(valueParts.fract, valueParts.exp - exponent) -
    ldexp(locationParts.fract, locationParts.exp - exponent);
  return getScaledQuotient(difference, scale, 1.0, exponent);
}
`;
