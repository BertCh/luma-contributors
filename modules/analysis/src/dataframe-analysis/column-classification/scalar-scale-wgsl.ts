// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/**
 * @internal Functions for finite, prepared-float32 native-style scalar scales, with
 * `isFiniteValue` from `analysis-core` (do not include `IS_FINITE_VALUE_WGSL` again).
 */
export const SCALAR_SCALE_WGSL = /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
fn scalarTransform(value: f32, scale: u32) -> f32 {
  if (scale == 1u) {
    return select(sqrt(abs(value)), -sqrt(abs(value)), value < 0.0);
  }
  if (scale == 2u) {
    // For negative log domains d3 uses -log(-x); the sign cancels in normalization.
    return log(abs(value));
  }
  return value;
}
fn scalarInterpolate(first: f32, second: f32, position: f32) -> f32 {
  if (position == 0.0) { return first; }
  if (position == 1.0) { return second; }
  // Same convex interpolation convention as d3; unlike first+(second-first)*t this
  // remains finite for opposite large endpoints when the result is in range.
  return first * (1.0 - position) + second * position;
}
`;
