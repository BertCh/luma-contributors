// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ERFC_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Float32 counterpart of shared getNormalIntervalMass, declared once for both density axes. */
export const DENSITY_NORMAL_INTERVAL_WGSL = /* wgsl */ `
${ERFC_VALUE_WGSL}
fn getDensityCenteredNormalMass(value: f32) -> f32 {
  if (value > 1.0) { return 0.5 - 0.5 * erfcValue(value * 0.7071067811865476); }
  var term = value;
  var total = term;
  for (var order = 1u; order <= 12u; order++) {
    term *= (-value * value) / (2.0 * f32(order));
    total += term / f32(2u * order + 1u);
  }
  return total * 0.3989422804014327;
}
fn getDensityNormalSurvival(value: f32) -> f32 {
  if (value <= 1.0) { return 0.5 - getDensityCenteredNormalMass(value); }
  return 0.5 * erfcValue(value * 0.7071067811865476);
}
fn getDensityNormalIntervalMassWithWidth(low: f32, high: f32, width: f32) -> f32 {
  if (width == 0.0) { return 0.0; }
  if (width * max(1.0, max(abs(low), abs(high))) <= 0.25) {
    let midpoint = low + 0.5 * width;
    let halfWidth = 0.5 * width;
    let nodes = array<f32, 4>(0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565);
    let weights = array<f32, 4>(0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363);
    var total = 0.0;
    for (var index = 0u; index < 4u; index++) {
      let offset = halfWidth * nodes[index];
      let lower = midpoint - offset;
      let upper = midpoint + offset;
      total += weights[index] * (exp(-0.5 * lower * lower) + exp(-0.5 * upper * upper));
    }
    return halfWidth * total * 0.3989422804014327;
  }
  if (low >= 0.0) { return getDensityNormalSurvival(low) - getDensityNormalSurvival(high); }
  if (high <= 0.0) { return getDensityNormalSurvival(-high) - getDensityNormalSurvival(-low); }
  return getDensityCenteredNormalMass(-low) + getDensityCenteredNormalMass(high);
}
fn getDensityNormalIntervalMass(low: f32, high: f32) -> f32 {
  return getDensityNormalIntervalMassWithWidth(low, high, high - low);
}
`;
