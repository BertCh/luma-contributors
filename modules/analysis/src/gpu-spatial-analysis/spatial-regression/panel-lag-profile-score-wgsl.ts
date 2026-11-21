// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Derivative of concentrated negative log likelihood:
 * n*(rho-rho*)/[S/(e1'e1)+(rho-rho*)^2] + T*tr(W*(I-rho W)^-1).
 * Return [derivative, usable]. Dividing both residual moments by e1'e1 makes response
 * rescaling cancel before arithmetic; no difference of rounded logarithms is required.
 * Callers use its sign after eight coarse golden steps, keeping the same bracket updates
 * and explicit search budget. Requires ROUNDED_DIFFERENCE_WGSL and IS_FINITE_BITS_WGSL.
 * @internal
 */
export const PANEL_LAG_PROFILE_SCORE_WGSL = /* wgsl */ `
fn getPanelLagProfileScore(rho: f32, direct: vec4f, spatialTrace: vec2f,
  rows: f32, periods: f32, runtimeZero: u32) -> vec2f {
  if (spatialTrace.y == 0.0) { return vec2f(0.0); }
  let offset = getRoundedDifference(rho, direct.z, runtimeZero) - direct.w;
  var residualScore = 0.0;
  if (direct.y > 0.0) {
    let denominator = direct.x / direct.y + offset * offset;
    if (!(denominator > 0.0) || !isFiniteBits(denominator)) { return vec2f(0.0); }
    residualScore = rows * (offset / denominator);
  } else if (!(direct.x > 0.0)) { return vec2f(0.0); }
  let score = residualScore + periods * spatialTrace.x;
  if (!isFiniteBits(score)) { return vec2f(0.0); }
  return vec2f(score, 1.0);
}
`;
