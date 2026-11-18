// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Tiled panel profile log variance with its rho-independent floor removed. Adding the squared offset
 * to the floor first makes the float32 objective flat across rho intervals much wider than the
 * search bracket. The small-increment series retains that offset; its omitted term at 0.01 is
 * below 2.1e-11. Reported likelihoods still use the uncentered log variance.
 * Requires ROUNDED_DIFFERENCE_WGSL and IS_FINITE_BITS_WGSL.
 * @internal
 */
export const PANEL_LAG_PROFILE_WGSL = /* wgsl */ `
fn getPanelLagProfileLogVariance(rho: f32, direct: vec4<f32>, rows: f32, runtimeZero: u32) -> f32 {
  let offset = getRoundedDifference(rho, direct.z, runtimeZero) - direct.w;
  let increment = direct.y * offset * offset;
  let variance = (direct.x + increment) / rows;
  if (direct.x > 0.0) {
    let relativeIncrement = increment / direct.x;
    if (isFiniteBits(relativeIncrement)) {
      if (relativeIncrement < 0.01) {
        return relativeIncrement * (1.0 + relativeIncrement * (-0.5 + relativeIncrement * (1.0 / 3.0 - relativeIncrement * 0.25)));
      }
      return log(1.0 + relativeIncrement);
    }
    return log(variance) - (log(direct.x) - log(rows));
  }
  return log(variance);
}
`;
