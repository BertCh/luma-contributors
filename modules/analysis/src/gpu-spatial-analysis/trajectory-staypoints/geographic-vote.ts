// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** @internal Representative used only for geographic distinct votes and center offsets. */
export function getTrajectoryVotePosition(
  longitude: number,
  latitude: number,
  geographic: boolean
): [number, number] {
  if (!geographic) return [longitude, latitude];
  // Reduce before testing endpoints: adding 180 can lose the last bit near the seam in f32.
  let representative = longitude % 360;
  if (representative < -180) representative += 360;
  if (representative >= 180) representative -= 360;
  return [Math.abs(latitude) === 90 ? 0 : representative, latitude];
}

/** @internal Shader counterpart; source coordinates and row identities stay untouched. */
export function getTrajectoryVotePositionWGSL(geographic: boolean): string {
  return `fn canonicalLongitude(longitude: f32) -> f32 {
  var representative = longitude % 360.0;
  if (representative < -180.0) { representative += 360.0; }
  if (representative >= 180.0) { representative -= 360.0; }
  return representative;
}
fn votePosition(position: vec2<f32>) -> vec2<f32> {
  ${
    geographic
      ? `if (abs(position.y) == 90.0) { return vec2<f32>(0.0, position.y); }
  return vec2<f32>(canonicalLongitude(position.x), position.y);`
      : 'return position;'
  }
}`;
}
