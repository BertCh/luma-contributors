// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL for Deck's GlobeView (project module `PROJECTION_MODE_GLOBE` and `PROJECTION_MODE_GLOBE_METER_OFFSETS`). Positions still go through
 * the shared project module; these helpers add the pieces Deck leaves to each layer.
 *
 * - Horizon: Deck draws no globe occluder, so a point is hidden when its tangent plane faces away
 *   from the camera (`dot(P, camera - P) <= 0` in common space, radius 256 plus altitude).
 * - Great circles: a port of Deck's ArcLayer spherical interpolation, which Kepler uses for
 *   globe arcs and lines. At exactly antipodal pairs it falls back to linear lng/lat like Deck.
 *
 * Triangle and stroke layers interpolate the common-space position and discard fragments past the
 * horizon with `gpuGlobeIsCommonHidden`, so partially hidden primitives clip exactly at the limb.
 *
 * CARTESIAN layers (for example resident aggregate cells) are read as preprojected Web Mercator
 * common space on a globe; see `gpuGlobeFromMercatorCommon`.
 *
 * Requires `project32`. Functions are no-ops unless the viewport is a globe.
 */
export const GPU_GLOBE_WGSL = /* wgsl */ `
// Globe projection, including the meter-offsets variant (deck 10 reports it as its own projection
// mode; deck 9 reported it as PROJECTION_MODE_GLOBE).
fn gpuGlobeIsActive() -> bool {
  return project.projectionMode == PROJECTION_MODE_GLOBE || project.projectionMode == PROJECTION_MODE_GLOBE_METER_OFFSETS;
}
fn gpuGlobeIsCommonHidden(commonPosition: vec3<f32>) -> bool {
  if (!gpuGlobeIsActive()) { return false; }
  return dot(commonPosition, project.cameraPosition - commonPosition) <= 0.0;
}
// Position in the layer's own coordinate frame (longitude/latitude, meter offsets or geocentric
// meters): resolved to common space through the project module before the horizon test.
fn gpuGlobeIsHidden(position: vec3<f32>) -> bool {
  if (!gpuGlobeIsActive()) { return false; }
  return gpuGlobeIsCommonHidden(project_position_vec3_f64(position, vec3<f32>(0.0)));
}
// Longitude, latitude and altitude that the great-circle code has already produced.
fn gpuGlobeIsLngLatHidden(lngLatAltitude: vec3<f32>) -> bool {
  if (!gpuGlobeIsActive()) { return false; }
  return gpuGlobeIsCommonHidden(project_globe_(lngLatAltitude));
}
// Deck interpolates arcs along great circles only for geodetic data on a globe; meter offsets and
// geocentric data stay in projected common space.
fn gpuGlobeUsesGreatCircle() -> bool {
  return project.projectionMode == PROJECTION_MODE_GLOBE && project.coordinateFrame == COORDINATE_SYSTEM_GEODETIC;
}
// Preprojected Web Mercator common space (Deck CARTESIAN with the origin and model matrix the layer
// supplies) mapped onto the globe: invert the Mercator projection to longitude, latitude and
// altitude, then project through the globe branch. Returns the globe common position.
fn gpuGlobeFromMercatorCommon(position: vec3<f32>) -> vec3<f32> {
  let mercator = (project.modelMatrix * vec4<f32>(position, 1.0)).xyz + project.coordinateOrigin;
  let longitude = degrees(mercator.x / WORLD_SCALE - PI);
  let latitudeRadians = 2.0 * atan(exp(mercator.y / WORLD_SCALE - PI)) - PI / 2.0;
  let altitude = mercator.z * EARTH_RADIUS * cos(latitudeRadians) / WORLD_SCALE;
  return project_globe_(vec3<f32>(longitude, degrees(latitudeRadians), altitude));
}
fn gpuGlobeUsesMercatorCommon() -> bool {
  // Deck 10 packs 'cartesian' attributes on a globe viewport as the globe's geocentric frame.
  return project.projectionMode == PROJECTION_MODE_GLOBE && project.coordinateFrame == COORDINATE_SYSTEM_GEOCENTRIC;
}
fn gpuGlobeAngularDistance(source: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let sourceRadians = radians(source);
  let targetRadians = radians(targetPosition);
  let sinHalfDelta = sin((sourceRadians - targetRadians) / 2.0);
  let sinHalfDeltaSquared = sinHalfDelta * sinHalfDelta;
  let a = sinHalfDeltaSquared.y + cos(sourceRadians.y) * cos(targetRadians.y) * sinHalfDeltaSquared.x;
  return 2.0 * asin(sqrt(a));
}
// Longitude, latitude and arc height at \`ratio\` along the great circle; height is a distance multiplier.
fn gpuGlobeInterpolate(source: vec2<f32>, targetPosition: vec2<f32>, angularDistance: f32, ratio: f32, height: f32) -> vec3<f32> {
  var longitudeLatitude: vec2<f32>;
  if (abs(angularDistance - PI) < 0.001) {
    longitudeLatitude = (1.0 - ratio) * source + ratio * targetPosition;
  } else {
    let a = sin((1.0 - ratio) * angularDistance);
    let b = sin(ratio * angularDistance);
    let p = project_globe_(vec3<f32>(source, 0.0)).yxz * a + project_globe_(vec3<f32>(targetPosition, 0.0)).yxz * b;
    longitudeLatitude = degrees(vec2<f32>(atan2(p.y, -p.x), atan2(p.z, length(p.xy))));
  }
  return vec3<f32>(longitudeLatitude, sqrt(max(ratio * (1.0 - ratio), 0.0)) * angularDistance * EARTH_RADIUS * height);
}
`;
