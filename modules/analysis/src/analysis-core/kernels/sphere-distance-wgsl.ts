// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Great-circle distance pieces for WGSL (contributor contract rule 11), in the two forms the
 * library's private copies use: the haversine term `h = sin^2(angle / 2)`, which is monotonic in
 * distance and serves as a sort and radius key, and the chord between unit vectors,
 * `c = 2 sin(angle / 2)`. Angles are radians on the unit sphere; multiply by a radius for a length.
 *
 * Point-to-point distance on `(longitude, latitude)` degrees is `geodesicHaversineDistance` (or
 * `geodesicCentralAngle`) of `GEODESIC_WGSL` in `gpu-spatial-analysis/geometry-measures`, which
 * handles the antimeridian and pole precision; this module holds the conversions around it that
 * kernels otherwise inline. The radius comes from the spatial context: `getGPUSphereRadius` (mean
 * radius) for distances, `getGPUAuthalicSphereRadius` (equal-area radius) for areas, both in
 * `contracts/spatial-context.ts` and in the context's length unit.
 *
 * Functions (names start with `getSphere`, so they cannot collide with `GEODESIC_WGSL`):
 * - `getSphereHaversineTerm(deltaLatitude, deltaLongitude, cosineProduct) -> f32`: `h` from the
 *   latitude and longitude differences (radians) and `cos(latitude1) cos(latitude2)` (or any
 *   bound on it, as the box lower bounds use), clamped to `[0, 1]`.
 * - `getSphereAngleFromHaversineTerm(term) -> f32`: the central angle `2 atan2(sqrt(h),
 *   sqrt(1 - h))` in `[0, pi]`. Near the antipode it is no more precise than `2 asin(sqrt(h))`:
 *   both read the same float32 `h`, and `h = 1 - sin^2((pi - angle) / 2)` rounds to exactly 1
 *   within about 3.4e-4 rad of the antipode, so the angle returned there is `pi` (up to 2 km on
 *   Earth), and it is about 150 m off at `pi - 1e-3`. For nearly antipodal points use the vector
 *   form, `geodesicCentralAngle` of `GEODESIC_WGSL`, which does not pass through `h`. The second
 *   `atan2` argument is zero at the antipode; that case returns `pi` without calling `atan2`,
 *   because on the test GPU `atan2(y, 0)` returned `-pi/2` for positive `y` in one kernel (FX-17).
 * - `getSphereDistanceFromHaversineTerm(term, radius) -> f32`: the angle times `radius`.
 * - `getSphereHaversineTermFromAngle(angle) -> f32`: `sin^2(min(angle, pi) / 2)`, the radius key of
 *   a search radius (`d <= r` iff `h(d) <= h(r)`).
 * - `getSphereChordFromAngle(angle) -> f32`: `2 sin(min(angle, pi) / 2)`, in `[0, 2]`.
 * - `getSphereAngleFromChord(chord) -> f32`: the central angle of a unit-sphere chord, clamped to
 *   `[0, 2]`; `2 atan2(c / 2, sqrt(1 - c^2 / 4))` with the same antipode guard.
 *
 * Checked against float64 in `test/utils/wgsl-sphere-rows.spec.ts`.
 *
 * @internal
 */
export const SPHERE_DISTANCE_WGSL = /* wgsl */ `
const SPHERE_DISTANCE_PI: f32 = 3.14159265358979;

fn getSphereHaversineTerm(deltaLatitude: f32, deltaLongitude: f32, cosineProduct: f32) -> f32 {
  let sinHalfLatitude = sin(0.5 * deltaLatitude);
  let sinHalfLongitude = sin(0.5 * deltaLongitude);
  return clamp(sinHalfLatitude * sinHalfLatitude + cosineProduct * sinHalfLongitude * sinHalfLongitude, 0.0, 1.0);
}

fn getSphereAngleFromHaversineTerm(term: f32) -> f32 {
  let clamped = clamp(term, 0.0, 1.0);
  let cosineHalf = sqrt(1.0 - clamped);
  // atan2(y, 0) is not trusted on every device (FX-17): the antipode is pi.
  if (cosineHalf <= 0.0) {
    return SPHERE_DISTANCE_PI;
  }
  return 2.0 * atan2(sqrt(clamped), cosineHalf);
}

fn getSphereDistanceFromHaversineTerm(term: f32, radius: f32) -> f32 {
  return getSphereAngleFromHaversineTerm(term) * radius;
}

fn getSphereHaversineTermFromAngle(angle: f32) -> f32 {
  let sine = sin(0.5 * clamp(angle, 0.0, SPHERE_DISTANCE_PI));
  return sine * sine;
}

fn getSphereChordFromAngle(angle: f32) -> f32 {
  return 2.0 * sin(0.5 * clamp(angle, 0.0, SPHERE_DISTANCE_PI));
}

fn getSphereAngleFromChord(chord: f32) -> f32 {
  let half = 0.5 * clamp(chord, 0.0, 2.0);
  let cosineHalf = sqrt(max(1.0 - half * half, 0.0));
  if (cosineHalf <= 0.0) {
    return SPHERE_DISTANCE_PI;
  }
  return 2.0 * atan2(half, cosineHalf);
}
`;
