// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Oracle for `GEODESIC_WGSL`, used by every geometry contributor test; great-circle terms delegate to
 * math.gl `EllipsoidGeodesic` on a sphere (Karney). Positions are
 * `[longitude, latitude]` in degrees; angles in radians unless the name says degrees.
 */

import {EllipsoidGeodesic} from '@math.gl/geospatial';

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;

/** Wraps a longitude difference into `[-180, 180]`. */
export function wrapLongitudeDelta(delta: number): number {
  return delta - 360 * Math.round(delta / 360);
}

/** Unit vector on the sphere for a longitude/latitude pair in degrees. */
export function getUnitVector(lngLat: readonly number[]): [number, number, number] {
  const lambda = lngLat[0] * DEGREES_TO_RADIANS;
  const phi = lngLat[1] * DEGREES_TO_RADIANS;
  return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
}

/** Longitude/latitude in degrees of a (not necessarily unit) vector. */
export function getLngLatFromVector(vector: readonly number[]): [number, number] {
  const horizontal = Math.hypot(vector[0], vector[1]);
  return [
    Math.atan2(vector[1], vector[0]) * RADIANS_TO_DEGREES,
    Math.atan2(vector[2], horizontal) * RADIANS_TO_DEGREES
  ];
}

const sphereGeodesics = new Map<number, EllipsoidGeodesic>();

/** Karney geodesic on a sphere of `radius` (an independent math.gl oracle for great circles). */
function getSphereGeodesic(radius: number): EllipsoidGeodesic {
  let geodesic = sphereGeodesics.get(radius);
  if (!geodesic) {
    geodesic = new EllipsoidGeodesic({semiMajorAxis: radius, semiMinorAxis: radius});
    sphereGeodesics.set(radius, geodesic);
  }
  return geodesic;
}

/** Great-circle central angle in radians. */
export function getCentralAngle(a: readonly number[], b: readonly number[]): number {
  return getSphereGeodesic(1).inverse(a[1], a[0], b[1], b[0]).s12;
}

/** Great-circle distance in `radius` units. */
export function getHaversineDistance(
  a: readonly number[],
  b: readonly number[],
  radius: number
): number {
  return getSphereGeodesic(radius).inverse(a[1], a[0], b[1], b[0]).s12;
}

/** Initial bearing in degrees clockwise from north, in `(-180, 180]`. */
export function getInitialBearingDegrees(a: readonly number[], b: readonly number[]): number {
  return getSphereGeodesic(1).inverse(a[1], a[0], b[1], b[0]).azi1;
}

/** Destination after travelling `angle` radians from `origin` at `bearingDegrees`, longitude continuous with the origin. */
export function getDestination(
  origin: readonly number[],
  bearingDegrees: number,
  angle: number
): [number, number] {
  const result = getSphereGeodesic(1).direct(origin[1], origin[0], bearingDegrees, angle, {
    unrollLongitude: true
  });
  return [result.lon2, result.lat2];
}

/**
 * Slerp at `fraction` of the great circle from `a` to `b`, with the longitude unwrapped to stay
 * continuous with `a`.
 */
export function interpolateGreatCircle(
  a: readonly number[],
  b: readonly number[],
  fraction: number
): [number, number] {
  const {s12, azi1} = getSphereGeodesic(1).inverse(a[1], a[0], b[1], b[0]);
  const {lat2, lon2} = getSphereGeodesic(1).direct(a[1], a[0], azi1, s12 * fraction);
  return [a[0] + wrapLongitudeDelta(lon2 - a[0]), lat2];
}
