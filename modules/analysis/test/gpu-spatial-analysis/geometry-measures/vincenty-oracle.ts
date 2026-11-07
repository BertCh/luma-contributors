// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EllipsoidGeodesicExact} from '@math.gl/geospatial';
import type {GPUSpatialEllipsoid} from '../../../src/analysis-core/contracts/spatial-context';
import {wrapLongitudeDelta} from './geodesic-oracle';

const DEGREES_TO_RADIANS = Math.PI / 180;
const WGS84_ELLIPSOID = {semiMajorAxis: 6378137, flattening: 1 / 298.257223563};

const geodesics = new Map<string, EllipsoidGeodesicExact>();

/** Karney exact geodesic (math.gl) for the ellipsoid, the independent oracle for the Vincenty kernels. */
function getGeodesic({semiMajorAxis, flattening}: GPUSpatialEllipsoid): EllipsoidGeodesicExact {
  const key = `${semiMajorAxis}:${flattening}`;
  let geodesic = geodesics.get(key);
  if (!geodesic) {
    geodesic = new EllipsoidGeodesicExact({
      semiMajorAxis,
      semiMinorAxis: semiMajorAxis * (1 - flattening)
    });
    geodesics.set(key, geodesic);
  }
  return geodesic;
}

/**
 * Whether Vincenty's iteration on the auxiliary-sphere longitude converges. This is the only
 * Vincenty-specific behavior kept: Karney's solver always converges, but the GPU kernel mirrors
 * Vincenty and falls back to the sphere where it fails (near-antipodal pairs).
 */
function getVincentyConverges(
  a: readonly number[],
  b: readonly number[],
  flattening: number
): boolean {
  const longitudeDelta = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const u1 = Math.atan((1 - flattening) * Math.tan(a[1] * DEGREES_TO_RADIANS));
  const u2 = Math.atan((1 - flattening) * Math.tan(b[1] * DEGREES_TO_RADIANS));
  const [sinU1, cosU1, sinU2, cosU2] = [Math.sin(u1), Math.cos(u1), Math.sin(u2), Math.cos(u2)];
  let lambda = longitudeDelta;
  for (let iteration = 0; iteration < 1000; iteration++) {
    const sinLambda = Math.sin(lambda);
    const cosLambda = Math.cos(lambda);
    const sinSigma = Math.hypot(cosU2 * sinLambda, cosU1 * sinU2 - sinU1 * cosU2 * cosLambda);
    if (sinSigma === 0) {
      return true;
    }
    const cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
    const sigma = Math.atan2(sinSigma, cosSigma);
    const sinAlpha = (cosU1 * cosU2 * sinLambda) / sinSigma;
    const cosSquaredAlpha = 1 - sinAlpha * sinAlpha;
    const cos2SigmaM = cosSquaredAlpha === 0 ? 0 : cosSigma - (2 * sinU1 * sinU2) / cosSquaredAlpha;
    const c = (flattening / 16) * cosSquaredAlpha * (4 + flattening * (4 - 3 * cosSquaredAlpha));
    const previous = lambda;
    lambda =
      longitudeDelta +
      (1 - c) *
        flattening *
        sinAlpha *
        (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1 + 2 * cos2SigmaM ** 2)));
    if (Math.abs(lambda - previous) < 1e-13) {
      return true;
    }
  }
  return false;
}

/**
 * Karney inverse (math.gl): distance in the ellipsoid semi-major axis unit and bearings in
 * `(-180, 180]` degrees. `converged` reports whether Vincenty's iteration would converge; when it
 * does not, the distance and bearings are NaN, as for the Vincenty kernel.
 */
export function getVincentyInverse(
  a: readonly number[],
  b: readonly number[],
  ellipsoid: GPUSpatialEllipsoid = WGS84_ELLIPSOID
): {distance: number; initialBearing: number; finalBearing: number; converged: boolean} {
  if (!getVincentyConverges(a, b, ellipsoid.flattening)) {
    return {distance: NaN, initialBearing: NaN, finalBearing: NaN, converged: false};
  }
  const {s12, azi1, azi2} = getGeodesic(ellipsoid).inverse(a[1], a[0], b[1], b[0]);
  return {distance: s12, initialBearing: azi1, finalBearing: azi2, converged: true};
}

/** Karney direct (math.gl): distance in the ellipsoid semi-major axis unit; destination (longitude continuous with the origin) and final bearing. */
export function getVincentyDirect(
  origin: readonly number[],
  bearingDegrees: number,
  distance: number,
  ellipsoid: GPUSpatialEllipsoid = WGS84_ELLIPSOID
): {destination: [number, number]; finalBearing: number} {
  const result = getGeodesic(ellipsoid).direct(origin[1], origin[0], bearingDegrees, distance, {
    unrollLongitude: true
  });
  return {destination: [result.lon2, result.lat2], finalBearing: result.azi2};
}
