// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_WGS84_ELLIPSOID,
  type GPUSpatialEllipsoid
} from '../../analysis-core/contracts/spatial-context';

/**
 * Earth model of the geodesic column contributors: great circles on a sphere, Vincenty on the context ellipsoid (`wgs84`),
 * or `'rhumb'` lines of constant bearing on a sphere (turf `rhumb*`, `geo` `Rhumb`).
 */
export type GPUGeodesicModel = 'sphere' | 'wgs84' | 'rhumb';

/** Default compile-time Vincenty iteration count. */
export const GPU_GEODESIC_DEFAULT_ITERATIONS = 16;

/** Bounded per-invocation work; caps shader iteration cost rather than row count. @internal */
export const GPU_GEODESIC_MAXIMUM_ITERATIONS = 200;

/**
 * Exclude the ill-conditioned inverse within 0.02 radians of an antipode from the
 * pairs accuracy envelope, even if float32 fixed-point iteration stops. @internal
 */
export const GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN = 0.02;

/**
 * Arrival azimuth amplifies normalized position error by 1 / cos(latitude).
 * A 2e-6 normalized position budget (12.8 m on Earth) divided by 0.03 is
 * below the measured 0.004-degree azimuth envelope. Exclude more ill-conditioned
 * arrivals even when the direct iteration converges. @internal
 */
export const GPU_GEODESIC_DIRECT_MINIMUM_AZIMUTH_COSINE = 0.03;

/** Result of `vincentyInverse` in WGSL. @internal */
const VINCENTY_INVERSE_STRUCT = /* wgsl */ `
struct VincentyInverse {
  distance: f32,
  initialBearing: f32,
  finalBearing: f32,
  converged: bool,
  iterationLimitReached: bool,
}

struct VincentyDirect {
  destination: vec2<f32>,
  finalBearing: f32,
  converged: bool,
  // Unrounded local quantities for short-edge area integration.
  longitudeDelta: f32,
  sinLatitudeDelta: f32,
  hasLocalDelta: bool,
}
`;

/**
 * Returns the WGSL of the ellipsoidal Vincenty inverse and direct solutions in f32, rewritten so short
 * lines stay well conditioned: the reduced-latitude difference comes from `Δφ` minus the small
 * gap `φ - β` (never from two rounded reduced latitudes), and `cos U1 sin U2 - sin U1 cos U2 cos λ`
 * is evaluated as `sin(ΔU) + 2 sin U1 cos U2 sin²(λ / 2)`. Iterations stop at a compile-time cap;
 * rows that do not converge (near-antipodal) report `converged = false`. Callers prepend
 * `GEODESIC_WGSL`. Distances use the unit of the ellipsoid's semi-major axis.
 *
 * @internal
 */
export function getVincentyWGSL(
  iterations: number,
  ellipsoid: GPUSpatialEllipsoid = GPU_WGS84_ELLIPSOID
): string {
  const {semiMajorAxis, flattening} = ellipsoid;
  const semiMinorAxis = semiMajorAxis * (1 - flattening);
  return /* wgsl */ `
${VINCENTY_INVERSE_STRUCT}
const VINCENTY_A: f32 = ${getWGSLFloatLiteral(semiMajorAxis)};
const VINCENTY_B: f32 = ${getWGSLFloatLiteral(semiMinorAxis)};
const VINCENTY_F: f32 = ${getWGSLFloatLiteral(flattening)};
// (a^2 - b^2) / b^2
const VINCENTY_SECOND_ECCENTRICITY_SQUARED: f32 = ${getWGSLFloatLiteral((semiMajorAxis * semiMajorAxis - semiMinorAxis * semiMinorAxis) / (semiMinorAxis * semiMinorAxis))};
const VINCENTY_ITERATIONS: u32 = ${iterations}u;
// Below a millidegree of colatitude, the rounded-radian gap loses relative precision.
const VINCENTY_POLAR_LATITUDE: f32 = 89.999;
// Midpoint inversion of reduced latitude: the omitted cubic term is below 1 cm
// on WGS84 over a 0.01-radian span. Polar spans retain the general direct solution.
const VINCENTY_LOCAL_AREA_MAXIMUM_ANGLE: f32 = 0.01;
const VINCENTY_LOCAL_AREA_MAXIMUM_LATITUDE: f32 = 85.0;
// The short-span cubic estimate assumes an Earth-like flattening.
const VINCENTY_LOCAL_AREA_MAXIMUM_FLATTENING: f32 = 0.005;

fn getReducedLatitudeGap(phi: f32) -> f32 {
  let sinPhi = sin(phi);
  let cosPhi = cos(phi);
  return atan(VINCENTY_F * sinPhi * cosPhi / (cosPhi * cosPhi + (1.0 - VINCENTY_F) * sinPhi * sinPhi));
}

// Keep the established colatitude-plus-gap arithmetic on ordinary paths, including
// geometry-area integration. Only the polar singularity needs normalization: there
// cos(phi) and gap(phi) lose relative precision after phi rounds to radians.
fn getCosReducedLatitude(phi: f32, latitudeDegrees: f32) -> f32 {
  if (abs(latitudeDegrees) <= VINCENTY_POLAR_LATITUDE) {
    return sin((90.0 - abs(latitudeDegrees)) * GEODESIC_DEGREES_TO_RADIANS + getReducedLatitudeGap(abs(phi)));
  }
  let cosine = geodesicCosLatitude(latitudeDegrees);
  let scaledSine = (1.0 - VINCENTY_F) * sin(phi);
  return cosine / length(vec2<f32>(cosine, scaledSine));
}

fn getReducedLatitude(phi: f32) -> f32 {
  return phi - getReducedLatitudeGap(phi);
}

fn getVincentyDeltaSigma(bigB: f32, sinSigma: f32, cosSigma: f32, cos2SigmaM: f32) -> f32 {
  let cos2SigmaMSquared = cos2SigmaM * cos2SigmaM;
  return bigB * sinSigma * (cos2SigmaM + bigB / 4.0 * (cosSigma * (-1.0 + 2.0 * cos2SigmaMSquared) -
    bigB / 6.0 * cos2SigmaM * (-3.0 + 4.0 * sinSigma * sinSigma) * (-3.0 + 4.0 * cos2SigmaMSquared)));
}

fn getVincentyCoefficients(cosSquaredAlpha: f32) -> vec2<f32> {
  let uSquared = cosSquaredAlpha * VINCENTY_SECOND_ECCENTRICITY_SQUARED;
  let bigA = 1.0 + uSquared / 16384.0 * (4096.0 + uSquared * (-768.0 + uSquared * (320.0 - 175.0 * uSquared)));
  let bigB = uSquared / 1024.0 * (256.0 + uSquared * (-128.0 + uSquared * (74.0 - 47.0 * uSquared)));
  return vec2<f32>(bigA, bigB);
}

fn vincentyInverse(a: vec2<f32>, b: vec2<f32>) -> VincentyInverse {
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let reduced1 = getReducedLatitude(phi1);
  let reduced2 = getReducedLatitude(phi2);
  // U2 - U1: the midpoint derivative dU/dphi for short spans (the gap difference would lose
  // relative precision there), else deltaPhi minus the difference of the small gaps.
  var deltaReduced = deltaPhi - (getReducedLatitudeGap(phi2) - getReducedLatitudeGap(phi1));
  if (abs(deltaPhi) < 0.01) {
    let sinMid = sin(0.5 * (phi1 + phi2));
    let cosMid = cos(0.5 * (phi1 + phi2));
    deltaReduced = deltaPhi * (1.0 - VINCENTY_F) /
      (cosMid * cosMid + (1.0 - VINCENTY_F) * (1.0 - VINCENTY_F) * sinMid * sinMid);
  }
  let sinU1 = sin(reduced1);
  let cosU1 = getCosReducedLatitude(phi1, a.y);
  let sinU2 = sin(reduced2);
  let cosU2 = getCosReducedLatitude(phi2, b.y);
  let sinDeltaU = sin(deltaReduced);
  // Seam-safe difference: wrapping b.x - a.x afterwards loses 1.7 m across the antimeridian.
  let longitudeDelta = geodesicLongitudeDelta(a.x, b.x) * GEODESIC_DEGREES_TO_RADIANS;
  var lambda = longitudeDelta;
  var result: VincentyInverse;
  result.converged = false;
  // At a pole longitude does not distinguish positions. Opposite poles are singular,
  // not coincident; test this before the zero-sine branch below.
  if (a.y == b.y && (longitudeDelta == 0.0 || abs(a.y) == 90.0)) {
    result.converged = true;
    return result;
  }
  if (GEODESIC_PI - geodesicCentralAngle(a, b) < 1e-6) { return result; }
  var sinSigma = 0.0;
  var cosSigma = 1.0;
  var sigma = 0.0;
  var cosSquaredAlpha = 1.0;
  var cos2SigmaM = 0.0;
  var sinLambda = 0.0;
  var halfVersine = 0.0;
  // Each pass evaluates the auxiliary sphere at lambda; the final pass after convergence (or the
  // cap) evaluates it at the converged lambda, so the distance never mixes two lambda values.
  for (var iteration = 0u; iteration <= VINCENTY_ITERATIONS; iteration++) {
    sinLambda = sin(lambda);
    let sinHalfLambda = sin(0.5 * lambda);
    halfVersine = 2.0 * sinHalfLambda * sinHalfLambda;
    let northTerm = sinDeltaU + sinU1 * cosU2 * halfVersine;
    let eastTerm = cosU2 * sinLambda;
    sinSigma = sqrt(eastTerm * eastTerm + northTerm * northTerm);
    if (sinSigma == 0.0) {
      result.distance = 0.0;
      result.initialBearing = 0.0;
      result.finalBearing = 0.0;
      return result;
    }
    cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * (1.0 - halfVersine);
    sigma = geodesicAtan2(sinSigma, cosSigma);
    let sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
    cosSquaredAlpha = 1.0 - sinAlpha * sinAlpha;
    cos2SigmaM = 0.0;
    if (cosSquaredAlpha != 0.0) {
      cos2SigmaM = cosSigma - 2.0 * sinU1 * sinU2 / cosSquaredAlpha;
    }
    if (result.converged || iteration == VINCENTY_ITERATIONS) {
      break;
    }
    let c = VINCENTY_F / 16.0 * cosSquaredAlpha * (4.0 + VINCENTY_F * (4.0 - 3.0 * cosSquaredAlpha));
    let previous = lambda;
    lambda = longitudeDelta + (1.0 - c) * VINCENTY_F * sinAlpha *
      (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));
    // Relative test: about four f32 ulps of lambda (exact zero for meridian lines).
    if (abs(lambda - previous) <= 4.8e-7 * abs(lambda)) {
      result.converged = true;
    }
  }
  result.iterationLimitReached = !result.converged;
  let coefficients = getVincentyCoefficients(cosSquaredAlpha);
  let deltaSigma = getVincentyDeltaSigma(coefficients.y, sinSigma, cosSigma, cos2SigmaM);
  result.distance = VINCENTY_B * coefficients.x * (sigma - deltaSigma);
  result.initialBearing = geodesicAtan2(cosU2 * sinLambda, sinDeltaU + sinU1 * cosU2 * halfVersine) *
    GEODESIC_RADIANS_TO_DEGREES;
  result.finalBearing = geodesicAtan2(cosU1 * sinLambda, sinDeltaU - cosU1 * sinU2 * halfVersine) *
    GEODESIC_RADIANS_TO_DEGREES;
  return result;
}

fn vincentyDirect(origin: vec2<f32>, bearingDegrees: f32, distance: f32) -> VincentyDirect {
  if (distance == 0.0) {
    return VincentyDirect(origin, geodesicWrapLongitudeDelta(bearingDegrees), true, 0.0, 0.0, true);
  }
  let phi1 = origin.y * GEODESIC_DEGREES_TO_RADIANS;
  let alpha1 = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let sinAlpha1 = sin(alpha1);
  let cosAlpha1 = cos(alpha1);
  let reduced1 = getReducedLatitude(phi1);
  let sinU1 = sin(reduced1);
  let cosU1 = getCosReducedLatitude(phi1, origin.y);
  let sigma1 = geodesicAtan2(sinU1, cosU1 * cosAlpha1);
  let sinAlpha = cosU1 * sinAlpha1;
  let cosSquaredAlpha = 1.0 - sinAlpha * sinAlpha;
  let coefficients = getVincentyCoefficients(cosSquaredAlpha);
  let baseSigma = distance / (VINCENTY_B * coefficients.x);
  var sigma = baseSigma;
  var sinSigma = sin(sigma);
  var cosSigma = cos(sigma);
  var cos2SigmaM = cos(2.0 * sigma1 + sigma);
  var result: VincentyDirect;
  result.converged = false;
  for (var iteration = 0u; iteration < VINCENTY_ITERATIONS; iteration++) {
    cos2SigmaM = cos(2.0 * sigma1 + sigma);
    sinSigma = sin(sigma);
    cosSigma = cos(sigma);
    let previous = sigma;
    sigma = baseSigma + getVincentyDeltaSigma(coefficients.y, sinSigma, cosSigma, cos2SigmaM);
    if (abs(sigma - previous) <= 2.4e-7 * max(1.0, sigma)) {
      result.converged = true;
      break;
    }
  }
  sinSigma = sin(sigma);
  cosSigma = cos(sigma);
  cos2SigmaM = cos(2.0 * sigma1 + sigma);
  let northTerm = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1;
  let phi2 = geodesicAtan2(
    sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
    (1.0 - VINCENTY_F) * sqrt(sinAlpha * sinAlpha + northTerm * northTerm)
  );
  let lambda = geodesicAtan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1);
  let c = VINCENTY_F / 16.0 * cosSquaredAlpha * (4.0 + VINCENTY_F * (4.0 - 3.0 * cosSquaredAlpha));
  let longitudeDelta = lambda - (1.0 - c) * VINCENTY_F * sinAlpha *
    (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));
  result.destination = vec2<f32>(
    origin.x + longitudeDelta * GEODESIC_RADIANS_TO_DEGREES,
    phi2 * GEODESIC_RADIANS_TO_DEGREES
  );
  result.finalBearing = geodesicAtan2(sinAlpha, -northTerm) * GEODESIC_RADIANS_TO_DEGREES;
  result.longitudeDelta = longitudeDelta * GEODESIC_RADIANS_TO_DEGREES;
  if (abs(sigma) < VINCENTY_LOCAL_AREA_MAXIMUM_ANGLE && abs(origin.y) < VINCENTY_LOCAL_AREA_MAXIMUM_LATITUDE && abs(VINCENTY_F) <= VINCENTY_LOCAL_AREA_MAXIMUM_FLATTENING) {
    // sin(U2) - sin(U1) directly, without subtracting two rounded sines.
    let halfSine = sin(0.5 * sigma);
    let sineChange = -2.0 * sinU1 * halfSine * halfSine + cosU1 * sinSigma * cosAlpha1;
    let targetReducedLatitude = geodesicAtan2(
      sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
      sqrt(sinAlpha * sinAlpha + northTerm * northTerm)
    );
    let reducedChange = 2.0 * asin(clamp(sineChange / (2.0 * cos(0.5 * (reduced1 + targetReducedLatitude))), -1.0, 1.0));
    let flatteningComplement = 1.0 - VINCENTY_F;
    var latitudeChange = reducedChange * (cos(phi1) * cos(phi1) + flatteningComplement * flatteningComplement * sin(phi1) * sin(phi1)) / flatteningComplement;
    let middleLatitude = phi1 + 0.5 * latitudeChange;
    latitudeChange = reducedChange * (cos(middleLatitude) * cos(middleLatitude) + flatteningComplement * flatteningComplement * sin(middleLatitude) * sin(middleLatitude)) / flatteningComplement;
    result.sinLatitudeDelta = 2.0 * cos(phi1 + 0.5 * latitudeChange) * sin(0.5 * latitudeChange);
    result.hasLocalDelta = true;
  }
  return result;
}
`;
}

/** Sphere final bearing: the reverse initial bearing turned by 180 degrees. @internal */
export const SPHERE_PAIR_WGSL = /* wgsl */ `
// Arrival azimuth from the direct solution, before destination coordinates round to f32.
fn sphereDirectFinalBearingDegrees(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> f32 {
  // Preserve the sphere contributor's existing coincident reverse-bearing convention.
  if (angle == 0.0) { return 180.0; }
  let bearing = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let cosine = geodesicCosLatitude(origin.y);
  let north = sin(origin.y * GEODESIC_DEGREES_TO_RADIANS) * sin(angle) -
    cosine * cos(angle) * cos(bearing);
  return geodesicAtan2(cosine * sin(bearing), -north) * GEODESIC_RADIANS_TO_DEGREES;
}

fn sphereFinalBearingDegrees(a: vec2<f32>, b: vec2<f32>) -> f32 {
  var bearing = geodesicInitialBearingDegrees(b, a) + 180.0;
  if (bearing > 180.0) {
    bearing -= 360.0;
  }
  return bearing;
}
`;

/**
 * Retains the ordinary inverse for easy rows; bracketed auxiliary-longitude roots handle
 * oscillating fixed-point iterations. Singular/discontinuous roots remain explicit failures.
 * This does not claim GeographicLib's all-case series accuracy or antipodal uniqueness.
 */
export function getSafeguardedVincentyWGSL(
  iterations: number,
  ellipsoid: GPUSpatialEllipsoid = GPU_WGS84_ELLIPSOID
): string {
  let source = getVincentyWGSL(iterations, ellipsoid);
  source = source.replace('  converged: bool,', '  converged: bool,\n  residual: f32,');
  const start = source.indexOf('fn vincentyInverse(');
  const end = source.indexOf('fn vincentyDirect(', start);
  let candidate = source
    .slice(start, end)
    .replace(
      'fn vincentyInverse(a: vec2<f32>, b: vec2<f32>)',
      'fn vincentyCandidate(a: vec2<f32>, b: vec2<f32>, auxiliaryLongitude: f32)'
    )
    .replace('var lambda = longitudeDelta;', 'var lambda = auxiliaryLongitude;')
    .replace('iteration <= VINCENTY_ITERATIONS', 'iteration <= 0u')
    .replace('iteration == VINCENTY_ITERATIONS', 'iteration == 0u');
  candidate = candidate.replace(
    '  let coefficients = getVincentyCoefficients(cosSquaredAlpha);',
    `  let sineAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
  let correction = VINCENTY_F / 16.0 * cosSquaredAlpha * (4.0 + VINCENTY_F * (4.0 - 3.0 * cosSquaredAlpha));
  result.residual = lambda - longitudeDelta - (1.0 - correction) * VINCENTY_F * sineAlpha * (sigma + correction * sinSigma * (cos2SigmaM + correction * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));
  let coefficients = getVincentyCoefficients(cosSquaredAlpha);`
  );
  source = source.replace('fn vincentyInverse(', 'fn vincentyInverseFast(');
  return `${source}\n${candidate}
fn vincentyInverse(a: vec2f, b: vec2f) -> VincentyInverse {
  var result = vincentyInverseFast(a, b);
  if (result.converged || GEODESIC_PI - geodesicCentralAngle(a, b) < 1e-6) { return result; }
  let longitude = geodesicLongitudeDelta(a.x, b.x) * GEODESIC_DEGREES_TO_RADIANS;
  let lower = max(-3.141592653589793, longitude - 4.0 * abs(VINCENTY_F) - 0.001);
  let upper = min(3.141592653589793, longitude + 4.0 * abs(VINCENTY_F) + 0.001);
  var previousLongitude = lower;
  var previous = vincentyCandidate(a, b, lower);
  var shortest = 3.4e38;
  for (var interval = 1u; interval <= 32u; interval++) {
    let nextLongitude = mix(lower, upper, f32(interval) / 32.0);
    let next = vincentyCandidate(a, b, nextLongitude);
    if (previous.residual * next.residual <= 0.0) {
      var left = previousLongitude; var right = nextLongitude; var leftResidual = previous.residual;
      var best = previous;
      if (abs(next.residual) < abs(best.residual)) { best = next; }
      for (var iteration = 0u; iteration < 32u; iteration++) {
        let middle = (left + right) * 0.5;
        if (middle == left || middle == right) { break; }
        let trial = vincentyCandidate(a, b, middle);
        if (abs(trial.residual) < abs(best.residual)) { best = trial; }
        if (leftResidual * trial.residual <= 0.0) { right = middle; } else { left = middle; leftResidual = trial.residual; }
      }
      if (abs(best.residual) <= 7e-7 && best.distance > 0.0 && best.distance < shortest) { best.converged = true; best.iterationLimitReached = false; result = best; shortest = best.distance; }
    }
    previousLongitude = nextLongitude; previous = next;
  }
  return result;
}`;
}
