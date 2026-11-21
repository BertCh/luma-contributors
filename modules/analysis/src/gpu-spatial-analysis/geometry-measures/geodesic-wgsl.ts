// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '../../analysis-core/contracts/earth';

/**
 * Mean Earth radius in meters (IUGG R1), the default sphere radius of the geodesic contributors and of
 * turf's `distance`, `length`, `destination` and `along`. Alias of `GPU_EARTH_MEAN_RADIUS_METERS`.
 */
export const GPU_GEODESIC_MEAN_EARTH_RADIUS = GPU_EARTH_MEAN_RADIUS_METERS;

/** WGS84 semi-major axis in meters. Alias of `GPU_WGS84_SEMI_MAJOR_AXIS_METERS`. */
export const GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS = GPU_WGS84_SEMI_MAJOR_AXIS_METERS;

/** WGS84 flattening. Alias of `GPU_WGS84_FLATTENING`. */
export const GPU_GEODESIC_WGS84_FLATTENING = GPU_WGS84_FLATTENING;

/**
 * Public WGSL source of f32 spherical geodesy functions, to be included once in a compute shader
 * that works on longitude/latitude positions. Positions are `vec2<f32>(longitude, latitude)` in
 * degrees; angles are radians unless the name says degrees; distances are in the units of the
 * `radius` argument (pass `GPU_EARTH_MEAN_RADIUS_METERS` from `contracts/earth.ts` for meters).
 * All function and constant names start with `geodesic` / `GEODESIC_`.
 *
 * Functions:
 * - `geodesicHaversineDistance(a, b, radius) -> f32`: great-circle distance, well conditioned from
 *   a millimeter to antipodal. The shared distance of neighbor search, joins, trajectories and
 *   network snapping.
 * - `geodesicHaversineTerm(a, b) -> f32`: haversine `h` in [0, 1], monotonic in distance; cheaper
 *   for ordering or radius tests: `d <= r` iff `h <= sin(r / (2 R))^2`.
 * - `geodesicCentralAngle(a, b) -> f32`: central angle in radians.
 * - `geodesicInitialBearingDegrees(a, b) -> f32`: initial bearing, degrees clockwise from north.
 * - `geodesicDestination(origin, bearingDegrees, angle) -> vec2<f32>`: point at central `angle`
 *   along a bearing; longitude continuous with the origin.
 * - `geodesicInterpolate(a, b, angle, fraction) -> vec2<f32>`: slerp along the great circle
 *   (`angle = geodesicCentralAngle(a, b)`); longitude continuous with `a`. Preserves known
 *   endpoints at fractions zero/one; returns NaN for interior interpolation within 1e-6
 *   radians of an antipode, where the interpolation path is not certified.
 * - `geodesicSegmentFraction(point, start, end) -> f32` and
 *   `geodesicSegmentDistance(point, start, end, fraction, radius) -> f32`: point-to-segment
 *   distance for segments that are straight in longitude/latitude (the edge model of the
 *   planar-in-degrees containment tests and of network edges). The closest fraction is found in
 *   the local equirectangular frame at the point on one wrapped start-to-end arc, then compared
 *   with both endpoint distances. The returned fraction selects the nearest of those three
 *   candidates. The distance is a haversine of coordinate
 *   differences. An interior projection survives a rounded distance tie; ties between endpoint
 *   candidates select the start. Short distances keep their precision near +-180.
 *   Exact to first order for
 *   segments of road length (kilometers); densify long edges. An explicitly supplied fraction
 *   is evaluated at that fraction; only `geodesicSegmentFraction` selects the nearest candidate.
 * - Helpers: `geodesicWrapLongitudeDelta(delta)` (wrap to [-180, 180]),
 *   `geodesicLongitudeDelta(source, destination)` (preserve short seam separations), `geodesicCosLatitude(lat)`,
 *   `geodesicUnitVector(lngLat)`, `geodesicFromUnitVector(v)`, `geodesicLocalOffset(origin, p)`,
 *   `geodesicSinLatitudeDelta(lat, lat0)` (`sin(lat) - sin(lat0)` without cancellation).
 *
 * Precision rules, measured against the f64 twins in the geometry-measures test oracle:
 * differences are taken in degrees first so nearby points keep their relative precision; cosines
 * of latitude come from the exact colatitude; bearings and destinations use half-angle forms; the
 * slerp falls back to a normalized lerp below `1e-6` rad.
 *
 * @example
 * ```ts
 * import {GEODESIC_WGSL} from '@luma-contributors/analysis/gpu-spatial-analysis';
 * import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma-contributors/analysis/analysis-core';
 * const source = `${GEODESIC_WGSL}
 * fn distanceMeters(a: vec2<f32>, b: vec2<f32>) -> f32 {
 *   return geodesicHaversineDistance(a, b, ${GPU_EARTH_MEAN_RADIUS_METERS});
 * }`;
 * ```
 */
export const GEODESIC_WGSL = /* wgsl */ `
${GUARDED_ATAN2_WGSL.replaceAll('guardedAtan2', 'geodesicAtan2')}
const GEODESIC_PI: f32 = 3.14159265358979;
// Component reconstruction avoids polar and long-path cancellation; these boundaries
// retain the established short-path arithmetic and are covered by the PROJ regressions.
const GEODESIC_DIRECT_COMPONENT_ANGLE: f32 = 0.25 * GEODESIC_PI;
const GEODESIC_DIRECT_POLAR_LATITUDE: f32 = 80.0;
const GEODESIC_DIRECT_POLAR_SINE: f32 = 0.984807753012208; // sin(80 degrees)
const GEODESIC_DEGREES_TO_RADIANS: f32 = 0.0174532925199433;
const GEODESIC_RADIANS_TO_DEGREES: f32 = 57.2957795130823;
// Haversine terms remain well conditioned below a 90-degree separation. Compare local snap
// candidates in that common representation, before asin/atan2 rounding can reverse a near tie.
const GEODESIC_LOCAL_COMPARISON_MAXIMUM_TERM: f32 = 0.5;

fn geodesicWrapLongitudeDelta(delta: f32) -> f32 {
  return delta - 360.0 * round(delta / 360.0);
}

// Exact turn reduction of finite f32 endpoints. Floating division/subtraction can erase the
// remainder of 2^62 degrees. Decode the significand instead; integer remainders cannot be
// reassociated by Metal. 104 is the largest shift (exponent 127 minus 23 significand bits).
fn geodesicReduceLongitude(longitude: f32) -> f32 {
  if (abs(longitude) <= 180.0) { return longitude; }
  let bits = bitcast<u32>(longitude);
  let exponentBits = (bits >> 23u) & 255u;
  if (exponentBits == 255u) { return bitcast<f32>(0x7fc00000u | (bits & 0u)); }
  let exponent = i32(exponentBits) - 127;
  let significand = (bits & 0x7fffffu) | 0x800000u;
  var remainder = 0.0;
  if (exponent < 23) {
    let denominator = 1u << u32(23 - exponent);
    remainder = f32(significand % (720u * denominator)) / f32(denominator);
  } else {
    var integerRemainder = significand % 720u;
    for (var shift = 0u; shift < 104u; shift++) {
      if (shift >= u32(exponent - 23)) { break; }
      integerRemainder = (integerRemainder * 2u) % 720u;
    }
    remainder = f32(integerRemainder);
  }
  // Retain modulo-720 turn parity for round-to-even half-turn ties, like the core CPU helper.
  return select(remainder, -remainder, (bits >> 31u) != 0u);
}

// Canonicalize a modulo-720 endpoint with integer units. Floating reduced - 360*turns
// could be reassociated into the caller's subtraction, losing a tiny trip from 360 degrees again.
fn geodesicCanonicalLongitude(reduced: f32) -> f32 {
  if (abs(reduced) <= 180.0) { return reduced; }
  let bits = bitcast<u32>(reduced);
  let exponentBits = (bits >> 23u) & 255u;
  if (exponentBits == 255u) { return bitcast<f32>(0x7fc00000u | (bits & 0u)); }
  let denominator = 1u << (150u - exponentBits);
  let significand = (bits & 0x7fffffu) | 0x800000u;
  let turnUnits = 360u * denominator;
  let halfTurnUnits = 180u * denominator;
  var turns = significand / turnUnits;
  let remainder = significand % turnUnits;
  if (remainder > halfTurnUnits || (remainder == halfTurnUnits && (turns % 2u) != 0u)) {
    turns += 1u;
  }
  let canonical = f32(i32(significand) - i32(turns * turnUnits)) / f32(denominator);
  return select(canonical, -canonical, (bits >> 31u) != 0u);
}

// Subtract near the shared antimeridian before forming the small angular separation.
// Wrapping (destination - source) afterward cannot recover bits lost by that large subtraction:
// 179.341 -> -179.877 loses about 1.6 meters on the OpenFlights sphere fixture.
fn geodesicLongitudeDelta(source: f32, destination: f32) -> f32 {
  let reducedSource = geodesicReduceLongitude(source);
  let reducedDestination = geodesicReduceLongitude(destination);
  let canonicalSource = geodesicCanonicalLongitude(reducedSource);
  let canonicalDestination = geodesicCanonicalLongitude(reducedDestination);
  // These differences are exact whole turns; avoid backend-dependent round(+-0.5) ties.
  let sourceTurns = i32((reducedSource - canonicalSource) / 360.0);
  let destinationTurns = i32((reducedDestination - canonicalDestination) / 360.0);
  // WGSL permits floating-point reassociation: the two seam-relative subtractions below
  // can become (destination - source) - 2 * halfTurn, restoring the cancellation.
  // For opposite seam quadrants, every f32 endpoint is exactly a multiple of 2^-17
  // degrees (the smallest ulp on [90, 180]). Scale by that power of two and wrap in
  // i32: magnitudes stay below 2^26, with no quantization or integer overflow. Only
  // the final, small difference is converted back to f32, so reassociation cannot
  // reintroduce a large floating-point subtraction.
  if (abs(canonicalSource) > 90.0 && abs(canonicalSource) <= 180.0 &&
      abs(canonicalDestination) > 90.0 && abs(canonicalDestination) <= 180.0 &&
      ((canonicalSource < 0.0) != (canonicalDestination < 0.0))) {
    let sourceUnits = i32(canonicalSource * 131072.0);
    let destinationUnits = i32(canonicalDestination * 131072.0);
    let turnUnits = select(-47185920, 47185920, canonicalSource > 0.0);
    return f32(destinationUnits - sourceUnits + turnUnits) * 0.00000762939453125;
  }
  // Canonicalize before subtraction: (tiny + 180) - 180 loses a sub-ulp target from a
  // 360-degree origin. Only exact half-turn ties depend on the removed whole-turn parity.
  let difference = canonicalDestination - canonicalSource;
  if (abs(difference) == 180.0) {
    let oddTurns = ((destinationTurns - sourceTurns) % 2) != 0;
    return select(difference, -difference, oddTurns);
  }
  return geodesicWrapLongitudeDelta(difference);
}

// cos(latitude) as sin(colatitude): 90 - |lat| is exact in f32, so the cosine keeps its relative
// precision near the poles where cos(lat * pi / 180) would not.
fn geodesicCosLatitude(latitudeDegrees: f32) -> f32 {
  return sin((90.0 - abs(latitudeDegrees)) * GEODESIC_DEGREES_TO_RADIANS);
}

fn geodesicUnitVector(lngLat: vec2<f32>) -> vec3<f32> {
  let lambda = geodesicCanonicalLongitude(geodesicReduceLongitude(lngLat.x)) * GEODESIC_DEGREES_TO_RADIANS;
  let phi = lngLat.y * GEODESIC_DEGREES_TO_RADIANS;
  let cosPhi = geodesicCosLatitude(lngLat.y);
  return vec3<f32>(cosPhi * cos(lambda), cosPhi * sin(lambda), sin(phi));
}

fn geodesicFromUnitVector(vector: vec3<f32>) -> vec2<f32> {
  let horizontal = sqrt(vector.x * vector.x + vector.y * vector.y);
  return vec2<f32>(
    geodesicAtan2(vector.y, vector.x) * GEODESIC_RADIANS_TO_DEGREES,
    geodesicAtan2(vector.z, horizontal) * GEODESIC_RADIANS_TO_DEGREES
  );
}

fn geodesicHaversineTerm(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicLongitudeDelta(a.x, b.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfPhi = sin(0.5 * deltaPhi);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let term = sinHalfPhi * sinHalfPhi +
    geodesicCosLatitude(a.y) * geodesicCosLatitude(b.y) *
    sinHalfLambda * sinHalfLambda;
  return clamp(term, 0.0, 1.0);
}

// East, north and up components of b in the local frame at a. Near an antipode,
// use the latitude sum and longitude supplement to avoid subtracting order-one terms.
fn geodesicBearingComponents(origin: vec2<f32>, targetPosition: vec2<f32>) -> vec3<f32> {
  if (origin.y == targetPosition.y && (geodesicLongitudeDelta(origin.x, targetPosition.x) == 0.0 || abs(origin.y) == 90.0)) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  let latitudeRadians = origin.y * GEODESIC_DEGREES_TO_RADIANS;
  let targetLatitudeRadians = targetPosition.y * GEODESIC_DEGREES_TO_RADIANS;
  let longitudeDeltaDegrees = geodesicLongitudeDelta(origin.x, targetPosition.x);
  let longitudeDeltaRadians = longitudeDeltaDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let originCosine = geodesicCosLatitude(origin.y);
  let targetCosine = geodesicCosLatitude(targetPosition.y);
  let halfSineLongitude = sin(0.5 * longitudeDeltaRadians);
  let halfVersine = 2.0 * halfSineLongitude * halfSineLongitude;
  var east = targetCosine * sin(longitudeDeltaRadians);
  var north = sin((targetPosition.y - origin.y) * GEODESIC_DEGREES_TO_RADIANS) +
    sin(latitudeRadians) * targetCosine * halfVersine;
  var longitudeCosine = 1.0 - halfVersine;
  // A small longitude supplement does not imply a near antipode: two points at
  // the same pole can have opposite longitudes. Their latitude sum is near 180
  // degrees, where taking its sine loses the colatitude's relative precision.
  if (abs(longitudeDeltaDegrees) > 90.0 && origin.y * targetPosition.y <= 0.0) {
    let supplement = (180.0 - abs(longitudeDeltaDegrees)) * GEODESIC_DEGREES_TO_RADIANS;
    let halfSine = sin(0.5 * supplement);
    east = targetCosine * sign(longitudeDeltaDegrees) * sin(supplement);
    north = sin((origin.y + targetPosition.y) * GEODESIC_DEGREES_TO_RADIANS) -
      2.0 * sin(latitudeRadians) * targetCosine * halfSine * halfSine;
    longitudeCosine = 2.0 * halfSine * halfSine - 1.0;
  }
  let vertical = sin(latitudeRadians) * sin(targetLatitudeRadians) + originCosine * targetCosine * longitudeCosine;
  return vec3<f32>(east, north, vertical);
}

fn geodesicCentralAngle(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let components = geodesicBearingComponents(a, b);
  return geodesicAtan2(sqrt(components.x * components.x + components.y * components.y), components.z);
}

fn geodesicInitialBearingDegrees(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let components = geodesicBearingComponents(a, b);
  return geodesicAtan2(components.x, components.y) * GEODESIC_RADIANS_TO_DEGREES;
}

fn geodesicDestination(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> vec2<f32> {
  if (angle == 0.0) { return origin; }
  let phi1 = origin.y * GEODESIC_DEGREES_TO_RADIANS;
  let theta = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi1 = sin(phi1);
  let cosPhi1 = geodesicCosLatitude(origin.y);
  let sinHalfAngle = sin(0.5 * angle);
  // sin(phi2) - sin(phi1), from the half-angle form so short moves keep their precision.
  let sinDelta = -2.0 * sinPhi1 * sinHalfAngle * sinHalfAngle + cosPhi1 * sin(angle) * cos(theta);
  let sinPhi2 = clamp(sinPhi1 + sinDelta, -1.0, 1.0);
  // The half-angle change below preserves ordinary short moves. At a polar
  // origin or arrival (within 10 degrees of a pole), use components instead of asin.
  // Long paths also cancel in sinPhi1 + sinDelta and in the longitude denominator.
  // Preserve the established half-angle change only for short ordinary paths.
  if (abs(angle) >= GEODESIC_DIRECT_COMPONENT_ANGLE || abs(origin.y) >= GEODESIC_DIRECT_POLAR_LATITUDE || abs(sinPhi2) >= GEODESIC_DIRECT_POLAR_SINE) {
    let east = sin(theta) * sin(angle);
    let north = cosPhi1 * cos(angle) - sinPhi1 * sin(angle) * cos(theta);
    let vertical = sinPhi1 * cos(angle) + cosPhi1 * sin(angle) * cos(theta);
    let horizontal = sqrt(east * east + north * north);
    // Above 45 degrees, recover the latitude from the colatitude: atan2(vertical, horizontal)
    // returns a value near pi/2 whose absolute error (about a meter in f32) would swamp short
    // polar moves, while the colatitude keeps the horizontal component's relative precision.
    var latitudeDegrees = geodesicAtan2(vertical, horizontal) * GEODESIC_RADIANS_TO_DEGREES;
    if (abs(vertical) > horizontal) {
      latitudeDegrees = sign(vertical) * (90.0 - geodesicAtan2(horizontal, abs(vertical)) * GEODESIC_RADIANS_TO_DEGREES);
    }
    // A short move from a polar origin still takes the half-angle latitude change, added in
    // degrees: converting a polar latitude to f32 radians alone costs about 0.4 m, which the
    // component form inherits. The midpoint cosine guard hands the last 0.06 degrees to the
    // colatitude.
    if (abs(angle) < GEODESIC_DIRECT_COMPONENT_ANGLE) {
      let midpointCosine = cos(0.5 * (phi1 + asin(sinPhi2)));
      if (abs(midpointCosine) > 1e-3) {
        latitudeDegrees = origin.y + 2.0 * asin(clamp(sinDelta / (2.0 * midpointCosine), -1.0, 1.0)) * GEODESIC_RADIANS_TO_DEGREES;
      }
    }
    return vec2<f32>(
      origin.x + geodesicAtan2(east, north) * GEODESIC_RADIANS_TO_DEGREES,
      latitudeDegrees
    );
  }
  let roughPhi2 = asin(sinPhi2);
  // sin(phi2) - sin(phi1) = 2 cos((phi1 + phi2) / 2) sin((phi2 - phi1) / 2); the midpoint cosine
  // is insensitive to the rough phi2, so the latitude change is recovered to relative precision.
  let midpointCosine = cos(0.5 * (phi1 + roughPhi2));
  var deltaPhi = roughPhi2 - phi1;
  if (abs(midpointCosine) > 1e-3) {
    deltaPhi = 2.0 * asin(clamp(sinDelta / (2.0 * midpointCosine), -1.0, 1.0));
  }
  let deltaLambda = geodesicAtan2(
    sin(theta) * sin(angle) * cosPhi1,
    cos(angle) - sinPhi1 * sinPhi2
  );
  return vec2<f32>(
    origin.x + deltaLambda * GEODESIC_RADIANS_TO_DEGREES,
    origin.y + deltaPhi * GEODESIC_RADIANS_TO_DEGREES
  );
}

// Apply a whole longitude turn to the known target, never reconstruct it as source + delta:
// the latter loses a small target longitude when the endpoints have very different magnitudes.
fn geodesicUnwrapTarget(source: vec2<f32>, destination: vec2<f32>) -> vec2<f32> {
  if (abs(destination.x - source.x) <= 180.0) { return destination; }
  let turn = round((destination.x - source.x) / 360.0);
  return vec2<f32>(destination.x - 360.0 * turn, destination.y);
}

fn geodesicInterpolate(a: vec2<f32>, b: vec2<f32>, angle: f32, fraction: f32) -> vec2<f32> {
  // Endpoints are known even when the interior path is nonunique or ill-conditioned.
  if (fraction == 0.0) { return a; }
  if (fraction == 1.0) { return geodesicUnwrapTarget(a, b); }
  if (GEODESIC_PI - angle < 1e-6) {
    return vec2<f32>(bitcast<f32>(0x7fc00000u | (bitcast<u32>(a.x) & 0u)));
  }
  let unitA = geodesicUnitVector(a);
  let unitB = geodesicUnitVector(b);
  var vector: vec3<f32>;
  if (angle < 1e-6) {
    vector = normalize(mix(unitA, unitB, fraction));
  } else {
    let sinAngle = sin(angle);
    vector = unitA * (sin((1.0 - fraction) * angle) / sinAngle) +
      unitB * (sin(fraction * angle) / sinAngle);
  }
  let lngLat = geodesicFromUnitVector(vector);
  // Float32 sin(pi) may choose the opposite sign of the half-turn meridian
  // after crossing a pole. Align interiors to the known endpoint's branch,
  // rather than letting that transcendental rounding create a 360-degree jump.
  let endpoint = geodesicUnwrapTarget(a, b);
  if (abs(geodesicLongitudeDelta(a.x, b.x)) == 180.0) {
    let branch = a.x + (endpoint.x - a.x) * fraction;
    return vec2<f32>(branch + geodesicWrapLongitudeDelta(lngLat.x - branch), lngLat.y);
  }
  // Keep ordinary longitudes continuous with the start (unwrapped across the antimeridian).
  return vec2<f32>(a.x + geodesicWrapLongitudeDelta(lngLat.x - a.x), lngLat.y);
}

fn geodesicSinLatitudeDelta(latitudeDegrees: f32, originLatitudeDegrees: f32) -> f32 {
  let halfSum = 0.5 * (latitudeDegrees + originLatitudeDegrees) * GEODESIC_DEGREES_TO_RADIANS;
  let halfDifference = 0.5 * (latitudeDegrees - originLatitudeDegrees) * GEODESIC_DEGREES_TO_RADIANS;
  return 2.0 * cos(halfSum) * sin(halfDifference);
}

fn geodesicHaversineDistance(a: vec2<f32>, b: vec2<f32>, radius: f32) -> f32 {
  return radius * geodesicCentralAngle(a, b);
}

// Offset of position from origin in the local equirectangular frame at origin, in degrees of
// latitude (east offsets scaled by cos(latitude of origin), longitudes wrapped at +-180).
fn geodesicLocalOffset(origin: vec2<f32>, position: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    geodesicLongitudeDelta(origin.x, position.x) * geodesicCosLatitude(origin.y),
    position.y - origin.y
  );
}

// Haversine term at a supplied segment fraction, using coordinate differences to preserve short
// offsets near the seam. Endpoint terms use the same haversine arithmetic as direct candidates.
fn geodesicSegmentHaversineTerm(
  point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32
) -> f32 {
  if (fraction == 0.0) { return geodesicHaversineTerm(point, start); }
  if (fraction == 1.0) { return geodesicHaversineTerm(point, end); }
  let startOffset = vec2<f32>(geodesicLongitudeDelta(point.x, start.x), start.y - point.y);
  let delta = vec2<f32>(geodesicLongitudeDelta(start.x, end.x), end.y - start.y);
  let offset = startOffset + fraction * delta;
  let sinHalfLatitude = sin(0.5 * offset.y * GEODESIC_DEGREES_TO_RADIANS);
  let sinHalfLongitude = sin(0.5 * offset.x * GEODESIC_DEGREES_TO_RADIANS);
  let term = sinHalfLatitude * sinHalfLatitude +
    geodesicCosLatitude(point.y) * geodesicCosLatitude(point.y + offset.y) *
    sinHalfLongitude * sinHalfLongitude;
  return clamp(term, 0.0, 1.0);
}

// Fraction in [0, 1] of the point of segment start-end closest to point, in the local
// equirectangular frame at point: exact to first order for segments of road length.
fn geodesicSegmentFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  // Pin coincident endpoints before projection: contracting a scaled offset subtraction
  // into an FMA can leave a nonzero residual even when both endpoints are identical.
  if (all(start == end)) {
    return 0.0;
  }
  let startOffset = geodesicLocalOffset(point, start);
  // Project the same wrapped start-to-end arc that distance evaluation interpolates. Wrapping
  // each endpoint relative to the query independently can instead project the opposite arc.
  let segment = vec2<f32>(
    geodesicLongitudeDelta(start.x, end.x) * geodesicCosLatitude(point.y), end.y - start.y
  );
  let denominator = dot(segment, segment);
  var fraction = 0.0;
  if (denominator > 0.0) {
    fraction = clamp(-dot(startOffset, segment) / denominator, 0.0, 1.0);
  }
  // The local projection is an approximation on the sphere. Always retain both endpoints as
  // candidates, and return their fraction as well as their distance to snapping consumers.
  let projectedTerm = geodesicSegmentHaversineTerm(point, start, end, fraction);
  let startTerm = geodesicHaversineTerm(point, start);
  let endTerm = geodesicHaversineTerm(point, end);
  if (max(projectedTerm, max(startTerm, endTerm)) < GEODESIC_LOCAL_COMPARISON_MAXIMUM_TERM) {
    // Comparing an asin projection with atan2 endpoints can manufacture a one-ulp preference
    // for an endpoint. On rounded local ties retain the interior position, which is still distinct.
    if (fraction == 0.0 || fraction == 1.0 || min(startTerm, endTerm) < projectedTerm) {
      return select(1.0, 0.0, startTerm <= endTerm);
    }
    return fraction;
  }
  // Central angles retain endpoint resolution near antipodes where haversine terms approach 1.
  let projectedDistance = geodesicSegmentDistance(point, start, end, fraction, 1.0);
  let startDistance = geodesicHaversineDistance(point, start, 1.0);
  let endDistance = geodesicHaversineDistance(point, end, 1.0);
  // Keep an interior projection on rounded distance ties: its snap position still matters.
  if (fraction == 0.0 || fraction == 1.0 || min(startDistance, endDistance) < projectedDistance) {
    return select(1.0, 0.0, startDistance <= endDistance);
  }
  return fraction;
}

// Great-circle distance (units of radius) from point to the position at fraction along the
// segment, built from coordinate differences so short distances keep their precision near +-180.
fn geodesicSegmentDistance(
  point: vec2<f32>,
  start: vec2<f32>,
  end: vec2<f32>,
  fraction: f32,
  radius: f32
) -> f32 {
  // Preserve exact endpoint distances without reconstructing their coordinates from offsets.
  // Other fractions retain the distance-at-fraction contract (including extrapolation).
  if (fraction == 0.0) { return geodesicHaversineDistance(point, start, radius); }
  if (fraction == 1.0) { return geodesicHaversineDistance(point, end, radius); }
  // Use the same seam-preserving differences as edge lengths and local projection. Mixing
  // rounded seam offsets with precise edge lengths biases map-matching likelihoods.
  return 2.0 * radius * asin(sqrt(geodesicSegmentHaversineTerm(point, start, end, fraction)));
}
`;
