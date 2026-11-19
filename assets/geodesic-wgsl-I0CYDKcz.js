import{s as e}from"./wgsl-common-DfyXiYB0.js";var t=`
${e.replaceAll(`guardedAtan2`,`geodesicAtan2`)}
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

// Subtract near the shared antimeridian before forming the small angular separation.
// Wrapping (destination - source) afterward cannot recover bits lost by that large subtraction:
// 179.341 -> -179.877 loses about 1.6 meters on the OpenFlights sphere fixture.
fn geodesicLongitudeDelta(source: f32, destination: f32) -> f32 {
  // WGSL permits floating-point reassociation: the two seam-relative subtractions below
  // can become (destination - source) - 2 * halfTurn, restoring the cancellation.
  // For opposite seam quadrants, every f32 endpoint is exactly a multiple of 2^-17
  // degrees (the smallest ulp on [90, 180]). Scale by that power of two and wrap in
  // i32: magnitudes stay below 2^26, with no quantization or integer overflow. Only
  // the final, small difference is converted back to f32, so reassociation cannot
  // reintroduce a large floating-point subtraction.
  if (abs(source) > 90.0 && abs(source) <= 180.0 &&
      abs(destination) > 90.0 && abs(destination) <= 180.0 &&
      ((source < 0.0) != (destination < 0.0))) {
    let sourceUnits = i32(source * 131072.0);
    let destinationUnits = i32(destination * 131072.0);
    let turnUnits = select(-47185920, 47185920, source > 0.0);
    return f32(destinationUnits - sourceUnits + turnUnits) * 0.00000762939453125;
  }
  let halfTurn = 180.0 * round((destination - source) / 360.0);
  return (destination - halfTurn) - (source + halfTurn);
}

// cos(latitude) as sin(colatitude): 90 - |lat| is exact in f32, so the cosine keeps its relative
// precision near the poles where cos(lat * pi / 180) would not.
fn geodesicCosLatitude(latitudeDegrees: f32) -> f32 {
  return sin((90.0 - abs(latitudeDegrees)) * GEODESIC_DEGREES_TO_RADIANS);
}

fn geodesicUnitVector(lngLat: vec2<f32>) -> vec3<f32> {
  let lambda = lngLat.x * GEODESIC_DEGREES_TO_RADIANS;
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

fn geodesicInterpolate(a: vec2<f32>, b: vec2<f32>, angle: f32, fraction: f32) -> vec2<f32> {
  // Endpoints are known even when the interior path is nonunique or ill-conditioned.
  if (fraction == 0.0) { return a; }
  if (fraction == 1.0) {
    if (abs(b.x - a.x) <= 180.0) { return b; }
    return vec2<f32>(a.x + geodesicLongitudeDelta(a.x, b.x), b.y);
  }
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
  // Keep longitudes continuous with the start point (unwrapped across the antimeridian).
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
`;export{t};