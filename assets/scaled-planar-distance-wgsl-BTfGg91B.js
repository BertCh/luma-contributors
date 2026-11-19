var e=`
struct PlanarDistanceValue {
  significand: f32,
  exponent: i32
}
struct PlanarDistanceVector {
  x: PlanarDistanceValue,
  y: PlanarDistanceValue
}
// Smaller aligned terms cannot affect a normal f32 significand. Skipping them also avoids
// subnormal intermediates in ldexp: 0.5 * 2^-125 is the smallest normal f32 value.
const PLANAR_MINIMUM_ALIGNMENT_EXPONENT: i32 = -125;
// Split conversion at 2^126 so no intermediate power of two overflows for a normal result.
const PLANAR_MAXIMUM_CONVERSION_EXPONENT: i32 = 126;
fn getPlanarDistanceValue(significand: f32, exponent: i32) -> PlanarDistanceValue {
  if (significand == 0.0) { return PlanarDistanceValue(0.0, 0); }
  let parts = frexp(significand);
  return PlanarDistanceValue(parts.fract, exponent + parts.exp);
}
fn addPlanarDistanceValues(first: PlanarDistanceValue, second: PlanarDistanceValue) -> PlanarDistanceValue {
  if (first.significand == 0.0) { return second; }
  if (second.significand == 0.0) { return first; }
  let exponent = max(first.exponent, second.exponent);
  let firstShift = first.exponent - exponent;
  let secondShift = second.exponent - exponent;
  var firstTerm = 0.0;
  var secondTerm = 0.0;
  if (firstShift >= PLANAR_MINIMUM_ALIGNMENT_EXPONENT) {
    firstTerm = ldexp(first.significand, firstShift);
  }
  if (secondShift >= PLANAR_MINIMUM_ALIGNMENT_EXPONENT) {
    secondTerm = ldexp(second.significand, secondShift);
  }
  return getPlanarDistanceValue(firstTerm + secondTerm, exponent);
}
fn negatePlanarDistanceValue(value: PlanarDistanceValue) -> PlanarDistanceValue {
  return PlanarDistanceValue(-value.significand, value.exponent);
}
fn multiplyPlanarDistanceValues(first: PlanarDistanceValue, second: PlanarDistanceValue) -> PlanarDistanceValue {
  return getPlanarDistanceValue(first.significand * second.significand, first.exponent + second.exponent);
}
fn getPlanarDistanceDifference(first: f32, second: f32) -> PlanarDistanceValue {
  return addPlanarDistanceValues(getPlanarDistanceValue(first, 0), getPlanarDistanceValue(-second, 0));
}
fn getPlanarDistanceVector(first: vec2f, second: vec2f) -> PlanarDistanceVector {
  return PlanarDistanceVector(getPlanarDistanceDifference(first.x, second.x),
    getPlanarDistanceDifference(first.y, second.y));
}
fn dotPlanarDistanceVectors(first: PlanarDistanceVector, second: PlanarDistanceVector) -> PlanarDistanceValue {
  return addPlanarDistanceValues(multiplyPlanarDistanceValues(first.x, second.x),
    multiplyPlanarDistanceValues(first.y, second.y));
}
fn getPlanarDistanceVectorLength(value: PlanarDistanceVector) -> PlanarDistanceValue {
  let squared = dotPlanarDistanceVectors(value, value);
  if (squared.significand == 0.0) { return squared; }
  var significand = squared.significand;
  var exponent = squared.exponent;
  if ((exponent & 1) != 0) {
    significand *= 2.0;
    exponent -= 1;
  }
  return getPlanarDistanceValue(sqrt(significand), exponent / 2);
}
fn convertPlanarDistanceValue(value: PlanarDistanceValue) -> f32 {
  if (value.significand == 0.0 || value.exponent < -125) { return 0.0; }
  if (value.exponent > 128) {
    return bitcast<f32>(0x7f800000u | (bitcast<u32>(value.significand) & 0u));
  }
  let firstExponent = min(value.exponent, PLANAR_MAXIMUM_CONVERSION_EXPONENT);
  return ldexp(ldexp(value.significand, firstExponent), value.exponent - firstExponent);
}
fn getPlanarPointSegmentDistance(point: vec2f, start: vec2f, end: vec2f) -> f32 {
  let direction = getPlanarDistanceVector(end, start);
  let fromStart = getPlanarDistanceVector(point, start);
  let directionLength = getPlanarDistanceVectorLength(direction);
  if (directionLength.significand == 0.0 ||
      dotPlanarDistanceVectors(fromStart, direction).significand <= 0.0) {
    return convertPlanarDistanceValue(getPlanarDistanceVectorLength(fromStart));
  }
  let fromEnd = getPlanarDistanceVector(point, end);
  if (dotPlanarDistanceVectors(fromEnd, direction).significand >= 0.0) {
    return convertPlanarDistanceValue(getPlanarDistanceVectorLength(fromEnd));
  }
  let cross = addPlanarDistanceValues(multiplyPlanarDistanceValues(fromStart.x, direction.y),
    negatePlanarDistanceValue(multiplyPlanarDistanceValues(fromStart.y, direction.x)));
  let distance = getPlanarDistanceValue(abs(cross.significand) / directionLength.significand,
    cross.exponent - directionLength.exponent);
  return convertPlanarDistanceValue(distance);
}
`;export{e as t};