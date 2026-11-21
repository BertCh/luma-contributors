// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Power-of-two edge normalization shared by circumcircles and outward hull normals. @internal */
export const TRIANGLE_SCALE_WGSL = /* wgsl */ `
fn scaleTriangleComponent(value: f32, exponent: i32) -> f32 {
  // Split the power so even the inverse of an exponent-128 extent uses ordinary ldexp exponents.
  let firstExponent = exponent / 2i;
  return ldexp(ldexp(value, firstExponent), exponent - firstExponent);
}
fn scaleTrianglePosition(value: vec2f, exponent: i32) -> vec2f {
  return vec2f(scaleTriangleComponent(value.x, exponent), scaleTriangleComponent(value.y, exponent));
}
struct ScaledTriangleEdge { direction: vec2f, exponent: i32 }
fn getScaledTriangleEdge(first: vec2f, second: vec2f) -> ScaledTriangleEdge {
  var edge = second - first;
  var exponent = 0i;
  if (!(abs(edge.x) <= 3.402823466e38 && abs(edge.y) <= 3.402823466e38)) {
    exponent = frexp(max(max(abs(first.x), abs(first.y)), max(abs(second.x), abs(second.y)))).exp;
    edge = scaleTrianglePosition(second, -exponent) - scaleTrianglePosition(first, -exponent);
  }
  let extent = max(abs(edge.x), abs(edge.y));
  if (extent == 0.0) { return ScaledTriangleEdge(vec2f(0.0), 0i); }
  let relativeExponent = frexp(extent).exp;
  return ScaledTriangleEdge(scaleTrianglePosition(edge, -relativeExponent), exponent + relativeExponent);
}
`;

/** Independently scaled edge equations; requires EXACT_ORIENTATION_WGSL. @internal */
export const TRIANGLE_CIRCUMCIRCLE_WGSL = /* wgsl */ `
${TRIANGLE_SCALE_WGSL}
struct TriangleCircumcircle {
  center: vec2f,
  radius: f32,
  orientation: i32,
}
fn getTriangleAlignedTerm(fraction: f32, exponent: i32) -> f32 {
  // Terms below the f32 significand range cannot contribute in this frame. Avoid invalid ldexp powers.
  if (exponent < -149i) { return 0.0; }
  return scaleTriangleComponent(fraction, exponent);
}
fn sumTriangleCoordinate(origin: f32, first: f32, firstExponent: i32, second: f32, secondExponent: i32) -> f32 {
  let originParts = frexp(origin);
  let firstParts = frexp(first);
  let secondParts = frexp(second);
  let originPower = select(-1000i, originParts.exp, origin != 0.0);
  let firstPower = select(-1000i, firstParts.exp + firstExponent, first != 0.0);
  let secondPower = select(-1000i, secondParts.exp + secondExponent, second != 0.0);
  let power = max(originPower, max(firstPower, secondPower));
  if (power == -1000i) { return 0.0; }
  let offset = getTriangleAlignedTerm(firstParts.fract, firstPower - power) + getTriangleAlignedTerm(secondParts.fract, secondPower - power);
  let centered = offset + getTriangleAlignedTerm(originParts.fract, originPower - power);
  // Add the origin before restoration: an unrepresentable offset can still give a finite center.
  return scaleTriangleComponent(centered, power);
}
// Exact product of two f32 values: (high * 2^32 + low) * 2^exponent, sign 0 (positive), 1 (negative)
// or 2 (zero).
struct TriangleProductPart { low: u32, high: u32, exponent: i32, sign: u32 }
fn getTriangleProductPart(first: f32, second: f32) -> TriangleProductPart {
  let firstBits = bitcast<u32>(first);
  let secondBits = bitcast<u32>(second);
  let firstMagnitude = firstBits & 0x7fffffffu;
  let secondMagnitude = secondBits & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u) {
    return TriangleProductPart(0u, 0u, 0i, 2u);
  }
  let firstExponentBits = firstMagnitude >> 23u;
  let secondExponentBits = secondMagnitude >> 23u;
  let firstSignificand = select(0x800000u | (firstMagnitude & 0x7fffffu), firstMagnitude & 0x7fffffu, firstExponentBits == 0u);
  let secondSignificand = select(0x800000u | (secondMagnitude & 0x7fffffu), secondMagnitude & 0x7fffffu, secondExponentBits == 0u);
  let firstExponent = select(i32(firstExponentBits) - 150i, -149i, firstExponentBits == 0u);
  let secondExponent = select(i32(secondExponentBits) - 150i, -149i, secondExponentBits == 0u);
  let product = multiply24(firstSignificand, secondSignificand);
  return TriangleProductPart(product.x, product.y, firstExponent + secondExponent, (firstBits ^ secondBits) >> 31u);
}
// first * second - third * fourth for finite scaled edge components whose rounded products cancel:
// the 48-bit integer products are subtracted exactly and the difference is rounded once (to a few
// units in the last place). Float error-free transforms are not used because compilers reassociate
// them. Products of opposite sign, or more than 16 binary orders apart, do not cancel and keep the
// float expression.
fn getTriangleProductDifference(first: f32, second: f32, third: f32, fourth: f32) -> f32 {
  let rounded = first * second - third * fourth;
  let left = getTriangleProductPart(first, second);
  let right = getTriangleProductPart(third, fourth);
  if (left.sign == 2u || right.sign == 2u || left.sign != right.sign) { return rounded; }
  let shift = left.exponent - right.exponent;
  if (shift > 16i || shift < -16i) { return rounded; }
  let exponent = min(left.exponent, right.exponent);
  if (exponent < -250i || exponent > 100i) { return rounded; }
  // Align the operand with the larger exponent onto the smaller one: at most 48 + 16 bits.
  var leftLow = left.low;
  var leftHigh = left.high;
  var rightLow = right.low;
  var rightHigh = right.high;
  if (shift > 0i) {
    let bits = u32(shift);
    leftHigh = (leftHigh << bits) | (leftLow >> (32u - bits));
    leftLow = leftLow << bits;
  } else if (shift < 0i) {
    let bits = u32(-shift);
    rightHigh = (rightHigh << bits) | (rightLow >> (32u - bits));
    rightLow = rightLow << bits;
  }
  let leftLarger = leftHigh > rightHigh || (leftHigh == rightHigh && leftLow >= rightLow);
  let largeLow = select(rightLow, leftLow, leftLarger);
  let largeHigh = select(rightHigh, leftHigh, leftLarger);
  let smallLow = select(leftLow, rightLow, leftLarger);
  let smallHigh = select(leftHigh, rightHigh, leftLarger);
  let differenceLow = largeLow - smallLow;
  let differenceHigh = largeHigh - smallHigh - select(0u, 1u, largeLow < smallLow);
  let magnitude = scaleTriangleComponent(f32(differenceHigh) * 4294967296.0 + f32(differenceLow), exponent);
  // left - right has the common sign when left is larger, the opposite sign otherwise.
  let negativeResult = select(!leftLarger, leftLarger, left.sign == 1u);
  return select(magnitude, -magnitude, negativeResult);
}
fn getTriangleCircumcircle(first: vec2f, second: vec2f, third: vec2f) -> TriangleCircumcircle {
  let orientation = orientSign(first, second, third);
  if (orientation == 0 || orientation == 2) {
    return TriangleCircumcircle(vec2f(0.0), 0.0, 0);
  }
  var origin = first;
  var firstScaled = getScaledTriangleEdge(first, second);
  var secondScaled = getScaledTriangleEdge(first, third);
  var firstEdge = firstScaled.direction;
  var secondEdge = secondScaled.direction;
  let leftProduct = firstEdge.x * secondEdge.y;
  let rightProduct = firstEdge.y * secondEdge.x;
  var cross = leftProduct - rightProduct;
  let constructionCancellation = abs(cross) <= 0.0009765625 * (abs(leftProduct) + abs(rightProduct));
  // A thin triangle's separately rounded products can cancel its nonzero cross product (to zero
  // for (0,0), (8192,8191), (8191,8190)); recompute it exactly when the cancellation exceeds 2^-10.
  if (abs(cross) <= 0.0009765625 * (abs(leftProduct) + abs(rightProduct))) {
    cross = getTriangleProductDifference(firstEdge.x, secondEdge.y, firstEdge.y, secondEdge.x);
  }
  if (constructionCancellation) {
    // Edge subtraction can compromise a nonzero determinant too. Start with the edge whose
    // maximum component magnitude is smallest, retaining the separation of nearby vertices.
    // Product correction alone cannot recover information already lost from the edge directions.
    let thirdScaled = getScaledTriangleEdge(second, third);
    let firstExtent = max(abs(firstScaled.direction.x), abs(firstScaled.direction.y));
    let secondExtent = max(abs(secondScaled.direction.x), abs(secondScaled.direction.y));
    let thirdExtent = max(abs(thirdScaled.direction.x), abs(thirdScaled.direction.y));
    let thirdShorterThanFirst = thirdScaled.exponent < firstScaled.exponent ||
      (thirdScaled.exponent == firstScaled.exponent && thirdExtent < firstExtent);
    let thirdShorterThanSecond = thirdScaled.exponent < secondScaled.exponent ||
      (thirdScaled.exponent == secondScaled.exponent && thirdExtent < secondExtent);
    if (thirdShorterThanFirst && thirdShorterThanSecond) {
      origin = second;
      firstScaled = thirdScaled;
      secondScaled = getScaledTriangleEdge(second, first);
    } else if (secondScaled.exponent < firstScaled.exponent ||
      (secondScaled.exponent == firstScaled.exponent && secondExtent < firstExtent)) {
      origin = third;
      firstScaled = getScaledTriangleEdge(third, first);
      secondScaled = getScaledTriangleEdge(third, second);
    }
    firstEdge = firstScaled.direction;
    secondEdge = secondScaled.direction;
    cross = getTriangleProductDifference(firstEdge.x, secondEdge.y, firstEdge.y, secondEdge.x);
  }
  let denominator = 2.0 * cross;
  if (denominator == 0.0) { return TriangleCircumcircle(vec2f(0.0), 0.0, 0); }
  let firstLength = dot(firstEdge, firstEdge);
  let secondLength = dot(secondEdge, secondEdge);
  let firstTerm = vec2f(secondEdge.y, -secondEdge.x) * (firstLength / denominator);
  let secondTerm = vec2f(-firstEdge.y, firstEdge.x) * (secondLength / denominator);
  let offset = vec2f(
    sumTriangleCoordinate(0.0, firstTerm.x, firstScaled.exponent, secondTerm.x, secondScaled.exponent),
    sumTriangleCoordinate(0.0, firstTerm.y, firstScaled.exponent, secondTerm.y, secondScaled.exponent));
  let center = vec2f(
    sumTriangleCoordinate(origin.x, firstTerm.x, firstScaled.exponent, secondTerm.x, secondScaled.exponent),
    sumTriangleCoordinate(origin.y, firstTerm.y, firstScaled.exponent, secondTerm.y, secondScaled.exponent));
  let offsetExtent = max(abs(offset.x), abs(offset.y));
  // Radius overflow does not invalidate a representable Voronoi center.
  var radius = offsetExtent;
  if (offsetExtent > 0.0 && orientIsFinite(offsetExtent)) { radius *= length(offset / offsetExtent); }
  return TriangleCircumcircle(center, radius, orientation);
}
`;
