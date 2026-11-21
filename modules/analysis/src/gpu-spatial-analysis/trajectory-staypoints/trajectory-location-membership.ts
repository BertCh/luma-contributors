// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
/** Exact integer coordinate in units of 2^-149, the smallest binary32 subnormal. */
function getLocationCoordinateInteger(value: number): bigint {
  const words = new Uint32Array(new Float32Array([value]).buffer);
  const bits = words[0];
  const exponent = (bits >>> 23) & 255;
  const significand = (bits & 0x7fffff) | (exponent === 0 ? 0 : 0x800000);
  return (bits >>> 31 ? -1n : 1n) * (BigInt(significand) << BigInt(Math.max(exponent - 1, 0)));
}
/** Inclusive squared-distance predicate on stored binary32 coordinates. @internal */
export function isTrajectoryLocationWithin(
  firstX: number,
  firstY: number,
  secondX: number,
  secondY: number,
  radius: number
): boolean {
  if (![firstX, firstY, secondX, secondY, radius].every(Number.isFinite) || radius < 0)
    return false;
  const horizontal = firstX - secondX;
  const vertical = firstY - secondY;
  const squaredDistance = horizontal * horizontal + vertical * vertical;
  const squaredRadius = radius * radius;
  const difference = squaredDistance - squaredRadius;
  if (Math.abs(difference) > 1e-12 * Math.max(squaredDistance, squaredRadius))
    return difference < 0;
  const exactHorizontal =
    getLocationCoordinateInteger(firstX) - getLocationCoordinateInteger(secondX);
  const exactVertical =
    getLocationCoordinateInteger(firstY) - getLocationCoordinateInteger(secondY);
  const exactRadius = getLocationCoordinateInteger(radius);
  return exactHorizontal ** 2n + exactVertical ** 2n <= exactRadius ** 2n;
}

/**
 * Ambiguous comparisons expand (a-b)^2 without first rounding the subtraction. Seven signed
 * raw-coordinate products are accumulated as exact integers in two 576-bit sums. Every finite
 * f32 coordinate fits: the largest squared difference has fewer than 559 bits in units 2^-298.
 * The fixed 18-word arrays are 72 bytes each; integer operations survive Metal reassociation.
 * @internal
 */
export const TRAJECTORY_LOCATION_MEMBERSHIP_WGSL = `
fn locationMultiply(first: u32, second: u32) -> vec2<u32> {
  let firstHigh = first >> 12u; let firstLow = first & 4095u;
  let secondHigh = second >> 12u; let secondLow = second & 4095u;
  let middle = firstHigh * secondLow + firstLow * secondHigh;
  let lower = firstLow * secondLow + ((middle & 4095u) << 12u);
  let upper = firstHigh * secondHigh + (middle >> 12u) + (lower >> 24u);
  return vec2<u32>((lower & 0xffffffu) | (upper << 24u), upper >> 8u);
}
fn withinRadius(firstPoint: vec2<f32>, secondPoint: vec2<f32>, radius: f32) -> bool {
  if (radius < 0.0 || (bitcast<u32>(radius) & 0x7f800000u) == 0x7f800000u ||
      (bitcast<u32>(firstPoint.x) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(firstPoint.y) & 0x7f800000u) == 0x7f800000u ||
      (bitcast<u32>(secondPoint.x) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(secondPoint.y) & 0x7f800000u) == 0x7f800000u) { return false; }
  let offset = abs(firstPoint-secondPoint);
  let exponent = clamp(i32((bitcast<u32>(radius) >> 23u) & 255u)-127, -126, 126);
  let scale = bitcast<f32>(u32(127-exponent) << 23u);
  let normalized = offset*scale; let normalizedRadius = radius*scale;
  let difference = dot(normalized,normalized)-normalizedRadius*normalizedRadius;
  // Tiny-radius subtraction can flush to zero; compare original bits in that range.
  if ((bitcast<u32>(radius) & 0x7f800000u) > 0x00800000u && abs(difference)>0.00001) {
    return difference<0.0;
  }
  var positive: array<u32,18>; var negative: array<u32,18>;
  let firstTerms=array<f32,7>(firstPoint.x,secondPoint.x,firstPoint.x,firstPoint.y,secondPoint.y,firstPoint.y,radius);
  let secondTerms=array<f32,7>(firstPoint.x,secondPoint.x,secondPoint.x,firstPoint.y,secondPoint.y,secondPoint.y,radius);
  for(var term=0u;term<7u;term++) {
  let firstBits = bitcast<u32>(firstTerms[term]); let secondBits = bitcast<u32>(secondTerms[term]);
  let firstMagnitude = firstBits & 0x7fffffffu; let secondMagnitude = secondBits & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u) { continue; }
  let firstBiased = firstMagnitude >> 23u; let secondBiased = secondMagnitude >> 23u;
  let firstSignificand = (firstMagnitude & 0x7fffffu) | select(0u, 0x800000u, firstBiased != 0u);
  let secondSignificand = (secondMagnitude & 0x7fffffu) | select(0u, 0x800000u, secondBiased != 0u);
  let firstExponent = select(-149, i32(firstBiased) - 150, firstBiased != 0u);
  let secondExponent = select(-149, i32(secondBiased) - 150, secondBiased != 0u);
  let product = locationMultiply(firstSignificand, secondSignificand);
  let shift = u32(firstExponent + secondExponent + 298) + select(0u, 1u, term == 2u || term == 5u);
  let firstWord = shift / 32u; let bitShift = shift % 32u;
  var pieces: array<u32, 3>;
  pieces[0] = product.x; pieces[1] = product.y;
  if (bitShift != 0u) {
    pieces[0] = product.x << bitShift;
    pieces[1] = (product.y << bitShift) | (product.x >> (32u-bitShift));
    pieces[2] = product.y >> (32u-bitShift);
  }
  let isNegative = (((firstBits ^ secondBits) >> 31u) != 0u) != (term == 2u || term == 5u || term == 6u);
  var carry = 0u;
  for (var word = firstWord; word < 18u; word++) {
    var piece = 0u;
    if (word-firstWord < 3u) { piece = pieces[word-firstWord]; }
    var before = positive[word];
    if (isNegative) { before = negative[word]; }
    let added = before + piece; let total = added + carry;
    carry = select(0u, 1u, added < before || total < added);
    if (isNegative) { negative[word] = total; } else { positive[word] = total; }
  }
  }
  for (var remaining=18u; remaining>0u; remaining--) {
    let word=remaining-1u;
    if (positive[word] != negative[word]) { return positive[word]<negative[word]; }
  }
  return true;
}`;
