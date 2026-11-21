// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Metric companion to exact-orientation-wgsl.ts. Reuses its integer product/accumulation
 * helpers, subtracts the exact signed integer sums and converts the leading 24 bits to f32.
 * The conversion error is less than one f32 ulp; unlike a floating error-free transform,
 * integer subtraction survives native shader fast-math reassociation. Used only when the
 * ordinary determinant is within its cancellation bound. Out-of-range metrics return NaN.
 */
export const POLYGON_OVERLAP_METRIC_WGSL = /* wgsl */ `
fn overlapExactDeterminant(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  var invalidBits = 0x7fc00000u;
  let invalid = bitcast<f32>(invalidBits);
  var xBits = array<u32, 6>(
    bitcast<u32>(a.x), bitcast<u32>(a.y) ^ 0x80000000u, bitcast<u32>(b.x),
    bitcast<u32>(b.y) ^ 0x80000000u, bitcast<u32>(c.x), bitcast<u32>(c.y) ^ 0x80000000u);
  var yBits = array<u32, 6>(
    bitcast<u32>(b.y), bitcast<u32>(b.x), bitcast<u32>(c.y),
    bitcast<u32>(c.x), bitcast<u32>(a.y), bitcast<u32>(a.x));
  var exponents: array<i32, 6>;
  var lows: array<u32, 6>;
  var highs: array<u32, 6>;
  var negatives: array<bool, 6>;
  var present: array<bool, 6>;
  var minimumExponent = 100000;
  var maximumExponent = -100000;
  for (var term = 0u; term < 6u; term++) {
    let xMagnitude = xBits[term] & 0x7fffffffu;
    let yMagnitude = yBits[term] & 0x7fffffffu;
    let xExponentBits = xMagnitude >> 23u;
    let yExponentBits = yMagnitude >> 23u;
    let xFraction = xMagnitude & 0x7fffffu;
    let yFraction = yMagnitude & 0x7fffffu;
    if (xMagnitude == 0u || yMagnitude == 0u) { continue; }
    let xSignificand = select(0x800000u | xFraction, xFraction, xExponentBits == 0u);
    let ySignificand = select(0x800000u | yFraction, yFraction, yExponentBits == 0u);
    let xExponent = select(i32(xExponentBits) - 150, -149, xExponentBits == 0u);
    let yExponent = select(i32(yExponentBits) - 150, -149, yExponentBits == 0u);
    let product = multiply24(xSignificand, ySignificand);
    present[term] = true;
    lows[term] = product.x;
    highs[term] = product.y;
    exponents[term] = xExponent + yExponent;
    negatives[term] = ((xBits[term] ^ yBits[term]) >> 31u) == 1u;
    minimumExponent = min(minimumExponent, exponents[term]);
    maximumExponent = max(maximumExponent, exponents[term]);
  }
  if (minimumExponent > maximumExponent) { return 0.0; }
  if (maximumExponent - minimumExponent > ORIENT_MAXIMUM_SPAN) { return invalid; }
  var positive: array<u32, 8>;
  var negative: array<u32, 8>;
  for (var term = 0u; term < 6u; term++) {
    if (!present[term]) { continue; }
    let shift = u32(exponents[term] - minimumExponent);
    if (negatives[term]) {
      accumulateShifted(&negative, lows[term], highs[term], shift);
    } else {
      accumulateShifted(&positive, lows[term], highs[term], shift);
    }
  }
  var negativeResult = false;
  for (var slot = 8u; slot > 0u; slot--) {
    let plus = positive[slot - 1u]; let minus = negative[slot - 1u];
    if (plus != minus) { negativeResult = plus < minus; break; }
  }
  var borrow = 0u;
  var highest = 0u;
  var highestWord = 0u;
  var precedingWord = 0u;
  var lowerWord = 0u;
  var lowestBit = 0u;
  // Keep conversion inputs in scalars. A dynamic reload from the difference array can
  // lose the leading word under backend lowering even while retaining its nonzero index.
  for (var slot = 0u; slot < 8u; slot++) {
    let larger = select(positive[slot], negative[slot], negativeResult);
    let smaller = select(negative[slot], positive[slot], negativeResult);
    let differenceWord = larger - smaller - borrow;
    borrow = select(0u, 1u, larger < smaller || (larger == smaller && borrow != 0u));
    if (differenceWord != 0u) {
      if (highestWord == 0u) { lowestBit = slot * 32u + countTrailingZeros(differenceWord); }
      highest = slot;
      highestWord = differenceWord;
      lowerWord = precedingWord;
    }
    precedingWord = differenceWord;
  }
  if (highestWord == 0u) { return 0.0; }
  let highestBit = 31u - countLeadingZeros(highestWord);
  let leadingBit = highest * 32u + highestBit;
  let exponent = minimumExponent + i32(leadingBit);
  // Coordinates and constructed metric intersections must have normal finite determinants.
  if (exponent < -126 || exponent > 127) { return invalid; }
  var significand = 0u;
  if (highestBit >= 23u) { significand = highestWord >> (highestBit - 23u); }
  else {
    significand = (highestWord << (23u - highestBit)) | (lowerWord >> (9u + highestBit));
  }
  // A set bit below the largest finite significand makes the exact value too large.
  if (exponent == 127 && significand == 0xffffffu && leadingBit > 23u &&
      lowestBit < leadingBit - 23u) { return invalid; }
  let signBit = select(0u, 0x80000000u, negativeResult);
  return bitcast<f32>(signBit | (u32(exponent + 127) << 23u) | (significand & 0x7fffffu));
}

fn overlapDeterminant(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  let first = (a.x - c.x) * (b.y - c.y);
  let second = (a.y - c.y) * (b.x - c.x);
  let determinant = first - second;
  if (abs(determinant) > 3.0e-7 * (abs(first) + abs(second))) { return determinant; }
  return overlapExactDeterminant(a, b, c);
}
`;
