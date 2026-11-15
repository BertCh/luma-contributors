// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL helpers for exact 64-bit arithmetic on vec2<u32> pairs (x = low word, y = high word, the
 * little-endian element order of first uint32x2 GPUVector). Algorithms are mirrored one-to-one by the
 * TypeScript emulation in wide-numeric-gpu-reference.ts. 128-bit values are vec4<u32> (little-endian words).
 */
export const WIDE_NUMERIC_WGSL_HELPERS = /* wgsl */ `
struct Float32Conversion { value: f32, exact: u32 }
struct NarrowingClass { kind: u32, errorKey: u32, errorTop: u32, errorUnits: vec2<u32> }
struct MantissaRounding { kept: u32, roundUp: u32, inexact: u32, errorUnits: vec2<u32> }

fn word64Add(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  let low = first.x + second.x;
  return vec2<u32>(low, first.y + second.y + select(0u, 1u, low < first.x));
}
fn word64Subtract(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  let low = first.x - second.x;
  return vec2<u32>(low, first.y - second.y - select(0u, 1u, first.x < second.x));
}
fn word64Negate(first: vec2<u32>) -> vec2<u32> { return word64Subtract(vec2<u32>(0u, 0u), first); }
fn word64LessUnsigned(first: vec2<u32>, second: vec2<u32>) -> bool {
  return first.y < second.y || (first.y == second.y && first.x < second.x);
}
fn word64LessSigned(first: vec2<u32>, second: vec2<u32>) -> bool {
  return word64LessUnsigned(vec2<u32>(first.x, first.y ^ 0x80000000u), vec2<u32>(second.x, second.y ^ 0x80000000u));
}
fn word64MinimumUnsigned(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> { return select(second, first, word64LessUnsigned(first, second)); }
fn word64MaximumUnsigned(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> { return select(first, second, word64LessUnsigned(first, second)); }
fn word64MinimumSigned(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> { return select(second, first, word64LessSigned(first, second)); }
fn word64MaximumSigned(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> { return select(first, second, word64LessSigned(first, second)); }
fn word64Multiply32(first: u32, second: u32) -> vec2<u32> {
  let firstLowHalf = first & 0xffffu; let firstHighHalf = first >> 16u; let secondLowHalf = second & 0xffffu; let secondHighHalf = second >> 16u;
  let lowHalfProduct = firstLowHalf * secondLowHalf; let firstCrossHalfProduct = firstLowHalf * secondHighHalf; let secondCrossHalfProduct = firstHighHalf * secondLowHalf; let highHalfProduct = firstHighHalf * secondHighHalf;
  let middle = (lowHalfProduct >> 16u) + (firstCrossHalfProduct & 0xffffu) + (secondCrossHalfProduct & 0xffffu);
  return vec2<u32>((lowHalfProduct & 0xffffu) | (middle << 16u), highHalfProduct + (firstCrossHalfProduct >> 16u) + (secondCrossHalfProduct >> 16u) + (middle >> 16u));
}
fn word64MultiplyLow(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  let product = word64Multiply32(first.x, second.x);
  return vec2<u32>(product.x, product.y + first.x * second.y + first.y * second.x);
}
fn word64Multiply128(first: vec2<u32>, second: vec2<u32>) -> vec4<u32> {
  let lowProduct = word64Multiply32(first.x, second.x);
  let firstCrossProduct = word64Multiply32(first.x, second.y);
  let secondCrossProduct = word64Multiply32(first.y, second.x);
  let highProduct = word64Multiply32(first.y, second.y);
  var carryOne = 0u;
  let middle = lowProduct.y + firstCrossProduct.x;
  carryOne += select(0u, 1u, middle < lowProduct.y);
  let middleSum = middle + secondCrossProduct.x;
  carryOne += select(0u, 1u, middleSum < middle);
  var carryTwo = 0u;
  var upper = firstCrossProduct.y;
  var nextSum = upper + secondCrossProduct.y;
  carryTwo += select(0u, 1u, nextSum < upper);
  upper = nextSum;
  nextSum = upper + highProduct.x;
  carryTwo += select(0u, 1u, nextSum < upper);
  upper = nextSum;
  nextSum = upper + carryOne;
  carryTwo += select(0u, 1u, nextSum < upper);
  upper = nextSum;
  return vec4<u32>(lowProduct.x, middleSum, upper, highProduct.y + carryTwo);
}
fn word64ShiftRight(first: vec2<u32>, amount: u32) -> vec2<u32> {
  if (amount == 0u) { return first; }
  if (amount < 32u) { return vec2<u32>((first.x >> amount) | (first.y << (32u - amount)), first.y >> amount); }
  return vec2<u32>(first.y >> (amount - 32u), 0u);
}
fn word64ShiftLeft(first: vec2<u32>, amount: u32) -> vec2<u32> {
  if (amount == 0u) { return first; }
  if (amount < 32u) { return vec2<u32>(first.x << amount, (first.y << amount) | (first.x >> (32u - amount))); }
  return vec2<u32>(0u, first.x << (amount - 32u));
}
fn word64LeadingZeros(first: vec2<u32>) -> u32 {
  return select(countLeadingZeros(first.x) + 32u, countLeadingZeros(first.y), first.y != 0u);
}
// Unsigned division by first 32-bit divisor: (quotient low, quotient high, remainder). Zero divisor returns all ones.
fn word64DivideSmall(first: vec2<u32>, divisor: u32) -> vec3<u32> {
  if (divisor == 0u) { return vec3<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu); }
  var quotient = vec2<u32>(0u, 0u);
  var remainder = 0u;
  for (var bit = 63i; bit >= 0i; bit = bit - 1i) {
    let amount = u32(bit) & 31u;
    let source = (select(first.x, first.y, bit >= 32i) >> amount) & 1u;
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | source;
    var quotientBit = 0u;
    if (carry != 0u || remainder >= divisor) { remainder = remainder - divisor; quotientBit = 1u; }
    if (bit >= 32i) { quotient.y = quotient.y | (quotientBit << amount); }
    else { quotient.x = quotient.x | (quotientBit << amount); }
  }
  return vec3<u32>(quotient, remainder);
}
// Truncating signed division: (quotient low, quotient high, remainder magnitude, remainder negative flag).
fn word64DivideSmallSigned(first: vec2<u32>, divisor: u32) -> vec4<u32> {
  let negative = (first.y & 0x80000000u) != 0u;
  let result = word64DivideSmall(select(first, word64Negate(first), negative), divisor);
  let quotient = select(result.xy, word64Negate(result.xy), negative && divisor != 0u);
  return vec4<u32>(quotient, result.z, select(0u, 1u, negative));
}
fn word64ToFloat32Unsigned(first: vec2<u32>) -> Float32Conversion {
  if (first.x == 0u && first.y == 0u) { return Float32Conversion(0.0, 1u); }
  let mostSignificantBit = 63u - word64LeadingZeros(first);
  if (mostSignificantBit < 24u) { return Float32Conversion(f32(first.x), 1u); }
  let shift = mostSignificantBit - 23u;
  let kept = word64ShiftRight(first, shift).x;
  let remainder = word64Subtract(first, word64ShiftLeft(vec2<u32>(kept, 0u), shift));
  let halfStep = word64ShiftLeft(vec2<u32>(1u, 0u), shift - 1u);
  let remainderIsHalf = remainder.x == halfStep.x && remainder.y == halfStep.y;
  let roundUp = word64LessUnsigned(halfStep, remainder) || (remainderIsHalf && (kept & 1u) == 1u);
  let mantissa = kept + select(0u, 1u, roundUp);
  let scale = bitcast<f32>((127u + shift) << 23u);
  return Float32Conversion(f32(mantissa) * scale, select(0u, 1u, remainder.x == 0u && remainder.y == 0u));
}
fn word64ToFloat32Signed(first: vec2<u32>) -> Float32Conversion {
  let negative = (first.y & 0x80000000u) != 0u;
  let result = word64ToFloat32Unsigned(select(first, word64Negate(first), negative));
  return Float32Conversion(select(result.value, -result.value, negative), result.exact);
}
fn roundMagnitudeToFloat32Mantissa(magnitude: vec2<u32>, dropBits: u32) -> MantissaRounding {
  if (dropBits >= 54u) {
    return MantissaRounding(0u, 0u, select(0u, 1u, magnitude.x != 0u || magnitude.y != 0u), magnitude);
  }
  let kept = word64ShiftRight(magnitude, dropBits);
  let remainder = word64Subtract(magnitude, word64ShiftLeft(kept, dropBits));
  let halfStep = word64ShiftLeft(vec2<u32>(1u, 0u), dropBits - 1u);
  let remainderIsHalf = remainder.x == halfStep.x && remainder.y == halfStep.y;
  let roundUp = word64LessUnsigned(halfStep, remainder) || (remainderIsHalf && (kept.x & 1u) == 1u);
  let errorUnits = select(remainder, word64Subtract(word64ShiftLeft(vec2<u32>(1u, 0u), dropBits), remainder), roundUp);
  return MantissaRounding(kept.x, select(0u, 1u, roundUp), select(0u, 1u, remainder.x != 0u || remainder.y != 0u), errorUnits);
}
// kind: 0 exact, 1 inexact, 2 overflow, 3 non-finite. errorKey is biased by 1100 so it is positive.
fn describeNarrowingError(errorUnits: vec2<u32>, unitExponent: i32) -> NarrowingClass {
  let bitLength = 64u - word64LeadingZeros(errorUnits);
  let top = select(word64ShiftLeft(errorUnits, 32u - bitLength).x, word64ShiftRight(errorUnits, bitLength - 32u).x, bitLength >= 32u);
  return NarrowingClass(1u, u32(i32(bitLength) - 1i + unitExponent + 1100i), top, errorUnits);
}
fn classifyIntegerNarrowing(pair: vec2<u32>, isSigned: bool) -> NarrowingClass {
  let negative = isSigned && (pair.y & 0x80000000u) != 0u;
  let magnitude = select(pair, word64Negate(pair), negative);
  if (magnitude.x == 0u && magnitude.y == 0u) { return NarrowingClass(0u, 0u, 0u, vec2<u32>(0u)); }
  let mostSignificantBit = 63u - word64LeadingZeros(magnitude);
  if (mostSignificantBit < 24u) { return NarrowingClass(0u, 0u, 0u, vec2<u32>(0u)); }
  let rounded = roundMagnitudeToFloat32Mantissa(magnitude, mostSignificantBit - 23u);
  if (rounded.inexact == 0u) { return NarrowingClass(0u, 0u, 0u, vec2<u32>(0u)); }
  return describeNarrowingError(rounded.errorUnits, 0i);
}
fn classifyDoubleNarrowing(pair: vec2<u32>) -> NarrowingClass {
  let biasedExponent = (pair.y >> 20u) & 0x7ffu;
  if (biasedExponent == 0x7ffu) { return NarrowingClass(3u, 0u, 0u, vec2<u32>(0u)); }
  let significand = vec2<u32>(pair.x, (pair.y & 0xfffffu) | select(0u, 0x100000u, biasedExponent != 0u));
  if (significand.x == 0u && significand.y == 0u) { return NarrowingClass(0u, 0u, 0u, vec2<u32>(0u)); }
  let exponent = select(i32(biasedExponent) - 1023i, -1022i, biasedExponent == 0u);
  if (exponent >= 128i) { return NarrowingClass(2u, 0u, 0u, vec2<u32>(0u)); }
  let dropBits = 29u + select(0u, u32(-126i - exponent), exponent < -126i);
  let rounded = roundMagnitudeToFloat32Mantissa(significand, dropBits);
  if (exponent == 127i && rounded.roundUp != 0u && rounded.kept == 0xffffffu) { return NarrowingClass(2u, 0u, 0u, vec2<u32>(0u)); }
  if (rounded.inexact == 0u) { return NarrowingClass(0u, 0u, 0u, vec2<u32>(0u)); }
  return describeNarrowingError(rounded.errorUnits, exponent - 52i);
}
`;
