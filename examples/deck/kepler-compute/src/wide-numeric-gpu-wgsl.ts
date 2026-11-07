// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL helpers for exact 64-bit arithmetic on vec2<u32> pairs (x = low word, y = high word, the
 * little-endian element order of a uint32x2 GPUVector). Algorithms are mirrored one-to-one by the
 * TypeScript emulation in wide-numeric-gpu-reference.ts. 128-bit values are vec4<u32> (little-endian words).
 */
export const WIDE_NUMERIC_WGSL_HELPERS = /* wgsl */ `
struct Float32Conversion { value: f32, exact: u32 }
struct NarrowingClass { kind: u32, errorKey: u32, errorTop: u32 }
struct MantissaRounding { kept: u32, roundUp: u32, inexact: u32, errorUnits: vec2<u32> }

fn word64Add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let low = a.x + b.x;
  return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}
fn word64Subtract(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let low = a.x - b.x;
  return vec2<u32>(low, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn word64Negate(a: vec2<u32>) -> vec2<u32> { return word64Subtract(vec2<u32>(0u, 0u), a); }
fn word64LessUnsigned(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.y < b.y || (a.y == b.y && a.x < b.x);
}
fn word64LessSigned(a: vec2<u32>, b: vec2<u32>) -> bool {
  return word64LessUnsigned(vec2<u32>(a.x, a.y ^ 0x80000000u), vec2<u32>(b.x, b.y ^ 0x80000000u));
}
fn word64MinimumUnsigned(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(b, a, word64LessUnsigned(a, b)); }
fn word64MaximumUnsigned(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, word64LessUnsigned(a, b)); }
fn word64MinimumSigned(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(b, a, word64LessSigned(a, b)); }
fn word64MaximumSigned(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, word64LessSigned(a, b)); }
fn word64Multiply32(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu; let a1 = a >> 16u; let b0 = b & 0xffffu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let middle = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return vec2<u32>((p00 & 0xffffu) | (middle << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (middle >> 16u));
}
fn word64MultiplyLow(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let product = word64Multiply32(a.x, b.x);
  return vec2<u32>(product.x, product.y + a.x * b.y + a.y * b.x);
}
fn word64Multiply128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
  let p0 = word64Multiply32(a.x, b.x);
  let p1 = word64Multiply32(a.x, b.y);
  let p2 = word64Multiply32(a.y, b.x);
  let p3 = word64Multiply32(a.y, b.y);
  var carryOne = 0u;
  let middle = p0.y + p1.x;
  carryOne += select(0u, 1u, middle < p0.y);
  let middleSum = middle + p2.x;
  carryOne += select(0u, 1u, middleSum < middle);
  var carryTwo = 0u;
  var upper = p1.y;
  var nextSum = upper + p2.y;
  carryTwo += select(0u, 1u, nextSum < upper);
  upper = nextSum;
  nextSum = upper + p3.x;
  carryTwo += select(0u, 1u, nextSum < upper);
  upper = nextSum;
  nextSum = upper + carryOne;
  carryTwo += select(0u, 1u, nextSum < upper);
  upper = nextSum;
  return vec4<u32>(p0.x, middleSum, upper, p3.y + carryTwo);
}
fn word64ShiftRight(a: vec2<u32>, amount: u32) -> vec2<u32> {
  if (amount == 0u) { return a; }
  if (amount < 32u) { return vec2<u32>((a.x >> amount) | (a.y << (32u - amount)), a.y >> amount); }
  return vec2<u32>(a.y >> (amount - 32u), 0u);
}
fn word64ShiftLeft(a: vec2<u32>, amount: u32) -> vec2<u32> {
  if (amount == 0u) { return a; }
  if (amount < 32u) { return vec2<u32>(a.x << amount, (a.y << amount) | (a.x >> (32u - amount))); }
  return vec2<u32>(0u, a.x << (amount - 32u));
}
fn word64LeadingZeros(a: vec2<u32>) -> u32 {
  return select(countLeadingZeros(a.x) + 32u, countLeadingZeros(a.y), a.y != 0u);
}
// Unsigned division by a 32-bit divisor: (quotient low, quotient high, remainder). Zero divisor returns all ones.
fn word64DivideSmall(a: vec2<u32>, divisor: u32) -> vec3<u32> {
  if (divisor == 0u) { return vec3<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu); }
  var quotient = vec2<u32>(0u, 0u);
  var remainder = 0u;
  for (var bit = 63i; bit >= 0i; bit = bit - 1i) {
    let amount = u32(bit) & 31u;
    let source = (select(a.x, a.y, bit >= 32i) >> amount) & 1u;
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
fn word64DivideSmallSigned(a: vec2<u32>, divisor: u32) -> vec4<u32> {
  let negative = (a.y & 0x80000000u) != 0u;
  let result = word64DivideSmall(select(a, word64Negate(a), negative), divisor);
  let quotient = select(result.xy, word64Negate(result.xy), negative && divisor != 0u);
  return vec4<u32>(quotient, result.z, select(0u, 1u, negative));
}
fn word64ToFloat32Unsigned(a: vec2<u32>) -> Float32Conversion {
  if (a.x == 0u && a.y == 0u) { return Float32Conversion(0.0, 1u); }
  let mostSignificantBit = 63u - word64LeadingZeros(a);
  if (mostSignificantBit < 24u) { return Float32Conversion(f32(a.x), 1u); }
  let shift = mostSignificantBit - 23u;
  let kept = word64ShiftRight(a, shift).x;
  let remainder = word64Subtract(a, word64ShiftLeft(vec2<u32>(kept, 0u), shift));
  let halfStep = word64ShiftLeft(vec2<u32>(1u, 0u), shift - 1u);
  let remainderIsHalf = remainder.x == halfStep.x && remainder.y == halfStep.y;
  let roundUp = word64LessUnsigned(halfStep, remainder) || (remainderIsHalf && (kept & 1u) == 1u);
  let mantissa = kept + select(0u, 1u, roundUp);
  let scale = bitcast<f32>((127u + shift) << 23u);
  return Float32Conversion(f32(mantissa) * scale, select(0u, 1u, remainder.x == 0u && remainder.y == 0u));
}
fn word64ToFloat32Signed(a: vec2<u32>) -> Float32Conversion {
  let negative = (a.y & 0x80000000u) != 0u;
  let result = word64ToFloat32Unsigned(select(a, word64Negate(a), negative));
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
  return NarrowingClass(1u, u32(i32(bitLength) - 1i + unitExponent + 1100i), top);
}
fn classifyIntegerNarrowing(pair: vec2<u32>, isSigned: bool) -> NarrowingClass {
  let negative = isSigned && (pair.y & 0x80000000u) != 0u;
  let magnitude = select(pair, word64Negate(pair), negative);
  if (magnitude.x == 0u && magnitude.y == 0u) { return NarrowingClass(0u, 0u, 0u); }
  let mostSignificantBit = 63u - word64LeadingZeros(magnitude);
  if (mostSignificantBit < 24u) { return NarrowingClass(0u, 0u, 0u); }
  let rounded = roundMagnitudeToFloat32Mantissa(magnitude, mostSignificantBit - 23u);
  if (rounded.inexact == 0u) { return NarrowingClass(0u, 0u, 0u); }
  return describeNarrowingError(rounded.errorUnits, 0i);
}
fn classifyDoubleNarrowing(pair: vec2<u32>) -> NarrowingClass {
  let biasedExponent = (pair.y >> 20u) & 0x7ffu;
  if (biasedExponent == 0x7ffu) { return NarrowingClass(3u, 0u, 0u); }
  let significand = vec2<u32>(pair.x, (pair.y & 0xfffffu) | select(0u, 0x100000u, biasedExponent != 0u));
  if (significand.x == 0u && significand.y == 0u) { return NarrowingClass(0u, 0u, 0u); }
  let exponent = select(i32(biasedExponent) - 1023i, -1022i, biasedExponent == 0u);
  if (exponent >= 128i) { return NarrowingClass(2u, 0u, 0u); }
  let dropBits = 29u + select(0u, u32(-126i - exponent), exponent < -126i);
  let rounded = roundMagnitudeToFloat32Mantissa(significand, dropBits);
  if (exponent == 127i && rounded.roundUp != 0u && rounded.kept == 0xffffffu) { return NarrowingClass(2u, 0u, 0u); }
  if (rounded.inexact == 0u) { return NarrowingClass(0u, 0u, 0u); }
  return describeNarrowingError(rounded.errorUnits, exponent - 52i);
}
`;
