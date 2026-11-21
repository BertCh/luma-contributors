// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Keep each Gram intermediate as a signed float32 significand and an integer exponent.
 * The mantissa retains float32 precision, while products of up to four weights retain their
 * exponent through the tile and probe sums. Integer bit operations keep Metal from merging
 * the scale into a product or flushing the reciprocal of a large intermediate.
 * @internal
 */
export const SPATIAL_WEIGHTS_GRAM_WGSL = /* wgsl */ `
fn getGramParts(value: f32) -> vec2<f32> {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u || magnitude >= 0x7f800000u) { return vec2<f32>(value, 0.0); }
  var exponent = i32(magnitude >> 23u) - 127;
  var fraction = magnitude & 0x007fffffu;
  if (exponent == -127) {
    let shift = countLeadingZeros(fraction) - 8u;
    fraction = (fraction << shift) & 0x007fffffu;
    exponent = -126 - i32(shift);
  }
  return vec2<f32>(bitcast<f32>((bits & 0x80000000u) | 0x3f800000u | fraction), f32(exponent));
}

fn multiplyGram(first: vec2<f32>, second: vec2<f32>) -> vec2<f32> {
  let product = getGramParts(first.x * second.x);
  return vec2<f32>(product.x, product.y + first.y + second.y);
}

fn addGram(first: vec2<f32>, second: vec2<f32>) -> vec2<f32> {
  if (first.x == 0.0) { return second; }
  if (second.x == 0.0) { return first; }
  let firstMagnitude = bitcast<u32>(first.x) & 0x7fffffffu;
  let secondMagnitude = bitcast<u32>(second.x) & 0x7fffffffu;
  if (firstMagnitude >= 0x7f800000u || secondMagnitude >= 0x7f800000u) {
    return vec2<f32>(first.x + second.x, 0.0);
  }
  let firstLarger = first.y >= second.y;
  let larger = select(second, first, firstLarger);
  let smaller = select(first, second, firstLarger);
  let gap = i32(larger.y - smaller.y);
  if (gap > 26) { return larger; }
  let relativeScale = bitcast<f32>(u32(127 - gap) << 23u);
  let sum = getGramParts(larger.x + smaller.x * relativeScale);
  return vec2<f32>(sum.x, sum.y + larger.y);
}

fn publishGram(value: vec2<f32>, divisor: f32) -> f32 {
  let quotient = getGramParts(value.x / divisor);
  let bits = bitcast<u32>(quotient.x);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u || magnitude >= 0x7f800000u) { return quotient.x; }
  let exponent = i32(value.y + quotient.y);
  let sign = bits & 0x80000000u;
  let fraction = bits & 0x007fffffu;
  if (exponent > 127) { return bitcast<f32>(sign | 0x7f800000u); }
  if (exponent >= -126) { return bitcast<f32>(sign | (u32(exponent + 127) << 23u) | fraction); }
  // Round the final subnormal once, using integer bits so Metal cannot flush it.
  if (exponent < -150) { return bitcast<f32>(sign); }
  let shift = u32(-126 - exponent);
  let significand = fraction | 0x00800000u;
  let retained = significand >> shift;
  let remainder = significand & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  let roundUp = remainder > halfway || (remainder == halfway && (retained & 1u) != 0u);
  return bitcast<f32>(sign | (retained + select(0u, 1u, roundUp)));
}
`;

/** The same fixed pairwise order as the scalar power reductions, with wide Gram terms. @internal */
export function getGramPairwiseSumWGSL(count: string, term: string): string {
  return /* wgsl */ `
  var blocks: array<vec2<f32>, 32>;
  var depth = 0u;
  for (var gramItem = 0u; gramItem < ${count}; gramItem++) {
    var subtotal = ${term};
    var carry = gramItem;
    for (var level = 0u; level < 32u; level++) {
      if ((carry & 1u) == 0u) { break; }
      depth--;
      subtotal = addGram(blocks[depth], subtotal);
      carry >>= 1u;
    }
    blocks[depth] = subtotal;
    depth++;
  }
  var gramSum = vec2<f32>(0.0);
  for (var level = 0u; level < 32u; level++) {
    if (depth == 0u) { break; }
    depth--;
    gramSum = addGram(blocks[depth], gramSum);
  }`;
}

/** Load a Gram pair from packed float32 word storage. @internal */
export function getGramLoadWGSL(binding: string, index: string): string {
  return `vec2<f32>(${binding}[${binding}Offset + (${index}) * 2u], ${binding}[${binding}Offset + (${index}) * 2u + 1u])`;
}
