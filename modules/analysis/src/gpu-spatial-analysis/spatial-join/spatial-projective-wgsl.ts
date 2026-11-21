// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact projective fallback for tied polygon crossings. Coordinates are integers in units of
 * 2^-149. A float32 coordinate needs 277 bits; the degree-four predicates need at most 1112
 * bits including carries, so 36 words suffice. Used only when rounded crossing parameters tie.
 * @internal
 */
export const SPATIAL_PROJECTIVE_WGSL = /* wgsl */ `
struct SpatialInteger { word0: u32, word1: u32, word2: u32, word3: u32, word4: u32, word5: u32, word6: u32, word7: u32, word8: u32, word9: u32, word10: u32, word11: u32, word12: u32, word13: u32, word14: u32, word15: u32, word16: u32, word17: u32, word18: u32, word19: u32, word20: u32, word21: u32, word22: u32, word23: u32, word24: u32, word25: u32, word26: u32, word27: u32, word28: u32, word29: u32, word30: u32, word31: u32, word32: u32, word33: u32, word34: u32, word35: u32, negative: u32 }
fn spatialFromWords(words: array<u32, 36>, negative: u32) -> SpatialInteger {
  return SpatialInteger(words[0u], words[1u], words[2u], words[3u], words[4u], words[5u], words[6u], words[7u], words[8u], words[9u], words[10u], words[11u], words[12u], words[13u], words[14u], words[15u], words[16u], words[17u], words[18u], words[19u], words[20u], words[21u], words[22u], words[23u], words[24u], words[25u], words[26u], words[27u], words[28u], words[29u], words[30u], words[31u], words[32u], words[33u], words[34u], words[35u], negative);
}
fn spatialWord(value: SpatialInteger, index: u32) -> u32 {
  if (index == 0u) { return value.word0; }
  if (index == 1u) { return value.word1; }
  if (index == 2u) { return value.word2; }
  if (index == 3u) { return value.word3; }
  if (index == 4u) { return value.word4; }
  if (index == 5u) { return value.word5; }
  if (index == 6u) { return value.word6; }
  if (index == 7u) { return value.word7; }
  if (index == 8u) { return value.word8; }
  if (index == 9u) { return value.word9; }
  if (index == 10u) { return value.word10; }
  if (index == 11u) { return value.word11; }
  if (index == 12u) { return value.word12; }
  if (index == 13u) { return value.word13; }
  if (index == 14u) { return value.word14; }
  if (index == 15u) { return value.word15; }
  if (index == 16u) { return value.word16; }
  if (index == 17u) { return value.word17; }
  if (index == 18u) { return value.word18; }
  if (index == 19u) { return value.word19; }
  if (index == 20u) { return value.word20; }
  if (index == 21u) { return value.word21; }
  if (index == 22u) { return value.word22; }
  if (index == 23u) { return value.word23; }
  if (index == 24u) { return value.word24; }
  if (index == 25u) { return value.word25; }
  if (index == 26u) { return value.word26; }
  if (index == 27u) { return value.word27; }
  if (index == 28u) { return value.word28; }
  if (index == 29u) { return value.word29; }
  if (index == 30u) { return value.word30; }
  if (index == 31u) { return value.word31; }
  if (index == 32u) { return value.word32; }
  if (index == 33u) { return value.word33; }
  if (index == 34u) { return value.word34; }
  if (index == 35u) { return value.word35; }
  return 0u;
}
struct SpatialQuery { x: SpatialInteger, y: SpatialInteger, denominator: SpatialInteger }
fn spatialSign(value: SpatialInteger) -> i32 {
  for (var index = 36u; index > 0u; index--) {
    if (spatialWord(value, index - 1u) != 0u) { return select(1i, -1i, value.negative != 0u); }
  }
  return 0i;
}
fn spatialNegate(value: SpatialInteger) -> SpatialInteger {
  var result = value; result.negative ^= 1u; return result;
}
fn spatialApplySign(value: SpatialInteger, negate: bool) -> SpatialInteger {
  if (negate) { return spatialNegate(value); }
  return value;
}
fn spatialEnabled(value: SpatialInteger, enabled: bool) -> SpatialInteger {
  if (enabled) { return value; }
  return spatialInteger(0.0);
}
fn spatialInteger(value: f32) -> SpatialInteger {
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
  let shift = select(0u, exponent - 1u, exponent != 0u);
  let word = shift / 32u; let bit = shift % 32u;
  var words: array<u32, 36>;
  var negative = 0u;
  negative = bits >> 31u;
  words[word] = significand << bit;
  if (bit != 0u) { words[word + 1u] = significand >> (32u - bit); }
  return spatialFromWords(words, negative);
}
fn spatialCompareMagnitude(first: SpatialInteger, second: SpatialInteger) -> i32 {
  for (var index = 36u; index > 0u; index--) {
    if (spatialWord(first, index - 1u) != spatialWord(second, index - 1u)) {
      return select(-1i, 1i, spatialWord(first, index - 1u) > spatialWord(second, index - 1u));
    }
  }
  return 0i;
}
fn spatialAdd(first: SpatialInteger, second: SpatialInteger) -> SpatialInteger {
  var words: array<u32, 36>;
  var negative = 0u;
  if (first.negative == second.negative) {
    negative = first.negative;
    var carry = 0u;
    for (var index = 0u; index < 36u; index++) {
      let partial = spatialWord(first, index) + spatialWord(second, index);
      let total = partial + carry;
      carry = select(0u, 1u, partial < spatialWord(first, index) || total < partial);
      words[index] = total;
    }
  } else {
    let firstLarger = spatialCompareMagnitude(first, second) >= 0i;
    negative = select(second.negative, first.negative, firstLarger);
    var borrow = 0u;
    for (var index = 0u; index < 36u; index++) {
      let firstWord = spatialWord(first, index); let secondWord = spatialWord(second, index);
      let largerWord = select(secondWord, firstWord, firstLarger);
      let smallerWord = select(firstWord, secondWord, firstLarger);
      let partial = largerWord - smallerWord;
      let total = partial - borrow;
      borrow = select(0u, 1u, largerWord < smallerWord || partial < borrow);
      words[index] = total;
    }
  }
  return spatialFromWords(words, negative);
}
fn spatialSubtract(first: SpatialInteger, second: SpatialInteger) -> SpatialInteger {
  return spatialAdd(first, spatialNegate(second));
}
fn spatialMultiply(first: SpatialInteger, second: SpatialInteger) -> SpatialInteger {
  var words: array<u32, 36>;
  var negative = 0u;
  negative = first.negative ^ second.negative;
  for (var firstIndex = 0u; firstIndex < 36u; firstIndex++) {
    let firstWord = spatialWord(first, firstIndex);
    if (firstWord == 0u) { continue; }
    for (var secondIndex = 0u; secondIndex + firstIndex < 36u; secondIndex++) {
      let secondWord = spatialWord(second, secondIndex);
      if (secondWord == 0u) { continue; }
      let lowProduct = (firstWord & 65535u) * (secondWord & 65535u);
      let middleFirst = (firstWord >> 16u) * (secondWord & 65535u);
      let middleSecond = (firstWord & 65535u) * (secondWord >> 16u);
      let middle = (lowProduct >> 16u) + (middleFirst & 65535u) + (middleSecond & 65535u);
      let low = (lowProduct & 65535u) | (middle << 16u);
      let high = (firstWord >> 16u) * (secondWord >> 16u) + (middleFirst >> 16u) + (middleSecond >> 16u) + (middle >> 16u);
      var carry = 0u;
      for (var slot = firstIndex + secondIndex; slot < 36u; slot++) {
        let relative = slot - firstIndex - secondIndex;
        var addend = 0u;
        if (relative == 0u) { addend = low; } else if (relative == 1u) { addend = high; } else if (carry == 0u) { break; }
        let partial = words[slot] + addend;
        let total = partial + carry;
        carry = select(0u, 1u, partial < addend || total < partial);
        words[slot] = total;
      }
    }
  }
  return spatialFromWords(words, negative);
}
fn spatialCross(firstX: SpatialInteger, firstY: SpatialInteger, secondX: SpatialInteger, secondY: SpatialInteger) -> SpatialInteger {
  return spatialSubtract(spatialMultiply(firstX, secondY), spatialMultiply(firstY, secondX));
}
fn spatialEndpointQuery(point: vec2f) -> SpatialQuery {
  var words: array<u32, 36>; words[0] = 1u;
  return SpatialQuery(spatialInteger(point.x), spatialInteger(point.y), spatialFromWords(words, 0u));
}
fn spatialCrossingQuery(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> SpatialQuery {
  let startX = spatialInteger(a.x); let startY = spatialInteger(a.y);
  let directionX = spatialSubtract(spatialInteger(b.x), startX);
  let directionY = spatialSubtract(spatialInteger(b.y), startY);
  let offsetX = spatialSubtract(spatialInteger(c.x), startX);
  let offsetY = spatialSubtract(spatialInteger(c.y), startY);
  let edgeX = spatialSubtract(spatialInteger(d.x), spatialInteger(c.x));
  let edgeY = spatialSubtract(spatialInteger(d.y), spatialInteger(c.y));
  let rawNumerator = spatialCross(offsetX, offsetY, edgeX, edgeY);
  let rawDenominator = spatialCross(directionX, directionY, edgeX, edgeY);
  let numerator = spatialApplySign(rawNumerator, spatialSign(rawDenominator) < 0i);
  let denominator = spatialApplySign(rawDenominator, spatialSign(rawDenominator) < 0i);
  return SpatialQuery(spatialAdd(spatialMultiply(startX, denominator), spatialMultiply(directionX, numerator)),
    spatialAdd(spatialMultiply(startY, denominator), spatialMultiply(directionY, numerator)), denominator);
}
// Exact midpoint of two represented parameter values; addition never rounds them together.
fn spatialParameterQuery(a: vec2f, b: vec2f, first: f32, second: f32) -> SpatialQuery {
  let numerator = spatialAdd(spatialInteger(first), spatialInteger(second));
  let denominator = spatialInteger(2.0);
  let startX = spatialInteger(a.x); let startY = spatialInteger(a.y);
  return SpatialQuery(spatialAdd(spatialMultiply(startX, denominator), spatialMultiply(spatialSubtract(spatialInteger(b.x), startX), numerator)),
    spatialAdd(spatialMultiply(startY, denominator), spatialMultiply(spatialSubtract(spatialInteger(b.y), startY), numerator)), denominator);
}
fn spatialOffsetSign(baseValue: SpatialInteger, direction: SpatialInteger, normal: SpatialInteger) -> i32 {
  var sign = spatialSign(baseValue);
  if (sign == 0i) { sign = spatialSign(direction); }
  if (sign == 0i) { sign = spatialSign(normal); }
  return sign;
}
fn spatialCrossingOrder(a: vec2f, b: vec2f, c: vec2f, d: vec2f, e: vec2f, f: vec2f) -> i32 {
  let directionX = spatialSubtract(spatialInteger(b.x), spatialInteger(a.x));
  let directionY = spatialSubtract(spatialInteger(b.y), spatialInteger(a.y));
  let firstX = spatialSubtract(spatialInteger(d.x), spatialInteger(c.x));
  let firstY = spatialSubtract(spatialInteger(d.y), spatialInteger(c.y));
  let secondX = spatialSubtract(spatialInteger(f.x), spatialInteger(e.x));
  let secondY = spatialSubtract(spatialInteger(f.y), spatialInteger(e.y));
  let rawFirstNumerator = spatialCross(spatialSubtract(spatialInteger(c.x), spatialInteger(a.x)), spatialSubtract(spatialInteger(c.y), spatialInteger(a.y)), firstX, firstY);
  let rawSecondNumerator = spatialCross(spatialSubtract(spatialInteger(e.x), spatialInteger(a.x)), spatialSubtract(spatialInteger(e.y), spatialInteger(a.y)), secondX, secondY);
  let rawFirstDenominator = spatialCross(directionX, directionY, firstX, firstY);
  let rawSecondDenominator = spatialCross(directionX, directionY, secondX, secondY);
  let firstDenominator = spatialApplySign(rawFirstDenominator, spatialSign(rawFirstDenominator) < 0i);
  let secondDenominator = spatialApplySign(rawSecondDenominator, spatialSign(rawSecondDenominator) < 0i);
  let firstNumerator = spatialApplySign(rawFirstNumerator, spatialSign(rawFirstDenominator) < 0i);
  let secondNumerator = spatialApplySign(rawSecondNumerator, spatialSign(rawSecondDenominator) < 0i);
  return spatialSign(spatialSubtract(spatialMultiply(firstNumerator, secondDenominator), spatialMultiply(secondNumerator, firstDenominator)));
}
`;
