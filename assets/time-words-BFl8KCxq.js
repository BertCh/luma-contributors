var e=new Uint8Array(new Uint32Array([1]).buffer)[0]===1;function t(t){if(!e)throw Error(`Int64 time words require a little-endian platform`);return new Uint32Array(t.buffer,t.byteOffset,t.length*2)}var n=`
fn timeWordsSubtract(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let borrow = select(0u, 1u, a.x < b.x);
  return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}

fn timeWordsIsNegative(difference: vec2<u32>) -> bool {
  return (difference.y & 0x80000000u) != 0u;
}

fn timeWordsToF32(difference: vec2<u32>) -> f32 {
  let negative = timeWordsIsNegative(difference);
  var low = difference.x;
  var high = difference.y;
  if (negative) {
    low = ~difference.x + 1u;
    high = ~difference.y + select(0u, 1u, low == 0u);
  }
  let magnitude = f32(high) * 4294967296.0 + f32(low);
  return select(magnitude, -magnitude, negative);
}

// Signed order: high words with the sign bit flipped compare as unsigned, then the low words.
fn timeWordsCompare(a: vec2<u32>, b: vec2<u32>) -> i32 {
  let aHigh = a.y ^ 0x80000000u;
  let bHigh = b.y ^ 0x80000000u;
  if (aHigh != bHigh) {
    return select(-1i, 1i, aHigh > bHigh);
  }
  if (a.x != b.x) {
    return select(-1i, 1i, a.x > b.x);
  }
  return 0i;
}

// The magnitude |a - b| is below 2^64, so subtracting the smaller from the larger is exact as an
// unsigned 64-bit value even when the signed difference overflows Int64.
fn timeWordsDifferenceToF32(a: vec2<u32>, b: vec2<u32>) -> f32 {
  let isForward = timeWordsCompare(a, b) >= 0i;
  let magnitudeWords = select(timeWordsSubtract(b, a), timeWordsSubtract(a, b), isForward);
  let magnitude = f32(magnitudeWords.y) * 4294967296.0 + f32(magnitudeWords.x);
  return select(-magnitude, magnitude, isForward);
}

fn isTimeWordsAtLeast(a: vec2<u32>, aFraction: f32, b: vec2<u32>, bFraction: f32) -> bool {
  let order = timeWordsCompare(a, b);
  if (order == 0i) {
    return aFraction >= bFraction;
  }
  return order > 0i;
}

fn timeWordsDifference(a: vec2<u32>, aFraction: f32, b: vec2<u32>, bFraction: f32) -> f32 {
  return timeWordsDifferenceToF32(a, b) + (aFraction - bFraction);
}
`;export{t as n,n as t};