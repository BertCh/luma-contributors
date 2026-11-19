var e=`
fn addUint64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.y + right.y;
  return vec2<u32>(left.x + right.x + select(0u, 1u, low < left.y), low);
}

fn subtractUint64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x - right.x - select(0u, 1u, left.y < right.y), left.y - right.y);
}

fn negateInt64(value: vec2<u32>) -> vec2<u32> {
  return subtractUint64(vec2<u32>(0u, 0u), value);
}

fn compareUint64(left: vec2<u32>, right: vec2<u32>) -> i32 {
  if (left.x != right.x) {
    return select(-1i, 1i, left.x > right.x);
  }
  if (left.y != right.y) {
    return select(-1i, 1i, left.y > right.y);
  }
  return 0i;
}

fn isLessUint64(left: vec2<u32>, right: vec2<u32>) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}

fn isLessOrEqualUint64(left: vec2<u32>, right: vec2<u32>) -> bool {
  return left.x < right.x || (left.x == right.x && left.y <= right.y);
}

// Signed order: flipping the sign bit of the high words makes them compare as unsigned.
fn compareInt64(left: vec2<u32>, right: vec2<u32>) -> i32 {
  return compareUint64(vec2<u32>(left.x ^ 0x80000000u, left.y), vec2<u32>(right.x ^ 0x80000000u, right.y));
}

fn shiftLeftUint64(value: vec2<u32>, bits: u32) -> vec2<u32> {
  if (bits == 0u) {
    return value;
  }
  if (bits >= 64u) {
    return vec2<u32>(0u, 0u);
  }
  if (bits >= 32u) {
    return vec2<u32>(value.y << (bits - 32u), 0u);
  }
  return vec2<u32>((value.x << bits) | (value.y >> (32u - bits)), value.y << bits);
}

fn shiftRightUint64(value: vec2<u32>, bits: u32) -> vec2<u32> {
  if (bits == 0u) {
    return value;
  }
  if (bits >= 64u) {
    return vec2<u32>(0u, 0u);
  }
  if (bits >= 32u) {
    return vec2<u32>(0u, value.x >> (bits - 32u));
  }
  return vec2<u32>(value.x >> bits, (value.y >> bits) | (value.x << (32u - bits)));
}

// Exact 32 x 32 -> 64 bit product from 16-bit limbs; no partial sum exceeds 2^32.
fn multiplyUint32Wide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowLow = leftLow * rightLow;
  let lowHigh = leftLow * rightHigh;
  let highLow = leftHigh * rightLow;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  return vec2<u32>(
    leftHigh * rightHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u),
    (lowLow & 0xffffu) | (middle << 16u)
  );
}

fn multiplyUint64ByUint32(left: vec2<u32>, right: u32) -> vec2<u32> {
  let wide = multiplyUint32Wide(left.y, right);
  return vec2<u32>(wide.x + left.x * right, wide.y);
}

fn multiplyUint64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let wide = multiplyUint32Wide(left.y, right.y);
  return vec2<u32>(wide.x + left.y * right.x + left.x * right.y, wide.y);
}

struct Uint64Division {
  quotient: vec2<u32>,
  remainder: u32,
}

// Restoring division one bit at a time. The running remainder stays below the divisor, so after a
// shift it is below 2^33; the bit shifted out of the u32 is carried in overflowBit.
fn divideUint64ByUint32(value: vec2<u32>, divisor: u32) -> Uint64Division {
  if (divisor == 0u) {
    return Uint64Division(vec2<u32>(0xffffffffu, 0xffffffffu), 0u);
  }
  var quotient = vec2<u32>(0u, 0u);
  var remainder = 0u;
  for (var bit = 0u; bit < 64u; bit++) {
    let position = 63u - bit;
    let word = select(value.y, value.x, position >= 32u);
    let incoming = (word >> (position & 31u)) & 1u;
    let overflowBit = remainder >> 31u;
    remainder = (remainder << 1u) | incoming;
    if (overflowBit != 0u || remainder >= divisor) {
      remainder = remainder - divisor;
      if (position >= 32u) {
        quotient.x = quotient.x | (1u << (position - 32u));
      } else {
        quotient.y = quotient.y | (1u << position);
      }
    }
  }
  return Uint64Division(quotient, remainder);
}

fn uint64ToFloat(value: vec2<u32>) -> f32 {
  return f32(value.x) * 4294967296.0 + f32(value.y);
}

fn int64ToFloat(value: vec2<u32>) -> f32 {
  if ((value.x & 0x80000000u) != 0u) {
    return -uint64ToFloat(negateInt64(value));
  }
  return uint64ToFloat(value);
}
`;export{e as t};