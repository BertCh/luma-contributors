// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * 64-bit integers as two `u32` words for WGSL, which has no 64-bit integer type (contributor
 * contract rule 11). A value is `vec2<u32>(high, low)`: `.x` holds bits 32 to 63 and `.y` bits 0 to
 * 31. This is the in-shader order of published luma.gl (`dggs_u64_*`, `fp64_u64_*`) and of the
 * local `CELL_KEY_WGSL`, so the module can be concatenated with them. Storage is a separate
 * matter: Arrow `Int64` columns, the time words (`time-words.ts`) and the cell tables store the
 * low word first, so build a value from storage as `vec2<u32>(words[2 * i + 1], words[2 * i])`.
 * The older private copies (`add64`, `addWide`) still use `(low, high)` in the shader; do not mix
 * them with this module without swapping the words. Signed values are two's complement in the
 * same words. Arithmetic wraps modulo `2^64`.
 *
 * Functions (all names end in `Uint64` or `Int64`, so they cannot collide with the older local
 * copies `add64`, `addWide`, `cellAddI64` while those are being replaced):
 * - `addUint64(left, right)`, `subtractUint64(left, right)`, `negateInt64(value)`.
 * - `compareUint64(left, right) -> i32` (-1, 0, 1), `isLessUint64`, `isLessOrEqualUint64`;
 *   `compareInt64(left, right) -> i32` for signed values.
 * - `shiftLeftUint64(value, bits)`, `shiftRightUint64(value, bits)` (logical) for `bits` in
 *   `[0, 63]`; larger shifts return 0.
 * - `multiplyUint32Wide(left: u32, right: u32) -> vec2<u32>`: the exact 64-bit product as
 *   `(high, low)`, from 16-bit limbs.
 * - `multiplyUint64ByUint32(left: vec2<u32>, right: u32) -> vec2<u32>` and
 *   `multiplyUint64(left, right)`: products modulo `2^64`.
 * - `divideUint64ByUint32(value, divisor) -> Uint64Division` with `quotient: vec2<u32>` and
 *   `remainder: u32`; a zero divisor returns quotient `0xffffffffffffffff` and remainder 0.
 * - `uint64ToFloat(value) -> f32`, `int64ToFloat(value) -> f32`: nearest-ish float (two
 *   roundings); the signed form converts the magnitude so small negatives do not cancel.
 *
 * Checked against BigInt arithmetic in `test/utils/wgsl-words-search.spec.ts`.
 *
 * @internal
 */
export const UINT64_WGSL = /* wgsl */ `
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
`;
