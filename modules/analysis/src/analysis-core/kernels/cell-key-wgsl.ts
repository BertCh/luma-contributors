// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from './wgsl-kernel-nodes';
import {DECODE_ORDERED_KEY_WGSL, GET_NAN_WGSL, GET_ORDERED_KEY_WGSL} from './wgsl-common';

/**
 * Default fixed-point scale of cell-table sums: 16 fractional bits.
 *
 * A fixed-point sum is exact as an integer sum, not as a sum of the input values: every row is
 * first rounded to the nearest multiple of `1 / sumScale` (half to even), so with the default each
 * row carries an absolute error of up to `2^-17` (7.6e-6), a value below that contributes 0, and a
 * constant column of `1e-4` sums as `7 / 65536 = 1.0681e-4` per row (6.8% high). The default suits
 * values of ordinary magnitude (about 0.01 and above). For small or very large values pass a
 * `sumScale` that follows the data, for example from {@link getGPUCellSumScale}.
 */
export const GPU_CELL_DEFAULT_SUM_SCALE = 65536;

/**
 * Target magnitude under {@link getGPUCellSumScale} when a normal float32 scale can satisfy it:
 * `2^61`, one
 * bit below the saturation limit of a scaled row (`CELL_MAXIMUM_SCALED_VALUE`, just under `2^62`)
 * and two below the sign bit of the signed 64-bit sum.
 */
const CELL_SUM_SCALE_HEADROOM = 2 ** 61;

/**
 * Picks a `sumScale` from the data: the largest power of two for which `maximumRowCount` rows of
 * magnitude `maximumMagnitude` cannot pass `2^61` in the signed 64-bit sum. A power of two keeps
 * `value * sumScale` exact in float32, so the only rounding is to the multiple of `1 / sumScale`.
 * The result is clamped to normal float32 powers in `[2^-126, 2^127]`; the headroom bound holds
 * whenever a scale in that range can satisfy it.
 *
 * When the upper cap `2^127` does not constrain the scale, the resolution is at most
 * `maximumMagnitude * maximumRowCount * 2^-60` per row: with the default row count of `2^31`, a
 * float32 row within a factor of 32 of `maximumMagnitude` is added with no rounding, and a smaller
 * row `v` keeps a relative error of at most `(maximumMagnitude / v) * 2^-30`. The upper cap can
 * quantize even normal float32 values: `1.25 * 2^-126` scales to 2.5, rounds to 2, and decodes as
 * `2^-126`. Pass the true row count of the largest group to keep more bits for small values.
 *
 * Every table that is rolled up or compared together must use the same `sumScale`.
 *
 * @param maximumMagnitude Largest absolute value any row can hold; positive and finite.
 * @param maximumRowCount Largest number of rows added into one sum; an integer in `[1, 2^32]`.
 * Defaults to `2^31`.
 * @returns A power of two in `[2^-126, 2^127]`, valid for the `sumScale` props.
 */
export function getGPUCellSumScale(maximumMagnitude: number, maximumRowCount = 2 ** 31): number {
  if (!Number.isFinite(maximumMagnitude) || maximumMagnitude <= 0) {
    throw new Error(
      `getGPUCellSumScale maximumMagnitude must be positive and finite, got ${maximumMagnitude}`
    );
  }
  if (!Number.isInteger(maximumRowCount) || maximumRowCount < 1 || maximumRowCount > 2 ** 32) {
    throw new Error(
      `getGPUCellSumScale maximumRowCount must be an integer in [1, 2^32], got ${maximumRowCount}`
    );
  }
  // Decode the exact binary64 argument. A rounded quotient/product can erase a one-unit excess
  // over the headroom, including at the 2^127 cap. This is constant-size host arithmetic.
  const words = new DataView(new ArrayBuffer(8));
  words.setFloat64(0, maximumMagnitude);
  const high = words.getUint32(0);
  const low = words.getUint32(4);
  const exponentBits = high >>> 20;
  const significand =
    (BigInt(high & 0xfffff) << 32n) + BigInt(low) + (exponentBits === 0 ? 0n : 1n << 52n);
  const magnitudeExponent = exponentBits === 0 ? -1074 : exponentBits - 1075;
  const product = significand * BigInt(maximumRowCount);
  const isPowerOfTwo = (product & (product - 1n)) === 0n;
  const ceilingLogarithm = product.toString(2).length - Number(isPowerOfTwo);
  const exponent = Math.log2(CELL_SUM_SCALE_HEADROOM) - magnitudeExponent - ceilingLogarithm;
  // The scale is a float32 constant in the kernel: keep it a normal float32 power of two.
  return 2 ** Math.min(Math.max(exponent, -126), 127);
}

/** Validates a fixed-point sum scale. @internal */
export function validateSumScale(id: string, sumScale: number): void {
  if (!Number.isFinite(sumScale) || sumScale <= 0 || Math.fround(sumScale) !== sumScale) {
    throw new Error(`${id} sumScale must be a positive finite float32 value`);
  }
}

/**
 * Order-preserving u32 encoding of f32 (`-0` below `+0`) and a NaN builder.
 *
 * @internal
 */
export const ORDERED_KEY_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${GET_NAN_WGSL}

${GET_ORDERED_KEY_WGSL}

${DECODE_ORDERED_KEY_WGSL}
`;

/** Largest f32 below 2^62: scaled values saturate here so 64-bit sums of 2^31 rows cannot wrap far. */
export const CELL_MAXIMUM_SCALED_VALUE = Math.fround(2 ** 62 - 2 ** 38);

/**
 * Signed 64-bit fixed-point helpers on `vec2u(high, low)`. `cellScaleValue` rounds each value to
 * the nearest multiple of `1 / sumScale` (half to even) before the integer sum, so the sum is
 * order independent and reproducible, and its accuracy is that quantisation: see
 * {@link GPU_CELL_DEFAULT_SUM_SCALE} and {@link getGPUCellSumScale}.
 * @internal
 */
export function getFixedPointWGSL(sumScale: number): string {
  return /* wgsl */ `
const SUM_SCALE: f32 = ${getWGSLFloatLiteral(sumScale)};
const MAXIMUM_SCALED_VALUE: f32 = ${getWGSLFloatLiteral(CELL_MAXIMUM_SCALED_VALUE)};

// Not dggs_u64_add and friends: pulling in the dggs module costs compile time (see cell-keys.ts).
fn cellAddI64(left: vec2u, right: vec2u) -> vec2u {
  let low = left.y + right.y;
  return vec2u(left.x + right.x + select(0u, 1u, low < left.y), low);
}

fn cellNegateI64(value: vec2u) -> vec2u {
  let low = ~value.y + 1u;
  return vec2u(~value.x + select(0u, 1u, low == 0u), low);
}

/** roundHalfEven(value * SUM_SCALE), saturated, as a two's-complement 64-bit integer. */
fn cellScaleValue(value: f32) -> vec2u {
  let scaled = clamp(round(value * SUM_SCALE), -MAXIMUM_SCALED_VALUE, MAXIMUM_SCALED_VALUE);
  let magnitude = abs(scaled);
  let highFloat = floor(magnitude * 2.3283064365386963e-10);
  let lowFloat = magnitude - highFloat * 4294967296.0;
  let result = vec2u(u32(highFloat), u32(lowFloat));
  return select(result, cellNegateI64(result), scaled < 0.0);
}

fn cellI64ToF32(value: vec2u) -> f32 {
  let isNegative = (value.x & 0x80000000u) != 0u;
  let magnitude = select(value, cellNegateI64(value), isNegative);
  let result = cellU64ToF32(magnitude);
  return select(result, -result, isNegative);
}

// Round the full integer once (nearest, ties to even), retaining both words' discarded bits.
fn cellU64ToF32(value: vec2u) -> f32 {
  if (value.x == 0u && value.y <= 0x1000000u) {
    return f32(value.y);
  }
  let highestBit = select(31u - countLeadingZeros(value.y), 63u - countLeadingZeros(value.x), value.x != 0u);
  let discardedCount = highestBit - 23u;
  var significand: u32;
  var aboveHalf: bool;
  var atHalf: bool;
  if (discardedCount < 32u) {
    significand = (value.x << (32u - discardedCount)) | (value.y >> discardedCount);
    let discarded = value.y & ((1u << discardedCount) - 1u);
    let halfway = 1u << (discardedCount - 1u);
    aboveHalf = discarded > halfway;
    atHalf = discarded == halfway;
  } else if (discardedCount == 32u) {
    significand = value.x;
    aboveHalf = value.y > 0x80000000u;
    atHalf = value.y == 0x80000000u;
  } else {
    let highDiscardedCount = discardedCount - 32u;
    significand = value.x >> highDiscardedCount;
    let discarded = value.x & ((1u << highDiscardedCount) - 1u);
    let halfway = 1u << (highDiscardedCount - 1u);
    aboveHalf = discarded > halfway || (discarded == halfway && value.y != 0u);
    atHalf = discarded == halfway && value.y == 0u;
  }
  let increment = select(0u, 1u, aboveHalf || (atHalf && (significand & 1u) != 0u));
  // A carry from rounding the significand naturally increments the float exponent.
  return bitcast<f32>(((highestBit + 127u) << 23u) + (significand - 0x800000u) + increment);
}
`;
}
