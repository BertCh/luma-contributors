// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Small WGSL helper declarations shared by the GPU contributor kernels.
 *
 * Each constant declares exactly one function, so a kernel interpolates only what it calls.
 * Declaring a function twice in one shader fails compilation, so kernels that concatenate several
 * declaration strings must include each constant once.
 *
 * Float tests (contributor contract rule 11). New kernels use exactly these three names:
 *
 * | Function | Constant | Meaning |
 * | --- | --- | --- |
 * | `getNaN()` | {@link GET_NAN_WGSL} | a quiet NaN (`0x7fc00000`) |
 * | `isFiniteValue(value)` | {@link IS_FINITE_VALUE_WGSL} | neither infinite nor NaN |
 * | `isNanValue(value)` | {@link IS_NAN_VALUE_WGSL} | NaN, any payload and sign |
 *
 * Both tests read the bit pattern. `value != value` and comparisons with infinity are not used:
 * WGSL lets implementations assume floats are finite, so a compiler may fold `x != x` to `false`.
 * The bit tests are pinned on the test GPU by `test/utils/wgsl-common-nan.spec.ts` and
 * `test/utils/wgsl-float-tests.spec.ts`. `getQuietNaN(seed)` ({@link GET_QUIET_NAN_SEEDED_WGSL})
 * stays for kernels that need a NaN depending on a runtime operand. The other names
 * (`getQuietNaN()`, `isNanBits`, `isFiniteBits`, `isFiniteFloat`, and `isSpatialFinite` /
 * `getSpatialNaN` in `gpu-spatial-analysis/shared`) are older spellings of the same tests.
 */

/**
 * `getNaN()`: a quiet NaN. WGSL rejects NaN constants, so it is built from a runtime bit pattern.
 * @internal
 */
export const GET_NAN_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
`;

/**
 * `getQuietNaN()`: same as {@link GET_NAN_WGSL} under the name the raster and terrain kernels use.
 * @deprecated Use {@link GET_NAN_WGSL} (`getNaN()`).
 * @internal
 */
export const GET_QUIET_NAN_WGSL = /* wgsl */ `
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
`;

/**
 * `getQuietNaN(seed)`: a quiet NaN that depends on a runtime operand (`seed & 0u` is always zero),
 * which keeps the bit pattern out of constant evaluation. @internal
 */
export const GET_QUIET_NAN_SEEDED_WGSL = /* wgsl */ `
fn getQuietNaN(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }
`;

/**
 * `isNanBits(x)`: NaN test on the bit pattern, immune to compiler folding of `x != x`.
 * @deprecated Use {@link IS_NAN_VALUE_WGSL} (`isNanValue(value)`).
 * @internal
 */
export const IS_NAN_BITS_WGSL = /* wgsl */ `
fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}
`;

/**
 * `isNanValue(value)`: true for every NaN (either sign, any payload). Reads the bit pattern: the
 * exponent is all ones and the mantissa is not zero.
 * @internal
 */
export const IS_NAN_VALUE_WGSL = /* wgsl */ `
fn isNanValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
`;

/**
 * `isFiniteBits(x)`: finite test on the exponent bits.
 * @deprecated Use {@link IS_FINITE_VALUE_WGSL} (`isFiniteValue(value)`).
 * @internal
 */
export const IS_FINITE_BITS_WGSL = /* wgsl */ `
fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
`;

/**
 * `isFiniteValue(value)`: false for infinities and NaN, true otherwise (zeros and subnormals are
 * finite). Reads the exponent bits.
 * @internal
 */
export const IS_FINITE_VALUE_WGSL = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`;

/**
 * `isFiniteFloat(value)`: same as {@link IS_FINITE_VALUE_WGSL} under the name the older kernels use.
 * @deprecated Use {@link IS_FINITE_VALUE_WGSL} (`isFiniteValue(value)`).
 * @internal
 */
export const IS_FINITE_FLOAT_WGSL = /* wgsl */ `
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`;

/**
 * `getOrderedKey(x)`: order-preserving `u32` key of an `f32`, so unsigned comparison orders values
 * numerically. The order is total: negative NaNs, `-inf`, negative values, `-0`, `+0`, positive
 * values, `+inf`, positive NaNs. Keys of distinct bit patterns are distinct, so
 * {@link DECODE_ORDERED_KEY_WGSL} restores the exact bits. Use it for sort keys, atomic min and max
 * on floats, and the `'f32'` binary search of `binary-search-wgsl.ts`. @internal
 */
export const GET_ORDERED_KEY_WGSL = /* wgsl */ `
fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;

/** `decodeOrderedKey(key)`: inverse of {@link GET_ORDERED_KEY_WGSL}. @internal */
export const DECODE_ORDERED_KEY_WGSL = /* wgsl */ `
fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`;

/** `spreadBits(value)`: spreads the low 16 bits of `value` to even bit positions (Morton order). @internal */
export const SPREAD_BITS_WGSL = /* wgsl */ `
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
`;

/**
 * `nodeOverlaps(node, queryMinimum, queryMaximum)`: closed-box overlap of a BVH node against a
 * query box. Requires `nodeMinima` and `nodeMaxima` bindings. @internal
 */
export const NODE_OVERLAPS_WGSL = /* wgsl */ `
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}
`;

/**
 * `guardedAtan2(y, x)`: `atan2(y, x)` with a zero second argument of either sign treated as +0.0.
 *
 * Contract (contributor contract rule 11, FX-17):
 * - When `x` is not zero, the result is exactly the builtin `atan2(y, x)`.
 * - When `x` is +0.0 or -0.0 and `y` is not zero, the result is `sign(y) * pi / 2`, as for
 *   `atan2(y, +0.0)`.
 * - When both arguments are zero (either sign), the result is zero (its sign is not specified),
 *   where the builtin returns NaN on the test GPU.
 * - The sign of a zero `y` with a negative `x` is not kept: the result is +pi or -pi, whichever the
 *   builtin returns (+pi on the test GPU for both signs).
 * - Subnormal arguments may be flushed to zero by the GPU (the test GPU flushes them), in which
 *   case they follow the zero rows.
 *
 * Why: on the headless test GPU (Apple, Chromium WebGPU) `atan2(y, -0.0)` returns
 * `-sign(y) * pi / 2` while `atan2(y, +0.0)` is correct. A second argument that is a negation, a
 * difference or a dot product can be -0.0. Use this helper for every `atan2` whose second argument
 * can be exactly zero. Pinned by `test/utils/wgsl-guarded-atan2.spec.ts`. @internal
 */
export const GUARDED_ATAN2_WGSL = /* wgsl */ `
fn guardedAtan2(y: f32, x: f32) -> f32 {
  return select(atan2(y, x), sign(y) * 1.5707963267948966, x == 0.0);
}
`;

/** `erfcValue(x)`: complementary error function, Numerical Recipes Chebyshev fit (relative error under 1.2e-7). @internal */
export const ERFC_VALUE_WGSL = /* wgsl */ `
fn erfcValue(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(2.0 - r, r, x >= 0.0);
}
`;
