// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared WGSL double-single (float-float) helpers. @internal
 *
 * A double-single value is a `vec2<f32>` whose `x` component is the rounded sum and whose `y`
 * component is the residual rounding error, so `x + y` carries roughly 48 bits of significand.
 *
 * Shader compilers with fast-math enabled (observed on Metal) reassociate float expressions, which
 * turns the textbook error-free transforms into zero. `keepFloat` routes each intermediate through
 * an integer OR with a runtime zero, which no float optimization can see through. The entry point
 * must assign `opaqueZero = workgroupId.x >> 31u;` (any runtime value that is always zero) before
 * calling these helpers.
 *
 * Comparisons use the high parts first and the low parts only on exact ties, which stays exact
 * even if the low parts were lost.
 */
export const DOUBLE_SINGLE_WGSL = /* wgsl */ `
var<private> opaqueZero: u32;

// Hides a float result from algebraic simplification of the surrounding expression.
fn keepFloat(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) | opaqueZero);
}

// Knuth two-sum: returns (sum, exact rounding error); non-finite sums carry a zero error.
fn twoSum(a: f32, b: f32) -> vec2<f32> {
  let sum = keepFloat(a + b);
  // The error of an overflowed or non-finite sum is undefined; the sum alone is the answer.
  if (!(abs(sum) <= 3.402823466e+38)) { return vec2<f32>(sum, 0.0); }
  let virtualB = keepFloat(sum - a);
  let aError = keepFloat(a - keepFloat(sum - virtualB));
  let bError = keepFloat(b - virtualB);
  return vec2<f32>(sum, aError + bError);
}

// Adds two double-single values and renormalizes so the high part holds the rounded total.
fn addDoubleSingle(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let highSum = twoSum(a.x, b.x);
  let low = a.y + b.y + highSum.y;
  let high = keepFloat(highSum.x + low);
  if (!(abs(high) <= 3.402823466e+38)) { return vec2<f32>(high, 0.0); }
  return vec2<f32>(high, low - keepFloat(high - highSum.x));
}

// Exact double-single a >= b: compare the high parts first and the low parts only on ties.
fn isDoubleSingleAtLeast(aHigh: f32, aLow: f32, bHigh: f32, bLow: f32) -> bool {
  return aHigh > bHigh || (aHigh == bHigh && aLow >= bLow);
}`;
