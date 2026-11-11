// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `getRoundedDifference(minuend: f32, subtrahend: f32, runtimeZero: u32) -> f32`: the float32
 * difference `minuend - subtrahend`, rounded once and then fixed, so later arithmetic cannot be
 * regrouped around it.
 *
 * Why it exists: a residual such as `(y - mean) - intercept - slope * (x - xmean)` is accurate in
 * float32 only when `y - mean` is formed first (it is exact when the two are close) and the small
 * terms are subtracted from that. The shader compiler of at least one supported GPU (Metal with
 * fast math) regroups such a chain as `y - (mean + intercept + ...)`: measured in
 * `math-review-restricted-regimes.spec.ts`, every residual of a response near 10,000 came out as
 * a multiple of the float32 step of 10,000, and a Chow statistic was 12% off. Writing the
 * difference on its own line does not help, since the compiler regroups across statements.
 *
 * The cure passes the bits of the difference through an integer exclusive-or with a word the
 * compiler cannot know: `runtimeZero` must be a `u32` READ FROM A BUFFER at run time whose value
 * is always 0 (never a literal or a constant expression, which would be folded away). The
 * exclusive-or leaves the value unchanged, and because its operand is the bit pattern of the
 * rounded difference, the subtraction has to be carried out as written.
 *
 * Use it for the first difference of two values that are large against the result, before sums
 * or products with small terms. It costs two bit casts and one integer operation.
 *
 * @internal
 */
export const ROUNDED_DIFFERENCE_WGSL = /* wgsl */ `
fn getRoundedDifference(minuend: f32, subtrahend: f32, runtimeZero: u32) -> f32 {
  return bitcast<f32>(bitcast<u32>(minuend - subtrahend) ^ runtimeZero);
}
`;
