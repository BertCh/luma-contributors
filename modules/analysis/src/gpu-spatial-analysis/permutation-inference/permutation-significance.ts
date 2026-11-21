// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Exact comparison for bounded permutation counts and the stored float32 cutoff. @internal */
export function isPermutationRatioSignificant(
  numerator: number,
  denominator: number,
  level: number
): boolean {
  const bits = new Uint32Array(Float32Array.of(level).buffer)[0];
  if (bits > 0x3f800000 || denominator === 0) return false;
  const exponent = (bits >>> 23) & 255;
  const significand = BigInt((bits & 0x7fffff) + (exponent === 0 ? 0 : 0x800000));
  // A 24-bit cutoff times a 31-bit count can exceed binary64's exact integer
  // range. Keep the dyadic comparison exact on the host as well as in WGSL.
  return (
    BigInt(numerator) << BigInt(150 - Math.max(exponent, 1)) <= significand * BigInt(denominator)
  );
}

/** Compare numerator/denominator to the exact dyadic value of a float32 level. @internal */
export const PERMUTATION_SIGNIFICANCE_WGSL = /* wgsl */ `
fn isPermutationRatioSignificant(numerator: u32, denominator: u32, level: f32) -> bool {
  let bits = bitcast<u32>(level);
  if (bits > 0x3f800000u || denominator == 0u) { return false; }
  if (numerator == 0u) { return true; }
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
  let shift = 150u - max(exponent, 1u);
  if (shift >= 64u) { return false; }
  // Exact significand * denominator, using 16-bit limbs. The threshold is its
  // floor after the binary exponent shift, so the comparison uses integers only.
  let firstLow = significand & 0xffffu;
  let firstHigh = significand >> 16u;
  let secondLow = denominator & 0xffffu;
  let secondHigh = denominator >> 16u;
  let lowProduct = firstLow * secondLow;
  let firstCross = firstHigh * secondLow;
  let secondCross = firstLow * secondHigh;
  let middle = (lowProduct >> 16u) + (firstCross & 0xffffu) + (secondCross & 0xffffu);
  let high = firstHigh * secondHigh + (firstCross >> 16u) + (secondCross >> 16u) + (middle >> 16u);
  let low = (lowProduct & 0xffffu) | (middle << 16u);
  var threshold = 0u;
  if (shift >= 32u) { threshold = high >> (shift - 32u); }
  else { threshold = (high << (32u - shift)) | (low >> shift); }
  return numerator <= threshold;
}
`;
