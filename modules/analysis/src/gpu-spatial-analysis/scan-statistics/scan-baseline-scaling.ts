// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Binary exposure scaling reads IEEE words before arithmetic, including positive subnormals. @internal */
export const SCAN_BASELINE_SCALING_WGSL = /* wgsl */ `
// A baseline word is valid when finite and not negative; -0 (0x80000000) counts as zero.
fn isScanBaselineValid(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u && bits <= 0x80000000u;
}
fn getScanBaselineExponent(bits: u32) -> i32 {
  let encoded = (bits >> 23u) & 255u;
  if (encoded != 0u) { return i32(encoded) - 127; }
  if (bits == 0u) { return 0; }
  return i32(31u - countLeadingZeros(bits)) - 149;
}
// The working exposure table deliberately uses normal float32 values on every backend.
// Report loss before a positive exposure becomes zero, even on devices that flush subnormals.
fn hasScanBaselineScalingLoss(bits: u32, maximumBits: u32) -> bool {
  let magnitude = bits & 0x7fffffffu;
  return isScanBaselineValid(bits) && magnitude != 0u && maximumBits != 0u &&
    getScanBaselineExponent(magnitude) - getScanBaselineExponent(maximumBits) < -126;
}
fn getScaledScanBaseline(bits: u32, maximumBits: u32) -> f32 {
  if (bits == 0u || maximumBits == 0u) { return 0.0; }
  var fraction = bits & 0x7fffffu;
  if ((bits & 0x7f800000u) == 0u) {
    let leading = 31u - countLeadingZeros(bits);
    fraction = (bits << (23u - leading)) & 0x7fffffu;
  }
  let significand = bitcast<f32>(fraction | 0x3f800000u);
  let shift = getScanBaselineExponent(bits) - getScanBaselineExponent(maximumBits);
  // Subnormal normalized contributions are below the working f32 range; avoid out-of-range ldexp.
  if (shift < -126) { return 0.0; }
  return ldexp(significand, shift);
}
`;
