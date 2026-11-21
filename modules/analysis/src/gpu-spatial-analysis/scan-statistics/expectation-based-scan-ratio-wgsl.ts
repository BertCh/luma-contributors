// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';

/** Exact ordering of ratios of finite nonnegative float32 sums, without forming a quotient. @internal */
export const EXPECTATION_BASED_SCAN_RATIO_WGSL = /* wgsl */ `${UINT64_WGSL}
struct ScanRatioParts { significand: u32, exponent: i32 }
fn getScanRatioParts(value: f32) -> ScanRatioParts {
  let bits = bitcast<u32>(value);
  let exponent = i32((bits >> 23u) & 255u);
  let fraction = bits & 0x7fffffu;
  if (exponent != 0i) { return ScanRatioParts(fraction | 0x800000u, exponent); }
  let shift = countLeadingZeros(fraction) - 8u;
  return ScanRatioParts(fraction << shift, 1i - i32(shift));
}
fn compareScanRatios(leftCases: f32, leftBaseline: f32, rightCases: f32, rightBaseline: f32) -> i32 {
  let leftValid = leftBaseline > 0.0;
  let rightValid = rightBaseline > 0.0;
  if (leftValid != rightValid) { return select(-1i, 1i, leftValid); }
  if (!leftValid) { return 0i; }
  if (leftCases == 0.0 || rightCases == 0.0) {
    if (leftCases == rightCases) { return 0i; }
    return select(-1i, 1i, leftCases > rightCases);
  }
  let leftCount = getScanRatioParts(leftCases);
  let rightCount = getScanRatioParts(rightCases);
  let leftWeight = getScanRatioParts(leftBaseline);
  let rightWeight = getScanRatioParts(rightBaseline);
  // Cross-products contain at most 48 significant bits. Their exponent difference decides
  // the order beyond one bit; otherwise align exactly in uint64 (at most 49 bits).
  let difference = leftCount.exponent + rightWeight.exponent - rightCount.exponent - leftWeight.exponent;
  if (difference > 1i) { return 1i; }
  if (difference < -1i) { return -1i; }
  var left = multiplyUint32Wide(leftCount.significand, rightWeight.significand);
  var right = multiplyUint32Wide(rightCount.significand, leftWeight.significand);
  if (difference > 0i) { left = shiftLeftUint64(left, 1u); }
  if (difference < 0i) { right = shiftLeftUint64(right, 1u); }
  return compareUint64(left, right);
}`;
