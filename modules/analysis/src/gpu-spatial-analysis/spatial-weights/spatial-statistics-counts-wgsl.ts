// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';

/**
 * Exact uint32 counts in float32 statistics. Above 2^24, integer significand arithmetic retains
 * every count bit until the final nearest-even float32 rounding. Below it the existing float32
 * operations are preserved. Uses the shared integer arithmetic, not float compensation (which
 * fast-math compilers may eliminate). @internal
 */
export const SPATIAL_STATISTICS_COUNTS_WGSL = /* wgsl */ `
${UINT64_WGSL}

fn roundSpatialCountValue(magnitude: vec2<u32>, exponentScale: i32, negative: u32, sticky: bool) -> f32 {
  let bitLength = select(32u - countLeadingZeros(magnitude.y), 64u - countLeadingZeros(magnitude.x), magnitude.x != 0u);
  if (bitLength == 0u) { return bitcast<f32>(negative); }
  let subnormal = exponentScale + i32(bitLength) - 1i < -126i;
  let shift = select(i32(bitLength) - 24i, -149i - exponentScale, subnormal);
  if (shift >= 64i) { return bitcast<f32>(negative); }
  var rounded = 0u;
  if (shift <= 0i) {
    rounded = shiftLeftUint64(magnitude, u32(-shift)).y;
  } else {
    let quotient = shiftRightUint64(magnitude, u32(shift));
    let discarded = subtractUint64(magnitude, shiftLeftUint64(quotient, u32(shift)));
    let halfway = shiftLeftUint64(vec2<u32>(0u, 1u), u32(shift - 1i));
    let comparison = compareUint64(discarded, halfway);
    rounded = quotient.y + u32(comparison > 0i || (comparison == 0i && (sticky || (quotient.y & 1u) != 0u)));
  }
  if (subnormal) { return bitcast<f32>(negative | rounded); }
  var exponent = exponentScale + shift + 150i;
  if (rounded >= 0x1000000u) { rounded = rounded >> 1u; exponent++; }
  if (exponent >= 255i) { return bitcast<f32>(negative | 0x7f800000u); }
  return bitcast<f32>(negative | (u32(exponent) << 23u) | (rounded & 0x7fffffu));
}

fn divideBySpatialCount(value: f32, count: u32) -> f32 {
  if (count <= 16777216u) { return value / f32(count); }
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
  if (exponent == 255u || significand == 0u) { return value; }
  let division = divideUint64ByUint32(vec2<u32>(significand, 0u), count);
  var quotient = division.quotient;
  // With no discarded integer bits, the division remainder supplies the rounding decision.
  if (quotient.x == 0u && quotient.y < 0x1000000u && exponent > 32u) {
    let complement = count - division.remainder;
    quotient.y += u32(division.remainder > complement ||
      (division.remainder == complement && (quotient.y & 1u) != 0u));
  }
  return roundSpatialCountValue(quotient, i32(max(exponent, 1u)) - 182i,
    bits & 0x80000000u, division.remainder != 0u && !(quotient.y <= 0x1000000u && exponent > 32u));
}

fn multiplyBySpatialCount(value: f32, count: u32) -> f32 {
  if (count <= 16777216u) { return value * f32(count); }
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
  if (exponent == 255u || significand == 0u) { return value; }
  return roundSpatialCountValue(multiplyUint32Wide(significand, count), i32(max(exponent, 1u)) - 150i,
    bits & 0x80000000u, false);
}

fn divideSpatialCountByValue(count: u32, value: f32) -> f32 {
  if (count <= 16777216u) { return f32(count) / value; }
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let fraction = bits & 0x7fffffu;
  let negative = bits & 0x80000000u;
  if (exponent == 255u) { return select(bitcast<f32>(negative), value, fraction != 0u); }
  let significand = fraction | select(0u, 0x800000u, exponent != 0u);
  if (significand == 0u) { return bitcast<f32>(negative | 0x7f800000u); }
  let division = divideUint64ByUint32(vec2<u32>(count, 0u), significand);
  return roundSpatialCountValue(division.quotient, 118i - i32(max(exponent, 1u)),
    negative, division.remainder != 0u);
}

fn divideSpatialCounts(numerator: u32, denominator: u32) -> f32 {
  if (numerator <= 16777216u && denominator <= 16777216u) { return f32(numerator) / f32(denominator); }
  if (numerator == 0u) { return 0.0; }
  let shift = 31u + countLeadingZeros(numerator);
  let division = divideUint64ByUint32(shiftLeftUint64(vec2<u32>(0u, numerator), shift), denominator);
  return roundSpatialCountValue(division.quotient, -i32(shift), 0u, division.remainder != 0u);
}

`;
