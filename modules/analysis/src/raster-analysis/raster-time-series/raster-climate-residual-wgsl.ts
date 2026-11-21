// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';

/** Climate mean and scaled residual arithmetic; requires FIXED_POINT_WGSL. @internal */
export const RASTER_CLIMATE_RESIDUAL_WGSL = /* wgsl */ `
${UINT64_WGSL}

fn climateScale(fraction: f32, power: i32) -> f32 {
  let originalBits = bitcast<u32>(fraction);
  let magnitudeBits = originalBits & 0x7fffffffu;
  if (magnitudeBits == 0u || magnitudeBits >= 0x7f800000u) { return fraction; }
  var normalFraction = fraction;
  var adjustment = 0;
  if (magnitudeBits < 0x00800000u) {
    normalFraction = ldexp(fraction, 24);
    adjustment = -24;
  }
  let basePower = i32((bitcast<u32>(normalFraction) >> 23u) & 255u) - 127;
  let finalPower = basePower + adjustment + power;
  if (finalPower > 127) { return bitcast<f32>((originalBits & 0x80000000u) | 0x7f800000u); }
  if (finalPower < -150) { return bitcast<f32>(originalBits & 0x80000000u); }
  let normalizedFraction = ldexp(normalFraction, -basePower);
  // Each ldexp exponent stays in WGSL's defined range, including large finite queries.
  if (finalPower < -126) { return ldexp(ldexp(normalizedFraction, finalPower + 126), -126); }
  return ldexp(normalizedFraction, finalPower);
}

fn climateLeadingBit(magnitude: vec3<u32>) -> i32 {
  if (magnitude.z != 0u) { return 64i + i32(firstLeadingBit(magnitude.z)); }
  if (magnitude.y != 0u) { return 32i + i32(firstLeadingBit(magnitude.y)); }
  if (magnitude.x != 0u) { return i32(firstLeadingBit(magnitude.x)); }
  return -1i;
}

fn climateMean(sum: vec3<u32>, divisor: u32, unit: i32) -> f32 {
  let sign = sum.z & 0x80000000u;
  let magnitude = select(sum, negateFixedPoint(sum), sign != 0u);
  let numeratorLeading = climateLeadingBit(magnitude);
  if (numeratorLeading < 0) { return 0.0; }

  let divisorLeading = i32(firstLeadingBit(divisor));
  let quotientStart = i32(numeratorLeading - divisorLeading);
  // Load the numerator prefix above the first possible quotient bit. It is always < divisor.
  var remainder = 0u;
  if (quotientStart < 0) {
    remainder = magnitude.x << u32(-quotientStart - 1);
  } else if (divisorLeading != 0) {
    remainder = shiftFixedPointRight(magnitude, u32(quotientStart + 1));
  }

  // The first bit may be zero. 26 steps keep all 24 significant bits plus a rounding bit.
  const quotientDigits = 26i;
  var quotient = 0u;
  for (var digit = 0i; digit < quotientDigits; digit++) {
    let position = i32(quotientStart - digit);
    var incoming = 0u;
    if (position >= 0) { incoming = getFixedPointBit(magnitude, u32(position)); }
    let overflow = remainder >> 31u;
    remainder = (remainder << 1u) | incoming;
    quotient = quotient << 1u;
    if (overflow != 0u || remainder >= divisor) {
      remainder = remainder - divisor;
      quotient = quotient | 1u;
    }
  }
  let lastPosition = i32(quotientStart - (quotientDigits - 1i));
  var sticky = remainder != 0u;
  if (lastPosition > 0) {
    sticky = sticky || hasFixedPointBitsBelow(magnitude, u32(lastPosition));
  }

  // Round once, directly to the normal significand or the subnormal quantum (2^-149).
  let digitPower = i32(unit + lastPosition);
  let significandShift = select(1i, 2i, quotient >= 0x2000000u);
  let subnormalShift = i32(-149i - digitPower);
  let roundingShift = max(significandShift, i32(subnormalShift));
  if (roundingShift > quotientDigits) { return bitcast<f32>(sign); }
  let roundBit = 1u << (u32(roundingShift) - 1u);
  let discarded = quotient & ((roundBit << 1u) - 1u);
  var rounded = quotient >> u32(roundingShift);
  if (discarded > roundBit || (discarded == roundBit && (sticky || (rounded & 1u) != 0u))) {
    rounded = rounded + 1u;
  }
  if (i32(subnormalShift) >= significandShift && rounded <= 0x800000u) {
    return bitcast<f32>(sign | rounded);
  }
  // Add the 23 significand bits and the 127 exponent bias, plus any rounding carry.
  let biasedExponent = u32(digitPower + roundingShift + 150i) +
    select(0u, 1u, rounded == 0x1000000u);
  if (biasedExponent >= 255u) { return bitcast<f32>(sign | 0x7f800000u); }
  return bitcast<f32>(sign | (biasedExponent << 23u) | (rounded & 0x7fffffu));
}

fn climateResidual(retainedSum: vec3<u32>, observationCount: u32, query: f32,
  encodingUnit: i32) -> vec2<f32> {
  let queryBits = bitcast<u32>(query);
  let exponentBits = (queryBits >> 23u) & 255u;
  let queryExponent = i32(exponentBits) - 126;
  let queryNegative = (queryBits >> 31u) != 0u;
  let unitShift = select(-149, i32(exponentBits) - 150, exponentBits != 0u) - encodingUnit;
  var querySignificand = (queryBits & 0x7fffffu) | select(0u, 0x800000u, exponentBits != 0u);
  if (unitShift < 0) {
    // Only sub-unit bits need rounding. The rounded significand still fits one word.
    let roundedQuery = quantizeFixedPoint(bitcast<f32>(queryBits & 0x7fffffffu), encodingUnit);
    querySignificand = roundedQuery.x;
  }
  let product = multiplyUint32Wide(querySignificand, observationCount);
  let shift = max(unitShift, 0);
  let leadingProduct = climateLeadingBit(vec3<u32>(product.y, product.x, 0u));
  let leadingEncoded = leadingProduct + shift;
  let isSignedMinimum = queryNegative && leadingEncoded == 95 &&
    countOneBits(product.x) + countOneBits(product.y) == 1u;
  if (leadingProduct >= 0 && leadingEncoded >= 95 && !isSignedMinimum) {
    // A query need not obey the producer's bound. Scale when the actual product cannot fit.
    let commonExponent = max(queryExponent, encodingUnit + 95);
    let scaledMean = climateMean(retainedSum, observationCount, i32(encodingUnit - commonExponent));
    let centered = climateScale(query, -commonExponent) - scaledMean;
    return vec2<f32>(centered, f32(commonExponent));
  }
  let retainedNegative = (retainedSum.z >> 31u) != 0u;
  var multipliedQuery = vec3<u32>(0u);
  if (product.y != 0u) {
    multipliedQuery = shiftFixedPointLeft(product.y, u32(shift));
  }
  if (product.x != 0u) {
    multipliedQuery = multipliedQuery | shiftFixedPointLeft(product.x, u32(shift) + 32u);
  }
  let retainedMagnitude = select(retainedSum, negateFixedPoint(retainedSum), retainedNegative);
  var residualMagnitude = addFixedPoint(multipliedQuery, retainedMagnitude);
  var residualNegative = queryNegative;
  if (queryNegative == retainedNegative) {
    let signedDifference = addFixedPoint(multipliedQuery, negateFixedPoint(retainedMagnitude));
    let differenceNegative = (signedDifference.z >> 31u) != 0u;
    residualMagnitude = select(signedDifference, negateFixedPoint(signedDifference), differenceNegative);
    residualNegative = queryNegative != differenceNegative;
  }
  let leading = climateLeadingBit(residualMagnitude);
  if (leading < 0) { return vec2<f32>(0.0); }
  var normalized = 0.0;
  if (leading == 95) {
    // Preserve unsigned bit 95 and a sticky low bit before signed conversion.
    let halfMagnitude = vec3<u32>((residualMagnitude.x >> 1u) | (residualMagnitude.y << 31u) | (residualMagnitude.x & 1u),
      (residualMagnitude.y >> 1u) | (residualMagnitude.z << 31u), residualMagnitude.z >> 1u);
    normalized = getFixedPointFloat(halfMagnitude, 1 - leading);
  } else { normalized = getFixedPointFloat(residualMagnitude, -leading); }
  return vec2<f32>(select(normalized, -normalized, residualNegative) / f32(observationCount),
    f32(encodingUnit + leading));
}
`;
