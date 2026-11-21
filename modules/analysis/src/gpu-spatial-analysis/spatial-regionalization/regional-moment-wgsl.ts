// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';

/** Split moments retain subtraction and reduction residuals through the final contrast. @internal */
export const REGIONAL_MOMENT_WGSL = `${ADDITION_ROUNDING_ERROR_WGSL}
${ROUNDED_DIFFERENCE_WGSL}
// The shared helper reports no error when the smaller operand is below 2^-103, because a general
// error in that range can be subnormal. Two exact cases still hold a normal residual: when the
// exponents are more than 25 apart the whole smaller operand is the error, and otherwise every
// value is below 2^-77, so scaling by 2^64 is exact and lifts the error out of the subnormal range.
fn getRegionalAdditionError(first: f32, second: f32, sum: f32) -> f32 {
  let firstMagnitude = bitcast<u32>(first) & 0x7fffffffu;
  let secondMagnitude = bitcast<u32>(second) & 0x7fffffffu;
  let smallerMagnitude = min(firstMagnitude, secondMagnitude);
  let largerMagnitude = max(firstMagnitude, secondMagnitude);
  if (smallerMagnitude == 0u || smallerMagnitude >= 0x0c000000u || largerMagnitude >= 0x7f800000u) {
    return getAdditionRoundingError(first, second, sum);
  }
  if ((largerMagnitude >> 23u) > max(smallerMagnitude >> 23u, 1u) + 25u) {
    let smaller = select(first, second, firstMagnitude >= secondMagnitude);
    let larger = select(second, first, firstMagnitude >= secondMagnitude);
    return select(0.0, smaller, bitcast<u32>(sum) == bitcast<u32>(larger));
  }
  return ldexp(getAdditionRoundingError(ldexp(first, 64), ldexp(second, 64), ldexp(sum, 64)), -64);
}
fn addRegionalMoment(first: vec2<f32>, second: vec2<f32>, zero: u32) -> vec2<f32> {
  let total = getRoundedDifference(first.x, -second.x, zero);
  let residual = first.y + second.y + getRegionalAdditionError(first.x, second.x, total);
  let high = getRoundedDifference(total, -residual, zero);
  return vec2<f32>(high, getRegionalAdditionError(total, residual, high));
}
fn subtractRegionalValue(first: f32, second: f32, zero: u32) -> vec2<f32> {
  let high = getRoundedDifference(first, second, zero);
  return vec2<f32>(high, getRegionalAdditionError(first, -second, high));
}
// The split product of a split moment and a factor: an exact two-product of the high lanes from
// 12-bit significand halves (each partial product is exact), plus the low lane times the factor.
fn multiplyRegionalMoment(value: vec2<f32>, factor: f32, zero: u32) -> vec2<f32> {
  let product = getRoundedDifference(value.x * factor, 0.0, zero);
  let valueHigh = bitcast<f32>(bitcast<u32>(value.x) & 0xfffff000u);
  let factorHigh = bitcast<f32>(bitcast<u32>(factor) & 0xfffff000u);
  let valueLow = getRoundedDifference(value.x, valueHigh, zero);
  let factorLow = getRoundedDifference(factor, factorHigh, zero);
  var productError = getRoundedDifference(valueHigh * factorHigh, product, zero);
  productError = getRoundedDifference(productError, -(valueHigh * factorLow), zero);
  productError = getRoundedDifference(productError, -(valueLow * factorHigh), zero);
  productError = getRoundedDifference(productError, -(valueLow * factorLow), zero);
  return addRegionalMoment(vec2<f32>(product, 0.0), vec2<f32>(productError, value.y * factor), zero);
}
fn divideRegionalMoment(value: vec2<f32>, count: f32, zero: u32) -> vec2<f32> {
  let quotient = getRoundedDifference(value.x / count, 0.0, zero);
  // Split the product into 12-bit significands. Every product in the error expression is
  // exact; runtime bit barriers stop Metal from folding the error back to zero.
  let quotientHigh = bitcast<f32>(bitcast<u32>(quotient) & 0xfffff000u);
  let countHigh = bitcast<f32>(bitcast<u32>(count) & 0xfffff000u);
  let quotientLow = getRoundedDifference(quotient, quotientHigh, zero);
  let countLow = getRoundedDifference(count, countHigh, zero);
  let product = getRoundedDifference(quotient * count, 0.0, zero);
  var productError = getRoundedDifference(quotientHigh * countHigh, product, zero);
  productError = getRoundedDifference(productError, -(quotientHigh * countLow), zero);
  productError = getRoundedDifference(productError, -(quotientLow * countHigh), zero);
  productError = getRoundedDifference(productError, -(quotientLow * countLow), zero);
  let residual = addRegionalMoment(subtractRegionalValue(value.x, product, zero),
    vec2<f32>(value.y, -productError), zero);
  return addRegionalMoment(vec2<f32>(quotient, 0.0),
    vec2<f32>((residual.x + residual.y) / count, 0.0), zero);
}
`;
