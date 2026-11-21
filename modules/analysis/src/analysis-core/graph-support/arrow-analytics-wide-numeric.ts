// vis.gl analysis
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataType, type Vector} from 'apache-arrow';
import type {
  ArrowAnalyticsColumnConversion,
  ArrowAnalyticsFloat64Policy
} from './arrow-analytics-normalization';

/** True for numeric Arrow types that only reach Float32 through the explicit Float64 precision policy. */
export function isWideNumericType(type: DataType): boolean {
  return (
    (DataType.isInt(type) && type.bitWidth === 64) ||
    (DataType.isDecimal(type) && type.bitWidth >= 64)
  );
}

/**
 * Converts one Int64, Uint64 or Decimal value to a JavaScript number. Rejects integers that
 * binary64 cannot represent; Decimal coefficients and scales round once to binary64. For precision-policy
 * enforcement against original coefficients use prepareArrowTableForGPUAnalytics instead.
 */
export function convertWideNumericValue(type: DataType, value: unknown): number | null {
  if (value === null || value === undefined) return null;
  // JavaScript numbers are already converted values, including for Decimal source metadata.
  if (typeof value === 'number') return value;
  if (DataType.isDecimal(type)) {
    const coefficient = getWideNumericCoefficient(value);
    if (coefficient === 0n) return 0;
    // ECMAScript decimal string conversion rounds the original exact coefficient/exponent
    // once, ties to even. No intermediate power of ten can overflow or underflow.
    return Number(`${coefficient}e${-type.scale}`);
  }
  if (typeof value === 'bigint') {
    const rounded = Number(value);
    // Float64 cannot hold this integer; do not hide the loss from the Float32 error budget.
    if (BigInt(rounded) !== value) throw new Error(`${type} column exceeds its precision policy`);
    return rounded;
  }
  // Decimal values are little-endian two's-complement words scaled by 10^-scale.
  const words = value as ArrayLike<number>;
  let integer = BigInt(0);
  for (let index = words.length - 1; index >= 0; index--)
    integer = (integer << BigInt(32)) | BigInt(words[index] >>> 0);
  const bitWidth = words.length * 32;
  if (integer >= BigInt(1) << BigInt(bitWidth - 1)) integer -= BigInt(1) << BigInt(bitWidth);
  return Number(integer);
}

/** Maps a chunk through JavaScript numbers; this does not check original Decimal precision. */
export function convertWideNumericVector(
  type: DataType,
  vector: Vector,
  convertNumber: (value: number | null) => number | null
): (number | null)[] {
  return Array.from(vector, value => convertNumber(convertWideNumericValue(type, value)));
}

type NumericRational = {numerator: bigint; denominator: bigint};

/** @internal Narrows original Arrow coefficients, without a lossy binary64 intermediate. */
export function narrowWideNumericValue(
  type: DataType,
  value: unknown,
  policy: ArrowAnalyticsFloat64Policy,
  report: ArrowAnalyticsColumnConversion
): number | null {
  if (value === null || value === undefined) return null;
  const reject = () => {
    throw new Error(`${type} column "${report.name}" exceeds its precision policy`);
  };
  const coefficient = getWideNumericCoefficient(value);
  if (coefficient === 0n) return 0;
  const scale = DataType.isDecimal(type) ? type.scale : 0;
  const decimalOrder = absoluteInteger(coefficient).toString().length - scale;
  // An order >= 40 has magnitude >= 10^39, beyond every finite Float32 candidate.
  if (decimalOrder >= 40) return reject();
  // An order <= -324 has magnitude < 10^-324, below half the smallest binary64
  // subnormal. Its Float32 candidate and reported absolute error both round to zero.
  // Every positive binary64 absolute budget exceeds it; relative error is exactly 1.
  if (decimalOrder <= -324) {
    if (
      policy.mode === 'exact' ||
      (policy.maximumAbsoluteError === 0 && policy.maximumRelativeError < 1)
    ) {
      return reject();
    }
    report.maximumRelativeError = Math.max(report.maximumRelativeError, 1);
    return coefficient < 0n ? -0 : 0;
  }
  // Only the bounded exponent range that can affect either Float32 or error reports
  // reaches a power of ten; Arrow scales themselves span all signed 32-bit integers.
  const source =
    scale >= 0
      ? {numerator: coefficient, denominator: 10n ** BigInt(scale)}
      : {numerator: coefficient * 10n ** BigInt(-scale), denominator: 1n};
  const magnitude = absoluteInteger(source.numerator);
  const rounded = Math.fround(roundPositiveRational(magnitude, source.denominator, 24, -149));
  const candidate = source.numerator < 0n ? -rounded : rounded;
  if (!Number.isFinite(candidate)) return reject();
  const target = getNumberRational(rounded);
  const errorNumerator = absoluteInteger(
    magnitude * target.denominator - target.numerator * source.denominator
  );
  if (errorNumerator === 0n) return candidate;
  if (policy.mode === 'exact') return reject();
  const absoluteBudget = getNumberRational(policy.maximumAbsoluteError);
  const relativeBudget = getNumberRational(policy.maximumRelativeError);
  // |source - candidate| <= absoluteBudget + relativeBudget * |source|.
  // Cross multiplication keeps even zero and boundary budgets exact.
  if (
    errorNumerator * absoluteBudget.denominator * relativeBudget.denominator >
    target.denominator *
      (absoluteBudget.numerator * relativeBudget.denominator * source.denominator +
        relativeBudget.numerator * absoluteBudget.denominator * magnitude)
  ) {
    return reject();
  }
  report.maximumAbsoluteError = Math.max(
    report.maximumAbsoluteError,
    roundPositiveRational(errorNumerator, source.denominator * target.denominator, 53, -1074)
  );
  report.maximumRelativeError = Math.max(
    report.maximumRelativeError,
    roundPositiveRational(errorNumerator, magnitude * target.denominator, 53, -1074)
  );
  return candidate;
}

function getWideNumericCoefficient(value: unknown): bigint {
  let coefficient: bigint;
  if (typeof value === 'bigint') {
    coefficient = value;
  } else {
    const words = value as ArrayLike<number>;
    coefficient = 0n;
    for (let index = words.length - 1; index >= 0; index--) {
      coefficient = (coefficient << 32n) | BigInt(words[index] >>> 0);
    }
    coefficient = BigInt.asIntN(words.length * 32, coefficient);
  }
  return coefficient;
}

function absoluteInteger(value: bigint): bigint {
  return value < 0n ? -value : value;
}

/** Exact binary64 fraction; also represents every finite Float32 candidate exactly. */
function getNumberRational(value: number): NumericRational {
  const bytes = new DataView(new ArrayBuffer(8));
  bytes.setFloat64(0, value);
  const bits = bytes.getBigUint64(0);
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const significand = (bits & ((1n << 52n) - 1n)) | (exponentBits ? 1n << 52n : 0n);
  const exponent = (exponentBits || 1) - 1023 - 52;
  return exponent >= 0
    ? {numerator: significand << BigInt(exponent), denominator: 1n}
    : {numerator: significand, denominator: 1n << BigInt(-exponent)};
}

/** Round once, ties to even, to the requested significand width and subnormal quantum. */
function roundPositiveRational(
  numerator: bigint,
  denominator: bigint,
  precision: number,
  minimumExponent: number
): number {
  if (numerator === 0n) return 0;
  let exponent = numerator.toString(2).length - denominator.toString(2).length;
  const belowPower =
    exponent >= 0
      ? numerator < denominator << BigInt(exponent)
      : numerator << BigInt(-exponent) < denominator;
  if (belowPower) exponent--;
  const shift = Math.max(minimumExponent, exponent - (precision - 1));
  const scaledNumerator = shift < 0 ? numerator << BigInt(-shift) : numerator;
  const scaledDenominator = shift > 0 ? denominator << BigInt(shift) : denominator;
  let significand = scaledNumerator / scaledDenominator;
  const doubledRemainder = (scaledNumerator % scaledDenominator) * 2n;
  if (
    doubledRemainder > scaledDenominator ||
    (doubledRemainder === scaledDenominator && (significand & 1n) !== 0n)
  ) {
    significand++;
  }
  return Number(significand) * 2 ** shift;
}
