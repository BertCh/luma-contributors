// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure TypeScript reference and WGSL-equivalent emulation for exact 64-bit (and 128-bit) integer
 * arithmetic on (low, high) uint32 word pairs. The emulation functions deliberately use only
 * 32-bit operations (`>>> 0`, `Math.imul`, `Math.clz32`) in the same order as the WGSL helpers in
 * wide-numeric-gpu-wgsl.ts, so node specs verify the algorithms without first GPU. The BigInt
 * functions are the independent oracle.
 */

import {
  diagnoseFloat32Narrowing,
  type NumericNarrowingReport,
  type WideNumericFormat
} from './numeric-narrowing-diagnostics';

/** A 64-bit value as [low, high] unsigned 32-bit words (little-endian element order). */
export type WordPair = [low: number, high: number];
export type WideNumericGpuFormat = 'float64' | 'int64' | 'uint64';

const TWO_POW_32 = 0x100000000;
const SIGN_BIT = 0x80000000;

// ---------- BigInt conversions ----------

export function splitBigIntToWords(value: bigint): WordPair {
  const bits = BigInt.asUintN(64, value);
  return [Number(bits & 0xffffffffn), Number(bits >> 32n)];
}

export function joinWordsToBigInt(pair: readonly number[], signed: boolean): bigint {
  const bits = (BigInt(pair[1] >>> 0) << 32n) | BigInt(pair[0] >>> 0);
  return signed ? BigInt.asIntN(64, bits) : bits;
}

export function splitDoubleToWords(value: number): WordPair {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, true);
  return [view.getUint32(0, true), view.getUint32(4, true)];
}

// ---------- 64-bit pair emulation (mirrors WGSL) ----------

export function addWords(first: readonly number[], second: readonly number[]): WordPair {
  const low = (first[0] + second[0]) >>> 0;
  return [low, (first[1] + second[1] + (low < first[0] ? 1 : 0)) >>> 0];
}

export function subtractWords(first: readonly number[], second: readonly number[]): WordPair {
  const low = (first[0] - second[0]) >>> 0;
  return [low, (first[1] - second[1] - (first[0] < second[0] ? 1 : 0)) >>> 0];
}

export function negateWords(first: readonly number[]): WordPair {
  return subtractWords([0, 0], first);
}

export function isUnsignedLess(first: readonly number[], second: readonly number[]): boolean {
  return first[1] < second[1] || (first[1] === second[1] && first[0] < second[0]);
}

export function isSignedLess(first: readonly number[], second: readonly number[]): boolean {
  return isUnsignedLess(
    [first[0], (first[1] ^ SIGN_BIT) >>> 0],
    [second[0], (second[1] ^ SIGN_BIT) >>> 0]
  );
}

/** Full 32x32 -> 64 bit product through 16-bit limbs. */
export function multiplyWords32(first: number, second: number): WordPair {
  const firstLowHalf = first & 0xffff;
  const firstHighHalf = first >>> 16;
  const secondLowHalf = second & 0xffff;
  const secondHighHalf = second >>> 16;
  const lowHalfProduct = (firstLowHalf * secondLowHalf) >>> 0;
  const firstCrossHalfProduct = (firstLowHalf * secondHighHalf) >>> 0;
  const secondCrossHalfProduct = (firstHighHalf * secondLowHalf) >>> 0;
  const highHalfProduct = (firstHighHalf * secondHighHalf) >>> 0;
  const middle =
    ((lowHalfProduct >>> 16) +
      (firstCrossHalfProduct & 0xffff) +
      (secondCrossHalfProduct & 0xffff)) >>>
    0;
  const low = ((lowHalfProduct & 0xffff) | (middle << 16)) >>> 0;
  const high =
    (highHalfProduct +
      (firstCrossHalfProduct >>> 16) +
      (secondCrossHalfProduct >>> 16) +
      (middle >>> 16)) >>>
    0;
  return [low, high];
}

/** Low 64 bits of the product (identical for signed and unsigned operands). */
export function multiplyWordsLow(first: readonly number[], second: readonly number[]): WordPair {
  const product = multiplyWords32(first[0], second[0]);
  return [
    product[0],
    (product[1] + Math.imul(first[0], second[1]) + Math.imul(first[1], second[0])) >>> 0
  ];
}

/** Full unsigned 64x64 -> 128 bit product as four little-endian words. */
export function multiplyWords128(
  first: readonly number[],
  second: readonly number[]
): [number, number, number, number] {
  const lowProduct = multiplyWords32(first[0], second[0]);
  const firstCrossProduct = multiplyWords32(first[0], second[1]);
  const secondCrossProduct = multiplyWords32(first[1], second[0]);
  const highProduct = multiplyWords32(first[1], second[1]);
  let carryOne = 0;
  let middle = (lowProduct[1] + firstCrossProduct[0]) >>> 0;
  if (middle < lowProduct[1]) carryOne++;
  const middleSum = (middle + secondCrossProduct[0]) >>> 0;
  if (middleSum < middle) carryOne++;
  middle = middleSum;
  let carryTwo = 0;
  let upper = firstCrossProduct[1];
  let next = (upper + secondCrossProduct[1]) >>> 0;
  if (next < upper) carryTwo++;
  upper = next;
  next = (upper + highProduct[0]) >>> 0;
  if (next < upper) carryTwo++;
  upper = next;
  next = (upper + carryOne) >>> 0;
  if (next < upper) carryTwo++;
  upper = next;
  return [lowProduct[0], middle, upper, (highProduct[1] + carryTwo) >>> 0];
}

export function shiftWordsRight(first: readonly number[], amount: number): WordPair {
  if (amount === 0) return [first[0], first[1]];
  if (amount < 32)
    return [((first[0] >>> amount) | (first[1] << (32 - amount))) >>> 0, first[1] >>> amount];
  return [first[1] >>> (amount - 32), 0];
}

export function shiftWordsLeft(first: readonly number[], amount: number): WordPair {
  if (amount === 0) return [first[0], first[1]];
  if (amount < 32)
    return [
      (first[0] << amount) >>> 0,
      ((first[1] << amount) | (first[0] >>> (32 - amount))) >>> 0
    ];
  return [0, (first[0] << (amount - 32)) >>> 0];
}

export function countLeadingZeroWords(first: readonly number[]): number {
  return first[1] !== 0 ? Math.clz32(first[1]) : 32 + Math.clz32(first[0]);
}

/** Unsigned division by first 32-bit divisor, bit by bit. Zero divisor returns all ones. */
export function divideWordsBySmall(
  first: readonly number[],
  divisor: number
): {quotient: WordPair; remainder: number} {
  if (divisor === 0) return {quotient: [0xffffffff, 0xffffffff], remainder: 0xffffffff};
  const quotient: WordPair = [0, 0];
  let remainder = 0;
  for (let bit = 63; bit >= 0; bit--) {
    const amount = bit & 31;
    const source = ((bit >= 32 ? first[1] : first[0]) >>> amount) & 1;
    const carry = remainder >>> 31;
    remainder = ((remainder << 1) | source) >>> 0;
    let quotientBit = 0;
    if (carry !== 0 || remainder >= divisor) {
      remainder = (remainder - divisor) >>> 0;
      quotientBit = 1;
    }
    if (bit >= 32) quotient[1] = (quotient[1] | (quotientBit << amount)) >>> 0;
    else quotient[0] = (quotient[0] | (quotientBit << amount)) >>> 0;
  }
  return {quotient, remainder};
}

/** Truncating signed division: quotient is negated for first negative dividend; remainder is first magnitude plus sign. */
export function divideSignedWordsBySmall(
  first: readonly number[],
  divisor: number
): {quotient: WordPair; remainderMagnitude: number; remainderNegative: boolean} {
  const negative = (first[1] & SIGN_BIT) !== 0;
  const result = divideWordsBySmall(negative ? negateWords(first) : first, divisor);
  return {
    quotient: negative && divisor !== 0 ? negateWords(result.quotient) : result.quotient,
    remainderMagnitude: result.remainder,
    remainderNegative: negative
  };
}

/** Round-to-nearest-even conversion of first 64-bit integer to Float32 with an "exact in Float32" flag. */
export function convertWordsToFloat32(
  first: readonly number[],
  signed: boolean
): {value: number; exact: boolean} {
  const negative = signed && (first[1] & SIGN_BIT) !== 0;
  const magnitude = negative ? negateWords(first) : [first[0], first[1]];
  const result = convertUnsignedWordsToFloat32(magnitude);
  return {value: negative ? -result.value : result.value, exact: result.exact};
}

function convertUnsignedWordsToFloat32(first: readonly number[]): {value: number; exact: boolean} {
  if (first[0] === 0 && first[1] === 0) return {value: 0, exact: true};
  const mostSignificantBit = 63 - countLeadingZeroWords(first);
  if (mostSignificantBit < 24) return {value: Math.fround(first[0]), exact: true};
  const shift = mostSignificantBit - 23;
  const kept = shiftWordsRight(first, shift)[0];
  const remainder = subtractWords(first, shiftWordsLeft([kept, 0], shift));
  const half = shiftWordsLeft([1, 0], shift - 1);
  const remainderIsHalf = remainder[0] === half[0] && remainder[1] === half[1];
  const roundUp = isUnsignedLess(half, remainder) || (remainderIsHalf && (kept & 1) === 1);
  const mantissa = kept + (roundUp ? 1 : 0);
  return {
    value: Math.fround(mantissa * 2 ** shift),
    exact: remainder[0] === 0 && remainder[1] === 0
  };
}

/**
 * Emulates the atomic sum: the low and high words are accumulated modulo 2^32 with first carry counter
 * each, then recombined exactly (the signed sum subtracts 2^64 per negative value).
 */
export function sumWordsWithCarry(values: readonly WordPair[], signed: boolean): bigint {
  let lowSum = 0;
  let lowCarries = 0;
  let highSum = 0;
  let highCarries = 0;
  let negativeCount = 0;
  for (const pair of values) {
    const lowOld = lowSum;
    lowSum = (lowSum + pair[0]) >>> 0;
    if (lowSum < lowOld) lowCarries++;
    const highOld = highSum;
    highSum = (highSum + pair[1]) >>> 0;
    if (highSum < highOld) highCarries++;
    if (signed && (pair[1] & SIGN_BIT) !== 0) negativeCount++;
  }
  return combineWordSumAccumulators(lowSum, lowCarries, highSum, highCarries, negativeCount);
}

export function combineWordSumAccumulators(
  lowSum: number,
  lowCarries: number,
  highSum: number,
  highCarries: number,
  negativeCount: number
): bigint {
  const unsignedSum =
    BigInt(lowSum >>> 0) +
    ((BigInt(lowCarries >>> 0) + BigInt(highSum >>> 0)) << 32n) +
    (BigInt(highCarries >>> 0) << 64n);
  return unsignedSum - (BigInt(negativeCount >>> 0) << 64n);
}

// ---------- Narrowing classification (mirrors WGSL) ----------

export type WordNarrowingClassification = {
  kind: 'exact' | 'inexact' | 'overflow' | 'nonFinite';
  /** floor(log2(absolute error)); only meaningful for kind inexact. */
  errorExponent: number;
  /** Top 32 significant bits of the absolute error, normalised so bit 31 is set. */
  errorTopWord: number;
};

const EXACT_CLASSIFICATION: WordNarrowingClassification = {
  kind: 'exact',
  errorExponent: 0,
  errorTopWord: 0
};

function roundMagnitudeToFloat32Mantissa(
  magnitude: readonly number[],
  drop: number
): {kept: number; roundUp: boolean; inexact: boolean; errorUnits: WordPair} {
  if (drop >= 54) {
    return {
      kept: 0,
      roundUp: false,
      inexact: magnitude[0] !== 0 || magnitude[1] !== 0,
      errorUnits: [magnitude[0], magnitude[1]]
    };
  }
  const kept = shiftWordsRight(magnitude, drop);
  const remainder = subtractWords(magnitude, shiftWordsLeft(kept, drop));
  const half = shiftWordsLeft([1, 0], drop - 1);
  const remainderIsHalf = remainder[0] === half[0] && remainder[1] === half[1];
  const roundUp = isUnsignedLess(half, remainder) || (remainderIsHalf && (kept[0] & 1) === 1);
  return {
    kept: kept[0],
    roundUp,
    inexact: remainder[0] !== 0 || remainder[1] !== 0,
    errorUnits: roundUp ? subtractWords(shiftWordsLeft([1, 0], drop), remainder) : remainder
  };
}

function describeError(
  errorUnits: readonly number[],
  unitExponent: number
): WordNarrowingClassification {
  const bitLength = 64 - countLeadingZeroWords(errorUnits);
  const errorTopWord =
    bitLength >= 32
      ? shiftWordsRight(errorUnits, bitLength - 32)[0]
      : shiftWordsLeft(errorUnits, 32 - bitLength)[0];
  return {kind: 'inexact', errorExponent: bitLength - 1 + unitExponent, errorTopWord};
}

/** Classifies one 64-bit word pair against the Float32 narrowing semantic of diagnoseFloat32Narrowing. */
export function classifyWordsNarrowing(
  format: WideNumericGpuFormat,
  pair: readonly number[]
): WordNarrowingClassification {
  if (format !== 'float64') {
    const negative = format === 'int64' && (pair[1] & SIGN_BIT) !== 0;
    const magnitude = negative ? negateWords(pair) : [pair[0], pair[1]];
    if (magnitude[0] === 0 && magnitude[1] === 0) return EXACT_CLASSIFICATION;
    const mostSignificantBit = 63 - countLeadingZeroWords(magnitude);
    if (mostSignificantBit < 24) return EXACT_CLASSIFICATION;
    const rounded = roundMagnitudeToFloat32Mantissa(magnitude, mostSignificantBit - 23);
    return rounded.inexact ? describeError(rounded.errorUnits, 0) : EXACT_CLASSIFICATION;
  }
  const biasedExponent = (pair[1] >>> 20) & 0x7ff;
  if (biasedExponent === 0x7ff) return {...EXACT_CLASSIFICATION, kind: 'nonFinite'};
  const significand: WordPair = [
    pair[0] >>> 0,
    ((pair[1] & 0xfffff) | (biasedExponent !== 0 ? 0x100000 : 0)) >>> 0
  ];
  if (significand[0] === 0 && significand[1] === 0) return EXACT_CLASSIFICATION;
  const exponent = biasedExponent === 0 ? -1022 : biasedExponent - 1023;
  if (exponent >= 128) return {...EXACT_CLASSIFICATION, kind: 'overflow'};
  const drop = 29 + (exponent < -126 ? -126 - exponent : 0);
  const rounded = roundMagnitudeToFloat32Mantissa(significand, drop);
  if (exponent === 127 && rounded.roundUp && rounded.kept === 0xffffff) {
    return {...EXACT_CLASSIFICATION, kind: 'overflow'};
  }
  return rounded.inexact ? describeError(rounded.errorUnits, exponent - 52) : EXACT_CLASSIFICATION;
}

/** Converts the GPU error key (exponent and top word) to an approximate absolute error (never above the true value). */
export function decodeNarrowingError(errorExponent: number, errorTopWord: number): number {
  const exponent = errorExponent - 31;
  const first = Math.max(exponent, -1000);
  return errorTopWord * 2 ** first * 2 ** (exponent - first);
}

// ---------- Independent BigInt oracle ----------

/** Exact nearest-even Float32 of first non-negative integer, independent of the word algorithm. */
export function roundBigIntToFloat32(magnitude: bigint): number {
  const approximate = Math.fround(Number(magnitude));
  const candidates = [approximate];
  const view = new DataView(new ArrayBuffer(4));
  if (Number.isFinite(approximate) && approximate > 0) {
    view.setFloat32(0, approximate);
    const bits = view.getUint32(0);
    for (const neighbor of [bits - 1, bits + 1]) {
      view.setUint32(0, neighbor);
      candidates.push(view.getFloat32(0));
    }
  }
  let best = candidates[0];
  let bestDistance: bigint | undefined;
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate) || !Number.isInteger(candidate)) continue;
    const distance = BigInt(candidate) - magnitude;
    const absolute = distance < 0n ? -distance : distance;
    if (
      bestDistance === undefined ||
      absolute < bestDistance ||
      (absolute === bestDistance && isEvenFloat32(candidate) && !isEvenFloat32(best))
    ) {
      best = candidate;
      bestDistance = absolute;
    }
  }
  return best;
}

function isEvenFloat32(value: number): boolean {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  return (view.getUint32(0) & 1) === 0;
}

export type WideInteger64Summary = {
  count: number;
  nullCount: number;
  minimum: bigint | null;
  maximum: bigint | null;
  sum: bigint;
};

/** Exact reference for the resident exact-aggregate kernel. */
export function summarizeInteger64Reference(
  values: readonly (bigint | null)[],
  signed: boolean
): WideInteger64Summary {
  const summary: WideInteger64Summary = {
    count: 0,
    nullCount: 0,
    minimum: null,
    maximum: null,
    sum: 0n
  };
  for (const raw of values) {
    if (raw === null) {
      summary.nullCount++;
      continue;
    }
    const value = signed ? BigInt.asIntN(64, raw) : BigInt.asUintN(64, raw);
    summary.count++;
    summary.sum += value;
    if (summary.minimum === null || value < summary.minimum) summary.minimum = value;
    if (summary.maximum === null || value > summary.maximum) summary.maximum = value;
  }
  return summary;
}

/** Independent narrowing reference: Float64 through Math.fround, 64-bit integers through exact BigInt rounding. */
export function summarizeNarrowingReference(
  format: WideNumericGpuFormat,
  values: readonly (number | bigint | null)[]
): NumericNarrowingReport {
  if (format === 'float64') return diagnoseFloat32Narrowing(format as WideNumericFormat, values);
  let nullCount = 0;
  let inexactCount = 0;
  let maximumAbsoluteError = 0n;
  for (const raw of values) {
    if (raw === null || raw === undefined) {
      nullCount++;
      continue;
    }
    const value = BigInt(raw);
    const magnitude = value < 0n ? -value : value;
    const rounded = roundBigIntToFloat32(magnitude);
    const difference = BigInt(rounded) - magnitude;
    if (difference !== 0n) {
      inexactCount++;
      const absoluteError = difference < 0n ? -difference : difference;
      if (absoluteError > maximumAbsoluteError) maximumAbsoluteError = absoluteError;
    }
  }
  return {
    format: format as WideNumericFormat,
    rowCount: values.length,
    nullCount,
    inexactCount,
    overflowCount: 0,
    nonFiniteCount: 0,
    maximumAbsoluteError,
    verdict: inexactCount ? 'lossy' : 'exact'
  };
}

export {TWO_POW_32 as WORD_RADIX};
