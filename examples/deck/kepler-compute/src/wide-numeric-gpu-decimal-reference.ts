// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact Decimal128 reference arithmetic. A value is a signed 128-bit unscaled integer held as four
 * little-endian uint32 words (two's complement) plus a column scale: value = unscaled / 10^scale.
 * BigInt is the oracle for the GPU kernel in wide-numeric-gpu-decimal-aggregate.ts.
 */

export type Decimal128Words = [number, number, number, number];
export type Decimal128CompareOperator = '<' | '<=' | '=' | '>=' | '>' | '!=';

export const DECIMAL128_MAXIMUM = (1n << 127n) - 1n;
export const DECIMAL128_MINIMUM = -(1n << 127n);

export function splitDecimal128ToWords(value: bigint): Decimal128Words {
  const bits = BigInt.asUintN(128, value);
  return [0, 1, 2, 3].map(index =>
    Number((bits >> BigInt(index * 32)) & 0xffffffffn)
  ) as Decimal128Words;
}

export function joinWordsToDecimal128(words: readonly number[]): bigint {
  let bits = 0n;
  for (let index = 3; index >= 0; index--) bits = (bits << 32n) | BigInt(words[index] >>> 0);
  return BigInt.asIntN(128, bits);
}

/** Signed sum from per-word sums modulo 2^32, per-word carry counts and the negative-value count. */
export function combineDecimal128SumAccumulators(
  wordSums: readonly number[],
  wordCarries: readonly number[],
  negativeCount: number
): bigint {
  let unsignedSum = 0n;
  for (let index = 0; index < 4; index++)
    unsignedSum +=
      (BigInt(wordSums[index] >>> 0) + (BigInt(wordCarries[index] >>> 0) << 32n)) <<
      BigInt(index * 32);
  return unsignedSum - (BigInt(negativeCount >>> 0) << 128n);
}

export type Decimal128Summary = {
  count: number;
  nullCount: number;
  minimum: bigint | null;
  maximum: bigint | null;
  /** Exact, not wrapped to 128 bits. */
  sum: bigint;
  scale: number;
};

export function summarizeDecimal128Reference(
  values: readonly (bigint | null)[],
  scale: number,
  /** SQL WHERE semantics: with a predicate, NULL rows fail it and are not counted as nulls. */
  keep?: (value: bigint) => boolean
): Decimal128Summary {
  const summary: Decimal128Summary = {
    count: 0,
    nullCount: 0,
    minimum: null,
    maximum: null,
    sum: 0n,
    scale
  };
  for (const raw of values) {
    if (raw === null) {
      if (!keep) summary.nullCount++;
      continue;
    }
    if (keep && !keep(raw)) continue;
    const value = BigInt.asIntN(128, raw);
    summary.count++;
    summary.sum += value;
    if (summary.minimum === null || value < summary.minimum) summary.minimum = value;
    if (summary.maximum === null || value > summary.maximum) summary.maximum = value;
  }
  return summary;
}

/** Decimal text of an unscaled value, for example (-12345n, 2) is "-123.45". */
export function formatDecimal128(unscaled: bigint, scale: number): string {
  const negative = unscaled < 0n;
  const digits = (negative ? -unscaled : unscaled).toString().padStart(scale + 1, '0');
  const point = digits.length - scale;
  return `${negative ? '-' : ''}${scale ? `${digits.slice(0, point)}.${digits.slice(point)}` : digits}`;
}

function divideFloor(numerator: bigint, divisor: bigint): bigint {
  const quotient = numerator / divisor;
  return numerator % divisor !== 0n && numerator < 0n !== divisor < 0n ? quotient - 1n : quotient;
}

function divideCeil(numerator: bigint, divisor: bigint): bigint {
  return -divideFloor(-numerator, divisor);
}

export type NormalizedDecimal128Comparison =
  | {kind: 'constant'; result: boolean}
  | {kind: 'compare'; operator: Decimal128CompareOperator; threshold: bigint};

/**
 * Rewrites `column op constant` (column scale `columnScale`, constant scale `constantScale`) as an
 * exact integer comparison on the stored unscaled values, or as a constant (for non-null rows)
 * when no representable stored value can differ. No rounding of the constant ever happens.
 */
export function normalizeDecimal128Comparison(
  operator: Decimal128CompareOperator,
  constant: bigint,
  constantScale: number,
  columnScale: number
): NormalizedDecimal128Comparison {
  let comparison: NormalizedDecimal128Comparison;
  if (constantScale <= columnScale) {
    comparison = {
      kind: 'compare',
      operator,
      threshold: constant * 10n ** BigInt(columnScale - constantScale)
    };
  } else {
    // value * d op constant, with d = 10^(constantScale - columnScale) > 1.
    const divisor = 10n ** BigInt(constantScale - columnScale);
    const exact = constant % divisor === 0n;
    if (operator === '=' || operator === '!=')
      comparison = exact
        ? {kind: 'compare', operator, threshold: constant / divisor}
        : {kind: 'constant', result: operator === '!='};
    else if (operator === '<')
      comparison = {kind: 'compare', operator, threshold: divideCeil(constant, divisor)};
    else if (operator === '>=')
      comparison = {kind: 'compare', operator, threshold: divideCeil(constant, divisor)};
    else comparison = {kind: 'compare', operator, threshold: divideFloor(constant, divisor)};
  }
  if (comparison.kind === 'constant') return comparison;
  const {threshold} = comparison;
  if (threshold > DECIMAL128_MAXIMUM || threshold < DECIMAL128_MINIMUM) {
    const above = threshold > DECIMAL128_MAXIMUM;
    // Every stored value is below an above-range threshold and above a below-range one.
    const result =
      operator === '!='
        ? true
        : operator === '='
          ? false
          : operator === '<' || operator === '<='
            ? above
            : !above;
    return {kind: 'constant', result};
  }
  return comparison;
}

export function compareDecimal128Reference(
  values: readonly (bigint | null)[],
  scale: number,
  operator: Decimal128CompareOperator,
  constant: bigint,
  constantScale: number
): number {
  const comparison = normalizeDecimal128Comparison(operator, constant, constantScale, scale);
  let count = 0;
  for (const value of values) {
    if (value === null) continue;
    if (comparison.kind === 'constant') count += comparison.result ? 1 : 0;
    else {
      const {threshold} = comparison;
      const passes =
        comparison.operator === '<'
          ? value < threshold
          : comparison.operator === '<='
            ? value <= threshold
            : comparison.operator === '>'
              ? value > threshold
              : comparison.operator === '>='
                ? value >= threshold
                : comparison.operator === '='
                  ? value === threshold
                  : value !== threshold;
      count += passes ? 1 : 0;
    }
  }
  return count;
}
