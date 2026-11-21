// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Host budget decisions use the exact binary64 cutoff, not a rounded H_m.
// Fast ordinary comparisons stay in binary64. Near a boundary, small families
// use rational H_m; large families use a certified Euler-Maclaurin interval.
const PRECISION_BITS = 256n;
const SCALE = 1n << PRECISION_BITS;
const EXACT_HARMONIC_ROWS = 128;
const LOGARITHM_TERMS = 96;
const EULER_GAMMA_DIGITS =
  '57721566490153286060651209008240243104215933593992359880576723488486772677766467093694';

function getGreatestCommonDivisor(first: bigint, second: bigint): bigint {
  while (second !== 0n) [first, second] = [second, first % second];
  return first;
}

function getExactHarmonicRatio(rows: number): [bigint, bigint] {
  let numerator = 0n,
    denominator = 1n;
  for (let index = 1; index <= rows; index++) {
    const divisor = BigInt(index);
    const common = getGreatestCommonDivisor(denominator, divisor);
    numerator = numerator * (divisor / common) + denominator / common;
    denominator *= divisor / common;
  }
  return [numerator, denominator];
}

/** Positive log(value) bounds, in SCALE units; value is an integer >= 1. */
function getLogarithmBounds(value: bigint): [bigint, bigint] {
  const exponent = value.toString(2).length - 1;
  const power = 1n << BigInt(exponent);
  // log(x)=2*sum(z^(2j+1)/(2j+1)), z=(x-1)/(x+1).
  // Range reduction ensures 0<=z<=1/3, including the log(2) series.
  const series = (first: bigint, second: bigint): [bigint, bigint] => {
    let numerator = first,
      denominator = second,
      lower = 0n;
    for (let index = 0; index < LOGARITHM_TERMS; index++) {
      lower += (2n * SCALE * numerator) / (denominator * BigInt(2 * index + 1));
      numerator *= first * first;
      denominator *= second * second;
    }
    // Each division loses <1 SCALE unit. Tail <= 2*z^(2N+1)/((2N+1)*(1-z²)).
    const tail =
      (2n * SCALE * numerator * second * second) /
        (denominator * BigInt(2 * LOGARITHM_TERMS + 1) * (second * second - first * first)) +
      1n;
    return [lower, lower + BigInt(LOGARITHM_TERMS) + tail];
  };
  const logarithmTwo = series(1n, 3n);
  const reduced = series(value - power, value + power);
  return [
    BigInt(exponent) * logarithmTwo[0] + reduced[0],
    BigInt(exponent) * logarithmTwo[1] + reduced[1]
  ];
}

function getHarmonicBounds(rows: number): [bigint, bigint] {
  const count = BigInt(rows);
  const gammaDenominator = 10n ** BigInt(EULER_GAMMA_DIGITS.length);
  const gammaNumerator = BigInt(EULER_GAMMA_DIGITS);
  const logarithm = getLogarithmBounds(count);
  let lower = logarithm[0] + (SCALE * gammaNumerator) / gammaDenominator;
  let upper = logarithm[1] + (SCALE * (gammaNumerator + 1n)) / gammaDenominator + 1n;
  const addFraction = (numerator: bigint, denominator: bigint) => {
    const quotient = (SCALE * numerator) / denominator;
    // BigInt truncates toward zero; the extra unit encloses either sign.
    lower += quotient - 1n;
    upper += quotient + 1n;
  };
  addFraction(1n, 2n * count);
  // H_n = log(n)+gamma+1/(2n)-sum B_2k/(2k*n^(2k)).
  // The remainder has the sign of the next term and magnitude at most it.
  const coefficients = [
    [-1n, 12n],
    [1n, 120n],
    [-1n, 252n],
    [1n, 240n],
    [-1n, 132n],
    [691n, 32760n],
    [-1n, 12n],
    [3617n, 8160n]
  ] as const;
  for (const [index, [numerator, denominator]] of coefficients.entries())
    addFraction(numerator, denominator * count ** BigInt(2 * (index + 1)));
  lower -= (SCALE * 43867n) / (14364n * count ** 18n) + 1n;
  return [lower, upper];
}

/** H_rows * numerator / denominator <= the exact host cutoff. @internal */
export function isHarmonicRatioAtMost(
  rows: number,
  numerator: bigint,
  denominator: bigint,
  cutoff: number,
  approximateHarmonic: number
): boolean {
  const approximate = (approximateHarmonic * Number(numerator)) / Number(denominator);
  if (Math.abs(approximate - cutoff) > 16 * Number.EPSILON * Math.max(approximate, cutoff))
    return approximate < cutoff;
  const bits = new DataView(new ArrayBuffer(8));
  bits.setFloat64(0, cutoff);
  const word = bits.getBigUint64(0);
  const exponent = Number((word >> 52n) & 2047n);
  const significand = (word & ((1n << 52n) - 1n)) + (exponent === 0 ? 0n : 1n << 52n);
  const cutoffNumerator = significand * denominator;
  const cutoffDenominator = (1n << BigInt(1075 - Math.max(exponent, 1))) * numerator;
  if (rows > EXACT_HARMONIC_ROWS) {
    const [lower, upper] = getHarmonicBounds(rows);
    if (upper * cutoffDenominator <= cutoffNumerator * SCALE) return true;
    if (lower * cutoffDenominator > cutoffNumerator * SCALE) return false;
  }
  // Exact fallback if even the certified high-precision interval straddles the
  // cutoff. Never turn an unresolved boundary into advice to reject.
  const harmonic = getExactHarmonicRatio(rows);
  return harmonic[0] * cutoffDenominator <= cutoffNumerator * harmonic[1];
}
