// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {diagnoseFloat32Narrowing} from '../../../examples/deck/kepler-compute/src/analysis-numeric-diagnostics';
import {
  addWords,
  classifyWordsNarrowing,
  convertWordsToFloat32,
  countLeadingZeroWords,
  decodeNarrowingError,
  divideSignedWordsBySmall,
  divideWordsBySmall,
  isSignedLess,
  isUnsignedLess,
  joinWordsToBigInt,
  multiplyWords128,
  multiplyWordsLow,
  negateWords,
  roundBigIntToFloat32,
  shiftWordsLeft,
  shiftWordsRight,
  splitBigIntToWords,
  splitDoubleToWords,
  subtractWords,
  sumWordsWithCarry,
  summarizeInteger64Reference,
  summarizeNarrowingReference,
  type WordPair
} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-reference';

import {MASK64, makeWideNumericTestValues} from './wide-numeric-test-values';

describe('wide numeric word emulation versus BigInt', () => {
  const values = makeWideNumericTestValues();
  it('split/join round trips and add/subtract/negate/compare agree', () => {
    for (const a of values)
      for (const b of values.slice(0, 40)) {
        const pa = splitBigIntToWords(a);
        const pb = splitBigIntToWords(b);
        expect(joinWordsToBigInt(addWords(pa, pb), false)).toBe((a + b) & MASK64);
        expect(joinWordsToBigInt(subtractWords(pa, pb), false)).toBe((a - b) & MASK64);
        expect(isUnsignedLess(pa, pb)).toBe(a < b);
        expect(isSignedLess(pa, pb)).toBe(BigInt.asIntN(64, a) < BigInt.asIntN(64, b));
      }
    for (const a of values)
      expect(joinWordsToBigInt(negateWords(splitBigIntToWords(a)), false)).toBe(-a & MASK64);
  });
  it('multiplies to the low 64 and the full 128 bits', () => {
    for (const a of values)
      for (const b of values.slice(0, 60)) {
        const pa = splitBigIntToWords(a);
        const pb = splitBigIntToWords(b);
        const product = a * b;
        expect(joinWordsToBigInt(multiplyWordsLow(pa, pb), false)).toBe(product & MASK64);
        const wide = multiplyWords128(pa, pb);
        const rebuilt = wide.reduce(
          (total, word, index) => total | (BigInt(word) << BigInt(32 * index)),
          0n
        );
        expect(rebuilt).toBe(product);
      }
  });
  it('divides by small divisors, signed and unsigned, with remainder', () => {
    const divisors = [1, 2, 3, 7, 10, 1000, 65535, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff];
    for (const a of values)
      for (const divisor of divisors) {
        const pa = splitBigIntToWords(a);
        const unsigned = divideWordsBySmall(pa, divisor);
        expect(joinWordsToBigInt(unsigned.quotient, false)).toBe(a / BigInt(divisor));
        expect(BigInt(unsigned.remainder)).toBe(a % BigInt(divisor));
        const signedValue = BigInt.asIntN(64, a);
        const signed = divideSignedWordsBySmall(pa, divisor);
        expect(joinWordsToBigInt(signed.quotient, true)).toBe(
          BigInt.asIntN(64, signedValue / BigInt(divisor))
        );
        const remainder = signedValue % BigInt(divisor);
        expect(BigInt(signed.remainderMagnitude) * (signed.remainderNegative ? -1n : 1n)).toBe(
          signedValue < 0n ? remainder : remainder
        );
      }
    expect(divideWordsBySmall([5, 0], 0).remainder).toBe(0xffffffff);
  });
  it('shifts and counts leading zeros', () => {
    for (const a of values) {
      const pa = splitBigIntToWords(a);
      expect(countLeadingZeroWords(pa)).toBe(a === 0n ? 64 : 64 - a.toString(2).length);
      for (const amount of [0, 1, 5, 31, 32, 33, 63]) {
        expect(joinWordsToBigInt(shiftWordsRight(pa, amount), false)).toBe(a >> BigInt(amount));
        expect(joinWordsToBigInt(shiftWordsLeft(pa, amount), false)).toBe(
          (a << BigInt(amount)) & MASK64
        );
      }
    }
  });
  it('converts to Float32 with nearest-even rounding and the exact flag', () => {
    for (const a of values) {
      const unsigned = convertWordsToFloat32(splitBigIntToWords(a), false);
      const expectedUnsigned = roundBigIntToFloat32(a);
      expect(unsigned.value).toBe(expectedUnsigned);
      expect(unsigned.exact).toBe(BigInt(expectedUnsigned) === a);
      const signedValue = BigInt.asIntN(64, a);
      const signed = convertWordsToFloat32(splitBigIntToWords(a), true);
      const magnitude = signedValue < 0n ? -signedValue : signedValue;
      const expectedSigned = roundBigIntToFloat32(magnitude) * (signedValue < 0n ? -1 : 1);
      expect(signed.value).toBe(expectedSigned);
      expect(signed.exact).toBe(BigInt(Math.abs(expectedSigned)) === magnitude);
    }
    // 2^24 + 1 is a tie that rounds to even; 2^24 + 3 rounds up.
    expect(convertWordsToFloat32([0x1000001, 0], false)).toEqual({value: 0x1000000, exact: false});
    expect(convertWordsToFloat32([0x1000003, 0], false)).toEqual({value: 0x1000004, exact: false});
  });
  it('sums with carry exactly beyond 64 bits, signed and unsigned', () => {
    const pairs: WordPair[] = values.map(splitBigIntToWords);
    for (const signed of [false, true]) {
      const expected = summarizeInteger64Reference(values, signed).sum;
      expect(sumWordsWithCarry(pairs, signed)).toBe(expected);
    }
    const maximum = Array.from({length: 100000}, () => splitBigIntToWords(MASK64));
    expect(sumWordsWithCarry(maximum, false)).toBe(MASK64 * 100000n);
    expect(sumWordsWithCarry(maximum, true)).toBe(-100000n);
  });
  it('summarizes min/max/null reference', () => {
    const summary = summarizeInteger64Reference([5n, null, -3n, 7n], true);
    expect(summary).toEqual({count: 3, nullCount: 1, minimum: -3n, maximum: 7n, sum: 9n});
    expect(summarizeInteger64Reference([null], false).minimum).toBeNull();
  });
});

describe('wide numeric narrowing classification', () => {
  it('matches the independent reference for Int64 and Uint64 per value', () => {
    for (const format of ['int64', 'uint64'] as const)
      for (const raw of makeWideNumericTestValues()) {
        const value = format === 'int64' ? BigInt.asIntN(64, raw) : BigInt.asUintN(64, raw);
        const reference = summarizeNarrowingReference(format, [value]);
        const classification = classifyWordsNarrowing(format, splitBigIntToWords(value));
        expect(classification.kind === 'inexact').toBe(reference.inexactCount === 1);
        expect(classification.kind === 'overflow' || classification.kind === 'nonFinite').toBe(
          false
        );
        if (classification.kind === 'inexact') {
          const decoded = decodeNarrowingError(
            classification.errorExponent,
            classification.errorTopWord
          );
          expect(decoded).toBeLessThanOrEqual(reference.maximumAbsoluteError);
          expect(decoded).toBeGreaterThan(reference.maximumAbsoluteError * (1 - 1e-8));
        }
      }
  });
  it('matches diagnoseFloat32Narrowing for doubles including subnormals, overflow and non-finite', () => {
    const doubles = [
      0,
      -0,
      1,
      -1,
      0.1,
      -0.1,
      0.5,
      1 / 3,
      16777217,
      16777216,
      1e10,
      1e38,
      3.4028234663852886e38,
      3.4028235677973366e38,
      3.4028235677973362e38,
      3.5e38,
      -3.5e38,
      1e300,
      -1e300,
      1e-40,
      1e-45,
      7e-46,
      1.401298464324817e-45,
      7.006492321624085e-46,
      7.006492321624087e-46,
      1.1754943508222875e-38,
      1.1754942807573643e-38,
      5e-324,
      -5e-324,
      2.2250738585072014e-308,
      Number.MAX_VALUE,
      Number.MIN_VALUE,
      Infinity,
      -Infinity,
      NaN,
      9007199254740993,
      123456.789,
      Math.PI,
      Math.fround(Math.PI),
      2 ** -126,
      2 ** -149,
      2 ** -150,
      2 ** -151
    ];
    let state = 12345;
    for (let index = 0; index < 500; index++) {
      state = (Math.imul(state, 1103515245) + 12345) >>> 0;
      doubles.push((state / 0xffffffff - 0.5) * 10 ** (((state >>> 7) % 80) - 40));
    }
    for (const value of doubles) {
      const reference = diagnoseFloat32Narrowing('float64', [value]);
      const classification = classifyWordsNarrowing('float64', splitDoubleToWords(value));
      const expectedKind = reference.nonFiniteCount
        ? 'nonFinite'
        : reference.overflowCount
          ? 'overflow'
          : reference.inexactCount
            ? 'inexact'
            : 'exact';
      expect(classification.kind, String(value)).toBe(expectedKind);
      if (classification.kind === 'inexact') {
        const decoded = decodeNarrowingError(
          classification.errorExponent,
          classification.errorTopWord
        );
        expect(decoded, String(value)).toBeLessThanOrEqual(
          reference.maximumAbsoluteError * (1 + 1e-12)
        );
        expect(decoded, String(value)).toBeGreaterThanOrEqual(
          reference.maximumAbsoluteError * (1 - 1e-8)
        );
      }
    }
  });
});
