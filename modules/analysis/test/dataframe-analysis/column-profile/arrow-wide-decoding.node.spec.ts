// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, it, vi} from 'vitest';
import {
  convertWideNumericValue,
  convertWideNumericVector
} from '../../../src/analysis-core/graph-support/arrow-analytics-wide-numeric';
import {joinWordsToBigInt} from '../../../src/dataframe-analysis/column-profile/wide-numeric-gpu-reference';
import {decodeGPUGroupSum} from '../../../src/dataframe-analysis/group-statistics/decode-gpu-group-sum';

it('ING-10 converts the original Decimal coefficient and scale with one binary64 rounding', () => {
  const type = new arrow.Decimal(309, 38, 128);
  const words = Uint32Array.of(0, 0, 0, 1);
  // Python: float(Decimal(2 ** 96).scaleb(-309)), with context precision 100.
  const expected = 7.922816251426434e-281;
  expect(convertWideNumericValue(type, words)).toBe(expected);
  const vector = new arrow.Vector([arrow.makeData({type, data: words, length: 1})]);
  expect(convertWideNumericVector(type, vector, value => value)).toEqual([expected]);
  const zeroType = new arrow.Decimal(-324, 38, 128);
  const zero = new Uint32Array(4);
  expect(convertWideNumericValue(zeroType, zero)).toBe(0);
  const zeroVector = new arrow.Vector([arrow.makeData({type: zeroType, data: zero, length: 1})]);
  expect(convertWideNumericVector(zeroType, zeroVector, value => value)).toEqual([0]);
  expect(
    convertWideNumericValue(
      new arrow.Decimal(2, 38, 128),
      Uint32Array.of(0xffffff9c, 0xffffffff, 0xffffffff, 0xffffffff)
    )
  ).toBe(-1);
  expect(convertWideNumericValue(new arrow.Decimal(2147483647, 38, 128), words)).toBe(0);
  expect(convertWideNumericValue(new arrow.Decimal(-2147483648, 38, 128), words)).toBe(Infinity);
});

it('EGR-06 checks both word lanes before coercion and accepts typed subarrays', () => {
  expect(() => joinWordsToBigInt([], true)).toThrow(/pair.*2.*0/);
  expect(() => joinWordsToBigInt([1], true)).toThrow(/pair.*2.*1/);
  expect(() => joinWordsToBigInt([undefined, 0] as never, true)).toThrow(/pair.*lane 0/);
  const backing = Uint32Array.of(999, 0xffff0000, 0xffffffff, 888);
  expect(joinWordsToBigInt(backing.subarray(1, 3), true)).toBe(-65536n);
  expect(joinWordsToBigInt(new Uint8Array(backing.buffer, 4, 8), true)).toBe(-65536n);
  expect(joinWordsToBigInt(backing.slice(1, 3).buffer, true)).toBe(-65536n);
});

it('EGR-12 decodes signed scaled sums, exact integers and offset byte readbacks', () => {
  const words = Uint32Array.of(99, 0, 0xffff0000, 0xffffffff, 1, 2097152);
  expect(decodeGPUGroupSum(words, 1, 65536)).toEqual({scaledInteger: -65536n, value: -1});
  expect(decodeGPUGroupSum(words, 2, 1).scaledInteger).toBe(9007199254740993n);
  const bytes = new Uint8Array(11);
  const view = new DataView(bytes.buffer);
  view.setUint32(1, 0xffff0000, true);
  view.setUint32(5, 0xffffffff, true);
  expect(decodeGPUGroupSum(bytes.subarray(1, 9), 0, 65536)).toEqual({
    scaledInteger: -65536n,
    value: -1
  });
  expect(decodeGPUGroupSum(words.slice(2, 4).buffer, 0, 65536).value).toBe(-1);
  expect(() => decodeGPUGroupSum(new Uint8Array(7), 0, 65536)).toThrow(/whole.*4-byte/);
  expect(() => decodeGPUGroupSum(Uint32Array.of(1), 0, 65536)).toThrow(/2.*1/);
  expect(() => decodeGPUGroupSum(words, 3, 65536)).toThrow(/8.*6/);
  expect(() => decodeGPUGroupSum(words, -1, 65536)).toThrow(/row.*-1/);
  expect(() => decodeGPUGroupSum(words, 0, 0)).toThrow(/sumScale/);
  expect(() => decodeGPUGroupSum(Uint32Array.of(0, 0, 1), 0, 65536)).toThrow(/2-word pairs.*3/);
});

it('B2 standalone Decimal numeric inputs retain their already converted values', () => {
  const type = new arrow.Decimal(5, 10, 128);
  expect(convertWideNumericValue(type, 0.125)).toBe(0.125);
  expect(convertWideNumericValue(type, -0)).toBe(-0);
  expect(convertWideNumericValue(type, NaN)).toBeNaN();
  // Coefficient representations still scale: 12500 * 10^-5 = 0.125.
  expect(convertWideNumericValue(type, Uint32Array.of(12500, 0, 0, 0))).toBe(0.125);
});

it('B3 scaled sums round the exact rational once, including a quotient that fits binary64', () => {
  // Integer division, without any floating input: (2^53 + 1) / 3 = 3002399751580331.
  expect(decodeGPUGroupSum(Uint32Array.of(1, 2097152), 0, 3)).toEqual({
    scaledInteger: 9007199254740993n,
    value: 3002399751580331
  });
  expect(decodeGPUGroupSum(Uint32Array.of(0xffffffff, 0xffdfffff), 0, 3)).toEqual({
    scaledInteger: -9007199254740993n,
    value: -3002399751580331
  });
  // Ties to even: 2^53+1 rounds down, 2^53+3 rounds up at scale 1.
  expect(decodeGPUGroupSum(Uint32Array.of(1, 2097152), 0, 1).value).toBe(9007199254740992);
  expect(decodeGPUGroupSum(Uint32Array.of(3, 2097152), 0, 1).value).toBe(9007199254740996);
  // Python float(Fraction(integer) / Fraction.from_float(scale)), using exact float32 scales.
  expect(decodeGPUGroupSum(Uint32Array.of(0xffffffff, 0x7fffffff), 0, 2 ** -149).value).toBe(
    6.582018229284824e63
  );
  expect(
    decodeGPUGroupSum(Uint32Array.of(0, 0x80000000), 0, Math.fround(3.4028234663852886e38)).value
  ).toBe(-2.710505592772484e-20);
  expect(decodeGPUGroupSum(Uint32Array.of(1, 2097152), 0, Math.fround(0.1)).value).toBe(
    9.007199120523267e16
  );
});

it('B6 rejects unsupported typed views and identifies the pair or sums port', () => {
  for (const input of [
    Int8Array.of(-1, -1),
    Uint8ClampedArray.of(1, 0),
    Uint16Array.of(1, 0),
    Float32Array.of(1, 0)
  ]) {
    expect(() => joinWordsToBigInt(input, true)).toThrow(/pair.*unsupported.*Array/);
    expect(() => decodeGPUGroupSum(input as never, 0, 1)).toThrow(/sums.*unsupported.*Array/);
  }
  expect(() => joinWordsToBigInt(new Uint8Array(4), true)).toThrow(/pair.*2.*1/);
  expect(() => decodeGPUGroupSum(new Uint8Array(4), 0, 1)).toThrow(/sums.*2.*1/);
  expect(joinWordsToBigInt(Int32Array.of(-65536, -1), true)).toBe(-65536n);
});

it('S1 decoding one row reads two words rather than copying the sums column', () => {
  const words = Uint32Array.of(99, 0, 0xffff0000, 0xffffffff, 7, 0, 9, 0);
  const readWord = vi.spyOn(DataView.prototype, 'getUint32');
  try {
    expect(decodeGPUGroupSum(new Uint8Array(words.buffer), 1, 65536)).toEqual({
      scaledInteger: -65536n,
      value: -1
    });
    expect(readWord).toHaveBeenCalledTimes(2);
  } finally {
    readWord.mockRestore();
  }
  const copyWords = vi.spyOn(Uint32Array, 'from');
  try {
    expect(decodeGPUGroupSum(words, 1, 65536).value).toBe(-1);
    expect(copyWords).not.toHaveBeenCalled();
  } finally {
    copyWords.mockRestore();
  }
});
