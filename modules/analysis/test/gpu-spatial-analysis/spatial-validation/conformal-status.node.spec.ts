// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {runInNewContext} from 'node:vm';
import {decodeConformalStatus} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-conformal-prediction-intervals';

it('B6 reads cross-realm buffers and rejects unsupported element types', () => {
  expect(decodeConformalStatus(runInNewContext('Uint32Array.of(35).buffer'))).toEqual(
    decodeConformalStatus(35)
  );
  expect(decodeConformalStatus(runInNewContext('Uint32Array.of(35)'))).toEqual(
    decodeConformalStatus(35)
  );
  for (const value of [
    Float32Array.of(3),
    Int32Array.of(3),
    new DataView(new ArrayBuffer(4)),
    {byteLength: 4}
  ]) {
    expect(() => decodeConformalStatus(value as Uint32Array)).toThrow(/uint32|Uint32Array/);
  }
});

it('UCO-16 independently decodes every conformal condition and combinations', () => {
  const fields = [
    'lowerUnbounded',
    'upperUnbounded',
    'invalidQuery',
    'noCalibration',
    'neighborsTruncated',
    'inputIncomplete'
  ];
  for (let bit = 0; bit < fields.length; bit++) {
    expect(Object.values(decodeConformalStatus(1 << bit))).toEqual(
      fields.map((_, index) => index === bit)
    );
  }
  expect(Object.values(decodeConformalStatus(63))).toEqual([true, true, true, true, true, true]);
  expect(decodeConformalStatus(17).neighborsTruncated).toBe(true);
  expect(decodeConformalStatus(17).lowerUnbounded).toBe(true);
});

it('UCO-16 decodes complete uint32 readbacks with byte offsets and rejects missing words', () => {
  const words = Uint32Array.of(999, 48, 999);
  const expected = decodeConformalStatus(48);
  expect(decodeConformalStatus(words.subarray(1, 2))).toEqual(expected);
  expect(decodeConformalStatus(new Uint8Array(words.buffer, 4, 4))).toEqual(expected);
  expect(decodeConformalStatus(Uint32Array.of(48).buffer)).toEqual(expected);
  const unaligned = new Uint8Array(5);
  unaligned.set(new Uint8Array(Uint32Array.of(48).buffer), 1);
  expect(decodeConformalStatus(unaligned.subarray(1))).toEqual(expected);
  for (const input of [
    new Uint32Array(),
    new Uint8Array(3),
    new Uint8Array(7),
    new ArrayBuffer(0),
    NaN,
    -1,
    1.5,
    2 ** 32
  ]) {
    expect(() => decodeConformalStatus(input)).toThrow();
  }
});
