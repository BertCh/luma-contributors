// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {makeDecimal128WordVectors} from '../../../src/dataframe-analysis/column-profile/wide-numeric-gpu-decimal-aggregate';
import {createWideNumericUploadDevice} from '../../utils/wide-numeric-upload-test-utils';

function makeDecimalColumn(
  values: readonly (bigint | null)[],
  type = new arrow.Decimal(2, 38, 128)
) {
  const words = Uint32Array.from(
    values.flatMap(value => {
      const bits = BigInt.asUintN(type.bitWidth, value ?? 0n);
      return Array.from({length: type.bitWidth / 32}, (_, wordIndex) =>
        Number((bits >> BigInt(wordIndex * 32)) & 0xffffffffn)
      );
    })
  );
  const nullBitmap = new Uint8Array(Math.ceil(values.length / 8));
  values.forEach((value, index) => {
    if (value !== null) nullBitmap[index >> 3] |= 1 << (index & 7);
  });
  return new arrow.Vector([
    arrow.makeData({
      type,
      length: values.length,
      data: words,
      nullBitmap,
      nullCount: values.filter(value => value === null).length
    })
  ]);
}

function readSignedWords(words: Uint32Array): bigint[] {
  return Array.from({length: words.length / 4}, (_, row) => {
    let bits = 0n;
    for (let word = 3; word >= 0; word--) bits = (bits << 32n) | BigInt(words[row * 4 + word]);
    return BigInt.asIntN(128, bits);
  });
}

test('Decimal128 uploads sliced, chunked and nullable Arrow words without applying the row offset twice', () => {
  const first = makeDecimalColumn([99n, 11n, null, (1n << 96n) + 7n, -55n]).slice(1, 4);
  const last = makeDecimalColumn([88n, -22n, 33n]).slice(1, 3);
  const column = new arrow.Vector([...first.data, ...last.data]);
  expect(column.data.map(chunk => chunk.offset)).toEqual([1, 1]);
  const {device, buffers} = createWideNumericUploadDevice();
  const vectors = makeDecimal128WordVectors(device, column);
  try {
    expect(vectors.words.data.map(chunk => chunk.length)).toEqual([3, 2]);
    expect(readSignedWords(buffers[0].data)).toEqual([11n, 0n, (1n << 96n) + 7n]);
    expect([...buffers[1].data]).toEqual([1, 0, 1]);
    expect(readSignedWords(buffers[2].data)).toEqual([-22n, 33n]);
    expect([...buffers[3].data]).toEqual([1, 1]);
    expect(vectors.validity?.data.map(chunk => chunk.length)).toEqual([3, 2]);
    expect(vectors).not.toHaveProperty('unusedValidity');
    expect(buffers.map(buffer => buffer.destroyCount)).toEqual([0, 0, 0, 0]);
  } finally {
    vectors.destroy();
  }
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1, 1, 1]);
});

test('Decimal128 uploads empty and non-null chunks with normal returned ownership', () => {
  const column = new arrow.Vector([
    ...makeDecimalColumn([]).data,
    ...makeDecimalColumn([11n, 22n, 33n]).slice(1, 3).data
  ]);
  const {device, buffers} = createWideNumericUploadDevice();
  const vectors = makeDecimal128WordVectors(device, column);
  expect(vectors.words.data.map(chunk => chunk.length)).toEqual([0, 2]);
  expect([...buffers[0].data]).toEqual([0, 0, 0, 0]);
  expect(buffers).toHaveLength(2);
  expect(readSignedWords(buffers[1].data)).toEqual([22n, 33n]);
  expect(vectors.validity).toBeUndefined();
  expect(vectors).not.toHaveProperty('unusedValidity');
  expectTypeOf(vectors).not.toHaveProperty('unusedValidity');
  vectors.destroy();
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1]);
});

test('Decimal128 frees earlier buffers when any word or validity allocation fails', () => {
  for (const values of [
    [11n, null, 33n],
    [11n, 22n, 33n]
  ]) {
    const original = makeDecimalColumn(values);
    const column = new arrow.Vector([...original.slice(0, 2).data, ...original.slice(2).data]);
    const allocationCount = values.includes(null) ? 4 : 2;
    for (let failAllocation = 1; failAllocation <= allocationCount; failAllocation++) {
      const {device, buffers, failure} = createWideNumericUploadDevice(failAllocation);
      expect(() => makeDecimal128WordVectors(device, column)).toThrow(failure);
      expect(buffers).toHaveLength(failAllocation - 1);
      expect(buffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
    }
  }
});

test('Decimal128 uploads validity for earlier non-null and empty chunks when a later chunk has nulls', () => {
  const column = new arrow.Vector([
    ...makeDecimalColumn([11n]).data,
    ...makeDecimalColumn([]).data,
    ...makeDecimalColumn([null]).data
  ]);
  const {device, buffers} = createWideNumericUploadDevice();
  const vectors = makeDecimal128WordVectors(device, column);
  try {
    expect(vectors.words.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
    expect(vectors.validity?.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
    expect(buffers).toHaveLength(6);
    expect([...buffers[1].data]).toEqual([1]);
    expect([...buffers[5].data]).toEqual([0]);
  } finally {
    vectors.destroy();
  }
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1, 1, 1, 1, 1]);
});

test('Decimal128 upload rejects Decimal256 before allocating buffers', () => {
  const type = new arrow.Decimal(0, 76, 256);
  const data = new Uint32Array(16);
  data[0] = 1;
  data[8] = 2;
  const column = new arrow.Vector([arrow.makeData({type, length: 2, data})]);
  expect(column.stride).toBe(8);
  const {device, buffers} = createWideNumericUploadDevice();
  expect(() => makeDecimal128WordVectors(device, column)).toThrow(/Decimal.*256/);
  expect(buffers).toHaveLength(0);
});

test('Decimal128 upload rejects non-Decimal runtime input before allocating buffers', () => {
  const column = arrow.vectorFromArray([1, 2], new arrow.Int32());
  const {device, buffers} = createWideNumericUploadDevice();
  expect(() =>
    makeDecimal128WordVectors(device, column as unknown as arrow.Vector<arrow.Decimal>)
  ).toThrow(/Decimal128.*Int32/);
  expect(buffers).toHaveLength(0);
});

test.each([
  ['width', new arrow.Decimal(2, 76, 256)],
  ['scale', new arrow.Decimal(3, 38, 128)]
] as const)('Decimal128 upload rejects a later chunk with a different %s before allocation', (_, type) => {
  // Arrow concat accepts both types without a cast; column.type describes only the first chunk.
  const column = makeDecimalColumn([3n]).concat(makeDecimalColumn([1n, 2n], type));
  expect(column.type.bitWidth).toBe(128);
  expect(column.type.scale).toBe(2);
  const {device, buffers} = createWideNumericUploadDevice();
  expect(() => makeDecimal128WordVectors(device, column)).toThrow(/chunk 1/);
  expect(buffers).toHaveLength(0);
});
