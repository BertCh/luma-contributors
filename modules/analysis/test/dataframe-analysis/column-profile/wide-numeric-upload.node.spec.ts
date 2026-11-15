// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {makeDecimal128WordVectors} from '../../../src/dataframe-analysis/column-profile/index';
import {createWideNumericUploadDevice} from '../../utils/wide-numeric-upload-test-utils';

function makeDecimalColumn(values: readonly (bigint | null)[]) {
  const type = new arrow.Decimal(2, 38, 128);
  const words = Uint32Array.from(
    values.flatMap(value => {
      const bits = BigInt.asUintN(128, value ?? 0n);
      return [0n, 32n, 64n, 96n].map(shift => Number((bits >> shift) & 0xffffffffn));
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
    expect(vectors.unusedValidity).toBeUndefined();
    expect(buffers.map(buffer => buffer.destroyCount)).toEqual([0, 0, 0, 0]);
  } finally {
    vectors.words.destroy();
    vectors.validity?.destroy();
    vectors.unusedValidity?.destroy();
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
  expect(readSignedWords(buffers[2].data)).toEqual([22n, 33n]);
  expect(vectors.validity).toBeUndefined();
  expect(vectors.unusedValidity).toBeDefined();
  vectors.words.destroy();
  vectors.unusedValidity?.destroy();
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1, 1, 1]);
});

test('Decimal128 frees earlier buffers when any word or validity allocation fails', () => {
  for (const values of [
    [11n, null, 33n],
    [11n, 22n, 33n]
  ]) {
    const original = makeDecimalColumn(values);
    const column = new arrow.Vector([...original.slice(0, 2).data, ...original.slice(2).data]);
    for (let failAllocation = 1; failAllocation <= 4; failAllocation++) {
      const {device, buffers, failure} = createWideNumericUploadDevice(failAllocation);
      expect(() => makeDecimal128WordVectors(device, column)).toThrow(failure);
      expect(buffers).toHaveLength(failAllocation - 1);
      expect(buffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
    }
  }
});
