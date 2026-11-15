// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {makeWideNumericWordVectors} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-summary';
import {createWideNumericUploadDevice} from '../../analysis/test/utils/wide-numeric-upload-test-utils';

function readWideValues(words: Uint32Array, format: 'int64' | 'uint64' | 'float64') {
  const bytes = new DataView(words.buffer, words.byteOffset, words.byteLength);
  return Array.from({length: words.length / 2}, (_, index) =>
    format === 'float64'
      ? bytes.getFloat64(index * 8, true)
      : format === 'int64'
        ? bytes.getBigInt64(index * 8, true)
        : bytes.getBigUint64(index * 8, true)
  );
}

test('64-bit uploads keep sliced values and original nullable bitmap offsets across chunks', () => {
  for (const type of [new arrow.Int64(), new arrow.Uint64(), new arrow.Float64()]) {
    const format = arrow.DataType.isFloat(type) ? 'float64' : type.isSigned ? 'int64' : 'uint64';
    const original = arrow.vectorFromArray(
      format === 'float64' ? [99, 11.5, null, 33.5, 44.5] : [99n, 11n, null, (1n << 60n) + 3n, 44n],
      type
    );
    const column = new arrow.Vector([...original.slice(1, 4).data, ...original.slice(4, 5).data]);
    expect(column.data.map(chunk => chunk.offset)).toEqual([1, 4]);
    const {device, buffers} = createWideNumericUploadDevice();
    const vectors = makeWideNumericWordVectors(device, column);
    try {
      expect(vectors.words.data.map(chunk => chunk.length)).toEqual([3, 1]);
      expect(readWideValues(buffers[0].data, format)).toEqual(
        format === 'float64' ? [11.5, 0, 33.5] : [11n, 0n, (1n << 60n) + 3n]
      );
      expect(readWideValues(buffers[2].data, format)).toEqual(
        format === 'float64' ? [44.5] : [44n]
      );
      expect([...buffers[1].data]).toEqual([1, 0, 1]);
      expect([...buffers[3].data]).toEqual([1]);
      expect(vectors.validity?.data.map(chunk => chunk.length)).toEqual([3, 1]);
    } finally {
      vectors.words.destroy();
      vectors.validity?.destroy();
      vectors.unusedValidity?.destroy();
    }
    expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1, 1, 1]);
  }
});

test('64-bit empty and non-null uploads return the unused validity owner', () => {
  const original = arrow.vectorFromArray([11n, 22n, 33n], new arrow.Int64());
  const column = new arrow.Vector([...original.slice(0, 0).data, ...original.slice(1, 3).data]);
  const {device, buffers} = createWideNumericUploadDevice();
  const vectors = makeWideNumericWordVectors(device, column);
  expect(vectors.words.data.map(chunk => chunk.length)).toEqual([0, 2]);
  expect([...buffers[0].data]).toEqual([0, 0]);
  expect(readWideValues(buffers[2].data, 'int64')).toEqual([22n, 33n]);
  expect(vectors.validity).toBeUndefined();
  expect(vectors.unusedValidity).toBeDefined();
  vectors.words.destroy();
  vectors.unusedValidity?.destroy();
  expect(buffers.map(buffer => buffer.destroyCount)).toEqual([1, 1, 1, 1]);
});

test('64-bit uploads free all previous buffers at every word or validity allocation failure', () => {
  for (const values of [
    [11n, null, 33n],
    [11n, 22n, 33n]
  ]) {
    const original = arrow.vectorFromArray(values, new arrow.Int64());
    const column = new arrow.Vector([...original.slice(0, 2).data, ...original.slice(2).data]);
    for (let failAllocation = 1; failAllocation <= 4; failAllocation++) {
      const {device, buffers, failure} = createWideNumericUploadDevice(failAllocation);
      expect(() => makeWideNumericWordVectors(device, column)).toThrow(failure);
      expect(buffers).toHaveLength(failAllocation - 1);
      expect(buffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
    }
  }
});
