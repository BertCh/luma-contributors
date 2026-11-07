// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {
  getOriginalFieldType,
  readOriginalColumnValues
} from '../../../examples/deck/kepler-compute/src/original-column-values';

// Load the native statistics module directly; the package index pulls in a second deck.gl copy.
const exampleRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const native = createRequire(exampleRequire.resolve('@kepler.gl/utils'))('./column-statistics');

test('Arrow types map to native Kepler field types', () => {
  expect(getOriginalFieldType(new arrow.Int64())).toBe('integer');
  expect(getOriginalFieldType(new arrow.Float64())).toBe('real');
  expect(getOriginalFieldType(new arrow.Utf8())).toBe('string');
  expect(getOriginalFieldType(new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32()))).toBe(
    'string'
  );
  expect(getOriginalFieldType(new arrow.TimestampMillisecond())).toBe('timestamp');
  expect(getOriginalFieldType(new arrow.Binary())).toBe('other');
});

test('population rows feed native column statistics without a full scan', () => {
  const vector = arrow.vectorFromArray([4, 1, null, 3, 100], new arrow.Float64());
  const values = readOriginalColumnValues(vector, [0, 1, 2, 3]);
  expect(values).toEqual([4, 1, null, 3]);
  const stats = native.computeColumnStats(values, 'real');
  expect(stats).toMatchObject({type: 'numeric', percentNulls: 0.25, mean: 8 / 3});
  expect(stats.quantiles[2]).toEqual({label: '50%', value: 3});
});

test('exact reads keep 64-bit integers lossless for copying', () => {
  const vector = arrow.vectorFromArray([2n ** 60n + 1n, null], new arrow.Int64());
  expect(readOriginalColumnValues(vector, [0, 1], true)).toEqual([2n ** 60n + 1n, null]);
  const text = native.computeColumnStats(
    readOriginalColumnValues(
      arrow.vectorFromArray(['a', 'b', 'a', null], new arrow.Utf8()),
      [0, 1, 2, 3]
    ),
    'string'
  );
  expect(text).toMatchObject({type: 'categorical', uniqueValues: 2, percentNulls: 0.25});
});
