// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {
  isOriginalSortType,
  sortOriginalRows
} from '../../../examples/deck/kepler-compute/src/original-table-sort';

const identity = (length: number) => Array.from({length}, (_, index) => index);

test('strings sort exactly with missing values last in both directions and stable ties', () => {
  const vector = arrow.vectorFromArray(['b', 'a', null, 'b', 'c', 'a'], new arrow.Utf8());
  expect(Array.from(sortOriginalRows(vector, identity(6), 'ascending'))).toEqual([
    1, 5, 0, 3, 4, 2
  ]);
  expect(Array.from(sortOriginalRows(vector, identity(6), 'descending'))).toEqual([
    4, 0, 3, 1, 5, 2
  ]);
});

test('dictionary columns sort by value, not code', () => {
  const vector = arrow.vectorFromArray(
    ['z', 'a', 'm', 'a'],
    new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32())
  );
  expect(Array.from(sortOriginalRows(vector, identity(4), 'ascending'))).toEqual([1, 3, 2, 0]);
});

test('Int64 and timestamp keys keep exact 64-bit order beyond double precision', () => {
  const big = 2n ** 60n;
  const ints = arrow.vectorFromArray([big + 1n, big, big + 2n, null], new arrow.Int64());
  expect(Array.from(sortOriginalRows(ints, identity(4), 'ascending'))).toEqual([1, 0, 2, 3]);
  const type = new arrow.TimestampNanosecond();
  const times = new arrow.Vector([
    arrow.makeData({type, length: 3, data: new BigInt64Array([big + 1n, big, big + 2n])})
  ]);
  expect(Array.from(sortOriginalRows(times, identity(3), 'descending'))).toEqual([2, 0, 1]);
});

test('NaN counts as missing and only the requested rows are ordered', () => {
  const vector = arrow.vectorFromArray([3, NaN, 1, 2], new arrow.Float64());
  expect(Array.from(sortOriginalRows(vector, [3, 1, 2, 0], 'ascending'))).toEqual([2, 3, 0, 1]);
});

test('only ordered types are sortable', () => {
  expect(isOriginalSortType(new arrow.Utf8())).toBe(true);
  expect(isOriginalSortType(new arrow.Int64())).toBe(true);
  expect(isOriginalSortType(new arrow.Binary())).toBe(false);
  expect(isOriginalSortType(new arrow.List(new arrow.Field('item', new arrow.Int32())))).toBe(
    false
  );
});
