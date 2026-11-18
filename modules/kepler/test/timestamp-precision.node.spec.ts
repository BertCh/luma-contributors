// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {formatTimestampTicks} from '@luma-contributors/analysis/dataframe-analysis';
import {
  readOriginalColumnValues,
  readOriginalTimestampTicks
} from '../../../examples/deck/kepler-compute/src/original-column-values';
import {formatCSVValue} from '../../../examples/deck/kepler-compute/src/source-export';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
import {
  getPreparedTimeDomainMilliseconds,
  prepareKeplerTimeRangeMilliseconds
} from '../../../examples/deck/kepler-compute/src/time-runtime';

test.each([
  new arrow.TimestampMicrosecond('UTC'),
  new arrow.TimestampNanosecond('UTC')
])('dictionary timestamp reads match plain ticks across slices and distinct chunk dictionaries: %s', type => {
  const makeTimestamp = (ticks: bigint[]) =>
    arrow.makeVector(
      arrow.makeData({
        type,
        data: BigInt64Array.from(ticks),
        length: ticks.length
      })
    );
  const dictionaryType = new arrow.Dictionary(type, new arrow.Int32(), 41);
  const makeDictionary = (ticks: bigint[], indices: number[]) =>
    arrow.makeVector(
      arrow.makeData({
        type: dictionaryType,
        data: Int32Array.from(indices),
        length: indices.length,
        dictionary: makeTimestamp([123n, ...ticks]).slice(1)
      })
    );
  const first = makeDictionary([1n, -1n, 0n], [0, 1, 2]);
  const second = makeDictionary([0n, 1n, -1n], [1, 2]);
  const combined = new arrow.Vector([...first.slice(1).data, ...second.data]);
  const fraction = type.unit === arrow.TimeUnit.MICROSECOND ? '999999' : '999999999';
  const positiveFraction = type.unit === arrow.TimeUnit.MICROSECOND ? '000001' : '000000001';
  const exact = [
    `1969-12-31T23:59:59.${fraction}Z`,
    '1970-01-01T00:00:00Z',
    `1970-01-01T00:00:00.${positiveFraction}Z`,
    `1969-12-31T23:59:59.${fraction}Z`
  ];
  const plain = makeTimestamp([-1n, 0n, 1n, -1n]);
  expect(readOriginalColumnValues(plain, [0, 1, 2, 3])).toEqual([-1, 0, 0, -1]);
  expect(readOriginalColumnValues(plain, [0, 1, 2, 3], true)).toEqual(exact);
  expect(readOriginalColumnValues(combined, [0, 1, 2, 3])).toEqual([-1, 0, 0, -1]);
  expect(readOriginalColumnValues(combined, [0, 1, 2, 3], true)).toEqual(exact);
  expect([0, 1, 2, 3].map(row => readOriginalTimestampTicks(combined, row))).toEqual([
    -1n,
    0n,
    1n,
    -1n
  ]);
  expect(readOriginalColumnValues(combined.slice(1, 3), [0, 1], true)).toEqual(exact.slice(1, 3));
  expect([0, 1, 2, 3].map(row => formatCSVValue(combined, row, dictionaryType))).toEqual(exact);
  const nullableDictionary = arrow.makeVector(
    arrow.makeData({
      type,
      data: new BigInt64Array([-1n, 0n]),
      length: 2,
      nullCount: 1,
      nullBitmap: new Uint8Array([1])
    })
  );
  const nullable = arrow.makeVector(
    arrow.makeData({
      type: dictionaryType,
      data: new Int32Array([0, 1, 0]),
      length: 3,
      nullCount: 1,
      nullBitmap: new Uint8Array([3]),
      dictionary: nullableDictionary
    })
  );
  expect(readOriginalColumnValues(nullable, [0, 1, 2])).toEqual([-1, null, null]);
  expect(readOriginalColumnValues(nullable, [0, 1, 2], true)).toEqual([exact[0], null, null]);
  expect([0, 1, 2].map(row => formatCSVValue(nullable, row, dictionaryType))).toEqual([
    exact[0],
    '',
    ''
  ]);
});

test.each([
  new arrow.TimestampMicrosecond('UTC'),
  new arrow.TimestampNanosecond('UTC')
])('timestamp ticks survive slices, chunks, copy, CSV and native time admission: %s', type => {
  const scale = type.unit === arrow.TimeUnit.MICROSECOND ? 1000n : 1_000_000n;
  const makeVector = (values: bigint[]) =>
    arrow.makeVector(
      arrow.makeData({type, data: BigInt64Array.from(values), length: values.length})
    );
  const epoch = 1_700_000_000_000n * scale;
  const first = makeVector([123n, epoch + 1n]).slice(1);
  const second = makeVector([-1n, epoch + scale - 1n]);
  const vector = new arrow.Vector([...first.data, ...second.data]);
  expect([0, 1, 2].map(row => readOriginalTimestampTicks(vector, row))).toEqual([
    epoch + 1n,
    -1n,
    epoch + scale - 1n
  ]);
  const expected = [epoch + 1n, -1n, epoch + scale - 1n].map(ticks =>
    formatTimestampTicks(ticks, type.unit)
  );
  expect(readOriginalColumnValues(vector, [0, 1, 2], true)).toEqual(expected);
  expect(readOriginalColumnValues(vector, [0, 1, 2])).toEqual([
    1_700_000_000_000, -1, 1_700_000_000_000
  ]);
  expect([0, 1, 2].map(row => formatCSVValue(vector, row, type))).toEqual(expected);
  const metadata = prepareArrowTableForGPUAnalytics(new arrow.Table({time: vector})).timeColumns[
    'time'
  ];
  expect(getPreparedTimeDomainMilliseconds(metadata)).toEqual([-1, 1_700_000_000_001]);
  expect(
    prepareKeplerTimeRangeMilliseconds(1_700_000_000_000, 1_700_000_000_001, metadata.unit)
  ).toEqual({start: epoch, end: epoch + scale, unit: metadata.unit});
});

test('exact timestamp text retains nanoseconds before epoch and outside Date range', () => {
  expect(formatTimestampTicks(-1n, arrow.TimeUnit.NANOSECOND)).toBe(
    '1969-12-31T23:59:59.999999999Z'
  );
  expect(formatTimestampTicks(1_700_000_000_000_000_001n, arrow.TimeUnit.NANOSECOND)).toBe(
    '2023-11-14T22:13:20.000000001Z'
  );
  expect(formatTimestampTicks((1n << 63n) - 1n, arrow.TimeUnit.SECOND)).toBe(
    '9223372036854775807 s since Unix epoch'
  );
});

test('host date presentation refuses unsafe pre-epoch milliseconds while exact copying keeps ticks', () => {
  const ticks = [-9_072_000_000_000_000_001n, -9_072_000_000_000_000_000n];
  const vector = arrow.makeVector(
    arrow.makeData({
      type: new arrow.TimestampMicrosecond('UTC'),
      data: BigInt64Array.from(ticks),
      length: 2
    })
  );
  for (const row of [0, 1])
    expect(() => readOriginalColumnValues(vector, [row])).toThrow(/Date range/);
  expect(readOriginalColumnValues(vector, [0, 1], true)).toEqual(
    ticks.map(value => `${value} us since Unix epoch`)
  );
});

test.each([
  new arrow.TimestampMicrosecond('UTC'),
  new arrow.TimestampNanosecond('UTC')
])('calendar timestamp presentation floors before Number conversion at calendar boundaries: %s', type => {
  const scale = type.unit === arrow.TimeUnit.MICROSECOND ? 1000n : 1_000_000n;
  const boundary = BigInt(Date.UTC(2026, 0, 1));
  const ticks = [
    boundary * scale - 1n,
    -boundary * scale + 1n,
    -1n,
    boundary * scale,
    boundary * scale + 1n
  ];
  const vector = arrow.makeVector(
    arrow.makeData({type, data: BigInt64Array.from(ticks), length: ticks.length})
  );
  const values = readOriginalColumnValues(vector, [0, 1, 2, 3, 4]);
  expect(values).toEqual([
    Number(boundary - 1n),
    Number(-boundary),
    -1,
    Number(boundary),
    Number(boundary)
  ]);
  expect(new Date(values[0] as number).toISOString()).toBe('2025-12-31T23:59:59.999Z');
});

test.each([
  new arrow.TimestampNanosecond('UTC'),
  new arrow.TimestampMicrosecond('UTC')
])('enclosing native domains remain selectable at both signed Int64 extrema: %s', type => {
  const minimum = -(1n << 63n);
  const maximum = (1n << 63n) - 1n;
  const scale = type.unit === arrow.TimeUnit.MICROSECOND ? 1000n : 1_000_000n;
  for (const ticks of [[minimum], [maximum], [minimum, maximum]]) {
    const vector = arrow.makeVector(
      arrow.makeData({type, data: BigInt64Array.from(ticks), length: ticks.length})
    );
    const metadata = prepareArrowTableForGPUAnalytics(new arrow.Table({time: vector})).timeColumns[
      'time'
    ];
    // Microsecond extrema exceed the native Date admission policy; nanosecond extrema do not.
    if (type.unit === arrow.TimeUnit.MICROSECOND) {
      expect(() => getPreparedTimeDomainMilliseconds(metadata)).toThrow(/Date range/);
      continue;
    }
    const domain = getPreparedTimeDomainMilliseconds(metadata)!;
    const window = prepareKeplerTimeRangeMilliseconds(domain[0], domain[1], metadata.unit);
    const expectedStart = BigInt(domain[0]) * scale;
    const expectedEnd = BigInt(domain[1]) * scale;
    expect(window.start).toBe(expectedStart < minimum ? minimum : expectedStart);
    expect(window.end).toBe(expectedEnd > maximum ? maximum : expectedEnd);
    for (const tick of ticks) expect(tick >= window.start && tick <= window.end).toBe(true);
  }
});
