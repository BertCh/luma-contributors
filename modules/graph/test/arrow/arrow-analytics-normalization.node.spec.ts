// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  prepareArrowTableForGPUAnalytics,
  makeGPUTimeVectorFromPreparedArrowColumn,
  makeGPUAnalyticsTableFromArrowTable,
  makeArrowTableFromGPUAnalyticsTable
} from '@luma-contributors/graph/arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {describe, expect, test, vi} from 'vitest';

function makeTable(columns: Record<string, arrow.Vector>[]): arrow.Table {
  const first = columns[0];
  const schema = new arrow.Schema(
    Object.entries(first).map(
      ([name, vector]) => new arrow.Field(name, vector.type, true, new Map([['source', name]]))
    ),
    new Map([['dataset', 'fixture']])
  );
  return new arrow.Table(
    schema,
    columns.map(
      values =>
        new arrow.RecordBatch(
          schema,
          arrow.makeData({
            type: new arrow.Struct(schema.fields),
            length: Object.values(values)[0].length,
            children: Object.values(values).map(vector => vector.data[0])
          })
        )
    )
  );
}

describe('explicit Arrow analytics normalization', () => {
  test('widens Boolean and narrow integers, preserves nulls, sliced rows, empty batches and source identity', async () => {
    const source = makeTable([
      {
        flag: arrow.vectorFromArray([true, false, null, true], new arrow.Bool()).slice(1, 4),
        signed: arrow.vectorFromArray([2, -128, null, 127], new arrow.Int8()).slice(1, 4),
        unsigned: arrow.vectorFromArray([3, 65535, 0, null], new arrow.Uint16()).slice(1, 4)
      },
      {
        flag: arrow.vectorFromArray([], new arrow.Bool()),
        signed: arrow.vectorFromArray([], new arrow.Int8()),
        unsigned: arrow.vectorFromArray([], new arrow.Uint16())
      },
      {
        flag: arrow.vectorFromArray([true], new arrow.Bool()),
        signed: arrow.vectorFromArray([-1], new arrow.Int8()),
        unsigned: arrow.vectorFromArray([42], new arrow.Uint16())
      }
    ]);
    const prepared = prepareArrowTableForGPUAnalytics(source);
    expect(prepared.table.batches.map(batch => batch.numRows)).toEqual([3, 0, 1]);
    expect(prepared.sourceBatches).toEqual([
      {sourceBatchIndex: 0, sourceRowIndexOffset: 0, sourceRowCount: 3},
      {sourceBatchIndex: 1, sourceRowIndexOffset: 3, sourceRowCount: 0},
      {sourceBatchIndex: 2, sourceRowIndexOffset: 3, sourceRowCount: 1}
    ]);
    expect(Array.from(prepared.table.getChild('flag')!)).toEqual([0, null, 1, 1]);
    expect(Array.from(prepared.table.getChild('signed')!)).toEqual([-128, null, 127, -1]);
    expect(Array.from(prepared.table.getChild('unsigned')!)).toEqual([65535, 0, null, 42]);
    expect(prepared.table.schema.metadata.get('dataset')).toBe('fixture');
    expect(prepared.table.schema.fields[0].metadata.get('source')).toBe('flag');
    const uploaded = makeGPUAnalyticsTableFromArrowTable(new NullDevice({}), prepared.table);
    try {
      const readback = await makeArrowTableFromGPUAnalyticsTable(uploaded);
      expect(Array.from(readback.getChild('flag')!)).toEqual([0, null, 1, 1]);
    } finally {
      uploaded.table.destroy();
      Object.values(uploaded.validity).forEach(vector => vector?.destroy());
    }
  });

  test('rejects Float64 by default and enforces exact or bounded rounding with nonfinite and signed-zero semantics', () => {
    const exact = makeTable([
      {value: arrow.vectorFromArray([1, -0, null, NaN, Infinity, -Infinity], new arrow.Float64())}
    ]);
    expect(() => prepareArrowTableForGPUAnalytics(exact)).toThrow(/explicit precision/);
    const prepared = prepareArrowTableForGPUAnalytics(exact, {float64: {mode: 'exact'}});
    const values = Array.from(prepared.table.getChild('value')!);
    expect(values).toEqual([1, -0, null, NaN, Infinity, -Infinity]);
    expect(Object.is(values[1], -0)).toBe(true);
    const inexact = makeTable([
      {value: arrow.vectorFromArray([0.1, 16777217], new arrow.Float64())}
    ]);
    expect(() => prepareArrowTableForGPUAnalytics(inexact, {float64: {mode: 'exact'}})).toThrow(
      /precision policy/
    );
    expect(() =>
      prepareArrowTableForGPUAnalytics(inexact, {
        float64: {mode: 'round', maximumAbsoluteError: 0, maximumRelativeError: 0}
      })
    ).toThrow(/precision policy/);
    const rounded = prepareArrowTableForGPUAnalytics(inexact, {
      float64: {mode: 'round', maximumAbsoluteError: 1, maximumRelativeError: 0}
    });
    expect(Array.from(rounded.table.getChild('value')!)).toEqual([Math.fround(0.1), 16777216]);
    expect(rounded.conversions[0].maximumAbsoluteError).toBe(1);
    const overflow = makeTable([{value: arrow.vectorFromArray([1e300], new arrow.Float64())}]);
    expect(() =>
      prepareArrowTableForGPUAnalytics(overflow, {
        float64: {mode: 'round', maximumAbsoluteError: 1e308, maximumRelativeError: 1}
      })
    ).toThrow(/precision policy/);
    expect(() =>
      prepareArrowTableForGPUAnalytics(exact, {
        float64: {mode: 'round', maximumAbsoluteError: Infinity, maximumRelativeError: 0}
      })
    ).toThrow(/budgets/);
  });

  test('reconciles different batch dictionaries and streams without mutating prior dictionaries', async () => {
    const dictionaryType = new arrow.Dictionary(new arrow.Utf8(), new arrow.Int8(), 42, false);
    const source = makeTable([
      {category: arrow.vectorFromArray(['beta', null, 'alpha'], dictionaryType)},
      {category: arrow.vectorFromArray(['alpha', 'gamma', 'beta'], dictionaryType)}
    ]);
    const prior = {category: {values: Object.freeze(['existing', 'beta']), ordered: false}};
    const prepared = prepareArrowTableForGPUAnalytics(source, {dictionaries: prior});
    expect(prior.category.values).toEqual(['existing', 'beta']);
    expect(prepared.dictionaries.category.values).toEqual(['existing', 'beta', 'alpha', 'gamma']);
    expect(
      prepared.table.batches.map(batch =>
        Array.from(batch.getChild('category')!.data[0].values).slice(0, 3)
      )
    ).toEqual([
      [1, 0, 2],
      [2, 3, 1]
    ]);
    expect(Array.from(prepared.table.getChild('category')!)).toEqual([
      'beta',
      null,
      'alpha',
      'alpha',
      'gamma',
      'beta'
    ]);
    const uploaded = makeGPUAnalyticsTableFromArrowTable(new NullDevice({}), prepared.table);
    try {
      expect(uploaded.dictionaries.category).toEqual(prepared.dictionaries.category);
    } finally {
      uploaded.table.destroy();
      Object.values(uploaded.validity).forEach(vector => vector?.destroy());
    }
    const next = prepareArrowTableForGPUAnalytics(
      makeTable([{category: arrow.vectorFromArray(['delta', 'alpha'], new arrow.Utf8())}]),
      {dictionaries: prepared.dictionaries}
    );
    expect(next.dictionaries.category.values).toEqual([
      'existing',
      'beta',
      'alpha',
      'gamma',
      'delta'
    ]);
  });

  test('refuses changing ordered categories and unsupported types', () => {
    const type = new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32(), 12, true);
    const source = makeTable([
      {category: arrow.vectorFromArray(['a', 'b'], type)},
      {category: arrow.vectorFromArray(['b', 'a'], type)}
    ]);
    expect(() => prepareArrowTableForGPUAnalytics(source)).toThrow(/Ordered dictionary/);
    expect(() =>
      prepareArrowTableForGPUAnalytics(
        makeTable([{id: arrow.vectorFromArray([9007199254740993n], new arrow.Int64())}])
      )
    ).toThrow(/Unsupported|requires an explicit/);
  });

  test('preserves signed nanoseconds, timezone, sliced validity and time-only source batches without number conversion', async () => {
    const type = new arrow.TimestampNanosecond('America/New_York');
    const vector = new arrow.Vector([
      arrow.makeData({
        type,
        data: new BigInt64Array([
          123n,
          -9223372036854775808n,
          9007199254740993n,
          9223372036854775807n
        ]),
        length: 4,
        nullBitmap: new Uint8Array([0b1011]),
        nullCount: 1
      })
    ]).slice(1, 4);
    const source = makeTable([{time: vector}]);
    const prepared = prepareArrowTableForGPUAnalytics(source);
    expect(prepared.sourceRowCount).toBe(3);
    expect(prepared.sourceBatches[0].sourceRowCount).toBe(3);
    expect(prepared.table.numCols).toBe(0);
    const time = prepared.timeColumns.time;
    expect(time.unit).toBe('nanosecond');
    expect(time.timeZone).toBe('America/New_York');
    expect(Array.from(time.chunks[0])).toEqual([0, 0x80000000, 0, 0, 0xffffffff, 0x7fffffff]);
    expect(Array.from(time.validity[0])).toEqual([1, 0, 1]);
    const device = new NullDevice({});
    const submit = vi.spyOn(device, 'submit');
    const uploaded = makeGPUTimeVectorFromPreparedArrowColumn(device, time);
    expect(uploaded.timestamps.format).toBe('uint32x2');
    expect(uploaded.timestamps.length).toBe(3);
    expect(submit).not.toHaveBeenCalled();
    const bytes = await uploaded.timestamps.data[0].buffer.readAsync();
    expect(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)).toEqual(
      time.chunks[0]
    );
    uploaded.destroy();
    uploaded.destroy();
    expect(() =>
      prepareArrowTableForGPUAnalytics(source, {timeColumns: {time: 'millisecond'}})
    ).toThrow(/reinterpreted/);
  });

  test('converts dates exactly and accepts signed Int64 time only with a unit declaration', () => {
    const days = new arrow.Vector([
      arrow.makeData({type: new arrow.DateDay(), data: new Int32Array([-1, 0, 1]), length: 3})
    ]);
    const prepared = prepareArrowTableForGPUAnalytics(makeTable([{date: days}]));
    expect(Array.from(prepared.timeColumns.date.chunks[0])).toEqual([
      4208567296, 0xffffffff, 0, 0, 86400000, 0
    ]);
    const source = makeTable([
      {time: arrow.vectorFromArray([9007199254740993n], new arrow.Int64())}
    ]);
    const words = prepareArrowTableForGPUAnalytics(source, {timeColumns: {time: 'microsecond'}})
      .timeColumns.time;
    expect(Array.from(words.chunks[0])).toEqual([1, 2097152]);
  });
});
