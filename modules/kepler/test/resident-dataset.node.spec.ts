// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  DEFAULT_KEPLER_RESIDENT_PAGE_ROWS,
  KeplerResidentRows,
  estimateKeplerResidentRowCacheByteLength,
  makeResidentPreparedTable,
  type KeplerResidentColumnSchema
} from '../src/session/resident-dataset';

type ReadCall = {offset: number; length: number};

type FakeVector = {
  vector: GPUVector;
  calls: ReadCall[];
};

/** Byte padding placed before each chunk to exercise nonzero `byteOffset`. */
const CHUNK_BYTE_OFFSET = 16;

function makeFakeVector(
  chunks: readonly (readonly number[])[],
  encode: 'float32' | 'sint32' | 'uint32'
): FakeVector {
  const calls: ReadCall[] = [];
  const data = chunks.map(values => {
    const bytes = new Uint8Array(CHUNK_BYTE_OFFSET + values.length * 4 + 8);
    const view = new DataView(bytes.buffer);
    values.forEach((value, index) => {
      const position = CHUNK_BYTE_OFFSET + index * 4;
      if (encode === 'float32') view.setFloat32(position, value, true);
      else if (encode === 'sint32') view.setInt32(position, value, true);
      else view.setUint32(position, value, true);
    });
    return {
      length: values.length,
      byteOffset: CHUNK_BYTE_OFFSET,
      buffer: {
        readAsync: vi.fn(async (offset: number, length: number) => {
          calls.push({offset, length});
          return bytes.slice(offset, offset + length);
        })
      }
    };
  });
  return {vector: {data} as unknown as GPUVector, calls};
}

function makeFixture(pageRows = 4) {
  const device = {isLost: false} as Device;
  const schema: KeplerResidentColumnSchema[] = [
    {name: 'value', format: 'float32', metadata: {unit: 'meters'}},
    {name: 'delta', format: 'sint32'},
    {name: 'count', format: 'uint32'},
    {name: 'category', format: 'uint32', dictionary: ['red', 'green', 'blue']}
  ];
  const value = makeFakeVector(
    [
      [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
      [10.5, 11.5, 12.5, 13.5, 14.5]
    ],
    'float32'
  );
  const delta = makeFakeVector(
    [
      [-1, -2, -3, -4, -5, -6],
      [-7, -8, -9, -10, -11]
    ],
    'sint32'
  );
  const count = makeFakeVector(
    [
      [100, 101, 102, 103, 104, 105],
      [106, 107, 108, 109, 110]
    ],
    'uint32'
  );
  // Codes 0..2 are valid labels; 7 is out of range and must read as null.
  const category = makeFakeVector(
    [
      [0, 1, 2, 0, 1, 7],
      [2, 1, 0, 9, 2]
    ],
    'uint32'
  );
  // Only `count` carries validity; row 1 and row 8 are null.
  const countValidity = makeFakeVector(
    [
      [1, 0, 1, 1, 1, 1],
      [1, 1, 0, 1, 1]
    ],
    'uint32'
  );
  const rows = new KeplerResidentRows(device, {
    schema,
    columns: {
      value: value.vector,
      delta: delta.vector,
      count: count.vector,
      category: category.vector
    },
    validity: {count: countValidity.vector},
    schemaMetadata: {origin: 'test'},
    pageRows
  });
  return {
    device,
    schema,
    rows,
    vectors: {value, delta, count, category, countValidity},
    allCalls: () =>
      [value, delta, count, category, countValidity].reduce(
        (total, fake) => total + fake.calls.length,
        0
      )
  };
}

describe('KeplerResidentRows table shape', () => {
  test('exposes real row count, batch layout, field names and metadata', () => {
    const {rows} = makeFixture();
    expect(rows.rowCount).toBe(11);
    expect(rows.pageRows).toBe(4);
    expect(rows.table.numRows).toBe(11);
    expect(rows.table.batches.map(batch => batch.numRows)).toEqual([6, 5]);
    expect(rows.table.schema.fields.map(field => field.name)).toEqual([
      'value',
      'delta',
      'count',
      'category'
    ]);
    expect(rows.table.schema.metadata.get('origin')).toBe('test');
    expect(rows.table.schema.fields[0].metadata.get('unit')).toBe('meters');
    expect(arrow.DataType.isDictionary(rows.table.schema.fields[3].type)).toBe(true);
    expect(arrow.DataType.isFloat(rows.table.schema.fields[0].type)).toBe(true);
    expect(arrow.DataType.isInt(rows.table.schema.fields[1].type)).toBe(true);
  });

  test('uses the default page size and rejects invalid page sizes', () => {
    const device = {isLost: false} as Device;
    const column = makeFakeVector([[1, 2]], 'uint32');
    const schema: KeplerResidentColumnSchema[] = [{name: 'count', format: 'uint32'}];
    const rows = new KeplerResidentRows(device, {
      schema,
      columns: {count: column.vector},
      validity: {}
    });
    expect(rows.pageRows).toBe(DEFAULT_KEPLER_RESIDENT_PAGE_ROWS);
    for (const pageRows of [0, -1, 1.5]) {
      expect(
        () =>
          new KeplerResidentRows(device, {
            schema,
            columns: {count: column.vector},
            validity: {},
            pageRows
          })
      ).toThrow();
    }
  });

  test('every cell is null before any rows are loaded', () => {
    const {rows, allCalls} = makeFixture();
    for (const name of ['value', 'delta', 'count', 'category']) {
      const vector = rows.table.getChild(name)!;
      for (let row = 0; row < rows.rowCount; row++) expect(vector.get(row)).toBeNull();
    }
    expect(rows.loadedRowCount).toBe(0);
    expect(rows.readbackByteLength).toBe(0);
    expect(rows.isRowLoaded(0)).toBe(false);
    expect(allCalls()).toBe(0);
  });

  test('an empty schema produces an empty table', () => {
    const rows = new KeplerResidentRows({isLost: false} as Device, {
      schema: [],
      columns: {},
      validity: {}
    });
    expect(rows.rowCount).toBe(0);
    expect(rows.table.numRows).toBe(0);
  });
});

describe('KeplerResidentRows.ensureRows', () => {
  test('loads only the containing page with exact byte ranges', async () => {
    const {rows, vectors, allCalls} = makeFixture();
    await rows.ensureRows([5]);
    // Row 5 is chunk 0, page 1 (rows 4..5 of the 6-row chunk), so two rows are read.
    expect(rows.isRowLoaded(4)).toBe(true);
    expect(rows.isRowLoaded(5)).toBe(true);
    expect(rows.isRowLoaded(3)).toBe(false);
    expect(rows.isRowLoaded(6)).toBe(false);
    expect(rows.loadedRowCount).toBe(2);
    // Four value columns plus one validity vector, each reading two 4-byte words.
    expect(allCalls()).toBe(5);
    expect(vectors.value.calls).toEqual([{offset: CHUNK_BYTE_OFFSET + 4 * 4, length: 8}]);
    expect(vectors.countValidity.calls).toEqual([{offset: CHUNK_BYTE_OFFSET + 4 * 4, length: 8}]);
    expect(rows.readbackByteLength).toBe(5 * 8);
    expect(rows.table.getChild('value')!.get(5)).toBeCloseTo(5.5);
    expect(rows.table.getChild('value')!.get(4)).toBeCloseTo(4.5);
    expect(rows.table.getChild('value')!.get(3)).toBeNull();
    expect(rows.table.getChild('value')!.get(6)).toBeNull();
  });

  test('a page never crosses a chunk boundary', async () => {
    const {rows, vectors} = makeFixture();
    // Row 6 starts chunk 1; a page of 4 rows from row 6 would span 6..9, entirely inside chunk 1.
    await rows.ensureRows([6]);
    expect(vectors.value.calls).toEqual([{offset: CHUNK_BYTE_OFFSET, length: 16}]);
    expect(rows.loadedRowCount).toBe(4);
    expect(rows.isRowLoaded(5)).toBe(false);
    expect(rows.isRowLoaded(9)).toBe(true);
    expect(rows.isRowLoaded(10)).toBe(false);
    await rows.ensureRows([10]);
    // Final page of chunk 1 holds a single row.
    expect(vectors.value.calls[1]).toEqual({offset: CHUNK_BYTE_OFFSET + 4 * 4, length: 4});
    expect(rows.loadedRowCount).toBe(5);
    // The last page of chunk 0 (2 rows) does not borrow rows from chunk 1.
    await rows.ensureRows([4]);
    expect(vectors.value.calls[2]).toEqual({offset: CHUNK_BYTE_OFFSET + 4 * 4, length: 8});
    expect(rows.loadedRowCount).toBe(7);
    expect(rows.table.getChild('value')!.get(6)).toBeCloseTo(10.5);
    expect(rows.table.getChild('value')!.get(10)).toBeCloseTo(14.5);
  });

  test('decodes float32, sint32 and uint32 values', async () => {
    const {rows} = makeFixture();
    await rows.ensureRows([0, 6, 10]);
    const value = rows.table.getChild('value')!;
    const delta = rows.table.getChild('delta')!;
    const count = rows.table.getChild('count')!;
    expect(value.get(2)).toBeCloseTo(2.5);
    expect(value.get(8)).toBeCloseTo(12.5);
    expect(delta.get(0)).toBe(-1);
    expect(delta.get(3)).toBe(-4);
    expect(delta.get(9)).toBe(-10);
    expect(delta.get(10)).toBe(-11);
    expect(count.get(0)).toBe(100);
    expect(count.get(7)).toBe(107);
    expect(count.get(10)).toBe(110);
  });

  test('validity vector marks null rows', async () => {
    const {rows} = makeFixture();
    await rows.ensureRows([0, 6]);
    const count = rows.table.getChild('count')!;
    expect(count.get(0)).toBe(100);
    expect(count.get(1)).toBeNull();
    expect(count.get(2)).toBe(102);
    expect(count.get(8)).toBeNull();
    expect(count.get(9)).toBe(109);
    // Columns without a validity vector stay valid once loaded.
    expect(rows.table.getChild('value')!.get(1)).toBeCloseTo(1.5);
  });

  test('dictionary column decodes labels and reads out-of-range codes as null', async () => {
    const {rows} = makeFixture();
    await rows.ensureRows([0, 4, 6, 10]);
    const category = rows.table.getChild('category')!;
    expect(category.get(0)).toBe('red');
    expect(category.get(1)).toBe('green');
    expect(category.get(2)).toBe('blue');
    expect(category.get(3)).toBe('red');
    expect(category.get(4)).toBe('green');
    expect(category.get(5)).toBeNull();
    expect(category.get(6)).toBe('blue');
    expect(category.get(8)).toBe('red');
    expect(category.get(9)).toBeNull();
    expect(category.get(10)).toBe('blue');
  });

  test('null counts are recomputed after pages load', async () => {
    const {rows} = makeFixture();
    const count = rows.table.getChild('count')!;
    expect(count.nullCount).toBe(11);
    await rows.ensureRows([0, 4, 6, 10]);
    const refreshed = rows.table.getChild('count')!;
    expect(refreshed.nullCount).toBe(2);
  });

  test('repeated and concurrent requests for a page read once', async () => {
    const {rows, vectors, allCalls} = makeFixture();
    await Promise.all([rows.ensureRows([0]), rows.ensureRows([1, 2]), rows.ensureRows([3])]);
    expect(allCalls()).toBe(5);
    expect(vectors.value.calls).toHaveLength(1);
    expect(rows.loadedRowCount).toBe(4);
    await rows.ensureRows([0, 1, 2, 3]);
    expect(allCalls()).toBe(5);
    expect(rows.loadedRowCount).toBe(4);
    expect(rows.readbackByteLength).toBe(5 * 16);
  });

  test('requesting rows from several pages loads each once', async () => {
    const {rows, vectors} = makeFixture();
    await rows.ensureRows([0, 4, 6, 10, 10]);
    expect(vectors.value.calls).toHaveLength(4);
    expect(rows.loadedRowCount).toBe(4 + 2 + 4 + 1);
    expect(rows.isRowLoaded(10)).toBe(true);
  });

  test('subscribe listeners fire once per loaded page and can unsubscribe', async () => {
    const {rows} = makeFixture();
    const listener = vi.fn();
    const unsubscribe = rows.subscribe(listener);
    await rows.ensureRows([0]);
    expect(listener).toHaveBeenCalledTimes(1);
    await rows.ensureRows([0]);
    expect(listener).toHaveBeenCalledTimes(1);
    await rows.ensureRows([6, 10]);
    expect(listener).toHaveBeenCalledTimes(3);
    unsubscribe();
    await rows.ensureRows([4]);
    expect(listener).toHaveBeenCalledTimes(3);
  });

  test('rows outside the dataset reject', async () => {
    const {rows, allCalls} = makeFixture();
    await expect(rows.ensureRows([11])).rejects.toThrow();
    await expect(rows.ensureRows([-1])).rejects.toThrow();
    await expect(rows.ensureRows([1.5])).rejects.toThrow();
    await expect(rows.ensureRows([0, 99])).rejects.toThrow();
    expect(allCalls()).toBe(0);
    expect(rows.isRowLoaded(11)).toBe(false);
    expect(rows.isRowLoaded(-1)).toBe(false);
  });

  test('close drops later reads', async () => {
    const {rows, allCalls} = makeFixture();
    const listener = vi.fn();
    rows.subscribe(listener);
    await rows.ensureRows([0]);
    const callsBefore = allCalls();
    rows.close();
    await rows.ensureRows([6]);
    expect(allCalls()).toBe(callsBefore);
    expect(rows.isRowLoaded(6)).toBe(false);
    expect(rows.loadedRowCount).toBe(4);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  test('close during an in-flight read discards its results', async () => {
    const {rows} = makeFixture();
    const pending = rows.ensureRows([0]);
    rows.close();
    await pending;
    expect(rows.isRowLoaded(0)).toBe(false);
    expect(rows.loadedRowCount).toBe(0);
    expect(rows.table.getChild('value')!.get(0)).toBeNull();
  });

  test('a lost device skips reads', async () => {
    const {rows, device, allCalls} = makeFixture();
    (device as {isLost: boolean}).isLost = true;
    await rows.ensureRows([0]);
    expect(allCalls()).toBe(0);
    expect(rows.isRowLoaded(0)).toBe(false);
  });
});

describe('KeplerResidentRows domains', () => {
  test('setDomains copies the supplied record', () => {
    const {rows} = makeFixture();
    expect(rows.domains).toEqual({});
    const domains: Record<string, readonly [number, number] | null> = {
      value: [0.5, 14.5],
      count: null
    };
    rows.setDomains(domains);
    expect(rows.domains).toEqual({value: [0.5, 14.5], count: null});
    domains.delta = [1, 2];
    expect(rows.domains.delta).toBeUndefined();
  });
});

describe('estimateKeplerResidentRowCacheByteLength', () => {
  test('sums values, bitmaps and dictionary label bytes', () => {
    const schema: KeplerResidentColumnSchema[] = [
      {name: 'value', format: 'float32'},
      {name: 'category', format: 'uint32', dictionary: ['red', 'green']}
    ];
    // Chunk of 6: 2 * (24 + 1) = 50. Chunk of 5: 2 * (20 + 1) = 42. Labels: (3 + 5) * 2 = 16.
    expect(estimateKeplerResidentRowCacheByteLength(schema, [6, 5])).toBe(50 + 42 + 16);
    // Bitmap rounds up: 9 rows need 2 bytes. 1 * (36 + 2) = 38.
    expect(
      estimateKeplerResidentRowCacheByteLength([{name: 'value', format: 'float32'}], [9])
    ).toBe(38);
    expect(estimateKeplerResidentRowCacheByteLength(schema, [])).toBe(16);
  });
});

describe('makeResidentPreparedTable', () => {
  test('reports source batch offsets and dictionaries', () => {
    const {rows, schema} = makeFixture();
    const prepared = makeResidentPreparedTable(rows, schema);
    expect(prepared.table).toBe(rows.table);
    expect(prepared.sourceRowCount).toBe(11);
    expect(prepared.sourceBatches).toEqual([
      {sourceBatchIndex: 0, sourceRowIndexOffset: 0, sourceRowCount: 6},
      {sourceBatchIndex: 1, sourceRowIndexOffset: 6, sourceRowCount: 5}
    ]);
    expect(prepared.dictionaries).toEqual({
      category: {values: ['red', 'green', 'blue'], ordered: false}
    });
    expect(prepared.timeColumns).toEqual({});
    expect(prepared.conversions).toEqual([]);
  });
});
