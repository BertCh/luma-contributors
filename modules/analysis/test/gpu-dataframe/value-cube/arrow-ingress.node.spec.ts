// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import * as ingress from '../../../src/analysis-core/graph-support/arrow-analytics-normalization';
import {
  createArrowIngressFixture,
  createNullableArrowMeasuresFixture
} from './arrow-ingress-fixture';

it('Arrow ingress preserves sliced nullable multi-batch time words and validity', () => {
  const {source, column} = createArrowIngressFixture();
  expect(source.batches).toHaveLength(2);
  expect(
    source.batches.map(batch => {
      const data = batch.getChild('time')!.data[0];
      return [data.nullCount, data.nullBitmap.byteLength > 0];
    })
  ).toEqual([
    [1, true],
    [0, false]
  ]);
  expect(source.batches.map(batch => batch.getChild('time')!.data[0].offset)).toEqual([1, 1]);
  expect(Array.from(source.getChild('time')!)).toEqual([-10, null, 0, 0, 10, 10, 20, 0]);
  expect(column.unit).toBe('millisecond');
  expect(column.validity.map(values => Array.from(values))).toEqual([
    [1, 0, 1, 1],
    [1, 1, 1, 1]
  ]);
  expect(column.chunks.map(values => Array.from(values))).toEqual([
    [0xfffffff6, 0xffffffff, 0, 0, 0, 0, 0, 0],
    [10, 0, 10, 0, 20, 0, 0, 0]
  ]);
  // Null row 1 and epoch-zero row 2 have identical words; only validity distinguishes them.
  expect(Array.from(column.chunks[0].slice(2, 4))).toEqual(
    Array.from(column.chunks[0].slice(4, 6))
  );
});

it('UDF-01 pairs differently named category keys and preserves integer identity', () => {
  const left = new arrow.Table({label: arrow.vectorFromArray(['A', 'B'], new arrow.Utf8())});
  const right = new arrow.Table({name: arrow.vectorFromArray(['B', 'A'], new arrow.Utf8())});
  // Label equality, by hand: A -> right row 1 -> 10; B -> right row 0 -> 20.
  const independentLeft = ingress.prepareArrowTableForGPUAnalytics(left);
  const independentRight = ingress.prepareArrowTableForGPUAnalytics(right);
  expect(Array.from(independentLeft.table.getChild('label')!.data[0].values)).toEqual([0, 1]);
  expect(Array.from(independentRight.table.getChild('name')!.data[0].values)).toEqual([0, 1]);
  const pair = ingress.prepareArrowKeyPairForGPUAnalytics(left, 'label', right, 'name');
  expect(Array.from(pair.left.chunks[0])).toEqual([0, 1]);
  expect(Array.from(pair.right.chunks[0])).toEqual([1, 0]);
  expect(pair.dictionary?.values).toEqual(['A', 'B']);
  expect(pair.left.validity.map(values => Array.from(values))).toEqual([[1, 1]]);
  const wide = new arrow.Table({
    id: arrow.vectorFromArray([9007199254740992n, 9007199254740993n], new arrow.Uint64())
  });
  const integers = ingress.prepareArrowKeyPairForGPUAnalytics(wide, 'id', wide, 'id');
  expect(integers.keyBits).toBe(64);
  expect(Array.from(integers.left.chunks[0])).toEqual([0, 2097152, 1, 2097152]);
  expect(() => ingress.prepareArrowTableForGPUAnalytics(wide)).toThrow(/Uint64/);
  expect(
    Array.from(
      ingress
        .prepareArrowTableForGPUAnalytics(wide, {
          float64: {mode: 'round', maximumAbsoluteError: 1, maximumRelativeError: 0}
        })
        .table.getChild('id')!
    )
  ).toEqual([9007199254740992, 9007199254740992]);
  const reserved = new arrow.Table({
    id: arrow.vectorFromArray([0xffffffffffffffffn], new arrow.Uint64())
  });
  expect(() => ingress.prepareArrowKeyPairForGPUAnalytics(reserved, 'id', wide, 'id')).toThrow(
    /id.*reserved.*all-ones/
  );
});

it('UDF-02 prepares independent nullable measures without excluding eligible keys', () => {
  const source = createNullableArrowMeasuresFixture();
  const columns = ingress.prepareArrowMeasureColumnsForGPUAnalytics(source, ['x', 'y']);
  expect(columns['x'].chunks.map(values => Array.from(values))).toEqual([[NaN, 0], [2]]);
  expect(columns['y'].chunks.map(values => Array.from(values))).toEqual([[12, NaN], [4]]);
  expect(columns['x'].validity.map(values => Array.from(values))).toEqual([[0, 1], [1]]);
  expect(columns['y'].validity.map(values => Array.from(values))).toEqual([[1, 0], [1]]);
  expect(ingress.prepareArrowTableForGPUAnalytics(source).sourceRowCount).toBe(3);
  // Hand calculation: x has 0+2 over 2 values; y has 12+4 over 2 values.
  for (const [name, expectedMean] of [
    ['x', 1],
    ['y', 8]
  ] as const) {
    const finite = columns[name].chunks.flatMap(chunk => Array.from(chunk)).filter(Number.isFinite);
    expect(finite).toHaveLength(2);
    expect(finite.reduce((sum, value) => sum + value, 0) / finite.length).toBe(expectedMean);
  }
});

function makeOrderedTable(labels: string[]) {
  const type = new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32(), 42, true);
  return new arrow.Table({
    rank: new arrow.Vector([
      arrow.makeData({
        type,
        data: Int32Array.of(0, 1),
        dictionary: arrow.vectorFromArray(labels, new arrow.Utf8())
      })
    ])
  });
}

it('ING-04 UDF-16 rejects duplicate ordered labels before first publication and preserves unique codes on reuse', () => {
  expect(() => ingress.prepareArrowTableForGPUAnalytics(makeOrderedTable(['x', 'x']))).toThrow(
    /rank.*0.*1/
  );
  const source = makeOrderedTable(['b', 'a']).concat(makeOrderedTable(['b', 'a']));
  const first = ingress.prepareArrowTableForGPUAnalytics(source);
  const second = ingress.prepareArrowTableForGPUAnalytics(source, {
    dictionaries: first.dictionaries
  });
  for (const prepared of [first, second]) {
    expect(
      prepared.table.batches.map(batch => Array.from(batch.getChild('rank')!.data[0].values))
    ).toEqual([
      [0, 1],
      [0, 1]
    ]);
    expect(prepared.dictionaries['rank'].values).toEqual(['b', 'a']);
  }
});

it('ING-05 rejects invalid precision modes, time units and own unknown keys before projection', () => {
  const source = new arrow.Table({
    x: arrow.vectorFromArray([16777217], new arrow.Float64()),
    time: arrow.vectorFromArray([16777217n], new arrow.Int64())
  });
  expect(() =>
    ingress.prepareArrowTableForGPUAnalytics(source, {
      columns: [],
      float64: {mode: 'excat'} as never
    })
  ).toThrow(/excat.*exact.*round/);
  expect(() =>
    ingress.prepareArrowTableForGPUAnalytics(source, {
      columns: [],
      timeColumns: {time: 'seconds' as never}
    })
  ).toThrow(/seconds.*second.*millisecond.*microsecond.*nanosecond/);
  expect(() =>
    ingress.prepareArrowTableForGPUAnalytics(source, {columns: [], timeColumns: {tyme: 'second'}})
  ).toThrow(/tyme/);
  expect(() =>
    ingress.prepareArrowTableForGPUAnalytics(source, {columns: [], timeColumns: {time: 'second'}})
  ).not.toThrow();
});

it('ING-11 rejects wrapped scalar totals and offers addressable batches above Uint32', () => {
  // Reuse one backing Mi-row array 4096 times: no multibillion-row allocation.
  const batch = new arrow.Table({x: arrow.makeVector(new Float32Array(1048576))}).batches[0];
  const final = new arrow.Table({x: arrow.makeVector(Float32Array.of(9))}).batches[0];
  const source = new arrow.Table(batch.schema, [...new Array(4096).fill(batch), final]);
  expect(source.numRows).toBe(4294967297);
  expect(source.getChild('x')!.length).toBe(1);
  expect(() => ingress.prepareArrowTableForGPUAnalytics(source)).toThrow(
    /4294967297.*4294967295.*prepareArrowBatchesForGPUAnalytics/
  );
  const prepared = [...ingress.prepareArrowBatchesForGPUAnalytics(source)];
  expect(prepared).toHaveLength(4097);
  expect(prepared.at(-1)!.sourceBatches[0].sourceRowIndexOffset).toBe(4294967296);
  expect(prepared.at(-1)!.table.getChild('x')!.get(0)).toBe(9);
  const smaller = ingress.prepareArrowTableForGPUAnalytics(
    new arrow.Table(batch.schema, [...new Array(16).fill(batch), final])
  );
  expect(smaller.sourceRowCount).toBe(16777217);
  expect(smaller.table.getChild('x')!.length).toBe(16777217);
});

it('ING-19 widens Float16 exactly and prepares LargeUtf8 categories with validity', () => {
  const source = new arrow.Table({
    x: arrow.vectorFromArray([1.5, null, -3.25], new arrow.Float16()),
    label: arrow.vectorFromArray(['a', null, 'b'], new arrow.LargeUtf8())
  });
  const prepared = ingress.prepareArrowTableForGPUAnalytics(source);
  expect(String(prepared.table.getChild('x')!.type)).toBe('Float32');
  expect(Array.from(prepared.table.getChild('x')!)).toEqual([1.5, null, -3.25]);
  expect(Array.from(prepared.table.getChild('label')!)).toEqual(['a', null, 'b']);
  expect(prepared.dictionaries['label'].values).toEqual(['a', 'b']);
});

it('UDF-01 paired eligibility preserves nulls and chunk offsets on each side', () => {
  const left = new arrow.Table({
    left: arrow.vectorFromArray(['unused', 'A', null], new arrow.Utf8()).slice(1)
  });
  const right = new arrow.Table({right: arrow.vectorFromArray([null, 'B', 'A'], new arrow.Utf8())});
  const pair = ingress.prepareArrowKeyPairForGPUAnalytics(left, 'left', right, 'right');
  expect(pair.left.validity.map(values => Array.from(values))).toEqual([[1, 0]]);
  expect(pair.right.validity.map(values => Array.from(values))).toEqual([[0, 1, 1]]);
  expect(pair.dictionary?.values).toEqual(['A', 'B']);
  expect(Array.from(pair.right.chunks[0])).toEqual([0, 1, 0]);
});

it('ING-11 batch preparation captures options and retains category codes across batches', () => {
  const first = new arrow.Table({label: arrow.vectorFromArray(['B'], new arrow.Utf8())});
  const last = new arrow.Table({label: arrow.vectorFromArray(['A', 'B'], new arrow.Utf8())});
  const columns = ['label'];
  const labels = ['seed'];
  const prepared = ingress.prepareArrowBatchesForGPUAnalytics(first.concat(last), {
    columns,
    dictionaries: {label: {values: labels, ordered: false}}
  });
  columns[0] = 'missing';
  labels[0] = 'changed';
  const batches = [...prepared];
  expect(batches[1].dictionaries['label'].values).toEqual(['seed', 'B', 'A']);
  expect(batches.map(batch => Array.from(batch.table.getChild('label')!.data[0].values))).toEqual([
    [1],
    [2, 1]
  ]);
  expect(batches[1].sourceBatches[0]).toEqual({
    sourceBatchIndex: 1,
    sourceRowIndexOffset: 1,
    sourceRowCount: 2
  });
});

function makeReviewDictionary(
  name: string,
  labels: (string | null)[],
  indices: number[],
  ordered: boolean
) {
  const type = new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32(), 72, ordered);
  return new arrow.Table({
    [name]: new arrow.Vector([
      arrow.makeData({
        type,
        data: Int32Array.from(indices),
        dictionary: arrow.vectorFromArray(labels, new arrow.Utf8())
      })
    ])
  });
}

it('B1 decoded null dictionary labels have no key eligibility or matches', () => {
  const left = makeReviewDictionary('p', ['p', null, 'q'], [1, 2, 0], false);
  const right = new arrow.Table({q: arrow.vectorFromArray(['q', 'p'], new arrow.Utf8())});
  expect(Array.from(left.getChild('p')!)).toEqual([null, 'q', 'p']);
  const pair = ingress.prepareArrowKeyPairForGPUAnalytics(left, 'p', right, 'q');
  expect(Array.from(pair.left.validity[0])).toEqual([0, 1, 1]);
  // Hand lookup by labels: missing, q, p have 0, 1, 1 matches respectively.
  const matches = Array.from(pair.left.chunks[0], (code, row) =>
    pair.left.validity[0][row]
      ? Array.from(pair.right.chunks[0]).filter(
          (other, index) => pair.right.validity[0][index] && other === code
        ).length
      : 0
  );
  expect(matches).toEqual([0, 1, 1]);
});

it('B4 rejects a Utf8 dictionary byte overflow independently of category count', () => {
  // 8192 labels * 262144 encoded bytes = 2147483648. The category count is harmless;
  // the final signed Int32 Utf8 offset wraps to -2147483648. No payload allocation is needed.
  expect(8192).toBeLessThan(ingress.MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT);
  expect(() =>
    ingress.validateArrowAnalyticsDictionaryByteLength('label', 2147483647)
  ).not.toThrow();
  expect(() => ingress.validateArrowAnalyticsDictionaryByteLength('label', 8192 * 262144)).toThrow(
    /label.*2147483648.*2147483647/
  );
});

it('B5 paired ordered dictionaries reconcile labels independently of ordinal order', () => {
  const left = makeReviewDictionary('p', ['a', 'b'], [0, 1], true);
  const right = makeReviewDictionary('q', ['b', 'a'], [0, 1], true);
  const pair = ingress.prepareArrowKeyPairForGPUAnalytics(left, 'p', right, 'q');
  expect(Array.from(pair.left.chunks[0])).toEqual([0, 1]);
  expect(Array.from(pair.right.chunks[0])).toEqual([1, 0]);
  expect(pair.dictionary?.ordered).toBe(false);
  const plain = new arrow.Table({q: arrow.vectorFromArray(['b', 'a'], new arrow.Utf8())});
  expect(
    Array.from(ingress.prepareArrowKeyPairForGPUAnalytics(left, 'p', plain, 'q').right.chunks[0])
  ).toEqual([1, 0]);
  // Equality across sources does not weaken ordinal consistency within one source.
  const changed = makeReviewDictionary('p', ['b', 'a'], [0, 1], true);
  expect(() =>
    ingress.prepareArrowKeyPairForGPUAnalytics(left.concat(changed), 'p', plain, 'q')
  ).toThrow(/Ordered dictionary.*p.*changes/);
  expect(() => ingress.prepareArrowTableForGPUAnalytics(left.concat(changed))).toThrow(
    /Ordered dictionary.*p.*changes/
  );
});

it('S2 batch projection validation runs for schema-only tables before iteration', () => {
  const schema = new arrow.Schema([new arrow.Field('x', new arrow.Float32())]);
  const source = new arrow.Table(schema, []);
  expect(() => ingress.prepareArrowTableForGPUAnalytics(source, {columns: ['typo']})).toThrow(
    /typo/
  );
  expect(() => ingress.prepareArrowBatchesForGPUAnalytics(source, {columns: ['typo']})).toThrow(
    /typo/
  );
  expect(() => ingress.prepareArrowBatchesForGPUAnalytics(source, {columns: ['x', 'x']})).toThrow(
    /Duplicate/
  );
  expect([...ingress.prepareArrowBatchesForGPUAnalytics(source, {columns: ['x']})]).toEqual([]);
});

it('S3 nullable measure preparation accepts float source fields consistently', () => {
  for (const type of [new arrow.Int32(), new arrow.Int64()]) {
    const source = new arrow.Table({
      x: arrow.vectorFromArray([type.bitWidth === 64 ? 1n : 1], type)
    });
    expect(() =>
      ingress.prepareArrowMeasureColumnsForGPUAnalytics(source, ['x'], {float64: {mode: 'exact'}})
    ).toThrow(/x.*float measure.*Int/);
  }
  for (const type of [new arrow.Float16(), new arrow.Float32(), new arrow.Float64()]) {
    const source = new arrow.Table({x: arrow.vectorFromArray([0, null, 2], type)});
    expect(
      Array.from(
        ingress.prepareArrowMeasureColumnsForGPUAnalytics(source, ['x'], {
          float64: {mode: 'exact'}
        })['x'].chunks[0]
      )
    ).toEqual([0, NaN, 2]);
  }
});

it('N1 schema-only ordered dictionaries remain unobserved until a dictionary arrives', () => {
  const actual = makeReviewDictionary('q', ['v', 'w'], [0, 1], true);
  const schemaOnly = new arrow.Table(actual.schema, []);
  const empty = ingress.prepareArrowTableForGPUAnalytics(schemaOnly);
  const prepared = ingress.prepareArrowTableForGPUAnalytics(actual, {
    dictionaries: empty.dictionaries
  });
  expect(prepared.dictionaries['q'].values).toEqual(['v', 'w']);
  expect(Array.from(prepared.table.getChild('q')!.data[0].values)).toEqual([0, 1]);
  const observedEmpty = makeReviewDictionary('q', [], [], true);
  const observed = ingress.prepareArrowTableForGPUAnalytics(observedEmpty);
  expect(() =>
    ingress.prepareArrowTableForGPUAnalytics(actual, {dictionaries: observed.dictionaries})
  ).toThrow(/Ordered dictionary.*q.*changes/);
});
