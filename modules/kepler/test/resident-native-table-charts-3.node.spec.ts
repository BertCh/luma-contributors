// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {
  applyResidentColumnDelete,
  findResidentColumnReferences,
  getResidentExportColumns,
  planResidentColumnDelete,
  type ResidentVisStateSnapshot
} from '../../../examples/deck/kepler-compute/src/resident-column-edit-propagation';
import {EMPTY_RESIDENT_COLUMN_EDITS} from '../../../examples/deck/kepler-compute/src/resident-column-edits';
import {
  planResidentChart,
  type NativeChartConfiguration,
  type ResidentChartField
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import {
  getResidentTopGroupLimit,
  selectResidentTopGroups
} from '../../../examples/deck/kepler-compute/src/resident-chart-top-groups';
import {makeResidentNumericSeriesLabels} from '../../../examples/deck/kepler-compute/src/resident-chart-numeric-series';
import {chooseTableSortRoute} from '../../../examples/deck/kepler-compute/src/resident-table-population-routes';
import {
  classifyTableSortType,
  compareWideSortWords,
  encodeWideSortWord,
  isDictionarySorted,
  makeDictionaryRankTable
} from '../../../examples/deck/kepler-compute/src/resident-table-sort-keys';

function splitFloat64(value: number): {lo: number; hi: number} {
  const words = new Uint32Array(new Float64Array([value]).buffer);
  return {lo: words[0], hi: words[1]};
}

test('dictionary rank table makes code order equal lexical order, with equal strings sharing a rank', () => {
  const values = ['pear', 'apple', 'fig', 'apple', 'Zed', ''];
  const ranks = makeDictionaryRankTable(values);
  const byRank = values.map((value, code) => ({value, rank: ranks[code]}));
  byRank.sort((first, second) => first.rank - second.rank);
  const sorted = [...values].sort();
  expect(byRank.map(entry => entry.value)).toEqual(sorted);
  expect(ranks[1]).toBe(ranks[3]);
  expect(isDictionarySorted(['a', 'b', 'c'])).toBe(true);
  expect(isDictionarySorted(['b', 'a'])).toBe(false);
});

test('Float64 words sort like the numbers: negatives, zero, infinities, NaN invalid', () => {
  const numbers = [-Infinity, -1e300, -2.5, -1e-300, -0, 0, 1e-300, 2.5, 1e300, Infinity];
  const encoded = numbers.map(number => {
    const {lo, hi} = splitFloat64(number);
    return encodeWideSortWord('float64', lo, hi);
  });
  for (let index = 1; index < encoded.length; index++)
    expect(compareWideSortWords(encoded[index - 1], encoded[index])).toBeLessThanOrEqual(0);
  expect(compareWideSortWords(encoded[4], encoded[5])).toBe(0);
  const nan = splitFloat64(NaN);
  expect(encodeWideSortWord('float64', nan.lo, nan.hi).valid).toBe(false);
});

test('Int64 and Uint64 words order across the 32-bit boundary and the sign', () => {
  const signed = [-(2n ** 63n), -(2n ** 32n) - 1n, -1n, 0n, 1n, 2n ** 32n, 2n ** 63n - 1n].map(
    value => {
      const words = new Uint32Array(new BigInt64Array([value]).buffer);
      return encodeWideSortWord('wide-int64', words[0], words[1]);
    }
  );
  for (let index = 1; index < signed.length; index++)
    expect(compareWideSortWords(signed[index - 1], signed[index])).toBe(-1);
  const unsigned = [0n, 1n, 2n ** 32n, 2n ** 64n - 1n].map(value => {
    const words = new Uint32Array(new BigUint64Array([value]).buffer);
    return encodeWideSortWord('wide-uint64', words[0], words[1]);
  });
  for (let index = 1; index < unsigned.length; index++)
    expect(compareWideSortWords(unsigned[index - 1], unsigned[index])).toBe(-1);
});

test('Arrow types classify into sort families', () => {
  expect(classifyTableSortType(new arrow.Float32())).toBe('float32');
  expect(classifyTableSortType(new arrow.Float64())).toBe('float64');
  expect(classifyTableSortType(new arrow.Int64())).toBe('wide-int64');
  expect(classifyTableSortType(new arrow.Uint64())).toBe('wide-uint64');
  expect(classifyTableSortType(new arrow.Utf8())).toBe('dictionary-string');
  expect(classifyTableSortType(new arrow.TimestampMillisecond())).toBe('wide-int64');
  expect(classifyTableSortType(new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32()))).toBe(
    'dictionary-string'
  );
  expect(classifyTableSortType(new arrow.Int32())).toBe('int32');
  expect(classifyTableSortType(new arrow.Uint32())).toBe('uint32');
});

test('sort routes: bounded CPU at or below 262,144 rows, GPU above, refused only when unsupported', () => {
  const below = {populationRows: 262_144, hasPreparedGPUColumn: true};
  const above = {populationRows: 262_145, hasPreparedGPUColumn: true};
  for (const orderKind of ['dictionary-string', 'wide-int64', 'wide-uint64', 'float64'] as const) {
    expect(chooseTableSortRoute({orderKind, ...below}).route).toBe('cpu');
    expect(chooseTableSortRoute({orderKind, ...above}).route).toBe('gpu');
  }
  expect(chooseTableSortRoute({orderKind: 'dictionary-string', ...above}).reason).toContain(
    'rank table'
  );
  expect(chooseTableSortRoute({orderKind: 'float64', ...above}).reason).toContain('hi/lo');
  const noDictionary = chooseTableSortRoute({
    orderKind: 'dictionary-string',
    populationRows: 1e6,
    hasPreparedGPUColumn: false
  });
  expect(noDictionary.route).toBe('refused');
  expect(noDictionary.reason).toContain('no prepared dictionary');
  const noWords = chooseTableSortRoute({
    orderKind: 'wide-int64',
    populationRows: 1e6,
    hasPreparedGPUColumn: false,
    hasRetainedWideData: false
  });
  expect(noWords.route).toBe('refused');
  expect(noWords.reason).toContain('retained 64-bit words');
  expect(
    chooseTableSortRoute({orderKind: 'float32', populationRows: 1e8, hasPreparedGPUColumn: true})
      .route
  ).toBe('gpu');
});

const state: ResidentVisStateSnapshot = {
  layers: [
    {
      id: 'layer-a',
      config: {
        dataId: 'd',
        label: 'Points',
        columns: {lat: {value: 'lat'}, lng: {value: 'lng'}},
        colorField: {name: 'speed'},
        textLabel: [{field: {name: 'name'}}]
      },
      visualChannels: {colorField: {field: 'colorField'}}
    }
  ],
  filters: [
    {id: 'f0', dataId: ['d'], name: ['speed']},
    {id: 'f1', dataId: ['other', 'd'], name: ['x', 'category']},
    {id: 'f2', dataId: ['other'], name: ['category']}
  ],
  interactionConfig: {
    tooltip: {
      config: {fieldsToShow: {d: [{name: 'speed'}, {name: 'category', format: 'x'}]}}
    }
  }
};

test('column references find layer roles, per-dataset filters and tooltip entries', () => {
  expect(findResidentColumnReferences(state, 'd', 'speed')).toEqual({
    layers: [{layerId: 'layer-a', label: 'Points', roles: ['channel colorField']}],
    filters: [{filterId: 'f0', index: 0}],
    tooltip: true
  });
  expect(findResidentColumnReferences(state, 'd', 'category').filters).toEqual([
    {filterId: 'f1', index: 1}
  ]);
});

test('delete is refused while a layer uses the column, otherwise removes filters (descending) and the tooltip entry', () => {
  const refused = planResidentColumnDelete(state, 'd', 'lat');
  expect(refused.ok).toBe(false);
  expect(!refused.ok && refused.reason).toContain('Points');
  const plan = planResidentColumnDelete(state, 'd', 'category');
  expect(plan).toEqual({
    ok: true,
    removeFilterIndices: [1],
    tooltipFieldsToShow: [{name: 'speed'}]
  });
  const calls: string[] = [];
  const result = applyResidentColumnDelete(
    {
      getVisState: () => state,
      removeFilter: index => calls.push(`remove:${index}`),
      setTooltipFieldsToShow: (datasetId, fields) =>
        calls.push(`tooltip:${datasetId}:${fields.map(field => field.name)}`)
    },
    'd',
    'category'
  );
  expect(result).toEqual({ok: true, removedFilters: 1, tooltipUpdated: true});
  expect(calls).toEqual(['remove:1', 'tooltip:d:speed']);
});

test('export columns omit deleted columns and use rename aliases', () => {
  expect(
    getResidentExportColumns(['a', 'b', 'c'], {
      ...EMPTY_RESIDENT_COLUMN_EDITS,
      names: {a: 'Alpha'},
      deleted: ['b']
    })
  ).toEqual([
    {source: 'a', header: 'Alpha'},
    {source: 'c', header: 'c'}
  ]);
});

test('top groups keep the highest by measure, fold the rest into Other, and drop absent entries', () => {
  const summaries = [
    {value: 5, count: 1, firstRow: 4},
    {value: 9, count: 2, firstRow: 1},
    {value: 7, count: 3, firstRow: 0},
    {value: 0, count: 0, firstRow: 0xffffffff},
    {value: 7, count: 1, firstRow: 9}
  ];
  const labels = ['a', 'b', 'c', 'd', 'e'];
  const selection = selectResidentTopGroups({summaries, labels, limit: 2});
  expect(selection.labels).toEqual(['b', 'c', 'Other']);
  expect(Array.from(selection.bucketOfCode)).toEqual([2, 0, 1, 0xffffffff, 2]);
  expect(selection.hasOther).toBe(true);
  expect(selection.otherCodeCount).toBe(2);
  const all = selectResidentTopGroups({summaries, labels, limit: 10});
  expect(all.hasOther).toBe(false);
  expect(all.labels).toEqual(['b', 'c', 'e', 'a']);
  expect(
    selectResidentTopGroups({summaries, labels: ['a', 'Other', 'c', 'd', 'e'], limit: 4}).labels
  ).not.toContain('Other (remaining groups)');
  expect(
    selectResidentTopGroups({summaries, labels: ['a', 'Other', 'c', 'd', 'e'], limit: 2}).labels.at(
      -1
    )
  ).toBe('Other (remaining groups)');
});

test('top group limit keeps the series cross product inside the cell budget', () => {
  expect(getResidentTopGroupLimit(undefined, 4096)).toBe(999);
  expect(getResidentTopGroupLimit(3, 4096)).toBe(999);
  expect(getResidentTopGroupLimit(7, 4096)).toBe(511);
  expect(getResidentTopGroupLimit(5000, 4096)).toBe(1);
});

const categoryChart = (extra: object = {}) =>
  ({
    id: 'c',
    type: 'barChart',
    dataId: 'd',
    numGroups: 10,
    xAxis: {field: {name: 'name', type: 'string'}, aggregation: 'uniqueBin'},
    yAxis: {field: null, aggregation: 'count'},
    ...extra
  }) as unknown as NativeChartConfiguration;

test('large dictionaries plan top groups with a note; series fit the cell budget', () => {
  const fields: ResidentChartField[] = [
    {name: 'name', format: 'uint32', categorical: true, categoryCount: 50_000},
    {name: 'kind', format: 'uint32', categorical: true, categoryCount: 7}
  ];
  const plain = planResidentChart(categoryChart(), 'd', fields);
  expect(plain.supported && plain.plan.kind === 'category' && plain.plan.topGroups).toEqual({
    limit: 999
  });
  expect(plain.supported && plain.notes?.[0]).toContain('50,000');
  const withSeries = planResidentChart(
    categoryChart({groupBy: {field: {name: 'kind', type: 'string'}}}),
    'd',
    fields
  );
  expect(
    withSeries.supported && withSeries.plan.kind === 'category' && withSeries.plan.topGroups
  ).toEqual({limit: 511});
});

test('numeric series are binned to fit the cell budget with a note', () => {
  const fields: ResidentChartField[] = [
    {name: 'name', format: 'uint32', categorical: true, categoryCount: 20},
    {name: 'speed', format: 'float32'}
  ];
  const chart = categoryChart({
    numGroups: 8,
    groupBy: {field: {name: 'speed', type: 'real'}}
  });
  const planned = planResidentChart(chart, 'd', fields);
  expect(planned.supported).toBe(true);
  if (!planned.supported || planned.plan.kind !== 'category') throw new Error('expected category');
  expect(planned.plan.series).toEqual({column: 'speed', categoryCount: 8});
  expect(planned.plan.numericSeries).toEqual({binFormat: 'float32'});
  expect(planned.notes?.some(note => note.includes('equal-width bins'))).toBe(true);
  const crowded = planResidentChart(chart, 'd', [
    {...fields[0], categoryCount: 1000} as ResidentChartField,
    fields[1]
  ]);
  expect(crowded.supported && crowded.plan.kind === 'category' && crowded.plan.series).toEqual({
    column: 'speed',
    categoryCount: 3
  });
});

test('numeric series labels name intervals and pad unused bins', () => {
  expect(makeResidentNumericSeriesLabels([0, 5, 10], 4)).toEqual(['0 to 5', '5 to 10', '', '']);
  expect(makeResidentNumericSeriesLabels([], 2)).toEqual(['', '']);
});
