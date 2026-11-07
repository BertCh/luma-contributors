// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {makeResidentCategoryChartBins} from '../../../examples/deck/kepler-compute/src/resident-chart-bins';
import {
  planResidentChart,
  type NativeChartConfiguration
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts');
const rows = [
  ['Beta', 4],
  ['Alpha', null],
  ['Gamma', -2],
  ['Beta', 2],
  ['Alpha', 3],
  ['Gamma', 3],
  ['', 100],
  [null, 100],
  ['Delta', null]
];
const dataset = {
  id: 'source',
  color: [10, 20, 30],
  allIndexes: rows.map((_, index) => index),
  filteredIndex: rows.map((_, index) => index),
  fields: [
    {name: 'category', type: 'string'},
    {name: 'value', type: 'real'}
  ],
  getValue: (name: string, index: number) => rows[index][name === 'category' ? 0 : 1]
};

test('bounded dictionary summaries match native first-occurrence colors, ties, nulls, top-N and Other', () => {
  for (const horizontal of [false, true])
    for (const aggregation of ['count', 'average', 'minimum', 'maximum', 'sum'])
      for (const groupOthers of [false, true])
        for (const colorBy of ['category', 'none']) {
          const categoryAxis = {
            field: {name: 'category', type: 'string'},
            aggregation: 'uniqueBin'
          };
          const measureAxis = {field: {name: 'value', type: 'real'}, aggregation};
          const chart = (horizontal ? native.createHorizontalBarChart : native.createBarChart)({
            id: 'bars',
            dataId: 'source',
            applyFilters: true,
            xAxis: horizontal ? measureAxis : categoryAxis,
            yAxis: horizontal ? categoryAxis : measureAxis,
            numGroups: 2,
            groupOthers,
            colorBy,
            chartDisplay: {colorRange: {colors: ['red', 'green', 'blue']}}
          });
          const values =
            aggregation === 'count'
              ? [2, 2, 2, 1]
              : aggregation === 'average'
                ? [3, 3, 0.5, 0]
                : aggregation === 'minimum'
                  ? [2, 3, -2, 0]
                  : aggregation === 'maximum'
                    ? [4, 3, 3, 0]
                    : [6, 3, 1, 0];
          // Deliberately different from native row insertion order and from lexical dictionary order.
          const summaries = [
            {category: 'Delta', value: values[3], count: 1, firstRow: 8},
            {category: 'Gamma', value: values[2], count: 2, firstRow: 2},
            {category: 'Alpha', value: values[1], count: 2, firstRow: 1},
            {category: 'Beta', value: values[0], count: 2, firstRow: 0},
            {category: '', value: 100, count: 1, firstRow: 6},
            {category: 'Unused', value: 0, count: 0, firstRow: 0xffffffff}
          ];
          expect(makeResidentCategoryChartBins(chart, summaries, dataset.color)).toEqual(
            native.computeDatasetChart(chart, dataset).bins
          );
        }
});

test('category planning retains native owned-filter exclusions and rejects unbounded or incompatible groups', () => {
  const chart: NativeChartConfiguration = native.createBarChart({
    id: 'bars',
    dataId: 'source',
    applyFilters: true,
    xAxis: {field: {name: 'category', type: 'string'}, aggregation: 'uniqueBin'},
    yAxis: {field: {name: 'value', type: 'real'}, aggregation: 'average'},
    crossFilter: {enabled: true, filterId: 'owned', fieldNames: {x: 'category'}, value: {x: 'Beta'}}
  });
  const fields = [
    {name: 'category', format: 'uint32' as const, categorical: true, categoryCount: 4},
    {name: 'value', format: 'float32' as const}
  ];
  expect(planResidentChart(chart, 'source', fields)).toMatchObject({
    supported: true,
    plan: {
      kind: 'category',
      categoryColumn: 'category',
      categoryCount: 4,
      operation: 'mean',
      excludedCpuFilterIds: ['owned', 'owned-x', 'owned-y'],
      excludedGpuFieldNames: ['category']
    }
  });
  expect(
    planResidentChart({...chart, crossFilter: {enabled: false}}, 'source', fields)
  ).toMatchObject({supported: true, plan: {excludedCpuFilterIds: [], excludedGpuFieldNames: []}});
  expect(
    planResidentChart(chart, 'source', [{...fields[0], categoryCount: 262_145}, fields[1]])
  ).toMatchObject({supported: false});
  expect(
    planResidentChart(chart, 'source', [{...fields[0], categoryCount: 1001}, fields[1]])
  ).toMatchObject({supported: true, plan: {topGroups: {limit: 999}, categoryCount: 1001}});
  expect(
    planResidentChart(
      {...chart, groupBy: {field: {name: 'category', type: 'string'}}},
      'source',
      fields
    )
  ).toMatchObject({supported: true, plan: {series: {column: 'category', categoryCount: 4}}});
});
