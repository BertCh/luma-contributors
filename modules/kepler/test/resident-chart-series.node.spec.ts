// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {foldResidentChartSeries} from '../../../examples/deck/kepler-compute/src/resident-chart-series';
import {makeResidentCategoryChartBins} from '../../../examples/deck/kepler-compute/src/resident-chart-bins';
import {makeResidentNumericChartBins} from '../../../examples/deck/kepler-compute/src/resident-numeric-bins';
import {
  getResidentChartPlanKey,
  planResidentChart,
  type NativeChartConfiguration
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts');
const rows: [string | null, string | null, number | null][] = [
  ['Beta', null, 100],
  ['Alpha', 'Two', 4],
  ['Beta', 'One', 2],
  ['Alpha', 'One', 8],
  ['Beta', 'One', 6],
  ['Alpha', 'Two', null],
  ['Beta', 'Two', 10],
  ['Alpha', '', 200],
  ['Gamma', 'Two', null],
  ['Delta', null, 400],
  ['', 'One', 600],
  [null, 'Two', 800]
];
const categories = ['Delta', 'Gamma', 'Alpha', 'Beta', ''];
const series = ['Two', '', 'One', 'Unused'];
function makeCells(operation: string) {
  return categories.flatMap(category =>
    [...series, null].map(seriesLabel => {
      const indexes = rows.flatMap((row, index) =>
        row[0] === category &&
        (seriesLabel === null ? !row[1] : row[1] === seriesLabel && row[1] !== '')
          ? [index]
          : []
      );
      const values = indexes
        .map(index => rows[index][2])
        .filter((value): value is number => value !== null);
      return {
        category: '',
        count: indexes.length,
        firstRow: indexes[0] ?? 0xffffffff,
        value:
          operation === 'count'
            ? indexes.length
            : values.length === 0
              ? 0
              : operation === 'sum'
                ? values.reduce((sum, value) => sum + value, 0)
                : operation === 'average'
                  ? values.reduce((sum, value) => sum + value, 0) / values.length
                  : operation === 'minimum'
                    ? Math.min(...values)
                    : Math.max(...values)
      };
    })
  );
}
const dataset = {
  id: 'source',
  color: [10, 20, 30],
  allIndexes: rows.map((_, index) => index),
  filteredIndex: rows.map((_, index) => index),
  fields: [
    {name: 'category', type: 'string'},
    {name: 'series', type: 'string'},
    {name: 'amount', type: 'real'}
  ],
  getValue: (name: string, index: number) =>
    rows[index][name === 'category' ? 0 : name === 'series' ? 1 : 2]
};

test('bounded series match native sum-of-statistics, missing-series row counts, local colors, ordering and Other', () => {
  for (const type of ['barChart', 'horizontalBar', 'lineChart'])
    for (const operation of ['count', 'sum', 'average', 'minimum', 'maximum'])
      for (const colorBy of ['none', 'category'])
        for (const groupOthers of [false, true]) {
          const categoryAxis = {
            field: {name: 'category', type: 'string'},
            aggregation: 'uniqueBin'
          };
          const measureAxis = {field: {name: 'amount', type: 'real'}, aggregation: operation};
          const chart = native.createBarChart({
            id: 'chart',
            dataId: 'source',
            type,
            xAxis: type === 'horizontalBar' ? measureAxis : categoryAxis,
            yAxis: type === 'horizontalBar' ? categoryAxis : measureAxis,
            groupBy: {field: {name: 'series', type: 'string'}},
            numGroups: 2,
            groupOthers,
            colorBy,
            chartDisplay: {colorRange: {colors: ['red', 'green', 'blue']}}
          });
          chart.type = type;
          const summaries = foldResidentChartSeries(makeCells(operation), categories, series);
          expect(makeResidentCategoryChartBins(chart, summaries, dataset.color)).toEqual(
            native.computeDatasetChart(chart, dataset).bins
          );
          expect(summaries.find(summary => summary.category === 'Delta')).toMatchObject({
            count: 1,
            value: 0,
            firstRow: 9,
            series: []
          });
        }
});

test('interval series preserve empty bins and native outer domain order', () => {
  const numericDataset = {
    ...dataset,
    fields: [{name: 'category', type: 'real'}, ...dataset.fields.slice(1)],
    getValue: (name: string, index: number) =>
      name === 'category'
        ? rows[index][0] === 'Beta'
          ? 0
          : rows[index][0] === 'Alpha'
            ? 2
            : null
        : dataset.getValue(name, index)
  };
  for (const type of ['barChart', 'horizontalBar', 'lineChart'])
    for (const operation of ['count', 'sum', 'average', 'minimum', 'maximum']) {
      const categoryAxis = {field: {name: 'category', type: 'real'}, aggregation: 'numericBin'};
      const measureAxis = {field: {name: 'amount', type: 'real'}, aggregation: operation};
      const chart = native.createBarChart({
        id: 'chart',
        dataId: 'source',
        xAxis: type === 'horizontalBar' ? measureAxis : categoryAxis,
        yAxis: type === 'horizontalBar' ? categoryAxis : measureAxis,
        groupBy: {field: {name: 'series', type: 'string'}},
        numGroups: 4,
        groupOthers: true
      });
      chart.type = type;
      const binCount = type === 'lineChart' ? 10 : 4;
      const boundaries = Array.from({length: binCount + 1}, (_, index) => index * (2 / binCount));
      const summaries = foldResidentChartSeries(makeCells(operation), categories, series);
      const bins = Array.from({length: binCount}, (_, index) =>
        index === 0
          ? summaries[3]
          : index === binCount - 1
            ? summaries[2]
            : {category: '', count: 0, value: 0, firstRow: 0xffffffff, series: []}
      );
      expect(makeResidentNumericChartBins(chart, bins, boundaries, dataset.color)).toEqual(
        native.computeDatasetChart(chart, numericDataset).bins
      );
    }
});

test('series plans bound cross-products and include series in execution identity', () => {
  const chart: NativeChartConfiguration = native.createBarChart({
    id: 'chart',
    dataId: 'source',
    xAxis: {field: {name: 'category', type: 'string'}},
    groupBy: {field: {name: 'series', type: 'string'}}
  });
  const fields = [
    {name: 'category', format: 'uint32' as const, categorical: true, categoryCount: 512},
    {name: 'series', format: 'uint32' as const, categorical: true, categoryCount: 7}
  ];
  const accepted = planResidentChart(chart, 'source', fields);
  expect(accepted).toMatchObject({
    supported: true,
    plan: {series: {column: 'series', categoryCount: 7}}
  });
  expect(
    planResidentChart(chart, 'source', [fields[0], {...fields[1], categoryCount: 8}])
  ).toMatchObject({supported: false});
  expect(planResidentChart(chart, 'source', [fields[0]])).toMatchObject({supported: false});
  expect(
    planResidentChart(
      {...chart, groupBy: {field: {name: 'series', type: 'integer'}}},
      'source',
      fields
    )
  ).toMatchObject({supported: false});
  if (!accepted.supported || accepted.plan.kind !== 'category')
    throw new Error('Expected category plan');
  expect(getResidentChartPlanKey(accepted.plan)).not.toBe(
    getResidentChartPlanKey({...accepted.plan, series: undefined})
  );
  const integer = {
    ...chart,
    xAxis: {field: {name: 'category', type: 'integer'}, aggregation: 'uniqueBin'},
    numGroups: 4
  };
  expect(
    planResidentChart(integer, 'source', [
      {name: 'category', format: 'sint32'},
      {...fields[1], categoryCount: 512}
    ])
  ).toMatchObject({supported: true});
});
