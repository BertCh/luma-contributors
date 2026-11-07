// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {makeResidentNumericBoundaries} from '../../../examples/deck/kepler-compute/src/resident-numeric-domain';
import {makeResidentNumericChartBins} from '../../../examples/deck/kepler-compute/src/resident-numeric-bins';
import {makeResidentCategoryChartBins} from '../../../examples/deck/kepler-compute/src/resident-chart-bins';
import {
  planResidentChart,
  getResidentChartPlanKey
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts');

test('integer automatic charts match native unique groups, interval fallback, ranking and singleton range filters', () => {
  const fixtures = [
    [3, 1, 3, null, 2, 1],
    [0, -0, 1, 2, NaN, Infinity, -Infinity],
    [4294967295, 4294967294, 4294967295, 4294967293],
    [-2147483648, 2147483647, -1, -2147483648, 0],
    [null, null],
    [7, 7, 7]
  ];
  for (const values of fixtures)
    for (const numGroups of [0, 1, 2, 3, 10])
      for (const horizontal of [false, true])
        for (const operation of ['count', 'sum', 'average', 'minimum', 'maximum']) {
          const amounts = values.map((_, index) => (index === 1 ? null : (index % 3) * 10));
          const categoryAxis = {field: {name: 'value', type: 'integer'}, aggregation: 'uniqueBin'};
          const measureAxis = {field: {name: 'amount', type: 'real'}, aggregation: operation};
          const chart = (horizontal ? native.createHorizontalBarChart : native.createBarChart)({
            id: 'integer',
            dataId: 'source',
            numGroups,
            groupOthers: true,
            xAxis: horizontal ? measureAxis : categoryAxis,
            yAxis: horizontal ? categoryAxis : measureAxis
          });
          const unique = [...new Set(values.filter(value => value !== null))];
          const useUnique = !(numGroups > 0 && unique.length > numGroups);
          const finite = values.filter(
            (value): value is number => value !== null && Number.isFinite(value)
          );
          const boundaries =
            useUnique || !finite.length
              ? []
              : makeResidentNumericBoundaries(
                  Math.min(...finite) || 0,
                  Math.max(...finite) || 0,
                  Math.max(2, numGroups)
                );
          const rows = Array.from(
            {length: useUnique ? unique.length : boundaries.length - 1},
            () => [] as number[]
          );
          values.forEach((value, index) => {
            if (value === null || (!useUnique && !Number.isFinite(value))) return;
            const group = useUnique
              ? unique.findIndex(candidate => String(candidate) === String(value))
              : boundaries.slice(1, -1).filter(edge => value >= edge).length;
            rows[group].push(index);
          });
          const summaries = rows.map((indexes, index) => {
            const measurements = indexes
              .map(row => amounts[row])
              .filter((value): value is number => value !== null);
            const sum = measurements.reduce((total, value) => total + value, 0);
            return {
              category: String(index),
              count: indexes.length,
              firstRow: indexes[0] ?? 0xffffffff,
              value:
                operation === 'count'
                  ? indexes.length
                  : !measurements.length
                    ? 0
                    : operation === 'sum'
                      ? sum
                      : operation === 'average'
                        ? sum / measurements.length
                        : operation === 'minimum'
                          ? Math.min(...measurements)
                          : Math.max(...measurements)
            };
          });
          const dataset = {
            id: 'source',
            fields: [categoryAxis.field, measureAxis.field],
            allIndexes: values.map((_, index) => index),
            filteredIndex: values.map((_, index) => index),
            getValue: (field: string, index: number) =>
              field === 'value' ? values[index] : amounts[index]
          };
          expect(
            makeResidentNumericChartBins(
              chart,
              summaries,
              boundaries,
              undefined,
              useUnique ? unique : undefined
            )
          ).toEqual(native.computeDatasetChart(chart, dataset).bins);
        }
});

test('integer planning retains native auto group budget in execution identity', () => {
  const chart = native.createBarChart({
    id: 'integer',
    dataId: 'source',
    xAxis: {field: {name: 'value', type: 'integer'}, aggregation: 'uniqueBin'},
    numGroups: 1
  });
  expect(planResidentChart(chart, 'source', [{name: 'value', format: 'uint32'}])).toMatchObject({
    supported: true,
    plan: {kind: 'numeric', integerGroupLimit: 1, binCount: 2}
  });
  expect(
    planResidentChart({...chart, numGroups: 0}, 'source', [{name: 'value', format: 'sint32'}])
  ).toMatchObject({supported: true, plan: {integerGroupLimit: 0, binCount: 10}});
  expect(
    planResidentChart({...chart, numGroups: 0.5}, 'source', [{name: 'value', format: 'float32'}])
  ).toMatchObject({supported: false});
  const one = planResidentChart(chart, 'source', [{name: 'value', format: 'uint32'}]);
  const two = planResidentChart({...chart, numGroups: 2}, 'source', [
    {name: 'value', format: 'uint32'}
  ]);
  if (!one.supported || !two.supported) throw new Error('Expected integer plans');
  // Both use two fallback bins, but must rebuild when the unique-group crossover changes.
  expect(getResidentChartPlanKey(one.plan)).not.toBe(getResidentChartPlanKey(two.plan));
});

test('single-series lines match native full-series ordering, palettes and numeric bins', () => {
  const fixtures = [
    {type: 'integer', values: [2, 10, 1, 10, 2]},
    {type: 'string', values: ['b', 'a', 'z', 'b', 'a']},
    {type: 'real', values: [0, 0.25, 0.75, 1, 0.25]}
  ];
  for (const fixture of fixtures) {
    const chart = native.createLineChart({
      id: 'line',
      dataId: 'source',
      xAxis: {field: {name: 'value', type: fixture.type}, aggregation: 'uniqueBin'},
      yAxis: {field: null, aggregation: 'count'},
      chartDisplay: {colorRange: {colors: ['red', 'blue']}}
    });
    const dataset = {
      id: 'source',
      fields: [{name: 'value', type: fixture.type}],
      allIndexes: fixture.values.map((_, index) => index),
      filteredIndex: fixture.values.map((_, index) => index),
      getValue: (_name: string, index: number) => fixture.values[index]
    };
    const groups = fixture.type === 'real' ? [] : [...new Set(fixture.values)];
    const boundaries = fixture.type === 'real' ? makeResidentNumericBoundaries(0, 1, 10) : [];
    const rows = Array.from(
      {length: fixture.type === 'real' ? 10 : groups.length},
      () => [] as number[]
    );
    fixture.values.forEach((value, row) =>
      rows[
        fixture.type === 'real'
          ? boundaries.slice(1, -1).filter(edge => Number(value) >= edge).length
          : groups.indexOf(value)
      ].push(row)
    );
    const summaries = rows.map((indexes, index) => ({
      category: String(groups[index]),
      count: indexes.length,
      value: indexes.length,
      firstRow: indexes[0] ?? 0xffffffff
    }));
    const actual =
      fixture.type === 'string'
        ? makeResidentCategoryChartBins(chart, summaries, [255, 0, 0])
        : makeResidentNumericChartBins(
            chart,
            summaries,
            boundaries,
            [255, 0, 0],
            fixture.type === 'integer' ? groups.map(Number) : undefined
          );
    expect(actual).toEqual(native.computeDatasetChart(chart, dataset).bins);
    const plan = planResidentChart(chart, 'source', [
      {
        name: 'value',
        format: fixture.type === 'real' ? 'float32' : 'uint32',
        ...(fixture.type === 'string' ? {categorical: true, categoryCount: groups.length} : {})
      }
    ]);
    expect(plan).toMatchObject({
      supported: true,
      plan: {kind: fixture.type === 'string' ? 'category' : 'numeric'}
    });
    if (fixture.type === 'integer') expect(plan).toMatchObject({plan: {integerGroupLimit: 0}});
  }
});
