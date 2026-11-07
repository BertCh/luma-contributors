// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {
  makeResidentNumericBoundaries,
  encodeResidentNumericThreshold,
  decodeResidentNumericOrder
} from '../../../examples/deck/kepler-compute/src/resident-numeric-domain';
import {makeResidentNumericChartBins} from '../../../examples/deck/kepler-compute/src/resident-numeric-bins';
import {planResidentChart} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import type {GPUResidentScalarFormat} from '../src/exploration/gpu-resident-explorer';

const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts');
const minimumSubnormal = new Float32Array(new Uint32Array([1]).buffer)[0];

test('numeric bins match native exact endpoints, empty intervals, constants and source-format edge ties', () => {
  const fixtures: {format: GPUResidentScalarFormat; values: number[]}[] = [
    {format: 'float32', values: [0, Math.fround(1 / 3), Math.fround(2 / 3), 1]},
    {format: 'float32', values: [1, 1 + 2 ** -23]},
    {format: 'float32', values: [-minimumSubnormal, -0, 0, minimumSubnormal]},
    {format: 'float32', values: [-3.4028234663852886e38, 0, 3.4028234663852886e38]},
    {format: 'float32', values: [7, 7, 7]},
    {format: 'sint32', values: [-2147483648, -1, 0, 2147483647]},
    {format: 'uint32', values: [4294967293, 4294967294, 4294967295]}
  ];
  for (const fixture of fixtures)
    for (const binCount of [2, 3, 10, 512])
      for (const horizontal of [false, true]) {
        const categoryAxis = {
          field: {name: 'value', type: fixture.format === 'float32' ? 'real' : 'integer'},
          aggregation: 'numericBin'
        };
        const measureAxis = {field: null, aggregation: 'count'};
        const chart = (horizontal ? native.createHorizontalBarChart : native.createBarChart)({
          id: 'bins',
          dataId: 'source',
          numGroups: binCount,
          groupOthers: true,
          xAxis: horizontal ? measureAxis : categoryAxis,
          yAxis: horizontal ? categoryAxis : measureAxis
        });
        const dataset = {
          id: 'source',
          fields: [{name: 'value', type: categoryAxis.field.type}],
          allIndexes: fixture.values.map((_, index) => index),
          filteredIndex: fixture.values.map((_, index) => index),
          getValue: (_name: string, index: number) => fixture.values[index]
        };
        const boundaries = makeResidentNumericBoundaries(
          Math.min(...fixture.values),
          Math.max(...fixture.values),
          binCount
        );
        const summaries = Array.from({length: binCount}, (_, index) => ({
          category: String(index),
          count: 0,
          value: 0,
          firstRow: 0xffffffff
        }));
        const thresholds = boundaries
          .slice(1, -1)
          .map(value => encodeResidentNumericThreshold(value, fixture.format));
        for (const value of fixture.values) {
          const ordered = encodeResidentNumericThreshold(value, fixture.format);
          expect(decodeResidentNumericOrder(ordered, fixture.format)).toBe(value === 0 ? 0 : value);
          const index = thresholds.filter(threshold => ordered >= threshold).length;
          summaries[index].count++;
          summaries[index].value++;
        }
        expect(makeResidentNumericChartBins(chart, summaries, boundaries)).toEqual(
          native.computeDatasetChart(chart, dataset).bins
        );
      }
});

test('numeric threshold conversion rounds upward rather than rounding a source value across an edge', () => {
  const threshold = 1 + 2 ** -25;
  expect(Math.fround(threshold)).toBe(1);
  expect(encodeResidentNumericThreshold(1, 'float32')).toBeLessThan(
    encodeResidentNumericThreshold(threshold, 'float32')
  );
  expect(
    decodeResidentNumericOrder(encodeResidentNumericThreshold(threshold, 'float32'), 'float32')
  ).toBe(1 + 2 ** -23);
});

test('planner distinguishes numeric intervals, string categories and unimplemented native integer auto bins', () => {
  const chart = native.createBarChart({
    id: 'bins',
    dataId: 'source',
    numGroups: 3,
    xAxis: {field: {name: 'value', type: 'real'}, aggregation: 'uniqueBin'}
  });
  expect(planResidentChart(chart, 'source', [{name: 'value', format: 'float32'}])).toMatchObject({
    supported: true,
    plan: {kind: 'numeric', binColumn: 'value', binCount: 3, binFormat: 'float32'}
  });
  expect(
    planResidentChart({...chart, xAxis: {...chart.xAxis, aggregation: 'timeBin'}}, 'source', [
      {name: 'value', format: 'float32'}
    ])
  ).toMatchObject({supported: false});
  expect(
    planResidentChart({...chart, numGroups: 513}, 'source', [{name: 'value', format: 'float32'}])
  ).toMatchObject({supported: true, plan: {kind: 'numeric', binCount: 513}});
  expect(
    planResidentChart(
      {...chart, xAxis: {field: {name: 'value', type: 'integer'}, aggregation: 'numericBin'}},
      'source',
      [{name: 'value', format: 'uint32'}]
    )
  ).toMatchObject({supported: true, plan: {kind: 'numeric'}});
});
