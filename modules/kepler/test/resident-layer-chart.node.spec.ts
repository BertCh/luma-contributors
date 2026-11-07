// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {LayerChartConfig} from '@kepler.gl/types';
import {
  getResidentLayerChartFieldNames,
  readResidentLayerChart,
  type ResidentLayerChartDependencies,
  type ResidentLayerChartRow
} from '../../../examples/deck/kepler-compute/src/resident-layer-chart';

const ROWS: ResidentLayerChartRow[] = [
  {station: 'a', kind: 'x', value: 1, time: '2024-01-01T00:00:00Z'},
  {station: 'a', kind: 'y', value: 2, time: '2024-01-02T00:00:00Z'},
  {station: 'a', kind: 'x', value: 4, time: '2024-01-03T00:00:00Z'},
  {station: 'b', kind: 'z', value: 8, time: '2024-01-04T00:00:00Z'}
];

function makeChart(overrides: Partial<LayerChartConfig>): LayerChartConfig {
  return {
    id: 'c',
    title: 'Layer chart',
    type: 'layerChart',
    dataId: 'd',
    layerId: 'l',
    applyFilters: false,
    display: {},
    layerChartType: 'BREAKDOWN_BY_CATEGORY',
    axis: {field: {name: 'kind', type: 'string'}, aggregation: 'uniqueBin'},
    yAxis: {field: {name: 'value', type: 'integer'}, aggregation: 'sum'},
    chartDisplay: {idField: 'station', numEntries: 5},
    ...overrides
  } as LayerChartConfig;
}

function makeDependencies(withSiblings: boolean): ResidentLayerChartDependencies {
  return {
    readRows: async (rowIndexes, fieldNames) =>
      new Map(
        rowIndexes.map(row => [
          row,
          Object.fromEntries(fieldNames.map(name => [name, ROWS[row][name]]))
        ])
      ),
    findSiblingRows: withSiblings
      ? async (idField, value) =>
          ROWS.flatMap((row, index) => (row[idField] === value ? [index] : []))
      : undefined
  };
}

test('layer chart groups all rows sharing the id field with the native computation', async () => {
  const chart = makeChart({});
  expect(getResidentLayerChartFieldNames(chart).sort()).toEqual(['kind', 'station', 'value']);
  const result = await readResidentLayerChart('d', chart, 0, makeDependencies(true));
  expect(result?.rowCount).toBe(3);
  expect(result?.view.kind).toBe('bars');
  const bins = result?.view.kind === 'bars' ? result.view.bins : [];
  expect(Object.fromEntries(bins.map(bin => [bin.key, bin.value]))).toEqual({x: 5, y: 2});
  expect(result?.notes).toEqual([]);
});

test('without a sibling lookup the hovered row alone is summarised and the limit is stated', async () => {
  const result = await readResidentLayerChart('d', makeChart({}), 3, makeDependencies(false));
  expect(result?.rowCount).toBe(1);
  expect(result?.notes[0]).toContain('station');
});

test('time series layer charts produce a line over the sibling rows', async () => {
  const chart = makeChart({
    layerChartType: 'TIME_SERIES',
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval: 'day'},
    chartDisplay: {idField: 'station', interval: 'day'}
  });
  const result = await readResidentLayerChart('d', chart, 1, makeDependencies(true));
  expect(result?.view.kind).toBe('line');
  expect(result?.view.kind === 'line' ? result.view.bins.length : 0).toBe(3);
});

test('a stale read returns null instead of a chart', async () => {
  const result = await readResidentLayerChart('d', makeChart({}), 0, {
    readRows: async () => null
  });
  expect(result).toBeNull();
});

test('no id field means exactly the hovered row', async () => {
  const result = await readResidentLayerChart(
    'd',
    makeChart({chartDisplay: {idField: null, numEntries: 5}}),
    1,
    makeDependencies(true)
  );
  expect(result?.rowCount).toBe(1);
  expect(result?.notes).toEqual([]);
});
