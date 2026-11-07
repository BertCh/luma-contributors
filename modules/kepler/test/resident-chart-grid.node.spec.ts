// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getResidentGridAxisLimit,
  makeResidentGridSyntheticChart,
  makeResidentHeatmapCells,
  makeResidentPivotTable,
  pickResidentGridEntries
} from '../../../examples/deck/kepler-compute/src/resident-chart-grid';
import {
  MAXIMUM_RESIDENT_GROUPS,
  clampIntegerGroupLimit,
  clampNumericGroupBudget,
  describeCategoryOverflow
} from '../../../examples/deck/kepler-compute/src/resident-chart-group-limits';
import {
  planResidentChart,
  type NativeChartConfiguration,
  type ResidentChartField
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const bin = (key: string, count: number, series: [string, number, number][]) => ({
  key,
  count,
  value: series.reduce((total, [, value]) => total + value, 0),
  color: '#000',
  series: series.map(([seriesKey, value, seriesCount]) => ({
    key: seriesKey,
    value,
    count: seriesCount,
    color: '#111'
  }))
});

test('axis limit follows native numGroups clamped to 1..10', () => {
  expect(getResidentGridAxisLimit(undefined)).toBe(10);
  expect(getResidentGridAxisLimit(0)).toBe(10);
  expect(getResidentGridAxisLimit(3)).toBe(3);
  expect(getResidentGridAxisLimit(500)).toBe(10);
});

test('categorical axes rank by count then name, ordered axes keep domain order', () => {
  const entries = [
    {key: 'b', count: 5},
    {key: 'a', count: 5},
    {key: 'c', count: 9},
    {key: 'd', count: 1}
  ];
  expect(pickResidentGridEntries(entries, 3, false).map(entry => entry.key)).toEqual([
    'a',
    'b',
    'c'
  ]);
  expect(pickResidentGridEntries(entries, 2, true).map(entry => entry.key)).toEqual(['b', 'a']);
});

test('heatmap cells form a full grid with zeros for empty combinations', () => {
  const bins = [
    bin('x1', 3, [
      ['y1', 2, 2],
      ['y2', 1, 1]
    ]),
    bin('x2', 1, [['y2', 7, 1]])
  ];
  const cells = makeResidentHeatmapCells(bins, {limit: 10, orderedX: false});
  expect(cells.map(cell => [cell.x, cell.y, cell.value])).toEqual([
    ['x1', 'y1', 2],
    ['x1', 'y2', 1],
    ['x2', 'y1', 0],
    ['x2', 'y2', 7]
  ]);
  expect(cells[0].filterValueY).toEqual(['y1']);
});

test('pivot tables keep Y series as rows and X groups as columns, bounded by the limit', () => {
  const bins = [
    bin('x1', 3, [['y1', 2, 2]]),
    bin('x2', 2, [
      ['y1', 4, 1],
      ['y2', 1, 1]
    ]),
    bin('x3', 1, [['y2', 9, 1]])
  ];
  const table = makeResidentPivotTable(bins, {limit: 2, orderedX: false});
  expect(table.columnKeys).toEqual(['x1', 'x2']);
  expect(table.rowKeys).toEqual(['y1', 'y2']);
  expect(table.values['y1']).toEqual({x1: 2, x2: 4});
  expect(table.values['y2']).toEqual({x1: 0, x2: 1});
});

test('heatmap and pivot become a category summary with the Y axis as the series', () => {
  const chart = {
    id: 'h',
    type: 'heatmapChart',
    dataId: 'd',
    numGroups: 10,
    xAxis: {field: {name: 'region', type: 'string'}, aggregation: 'uniqueBin'},
    yAxis: {field: {name: 'kind', type: 'string'}, aggregation: 'uniqueBin'},
    value: {field: null, aggregation: 'count'}
  } as unknown as NativeChartConfiguration;
  const synthetic = makeResidentGridSyntheticChart(chart)! as unknown as {
    type: string;
    groupBy: {field: {name: string}};
    numGroups: number;
  };
  expect(synthetic.type).toBe('barChart');
  expect(synthetic.groupBy.field.name).toBe('kind');
  expect(synthetic.numGroups).toBe(0);
  const fields: ResidentChartField[] = [
    {name: 'region', format: 'uint32', categorical: true, categoryCount: 20},
    {name: 'kind', format: 'uint32', categorical: true, categoryCount: 4}
  ];
  const planned = planResidentChart(synthetic as unknown as NativeChartConfiguration, 'd', fields);
  expect(planned.supported).toBe(true);
  expect(planned.supported && planned.plan.kind).toBe('category');
  expect(makeResidentGridSyntheticChart({...chart, type: 'barChart'} as never)).toBeUndefined();
});

test('numeric group budgets cap at 1,000 equal-width bins with a stated note', () => {
  expect(clampNumericGroupBudget(10)).toEqual({groups: 10});
  expect(clampNumericGroupBudget(1000)).toEqual({groups: 1000});
  const clamped = clampNumericGroupBudget(5000);
  expect(clamped.groups).toBe(MAXIMUM_RESIDENT_GROUPS);
  expect(clamped.requested).toBe(5000);
  expect(clamped.note).toContain('1,000 equal-width bins');
  expect(clampIntegerGroupLimit(0)).toEqual({groups: 0});
  expect(clampIntegerGroupLimit(2000).groups).toBe(1000);
  expect(describeCategoryOverflow(5000)).toContain('5,000');
});

function numericChart(numGroups: number, type: 'real' | 'integer' = 'real') {
  return {
    id: 'n',
    type: 'barChart',
    dataId: 'd',
    numGroups,
    applyFilters: false,
    xAxis: {field: {name: 'value', type}, aggregation: 'numericBin'},
    yAxis: {field: null, aggregation: 'count'}
  } as unknown as NativeChartConfiguration;
}
const numericFields: ResidentChartField[] = [{name: 'value', format: 'float32'}];

test('numeric charts above 512 bins are admitted up to 1,000 and clamp beyond with a note', () => {
  const within = planResidentChart(numericChart(900), 'd', numericFields);
  expect(within.supported && within.plan.kind === 'numeric' && within.plan.binCount).toBe(900);
  expect(within.supported && within.notes).toBeFalsy();
  const beyond = planResidentChart(numericChart(5000), 'd', numericFields);
  expect(beyond.supported && beyond.plan.kind === 'numeric' && beyond.plan.binCount).toBe(1000);
  expect(beyond.supported && beyond.notes?.[0]).toContain('1,000 equal-width bins');
});

test('category dictionaries up to 1,000 entries are admitted, larger ones use top groups plus Other, huge ones are refused', () => {
  const chart = {
    id: 'c',
    type: 'barChart',
    dataId: 'd',
    numGroups: 10,
    xAxis: {field: {name: 'name', type: 'string'}, aggregation: 'uniqueBin'},
    yAxis: {field: null, aggregation: 'count'}
  } as unknown as NativeChartConfiguration;
  const field = (categoryCount: number): ResidentChartField[] => [
    {name: 'name', format: 'uint32', categorical: true, categoryCount}
  ];
  expect(planResidentChart(chart, 'd', field(1000)).supported).toBe(true);
  const top = planResidentChart(chart, 'd', field(1001));
  expect(top.supported && top.plan.kind === 'category' && top.plan.topGroups?.limit).toBe(999);
  expect(top.supported && top.notes?.[0]).toContain('1,001');
  const refused = planResidentChart(chart, 'd', field(300_000));
  expect(refused.supported).toBe(false);
  expect(!refused.supported && refused.reason).toContain('300,000');
});
