// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {
  getResidentChartPlanKey,
  filterResidentHoverCharts,
  isResidentChartOrderOperation,
  needsResidentChartMeasureWidening,
  planResidentChart,
  type NativeChartConfiguration,
  type ResidentChartField
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

// Load the example's exact pinned native builder rather than an unrelated hoisted chart version.
const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts') as {
  createBigNumberChart(props: Record<string, unknown>): NativeChartConfiguration;
  createBarChart(props: Record<string, unknown>): NativeChartConfiguration;
};
const fields: ResidentChartField[] = [
  {name: 'amount', format: 'float32'},
  {name: 'integer', format: 'sint32'},
  {name: 'category', format: 'uint32', categorical: true},
  {kind: 'time', name: 'time', unit: 'ms', sourceType: 'timestamp'}
];
const chart = (aggregation = 'count', field: string | null = 'amount', applyFilters = true) =>
  native.createBigNumberChart({
    id: 'chart',
    dataId: 'source',
    applyFilters,
    axis: {field: field ? {name: field, type: 'real'} : null, aggregation}
  });

test('native scalar charts map population and numeric measures without using CPU indices', () => {
  for (const [aggregation, operation] of Object.entries({
    count: 'count',
    sum: 'sum',
    average: 'mean',
    minimum: 'min',
    maximum: 'max'
  })) {
    const result = planResidentChart(chart(aggregation), 'source', fields);
    expect(result).toEqual({
      supported: true,
      plan: {
        kind: 'scalar',
        chartId: 'chart',
        datasetId: 'source',
        population: 'selected',
        operation,
        ...(operation === 'count' ? {} : {column: 'amount'})
      }
    });
  }
  expect(planResidentChart(chart('sum', null, false), 'source', fields)).toMatchObject({
    supported: true,
    plan: {operation: 'count', population: 'all'}
  });
  expect(planResidentChart(chart('count', 'category'), 'source', fields)).toMatchObject({
    supported: true,
    plan: {operation: 'count'}
  });
});

test('scalar charts offer median and deviation, and integer measures plan for exact widening', () => {
  expect(planResidentChart(chart('median'), 'source', fields)).toMatchObject({
    supported: true,
    plan: {kind: 'scalar', operation: 'median', column: 'amount'}
  });
  expect(planResidentChart(chart('stdev', 'integer'), 'source', fields)).toMatchObject({
    supported: true,
    plan: {operation: 'standardDeviation', column: 'integer'}
  });
  expect(planResidentChart(chart('sum', 'integer'), 'source', fields)).toMatchObject({
    supported: true,
    plan: {operation: 'sum', column: 'integer'}
  });
  expect(planResidentChart(chart('median', 'category'), 'source', fields)).toMatchObject({
    supported: false,
    reason: expect.stringContaining('dictionary')
  });
  expect(isResidentChartOrderOperation('median')).toBe(true);
  expect(isResidentChartOrderOperation('mean')).toBe(false);
  // Float32 never widens; scalar integer extrema stay native; everything else widens.
  expect(needsResidentChartMeasureWidening('scalar', 'sum', 'float32')).toBe(false);
  expect(needsResidentChartMeasureWidening('scalar', 'min', 'sint32')).toBe(false);
  expect(needsResidentChartMeasureWidening('category', 'max', 'sint32')).toBe(true);
  expect(needsResidentChartMeasureWidening('scalar', 'median', 'uint32')).toBe(true);
  expect(needsResidentChartMeasureWidening('time', 'count', 'uint32')).toBe(false);
});

test('unsupported measure, stale field, dictionary and integer overflow contracts fail explicitly', () => {
  for (const aggregation of [
    'variance',
    'p05',
    'p25',
    'p50',
    'p75',
    'p95',
    'mode',
    'countUnique'
  ]) {
    expect(planResidentChart(chart(aggregation), 'source', fields)).toMatchObject({
      supported: false,
      reason: expect.stringContaining(aggregation)
    });
  }
  expect(planResidentChart(chart('sum', 'missing'), 'source', fields)).toMatchObject({
    supported: false,
    reason: expect.stringContaining('unavailable')
  });
  expect(planResidentChart(chart('maximum', 'time'), 'source', fields)).toMatchObject({
    supported: false,
    reason: expect.stringContaining('timestamps')
  });
  expect(planResidentChart(chart('minimum', 'integer'), 'source', fields)).toMatchObject({
    supported: true,
    plan: {operation: 'min'}
  });
  expect(planResidentChart(chart('maximum', 'category'), 'source', fields)).toMatchObject({
    supported: false,
    reason: expect.stringContaining('dictionary')
  });
  expect(planResidentChart(chart(), 'other', fields)).toMatchObject({supported: false});
  expect(
    planResidentChart(native.createBarChart({id: 'bar', dataId: 'source'}), 'source', fields)
  ).toMatchObject({supported: false, reason: expect.stringContaining('grouped-summary')});
});

test('native title/palette changes retain the execution key while population and measure replace it', () => {
  const base = chart('average');
  const result = planResidentChart(base, 'source', fields);
  const renamed = planResidentChart({...base, title: 'New title'}, 'source', fields);
  const all = planResidentChart(chart('average', 'amount', false), 'source', fields);
  const sum = planResidentChart(chart('sum'), 'source', fields);
  if (!result.supported || !renamed.supported || !all.supported || !sum.supported)
    throw new Error('Expected native scalar plans');
  expect(getResidentChartPlanKey(result.plan)).toBe(getResidentChartPlanKey(renamed.plan));
  expect(getResidentChartPlanKey(result.plan)).not.toBe(getResidentChartPlanKey(all.plan));
  expect(getResidentChartPlanKey(result.plan)).not.toBe(getResidentChartPlanKey(sum.plan));
});

test('resident tooltip charts cannot enter the native CPU chart consumer, including mismatched source IDs', () => {
  const layerChart = {
    id: 'tooltip',
    type: 'layerChart',
    layerId: 'points',
    dataId: 'resident',
    chartType: 'BREAKDOWN_BY_CATEGORY',
    title: 'Tooltip'
  } as NativeChartConfiguration;
  const hostChart = {...layerChart, id: 'host', dataId: 'host'};
  const otherLayer = {...layerChart, id: 'other', layerId: 'other-points'};
  const isResident = (datasetId: string) => datasetId === 'resident';
  expect(
    filterResidentHoverCharts([layerChart, hostChart, otherLayer], 'points', 'resident', isResident)
  ).toEqual({charts: [otherLayer], blocked: true});
  expect(filterResidentHoverCharts([layerChart, hostChart], 'points', 'host', isResident)).toEqual({
    charts: [hostChart],
    blocked: true
  });
  expect(filterResidentHoverCharts([hostChart], 'points', 'host', isResident)).toEqual({
    charts: [hostChart],
    blocked: false
  });
  expect(filterResidentHoverCharts([layerChart], undefined, undefined, isResident)).toEqual({
    charts: [layerChart],
    blocked: false
  });
});
