// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {expect, test} from 'vitest';
import {
  makeResidentTimeHistogramPlan,
  makeTimeHistogramWordEdges
} from '../../../examples/deck/kepler-compute/src/resident-time-histogram';
import {installKeplerComputeTableBridge} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

const require = createRequire(import.meta.url);
const nativeRequire = createRequire(require.resolve('@kepler.gl/utils'));
const constants = nativeRequire('@kepler.gl/constants');
const arrays = nativeRequire('d3-array');
const moment = nativeRequire('moment');
const directory = join(dirname(require.resolve('@kepler.gl/utils')), '../esm');
const extract = (file: string, name: string) => {
  const source = readFileSync(join(directory, file), 'utf8');
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`Pinned native helper missing: ${name}`);
  return source.slice(start, source.indexOf('\n}', start) + 2);
};
// Execute pinned native preprocessing and binning, not a reimplementation of their null policy.
const native = runInNewContext(
  [
    extract('data-utils.js', 'timeToUnixMilli'),
    extract('time.js', 'intervalToFunction'),
    extract('time.js', 'getFilterMappedValue'),
    extract('plot.js', 'getBinThresholds'),
    extract('plot.js', 'histogramFromThreshold'),
    extract('plot.js', 'binByTime'),
    '({timeToUnixMilli,getBinThresholds,binByTime})'
  ].join('\n'),
  {
    TIME_INTERVALS: constants.TIME_INTERVALS,
    durationDay: 86400000,
    moment,
    Date,
    _slicedToArray: (value: unknown[]) => value,
    d3Histogram: arrays.histogram,
    notNullorUndefined: (value: unknown) => value != null,
    getEndMappedValue: () => null
  }
) as {
  timeToUnixMilli(value: unknown, format?: string): number | null;
  getBinThresholds(interval: string, domain: number[]): number[];
  binByTime(
    indexes: number[],
    dataset: unknown,
    interval: string,
    filter: unknown
  ): {x0: number; x1: number; count: number}[];
};

test('bounded native time edges match pinned calendar floor/ceil/offset including leap months and negative epochs', () => {
  for (const [interval, domain] of [
    ['1-month', [Date.UTC(2024, 0, 31), Date.UTC(2024, 3, 1)]],
    ['3-month', [Date.UTC(1968, 11, 13), Date.UTC(1970, 1, 7)]],
    ['1-week', [-20 * 86400000, 20 * 86400000]],
    ['2-day', [Date.UTC(2024, 0, 30), Date.UTC(2024, 2, 4)]],
    ['1-year', [Date.UTC(1999, 11, 31), Date.UTC(2001, 0, 1)]],
    ['1-second', [-1000, 0]],
    ['1-millisecond', [0, 0]]
  ] as const) {
    const plan = makeResidentTimeHistogramPlan(domain, interval);
    expect(plan.thresholds).toEqual(native.getBinThresholds(interval, [...domain]));
  }
});

test('pinned native Timestamp preprocessing and histogram coerce null to epoch while NaN is excluded', () => {
  const values = [new Date(-1000), new Date(0), null, new Date(1000), new Date(NaN)];
  const mapped = values.map(value => native.timeToUnixMilli(value));
  expect(mapped).toEqual([-1000, 0, null, 1000, NaN]);
  const dataset = {id: 'source', getColumnField: () => ({filterProps: {mappedValue: mapped}})};
  const filter = {dataId: ['source'], name: ['time'], domain: [-1000, 1000]};
  const bins = native.binByTime([0, 1, 2, 3, 4], dataset, '1-second', filter);
  expect(bins.map(({x0, x1, count}) => ({x0, x1, count}))).toEqual([
    {x0: -1000, x1: 0, count: 1},
    {x0: 0, x1: 1000, count: 2},
    {x0: 1000, x1: 2000, count: 1}
  ]);
  // A source value exactly at the last threshold occupies d3's terminal zero-width bin.
  const terminal = native.binByTime(
    [0],
    {id: 'source', getColumnField: () => ({filterProps: {mappedValue: [2000]}})},
    '1-second',
    filter
  );
  expect(terminal.map(({x0, x1, count}) => ({x0, x1, count}))).toEqual([
    {x0: 2000, x1: 2000, count: 1}
  ]);
});

test('time histogram edges and default selection are bounded before allocating an interval range', () => {
  expect(() => makeResidentTimeHistogramPlan([0, 1_000_000], '1-millisecond')).toThrow('16384');
  // Beyond the former 256 bin cap: 1000 and 16000 one-millisecond bins are planned, with exact edges.
  expect(makeResidentTimeHistogramPlan([0, 1000], '1-millisecond').thresholds.length).toBe(1002);
  expect(makeResidentTimeHistogramPlan([0, 16000], '1-millisecond').thresholds.length).toBe(16002);
  expect(() => makeResidentTimeHistogramPlan([0, 1000], 'invalid')).toThrow('interval');
  expect(() => makeResidentTimeHistogramPlan([0, Infinity])).toThrow('finite');
  expect(
    makeResidentTimeHistogramPlan([Date.UTC(2000, 0, 1), Date.UTC(2020, 0, 1)]).thresholds.length
  ).toBeLessThanOrEqual(65);
});

test('native bridge advertises certified defaults and never reuses counts across interval changes', () => {
  let interval = '1-month';
  const table = installKeplerComputeTableBridge(
    {
      id: 'source',
      fields: [{name: 'time', type: 'timestamp'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable() {}
    },
    {
      fields: {time: {kind: 'time', dimension: 'time', domain: [0, 1000]}},
      filterScopedHistograms: true,
      getTimeHistogramInterval: (_field, filter) => {
        if (filter?.plotType?.interval === 'invalid')
          throw new Error('Unsupported native interval');
        return filter?.plotType?.interval ?? interval;
      },
      explorer: {clear() {}, setRange() {}, setCategories() {}},
      onSelectionChange() {},
      onUnsupported() {}
    }
  );
  expect(table.getColumnFilterProps('time')?.plotType).toEqual({
    type: 'histogram',
    interval: '1-month'
  });
  const filter = {
    id: 'time-filter',
    fixedDomain: true,
    dataId: ['source'],
    name: ['time'],
    type: 'timeRange',
    value: [0, 1000],
    plotType: {type: 'histogram', interval}
  };
  table.filterTable([filter]);
  table.lumaCompute.setFilterHistogram(
    filter.id,
    [{x0: 0, x1: 1000, count: 2, indexes: []}],
    table.lumaCompute.revision,
    interval
  );
  expect(table.lumaCompute.updateFilterPlot(filter).lumaComputePlot?.status).toBe('ready');
  interval = '1-year';
  const next = table.lumaCompute.updateFilterPlot({
    ...filter,
    plotType: {type: 'histogram', interval}
  });
  expect(next.lumaComputePlot?.status).toBe('pending');
  expect(next.timeBins?.source?.[interval]).toBeUndefined();
  const bad = table.lumaCompute.updateFilterPlot({
    ...filter,
    plotType: {type: 'histogram', interval: 'invalid'}
  });
  expect(bad.lumaComputePlot?.status).toBe('unsupported');
  // Plot errors do not replace the validated display selection plan.
  expect(table.lumaCompute.plan.display).toHaveLength(1);
});

test('second sources use directed edges without rounding an instant into the wrong native bin', () => {
  expect(Array.from(makeTimeHistogramWordEdges([-1500, -500, 500, 1500], 'second'))).toEqual([
    -1n,
    0n,
    1n,
    2n,
    1n
  ]);
  expect(Array.from(makeTimeHistogramWordEdges([0, 1, 2], 'microsecond'))).toEqual([
    0n,
    1000n,
    2000n,
    2000n
  ]);
});
