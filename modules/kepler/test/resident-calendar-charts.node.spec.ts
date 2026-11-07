// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  planResidentChart,
  getResidentChartTimeNullEpochFilters,
  getResidentChartPlanKey,
  type ResidentChartField,
  type ResidentTimeChartPlan
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import {chooseNativeTimeInterval} from '../../../examples/deck/kepler-compute/src/resident-time-extent';
import {makeResidentTimeBarPeriods} from '../../../examples/deck/kepler-compute/src/resident-time-bins';
import {makeResidentTimePeriods} from '../../../examples/deck/kepler-compute/src/resident-time-periods';
import {makeResidentTimeChartBins} from '../../../examples/deck/kepler-compute/src/resident-time-bins';
const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts');
const instants = [
  '1969-12-31T23:59:59.999Z',
  '2024-02-29T23:59:59.999Z',
  '2024-03-01T00:00:00.000Z',
  '2024-03-10T01:59:59-05:00',
  '2024-03-10T03:00:00-04:00',
  '2024-11-03T01:30:00-04:00',
  '2024-11-03T01:30:00-05:00',
  '0000-02-01T00:00:00Z',
  '0099-12-31T23:59:59.999Z',
  '-000001-12-31T23:59:59.999Z',
  '1970-01-01T00:00:00.000Z',
  '1970-01-01T00:00:00.001Z',
  '2024-03-02T23:59:59.999Z',
  '2024-03-03T00:00:00.000Z',
  '2024-03-03T00:00:00.001Z',
  '1969-12-27T23:59:59.999Z',
  '1969-12-28T00:00:00.000Z',
  '1969-12-28T00:00:00.001Z'
].map(value => Date.parse(value));
const values = [1, 2, null, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];

function getPeriodOrdinal(instant: number, interval: ResidentTimeChartPlan['interval']): number {
  const date = new Date(instant);
  if (interval === 'year') return date.getUTCFullYear();
  if (interval === 'month') return date.getUTCFullYear() * 12 + date.getUTCMonth();
  const day = Math.floor(instant / 86400000);
  return interval === 'week' ? day - ((((day + 4) % 7) + 7) % 7) : day;
}

const dataset = {
  id: 'source',
  color: [1, 2, 3],
  allIndexes: instants.map((_, index) => index),
  filteredIndex: instants.map((_, index) => index),
  fields: [
    {name: 'time', type: 'timestamp'},
    {name: 'amount', type: 'real'}
  ],
  getValue: (name: string, index: number) => (name === 'time' ? instants[index] : values[index])
};

test('UTC calendar summaries match native day/week/month/year ordering, leap dates, pre-epoch years, DST instants and ranges', () => {
  for (const interval of ['day', 'week', 'month', 'year'] as const)
    for (const operation of ['count', 'sum', 'average', 'minimum', 'maximum']) {
      const groups = new Map<number, number[]>();
      instants.forEach((instant, index) => {
        const key = getPeriodOrdinal(instant, interval);
        groups.set(key, [...(groups.get(key) ?? []), index]);
      });
      const ordered = [...groups.keys()].sort((first, second) => first - second);
      const periods = makeResidentTimePeriods(
        ordered.map(value => (value ^ 0x80000000) >>> 0),
        interval
      );
      const summaries = ordered.map((key, index) => {
        const rows = groups.get(key)!;
        const measures = rows
          .map(row => values[row])
          .filter((value): value is number => value !== null);
        const sum = measures.reduce((total, value) => total + value, 0);
        return {
          category: String(index),
          count: rows.length,
          firstRow: rows[0],
          value:
            operation === 'count'
              ? rows.length
              : !measures.length
                ? 0
                : operation === 'sum'
                  ? sum
                  : operation === 'average'
                    ? sum / measures.length
                    : operation === 'minimum'
                      ? Math.min(...measures)
                      : Math.max(...measures)
        };
      });
      const chart = native.createLineChart({
        id: 'line',
        dataId: 'source',
        xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval},
        yAxis: {field: {name: 'amount', type: 'real'}, aggregation: operation}
      });
      expect(makeResidentTimeChartBins(chart, summaries, periods)).toEqual(
        native.computeDatasetChart(chart, dataset).bins
      );
    }
});

test('calendar admission distinguishes exact sidecars and native interval precedence', () => {
  const fields: ResidentChartField[] = [
    {kind: 'time', name: 'time', unit: 'millisecond', sourceType: 'Timestamp<MILLISECOND>'},
    {name: 'amount', format: 'float32'}
  ];
  const chart = native.createLineChart({
    id: 'line',
    dataId: 'source',
    applyFilters: true,
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval: 'month'},
    yAxis: {field: {name: 'amount', type: 'real'}, aggregation: 'average'},
    chartDisplay: {interval: 'year'},
    crossFilter: {enabled: true, filterId: 'owned'}
  });
  const accepted = planResidentChart(chart, 'source', fields);
  expect(accepted).toMatchObject({
    supported: true,
    plan: {
      kind: 'time',
      interval: 'month',
      operation: 'mean',
      excludedGpuFieldNames: ['time'],
      excludedCpuFilterIds: ['owned', 'owned-x', 'owned-y']
    }
  });
  const fallback = planResidentChart(
    {...chart, xAxis: {...chart.xAxis, interval: null}},
    'source',
    fields
  );
  expect(fallback).toMatchObject({supported: true, plan: {interval: 'year'}});
  if (!accepted.supported || !fallback.supported) throw new Error('Expected calendar plans');
  expect(getResidentChartPlanKey(accepted.plan)).not.toBe(getResidentChartPlanKey(fallback.plan));
  for (const interval of ['hour', 'day', 'week'])
    expect(
      planResidentChart({...chart, xAxis: {...chart.xAxis, interval}}, 'source', fields)
    ).toMatchObject({supported: true, plan: {kind: 'time', interval}});
  expect(
    planResidentChart({...chart, xAxis: {...chart.xAxis, interval: ''}}, 'source', fields)
  ).toMatchObject({supported: true, plan: {kind: 'time', interval: 'auto'}});
  expect(
    planResidentChart({...chart, xAxis: {...chart.xAxis, interval: 'auto'}}, 'source', fields)
  ).toMatchObject({supported: true, plan: {interval: 'day', dateTimeLabels: true}});
  for (const aggregation of ['uniqueBin', 'numericBin'])
    expect(
      planResidentChart({...chart, xAxis: {...chart.xAxis, aggregation}}, 'source', fields)
    ).toMatchObject({supported: false});
  expect(
    planResidentChart(chart, 'source', [
      {name: 'time', kind: 'time', unit: 'second', sourceType: 'Timestamp<SECOND>'},
      fields[1]
    ])
  ).toMatchObject({supported: true, plan: {kind: 'time', interval: 'month'}});
  for (const unit of ['microsecond', 'nanosecond'] as const)
    expect(
      planResidentChart(chart, 'source', [
        {name: 'time', kind: 'time', unit, sourceType: `Timestamp<${unit}>`},
        fields[1]
      ])
    ).toMatchObject({supported: true, plan: {kind: 'time', interval: 'month'}});
  expect(
    planResidentChart(chart, 'source', [
      {name: 'time', kind: 'time', unit: 'second', sourceType: 'Int64'},
      fields[1]
    ])
  ).toMatchObject({supported: false});
  expect(
    planResidentChart({...chart, type: 'barChart', numGroups: 0}, 'source', fields)
  ).toMatchObject({supported: true, plan: {kind: 'time', interval: 'month'}});
  expect(() => makeResidentTimePeriods([(275760 ^ 0x80000000) >>> 0], 'year')).toThrow(
    'Date boundary'
  );
});

test('hour admission preserves source-unit gates, bounded series and owned-filter identity', () => {
  const chart = native.createLineChart({
    id: 'hour',
    dataId: 'source',
    applyFilters: true,
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval: 'hour'},
    yAxis: {field: {name: 'amount', type: 'real'}, aggregation: 'average'},
    crossFilter: {enabled: true, filterId: 'persisted-hour', fieldNames: {x: 'time'}}
  });
  for (const unit of ['second', 'millisecond', 'microsecond', 'nanosecond'] as const) {
    const fields: ResidentChartField[] = [
      {kind: 'time', name: 'time', unit, sourceType: `Timestamp<${unit.toUpperCase()}>`},
      {name: 'amount', format: 'float32'},
      {name: 'series', format: 'uint32', categorical: true, categoryCount: 7}
    ];
    const planned = planResidentChart(chart, 'source', fields);
    expect(planned).toMatchObject({
      supported: true,
      plan: {
        kind: 'time',
        interval: 'hour',
        operation: 'mean',
        population: 'selected',
        excludedCpuFilterIds: ['persisted-hour', 'persisted-hour-x', 'persisted-hour-y'],
        excludedGpuFieldNames: ['time']
      }
    });
    const daily = planResidentChart(
      {...chart, xAxis: {...chart.xAxis, interval: 'day'}},
      'source',
      fields
    );
    if (!planned.supported || !daily.supported) throw new Error('Expected explicit calendar plans');
    expect(getResidentChartPlanKey(planned.plan)).not.toBe(getResidentChartPlanKey(daily.plan));
    expect(
      planResidentChart(
        {...chart, xAxis: {...chart.xAxis, interval: undefined}, chartDisplay: {}},
        'source',
        fields
      )
    ).toMatchObject({supported: true, plan: {interval: 'auto'}});
    const grouped = {
      ...chart,
      groupBy: {field: {name: 'series', type: 'string'}}
    };
    expect(planResidentChart(grouped, 'source', fields)).toMatchObject({
      supported: true,
      plan: {
        interval: 'hour',
        series: {column: 'series', categoryCount: 7}
      }
    });
    expect(
      planResidentChart(grouped, 'source', [
        fields[0],
        fields[1],
        {name: 'series', format: 'uint32', categorical: true, categoryCount: 8}
      ])
    ).toMatchObject({supported: false, reason: expect.stringContaining('4,096')});
  }
  expect(getResidentChartTimeNullEpochFilters([chart], 'source', ['time'])).toEqual([
    {filterId: 'persisted-hour', field: 'time'}
  ]);
});

test('native null timestamp coercion is epoch grouping while invalid dates are excluded', () => {
  for (const interval of ['day', 'week', 'month', 'year'] as const) {
    const chart = native.createLineChart({
      id: 'line',
      dataId: 'source',
      xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval}
    });
    const result = native.computeDatasetChart(chart, {
      ...dataset,
      allIndexes: [0, 1, 2],
      filteredIndex: [0, 1, 2],
      getValue: (_name: string, index: number) =>
        [null, NaN, Date.parse('2024-01-01T00:00:00Z')][index]
    });
    const periods = makeResidentTimePeriods(
      [0, Date.parse('2024-01-01T00:00:00Z')].map(
        instant => (getPeriodOrdinal(instant, interval) ^ 0x80000000) >>> 0
      ),
      interval
    );
    expect(
      result.bins.map((bin: {key: string; count: number; filterValue: number[]}) => [
        bin.key,
        bin.count,
        bin.filterValue
      ])
    ).toEqual(periods.map(period => [period.label, 1, [period.start, period.end]]));
    if (interval === 'week') expect(periods[0].label).toBe('Week of 1969-12-28');
  }
});

test('daily and Sunday-week periods preserve exact negative boundaries and reject clipped endpoints', () => {
  const decode = (ordinal: number, interval: 'day' | 'week') =>
    makeResidentTimePeriods([(ordinal ^ 0x80000000) >>> 0], interval)[0];
  expect(decode(-1, 'day')).toEqual({label: '1969-12-31', start: -86400000, end: 0});
  expect(decode(-4, 'week')).toEqual({
    label: 'Week of 1969-12-28',
    start: -345600000,
    end: 259200000
  });
  expect(getPeriodOrdinal(-345600001, 'week')).toBe(-11);
  expect(getPeriodOrdinal(-345600000, 'week')).toBe(-4);
  expect(getPeriodOrdinal(-345599999, 'week')).toBe(-4);
  const minimum = -8640000000000000;
  const maximum = 8640000000000000;
  expect(decode(-100000000, 'day')).toEqual({
    label: '-271821-04-20',
    start: minimum,
    end: minimum + 86400000
  });
  expect(decode(99999999, 'day').end).toBe(maximum);
  expect(() => decode(100000000, 'day')).toThrow('Date boundary');
  expect(() => decode(-100000001, 'day')).toThrow('Date boundary');
  expect(() => decode(getPeriodOrdinal(minimum, 'week'), 'week')).toThrow('Date boundary');
  expect(() => decode(getPeriodOrdinal(maximum, 'week'), 'week')).toThrow('Date boundary');
  const firstCompleteWeek = decode(getPeriodOrdinal(minimum, 'week') + 7, 'week');
  expect(firstCompleteWeek.start).toBeGreaterThanOrEqual(minimum);
  expect(firstCompleteWeek.end - firstCompleteWeek.start).toBe(604800000);
});

test('chart time-null ownership requires active exact native ID, field and dataset', () => {
  const chart = native.createLineChart({
    id: 'line',
    dataId: 'source',
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin'},
    crossFilter: {
      enabled: true,
      filterId: 'persisted-arbitrary-id',
      fieldNames: {x: 'time'},
      value: {x: '1970'}
    }
  });
  expect(getResidentChartTimeNullEpochFilters([chart], 'source', ['time'])).toEqual([
    {filterId: 'persisted-arbitrary-id', field: 'time'}
  ]);
  expect(getResidentChartTimeNullEpochFilters([chart], 'other', ['time'])).toEqual([]);
  expect(getResidentChartTimeNullEpochFilters([chart], 'source', [])).toEqual([]);
  expect(
    getResidentChartTimeNullEpochFilters(
      [{...chart, crossFilter: {...chart.crossFilter, enabled: false}}],
      'source',
      ['time']
    )
  ).toEqual([]);
  expect(
    getResidentChartTimeNullEpochFilters(
      [{...chart, crossFilter: {...chart.crossFilter, fieldNames: {x: 'other'}}}],
      'source',
      ['time']
    )
  ).toEqual([]);
  expect(
    getResidentChartTimeNullEpochFilters(
      [{...chart, xAxis: {...chart.xAxis, aggregation: 'uniqueBin'}}],
      'source',
      ['time']
    )
  ).toEqual([]);
  expect(getResidentChartTimeNullEpochFilters([], 'source', ['time'])).toEqual([]);
});

test('second source calendar groups match native Arrow Date values and exact click endpoints', () => {
  const monthStart = BigInt(Date.parse('2024-03-01T00:00:00Z') / 1000);
  const secondValues = [-1n, 0n, monthStart - 1n, monthStart, monthStart + 1n];
  const timestamps = arrow.makeVector(
    arrow.makeData({
      type: new arrow.TimestampSecond(),
      data: new BigInt64Array(secondValues)
    })
  );
  for (const interval of ['day', 'week', 'month', 'year'] as const) {
    const chart = native.createLineChart({
      id: 'seconds',
      dataId: 'source',
      xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval}
    });
    const result = native.computeDatasetChart(chart, {
      ...dataset,
      allIndexes: secondValues.map((_, index) => index),
      filteredIndex: secondValues.map((_, index) => index),
      getValue: (_name: string, index: number) => timestamps.get(index)
    });
    const ordinals = secondValues.map(value => getPeriodOrdinal(Number(value * 1000n), interval));
    const ordered = [...new Set(ordinals)].sort((first, second) => first - second);
    const periods = makeResidentTimePeriods(
      ordered.map(value => (value ^ 0x80000000) >>> 0),
      interval
    );
    expect(periods.map(period => period.label)).toEqual(
      result.bins.map((bin: {key: string}) => bin.key)
    );
    expect(ordered.map(value => ordinals.filter(ordinal => ordinal === value).length)).toEqual(
      result.bins.map((bin: {count: number}) => bin.count)
    );
    for (const period of periods) {
      expect(Number.isInteger(period.start / 1000)).toBe(true);
      expect(Number.isInteger(period.end / 1000)).toBe(true);
      const start = BigInt(period.start / 1000);
      const end = BigInt(period.end / 1000);
      // Native range endpoints remain inclusive; exact unit conversion preserves that policy.
      expect(secondValues.map(value => value >= start && value <= end)).toEqual(
        secondValues.map(
          (_, index) =>
            timestamps.get(index)! >= period.start && timestamps.get(index)! <= period.end
        )
      );
    }
  }
});

test('automatic interval selection mirrors native span thresholds and period labels', () => {
  const day = 86400000;
  const labelInterval = (label: string) =>
    /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/.test(label)
      ? 'hour'
      : label.startsWith('Week of')
        ? 'week'
        : /^\d{4}$/.test(label)
          ? 'year'
          : /^\d{4}-\d{2}$/.test(label)
            ? 'month'
            : 'day';
  const start = Date.UTC(2020, 0, 1);
  for (const span of [
    0,
    3600000,
    199 * 3600000,
    200 * 3600000,
    200 * 3600000 + 1,
    199 * day,
    200 * day,
    201 * day,
    200 * 7 * day,
    201 * 7 * day,
    200 * 30 * day,
    201 * 30 * day,
    200 * 365 * day,
    201 * 365 * day
  ]) {
    const chart = native.createLineChart({
      id: 'auto',
      dataId: 'source',
      xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin'}
    });
    const result = native.computeDatasetChart(chart, {
      ...dataset,
      allIndexes: [0, 1],
      filteredIndex: [0, 1],
      getValue: (_name: string, index: number) => (index ? start + span : start)
    });
    expect(labelInterval(result.bins[0].key), `span ${span}`).toBe(chooseNativeTimeInterval(span));
  }
});

test('literal auto interval is native UTC days with date-time labels', () => {
  const chart = native.createLineChart({
    id: 'literal',
    dataId: 'source',
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval: 'auto'}
  });
  const fields: ResidentChartField[] = [
    {kind: 'time', name: 'time', unit: 'millisecond', sourceType: 'Timestamp<MILLISECOND>'}
  ];
  expect(planResidentChart(chart, 'source', fields)).toMatchObject({
    supported: true,
    plan: {kind: 'time', interval: 'day', dateTimeLabels: true}
  });
  const result = native.computeDatasetChart(chart, dataset);
  const ordinals = [...new Set(instants.map(instant => getPeriodOrdinal(instant, 'day')))].sort(
    (first, second) => first - second
  );
  const periods = makeResidentTimePeriods(
    ordinals.map(value => (value ^ 0x80000000) >>> 0),
    'day'
  ).map(period => `${period.label} 00:00`);
  expect(result.bins.map((bin: {key: string}) => bin.key)).toEqual(periods);
});

test('time bars match native equal-width bins, labels and exclusive upper edges', () => {
  for (const [count, spanMilliseconds] of [
    [2, 1],
    [10, 86400000 * 3 + 12345],
    [10, 1000],
    [7, 86400000 * 400],
    [10, 5]
  ] as const) {
    const start = Date.UTC(2020, 5, 17, 13, 7, 9);
    const times = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23].map(
      step => start + Math.floor((spanMilliseconds * step) / 23)
    );
    const chart = native.createBarChart({
      id: 'bars',
      dataId: 'source',
      numGroups: count,
      xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin'}
    });
    const result = native.computeDatasetChart(chart, {
      ...dataset,
      allIndexes: times.map((_, index) => index),
      filteredIndex: times.map((_, index) => index),
      getValue: (_name: string, index: number) => times[index]
    });
    const layout = makeResidentTimeBarPeriods(Math.min(...times), Math.max(...times), count);
    const edges = layout.thresholds.map(value => Math.ceil(value));
    const rawCounts = new Array(layout.thresholds.length + 1).fill(0);
    for (const time of times) rawCounts[edges.filter(edge => edge <= time).length]++;
    expect(result.bins.map((bin: {key: string}) => bin.key)).toEqual(
      layout.periods.map(period => period.label)
    );
    expect(result.bins.map((bin: {count: number}) => bin.count)).toEqual(
      layout.sourceBins.map(index => rawCounts[index])
    );
    expect(result.bins.map((bin: {filterValue: number[]}) => bin.filterValue)).toEqual(
      layout.periods.map(period => [period.start, period.end])
    );
  }
});

test('time bars plan from native group counts and reject calendar-period bars', () => {
  const fields: ResidentChartField[] = [
    {kind: 'time', name: 'time', unit: 'nanosecond', sourceType: 'Timestamp<NANOSECOND>'},
    {name: 'amount', format: 'float32'}
  ];
  const bar = (extra: object, type = 'createBarChart') =>
    native[type]({
      id: 'bars',
      dataId: 'source',
      xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin'},
      yAxis: {field: {name: 'amount', type: 'real'}, aggregation: 'sum'},
      ...extra
    });
  expect(planResidentChart(bar({}), 'source', fields)).toMatchObject({
    supported: true,
    plan: {kind: 'time', interval: 'equal-width', binCount: 10, operation: 'sum'}
  });
  expect(planResidentChart(bar({numGroups: 3}), 'source', fields)).toMatchObject({
    supported: true,
    plan: {binCount: 3}
  });
  // A non-positive bar group count selects native calendar periods, like a line.
  expect(planResidentChart(bar({numGroups: 0}), 'source', fields)).toMatchObject({
    supported: true,
    plan: {kind: 'time', interval: 'auto'}
  });
  expect(planResidentChart(bar({numGroups: -3}), 'source', fields)).toMatchObject({
    supported: true,
    plan: {interval: 'auto'}
  });
  expect(planResidentChart(bar({numGroups: 513}), 'source', fields)).toMatchObject({
    supported: true,
    plan: {binCount: 513}
  });
  expect(planResidentChart(bar({numGroups: 1000}), 'source', fields)).toMatchObject({
    supported: true,
    plan: {binCount: 1000}
  });
  expect(planResidentChart(bar({numGroups: 1001}), 'source', fields)).toMatchObject({
    supported: false
  });
});
