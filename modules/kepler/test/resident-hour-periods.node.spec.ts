// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {makeResidentHourPeriods} from '../../../examples/deck/kepler-compute/src/resident-hour-periods';

const native = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
)('@kepler.gl/charts');
const encodeDay = (day: number) => (day ^ 0x80000000) >>> 0;

function makeHourDomain(instants: readonly number[]) {
  const days = [...new Set(instants.map(instant => Math.floor(instant / 86400000)))].sort(
    (first, second) => first - second
  );
  const hourKeys = [
    ...new Set(
      instants.map(
        instant =>
          days.indexOf(Math.floor(instant / 86400000)) * 24 + new Date(instant).getUTCHours()
      )
    )
  ].sort((first, second) => first - second);
  return {hourKeys, dayKeys: days.map(encodeDay)};
}

test('bounded hour periods match native UTC labels and endpoints across the full Date range', () => {
  const values = [
    Date.parse('2024-03-10T03:00:00-04:00'),
    -8640000000000000,
    8640000000000000 - 1,
    -3600001,
    -3600000,
    -3599999,
    -1,
    0,
    1,
    3599999,
    3600000,
    3600001,
    Date.parse('0000-02-29T23:59:59.999Z'),
    Date.parse('0099-12-31T23:59:59.999Z'),
    Date.parse('-000001-12-31T23:59:59.999Z'),
    Date.parse('2024-02-29T23:59:59.999Z'),
    Date.parse('2024-03-01T00:00:00.000Z'),
    Date.parse('2024-03-10T01:59:59-05:00'),
    Date.parse('2024-11-03T01:30:00-04:00'),
    Date.parse('2024-11-03T01:30:00-05:00'),
    null,
    NaN
  ];
  const instants = values.flatMap(value => (Number.isNaN(value) ? [] : [value ?? 0]));
  const {hourKeys, dayKeys} = makeHourDomain(instants);
  const periods = makeResidentHourPeriods(hourKeys, dayKeys);
  const chart = native.createLineChart({
    id: 'hour',
    dataId: 'source',
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval: 'hour'}
  });
  const result = native.computeDatasetChart(chart, {
    id: 'source',
    color: [1, 2, 3],
    allIndexes: values.map((_, index) => index),
    filteredIndex: values.map((_, index) => index),
    fields: [{name: 'time', type: 'timestamp'}],
    getValue: (_name: string, index: number) => values[index]
  });
  expect(periods).toEqual(
    result.bins.map((bin: {key: string; filterValue: number[]}) => ({
      label: bin.key,
      start: bin.filterValue[0],
      end: bin.filterValue[1]
    }))
  );
  expect(periods[0].start / 3600000).toBe(-2400000000);
  expect(periods.at(-1)!.end / 3600000).toBe(2400000000);
  for (const period of periods) {
    expect(Number.isSafeInteger(period.start)).toBe(true);
    expect(period.end - period.start).toBe(3600000);
  }
  expect(periods.find(period => period.start === -3600000)).toEqual({
    label: '1969-12-31 23:00',
    start: -3600000,
    end: 0
  });
  expect(result.bins.find((bin: {key: string}) => bin.key === '1970-01-01 00:00').count).toBe(4);
});

test('hour keys preserve the complete bounded dense-day domain without interpreting it as epoch hours', () => {
  const dayKeys = Array.from({length: 512}, (_, index) => encodeDay(index - 256));
  const periods = makeResidentHourPeriods([0, 23, 24, 12287], dayKeys);
  expect(periods.map(period => period.start)).toEqual([
    -256 * 86400000,
    -256 * 86400000 + 23 * 3600000,
    -255 * 86400000,
    255 * 86400000 + 23 * 3600000
  ]);
  expect(makeResidentHourPeriods([], [])).toEqual([]);
});

test('hour periods reject clipped Date endpoints and require a matching day domain', () => {
  expect(makeResidentHourPeriods([0], [encodeDay(-100000000)])[0].start).toBe(-8640000000000000);
  expect(makeResidentHourPeriods([23], [encodeDay(99999999)])[0].end).toBe(8640000000000000);
  expect(() => makeResidentHourPeriods([0], [encodeDay(100000000)])).toThrow('Date boundary');
  expect(() => makeResidentHourPeriods([23], [encodeDay(-100000001)])).toThrow('Date boundary');
  expect(() => makeResidentHourPeriods([24], [encodeDay(0)])).toThrow();
  expect(() => makeResidentHourPeriods([-1], [encodeDay(0)])).toThrow();
});
