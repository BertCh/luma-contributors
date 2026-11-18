// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import type {ArrowAnalyticsTimeUnit} from '../../../src/analysis-core/graph-support/arrow-analytics-normalization';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUCalendarBucketsParameterValues,
  GPUCalendarBuckets
} from '../../../src/dataframe-analysis/calendar-buckets/index';
import {
  GPUPeriodIndex,
  GPU_PERIOD_INDEX_INVALID,
  GPU_PERIOD_INDEX_OUT_OF_RANGE,
  GPU_PERIOD_INDEX_MAXIMUM_COUNT,
  getGPUPeriodIndexCalendarPeriod,
  type GPUPeriodIndexCalendarPeriods,
  type GPUPeriodIndexFixedPeriods
} from '../../../src/dataframe-analysis/period-index/index';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/index';
import {
  formatTimestampTicks,
  getTimestampCalendarMilliseconds
} from '../../../src/dataframe-analysis/temporal-reduction/index';
import {
  createVectorView,
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createSeededRandom,
  getEpochMilliseconds,
  getExpectedCalendarPeriod,
  getExpectedFixedPeriod
} from './period-index-oracle';

const INT64_MINIMUM = -(2n ** 63n);
const INT64_MAXIMUM = 2n ** 63n - 1n;
const TICKS_PER_MILLISECOND = {
  second: 1n,
  millisecond: 1n,
  microsecond: 1000n,
  nanosecond: 1_000_000n
};

it('indexes hours across signed boundaries and the full unsigned period count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const hourIndices = [-0x80000001, -1, 0, 0x7ffffffc, 0x7ffffffd, 0x7fffffff, 0x80000000];
  for (const originHour of [0, -24, -0x80000000]) {
    const instant = new Date(originHour * 3_600_000);
    const origin = {
      year: instant.getUTCFullYear(),
      month: instant.getUTCMonth() + 1,
      day: instant.getUTCDate(),
      hour: instant.getUTCHours()
    };
    for (const interval of [1, 3]) {
      const step = {calendarUnit: 'hour', interval} as const;
      const expected = hourIndices.map(hourIndex => {
        const difference = hourIndex - originHour;
        const candidate = Math.floor(difference / interval);
        return difference >= 0 && candidate < GPU_PERIOD_INDEX_MAXIMUM_COUNT
          ? candidate
          : GPU_PERIOD_INDEX_OUT_OF_RANGE;
      });
      const cpuPeriods = hourIndices.map(hourIndex => {
        const date = new Date(hourIndex * 3_600_000);
        const period = getGPUPeriodIndexCalendarPeriod(
          step,
          origin,
          GPU_PERIOD_INDEX_MAXIMUM_COUNT,
          {
            year: date.getUTCFullYear(),
            month: date.getUTCMonth() + 1,
            day: date.getUTCDate(),
            hour: date.getUTCHours()
          }
        );
        return period < 0 ? GPU_PERIOD_INDEX_OUT_OF_RANGE : period;
      });
      expect(cpuPeriods).toEqual(expected);
      for (const chunkLengths of [undefined, [1, 3, 3]]) {
        const result = await runCalendar(
          device,
          hourIndices.map(hourIndex => BigInt(hourIndex) * 3_600_000n),
          'millisecond',
          {kind: 'calendar', step, origin, count: GPU_PERIOD_INDEX_MAXIMUM_COUNT},
          {chunkLengths, withStatus: true}
        );
        expect(result.periods).toEqual(expected);
        expect(result.invalidCount).toBe(
          expected.filter(period => period === GPU_PERIOD_INDEX_OUT_OF_RANGE).length
        );
      }
    }
  }
});

it('keeps pre-epoch fractional ticks on the same calendar day in display and bucketing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const ticks = [-1n, 0n, 1n];
  for (const [unit, arrowUnit] of [
    ['microsecond', 2],
    ['nanosecond', 3]
  ] as const) {
    const result = await runCalendar(
      device,
      ticks,
      unit,
      {
        kind: 'calendar',
        step: {calendarUnit: 'day'},
        origin: {year: 1969, month: 12, day: 31},
        count: 2
      },
      {withStatus: true}
    );
    expect(result.periods).toEqual([0, 1, 1]);
    expect(result.invalidCount).toBe(0);
    expect(
      ticks.map(
        tick => Math.floor(getTimestampCalendarMilliseconds(tick, arrowUnit) / 86_400_000) + 1
      )
    ).toEqual(result.periods);
    expect(ticks.map(tick => formatTimestampTicks(tick, arrowUnit).slice(0, 10))).toEqual([
      '1969-12-31',
      '1970-01-01',
      '1970-01-01'
    ]);
  }
});

it('compiles maximum signed calendar divisors for packed and chunked rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const ticks = [0n, 86_400_000n, 172_800_000n];
  for (const [calendarUnit, interval] of [
    ['year', 0x7fffffff],
    ['week', 306_783_378]
  ] as const) {
    for (const chunkLengths of [undefined, [1, 1, 1]]) {
      const result = await runCalendar(
        device,
        ticks,
        'millisecond',
        {
          kind: 'calendar',
          step: {calendarUnit, interval},
          origin: {year: 1970},
          count: 1
        },
        {chunkLengths, withStatus: true}
      );
      // All three dates are inside the first period, even with the largest supported divisor.
      expect(result.periods).toEqual([0, 0, 0]);
      expect(result.invalidCount).toBe(0);
    }
  }
});

it('compiles both signed hour-origin literal boundaries with independent periods', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const hourIndex of [-0x80000000, 0x7fffffff]) {
    const milliseconds = hourIndex * 3_600_000;
    const instant = new Date(milliseconds);
    const offsets = hourIndex < 0 ? [0, 1, 2] : [-2, -1, 0];
    const expected =
      hourIndex < 0 ? [0, 1, 2] : [GPU_PERIOD_INDEX_OUT_OF_RANGE, GPU_PERIOD_INDEX_OUT_OF_RANGE, 0];
    for (const chunkLengths of [undefined, [1, 1, 1]]) {
      const result = await runCalendar(
        device,
        offsets.map(offset => BigInt(milliseconds + offset * 3_600_000)),
        'millisecond',
        {
          kind: 'calendar',
          step: {calendarUnit: 'hour'},
          count: 3,
          origin: {
            year: instant.getUTCFullYear(),
            month: instant.getUTCMonth() + 1,
            day: instant.getUTCDate(),
            hour: instant.getUTCHours()
          }
        },
        {chunkLengths, withStatus: true}
      );
      expect(result.periods).toEqual(expected);
      expect(result.invalidCount).toBe(hourIndex < 0 ? 0 : 2);
    }
  }
});

function getContext(unit: ArrowAnalyticsTimeUnit): GPUTemporalContext {
  return {unit, epoch: 'unix', encoding: 'int64-words', axis: {kind: 'irregular'}};
}

/** Ticks of `unit` at an epoch millisecond (seconds are truncated to a whole second). */
function toTicks(milliseconds: number, unit: ArrowAnalyticsTimeUnit): bigint {
  return unit === 'second'
    ? BigInt(Math.floor(milliseconds / 1000))
    : BigInt(milliseconds) * TICKS_PER_MILLISECOND[unit];
}

type RunOptions = {
  mask?: number[];
  withStatus?: boolean;
  chunkLengths?: readonly number[];
  repeat?: number;
};

type RunResult = {periods: number[]; invalidCount?: number};

function splitRows<Format extends 'uint32' | 'sint32' | 'uint32x2'>(
  graph: GPUCommandGraph,
  view: GraphDataView<Format>,
  lengths?: readonly number[]
) {
  if (!lengths) return view;
  let rowBase = 0;
  return createVectorView(
    view.buffer.id,
    view.format,
    lengths.map(length => {
      const chunk = graph.createDataView(view.buffer, {
        format: view.format,
        byteOffset: rowBase * view.rowByteLength,
        length
      });
      rowBase += length;
      return chunk;
    })
  );
}

async function runCalendar(
  device: Device,
  ticks: readonly bigint[],
  unit: ArrowAnalyticsTimeUnit,
  periods: GPUPeriodIndexCalendarPeriods,
  options: RunOptions = {}
): Promise<RunResult> {
  const rowCount = ticks.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer): Buffer => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'period-index-calendar'});
  const parameters = new GPUParameterBuffer(device, {
    id: 'period-index-calendar-parameters',
    format: 'sint32',
    length: 2
  });
  const timestamps = track(createInputBuffer(device, getInt64TimeWords(BigInt64Array.from(ticks))));
  const year = track(createOutputBuffer(device, rowCount));
  const month = track(createOutputBuffer(device, rowCount));
  const dayOfMonth = track(createOutputBuffer(device, rowCount));
  const hour = track(createOutputBuffer(device, rowCount));
  const periodsBuffer = track(createOutputBuffer(device, rowCount));
  const invalidCount = track(createOutputBuffer(device, 1));
  const mask = options.mask
    ? track(createInputBuffer(device, Uint32Array.from(options.mask)))
    : null;
  const temporalContext = getContext(unit);
  const yearView = splitRows(
    graph,
    importGraphBuffer(graph, 'year', year, 'sint32', rowCount),
    options.chunkLengths
  );
  const monthView = splitRows(
    graph,
    importGraphBuffer(graph, 'month', month, 'uint32', rowCount),
    options.chunkLengths
  );
  const dayView = splitRows(
    graph,
    importGraphBuffer(graph, 'day', dayOfMonth, 'uint32', rowCount),
    options.chunkLengths
  );
  const hourView = splitRows(
    graph,
    importGraphBuffer(graph, 'hour', hour, 'uint32', rowCount),
    options.chunkLengths
  );
  graph.add(
    new GPUCalendarBuckets({
      id: 'calendar',
      temporalContext,
      timestamps: splitRows(
        graph,
        importGraphBuffer(graph, 'times', timestamps, 'uint32x2', rowCount),
        options.chunkLengths
      ),
      parameters: parameters.importToGraph(graph),
      output: {year: yearView, month: monthView, dayOfMonth: dayView, hour: hourView}
    })
  );
  graph.add(
    new GPUPeriodIndex({
      id: 'period-index',
      temporalContext,
      periods,
      columns: {year: yearView, month: monthView, dayOfMonth: dayView, hour: hourView},
      mask: mask
        ? splitRows(
            graph,
            importGraphBuffer(graph, 'mask', mask, 'uint32', rowCount),
            options.chunkLengths
          )
        : undefined,
      output: {
        periods: splitRows(
          graph,
          importGraphBuffer(graph, 'periods', periodsBuffer, 'uint32', rowCount),
          options.chunkLengths
        ),
        ...(options.withStatus
          ? {status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidCount, 'uint32', 1)}}
          : {})
      }
    })
  );
  const compiled = graph.compile();
  parameters.write(getGPUCalendarBucketsParameterValues(0, 0));
  for (let encoding = 0; encoding < (options.repeat ?? 1); encoding++)
    submitGraph(device, compiled, undefined);
  const result: RunResult = {periods: await readUint32(periodsBuffer, rowCount)};
  if (options.withStatus) {
    result.invalidCount = (await readUint32(invalidCount, 1))[0];
  }
  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

async function runFixed(
  device: Device,
  ticks: readonly bigint[],
  unit: ArrowAnalyticsTimeUnit,
  periods: GPUPeriodIndexFixedPeriods,
  options: RunOptions = {}
): Promise<RunResult> {
  const rowCount = ticks.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer): Buffer => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'period-index-fixed'});
  const timestamps = track(createInputBuffer(device, getInt64TimeWords(BigInt64Array.from(ticks))));
  const periodsBuffer = track(createOutputBuffer(device, rowCount));
  const invalidCount = track(createOutputBuffer(device, 1));
  const mask = options.mask
    ? track(createInputBuffer(device, Uint32Array.from(options.mask)))
    : null;
  graph.add(
    new GPUPeriodIndex({
      id: 'period-index',
      temporalContext: getContext(unit),
      periods,
      timestamps: splitRows(
        graph,
        importGraphBuffer(graph, 'times', timestamps, 'uint32x2', rowCount),
        options.chunkLengths
      ),
      mask: mask
        ? splitRows(
            graph,
            importGraphBuffer(graph, 'mask', mask, 'uint32', rowCount),
            options.chunkLengths
          )
        : undefined,
      output: {
        periods: splitRows(
          graph,
          importGraphBuffer(graph, 'periods', periodsBuffer, 'uint32', rowCount),
          options.chunkLengths
        ),
        ...(options.withStatus
          ? {status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidCount, 'uint32', 1)}}
          : {})
      }
    })
  );
  const compiled = graph.compile();
  for (let encoding = 0; encoding < (options.repeat ?? 1); encoding++)
    submitGraph(device, compiled, undefined);
  const result: RunResult = {periods: await readUint32(periodsBuffer, rowCount)};
  if (options.withStatus) {
    result.invalidCount = (await readUint32(invalidCount, 1))[0];
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** GPU periods as the oracle's convention: `-1` for out of range, `-2` for invalid. */
function toSigned(periods: readonly number[]): number[] {
  return periods.map(period =>
    period === GPU_PERIOD_INDEX_OUT_OF_RANGE
      ? -1
      : period === GPU_PERIOD_INDEX_INVALID
        ? -2
        : period
  );
}

const BOUNDARY_TIMES = [
  // Leap day and its neighbors.
  Date.UTC(2024, 1, 28, 23, 59, 59, 999),
  Date.UTC(2024, 1, 29, 0, 0, 0, 0),
  Date.UTC(2024, 1, 29, 23, 59, 59, 999),
  Date.UTC(2024, 2, 1, 0, 0, 0, 0),
  Date.UTC(2023, 1, 28, 23, 59, 59, 999),
  Date.UTC(2023, 2, 1, 0, 0, 0, 0),
  Date.UTC(2000, 1, 29, 12),
  Date.UTC(1900, 2, 1),
  // Year and quarter starts.
  Date.UTC(2023, 11, 31, 23, 59, 59, 999),
  Date.UTC(2024, 0, 1, 0, 0, 0, 0),
  Date.UTC(2024, 2, 31, 23, 59, 59, 999),
  Date.UTC(2024, 3, 1, 0, 0, 0, 0),
  Date.UTC(2024, 5, 30, 23, 59, 59, 999),
  Date.UTC(2024, 6, 1, 0, 0, 0, 0),
  Date.UTC(2024, 8, 30, 23, 59, 59, 999),
  Date.UTC(2024, 9, 1, 0, 0, 0, 0),
  // Week starts relative to a Wednesday origin (2023-03-15) and a Monday one (2024-01-01).
  Date.UTC(2023, 2, 21, 23, 59, 59, 999),
  Date.UTC(2023, 2, 22, 0, 0, 0, 0),
  Date.UTC(2024, 0, 7, 23, 59, 59, 999),
  Date.UTC(2024, 0, 8, 0, 0, 0, 0),
  // Before 1970.
  Date.UTC(1969, 11, 31, 23, 59, 59, 999),
  Date.UTC(1970, 0, 1, 0, 0, 0, 0),
  Date.UTC(1960, 1, 29, 12),
  Date.UTC(1900, 0, 1),
  Date.UTC(1899, 11, 31, 23, 59, 59, 999),
  // Around the origins.
  Date.UTC(2023, 2, 14, 23),
  Date.UTC(2023, 2, 15, 0),
  Date.UTC(1969, 11, 30, 23)
];

const CALENDAR_CASES: {
  name: string;
  step: GPUPeriodIndexCalendarPeriods['step'];
  origin: GPUPeriodIndexCalendarPeriods['origin'];
  count: number;
}[] = [
  {name: 'year', step: {calendarUnit: 'year'}, origin: {year: 1899}, count: 200},
  {
    name: 'year interval 5, short',
    step: {calendarUnit: 'year', interval: 5},
    origin: {year: 1960},
    count: 8
  },
  {name: 'quarter', step: {calendarUnit: 'quarter'}, origin: {year: 1969, month: 10}, count: 400},
  {
    name: 'quarter from a mid-quarter origin',
    step: {calendarUnit: 'quarter', interval: 2},
    origin: {year: 2023, month: 5, day: 20},
    count: 6
  },
  {name: 'month', step: {calendarUnit: 'month'}, origin: {year: 1900, month: 1}, count: 2000},
  {
    name: 'month interval 3, short',
    step: {calendarUnit: 'month', interval: 3},
    origin: {year: 2023, month: 11},
    count: 4
  },
  {
    name: 'week from a Wednesday',
    step: {calendarUnit: 'week'},
    origin: {year: 2023, month: 3, day: 15},
    count: 120
  },
  {
    name: 'week from a Monday, interval 2',
    step: {calendarUnit: 'week', interval: 2},
    origin: {year: 2024, month: 1, day: 1},
    count: 60
  },
  {
    name: 'day from before 1970',
    step: {calendarUnit: 'day'},
    origin: {year: 1969, month: 12, day: 30},
    count: 70_000
  },
  {
    name: 'day interval 10',
    step: {calendarUnit: 'day', interval: 10},
    origin: {year: 2023, month: 12, day: 1},
    count: 20
  },
  {
    name: 'hour from before 1970',
    step: {calendarUnit: 'hour'},
    origin: {year: 1969, month: 12, day: 30, hour: 22},
    count: 100_000
  },
  {
    name: 'hour interval 6',
    step: {calendarUnit: 'hour', interval: 6},
    origin: {year: 2024, month: 2, day: 28, hour: 3},
    count: 40
  }
];

describe('GPUPeriodIndex calendar periods', () => {
  for (const calendarCase of CALENDAR_CASES) {
    it(`matches Date arithmetic at calendar boundaries: ${calendarCase.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const random = createSeededRandom(calendarCase.name.length * 7919);
      const times = [...BOUNDARY_TIMES];
      // Random times from 1850 to 2100, so rows fall before, inside and past the periods.
      for (let row = 0; row < 150; row++) {
        times.push(
          Math.floor(
            Date.UTC(1850, 0, 1) + random() * (Date.UTC(2100, 0, 1) - Date.UTC(1850, 0, 1))
          )
        );
      }
      const periods: GPUPeriodIndexCalendarPeriods = {kind: 'calendar', ...calendarCase};
      const result = await runCalendar(
        device,
        times.map(time => toTicks(time, 'millisecond')),
        'millisecond',
        periods,
        {withStatus: true}
      );
      const expected = times.map(time =>
        getExpectedCalendarPeriod(calendarCase.step, calendarCase.origin, calendarCase.count, time)
      );
      expect(toSigned(result.periods)).toEqual(expected);
      expect(
        expected.some(period => period >= 0),
        'some rows in range'
      ).toBe(true);
      expect(result.invalidCount).toBe(expected.filter(period => period === -1).length);
    });
  }

  for (const unit of ['second', 'microsecond', 'nanosecond'] as const) {
    it(`takes ${unit} words through the calendar decode`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Times one tick either side of leap-day, quarter and year starts, plus a pre-1970 instant.
      const boundaries = [
        Date.UTC(2024, 1, 29),
        Date.UTC(2024, 2, 1),
        Date.UTC(2024, 3, 1),
        Date.UTC(2024, 0, 1),
        Date.UTC(1969, 11, 31, 23, 0, 0)
      ];
      const ticks = boundaries.flatMap(time => {
        const base = toTicks(time, unit);
        return [base - 1n, base, base + 1n];
      });
      for (const calendarCase of CALENDAR_CASES.filter(({name}) =>
        /^(quarter$|month$|day|hour$)/.test(name)
      )) {
        const result = await runCalendar(device, ticks, unit, {kind: 'calendar', ...calendarCase});
        const expected = ticks.map(tick =>
          getExpectedCalendarPeriod(
            calendarCase.step,
            calendarCase.origin,
            calendarCase.count,
            getEpochMilliseconds(tick, unit)
          )
        );
        expect(toSigned(result.periods), `${unit} ${calendarCase.name}`).toEqual(expected);
      }
    });
  }

  it('marks masked and undecodable rows invalid and does not count them', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const times = [
      toTicks(Date.UTC(2023, 5, 15), 'millisecond'),
      toTicks(Date.UTC(2023, 5, 16), 'millisecond'),
      INT64_MAXIMUM,
      INT64_MINIMUM,
      toTicks(Date.UTC(2020, 0, 1), 'millisecond'),
      toTicks(Date.UTC(2023, 7, 1), 'millisecond')
    ];
    const result = await runCalendar(
      device,
      times,
      'millisecond',
      {kind: 'calendar', step: {calendarUnit: 'month'}, origin: {year: 2023, month: 1}, count: 12},
      {mask: [1, 0, 1, 1, 1, 1], withStatus: true}
    );
    expect(result.periods).toEqual([
      5,
      GPU_PERIOD_INDEX_INVALID,
      GPU_PERIOD_INDEX_INVALID,
      GPU_PERIOD_INDEX_INVALID,
      GPU_PERIOD_INDEX_OUT_OF_RANGE,
      7
    ]);
    expect(result.invalidCount).toBe(1);
  });
});

describe('GPUPeriodIndex fixed periods', () => {
  const HOUR = 3_600_000n;

  it('divides the word difference by an hour width, before and after 1970', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const origin = BigInt(Date.UTC(1969, 11, 31, 12));
    const random = createSeededRandom(5);
    const ticks = [
      origin - 1n,
      origin,
      origin + HOUR - 1n,
      origin + HOUR,
      origin + 24n * HOUR - 1n,
      origin + 24n * HOUR,
      origin + 25n * HOUR,
      BigInt(Date.UTC(1970, 0, 1)),
      INT64_MINIMUM,
      INT64_MAXIMUM
    ];
    for (let row = 0; row < 200; row++) {
      ticks.push(origin - 10n * HOUR + BigInt(Math.floor(random() * 48 * Number(HOUR))));
    }
    const periods: GPUPeriodIndexFixedPeriods = {kind: 'fixed', width: HOUR, origin, count: 24};
    const result = await runFixed(device, ticks, 'millisecond', periods, {withStatus: true});
    const expected = ticks.map(tick => getExpectedFixedPeriod(tick, HOUR, origin, 24));
    expect(toSigned(result.periods)).toEqual(expected);
    expect(expected.filter(period => period >= 0).length).toBeGreaterThan(20);
    expect(result.invalidCount).toBe(expected.filter(period => period === -1).length);
    // A numeric width is the same as the bigint.
    const numeric = await runFixed(device, ticks, 'millisecond', {...periods, width: 3_600_000});
    expect(numeric.periods).toEqual(result.periods);
  });

  it('is exact for nanosecond words and widths beyond 32 bits', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const origin = 1_700_000_000_000_000_000n;
    const random = createSeededRandom(11);
    const cases: [bigint, number][] = [
      [86_400_000_000_000n, 40],
      [2n ** 32n, 1000],
      [2n ** 32n + 3n, 1000],
      [2n ** 40n + 12345n, 1000],
      [2n ** 62n, 6],
      [1n, 4_000_000_000],
      [7n, 4_000_000_000]
    ];
    for (const [width, count] of cases) {
      const ticks: bigint[] = [
        origin - 1n,
        origin,
        origin + width - 1n,
        origin + width,
        origin + 2n * width
      ];
      ticks.push(origin + width * BigInt(count) - 1n, origin + width * BigInt(count));
      for (let row = 0; row < 100; row++) {
        const fraction = BigInt(Math.floor(random() * 1_000_000));
        ticks.push(origin - width + (width * BigInt(count + 2) * fraction) / 1_000_000n);
      }
      const clamped = ticks.map(tick => (tick > INT64_MAXIMUM ? INT64_MAXIMUM : tick));
      const result = await runFixed(device, clamped, 'nanosecond', {
        kind: 'fixed',
        width,
        origin,
        count
      });
      const expected = clamped.map(tick => getExpectedFixedPeriod(tick, width, origin, count));
      expect(toSigned(result.periods), `width ${width}`).toEqual(expected);
    }
  });

  it('reports a quotient beyond 32 bits as out of range and honors the mask', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runFixed(
      device,
      [INT64_MAXIMUM, 5n, 10n, 3n],
      'millisecond',
      {kind: 'fixed', width: 1n, origin: 0n, count: 100},
      {mask: [1, 1, 0, 1], withStatus: true}
    );
    expect(result.periods).toEqual([GPU_PERIOD_INDEX_OUT_OF_RANGE, 5, GPU_PERIOD_INDEX_INVALID, 3]);
    expect(result.invalidCount).toBe(1);
  });
});

it('GPUPeriodIndex calendar chunks match packed input and the Date oracle for every step', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const milliseconds = [
    Date.UTC(2023, 0, 1),
    Date.UTC(2024, 1, 29),
    Date.UTC(2024, 2, 1, 12),
    Date.UTC(1969, 11, 31),
    Date.UTC(2024, 11, 31),
    Date.UTC(2030, 0, 1),
    Date.UTC(2024, 0, 1),
    Date.UTC(2023, 11, 31, 23),
    Date.UTC(2025, 0, 1)
  ];
  const ticks = milliseconds.map(BigInt);
  const mask = [1, 1, 0, 1, 1, 1, 1, 1, 1];
  for (const calendarUnit of ['year', 'quarter', 'month', 'week', 'day', 'hour'] as const) {
    const periods: GPUPeriodIndexCalendarPeriods = {
      kind: 'calendar',
      step: {calendarUnit, interval: 2},
      origin: {year: 2023},
      count: 8
    };
    const packed = await runCalendar(device, ticks, 'millisecond', periods, {
      mask,
      withStatus: true
    });
    const chunked = await runCalendar(device, ticks, 'millisecond', periods, {
      mask,
      withStatus: true,
      chunkLengths: [1, 3, 5],
      repeat: 2
    });
    const expected = milliseconds.map((time, index) =>
      mask[index] === 0
        ? -2
        : getExpectedCalendarPeriod(periods.step, periods.origin, periods.count, time)
    );
    expect(chunked).toEqual(packed);
    expect(toSigned(chunked.periods)).toEqual(expected);
    expect(chunked.invalidCount).toBe(expected.filter(period => period === -1).length);
  }
});

it('GPUPeriodIndex fixed chunks match packed and exact BigInt division including sentinels', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const periods: GPUPeriodIndexFixedPeriods = {kind: 'fixed', origin: -10n, width: 3n, count: 4};
  const ticks = [INT64_MINIMUM, -11n, -10n, -9n, -7n, -1n, 1n, 2n, INT64_MAXIMUM];
  const mask = [1, 1, 1, 0, 1, 1, 1, 1, 1];
  const expected = ticks.map((time, index) =>
    mask[index] === 0 ? -2 : time < -10n || time >= 2n ? -1 : Number((time + 10n) / 3n)
  );
  const packed = await runFixed(device, ticks, 'nanosecond', periods, {mask, withStatus: true});
  const chunked = await runFixed(device, ticks, 'nanosecond', periods, {
    mask,
    withStatus: true,
    chunkLengths: [1, 3, 5],
    repeat: 2
  });
  expect(chunked).toEqual(packed);
  expect(toSigned(chunked.periods)).toEqual(expected);
  expect(chunked.invalidCount).toBe(expected.filter(period => period === -1).length);
});
