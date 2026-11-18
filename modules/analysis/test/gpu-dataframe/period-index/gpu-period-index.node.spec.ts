// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {
  getGPUPeriodIndexCalendarPeriod,
  getGPUPeriodIndexDaysFromCivil,
  getGPUPeriodIndexStepIndex,
  GPUPeriodIndex,
  GPU_PERIOD_INDEX_MAXIMUM_COUNT,
  GPU_PERIOD_INDEX_OUT_OF_RANGE,
  type GPUPeriodIndexProps
} from '../../../src/dataframe-analysis/period-index/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getExpectedCalendarPeriod} from './period-index-oracle';

const CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};
const ROWS = 6;
let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUPeriodIndexProps> = {}
): GPUPeriodIndexProps {
  const view = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(format: Format, length = ROWS) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    temporalContext: CONTEXT,
    periods: {kind: 'calendar', step: {calendarUnit: 'month'}, origin: {year: 2023}, count: 12},
    columns: {year: view('sint32'), month: view('uint32')},
    output: {periods: view('uint32')},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUPeriodIndexProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUPeriodIndex(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const viewOf = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length = ROWS
) => createTransientView(graph, `v-${serial++}`, format, length);

describe('GPUPeriodIndex validation', () => {
  it('accepts a calendar and a fixed axis and reports the row count', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    expect(new GPUPeriodIndex(createProps(graph)).rowCount).toBe(ROWS);
    const fixed = new GPUPeriodIndex(
      createProps(graph, {
        periods: {kind: 'fixed', width: 3_600_000, origin: -5n, count: 24},
        columns: undefined,
        timestamps: viewOf(graph, 'uint32x2', 4)
      })
    );
    expect(fixed.rowCount).toBe(4);
    device.destroy();
  });

  it('resolves the temporal context with Int64 words from the Unix epoch', () => {
    expectThrows(
      () => ({temporalContext: {...CONTEXT, encoding: 'float32-relative', epoch: 1000n}}),
      /int64-words/
    );
    expectThrows(() => ({temporalContext: {...CONTEXT, encoding: 'double-single'}}), /int64-words/);
    expectThrows(() => ({temporalContext: {...CONTEXT, epoch: 1000n}}), /Unix epoch/);
    expectThrows(() => ({temporalContext: {...CONTEXT, axis: {kind: 'regular', step: 0}}}), /step/);
  });

  it('keeps the period count within the sentinel-free range', () => {
    for (const count of [0, 1.5, GPU_PERIOD_INDEX_MAXIMUM_COUNT + 1]) {
      expectThrows(
        () => ({
          periods: {kind: 'calendar', step: {calendarUnit: 'year'}, origin: {year: 2000}, count}
        }),
        /periods.count must be an integer in \[1, 4294967293\]/
      );
    }
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    expect(
      () =>
        new GPUPeriodIndex(
          createProps(graph, {
            periods: {
              kind: 'calendar',
              step: {calendarUnit: 'month'},
              origin: {year: 2000},
              count: GPU_PERIOD_INDEX_MAXIMUM_COUNT
            }
          })
        )
    ).not.toThrow();
    device.destroy();
  });

  it('requires the columns its calendar step reads', () => {
    expectThrows(() => ({columns: undefined}), /calendar periods need columns/);
    expectThrows(
      graph => ({columns: {year: viewOf(graph, 'sint32')}}),
      /columns.month is required for a month step/
    );
    expectThrows(
      graph => ({
        periods: {kind: 'calendar', step: {calendarUnit: 'day'}, origin: {year: 2023}, count: 3},
        columns: {year: viewOf(graph, 'sint32'), month: viewOf(graph, 'uint32')}
      }),
      /columns.dayOfMonth is required for a day step/
    );
    expectThrows(
      graph => ({
        periods: {kind: 'calendar', step: {calendarUnit: 'hour'}, origin: {year: 2023}, count: 3},
        columns: {
          year: viewOf(graph, 'sint32'),
          month: viewOf(graph, 'uint32'),
          dayOfMonth: viewOf(graph, 'uint32')
        }
      }),
      /columns.hour is required for a hour step/
    );
    expectThrows(
      graph => ({columns: {year: viewOf(graph, 'sint32'), month: viewOf(graph, 'uint32', 5)}}),
      /columns.month must have 6 rows/
    );
    expectThrows(
      graph => ({
        columns: {year: viewOf(graph, 'uint32') as never, month: viewOf(graph, 'uint32')}
      }),
      /columns.year/
    );
  });

  it('validates the calendar step and origin', () => {
    const periods = (step: object, origin: object) =>
      ({kind: 'calendar', step, origin, count: 3}) as never;
    expectThrows(
      () => ({periods: periods({calendarUnit: 'weekday'}, {year: 2023})}),
      /calendar unit and a positive integer interval/
    );
    expectThrows(
      () => ({periods: periods({calendarUnit: 'month', interval: 0}, {year: 2023})}),
      /positive integer interval/
    );
    expectThrows(
      () => ({periods: periods({calendarUnit: 'month', interval: 1.5}, {year: 2023})}),
      /positive integer interval/
    );
    expectThrows(
      () => ({periods: periods({calendarUnit: 'month'}, {year: 2023, month: 13})}),
      /periods.origin.month must be an integer in 1..12/
    );
    expectThrows(
      () => ({periods: periods({calendarUnit: 'month'}, {year: 2023, day: 0})}),
      /periods.origin.day/
    );
    expectThrows(
      () => ({periods: periods({calendarUnit: 'month'}, {year: 2023, hour: 24})}),
      /periods.origin.hour/
    );
    expectThrows(
      () => ({periods: periods({calendarUnit: 'month'}, {year: 1.5})}),
      /periods.origin.year/
    );
  });

  it('validates the fixed width, origin and timestamps', () => {
    const fixed = (width: number | bigint, origin: bigint) =>
      ({kind: 'fixed', width, origin, count: 4}) as const;
    const withStamps = (
      graph: GPUCommandGraph,
      width: number | bigint,
      origin: bigint
    ): Partial<GPUPeriodIndexProps> => ({
      periods: fixed(width, origin),
      columns: undefined,
      timestamps: viewOf(graph, 'uint32x2')
    });
    expectThrows(
      () => ({periods: fixed(10, 0n), columns: undefined}),
      /fixed periods need timestamps/
    );
    expectThrows(graph => withStamps(graph, 0, 0n), /periods.width must be a positive integer/);
    expectThrows(graph => withStamps(graph, -3n, 0n), /periods.width must be a positive integer/);
    expectThrows(graph => withStamps(graph, 1.5, 0n), /periods.width must be a positive integer/);
    expectThrows(
      graph => withStamps(graph, 2n ** 63n, 0n),
      /periods.width must be a positive integer/
    );
    expectThrows(graph => withStamps(graph, 1, 2n ** 63n), /periods.origin must be a signed Int64/);
    expectThrows(
      graph => ({...withStamps(graph, 1, 0n), timestamps: viewOf(graph, 'uint32') as never}),
      /timestamps/
    );
  });

  it('validates the mask, the periods output and the status words', () => {
    expectThrows(graph => ({mask: viewOf(graph, 'uint32', 5)}), /mask must have 6 rows/);
    expectThrows(
      graph => ({output: {periods: viewOf(graph, 'uint32', 5)}}),
      /one row per input row/
    );
    expectThrows(
      graph => ({output: {periods: viewOf(graph, 'sint32') as never}}),
      /output.periods/
    );
    expectThrows(
      graph => ({
        output: {
          periods: viewOf(graph, 'uint32'),
          status: {invalidCount: viewOf(graph, 'sint32') as never}
        }
      }),
      /invalidCount must contain one uint32 row/
    );
  });

  it('rejects an output that shares a buffer with an input', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    expect(
      () =>
        new GPUPeriodIndex({
          ...props,
          output: {periods: props.columns!.month as never}
        })
    ).toThrow(/outputs must not share buffers with inputs/);
    device.destroy();
  });

  it('refuses a fixed axis whose timestamps exceed one storage binding', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const limit = device.limits.maxStorageBufferBindingSize / 8;
    const build = (rows: number) => {
      const props = createProps(graph, {
        periods: {kind: 'fixed', width: 1000, origin: 0n, count: 8},
        columns: undefined,
        timestamps: viewOf(graph, 'uint32x2', rows),
        output: {periods: viewOf(graph, 'uint32', rows)}
      });
      graph.add(new GPUPeriodIndex({...props, id: `period-index-${rows}`}));
    };
    expect(() => build(limit)).not.toThrow();
    expect(() => build(limit + 1)).toThrow(
      /timestamps chunk 0 needs 134217736 bytes in one binding/
    );
    device.destroy();
  });

  it('declares the clear and the period kernel', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const withStatus = new GPUPeriodIndex({
      ...props,
      output: {...props.output, status: {invalidCount: viewOf(graph, 'uint32', 1)}}
    });
    expect(withStatus.getCommandNodes(graph).map(node => node.id)).toEqual([
      'period-index-clear-invalid-count',
      'period-index-periods-chunk-0'
    ]);
    expect(
      new GPUPeriodIndex({...props, id: 'without-status'})
        .getCommandNodes(graph)
        .map(node => node.id)
    ).toEqual(['without-status-periods-chunk-0']);
    device.destroy();
  });
});

describe('period index CPU twin', () => {
  it('preserves signed hour and unsigned period boundaries in CPU results and shaders', () => {
    const hourIndices = [-0x80000001, -1, 0, 0x7ffffffc, 0x7ffffffd, 0x7fffffff, 0x80000000];
    const fieldsAtHour = (hourIndex: number) => {
      const instant = new Date(hourIndex * 3_600_000);
      return {
        year: instant.getUTCFullYear(),
        month: instant.getUTCMonth() + 1,
        day: instant.getUTCDate(),
        hour: instant.getUTCHours()
      };
    };
    const fields = hourIndices.map(fieldsAtHour);
    for (const originHour of [0, -24, -0x80000000]) {
      for (const interval of [1, 3]) {
        const device = createNullWebGPUDevice();
        try {
          const graph = new GPUCommandGraph(device);
          const column = <Format extends 'uint32' | 'sint32'>(name: string, format: Format) =>
            createTransientView(graph, name, format, fields.length);
          const columns = {
            year: column('year', 'sint32'),
            month: column('month', 'uint32'),
            dayOfMonth: column('day', 'uint32'),
            hour: column('hour', 'uint32')
          };
          const output = {
            periods: column('periods', 'uint32'),
            status: {invalidCount: createTransientView(graph, 'invalid-count', 'uint32', 1)}
          };
          const step = {calendarUnit: 'hour', interval} as const;
          const origin = fieldsAtHour(originHour);
          const contributor = new GPUPeriodIndex({
            temporalContext: CONTEXT,
            periods: {kind: 'calendar', step, origin, count: GPU_PERIOD_INDEX_MAXIMUM_COUNT},
            columns,
            output
          });
          const expected = hourIndices.map(hourIndex => {
            const difference = hourIndex - originHour;
            const candidate = Math.floor(difference / interval);
            return difference >= 0 && candidate < GPU_PERIOD_INDEX_MAXIMUM_COUNT
              ? candidate
              : GPU_PERIOD_INDEX_OUT_OF_RANGE;
          });
          expect(
            fields.map(value => {
              const candidate = getGPUPeriodIndexCalendarPeriod(
                step,
                origin,
                GPU_PERIOD_INDEX_MAXIMUM_COUNT,
                value
              );
              return candidate < 0 ? GPU_PERIOD_INDEX_OUT_OF_RANGE : candidate;
            })
          ).toEqual(expected);
          const nodes = contributor.getCommandNodes(graph);
          getParsedKernelSources(nodes);
          const source = getGPUCommandNodeKernelSources(nodes).at(-1)!.source;
          expect(source).toContain('candidate.y < 4294967293u');
          expect(source).toContain('multiplyUint64ByUint32(calendarStepWords(daysFromCivil');
          expect(source).not.toContain('candidate < 2147483647');
        } finally {
          device.destroy();
        }
      }
    }
  });

  it('matches Date arithmetic at calendar boundaries', () => {
    const origin = {year: 2023, month: 11, day: 15, hour: 6};
    const times = [
      Date.UTC(2024, 1, 29),
      Date.UTC(2024, 1, 29, 23, 59, 59, 999),
      Date.UTC(2024, 2, 1),
      Date.UTC(2024, 2, 31, 23, 59, 59, 999),
      Date.UTC(2024, 3, 1),
      Date.UTC(2023, 11, 31, 23),
      Date.UTC(2023, 10, 15, 5),
      Date.UTC(1969, 11, 31, 23, 59, 59, 999),
      Date.UTC(1960, 1, 29, 12)
    ];
    for (const calendarUnit of ['year', 'quarter', 'month', 'week', 'day', 'hour'] as const) {
      for (const interval of [1, 3]) {
        for (const time of times) {
          const date = new Date(time);
          const fields = {
            year: date.getUTCFullYear(),
            month: date.getUTCMonth() + 1,
            day: date.getUTCDate(),
            hour: date.getUTCHours()
          };
          expect(
            getGPUPeriodIndexCalendarPeriod({calendarUnit, interval}, origin, 100_000, fields),
            `${calendarUnit} ${interval} ${date.toISOString()}`
          ).toBe(getExpectedCalendarPeriod({calendarUnit, interval}, origin, 100_000, time));
        }
      }
    }
  });

  it('numbers days from 1970 across the leap rules', () => {
    expect(getGPUPeriodIndexDaysFromCivil(1970, 1, 1)).toBe(0);
    expect(getGPUPeriodIndexDaysFromCivil(1969, 12, 31)).toBe(-1);
    expect(
      getGPUPeriodIndexDaysFromCivil(2000, 3, 1) - getGPUPeriodIndexDaysFromCivil(2000, 2, 28)
    ).toBe(2);
    expect(
      getGPUPeriodIndexDaysFromCivil(1900, 3, 1) - getGPUPeriodIndexDaysFromCivil(1900, 2, 28)
    ).toBe(1);
    expect(getGPUPeriodIndexStepIndex('quarter', {year: 2024, month: 4})).toBe(2024 * 4 + 1);
    expect(getGPUPeriodIndexStepIndex('hour', {year: 1970, month: 1, day: 2, hour: 3})).toBe(27);
  });
});
