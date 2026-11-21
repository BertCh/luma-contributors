// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';
import {getGPUPeriodIndexCalendarPeriod} from '../../../src/dataframe-analysis/period-index/period-index-calendar';
import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  GPUCalendarBuckets,
  type GPUCalendarBucketsOutput
} from '../../../src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets';
import {GPUForestInference} from '../../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {GPUPeriodIndex} from '../../../src/dataframe-analysis/period-index/gpu-period-index';
import {GPUTimeWordsRelative} from '../../../src/dataframe-analysis/time-words-relative/gpu-time-words-relative';
import {GPUCellPeriodKeys} from '../../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {GPUTimeWindowFilter} from '../../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

const foreign = createRequire(import.meta.url)(
  '@luma.gl/graph/gpu-core'
) as typeof import('@luma.gl/graph/gpu-core');
const context = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
} as const;

it('six constructors reject extra top-level spread keys and accept their declared branches (TYP-04)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'sint32' | 'uint32' | 'uint32x2'>(
      format: Format,
      length = 1
    ) => createTransientView(graph, `v-${serial++}`, format, length);
    const calendar = {
      timestamps: view('uint32x2'),
      parameters: view('sint32', 2),
      output: {year: view('sint32')}
    };
    const relative = {
      timestamps: view('uint32x2'),
      temporalContext: context,
      origin: 0n,
      output: view('float32')
    };
    const period = {
      timestamps: view('uint32x2'),
      temporalContext: context,
      periods: {kind: 'fixed', width: 1n, origin: 0n, count: 1},
      output: {periods: view('uint32')}
    } as const;
    const keys = {
      cells: view('uint32x2'),
      periods: view('uint32'),
      periodCount: 1,
      fill: 'zeros',
      layout: 'legacy-one-bit',
      output: {keys: view('uint32x2')}
    } as const;
    const forest = {
      rowCount: 1,
      model: {
        featureCount: 1,
        treeCount: 1,
        nodeCount: 1,
        outputCount: 1,
        maximumDepth: 0,
        task: 'regression',
        transform: 'identity'
      },
      nodes: view('uint32', 6),
      trees: view('uint32', 4),
      leafValues: view('float32'),
      parameters: view('float32', 2),
      features: view('float32'),
      output: {predictions: view('float32'), validity: view('uint32')}
    } as const;
    const group = {
      keys: view('uint32'),
      columns: [{values: view('float32'), statistics: ['count'], output: {counts: view('uint32')}}],
      output: {
        keys: view('uint32'),
        counts: view('uint32'),
        status: {count: view('uint32'), overflow: view('uint32')}
      }
    } as const;
    const constructors = [
      () => new GPUCalendarBuckets({...calendar, optoin: true} as never),
      () => new GPUTimeWordsRelative({...relative, optoin: true} as never),
      () => new GPUPeriodIndex({...period, optoin: true} as never),
      () => new GPUCellPeriodKeys({...keys, optoin: true} as never),
      () => new GPUForestInference({...forest, optoin: true} as never),
      () => new GPUGroupStatistics({...group, optoin: true} as never)
    ];
    for (const construct of constructors) expect(construct).toThrow(/unknown prop optoin/);
    expect(() => new GPUCalendarBuckets(calendar)).not.toThrow();
    expect(() => new GPUTimeWordsRelative(relative)).not.toThrow();
    expect(() => new GPUPeriodIndex(period)).not.toThrow();
    expect(() => new GPUCellPeriodKeys(keys)).not.toThrow();
    expect(new GPUCellPeriodKeys({...keys, layout: 'zero-period-bit-identity'}).periodBits).toBe(0);
    expect(
      () => new GPUCellPeriodKeys({...keys, periodCount: 2, layout: 'zero-period-bit-identity'})
    ).toThrow(/periodCount = 1.*2/);
    expect(() => new GPUCellPeriodKeys({...keys, layout: 'typo' as never})).toThrow(
      /legacy-one-bit.*zero-period-bit-identity/
    );

    expect(() => new GPUForestInference(forest)).not.toThrow();
    expect(() => new GPUGroupStatistics(group)).not.toThrow();
    const realm = <Value extends object>(props: Value): Value =>
      runInNewContext('({...props})', {props});
    for (const construct of [
      () => new GPUCalendarBuckets(realm(calendar)),
      () => new GPUTimeWordsRelative(realm(relative)),
      () => new GPUPeriodIndex(realm(period)),
      () => new GPUCellPeriodKeys(realm(keys)),
      () => new GPUForestInference(realm(forest))
    ])
      expect(construct).not.toThrow();
    expect(() => new GPUCalendarBuckets(realm({...calendar, optoin: true}) as never)).toThrow(
      /unknown prop optoin/
    );

    if (false) {
      // @ts-expect-error fresh literals must reject extra keys while retaining inferred view types
      new GPUCalendarBuckets({...calendar, optoin: true});
      // @ts-expect-error extra spread keys must also be rejected
      new GPUTimeWordsRelative({...relative, ...{optoin: true}});
      // @ts-expect-error period constructor exact keys
      new GPUPeriodIndex({...period, optoin: true});
      // @ts-expect-error key constructor exact keys
      new GPUCellPeriodKeys({...keys, optoin: true});
      // @ts-expect-error forest union includes both branches, but not arbitrary keys
      new GPUForestInference({...forest, optoin: true});
      // @ts-expect-error group constructor exact keys (implemented by grouped-reductions owner)
      new GPUGroupStatistics({...group, optoin: true});
      // @ts-expect-error empty calendar selection has no terminal output
      const empty: GPUCalendarBucketsOutput = {};
      void empty;
      const calendarUnion = null as unknown as typeof calendar | (typeof calendar & {optoin: true});
      const relativeUnion = null as unknown as typeof relative | (typeof relative & {optoin: true});
      const periodUnion = null as unknown as typeof period | (typeof period & {optoin: true});
      const keysUnion = null as unknown as typeof keys | (typeof keys & {optoin: true});
      const forestUnion = null as unknown as typeof forest | (typeof forest & {optoin: true});
      // @ts-expect-error union alternatives must not conceal an extra key (S3)
      new GPUCalendarBuckets(calendarUnion);
      // @ts-expect-error inferred keys are distributive for relative time too
      new GPUTimeWordsRelative(relativeUnion);
      // @ts-expect-error inferred keys are distributive for period indexing too
      new GPUPeriodIndex(periodUnion);
      // @ts-expect-error inferred keys are distributive for cell keys too
      new GPUCellPeriodKeys(keysUnion);
      // @ts-expect-error inferred keys are distributive for every forest input branch
      new GPUForestInference(forestUnion);
      type GroupUnion = typeof group | (typeof group & {optoin: true});
      const groupUnion: GroupUnion = Math.random() > 0.5 ? group : {...group, optoin: true};
      // @ts-expect-error infer the union argument without an explicit generic (review S2)
      new GPUGroupStatistics(groupUnion);
      // A union variable must fail the constructor parameter even if one branch has no typo.
      expectTypeOf<GroupUnion>().not.toExtend<
        ConstructorParameters<typeof GPUGroupStatistics<GroupUnion>>[0]
      >();
    }
  } finally {
    device.destroy();
  }
});

it('TSC-01 and S3 preserve packed/chunked calendar and independent group inference', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'sint32' | 'float32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `inference-${serial++}`, format, length);
    const timestamps = view('uint32x2', 2);
    const chunkedTimestamps = createVectorView('times', 'uint32x2', [
      view('uint32x2', 1),
      view('uint32x2', 1)
    ]);
    const year = view('sint32', 2);
    const chunkedYear = createVectorView('years', 'sint32', [view('sint32', 1), view('sint32', 1)]);
    const packed = new GPUCalendarBuckets({
      timestamps,
      parameters: view('sint32', 2),
      temporalContext: context,
      output: {year}
    });
    const chunked = new GPUCalendarBuckets({
      timestamps: chunkedTimestamps,
      parameters: view('sint32', 2),
      temporalContext: context,
      output: {year: chunkedYear}
    });
    expectTypeOf(packed.props.timestamps).toExtend<GraphDataView<'uint32x2'>>();
    expectTypeOf(chunked.props.timestamps).toExtend<GraphVectorView<'uint32x2'>>();
    expectTypeOf(packed.props.output.year).toExtend<GraphDataView<'sint32'>>();
    expectTypeOf(chunked.props.output.year).toExtend<GraphVectorView<'sint32'>>();
    expect(() => new GPUCalendarBuckets({...packed.props, unit: 'second'})).toThrow(
      /contradicts temporalContext/
    );
    expect(() => new GPUCalendarBuckets({...chunked.props, unit: 'second'})).toThrow(
      /contradicts temporalContext/
    );

    const keys = view('uint32', 2);
    const values = view('float32', 2);
    const chunkedKeys = createVectorView('keys', 'uint32', [view('uint32', 1), view('uint32', 1)]);
    const chunkedValues = createVectorView('values', 'float32', [
      view('float32', 1),
      view('float32', 1)
    ]);
    const output = () => ({
      keys: view('uint32', 1),
      counts: view('uint32', 1),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    });
    const chunkedColumns = new GPUGroupStatistics({
      keys,
      columns: [{values: chunkedValues, statistics: ['mean'], output: {means: view('float32', 1)}}],
      output: output()
    });
    const chunkedGroups = new GPUGroupStatistics({
      keys: chunkedKeys,
      columns: [{values, statistics: ['mean'], output: {means: view('float32', 1)}}],
      output: output()
    });
    expectTypeOf(chunkedColumns.props.keys).toExtend<GraphDataView<'uint32'>>();
    expectTypeOf(chunkedColumns.props.columns[0].values).toExtend<GraphVectorView<'float32'>>();
    expectTypeOf(chunkedGroups.props.keys).toExtend<GraphVectorView<'uint32'>>();
    expectTypeOf(chunkedGroups.props.columns[0].values).toExtend<GraphDataView<'float32'>>();
  } finally {
    device.destroy();
  }
});

it('calendar selections accept every minimal terminal leaf (DEF-09)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'sint32'>(format: Format, length = 1) =>
      createTransientView(graph, `terminal-${serial++}`, format, length);
    const selections = [
      {year: view('sint32')} satisfies GPUCalendarBucketsOutput,
      {month: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {dayOfMonth: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {hour: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {minute: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {weekday: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {dayOfYear: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {isoWeek: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {isoWeekYear: view('sint32')} satisfies GPUCalendarBucketsOutput,
      {quarter: view('uint32')} satisfies GPUCalendarBucketsOutput,
      {hourWeekdayCounts: view('uint32', 168)} satisfies GPUCalendarBucketsOutput
    ];
    for (const [index, output] of selections.entries()) {
      expect(
        () =>
          new GPUCalendarBuckets({
            timestamps: createTransientView(graph, `time-${index}`, 'uint32x2', 1),
            parameters: createTransientView(graph, `parameters-${index}`, 'sint32', 2),
            output
          })
      ).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('review S1 rejects unknown explicit calendar units before generating a shader', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      id: 'unit-review',
      timestamps: createTransientView(graph, 'unit-times', 'uint32x2', 1),
      parameters: createTransientView(graph, 'unit-parameters', 'sint32', 2),
      output: {year: createTransientView(graph, 'unit-year', 'sint32', 1)}
    };
    for (const temporalContext of [undefined, context]) {
      expect(() =>
        Reflect.construct(GPUCalendarBuckets, [{...props, unit: 'milliseconds', temporalContext}])
      ).toThrow(
        /^unit-review.*unit.*second.*millisecond.*microsecond.*nanosecond.*received milliseconds/
      );
    }
    for (const unit of ['second', 'millisecond', 'microsecond', 'nanosecond'] as const) {
      expect(new GPUCalendarBuckets({...props, unit}).unit).toBe(unit);
    }
  } finally {
    device.destroy();
  }
});

it('foreign-format packed outputs retain capacity and foreign vectors cannot enter the sorted shortcut (ENT-02)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'sint32' | 'uint32' | 'uint32x2'>(
      format: Format,
      length = 1
    ) => createTransientView(graph, `foreign-${serial++}`, format, length);
    const words = view('uint32x2');
    const output = view('sint32', 2);
    const foreignOutput = new foreign.GraphDataView(output.buffer, {...output});
    expect(foreignOutput instanceof GraphDataView).toBe(false);
    expect(
      () =>
        new GPUCalendarBuckets({
          timestamps: words,
          parameters: view('sint32', 2),
          output: {year: foreignOutput}
        })
    ).not.toThrow();
    const periodOutput = view('uint32', 2);
    expect(
      () =>
        new GPUPeriodIndex({
          timestamps: words,
          temporalContext: context,
          periods: {kind: 'fixed', width: 1n, origin: 0n, count: 1},
          output: {periods: new foreign.GraphDataView(periodOutput.buffer, {...periodOutput})}
        })
    ).not.toThrow();
    const timestamps = new foreign.GraphVectorView({
      ...createVectorView('foreign-times', 'float32', [view('float32'), view('float32')])
    });
    expect(timestamps instanceof GraphVectorView).toBe(false);
    const props = {
      timestamps,
      window: view('float32', 8),
      output: {rows: view('uint32', 2), status: {count: view('uint32'), overflow: view('uint32')}}
    };
    expect(() => new GPUTimeWindowFilter({...props, sorted: true})).toThrow(
      /single packed timestamps/
    );
    // The unsorted classifier must unpack ordered foreign vector chunks, not bind the vector itself.
    expect(() => new GPUTimeWindowFilter(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('foreign chunked forests retain per-chunk work admission (ENT-02)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `forest-foreign-${serial++}`, format, length);
    const vector = <Format extends 'float32' | 'uint32'>(format: Format) =>
      new foreign.GraphVectorView({
        ...createVectorView(`forest-vector-${serial++}`, format, [
          view(format, 50_000),
          view(format, 50_000)
        ])
      });
    // 100000 * 1024 * (64+1+1) = 6,758,400,000 total work; each 50000-row batch is below 5e9.
    expect(
      () =>
        new GPUForestInference({
          rowCount: 100_000,
          model: {
            featureCount: 1,
            treeCount: 1024,
            nodeCount: 1,
            outputCount: 1,
            maximumDepth: 64,
            task: 'regression',
            transform: 'identity'
          },
          features: vector('float32'),
          nodes: view('uint32', 6),
          trees: view('uint32', 4096),
          leafValues: view('float32', 1),
          parameters: view('float32', 2),
          output: {predictions: vector('float32'), validity: vector('uint32')}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('captures non-enumerable nested origin fields and output ports (B3)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const origin = {year: 2025, month: 7, day: 1};
    Object.defineProperty(origin, 'month', {value: 7, enumerable: false, writable: true});
    const producer = new GPUPeriodIndex({
      temporalContext: context,
      periods: {kind: 'calendar', step: {calendarUnit: 'day'}, origin, count: 365},
      columns: {
        year: createTransientView(graph, 'year', 'sint32', 1),
        month: createTransientView(graph, 'month', 'uint32', 1),
        dayOfMonth: createTransientView(graph, 'day', 'uint32', 1)
      },
      output: {periods: createTransientView(graph, 'period', 'uint32', 1)}
    });
    const periods = producer.props.periods;
    if (periods.kind !== 'calendar') throw new Error('calendar fixture');
    expect(
      getGPUPeriodIndexCalendarPeriod(periods.step, periods.origin, periods.count, {
        year: 2025,
        month: 7,
        day: 1
      })
    ).toBe(0);
    origin.month = 9;
    expect(periods.origin.month).toBe(7);
    const month = createTransientView(graph, 'calendar-month', 'uint32', 1);
    const output = {month};
    Object.defineProperty(output, 'month', {value: month, enumerable: false});
    const calendar = new GPUCalendarBuckets({
      timestamps: createTransientView(graph, 'words', 'uint32x2', 1),
      parameters: createTransientView(graph, 'parameters', 'sint32', 2),
      output
    });
    expect(calendar.props.output.month).toBe(month);
  } finally {
    device.destroy();
  }
});
