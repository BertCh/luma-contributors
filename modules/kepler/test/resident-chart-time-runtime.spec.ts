// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerScalarSelection} from '../src/selection/gpu-scalar-selection';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import {GPUKeplerTimeSelection} from '../src/selection/gpu-time-selection';
import {ResidentTimeChartRuntime} from '../../../examples/deck/kepler-compute/src/resident-chart-time-runtime';
import type {ResidentChartSelection} from '../../../examples/deck/kepler-compute/src/resident-chart-group-runtime';
import type {ResidentTimeChartPlan} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

test.each([
  'millisecond',
  'second'
] as const)('UTC calendar GPU charts retain exact %s time, chunks, selected scopes, periods and series', async unit => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const instants = [
    '1969-12-31T23:59:59.999Z',
    '2024-02-29T23:59:59.999Z',
    '2024-03-01T00:00:00Z',
    '2024-03-10T01:59:59-05:00',
    '2024-03-10T03:00:00-04:00',
    '2024-11-03T01:30:00-04:00',
    '2024-11-03T01:30:00-05:00',
    '0000-02-01T00:00:00Z',
    '0099-12-31T23:59:59.999Z',
    '-000001-12-31T23:59:59.999Z'
  ].map(value => {
    const instant = Date.parse(value);
    return unit === 'second' ? Math.floor(instant / 1000) * 1000 : instant;
  });
  const source = new arrow.Table({
    time: arrow.vectorFromArray(
      instants,
      unit === 'second'
        ? new arrow.TimestampSecond('America/New_York')
        : new arrow.TimestampMillisecond('America/New_York')
    ),
    amount: arrow.vectorFromArray([1, 2, null, 4, 6, 8, 10, 12, 14, 16], new arrow.Float32()),
    series: arrow.vectorFromArray([
      'One',
      'One',
      'Two',
      'One',
      'Two',
      'One',
      'Two',
      'One',
      'One',
      'One'
    ])
  });
  const dataset = session.importArrowDataset({
    id: 'calendar',
    table: source.slice(0, 3).concat(source.slice(3, 7), source.slice(7))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const time = dataset.value.timeColumns['time'];
  const filter = new GPUKeplerTimeSelection({
    device,
    timestamps: time.timestamps,
    validity: time.validity,
    unit,
    window: {
      start: Date.parse('2024-03-01T00:00:00Z'),
      end: Date.parse('2024-04-01T00:00:00Z'),
      unit: 'millisecond'
    }
  });
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: time.timestamps,
    scalarFilters: [
      {
        filterId: 'owned',
        mask: filter.mask,
        nativeTopology: true,
        nativeFieldName: 'time',
        dynamicDomain: false
      }
    ]
  });
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    isLayerRegistrationPending: false,
    canConsume: true,
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => {
      filter.encode(encoder);
      scopes.encode(encoder);
    }
  };
  const readers: ResidentTimeChartRuntime[] = [];
  const baseline = session.registry.memoryUsage;
  const buffers = time.timestamps.data.map(chunk => chunk.buffer);
  const create = (changes: Partial<ResidentTimeChartPlan> = {}, budget = 8 * 1024 * 1024) => {
    const reader = new ResidentTimeChartRuntime({
      session,
      explorer,
      selection,
      workingMemoryBudgetBytes: budget,
      plan: {
        kind: 'time',
        chartId: `calendar-${readers.length}`,
        datasetId: dataset.id,
        population: 'all',
        operation: 'count',
        timeColumn: 'time',
        interval: 'month',
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: [],
        ...changes
      }
    });
    readers.push(reader);
    return reader;
  };
  try {
    const months = await create().read();
    expect(months?.periods.map(period => period.label)).toEqual([
      '-1-12',
      '0-02',
      '99-12',
      '1969-12',
      '2024-02',
      '2024-03',
      '2024-11'
    ]);
    expect(months?.summaries.map(summary => summary.count)).toEqual([1, 1, 1, 1, 1, 3, 2]);
    expect(months?.periods[5]).toEqual({
      label: '2024-03',
      start: Date.parse('2024-03-01T00:00:00Z'),
      end: Date.parse('2024-04-01T00:00:00Z')
    });
    const expectations = {sum: 10, mean: 5, min: 4, max: 6};
    for (const operation of ['sum', 'mean', 'min', 'max'] as const) {
      const result = await create({operation, column: 'amount', population: 'selected'}).read();
      expect(result?.periods.map(period => period.label)).toEqual(['2024-03']);
      expect(result?.summaries[0]).toMatchObject({
        value: expectations[operation],
        count: 3,
        firstRow: 2
      });
    }
    const excluding = await create({
      population: 'selected',
      excludedCpuFilterIds: ['owned'],
      excludedGpuFieldNames: ['time']
    }).read();
    expect(excluding?.periods).toEqual(months?.periods);
    const series = await create({
      operation: 'mean',
      column: 'amount',
      series: {column: 'series', categoryCount: 2}
    }).read();
    expect(series?.summaries[5]).toMatchObject({
      value: 10,
      count: 3,
      series: [
        {category: 'Two', value: 6, count: 2},
        {category: 'One', value: 4, count: 1}
      ]
    });
    const years = await create({interval: 'year'}).read();
    expect(years?.periods.map(period => period.label)).toEqual(['-1', '0', '99', '1969', '2024']);
    expect(years?.summaries.map(summary => summary.count)).toEqual([1, 1, 1, 1, 6]);
    for (const interval of ['day', 'week'] as const) {
      // Epoch-day arithmetic is an independent oracle that also handles negative years.
      const starts = instants.map(instant => {
        const dayStart = Math.floor(instant / 86400000) * 86400000;
        return interval === 'week'
          ? dayStart - new Date(dayStart).getUTCDay() * 86400000
          : dayStart;
      });
      const ordered = [...new Set(starts)].sort((first, second) => first - second);
      const result = await create({interval}).read();
      expect(result?.periods.map(period => [period.start, period.end])).toEqual(
        ordered.map(start => [start, start + (interval === 'week' ? 7 : 1) * 86400000])
      );
      expect(result?.summaries.map(summary => summary.count)).toEqual(
        ordered.map(start => starts.filter(value => value === start).length)
      );
    }
    const cancelled = create();
    const pending = cancelled.read();
    cancelled.cancel();
    expect(await pending).toBeNull();
    expect(() => create({}, 1)).toThrow('before allocation');
    expect(time.timestamps.data.map(chunk => chunk.buffer)).toEqual(buffers);
    expect(time.timestamps.data.map(chunk => chunk.length)).toEqual([3, 4, 3]);
  } finally {
    await Promise.all(readers.map(reader => reader.destroy()));
    expect(session.registry.memoryUsage).toEqual(baseline);
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await filter.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('calendar charts group nulls at epoch, exclude invalid dates and bound cardinality without changing source ownership', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  try {
    for (const count of [2, 3, 513]) {
      const instants = Array.from({length: count}, (_, index) => {
        const date = new Date(0);
        date.setUTCFullYear(1900 + index, 0, 1);
        return date.getTime();
      });
      const dataset = session.importArrowDataset({
        id: `calendar-${count}`,
        table: new arrow.Table({
          time: arrow.vectorFromArray(
            count === 2
              ? [null, instants[1]]
              : count === 3
                ? [null, instants[1], 8640000000000001]
                : instants,
            new arrow.TimestampMillisecond()
          ),
          amount: arrow.makeVector(new Float32Array(count)),
          selected: arrow.makeVector(
            Uint32Array.from({length: count}, (_, index) => (index === 0 ? 0 : 1))
          )
        })
      });
      const explorer = session.openExplorer(dataset.id, {
        dimensions: [],
        workingMemoryBudgetBytes: 8 * 1024 * 1024
      });
      const sourceMask = new GPUVector({
        name: 'selected',
        type: 'data',
        format: 'uint32',
        data: dataset.value.columns['selected'].data,
        ownsData: false
      });
      const scopes = new GPUKeplerSelectionScopes({
        device,
        rows: dataset.value.timeColumns['time'].timestamps,
        scalarFilters: [
          {
            filterId: 'other',
            mask: sourceMask,
            nativeTopology: true,
            nativeFieldName: 'other',
            dynamicDomain: false
          }
        ]
      });
      const selection: ResidentChartSelection = {
        get currentRevision() {
          return scopes.currentRevision;
        },
        isLayerRegistrationPending: false,
        canConsume: true,
        getConsumerMask: request => scopes.getConsumerMask(request),
        removeConsumerMask: id => scopes.removeConsumerMask(id),
        encode: encoder => {
          scopes.encode(encoder);
        }
      };
      const baseline = session.registry.memoryUsage;
      const plan: ResidentTimeChartPlan = {
        kind: 'time',
        chartId: 'guard',
        datasetId: dataset.id,
        population: 'all',
        operation: 'count',
        timeColumn: 'time',
        interval: 'year',
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: []
      };
      const reader = new ResidentTimeChartRuntime({
        session,
        explorer,
        selection,
        plan,
        workingMemoryBudgetBytes: 8 * 1024 * 1024
      });
      const selected = new ResidentTimeChartRuntime({
        session,
        explorer,
        selection,
        plan: {...plan, chartId: 'selected', population: 'selected'},
        workingMemoryBudgetBytes: 8 * 1024 * 1024
      });
      try {
        // Native shows up to 1,000 periods before falling back to equal-width bins.
        if (count === 513) expect((await reader.read())?.periods).toHaveLength(513);
        else
          expect((await reader.read())?.periods.map(period => period.label)).toEqual([
            '1901',
            '1970'
          ]);
        expect((await selected.read())?.periods).toHaveLength(count === 513 ? 512 : 1);
      } finally {
        await reader.destroy();
        await selected.destroy();
        expect(session.registry.memoryUsage).toEqual(baseline);
        const fence = device.createFence();
        try {
          await scopes.destroy(fence.signaled);
        } finally {
          fence.destroy();
        }
      }
    }
  } finally {
    const fence = device.createFence();
    try {
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('second calendar charts normalize exact source words while preserving nulls and invalid Date clipping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const monthStart = BigInt(Date.parse('2024-03-01T00:00:00Z') / 1000);
  const source = new arrow.Table({
    time: arrow.makeVector(
      arrow.makeData({
        type: new arrow.TimestampSecond(),
        data: new BigInt64Array([
          (1n << 63n) - 1n, // Null words must not make a source null into an invalid instant.
          -1n,
          monthStart - 1n,
          monthStart,
          8640000000001n,
          -8640000000001n,
          (1n << 63n) - 1n,
          -(1n << 63n)
        ]),
        length: 8,
        nullCount: 1,
        nullBitmap: new Uint8Array([0xfe])
      })
    )
  });
  const dataset = session.importArrowDataset({
    id: 'second-boundaries',
    table: source.slice(0, 3).concat(source.slice(3))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const baseline = session.registry.memoryUsage;
  const time = dataset.value.timeColumns['time'];
  const buffers = time.timestamps.data.map(chunk => chunk.buffer);
  const filter = new GPUKeplerScalarSelection({
    device,
    filters: [
      {
        filterId: 'chart-click',
        kind: 'time',
        timestamps: time.timestamps,
        validity: time.validity,
        unit: 'second',
        chartNullAsEpoch: true,
        window: {start: Date.parse('1969-12-01T00:00:00Z'), end: 0, unit: 'millisecond'}
      }
    ]
  });
  filter.setTimeWindow('chart-click', {
    start: Date.parse('1969-12-01T00:00:00Z'),
    end: 0,
    unit: 'millisecond'
  });
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: time.timestamps,
    scalarFilters: [
      {
        filterId: 'chart-click',
        mask: filter.getMask('chart-click'),
        nativeTopology: true,
        nativeFieldName: 'time',
        dynamicDomain: false
      }
    ]
  });
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    isLayerRegistrationPending: false,
    canConsume: true,
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => {
      filter.encode(encoder);
      scopes.encode(encoder);
    }
  };
  const plan: ResidentTimeChartPlan = {
    kind: 'time',
    chartId: 'seconds',
    datasetId: dataset.id,
    population: 'all',
    operation: 'count',
    timeColumn: 'time',
    interval: 'month',
    excludedCpuFilterIds: [],
    excludedGpuFieldNames: []
  };
  const reader = new ResidentTimeChartRuntime({
    session,
    explorer,
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    plan
  });
  const selected = new ResidentTimeChartRuntime({
    session,
    explorer,
    selection,
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    plan: {...plan, chartId: 'selected-seconds', population: 'selected'}
  });
  try {
    for (let iteration = 0; iteration < 2; iteration++) {
      const result = await reader.read();
      expect(result?.periods.map(period => period.label)).toEqual([
        '1969-12',
        '1970-01',
        '2024-02',
        '2024-03'
      ]);
      expect(result?.summaries.map(summary => summary.count)).toEqual([1, 1, 1, 1]);
    }
    // Native month clicks include both endpoints: the null epoch sits at December's upper bound.
    const december = await selected.read();
    expect(december?.periods.map(period => period.label)).toEqual(['1969-12', '1970-01']);
    expect(december?.summaries.map(summary => summary.count)).toEqual([1, 1]);
    filter.setTimeWindow('chart-click', {
      start: Date.parse('2024-02-01T00:00:00Z'),
      end: Date.parse('2024-03-01T00:00:00Z'),
      unit: 'millisecond'
    });
    scopes.invalidate();
    const february = await selected.read();
    expect(february?.periods.map(period => period.label)).toEqual(['2024-02', '2024-03']);
    expect(february?.summaries.map(summary => summary.count)).toEqual([1, 1]);
    expect(time.timestamps.data.map(chunk => chunk.length)).toEqual([3, 5]);
    expect(time.timestamps.data.map(chunk => chunk.buffer)).toEqual(buffers);
  } finally {
    await reader.destroy();
    await selected.destroy();
    expect(session.registry.memoryUsage).toEqual(baseline);
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await filter.close(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('UTC day/week charts preserve Sunday boundaries and null epoch, guard Date edges and cap observed periods', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const sunday = Date.parse('2024-03-03T00:00:00Z');
  const minimumDate = -8640000000000000;
  const maximumDate = 8640000000000000;
  try {
    for (const interval of ['day', 'week'] as const) {
      const periodMilliseconds = (interval === 'week' ? 7 : 1) * 86400000;
      for (const scenario of ['boundaries', 'minimum', 'maximum', '512', '513'] as const) {
        const instants =
          scenario === 'boundaries'
            ? [null, -1, 0, sunday - 1, sunday, sunday + 1, maximumDate + 1]
            : scenario === 'minimum'
              ? [minimumDate]
              : scenario === 'maximum'
                ? [maximumDate]
                : Array.from(
                    {length: Number(scenario)},
                    (_, index) => sunday + index * periodMilliseconds
                  );
        const source = new arrow.Table({
          time: arrow.vectorFromArray(instants, new arrow.TimestampMillisecond())
        });
        const dataset = session.importArrowDataset({
          id: `${interval}-${scenario}`,
          table: source.slice(0, 1).concat(source.slice(1))
        });
        const explorer = session.openExplorer(dataset.id, {
          dimensions: [],
          workingMemoryBudgetBytes: 8 * 1024 * 1024
        });
        const baseline = session.registry.memoryUsage;
        const time = dataset.value.timeColumns['time'];
        const buffers = time.timestamps.data.map(chunk => chunk.buffer);
        const reader = new ResidentTimeChartRuntime({
          session,
          explorer,
          workingMemoryBudgetBytes: 8 * 1024 * 1024,
          plan: {
            kind: 'time',
            chartId: `${interval}-${scenario}`,
            datasetId: dataset.id,
            population: 'all',
            operation: 'count',
            timeColumn: 'time',
            interval,
            excludedCpuFilterIds: [],
            excludedGpuFieldNames: []
          }
        });
        try {
          if (scenario === '513') expect((await reader.read())?.periods).toHaveLength(513);
          else if (scenario === 'maximum' || (scenario === 'minimum' && interval === 'week'))
            await expect(reader.read()).rejects.toThrow('Date boundary');
          else {
            const result = await reader.read();
            if (scenario === '512') expect(result?.periods).toHaveLength(512);
            else if (scenario === 'minimum')
              expect(result?.periods).toEqual([
                {label: '-271821-04-20', start: minimumDate, end: minimumDate + 86400000}
              ]);
            else {
              expect(result?.periods.map(period => period.label)).toEqual(
                interval === 'day'
                  ? ['1969-12-31', '1970-01-01', '2024-03-02', '2024-03-03']
                  : ['Week of 1969-12-28', 'Week of 2024-02-25', 'Week of 2024-03-03']
              );
              expect(result?.summaries.map(summary => summary.count)).toEqual(
                interval === 'day' ? [1, 2, 1, 2] : [3, 1, 2]
              );
            }
          }
          expect(time.timestamps.data.map(chunk => chunk.buffer)).toEqual(buffers);
        } finally {
          await reader.destroy();
          expect(session.registry.memoryUsage).toEqual(baseline);
        }
      }
    }
  } finally {
    const fence = device.createFence();
    try {
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
