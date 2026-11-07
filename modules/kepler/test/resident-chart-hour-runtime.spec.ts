// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerScalarSelection} from '../src/selection/gpu-scalar-selection';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import {ResidentTimeChartRuntime} from '../../../examples/deck/kepler-compute/src/resident-chart-time-runtime';
import {
  INTEGER_DOMAIN_READ_BYTES,
  ResidentIntegerChartDomain
} from '../../../examples/deck/kepler-compute/src/resident-integer-domain';
import type {ResidentChartSelection} from '../../../examples/deck/kepler-compute/src/resident-chart-group-runtime';
import {MAXIMUM_RESIDENT_TIME_GROUPS} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import type {ResidentTimeChartPlan} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const HOUR_MILLISECONDS = 3600000;
// Series-free time charts reserve the native 1,000-period capacity for count, sum and result readback.
const COUNT_READ_BYTES = INTEGER_DOMAIN_READ_BYTES * 2 + MAXIMUM_RESIDENT_TIME_GROUPS * 4 * 3;

test.each([
  'millisecond',
  'second'
] as const)('hour charts retain full Date range, original %s chunks, series and exact selected scopes', async unit => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const instants = [
    -8640000000000000,
    8640000000000000 - 1000,
    -3600000,
    -1,
    0,
    1,
    Date.parse('0000-02-29T23:59:59.999Z'),
    Date.parse('0099-12-31T23:59:59.999Z'),
    Date.parse('-000001-12-31T23:59:59.999Z'),
    Date.parse('2024-03-10T01:59:59-05:00'),
    Date.parse('2024-03-10T03:00:00-04:00'),
    null,
    8640000000000000 + 1000
  ].map(instant =>
    instant === null || unit === 'millisecond' ? instant : Math.floor(instant / 1000) * 1000
  );
  const source = new arrow.Table({
    time: arrow.vectorFromArray(
      instants,
      unit === 'second'
        ? new arrow.TimestampSecond('America/New_York')
        : new arrow.TimestampMillisecond('America/New_York')
    ),
    amount: arrow.makeVector(Float32Array.from(instants, (_, index) => index + 1)),
    series: arrow.vectorFromArray(instants.map((_, index) => (index % 2 ? 'Odd' : 'Even')))
  });
  const dataset = session.importArrowDataset({
    id: `hours-${unit}`,
    table: source.slice(0, 3).concat(source.slice(3, 8), source.slice(8))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const time = dataset.value.timeColumns['time'];
  const buffers = time.timestamps.data.map(chunk => chunk.buffer);
  const filter = new GPUKeplerScalarSelection({
    device,
    filters: [
      {
        filterId: 'owned-hour',
        kind: 'time',
        timestamps: time.timestamps,
        validity: time.validity,
        unit,
        chartNullAsEpoch: true,
        window: {start: -3600000, end: 0, unit: 'millisecond'}
      }
    ]
  });
  filter.setTimeWindow('owned-hour', {start: -3600000, end: 0, unit: 'millisecond'});
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: time.timestamps,
    scalarFilters: [
      {
        filterId: 'owned-hour',
        mask: filter.getMask('owned-hour'),
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
  const baseline = session.registry.memoryUsage;
  const readers: ResidentTimeChartRuntime[] = [];
  const create = (changes: Partial<ResidentTimeChartPlan> = {}, budget = 8 * 1024 * 1024) => {
    const reader = new ResidentTimeChartRuntime({
      session,
      explorer,
      selection,
      workingMemoryBudgetBytes: budget,
      plan: {
        kind: 'time',
        chartId: `hours-${readers.length}`,
        datasetId: dataset.id,
        population: 'all',
        operation: 'count',
        timeColumn: 'time',
        interval: 'hour',
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: [],
        ...changes
      }
    });
    readers.push(reader);
    return reader;
  };
  try {
    const starts = instants.flatMap(instant =>
      instant !== null && Math.abs(instant) > 8640000000000000
        ? []
        : [Math.floor((instant ?? 0) / HOUR_MILLISECONDS) * HOUR_MILLISECONDS]
    );
    const ordered = [...new Set(starts)].sort((first, second) => first - second);
    const reader = create();
    const result = await reader.read();
    expect(result?.periods.map(period => [period.start, period.end])).toEqual(
      ordered.map(start => [start, start + HOUR_MILLISECONDS])
    );
    expect(result?.summaries.map(summary => summary.count)).toEqual(
      ordered.map(start => starts.filter(value => value === start).length)
    );
    expect(result?.readbackBytes).toBe(COUNT_READ_BYTES);
    expect(reader.readbackByteLength).toBe(COUNT_READ_BYTES);
    expect((await reader.read())?.periods).toEqual(result?.periods);
    expect(reader.readbackByteLength).toBe(COUNT_READ_BYTES * 2);
    const selected = await create({population: 'selected'}).read();
    // Native inclusive upper endpoints retain both the original zero and synthetic null epoch.
    expect(selected?.periods.map(period => period.start)).toEqual([-3600000, 0]);
    expect(selected?.summaries.map(summary => summary.count)).toEqual([
      2,
      unit === 'second' ? 3 : 2
    ]);
    expect(
      (
        await create({
          population: 'selected',
          excludedCpuFilterIds: ['owned-hour'],
          excludedGpuFieldNames: ['time']
        }).read()
      )?.periods
    ).toEqual(result?.periods);
    const series = await create({
      operation: 'mean',
      column: 'amount',
      series: {column: 'series', categoryCount: 2}
    }).read();
    const epochIndex = series!.periods.findIndex(period => period.start === 0);
    expect(series?.summaries[epochIndex]).toMatchObject({
      count: 3,
      value: 14,
      series: [
        {category: 'Even', count: 1, value: 5},
        {category: 'Odd', count: 2, value: 9}
      ]
    });
    expect(() => create({}, 1)).toThrow('before allocation');
    expect(time.timestamps.data.map(chunk => chunk.buffer)).toEqual(buffers);
    expect(time.timestamps.data.map(chunk => chunk.length)).toEqual([3, 5, 5]);
  } finally {
    await Promise.all(readers.map(reader => reader.destroy()));
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

test('hour charts accept 512 and 513 observed hours or days (native limit 1,000) and reject clipped endpoints', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  try {
    for (const scenario of ['512-hours', '513-hours', '513-days', 'date-end'] as const) {
      const values =
        scenario === 'date-end'
          ? [8640000000000000]
          : Array.from(
              {length: scenario === '512-hours' ? 512 : 513},
              (_, index) => index * HOUR_MILLISECONDS * (scenario === '513-days' ? 24 : 1)
            );
      const source = new arrow.Table({
        time: arrow.vectorFromArray(values, new arrow.TimestampMillisecond())
      });
      const dataset = session.importArrowDataset({
        id: scenario,
        table: source.slice(0, 1).concat(source.slice(1))
      });
      const explorer = session.openExplorer(dataset.id, {
        dimensions: [],
        workingMemoryBudgetBytes: 8 * 1024 * 1024
      });
      const baseline = session.registry.memoryUsage;
      const reader = new ResidentTimeChartRuntime({
        session,
        explorer,
        workingMemoryBudgetBytes: 8 * 1024 * 1024,
        plan: {
          kind: 'time',
          chartId: scenario,
          datasetId: dataset.id,
          population: 'all',
          operation: 'count',
          timeColumn: 'time',
          interval: 'hour',
          excludedCpuFilterIds: [],
          excludedGpuFieldNames: []
        }
      });
      try {
        if (scenario === '512-hours') {
          const result = await reader.read();
          expect(result?.periods).toHaveLength(512);
          expect(result?.summaries.every(summary => summary.count === 1)).toBe(true);
          expect(result?.readbackBytes).toBe(COUNT_READ_BYTES);
        } else if (scenario === 'date-end') {
          await expect(reader.read()).rejects.toThrow('Date boundary');
        } else {
          // Native shows up to 1,000 periods before falling back to equal-width bins.
          expect((await reader.read())?.periods).toHaveLength(513);
        }
      } finally {
        await reader.destroy();
        expect(session.registry.memoryUsage).toEqual(baseline);
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

test('hour chart cancellation and stale selections after either bounded domain read cannot publish', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'hour-stale',
    table: new arrow.Table({
      time: arrow.vectorFromArray([-1, 0, 3600000, 86400000], new arrow.TimestampMillisecond())
    })
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: dataset.value.timeColumns['time'].timestamps
  });
  let canConsume = true;
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    isLayerRegistrationPending: false,
    get canConsume() {
      return canConsume;
    },
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => scopes.encode(encoder)
  };
  const baseline = session.registry.memoryUsage;
  try {
    for (const mutation of ['cancel', 'selection', 'eligibility'] as const)
      for (const stage of [1, 2]) {
        const reader = new ResidentTimeChartRuntime({
          session,
          explorer,
          selection,
          workingMemoryBudgetBytes: 8 * 1024 * 1024,
          plan: {
            kind: 'time',
            chartId: `${mutation}-${stage}`,
            datasetId: dataset.id,
            population: 'selected',
            operation: 'count',
            timeColumn: 'time',
            interval: 'hour',
            excludedCpuFilterIds: [],
            excludedGpuFieldNames: []
          }
        });
        const readDomain = ResidentIntegerChartDomain.prototype.read;
        let reads = 0;
        const spy = vi
          .spyOn(ResidentIntegerChartDomain.prototype, 'read')
          .mockImplementation(async function (this: ResidentIntegerChartDomain) {
            const result = await readDomain.call(this);
            if (++reads === stage) {
              if (mutation === 'cancel') reader.cancel();
              else if (mutation === 'selection') scopes.invalidate();
              else canConsume = false;
            }
            return result;
          });
        try {
          expect(await reader.read()).toBeNull();
          expect(reads).toBe(stage);
          expect(reader.readbackByteLength).toBe(stage * INTEGER_DOMAIN_READ_BYTES);
          spy.mockRestore();
          canConsume = true;
          const recovered = await reader.read();
          expect(recovered?.periods).toHaveLength(4);
          expect(reader.readbackByteLength).toBe(
            stage * INTEGER_DOMAIN_READ_BYTES + COUNT_READ_BYTES
          );
        } finally {
          spy.mockRestore();
          canConsume = true;
          await reader.destroy();
          expect(session.registry.memoryUsage).toEqual(baseline);
        }
      }
  } finally {
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
