// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import {ResidentChartProvider} from '../../../examples/deck/kepler-compute/src/resident-chart-provider';
import {ResidentTimeChartRuntime} from '../../../examples/deck/kepler-compute/src/resident-chart-time-runtime';
import type {NativeChartConfiguration} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import type {ResidentChartSelection} from '../../../examples/deck/kepler-compute/src/resident-chart-group-runtime';

test('same-ID day chart replay waits for retired readers before borrowing a selection mask', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = new arrow.Table({
    time: arrow.vectorFromArray([0, 3600000, 86400000], new arrow.TimestampMillisecond())
  });
  const dataset = session.importArrowDataset({
    id: 'calendar-replay',
    table: source.slice(0, 2).concat(source.slice(2))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const timestamps = dataset.value.timeColumns['time'].timestamps;
  const sourceBuffers = timestamps.data.map(chunk => chunk.buffer);
  const sourceBytes = await Promise.all(sourceBuffers.map(buffer => buffer.readAsync()));
  const scopes = new GPUKeplerSelectionScopes({device, rows: timestamps});
  const maskRequests = vi.spyOn(scopes, 'getConsumerMask');
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    isLayerRegistrationPending: false,
    canConsume: true,
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => scopes.encode(encoder)
  };
  const provider = new ResidentChartProvider({
    session,
    explorer,
    selection,
    subscribe: () => () => {},
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const baseline = session.registry.memoryUsage;
  const chart: NativeChartConfiguration = {
    id: 'persisted-calendar-id',
    title: 'Day chart',
    display: {},
    chartDisplay: {},
    type: 'lineChart',
    dataId: dataset.id,
    applyFilters: true,
    xAxis: {field: {name: 'time', type: 'timestamp'}, aggregation: 'timeBin', interval: 'day'},
    yAxis: {field: null, aggregation: 'count'}
  };
  let releaseRead!: () => void;
  let announceRead!: () => void;
  const readGate = new Promise<void>(resolve => {
    releaseRead = resolve;
  });
  const readReachedGate = new Promise<void>(resolve => {
    announceRead = resolve;
  });
  const readRuntime = ResidentTimeChartRuntime.prototype.read;
  const heldRead = vi
    .spyOn(ResidentTimeChartRuntime.prototype, 'read')
    .mockImplementationOnce(async function (this: ResidentTimeChartRuntime) {
      // Finish real GPU execution, but keep its provider request pending across remove/recreate.
      const result = await readRuntime.call(this);
      announceRead();
      await readGate;
      return result;
    });
  const pending: Promise<unknown>[] = [];
  try {
    const original = provider.read(chart);
    pending.push(original);
    await Promise.race([readReachedGate, original.then(() => undefined)]);
    heldRead.mockRestore();
    const oldMask = maskRequests.mock.results[0].value;
    const previousRequests = maskRequests.mock.calls.length;
    const removal = provider.remove(chart.id);
    const restored = provider.read({...chart, title: 'Restored saved day chart'});
    pending.push(removal, restored);
    // Drain the new entry's promise callbacks. It must remain behind the old reader's retirement.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(maskRequests).toHaveBeenCalledTimes(previousRequests);
    releaseRead();
    expect(await original).toBeNull();
    await removal;
    expect((await restored)?.status).toBe('ready');
    const currentMask = maskRequests.mock.results.at(-1)!.value;
    expect(currentMask).not.toBe(oldMask);

    const fence = device.createFence();
    try {
      await scopes.collectRetired(fence.signaled);
    } finally {
      fence.destroy();
    }
    // Retiring the old mask must not invalidate the restored reader's compiled GPU bindings.
    const replayed = await provider.read(chart);
    expect(replayed?.status).toBe('ready');
    if (
      replayed?.status !== 'ready' ||
      !('kind' in replayed.result) ||
      replayed.result.kind !== 'time'
    )
      throw new Error('Expected restored calendar result');
    expect(replayed.result.periods.map(period => period.label)).toEqual([
      '1970-01-01',
      '1970-01-02'
    ]);
    expect(replayed.result.summaries.map(summary => summary.count)).toEqual([2, 1]);
    expect(timestamps.data.map(chunk => chunk.length)).toEqual([2, 1]);
    expect(timestamps.data.map(chunk => chunk.buffer)).toEqual(sourceBuffers);
    expect(await Promise.all(sourceBuffers.map(buffer => buffer.readAsync()))).toEqual(sourceBytes);
  } finally {
    releaseRead();
    heldRead.mockRestore();
    await Promise.allSettled(pending);
    await provider.destroy();
    expect(session.registry.memoryUsage).toEqual(baseline);
    maskRequests.mockRestore();
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
