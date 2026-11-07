// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import type {KeplerComputeSession, KeplerSessionExplorer} from '@luma-contributors/kepler';
import {ResidentChartProvider} from '../../../examples/deck/kepler-compute/src/resident-chart-provider';
import type {NativeChartConfiguration} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const runtime = vi.hoisted(() => {
  // The node project shares its module cache. Other host tests can import this provider first.
  vi.resetModules();
  return {
    created: 0,
    destroyed: 0,
    pending: [] as ((result: unknown) => void)[],
    retirement: undefined as Promise<void> | undefined
  };
});
vi.mock('../../../examples/deck/kepler-compute/src/resident-chart-runtime', () => ({
  ResidentScalarChartRuntime: class {
    constructor() {
      runtime.created++;
    }
    read() {
      return new Promise(resolve => runtime.pending.push(resolve));
    }
    cancel() {}
    async destroy() {
      await runtime.retirement;
      runtime.destroyed++;
    }
  }
}));

const chart: NativeChartConfiguration = {
  id: 'summary',
  type: 'bigNumber',
  title: 'Count',
  dataId: 'source',
  applyFilters: true,
  axis: {field: null, aggregation: 'count'}
};
function fixture() {
  runtime.created = 0;
  runtime.destroyed = 0;
  runtime.pending = [];
  runtime.retirement = undefined;
  const explorer = {
    dataset: {
      id: 'source',
      revision: 1,
      value: {columns: {value: {format: 'float32'}}, timeColumns: {}, prepared: {dictionaries: {}}}
    },
    currentRevision: 2,
    isCurrent: true
  } as unknown as KeplerSessionExplorer;
  return new ResidentChartProvider({
    session: {} as KeplerComputeSession,
    explorer,
    subscribe: () => () => {}
  });
}
const value = {value: 7, sourceRevision: 1, selectionRevision: 2, hasValue: true, readbackBytes: 4};

test('chart removal waits for an active read and suppresses its obsolete result', async () => {
  const provider = fixture();
  const pending = provider.read(chart);
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(1));
  let completed = false;
  const removal = provider.remove(chart.id).then(() => {
    completed = true;
  });
  await Promise.resolve();
  expect(completed).toBe(false);
  runtime.pending[0](value);
  expect(await pending).toBeNull();
  await removal;
  expect(runtime.destroyed).toBe(1);
  await provider.destroy();
});

test('title-only updates reuse the reader and an unsupported configuration retires it', async () => {
  const provider = fixture();
  const first = provider.read(chart);
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(1));
  runtime.pending[0](value);
  expect(await first).toEqual({status: 'ready', result: value});
  const renamed = provider.read({...chart, title: 'Renamed'});
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(2));
  runtime.pending[1](value);
  expect(await renamed).toEqual({status: 'ready', result: value});
  expect(runtime.created).toBe(1);
  expect(
    await provider.read({
      ...chart,
      axis: {field: {name: 'value', type: 'real'}, aggregation: 'variance'}
    })
  ).toMatchObject({status: 'unsupported'});
  await provider.destroy();
  expect(runtime.destroyed).toBe(1);
  expect(provider.getRevision()).toContain('retired');
});

test('destroy cancels queued charts before allocation and blocks later requests', async () => {
  const provider = fixture();
  const pending = provider.read(chart);
  await provider.destroy();
  expect(await pending).toBeNull();
  expect(runtime.created).toBe(0);
  expect(await provider.read(chart)).toBeNull();
});

test('saved-chart remount waits for the previous read and fenced mask retirement', async () => {
  const provider = fixture();
  let finishRetirement!: () => void;
  runtime.retirement = new Promise(resolve => {
    finishRetirement = resolve;
  });
  const previous = provider.read(chart);
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(1));
  const removal = provider.remove(chart.id);
  const restored = provider.read(chart);
  await Promise.resolve();
  expect(runtime.created).toBe(1);
  runtime.pending[0](value);
  expect(await previous).toBeNull();
  await Promise.resolve();
  expect(runtime.created).toBe(1);
  expect(runtime.destroyed).toBe(0);
  finishRetirement();
  await removal;
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(2));
  expect(runtime.destroyed).toBe(1);
  expect(runtime.created).toBe(2);
  runtime.pending[1](value);
  expect(await restored).toEqual({status: 'ready', result: value});
  await provider.destroy();
});

test('repeated remount cleanup stays behind the original retirement', async () => {
  const provider = fixture();
  const previous = provider.read(chart);
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(1));
  const firstRemoval = provider.remove(chart.id);
  const interruptedRestore = provider.read(chart);
  const secondRemoval = provider.remove(chart.id);
  const restored = provider.read(chart);
  await Promise.resolve();
  expect(runtime.created).toBe(1);
  runtime.pending[0](value);
  expect(await previous).toBeNull();
  expect(await interruptedRestore).toBeNull();
  await Promise.all([firstRemoval, secondRemoval]);
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(2));
  expect(runtime.destroyed).toBe(1);
  expect(runtime.created).toBe(2);
  runtime.pending[1](value);
  expect(await restored).toEqual({status: 'ready', result: value});
  await provider.destroy();
});

test('retirement leaves other charts independent and provider disposal cancels replacement', async () => {
  const provider = fixture();
  let finishRetirement!: () => void;
  runtime.retirement = new Promise(resolve => {
    finishRetirement = resolve;
  });
  const previous = provider.read(chart);
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(1));
  const removal = provider.remove(chart.id);
  const restored = provider.read(chart);
  const independent = provider.read({...chart, id: 'another-chart'});
  await vi.waitFor(() => expect(runtime.pending).toHaveLength(2));
  expect(runtime.created).toBe(2);
  runtime.pending[1](value);
  expect(await independent).toEqual({status: 'ready', result: value});
  const destruction = provider.destroy();
  runtime.pending[0](value);
  expect(await previous).toBeNull();
  finishRetirement();
  await removal;
  expect(await restored).toBeNull();
  await destruction;
  expect(runtime.created).toBe(2);
  expect(runtime.destroyed).toBe(2);
});
