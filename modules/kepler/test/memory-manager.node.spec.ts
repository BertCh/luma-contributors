// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {buildMemoryManagerModel} from '../../../examples/deck/kepler-compute/src/memory-manager-model';
import {createMemoryManagerHost} from '../../../examples/deck/kepler-compute/src/memory-manager-host';
import {MemoryNotifier} from '../../../examples/deck/kepler-compute/src/memory-notifier';
import {ResidentEvictionHost} from '../../../examples/deck/kepler-compute/src/resident-eviction-host';

function makeRegistry() {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1000});
  const register = (id: string, gpuBytes: number) =>
    registry.registerDataset(
      {
        id,
        value: {},
        batches: [{id: `${id}-b`, rowCount: 5}],
        resources: [
          {
            id: `${id}-r`,
            value: {},
            byteLength: gpuBytes,
            memoryKind: 'gpu',
            ownership: 'owned',
            destroy() {}
          }
        ]
      },
      registry.reserveMemory(gpuBytes, 'gpu', 'source')
    );
  return {registry, register};
}

test('model attributes dataset bytes to layers and gates actions', () => {
  const {registry, register} = makeRegistry();
  register('base', 100);
  register('extra', 300);
  const model = buildMemoryManagerModel({
    registry,
    layers: [
      {id: 'l1', label: 'One', dataId: 'extra', isVisible: true},
      {id: 'l2', label: 'Two', dataId: 'extra', isVisible: false},
      {id: 'l3', label: 'Three', dataId: null, isVisible: true}
    ],
    datasetLabels: {extra: 'Extra data'},
    pinnedIds: new Set(['base']),
    evictedIds: [],
    canEvict: id => id === 'extra',
    isEvictionEnabled: true,
    layerRenderBuffers: new Map([['l1', {bytes: 4096, bufferCount: 3}]])
  });
  expect(model.usedBytes).toBe(400);
  expect(model.pressure).toBe('ok');
  const base = model.datasets.find(row => row.id === 'base')!;
  const extra = model.datasets.find(row => row.id === 'extra')!;
  expect(base).toMatchObject({canUnload: false, canEvict: false, isPinned: true});
  expect(extra).toMatchObject({
    label: 'Extra data',
    canUnload: true,
    canEvict: true,
    releasableBytes: 300,
    gpuBytes: 300,
    isVisible: true
  });
  expect(model.layers.find(row => row.id === 'l1')).toMatchObject({
    datasetBytes: 300,
    renderBufferBytes: 4096,
    renderBufferCount: 3,
    isMeasured: true,
    sharedWithLayers: 2
  });
  expect(model.layers.find(row => row.id === 'l2')).toMatchObject({
    renderBufferBytes: 0,
    isMeasured: false
  });
  expect(model.layerRenderBufferBytes).toBe(4096);
  expect(model.reservations.source).toBe(0);
});

test('unloading frees registry bytes and lowering budget is bounded by usage', () => {
  const {registry, register} = makeRegistry();
  register('extra', 300);
  expect(registry.memoryUsage.totalBytes).toBe(300);
  expect(() => registry.setMemoryBudget(200)).toThrow();
  registry.removeDataset('extra');
  expect(registry.memoryUsage.totalBytes).toBe(0);
  registry.setMemoryBudget(200);
  expect(registry.memoryBudgetBytes).toBe(200);
});

test('notifier warns once per upward crossing with hysteresis', () => {
  let totalBytes = 0;
  const notices: string[] = [];
  const notifier = new MemoryNotifier({
    getUsage: () => ({totalBytes, budgetBytes: 1000}),
    emit: notice => notices.push(`${notice.type}:${notice.key}`)
  });
  totalBytes = 700;
  notifier.check();
  totalBytes = 850;
  notifier.check();
  notifier.check();
  totalBytes = 790;
  notifier.check();
  totalBytes = 850;
  notifier.check();
  expect(notices).toEqual(['warning:memory-pressure-warning']);
  totalBytes = 960;
  notifier.check();
  expect(notices).toEqual(['warning:memory-pressure-warning', 'error:memory-pressure-critical']);
  totalBytes = 100;
  notifier.check();
  totalBytes = 900;
  notifier.check();
  expect(notices.length).toBe(3);
  notifier.reportRefusal();
  expect(notices[3]).toBe('error:memory-refused');
});

test('manual eviction requires enabled policy and a rebuild source', () => {
  const evicted: string[] = [];
  const makeHost = (enabled: boolean) =>
    new ResidentEvictionHost<never>({
      enabled,
      getVisibleDatasetIds: () => new Set(),
      canRebuild: id => id === 'a',
      evict: id => void evicted.push(id),
      rebuild: async () => {}
    });
  expect(makeHost(false).evictDataset('a')).toBe(false);
  const host = makeHost(true);
  expect(host.evictDataset('b')).toBe(false);
  expect(host.evictDataset('a')).toBe(true);
  expect(host.evictDataset('a')).toBe(false);
  expect(host.status.evictedIds).toEqual(['a']);
  expect(evicted).toEqual(['a']);
});

test('render-time reservations defer and coalesce memory panel updates', async () => {
  const {registry} = makeRegistry();
  const host = createMemoryManagerHost({
    getRegistry: () => registry,
    getLayers: () => [],
    getDatasetLabels: () => ({}),
    getPinnedIds: () => new Set<string>(),
    getEvictedIds: () => [],
    isEvictionEnabled: false,
    canEvict: () => false,
    evictDataset: () => false,
    unloadDataset() {},
    emitNotice() {},
    openManager() {}
  });
  host.bindRegistry();
  await Promise.resolve();
  let updates = 0;
  const unsubscribe = host.source.subscribe(() => updates++);
  const first = registry.reserveMemory(10, 'gpu', 'output');
  const second = registry.reserveMemory(20, 'gpu', 'scratch');
  expect(host.source.getModel()?.usedBytes).toBe(30);
  expect(updates).toBe(0);
  await Promise.resolve();
  expect(updates).toBe(1);
  first.release();
  second.release();
  unsubscribe();
  await Promise.resolve();
  expect(updates).toBe(1);
});

test('a real registry pressure crossing defers the Redux notice and emits it once', async () => {
  const {registry} = makeRegistry();
  const notices: string[] = [];
  const host = createMemoryManagerHost({
    getRegistry: () => registry,
    getLayers: () => [],
    getDatasetLabels: () => ({}),
    getPinnedIds: () => new Set<string>(),
    getEvictedIds: () => [],
    isEvictionEnabled: false,
    canEvict: () => false,
    evictDataset: () => false,
    unloadDataset() {},
    emitNotice: notice => {
      notices.push(notice.key);
    },
    openManager() {}
  });
  host.bindRegistry();
  await Promise.resolve();
  const reservation = registry.reserveMemory(850, 'gpu', 'output');
  expect(registry.memoryUsage.totalBytes).toBe(850);
  expect(notices).toEqual([]);
  host.refresh();
  expect(notices).toEqual([]);
  await Promise.resolve();
  expect(notices).toEqual(['memory-pressure-warning']);
  host.refresh();
  await Promise.resolve();
  expect(notices).toEqual(['memory-pressure-warning']);
  reservation.release();
  await Promise.resolve();
});
