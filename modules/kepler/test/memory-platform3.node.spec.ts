// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, beforeEach, expect, test, vi} from 'vitest';
import {
  createMemoryManagerHost,
  getOperationKind,
  listMemoryOperations
} from '../../../examples/deck/kepler-compute/src/memory-manager-host';
import {measureLayerRenderBuffers} from '../../../examples/deck/kepler-compute/src/memory-layer-buffers';
import {
  MEMORY_BUDGET_STORAGE_KEY,
  readStoredMemoryBudget,
  writeStoredMemoryBudget
} from '../../../examples/deck/kepler-compute/src/memory-budget-persistence';
import {listTrackedOperations} from '../../../examples/deck/kepler-compute/src/operation-tracker';
import {
  runTrackedOperation,
  trackDeferredOperation
} from '../../../examples/deck/kepler-compute/src/tracked-operations';
import {parseKeplerSandboxState, serializeKeplerSandboxState} from '../src/recipes/sandbox-state';

function makeStorage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
    removeItem: (key: string) => void values.delete(key),
    values
  };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

test('layer render buffers count owned buffers once and skip borrowed ones', () => {
  const shared = {byteLength: 100};
  const owned = {byteLength: 40};
  const borrowed = {byteLength: 999};
  const parent = {id: 'layer-a'};
  const result = measureLayerRenderBuffers(
    [
      {
        id: 'layer-a-fill',
        parent,
        state: {
          attributeManager: {
            getAttributes: () => ({
              a: {buffer: shared},
              b: {buffer: shared},
              c: {buffer: owned},
              d: {buffer: borrowed, state: {externalBuffer: borrowed}}
            })
          }
        }
      },
      {
        id: 'layer-b-sub',
        state: {attributeManager: {getAttributes: () => ({a: {buffer: {byteLength: 8}}})}}
      },
      {
        id: 'unrelated',
        state: {attributeManager: {getAttributes: () => ({a: {buffer: {byteLength: 1}}})}}
      }
    ],
    ['layer-a', 'layer-b', 'layer-c']
  );
  expect(result.get('layer-a')).toEqual({bytes: 140, bufferCount: 2});
  expect(result.get('layer-b')).toEqual({bytes: 8, bufferCount: 1});
  expect(result.get('layer-c')).toEqual({bytes: 0, bufferCount: 0});
});

test('deferred operations appear only after the delay and cancel through the tracker', () => {
  const cancel = vi.fn();
  const handle = trackDeferredOperation('Chart c1', cancel, 100);
  expect(listTrackedOperations()).toHaveLength(0);
  vi.advanceTimersByTime(100);
  expect(listTrackedOperations().map(operation => operation.label)).toEqual(['Chart c1']);
  handle.update({fraction: 0.5});
  expect(listTrackedOperations()[0].fraction).toBe(0.5);
  listTrackedOperations()[0].cancel();
  expect(cancel).toHaveBeenCalled();
  handle.finish();
  expect(listTrackedOperations()).toHaveLength(0);
  const quick = trackDeferredOperation('Analysis fast', cancel, 100);
  quick.finish();
  vi.advanceTimersByTime(500);
  expect(listTrackedOperations()).toHaveLength(0);
});

test('runTrackedOperation aborts its signal when cancelled from the manager', async () => {
  let aborted = false;
  const run = runTrackedOperation(
    'Join a with b',
    undefined,
    signal => {
      return new Promise<string>(resolve => {
        signal.addEventListener('abort', () => {
          aborted = true;
          resolve('cancelled');
        });
      });
    },
    10
  );
  vi.advanceTimersByTime(10);
  const operations = listMemoryOperations(undefined);
  expect(operations).toHaveLength(1);
  expect(operations[0]).toMatchObject({kind: 'join', label: 'Join a with b'});
  operations[0].cancel();
  expect(await run).toBe('cancelled');
  expect(aborted).toBe(true);
  expect(listTrackedOperations()).toHaveLength(0);
});

test('registry requests are listed and tracked joins hide their duplicate request', () => {
  const cancelRequest = vi.fn();
  const registry = {
    getActiveRequests: () => [
      {id: 1, slot: 'resident-join:r', inputs: [{datasetId: 'a'}], cancel: cancelRequest},
      {id: 2, slot: 'resident-group-by:g', inputs: [{datasetId: 'a'}], cancel: cancelRequest}
    ]
  };
  expect(listMemoryOperations(registry).map(operation => operation.key)).toEqual([
    'request-1',
    'request-2'
  ]);
  const handle = trackDeferredOperation('Join a with b', () => {}, 0);
  vi.advanceTimersByTime(0);
  expect(listMemoryOperations(registry).map(operation => operation.key)).toEqual([
    expect.stringMatching(/^operation-/),
    'request-2'
  ]);
  handle.finish();
  expect(getOperationKind('Stream download x')).toBe('transfer');
  expect(getOperationKind('Analysis ols')).toBe('analysis');
});

test('storage helpers never throw and validate values', () => {
  const storage = makeStorage();
  writeStoredMemoryBudget(512 * 1024 ** 2, storage);
  expect(readStoredMemoryBudget(storage)).toBe(512 * 1024 ** 2);
  storage.values.set(MEMORY_BUDGET_STORAGE_KEY, 'junk');
  expect(readStoredMemoryBudget(storage)).toBeUndefined();
  const broken = {
    getItem() {
      throw new Error('blocked');
    },
    setItem() {
      throw new Error('blocked');
    },
    removeItem() {
      throw new Error('blocked');
    }
  };
  expect(readStoredMemoryBudget(broken)).toBeUndefined();
  expect(() => writeStoredMemoryBudget(1, broken)).not.toThrow();
});

function makeHostRegistry(budget: number, used = 0) {
  const state = {budget};
  return {
    state,
    get memoryBudgetBytes() {
      return state.budget;
    },
    memoryUsage: {
      allocatedBytes: {cpu: 0, gpu: used},
      reservedBytes: {cpu: 0, gpu: 0},
      totalBytes: used,
      availableBytes: Math.max(0, budget - used)
    },
    getReservationBreakdown: () => ({source: 0, output: 0, scratch: 0, readback: 0}),
    getDatasetMemory: () => [],
    getActiveRequests: () => [],
    subscribeMemory: () => () => {},
    setMemoryBudget(next: number) {
      if (next < used) throw new Error('Budget below usage');
      state.budget = next;
    }
  };
}

function makeHost(registry: ReturnType<typeof makeHostRegistry>, storage = makeStorage()) {
  const notices: string[] = [];
  const host = createMemoryManagerHost({
    getRegistry: () => registry,
    getLayers: () => [],
    getDatasetLabels: () => ({}),
    getPinnedIds: () => new Set(),
    getEvictedIds: () => [],
    isEvictionEnabled: false,
    canEvict: () => false,
    evictDataset: () => false,
    unloadDataset() {},
    emitNotice: notice => void notices.push(notice.key),
    openManager() {},
    storage
  });
  return {host, notices, storage};
}

test('the chosen budget persists in localStorage and reapplies to a new registry', () => {
  const first = makeHostRegistry(1000);
  const {host, storage} = makeHost(first);
  expect(host.source.setBudgetMebibytes(2)).toBeNull();
  expect(storage.values.get(MEMORY_BUDGET_STORAGE_KEY)).toBe(String(2 * 1024 * 1024));
  expect(host.settings.get()).toEqual({memoryBudgetBytes: 2 * 1024 * 1024});
  const second = makeHostRegistry(1000);
  const reload = makeHost(second, storage);
  reload.host.bindRegistry();
  expect(second.state.budget).toBe(2 * 1024 * 1024);
});

test('a bundle budget applies, and one below current usage is kept with a notice', () => {
  const registry = makeHostRegistry(1000, 500);
  const {host, notices} = makeHost(registry);
  expect(host.settings.get()).toBeUndefined();
  host.settings.apply({memoryBudgetBytes: 4000});
  expect(registry.state.budget).toBe(4000);
  host.settings.apply({memoryBudgetBytes: 100});
  expect(registry.state.budget).toBe(4000);
  expect(notices).toContain('memory-budget-not-applied');
  expect(host.preferredBudgetBytes).toBe(100);
});

test('the sandbox state carries the budget setting and rejects unknown settings', () => {
  const base = {
    schemaVersion: 1 as const,
    sources: [],
    nativeConfig: {},
    selections: [],
    analyses: []
  };
  const restored = parseKeplerSandboxState(
    JSON.parse(serializeKeplerSandboxState({...base, settings: {memoryBudgetBytes: 1 << 30}}))
  );
  expect(restored.settings).toEqual({memoryBudgetBytes: 1 << 30});
  expect(parseKeplerSandboxState(base).settings).toBeUndefined();
  expect(() => parseKeplerSandboxState({...base, settings: {other: 1}})).toThrow();
  expect(() => parseKeplerSandboxState({...base, settings: {memoryBudgetBytes: -1}})).toThrow();
});
