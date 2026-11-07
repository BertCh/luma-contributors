// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {KeplerDatasetRegistry, type KeplerDatasetResource} from '../src/runtime/dataset-registry';
import {
  ResidentEvictionHost,
  collectVisibleDatasetIds
} from '../../../examples/deck/kepler-compute/src/resident-eviction-host';

function createOwnedResource(id: string, byteLength: number): KeplerDatasetResource {
  return {id, value: {id}, byteLength, memoryKind: 'gpu', ownership: 'owned', destroy() {}};
}

function registerTable(registry: KeplerDatasetRegistry, id: string, byteLength: number): void {
  registry.registerDataset({
    id,
    value: {},
    batches: [{id: `${id}-batch`, rowCount: 1}],
    resources: [createOwnedResource(`${id}-resource`, byteLength)]
  });
}

async function flushMicrotasks(): Promise<void> {
  for (let tick = 0; tick < 10; tick++) await Promise.resolve();
}

function createHost(props: {
  enabled: boolean;
  registry: KeplerDatasetRegistry;
  visible: Set<string>;
  rebuilt?: string[];
}) {
  const evicted: string[] = [];
  const host = new ResidentEvictionHost({
    enabled: props.enabled,
    getVisibleDatasetIds: () => props.visible,
    canRebuild: id => id !== 'unrebuildable',
    evict(id) {
      evicted.push(id);
      props.registry.removeDataset(id);
    },
    async rebuild(id) {
      props.rebuilt?.push(id);
    }
  });
  host.attach(props.registry);
  return {host, evicted};
}

describe('collectVisibleDatasetIds', () => {
  it('counts visible layers, filter targets and pins but not hidden layers', () => {
    const visible = collectVisibleDatasetIds({
      layers: [
        {isVisible: true, dataId: 'a'},
        {isVisible: false, dataId: 'b'}
      ],
      filters: [{dataId: ['c', 'd']}, {dataId: 'e'}],
      pinnedIds: ['pinned']
    });
    expect([...visible].sort()).toEqual(['a', 'c', 'd', 'e', 'pinned']);
  });
});

describe('ResidentEvictionHost', () => {
  it('leaves the registry untouched when disabled', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registerTable(registry, 'hidden', 80);
    const {evicted} = createHost({enabled: false, registry, visible: new Set()});
    expect(() => registry.reserveMemory(50)).toThrow();
    expect(evicted).toEqual([]);
  });

  it('evicts the least recently used hidden table when a reservation exceeds the budget', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registerTable(registry, 'old', 40);
    registerTable(registry, 'recent', 40);
    const visible = new Set<string>();
    const {host, evicted} = createHost({enabled: true, registry, visible});
    host.touch('old');
    host.touch('recent');
    const reservation = registry.reserveMemory(50);
    expect(reservation.isActive).toBe(true);
    expect(evicted).toEqual(['old']);
    expect(host.status.evictedIds).toEqual(['old']);
    expect(registry.getDataset('recent')).toBeDefined();
  });

  it('never evicts visible or unrebuildable tables and surfaces the budget error', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registerTable(registry, 'shown', 40);
    registerTable(registry, 'unrebuildable', 40);
    const {evicted} = createHost({enabled: true, registry, visible: new Set(['shown'])});
    expect(() => registry.reserveMemory(50)).toThrow();
    expect(evicted).toEqual([]);
  });

  it('does not count borrowed resources as freed bytes', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registry.registerDataset({
      id: 'borrowed',
      value: {},
      batches: [{id: 'batch', rowCount: 1}],
      resources: [{...createOwnedResource('borrowed-resource', 80), ownership: 'borrowed'}]
    });
    expect(registry.getDatasetFootprints()).toEqual([{id: 'borrowed', ownedByteLength: 0}]);
  });

  it('rebuilds an evicted table once on demand and reports status', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registerTable(registry, 'old', 80);
    const rebuilt: string[] = [];
    const {host} = createHost({enabled: true, registry, visible: new Set(), rebuilt});
    registry.reserveMemory(50);
    expect(host.isEvicted('old')).toBe(true);
    host.requestResident('old');
    host.requestResident('old');
    expect(host.status.rebuildingIds).toEqual(['old']);
    await flushMicrotasks();
    expect(rebuilt).toEqual(['old']);
    expect(host.isEvicted('old')).toBe(false);
    expect(host.status).toMatchObject({rebuildCount: 1, evictedIds: [], rebuildingIds: []});
  });

  it('records a failed rebuild once and retries only on request', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registerTable(registry, 'old', 80);
    let attempts = 0;
    const host = new ResidentEvictionHost({
      enabled: true,
      getVisibleDatasetIds: () => new Set<string>(),
      canRebuild: () => true,
      evict: id => void registry.removeDataset(id),
      async rebuild() {
        if (++attempts === 1) throw new Error('upload failed');
      }
    });
    host.attach(registry);
    registry.reserveMemory(50);
    host.requestResident('old');
    await flushMicrotasks();
    host.requestResident('old');
    await flushMicrotasks();
    expect(attempts).toBe(1);
    expect(host.status).toMatchObject({failedIds: ['old'], lastError: 'upload failed'});
    host.retryFailed();
    await flushMicrotasks();
    expect(attempts).toBe(2);
    expect(host.status).toMatchObject({failedIds: [], evictedIds: []});
  });

  it('restores default reservation after detach', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registerTable(registry, 'old', 80);
    const {host, evicted} = createHost({enabled: true, registry, visible: new Set()});
    host.detach();
    expect(() => registry.reserveMemory(50)).toThrow();
    expect(evicted).toEqual([]);
  });
});
