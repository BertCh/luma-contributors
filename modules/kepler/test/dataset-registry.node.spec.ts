// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  KeplerDatasetRegistry,
  type KeplerDatasetDescriptor,
  type KeplerDatasetResource
} from '../src/runtime/dataset-registry';

function createResource(
  id: string,
  byteLength = 64,
  ownership: 'owned' | 'borrowed' = 'owned'
): KeplerDatasetResource {
  return {id, value: {}, byteLength, memoryKind: 'gpu', ownership, destroy: vi.fn()};
}

function createDataset(
  id: string,
  resources: readonly KeplerDatasetResource[]
): KeplerDatasetDescriptor<{column: string}> {
  return {
    id,
    value: {column: 'population'},
    batches: [
      {id: 'first', rowCount: 2},
      {id: 'empty', rowCount: 0},
      {id: 'last', rowCount: 3, sourceRowOffset: 17}
    ],
    resources
  };
}

function createFence(): {
  promise: Promise<void>;
  resolve: () => void;
  reject: (error: Error) => void;
} {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return {promise, resolve, reject};
}

describe('Kepler stream population admission', () => {
  test.each([
    'open',
    'cancelled',
    'failed'
  ] as const)('%s prefixes require explicit acquisition and remain fenced on removal', async state => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const resource = createResource(`stream-${state}`);
    const population = {kind: 'stream' as const, streamId: 'stream', state};
    const snapshot = registry.registerDataset({...createDataset('prefix', [resource]), population});
    expect(snapshot.population).toEqual(population);
    expect(snapshot.population).not.toBe(population);
    expect(Object.isFrozen(snapshot.population)).toBe(true);
    expect(() => registry.acquireDataset('prefix')).toThrow('population is incomplete');
    expect(() => registry.beginRequest('analysis', ['prefix'])).toThrow('population is incomplete');
    const lease = registry.acquireDataset('prefix', {allowIncomplete: true});
    const fence = createFence();
    registry.removeDataset('prefix');
    const release = lease.release(fence.promise);
    expect(resource.destroy).not.toHaveBeenCalled();
    fence.resolve();
    await release;
    expect(resource.destroy).toHaveBeenCalledOnce();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    await registry.destroy();
  });

  test('failed complete-population admission takes no partial leases and does not supersede current work', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const resource = createResource('finite');
    registry.registerDataset(createDataset('finite', [resource]));
    registry.registerDataset({
      ...createDataset('prefix', []),
      population: {kind: 'stream', streamId: 'stream', state: 'open'}
    });
    const current = registry.beginRequest('analysis', ['finite']);
    expect(() => registry.beginRequest('analysis', ['finite', 'prefix'])).toThrow('incomplete');
    expect(current.signal.aborted).toBe(false);
    registry.removeDataset('finite');
    await registry.finishRequest(current);
    expect(resource.destroy).toHaveBeenCalledOnce();
    await registry.destroy();
  });

  test('explicit preview requests cannot publish an unmarked or falsely complete result', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    registry.registerDataset({
      ...createDataset('prefix', []),
      population: {kind: 'stream', streamId: 'stream', state: 'open'}
    });
    const request = registry.beginRequest('preview', ['prefix'], {allowIncomplete: true});
    const result = createDataset('result', []);
    expect(registry.publishResult(request, result)).toBeNull();
    expect(
      registry.publishResult(request, {
        ...result,
        population: {kind: 'stream', streamId: 'derived', state: 'complete'}
      })
    ).toBeNull();
    expect(registry.getDataset('result')).toBeUndefined();
    expect(
      registry.publishResult(request, {
        ...result,
        population: {kind: 'stream', streamId: 'derived', state: 'open'}
      })?.population?.state
    ).toBe('open');
    expect(() => registry.acquireDataset('result')).toThrow('incomplete');
    await registry.finishRequest(request);
    await registry.destroy();
  });

  test('only a new completed stream revision admits ordinary population consumers', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const descriptor = createDataset('streamed', []);
    registry.registerDataset({
      ...descriptor,
      population: {kind: 'stream', streamId: 'stream', state: 'open'}
    });
    const preview = registry.beginRequest('preview', ['streamed'], {allowIncomplete: true});
    const completed = registry.registerDataset({
      ...descriptor,
      population: {kind: 'stream', streamId: 'stream', state: 'complete'}
    });
    expect(preview.signal.aborted).toBe(true);
    const lease = registry.acquireDataset('streamed');
    expect(lease.dataset).toBe(completed);
    const analysis = registry.beginRequest('analysis', ['streamed']);
    expect(analysis.inputs).toEqual([{datasetId: 'streamed', revision: 2}]);
    await lease.release();
    await registry.finishRequest(preview);
    await registry.finishRequest(analysis);
    await registry.destroy();
  });

  test('device recovery preserves the unfinished population marker and retained CPU ownership', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const source = {...createResource('retained-prefix'), memoryKind: 'cpu' as const};
    registry.registerDataset({
      ...createDataset('prefix', [source]),
      source: {kind: 'arrow', value: {}},
      population: {kind: 'stream', streamId: 'stream', state: 'failed'}
    });
    registry.handleDeviceLoss();
    const recovery = registry.getRecoverySources()[0];
    expect(recovery.population).toEqual({kind: 'stream', streamId: 'stream', state: 'failed'});
    expect(source.destroy).not.toHaveBeenCalled();
    registry.recoverDevice();
    registry.registerDataset({
      ...createDataset('prefix', recovery.resources),
      source: recovery.source,
      population: recovery.population
    });
    expect(() => registry.acquireDataset('prefix')).toThrow('incomplete');
    registry.removeDataset('prefix');
    expect(source.destroy).toHaveBeenCalledOnce();
    await registry.destroy();
  });
});

describe('Kepler dataset registry ownership and identity', () => {
  test('counts shared allocations once and releases owned resources after the last dataset and lease', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const resource = createResource('shared');
    registry.registerDataset(createDataset('points', [resource, resource]));
    registry.registerDataset(createDataset('selection', [resource]));
    expect(registry.memoryUsage.totalBytes).toBe(64);
    const firstLease = registry.acquireDataset('points');
    const secondLease = registry.acquireDataset('selection');
    registry.removeDataset('points');
    registry.removeDataset('selection');
    expect(resource.destroy).not.toHaveBeenCalled();
    await firstLease.release();
    expect(resource.destroy).not.toHaveBeenCalled();
    await secondLease.release();
    await secondLease.release();
    expect(resource.destroy).toHaveBeenCalledTimes(1);
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });

  test('never calls a borrowed resource destructor, including device loss and registry destruction', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const borrowed = createResource('borrowed', 64, 'borrowed');
    registry.registerDataset(createDataset('points', [borrowed]));
    const lease = registry.acquireDataset('points');
    registry.handleDeviceLoss();
    expect(lease.isValid()).toBe(false);
    await lease.release();
    await registry.destroy();
    expect(borrowed.destroy).not.toHaveBeenCalled();
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });

  test('retains a replaced revision until its rendering fence settles', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 192});
    const original = createResource('original');
    const updated = createResource('updated');
    const first = registry.registerDataset(createDataset('points', [original]));
    const lease = registry.acquireDataset('points');
    const second = registry.registerDataset(createDataset('points', [updated]));
    expect(first.revision).toBe(1);
    expect(second.revision).toBe(2);
    expect(lease.dataset).toBe(first);
    expect(lease.isValid()).toBe(true);
    const fence = createFence();
    const release = lease.release(fence.promise);
    expect(lease.isValid()).toBe(false);
    expect(original.destroy).not.toHaveBeenCalled();
    expect(registry.memoryUsage.totalBytes).toBe(128);
    fence.resolve();
    await release;
    expect(original.destroy).toHaveBeenCalledOnce();
    registry.removeDataset('points');
    expect(registry.registerDataset(createDataset('points', [])).revision).toBe(3);
  });

  test('cleans up after a rejected completion fence without swallowing the queue error', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64});
    const resource = createResource('points');
    registry.registerDataset(createDataset('points', [resource]));
    const lease = registry.acquireDataset('points');
    registry.removeDataset('points');
    const fence = createFence();
    const completion = lease.release(fence.promise);
    fence.reject(new Error('device submission failed'));
    await expect(completion).rejects.toThrow('device submission failed');
    expect(resource.destroy).toHaveBeenCalledOnce();
    expect(lease.release()).toBe(completion);
  });

  test('preserves batch identities and explicit source lineage independently of resident positions', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 0});
    const dataset = registry.registerDataset(createDataset('points', []));
    expect(dataset.rowCount).toBe(5);
    expect(dataset.batches.map(batch => batch.rowCount)).toEqual([2, 0, 3]);
    expect(registry.resolveSourceRow(dataset, 2)).toEqual({
      datasetId: 'points',
      revision: 1,
      batchId: 'last',
      sourceRowIndex: 17
    });
    const result = registry.registerDataset({
      ...createDataset('result', []),
      batches: [
        {
          id: 'projection',
          rowCount: 2,
          sourceRowOffset: 18,
          source: {datasetId: 'points', revision: 1, batchId: 'last'}
        }
      ]
    });
    expect(registry.resolveSourceRow(result, 1)).toEqual({
      datasetId: 'points',
      revision: 1,
      batchId: 'last',
      sourceRowIndex: 19
    });
    const indirect = registry.registerDataset({
      ...createDataset('sorted', []),
      rowIdentity: 'indirect'
    });
    expect(() => registry.resolveSourceRow(indirect, 0)).toThrow('explicit mapping');
    expect(() => registry.resolveSourceRow(dataset, 5)).toThrow('out of range');
    expect(() => registry.resolveSourceRow(dataset, -1)).toThrow();
  });
});

describe('Kepler dataset registry memory transactions', () => {
  test('reserves allocation capacity, adopts it without double counting and prevents oversubscription', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const reservation = registry.reserveMemory(96);
    expect(registry.memoryUsage.availableBytes).toBe(32);
    expect(() => registry.reserveMemory(64)).toThrow('budget');
    registry.registerDataset(createDataset('points', [createResource('first', 64)]), reservation);
    expect(registry.memoryUsage).toEqual({
      allocatedBytes: {cpu: 0, gpu: 64},
      reservedBytes: {cpu: 0, gpu: 0},
      totalBytes: 64,
      availableBytes: 64
    });
    reservation.release();
    expect(() => registry.registerDataset(createDataset('other', []), reservation)).toThrow(
      'Inactive'
    );
  });

  test('reservation resize admits growth atomically, releases shrink capacity and cannot revive a released charge', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const graphics = registry.reserveMemory(64);
    expect(graphics.isActive).toBe(true);
    const source = registry.reserveMemory(32, 'cpu');
    graphics.resize(96);
    expect(graphics.byteLength).toBe(96);
    expect(registry.memoryUsage.reservedBytes).toEqual({gpu: 96, cpu: 32});
    expect(() => graphics.resize(97)).toThrow('budget');
    expect(graphics.byteLength).toBe(96);
    expect(registry.memoryUsage.totalBytes).toBe(128);
    for (const size of [-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])
      expect(() => graphics.resize(size)).toThrow();
    expect(registry.memoryUsage.totalBytes).toBe(128);
    graphics.resize(16);
    source.resize(64);
    expect(registry.memoryUsage.reservedBytes).toEqual({gpu: 16, cpu: 64});
    graphics.resize(0);
    graphics.resize(64);
    expect(registry.memoryUsage.totalBytes).toBe(128);
    graphics.release();
    graphics.release();
    expect(graphics.isActive).toBe(false);
    expect(registry.memoryUsage.totalBytes).toBe(64);
    expect(() => graphics.resize(0)).toThrow('Inactive');
    expect(() => graphics.resize(64)).toThrow('Inactive');
    source.release();
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });

  test('resized reservations adopt their current charge and device loss prevents later growth', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const reservation = registry.reserveMemory(96);
    reservation.resize(32);
    expect(() =>
      registry.registerDataset(createDataset('large', [createResource('large', 64)]), reservation)
    ).toThrow('Reservation exceeded');
    expect(reservation.byteLength).toBe(32);
    registry.registerDataset(createDataset('small', [createResource('small', 32)]), reservation);
    expect(registry.memoryUsage).toMatchObject({
      allocatedBytes: {gpu: 32},
      reservedBytes: {gpu: 0}
    });
    expect(() => reservation.resize(1)).toThrow('Inactive');
    const next = registry.reserveMemory(64);
    next.resize(16);
    registry.handleDeviceLoss();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    expect(() => next.resize(64)).toThrow('Inactive');
  });

  test('a rejected replacement preserves the previous dataset, reservation and caller-owned output', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const original = createResource('original');
    const previous = registry.registerDataset(createDataset('points', [original]));
    const reservation = registry.reserveMemory(64);
    const output = createResource('output', 96);
    expect(() => registry.registerDataset(createDataset('points', [output]), reservation)).toThrow(
      'Reservation exceeded'
    );
    expect(registry.getDataset('points')).toBe(previous);
    expect(registry.memoryUsage.totalBytes).toBe(128);
    expect(original.destroy).not.toHaveBeenCalled();
    expect(output.destroy).not.toHaveBeenCalled();
    reservation.release();
    expect(registry.memoryUsage.totalBytes).toBe(64);
  });

  test('rejects double-accounted aliases and ownership conflicts without replacing live resources', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 256});
    const resource = createResource('original');
    registry.registerDataset(createDataset('points', [resource]));
    expect(() => registry.registerDataset(createDataset('duplicate-id', [{...resource}]))).toThrow(
      'one descriptor'
    );
    expect(() =>
      registry.registerDataset(createDataset('duplicate-value', [{...resource, id: 'alias'}]))
    ).toThrow('one descriptor');
    expect(() =>
      registry.registerDataset(
        createDataset('conflicting-owner', [{...resource, ownership: 'borrowed'}])
      )
    ).toThrow('one descriptor');
    const fresh = createResource('fresh');
    expect(() =>
      registry.registerDataset(createDataset('local-alias', [fresh, {...fresh, id: 'another'}]))
    ).toThrow('one descriptor');
    expect(registry.memoryUsage.totalBytes).toBe(64);
    expect(resource.destroy).not.toHaveBeenCalled();
  });

  test('accounts CPU sources alongside GPU buffers and validates batch metadata before adoption', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const source = {...createResource('source', 32, 'borrowed'), memoryKind: 'cpu' as const};
    registry.registerDataset(createDataset('points', [source, createResource('values', 64)]));
    expect(registry.memoryUsage.allocatedBytes).toEqual({cpu: 32, gpu: 64});
    const reservation = registry.reserveMemory(16, 'cpu');
    expect(() =>
      registry.registerDataset(
        createDataset('wrong-kind', [createResource('gpu', 16)]),
        reservation
      )
    ).toThrow('memory kind');
    expect(() =>
      registry.registerDataset({
        ...createDataset('bad-batches', []),
        batches: [
          {id: 'same', rowCount: 1},
          {id: 'same', rowCount: 1}
        ]
      })
    ).toThrow('batch');
    expect(() =>
      registry.registerDataset({
        ...createDataset('bad-size', []),
        batches: [{id: 'first', rowCount: Number.MAX_SAFE_INTEGER + 1}]
      })
    ).toThrow('safe integer');
    reservation.release();
  });
});

describe('Kepler atomic mixed memory reservations', () => {
  test('adopts CPU and GPU charges together while old generations remain fenced', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const previousCPU = {...createResource('previous-cpu', 16), memoryKind: 'cpu' as const};
    const previousGPU = createResource('previous-gpu', 16);
    registry.registerDataset(createDataset('result', [previousCPU, previousGPU]));
    const lease = registry.acquireDataset('result');
    const fence = createFence();
    const cpu = registry.reserveMemory(32, 'cpu');
    const gpu = registry.reserveMemory(64, 'gpu');
    const currentCPU = {...createResource('current-cpu', 32), memoryKind: 'cpu' as const};
    const currentGPU = createResource('current-gpu', 64);
    const result = registry.registerDataset(createDataset('result', [currentCPU, currentGPU]), [
      cpu,
      gpu
    ]);
    expect(result.revision).toBe(2);
    expect(registry.memoryUsage.allocatedBytes).toEqual({cpu: 48, gpu: 80});
    expect(registry.memoryUsage.reservedBytes).toEqual({cpu: 0, gpu: 0});
    expect(cpu.isActive || gpu.isActive).toBe(false);
    const releasing = lease.release(fence.promise);
    expect(previousGPU.destroy).not.toHaveBeenCalled();
    fence.resolve();
    await releasing;
    expect(registry.memoryUsage.allocatedBytes).toEqual({cpu: 32, gpu: 64});
    expect(previousCPU.destroy).toHaveBeenCalledOnce();
    expect(previousGPU.destroy).toHaveBeenCalledOnce();
    await registry.destroy();
  });

  test.each([
    'duplicate',
    'foreign',
    'released',
    'cpu-capacity',
    'gpu-capacity',
    'metadata'
  ])('rejects %s without consuming either reservation or result ownership', async failure => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 256});
    const foreignRegistry = new KeplerDatasetRegistry({memoryBudgetBytes: 256});
    const cpu = registry.reserveMemory(32, 'cpu');
    const gpu = registry.reserveMemory(64, 'gpu');
    const foreign = foreignRegistry.reserveMemory(32, 'cpu');
    if (failure === 'released') cpu.release();
    const cpuOutput = {
      ...createResource('output-cpu', failure === 'cpu-capacity' ? 33 : 32),
      memoryKind: 'cpu' as const
    };
    const gpuOutput = createResource('output-gpu', failure === 'gpu-capacity' ? 65 : 64);
    const descriptor = createDataset('output', [cpuOutput, gpuOutput]);
    const before = registry.memoryUsage;
    expect(() =>
      registry.registerDataset(
        failure === 'metadata' ? {...descriptor, batches: [{id: '', rowCount: 1}]} : descriptor,
        failure === 'duplicate'
          ? [cpu, gpu, cpu]
          : failure === 'foreign'
            ? [foreign, gpu]
            : [cpu, gpu]
      )
    ).toThrow();
    expect(registry.memoryUsage).toEqual(before);
    expect(registry.getDataset('output')).toBeUndefined();
    expect(gpu.isActive).toBe(true);
    expect(foreign.isActive).toBe(true);
    expect(cpuOutput.destroy).not.toHaveBeenCalled();
    expect(gpuOutput.destroy).not.toHaveBeenCalled();
    cpu.release();
    gpu.release();
    foreign.release();
    await registry.destroy();
    await foreignRegistry.destroy();
  });

  test('stale publication leaves both kinds reserved until caller cleanup', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    registry.registerDataset(createDataset('source', []));
    const stale = registry.beginRequest('group-by', ['source']);
    const current = registry.beginRequest('group-by', ['source']);
    const cpu = registry.reserveMemory(32, 'cpu');
    const gpu = registry.reserveMemory(64, 'gpu');
    const cpuOutput = {...createResource('output-cpu', 32), memoryKind: 'cpu' as const};
    const gpuOutput = createResource('output-gpu', 64);
    const before = registry.memoryUsage;
    expect(
      registry.publishResult(stale, createDataset('result', [cpuOutput, gpuOutput]), [cpu, gpu])
    ).toBeNull();
    expect(registry.memoryUsage).toEqual(before);
    expect(cpu.isActive && gpu.isActive).toBe(true);
    expect(registry.getDataset('result')).toBeUndefined();
    expect(gpuOutput.destroy).not.toHaveBeenCalled();
    cpu.release();
    gpu.release();
    await Promise.all([registry.finishRequest(stale), registry.finishRequest(current)]);
    await registry.destroy();
  });
});

describe('Kepler asynchronous request publication and recovery', () => {
  test('registry destruction waits for the supplied fence and is idempotent while pending', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64});
    const resource = createResource('input');
    registry.registerDataset(createDataset('points', [resource]));
    const fence = createFence();
    const completion = registry.destroy(fence.promise);
    expect(resource.destroy).not.toHaveBeenCalled();
    expect(registry.destroy()).toBe(completion);
    expect(registry.memoryUsage.totalBytes).toBe(64);
    fence.resolve();
    await completion;
    expect(resource.destroy).toHaveBeenCalledOnce();
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });
  test('an empty-input request still waits for its submitted work', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 0});
    const request = registry.beginRequest('generate', []);
    const fence = createFence();
    const finished = vi.fn();
    const completion = registry.finishRequest(request, fence.promise).then(finished);
    await Promise.resolve();
    await Promise.resolve();
    expect(finished).not.toHaveBeenCalled();
    fence.resolve();
    await completion;
    expect(finished).toHaveBeenCalledOnce();
  });

  test('retains and accounts owned CPU source allocations through device restoration', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const sourceResource = {...createResource('source', 32), memoryKind: 'cpu' as const};
    const gpuResource = createResource('first-device', 64);
    const source = {kind: 'arrow' as const, value: sourceResource.value};
    registry.registerDataset({...createDataset('points', [sourceResource, gpuResource]), source});
    registry.handleDeviceLoss();
    expect(gpuResource.destroy).toHaveBeenCalledOnce();
    expect(sourceResource.destroy).not.toHaveBeenCalled();
    expect(registry.memoryUsage.totalBytes).toBe(32);
    expect(registry.getRecoverySources()[0]?.resources).toEqual([sourceResource]);
    registry.recoverDevice();
    registry.registerDataset({
      ...createDataset('points', [sourceResource, createResource('second-device', 64)]),
      source
    });
    expect(sourceResource.destroy).not.toHaveBeenCalled();
    expect(registry.memoryUsage.totalBytes).toBe(96);
    registry.handleDeviceLoss();
    expect(registry.removeDataset('points')).toBe(true);
    expect(sourceResource.destroy).toHaveBeenCalledOnce();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    await registry.destroy();
  });

  test('reports cleanup exceptions without partially disposing other resources or failing an adopted replacement', () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 192});
    const failure = new Error('external cleanup failed');
    const first = {
      ...createResource('first'),
      destroy: vi.fn<() => void>(() => {
        throw failure;
      })
    };
    const second = createResource('second');
    registry.registerDataset(createDataset('points', [first, second]));
    expect(registry.registerDataset(createDataset('points', []))?.revision).toBe(2);
    expect(second.destroy).toHaveBeenCalledOnce();
    expect(registry.takeCleanupErrors()).toEqual([{resourceId: 'first', error: failure}]);
    expect(registry.takeCleanupErrors()).toEqual([]);
    expect(registry.memoryUsage.totalBytes).toBe(64);
    expect(() => registry.registerDataset(createDataset('retry', [first]))).toThrow(
      'cleanup unresolved'
    );
    first.destroy.mockImplementation(() => undefined);
    expect(registry.retryResourceCleanup('first')).toBe(true);
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });

  test('superseded work cannot publish but retains inputs until explicitly finished', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    const input = createResource('input');
    registry.registerDataset(createDataset('points', [input]));
    const stale = registry.beginRequest('histogram', ['points', 'points']);
    const current = registry.beginRequest('histogram', ['points']);
    expect(stale.signal.aborted).toBe(true);
    expect(stale.inputs).toHaveLength(1);
    const rejectedOutput = createResource('rejected');
    expect(registry.publishResult(stale, createDataset('histogram', [rejectedOutput]))).toBeNull();
    expect(rejectedOutput.destroy).not.toHaveBeenCalled();
    expect(registry.publishResult(current, createDataset('histogram', []))?.revision).toBe(1);
    expect(registry.publishResult(current, createDataset('histogram', []))).toBeNull();
    registry.removeDataset('points');
    const fence = createFence();
    const completion = registry.finishRequest(stale, fence.promise);
    await registry.finishRequest(current);
    expect(input.destroy).not.toHaveBeenCalled();
    expect(registry.finishRequest(stale)).toBe(completion);
    fence.resolve();
    await completion;
    expect(input.destroy).toHaveBeenCalledOnce();
  });

  test('changed dataset revision aborts dependent operations without affecting unrelated slots', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 128});
    registry.registerDataset(createDataset('points', []));
    registry.registerDataset(createDataset('areas', []));
    const pointsRequest = registry.beginRequest('point-summary', ['points']);
    const areaRequest = registry.beginRequest('area-summary', ['areas']);
    registry.registerDataset(createDataset('points', []));
    expect(pointsRequest.signal.aborted).toBe(true);
    expect(areaRequest.signal.aborted).toBe(false);
    expect(registry.publishResult(pointsRequest, createDataset('stale', []))).toBeNull();
    expect(registry.publishResult(areaRequest, createDataset('current', []))).not.toBeNull();
    await registry.destroy();
  });

  test('failed request preparation does not supersede valid work or leak leases', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64});
    const input = createResource('input');
    registry.registerDataset(createDataset('points', [input]));
    const request = registry.beginRequest('summary', ['points']);
    expect(() => registry.beginRequest('summary', ['points', 'missing'])).toThrow(
      'Unknown dataset'
    );
    expect(request.signal.aborted).toBe(false);
    await registry.finishRequest(request);
    registry.removeDataset('points');
    expect(input.destroy).toHaveBeenCalledOnce();
  });

  test('device loss invalidates GPU handles and preserves only explicit recovery information', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 256});
    const source = {
      kind: 'arrow' as const,
      value: {batches: 'retained source'},
      metadata: {name: 'observations'}
    };
    const original = createResource('original');
    registry.registerDataset({...createDataset('points', [original]), source});
    const request = registry.beginRequest('summary', ['points']);
    const lease = registry.acquireDataset('points');
    const reservation = registry.reserveMemory(32);
    registry.handleDeviceLoss();
    expect(request.signal.aborted).toBe(true);
    expect(registry.getDataset('points')).toBeUndefined();
    expect(lease.isValid()).toBe(false);
    expect(registry.deviceEpoch).toBe(2);
    expect(registry.getRecoverySources()[0]).toMatchObject({id: 'points', revision: 1, source});
    expect('value' in registry.getRecoverySources()[0]!).toBe(false);
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(0);
    expect(() => registry.registerDataset(createDataset('points', []))).toThrow('Device lost');
    registry.recoverDevice();
    const restored = registry.registerDataset({
      ...createDataset('points', [createResource('restored')]),
      source
    });
    expect(restored.revision).toBe(2);
    expect(restored.deviceEpoch).toBe(2);
    expect(registry.getRecoverySources()).toEqual([]);
    expect(registry.publishResult(request, createDataset('stale', []))).toBeNull();
    await lease.release();
    await registry.finishRequest(request);
    reservation.release();
    expect(original.destroy).toHaveBeenCalledOnce();
    await registry.destroy();
    expect(registry.memoryUsage.totalBytes).toBe(0);
    expect(() => registry.recoverDevice()).toThrow('destroyed');
  });
});
