// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/graph/arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  KeplerComputeSession,
  KeplerSessionExplorer,
  getPreparedArrowUploadByteLength,
  getRetainedArrowArrayBufferBytes
} from '../../src/session/compute-session';
import type {GPUResidentExplorer} from '../../src/exploration/gpu-resident-explorer';

test('session counts source backing allocations once and predicts nullable chunk upload bytes', () => {
  const values = new Float32Array([1, 2, 3, 4]);
  const source = new arrow.Table({
    first: arrow.makeVector(values),
    second: arrow.makeVector(values)
  }).slice(1, 3);
  const prepared = prepareArrowTableForGPUAnalytics(source);
  expect(getRetainedArrowArrayBufferBytes(source, prepared)).toBe(values.buffer.byteLength);
  // makeVector marks these typed-array columns non-nullable.
  expect(getPreparedArrowUploadByteLength(prepared)).toBe(2 * 2 * 4);
  const nullable = new arrow.Table({
    value: arrow.vectorFromArray([1, null, 3], new arrow.Float32())
  });
  expect(getPreparedArrowUploadByteLength(prepareArrowTableForGPUAnalytics(nullable))).toBe(
    3 * 4 * 2
  );
});

test('session capacity rejection happens before any GPU allocation and retains no reservation', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 0});
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const source = new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2, 3]))});
  expect(() => session.importArrowDataset({id: 'observations', table: source})).toThrow(/budget/);
  expect(createBuffer).not.toHaveBeenCalled();
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  expect(session.registry.getDataset('observations')).toBeUndefined();
  await session.destroy(Promise.resolve());
  createBuffer.mockRestore();
  device.destroy();
});

test('session upload preflight accounts exact time words and validity', () => {
  const time = new arrow.Vector([
    arrow.makeData({
      type: new arrow.TimestampNanosecond('UTC'),
      data: new BigInt64Array([1n, -1n]),
      length: 2
    })
  ]);
  const source = new arrow.Table({time});
  const prepared = prepareArrowTableForGPUAnalytics(source);
  expect(getPreparedArrowUploadByteLength(prepared)).toBe(24);
  expect(getRetainedArrowArrayBufferBytes(source, prepared)).toBeGreaterThanOrEqual(40);
});

test('empty Arrow scalar sources retain their schema without requiring child storage', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024});
  const dataset = session.importArrowDataset({
    id: 'empty',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array())})
  });
  expect(dataset.rowCount).toBe(0);
  expect(dataset.value.columns.value.format).toBe('float32');
  expect(dataset.value.columns.value.length).toBe(0);
  await session.destroy(Promise.resolve());
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  device.destroy();
});

test('failed source replacement preserves the published dataset and its source allocation', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const original = session.importArrowDataset({
    id: 'observations',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))})
  });
  const allocatedBytes = session.registry.memoryUsage.totalBytes;
  const createBuffer = vi.spyOn(device, 'createBuffer');
  expect(() =>
    session.importArrowDataset({
      id: 'observations',
      table: new arrow.Table({value: arrow.makeVector(new Float64Array([1.1, 2.2]))})
    })
  ).toThrow(/Float64/);
  expect(createBuffer).not.toHaveBeenCalled();
  expect(session.registry.getDataset('observations')).toBe(original);
  expect(session.registry.memoryUsage.totalBytes).toBe(allocatedBytes);
  expect(original.value.columns.value.data[0].buffer.destroyed).toBe(false);
  await session.destroy(Promise.resolve());
  expect(original.value.columns.value.data[0].buffer.destroyed).toBe(true);
  createBuffer.mockRestore();
  device.destroy();
});

test('partial upload failure releases allocated GPU buffers and both capacity reservations', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const createOriginalBuffer = device.createBuffer.bind(device);
  const allocated = createOriginalBuffer({byteLength: 4});
  const createBuffer = vi
    .spyOn(device, 'createBuffer')
    .mockImplementationOnce(() => allocated)
    .mockImplementationOnce(() => {
      throw new Error('Injected allocation failure');
    });
  expect(() =>
    session.importArrowDataset({
      id: 'observations',
      table: new arrow.Table({
        first: arrow.makeVector(new Float32Array([1])),
        second: arrow.makeVector(new Float32Array([2]))
      })
    })
  ).toThrow('Injected allocation failure');
  expect(allocated.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  expect(session.registry.getDataset('observations')).toBeUndefined();
  await session.destroy(Promise.resolve());
  createBuffer.mockRestore();
  device.destroy();
});

test.each([
  'remove',
  'destroy'
] as const)('session %s honors the submission fence without an explorer', async action => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'observations',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1]))})
  });
  const buffer = dataset.value.columns.value.data[0].buffer;
  let resolveFence!: () => void;
  const fence = new Promise<void>(resolve => {
    resolveFence = resolve;
  });
  const completion =
    action === 'remove' ? session.removeDataset('observations', fence) : session.destroy(fence);
  await Promise.resolve();
  expect(buffer.destroyed).toBe(false);
  resolveFence();
  await completion;
  expect(buffer.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('device loss invalidates immediately and restores the retained prepared source at a new revision', async () => {
  const device = new NullDevice({});
  let resolveLoss!: () => void;
  Object.defineProperty(device, 'lost', {
    value: new Promise<void>(resolve => {
      resolveLoss = resolve;
    })
  });
  const source = new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))});
  // Recovery must fit the original source+upload budget, without double-counting the retained source.
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 16});
  const original = session.importArrowDataset({id: 'observations', table: source});
  const groupBy = {kind: 'group-by', source: {datasetId: 'parent', revision: 4}, population: 'all'};
  Object.assign(original.source!.metadata!, {groupBy});
  const oldBuffer = original.value.columns.value.data[0].buffer;
  resolveLoss();
  await Promise.resolve();
  expect(session.deviceLost).toBe(true);
  expect(session.registry.getDataset(original.id)).toBeUndefined();
  expect(() => session.importArrowDataset({id: 'other', table: source})).toThrow(/lost/);
  await session.handleDeviceLoss();
  expect(oldBuffer.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(8);
  expect(session.registry.getRecoverySources()[0].source.value).toBe(source);
  const replacementDevice = new NullDevice({});
  const [restored] = await session.recoverDevice(replacementDevice);
  expect(session.device).toBe(replacementDevice);
  expect(session.deviceLost).toBe(false);
  expect(restored.revision).toBe(original.revision + 1);
  expect(restored.deviceEpoch).toBeGreaterThan(original.deviceEpoch);
  expect(restored.value.prepared).toBe(original.value.prepared);
  expect(restored.value.source).toBe(source);
  expect(restored.source?.metadata?.['groupBy']).toEqual(groupBy);
  expect(restored.value.retainedArrayBufferBytes).toBe(8);
  expect(restored.value.columns.value.data[0].buffer.device).toBe(replacementDevice);
  expect(session.registry.getRecoverySources()).toHaveLength(0);
  expect(session.registry.memoryUsage.totalBytes).toBe(16);
  await session.destroy(Promise.resolve());
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  device.destroy();
  replacementDevice.destroy();
});

test('failed recovery keeps CPU sources available for an explicit upload retry', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 16});
  const original = session.importArrowDataset({
    id: 'observations',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))})
  });
  await session.handleDeviceLoss();
  const replacementDevice = new NullDevice({});
  const allocation = vi.spyOn(replacementDevice, 'createBuffer').mockImplementationOnce(() => {
    throw new Error('Injected recovery allocation failure');
  });
  await expect(session.recoverDevice(replacementDevice)).rejects.toThrow('Injected recovery');
  expect(session.registry.memoryUsage.totalBytes).toBe(8);
  expect(session.registry.getRecoverySources()).toHaveLength(1);
  const [restored] = session.restoreArrowDatasets();
  expect(restored.revision).toBe(original.revision + 1);
  expect(session.registry.getRecoverySources()).toHaveLength(0);
  allocation.mockRestore();
  await session.destroy(Promise.resolve());
  device.destroy();
  replacementDevice.destroy();
});

test.each([
  'remove',
  'destroy'
] as const)('device loss settles %s waiting on an abandoned submission fence', async action => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 16});
  session.importArrowDataset({
    id: 'observations',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))})
  });
  const fence = new Promise<void>(() => {});
  const completion =
    action === 'remove' ? session.removeDataset('observations', fence) : session.destroy(fence);
  await session.handleDeviceLoss();
  await completion;
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  expect(session.registry.getRecoverySources()).toHaveLength(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('loss settles a pending explorer close and suppresses rejected stale readback publication', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024});
  const dataset = session.importArrowDataset({
    id: 'observations',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1]))})
  });
  let rejectRead!: (error: Error) => void;
  const readback = new Promise<null>((_resolve, reject) => {
    rejectRead = reject;
  });
  const destroyExplorer = vi.fn(async () => {
    rejectRead(new Error('Device lost during mapping'));
  });
  const handle = new KeplerSessionExplorer({
    explorer: {
      selectionMask: dataset.value.columns.value,
      readSummary: () => readback,
      destroyAsync: destroyExplorer
    } as unknown as GPUResidentExplorer,
    lease: session.registry.acquireDataset(dataset.id),
    reservation: session.registry.reserveMemory(16),
    isCurrent: () => session.registry.getDataset(dataset.id) === dataset,
    onClose: () => {}
  });
  const pendingRead = handle.readSummary();
  const pendingClose = handle.close(new Promise<void>(() => {}));
  await session.handleDeviceLoss();
  await handle.closeAfterDeviceLoss();
  await pendingClose;
  expect(await pendingRead).toBeNull();
  expect(destroyExplorer).toHaveBeenCalledOnce();
  expect(handle.isCurrent).toBe(false);
  expect(dataset.value.columns.value.data[0].buffer.destroyed).toBe(true);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('source-only datasets preserve batch identities and null geometry without scalar uploads', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const first = new arrow.Table({
    geometry: arrow.vectorFromArray(
      ['{"type":"Point","coordinates":[0,0]}', null],
      new arrow.Utf8()
    ),
    h3: arrow.vectorFromArray([0x8928308280fffffn, null], new arrow.Uint64())
  });
  const second = new arrow.Table({
    geometry: arrow.vectorFromArray(['{}'], new arrow.Utf8()),
    h3: arrow.vectorFromArray([0x8928308280bffffn], new arrow.Uint64())
  });
  const source = first.concat(second);
  const createBuffer = vi.spyOn(device, 'createBuffer');
  try {
    const dataset = session.importArrowDataset({
      id: 'geometry',
      table: source,
      normalization: {columns: []}
    });
    expect(dataset.rowCount).toBe(3);
    expect(dataset.value.source).toBe(source);
    expect(dataset.value.source.getChild('geometry')!.get(1)).toBeNull();
    expect(dataset.value.source.getChild('h3')!.get(0)).toBe(0x8928308280fffffn);
    expect(dataset.value.columns).toEqual({});
    expect(dataset.value.analytics.table.schema.fields).toEqual([]);
    expect(dataset.value.analytics.table.numRows).toBe(3);
    expect(dataset.value.analytics.table.batches.map(batch => batch.numRows)).toEqual([2, 1]);
    expect(dataset.value.prepared.sourceBatches).toEqual([
      {sourceBatchIndex: 0, sourceRowIndexOffset: 0, sourceRowCount: 2},
      {sourceBatchIndex: 1, sourceRowIndexOffset: 2, sourceRowCount: 1}
    ]);
    expect(dataset.value.uploadedBufferBytes).toBe(0);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(session.registry.memoryUsage.totalBytes).toBe(dataset.value.retainedArrayBufferBytes);
    await session.handleDeviceLoss();
    expect(session.registry.getRecoverySources()[0].source.value).toBe(source);
    const replacement = new NullDevice({});
    try {
      const [restored] = await session.recoverDevice(replacement);
      expect(restored.rowCount).toBe(3);
      expect(restored.value.columns).toEqual({});
      expect(restored.value.analytics.table.batches.map(batch => batch.numRows)).toEqual([2, 1]);
    } finally {
      await session.destroy(Promise.resolve());
      replacement.destroy();
    }
    expect(session.registry.memoryUsage.totalBytes).toBe(0);
  } finally {
    createBuffer.mockRestore();
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
