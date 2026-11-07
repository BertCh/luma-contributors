// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';
import {
  KeplerComputeSession,
  getRetainedArrowArrayBufferBytes
} from '../../src/session/compute-session';
import {
  KeplerArrowStreamPublisher,
  KeplerStreamBackpressureError
} from '../../src/session/arrow-stream-publisher';

function batch(values: (number | null)[], categories: (string | null)[] = values.map(() => 'a')) {
  const source = new arrow.Table({
    value: arrow.vectorFromArray(values, new arrow.Float32()),
    category: arrow.vectorFromArray(categories, new arrow.Utf8())
  });
  // IPC yields canonical Data children even for empty builder vectors.
  return arrow.tableFromIPC(arrow.tableToIPC(source, 'stream')).batches[0];
}
function fixture(
  overrides: Partial<ConstructorParameters<typeof KeplerArrowStreamPublisher>[0]> = {}
) {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 4 * 1024 * 1024});
  const stream = new KeplerArrowStreamPublisher({
    session,
    id: 'stream',
    maximumQueuedBytes: 1 << 20,
    maximumQueuedBatches: 2,
    maximumBatchRows: 1024,
    dictionaryEvolution: 'append-unordered',
    ...overrides
  });
  return {device, session, stream};
}

test('Arrow streaming applies backpressure before retaining input and appends immutable chunk views without reupload', async () => {
  const {device, session, stream} = fixture();
  const firstSource = batch([1, null], ['a', 'b']);
  const emptySource = batch([], []);
  const lastSource = batch([3], ['c']);
  const first = stream.enqueue(firstSource);
  const empty = stream.enqueue(emptySource);
  expect(stream.queuedBatches).toBe(2);
  expect(stream.queuedBytes).toBe(
    getRetainedArrowArrayBufferBytes(new arrow.Table(firstSource.schema, [firstSource])) +
      getRetainedArrowArrayBufferBytes(new arrow.Table(emptySource.schema, [emptySource]))
  );
  expect(() => stream.enqueue(lastSource)).toThrow(KeplerStreamBackpressureError);
  const ready = stream.ready;
  expect(stream.ready).toBe(ready);
  const firstSnapshot = await first;
  const prefix = session.registry.acquireDataset('stream', {allowIncomplete: true});
  expect(() => session.registry.acquireDataset('stream')).toThrow(/incomplete/);
  const buffer = firstSnapshot.value.columns['value'].data[0].buffer;
  const writes = vi.spyOn(buffer, 'write');
  await ready;
  await empty;
  const lastSnapshot = await stream.enqueue(lastSource);
  expect(lastSnapshot.value.columns['value'].data.map(chunk => chunk.length)).toEqual([2, 0, 1]);
  expect(lastSnapshot.batches.map(part => part.sourceRowOffset)).toEqual([0, 2, 2]);
  expect(lastSnapshot.value.source.batches).toEqual([firstSource, emptySource, lastSource]);
  expect(lastSnapshot.value.columns['value'].data[0].buffer).toBe(buffer);
  expect(
    lastSnapshot.value.analytics.table.batches.map(part => part.sourceInfo?.sourceRowIndexOffset)
  ).toEqual([0, 2, 2]);
  expect(firstSnapshot.value.prepared.dictionaries['category'].values).toEqual(['a', 'b']);
  expect(lastSnapshot.value.prepared.dictionaries['category'].values).toEqual(['a', 'b', 'c']);
  expect(lastSnapshot.value.prepared.table.batches[0].data.children[1].values).toBe(
    firstSnapshot.value.prepared.table.batches[0].data.children[1].values
  );
  expect(
    Array.from(firstSnapshot.value.prepared.table.batches[0].data.children[1].dictionary!)
  ).toEqual(['a', 'b']);
  expect(
    Array.from(lastSnapshot.value.prepared.table.batches[0].data.children[1].dictionary!)
  ).toEqual(['a', 'b', 'c']);
  expect(writes).not.toHaveBeenCalled();
  const allocate = vi.spyOn(device, 'createBuffer');
  const complete = await stream.finish();
  expect(complete.population?.state).toBe('complete');
  expect(complete.revision).toBe(lastSnapshot.revision + 1);
  expect(allocate).not.toHaveBeenCalled();
  const completeLease = session.registry.acquireDataset('stream');
  await completeLease.release();
  await prefix.release();
  let release!: () => void;
  const disposal = stream.dispose(
    new Promise<void>(resolve => {
      release = resolve;
    })
  );
  expect(buffer.destroyed).toBe(false);
  release();
  await disposal;
  expect(stream.snapshot).toBeUndefined();
  await expect(stream.finish()).rejects.toThrow(/disposed/);
  expect(buffer.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  writes.mockRestore();
  allocate.mockRestore();
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('finish drains accepted batches, cancellation rejects queued inputs and retains an explicitly incomplete prefix', async () => {
  const {device, session, stream} = fixture();
  const first = await stream.enqueue(batch([1]));
  const pending = stream.enqueue(batch([2]));
  const settled = Promise.allSettled([pending]);
  const finished = stream.finish();
  const completion = Promise.allSettled([finished]);
  expect(() => stream.enqueue(batch([3]))).toThrow(/closed/);
  stream.cancel(new Error('Producer stopped'));
  expect((await settled)[0].status).toBe('rejected');
  expect((await completion)[0].status).toBe('rejected');
  expect(stream.state).toBe('cancelled');
  expect(stream.snapshot?.rowCount).toBe(1);
  expect(stream.snapshot?.population?.state).toBe('cancelled');
  expect(stream.snapshot?.value.columns['value'].data[0].buffer).toBe(
    first.value.columns['value'].data[0].buffer
  );
  expect(stream.queuedBytes).toBe(0);
  expect(() => session.registry.beginRequest('analysis', ['stream'])).toThrow(/incomplete/);
  await stream.dispose(Promise.resolve());
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('schema and fixed dictionary evolution reject explicitly without mutating earlier metadata', async () => {
  const {device, session, stream} = fixture({dictionaryEvolution: 'fixed'});
  const first = await stream.enqueue(batch([1], ['a']));
  const changed = new arrow.Table({
    value: arrow.vectorFromArray([1], new arrow.Int32()),
    category: arrow.vectorFromArray(['a'], new arrow.Utf8())
  }).batches[0];
  expect(() => stream.enqueue(changed)).toThrow(/schema evolution/);
  expect(stream.state).toBe('open');
  await expect(stream.enqueue(batch([2], ['b']))).rejects.toThrow(/dictionary evolution/);
  expect(stream.state).toBe('failed');
  expect(stream.snapshot?.population?.state).toBe('failed');
  expect(first.value.prepared.dictionaries['category'].values).toEqual(['a']);
  expect(stream.snapshot?.rowCount).toBe(1);
  await stream.dispose(Promise.resolve());
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('incremental exact timestamps retain original source offsets and finish without scalar columns', async () => {
  const {device, session, stream} = fixture({normalization: {timeColumns: {time: 'nanosecond'}}});
  const timeBatch = (values: (bigint | null)[]) => {
    const table = new arrow.Table({time: arrow.vectorFromArray(values, new arrow.Int64())});
    return arrow.tableFromIPC(arrow.tableToIPC(table, 'stream')).batches[0];
  };
  const first = await stream.enqueue(timeBatch([9007199254740993n, null]));
  const buffer = first.value.timeColumns['time'].timestamps.data[0].buffer;
  await stream.enqueue(timeBatch([]));
  await stream.enqueue(timeBatch([-9007199254740993n]));
  const completed = await stream.finish();
  expect(Object.keys(completed.value.columns)).toEqual([]);
  expect(completed.value.timeColumns['time'].timestamps.data.map(chunk => chunk.length)).toEqual([
    2, 0, 1
  ]);
  expect(completed.value.timeColumns['time'].timestamps.data[0].buffer).toBe(buffer);
  expect(completed.value.source.getChild('time')?.get(2)).toBe(-9007199254740993n);
  expect(completed.value.analytics.table.numRows).toBe(3);
  await stream.dispose(Promise.resolve());
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('upload failure rolls back only the unadopted batch and iterator consumption never pulls ahead', async () => {
  const {device, session, stream} = fixture();
  const original = device.createBuffer.bind(device);
  const buffers: ReturnType<typeof original>[] = [];
  const allocate = vi.spyOn(device, 'createBuffer').mockImplementation(options => {
    if (buffers.length === 1) throw new Error('Injected upload failure');
    const buffer = original(options);
    buffers.push(buffer);
    return buffer;
  });
  let pulled = 0;
  let returned = false;
  async function* source() {
    try {
      pulled++;
      yield batch([1]);
      pulled++;
      yield batch([2]);
    } finally {
      returned = true;
    }
  }
  await expect(stream.consume(source())).rejects.toThrow(/Injected/);
  expect(pulled).toBe(1);
  expect(returned).toBe(true);
  expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  allocate.mockRestore();
  await stream.dispose(Promise.resolve());
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('device loss preserves explicit stream recovery metadata and never automatically restores a prefix as complete', async () => {
  const {device, session, stream} = fixture();
  await stream.enqueue(batch([1]));
  await session.handleDeviceLoss();
  const recovery = session.registry.getRecoverySources()[0];
  expect(recovery.population?.state).toBe('open');
  expect(recovery.source.metadata?.['keplerStreamId']).toBe(stream.streamId);
  const replacement = new NullDevice({});
  expect(await session.recoverDevice(replacement)).toEqual([]);
  expect(session.registry.getDataset('stream')).toBeUndefined();
  await stream.dispose(Promise.resolve());
  expect(session.registry.getRecoverySources()).toEqual([]);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
  replacement.destroy();
});

test('finish linearizes after all admitted batches and terminal streams cannot enqueue again', async () => {
  const {device, session, stream} = fixture();
  const first = stream.enqueue(batch([1]));
  const second = stream.enqueue(batch([2]));
  const completion = stream.finish();
  expect(stream.finish()).toBe(completion);
  const snapshots = await Promise.all([first, second, completion]);
  expect(snapshots.map(snapshot => snapshot.population?.state)).toEqual([
    'open',
    'open',
    'complete'
  ]);
  expect(snapshots.map(snapshot => snapshot.rowCount)).toEqual([1, 2, 2]);
  expect(snapshots.map(snapshot => snapshot.revision)).toEqual([1, 2, 3]);
  expect(stream.queuedBatches).toBe(0);
  expect(stream.queuedBytes).toBe(0);
  expect(() => stream.enqueue(batch([3]))).toThrow(/closed/);
  stream.cancel();
  expect(stream.state).toBe('complete');
  await stream.dispose(Promise.resolve());
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('cancellation during an upload cannot publish the active batch after cancellation', async () => {
  const {device, session, stream} = fixture();
  const original = device.createBuffer.bind(device);
  const buffers: ReturnType<typeof original>[] = [];
  const allocation = vi.spyOn(device, 'createBuffer').mockImplementation(options => {
    const buffer = original(options);
    buffers.push(buffer);
    stream.cancel(new Error('Stop during allocation'));
    return buffer;
  });
  await expect(stream.enqueue(batch([1]))).rejects.toThrow(/Stop during allocation/);
  expect(stream.state).toBe('cancelled');
  expect(stream.snapshot).toBeUndefined();
  expect(session.registry.getDataset('stream')).toBeUndefined();
  expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  allocation.mockRestore();
  await stream.dispose(Promise.resolve());
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('completed streams recover prepared chunks and dictionary identities across retry and repeated device loss', async () => {
  const {device, session, stream} = fixture();
  await stream.enqueue(batch([1, 2], ['b', 'a']));
  await stream.enqueue(batch([], []));
  await stream.enqueue(batch([3], ['c']));
  const complete = await stream.finish();
  const prepared = complete.value.prepared;
  const cpuBytes = session.registry.memoryUsage.allocatedBytes.cpu;
  await session.handleDeviceLoss();
  const replacement = new NullDevice({});
  const allocate = vi.spyOn(replacement, 'createBuffer').mockImplementationOnce(() => {
    throw new Error('Injected restore failure');
  });
  await expect(session.recoverDevice(replacement)).rejects.toThrow(/Injected/);
  expect(session.registry.getDataset('stream')).toBeUndefined();
  expect(session.registry.getRecoverySources()[0].population?.state).toBe('complete');
  expect(session.registry.memoryUsage.allocatedBytes.cpu).toBe(cpuBytes);
  allocate.mockRestore();
  const [restored] = session.restoreArrowDatasets();
  expect(restored.population).toEqual(complete.population);
  expect(restored.revision).toBeGreaterThan(complete.revision);
  expect(restored.value.prepared).toBe(prepared);
  expect(restored.value.source.batches).toEqual(complete.value.source.batches);
  expect(restored.value.columns['value'].data.map(chunk => chunk.length)).toEqual([2, 0, 1]);
  expect(restored.value.prepared.dictionaries['category'].values).toEqual(['b', 'a', 'c']);
  expect(session.registry.memoryUsage.allocatedBytes.cpu).toBe(cpuBytes);
  const lease = session.registry.acquireDataset('stream');
  await lease.release();
  await session.handleDeviceLoss();
  const third = new NullDevice({});
  const [restoredAgain] = await session.recoverDevice(third);
  expect(restoredAgain.population).toEqual(complete.population);
  expect(restoredAgain.value.prepared).toBe(prepared);
  expect(session.registry.memoryUsage.allocatedBytes.cpu).toBe(cpuBytes);
  // The old publisher cannot remove the explicitly rebound replacement-device revision.
  await stream.dispose(Promise.resolve());
  expect(session.registry.getDataset('stream')).toBe(restoredAgain);
  await session.removeDataset('stream', Promise.resolve());
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
  replacement.destroy();
  third.destroy();
});

test('rejected disposal fences release retained prefix references and resources', async () => {
  const {device, session, stream} = fixture();
  const snapshot = await stream.enqueue(batch([1]));
  const buffer = snapshot.value.columns['value'].data[0].buffer;
  await expect(stream.dispose(Promise.reject(new Error('Fence rejected')))).rejects.toThrow(
    /Fence rejected/
  );
  expect(stream.snapshot).toBeUndefined();
  expect(buffer.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});
