// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ArrowStreamController} from '../../../examples/deck/kepler-compute/src/arrow-stream-controller';

function createFixture(
  overrides: Partial<ConstructorParameters<typeof ArrowStreamController>[0]> = {}
) {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 << 20});
  const controller = new ArrowStreamController({
    session,
    id: 'stream',
    normalization: {},
    maximumInputBytes: 4 << 20,
    maximumInputChunkBytes: 1 << 20,
    maximumQueuedBytes: 4 << 20,
    maximumBatchRows: 100,
    getCompletionFence: () => Promise.resolve(),
    ...overrides
  });
  return {device, session, controller};
}
function makeBytes(): Uint8Array {
  const batches = [[1, 2], [], [3, 4, 5]].map(values => {
    const table = new arrow.Table({value: arrow.vectorFromArray(values, new arrow.Float32())});
    return arrow.tableFromIPC(arrow.tableToIPC(table, 'stream')).batches[0];
  });
  return arrow.tableToIPC(new arrow.Table(batches[0].schema, batches), 'stream');
}
function input(bytes: Uint8Array, chunkBytes = 32, cancelled = () => {}) {
  let offset = 0;
  return new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        if (offset === bytes.length) controller.close();
        else {
          controller.enqueue(bytes.subarray(offset, offset + chunkBytes));
          offset = Math.min(bytes.length, offset + chunkBytes);
        }
      },
      cancel: cancelled
    },
    {highWaterMark: 0}
  );
}

test('native streaming pauses per original batch, guards incomplete analysis, and transfers completed buffers through panel disposal', async () => {
  const {device, session, controller} = createFixture();
  const bytes = makeBytes();
  await controller.start(input(bytes));
  expect(controller.progress).toMatchObject({phase: 'paused', rows: 2, batches: 1});
  expect(controller.progress.bytesRead).toBeLessThan(bytes.length);
  expect(() => session.registry.acquireDataset('stream')).toThrow(/incomplete/);
  await expect(controller.finish()).rejects.toThrow(/end of/);
  const original = controller.publisher.snapshot!.value.columns['value'].data[0].buffer;
  const writes = vi.spyOn(original, 'write');
  await controller.next();
  expect(controller.progress).toMatchObject({phase: 'paused', rows: 2, batches: 2});
  await controller.resume();
  expect(controller.progress).toMatchObject({phase: 'exhausted', rows: 5, batches: 3});
  const allocate = vi.spyOn(device, 'createBuffer');
  const complete = await controller.finish();
  expect(complete.batches.map(batch => batch.rowCount)).toEqual([2, 0, 3]);
  expect(complete.value.columns['value'].data[0].buffer).toBe(original);
  expect(allocate).not.toHaveBeenCalled();
  expect(writes).not.toHaveBeenCalled();
  let release!: () => void;
  const attach = controller.publish(async snapshot => {
    expect(snapshot).toBe(complete);
    await new Promise<void>(resolve => {
      release = resolve;
    });
  });
  const disposal = controller.dispose();
  release();
  await Promise.all([attach, disposal]);
  expect(controller.progress.phase).toBe('published');
  expect(session.registry.getDataset('stream')).toBe(complete);
  expect(original.destroyed).toBe(false);
  expect(controller.publisher.snapshot).toBeUndefined();
  await session.removeDataset('stream', Promise.resolve());
  expect(original.destroyed).toBe(true);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('cancellation releases a stalled IPC reader and prevents late batch admission', async () => {
  const {device, session, controller} = createFixture();
  const cancelled = vi.fn();
  const source = new ReadableStream<Uint8Array>({cancel: cancelled});
  const starting = controller.start(source);
  await Promise.resolve();
  controller.cancel();
  await starting;
  await controller.dispose();
  expect(cancelled).toHaveBeenCalledOnce();
  expect(source.locked).toBe(false);
  expect(controller.progress.phase).toBe('cancelled');
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('native IPC reader rejects random-access files and oversize chunks before publication', async () => {
  for (const bytes of [new TextEncoder().encode('ARROW1'), new Uint8Array(1025)]) {
    const {device, session, controller} = createFixture({maximumInputChunkBytes: 1024});
    const cancelled = vi.fn();
    await expect(controller.start(input(bytes, bytes.length, cancelled))).rejects.toThrow(
      /stream format|capacity/
    );
    await controller.dispose();
    expect(cancelled).toHaveBeenCalled();
    expect(session.registry.getDataset('stream')).toBeUndefined();
    expect(session.registry.memoryUsage.totalBytes).toBe(0);
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('failed native attach cannot retry a removed stream revision or detach its ownership', async () => {
  const {device, session, controller} = createFixture();
  await controller.start(input(makeBytes()));
  await controller.resume();
  await controller.finish();
  await expect(
    controller.publish(async snapshot => {
      await session.removeDataset(snapshot.id, Promise.resolve());
      throw new Error('Native registration failed');
    })
  ).rejects.toThrow(/Native registration/);
  expect(controller.progress.phase).toBe('failed');
  await expect(controller.publish(async () => {})).rejects.toThrow(/Finish/);
  await controller.dispose();
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('source preview bounds UTF-8 decoding before value access and cleanup survives reader cancellation rejection', async () => {
  const {device, session, controller} = createFixture({normalization: {columns: ['value']}});
  const table = new arrow.Table({
    value: arrow.vectorFromArray([1], new arrow.Float32()),
    text: arrow.vectorFromArray(['x'.repeat(100000)], new arrow.Utf8())
  });
  await controller.start(input(arrow.tableToIPC(table, 'stream'), 65536));
  const decode = vi.spyOn(TextDecoder.prototype, 'decode');
  const preview = controller.preview()!;
  expect(preview.rows).toHaveLength(1);
  expect(preview.rows[0][1].length).toBeLessThanOrEqual(120);
  for (const call of decode.mock.calls) expect(call[0]?.byteLength ?? 0).toBeLessThanOrEqual(480);
  decode.mockRestore();
  const reader: arrow.RecordBatchReader = Reflect.get(controller, 'reader');
  vi.spyOn(reader, 'cancel').mockRejectedValue(new Error('Reader cancel failed'));
  await expect(controller.dispose()).rejects.toThrow(/Reader cancel/);
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
});

test('successful native handoff after device recovery detaches the old publisher without removing the restored generation', async () => {
  const {device, session, controller} = createFixture();
  await controller.start(input(makeBytes()));
  await controller.resume();
  await controller.finish();
  const replacement = new NullDevice({});
  await controller.publish(async () => {
    await session.handleDeviceLoss();
    await session.recoverDevice(replacement);
  });
  const restored = session.registry.getDataset('stream')!;
  await controller.dispose();
  expect(session.registry.getDataset('stream')).toBe(restored);
  expect(restored.population?.state).toBe('complete');
  expect(controller.progress.phase).toBe('published');
  await session.removeDataset('stream', Promise.resolve());
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
  await session.destroy(Promise.resolve());
  device.destroy();
  replacement.destroy();
});

test('explicitly confirmed partial prefixes record row and batch counts and stop remaining input', async () => {
  const cancelled = vi.fn();
  const {session, controller} = createFixture({remoteUrl: 'https://data.example/stream.arrows'});
  await controller.start(input(makeBytes(), 32, cancelled));
  await expect(controller.finishPartial(true as never)).rejects.toThrow(/confirmation/);
  const dataset = await controller.finishPartial('save-incomplete-prefix');
  expect(controller.progress).toMatchObject({
    phase: 'complete',
    partial: {rowCount: 2, batchCount: 1},
    remoteUrl: 'https://data.example/stream.arrows'
  });
  expect(dataset.population).toMatchObject({
    state: 'complete',
    partial: {rowCount: 2, batchCount: 1}
  });
  expect(dataset.source?.metadata?.['nativeStreamOrigin']).toEqual({
    kind: 'remote',
    url: 'https://data.example/stream.arrows'
  });
  expect(cancelled).toHaveBeenCalled();
  expect(session.registry.getDataset('stream')?.population?.partial).toBeDefined();
  await controller.dispose();
});

test('a cancelled prefix cannot be saved as partial and remote startup aborts on cancel', async () => {
  const {controller} = createFixture({
    remoteUrl: 'https://data.example/stream.arrows',
    fetchImplementation: ((_url: unknown, init?: RequestInit) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')));
      })) as typeof fetch
  });
  const opening = controller.startRemote().catch(() => undefined);
  controller.cancel();
  await opening;
  await expect(controller.finishPartial('save-incomplete-prefix')).rejects.toThrow();
  expect(controller.progress.phase).toBe('cancelled');
  await controller.dispose();
});

test('remote streams read a fetch body through bounded admission and reject failed responses', async () => {
  const bytes = makeBytes();
  const {controller} = createFixture({
    remoteUrl: 'https://data.example/stream.arrows',
    fetchImplementation: (async () =>
      new Response(input(bytes), {
        headers: {'content-length': String(bytes.length)}
      })) as typeof fetch
  });
  await controller.startRemote();
  expect(controller.progress).toMatchObject({phase: 'paused', rows: 2});
  await controller.dispose();
  const failing = createFixture({
    remoteUrl: 'https://data.example/missing.arrows',
    fetchImplementation: (async () => new Response('', {status: 404})) as typeof fetch
  });
  await expect(failing.controller.startRemote()).rejects.toThrow(/HTTP 404/);
  expect(failing.controller.progress.phase).toBe('failed');
  await failing.controller.dispose();
});
