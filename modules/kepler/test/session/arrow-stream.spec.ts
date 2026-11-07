// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {requireTestDevice} from '../../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {KeplerArrowStreamPublisher} from '../../src/session/arrow-stream-publisher';

test('streamed GPU revisions preserve chunk buffers, dictionary codes, nulls and original row identities through preview and completion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 << 20});
  const stream = new KeplerArrowStreamPublisher({
    session,
    id: 'live',
    maximumQueuedBytes: 1 << 20,
    maximumQueuedBatches: 3,
    maximumBatchRows: 16,
    dictionaryEvolution: 'append-unordered',
    normalization: {timeColumns: {time: 'nanosecond'}}
  });
  const makeBatch = (
    values: (number | null)[],
    categories: (string | null)[],
    times: (bigint | null)[]
  ) => {
    const table = new arrow.Table({
      value: arrow.vectorFromArray(values, new arrow.Float32()),
      category: arrow.vectorFromArray(categories, new arrow.Utf8()),
      time: arrow.vectorFromArray(times, new arrow.Int64())
    });
    return arrow.tableFromIPC(arrow.tableToIPC(table, 'stream')).batches[0];
  };
  const first = await stream.enqueue(makeBatch([1, null], ['a', 'b'], [9007199254740993n, null]));
  const firstBuffer = first.value.columns['value'].data[0].buffer;
  const writes = vi.spyOn(firstBuffer, 'write');
  const preview = session.openExplorer('live', {
    allowIncompletePopulation: true,
    workingMemoryBudgetBytes: 8 << 20,
    dimensions: [{id: 'value', kind: 'range', column: 'value'}]
  });
  try {
    expect(() =>
      session.openExplorer('live', {workingMemoryBudgetBytes: 8 << 20, dimensions: []})
    ).toThrow(/incomplete/);
    preview.setRange('value', 0, 10);
    const prefixEncoder = device.createCommandEncoder();
    preview.encode(prefixEncoder);
    device.submit(prefixEncoder.finish());
    expect((await preview.readSummary())?.selectedCount).toBe(1);
    const empty = stream.enqueue(makeBatch([], [], []));
    const last = stream.enqueue(
      makeBatch([3, 4, 5], ['c', 'a', 'c'], [-9007199254740993n, 0n, 5n])
    );
    const completion = stream.finish();
    await Promise.all([empty, last]);
    const completed = await completion;
    expect(completed.population?.state).toBe('complete');
    expect(completed.batches.map(batch => batch.rowCount)).toEqual([2, 0, 3]);
    expect(completed.batches.map(batch => batch.sourceRowOffset)).toEqual([0, 2, 2]);
    expect(completed.value.columns['value'].data[0].buffer).toBe(firstBuffer);
    expect(firstBuffer.destroyed).toBe(false);
    expect(writes).not.toHaveBeenCalled();
    expect(preview.isCurrent).toBe(false);
    expect(await preview.readPage(0, 2, ['value'])).toBeNull();
    expect(await readWords(completed.value.columns['category'])).toEqual([0, 1, 2, 0, 2]);
    expect(await readWords(completed.value.analytics.validity['value']!)).toEqual([1, 0, 1, 1, 1]);
    expect(await readWords(completed.value.timeColumns['time'].timestamps, 2)).toEqual([
      1, 2097152, 0, 0, 4294967295, 4292870143, 0, 0, 5, 0
    ]);
    const explorer = session.openExplorer('live', {
      workingMemoryBudgetBytes: 8 << 20,
      dimensions: [{id: 'value', kind: 'range', column: 'value'}]
    });
    explorer.setRange('value', 3, 4);
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    device.submit(encoder.finish());
    expect((await explorer.readSummary())?.selectedCount).toBe(2);
    expect((await explorer.readPage(0, 8, ['value']))?.sourceRowIndices).toEqual(
      new Uint32Array([2, 3])
    );
    await preview.close(Promise.resolve());
    await stream.dispose(Promise.resolve());
    expect(explorer.isCurrent).toBe(false);
    expect(firstBuffer.destroyed).toBe(true);
    expect(session.registry.memoryUsage.totalBytes).toBe(0);
  } finally {
    writes.mockRestore();
    await preview.close(Promise.resolve());
    await stream.dispose(Promise.resolve());
    await session.destroy(Promise.resolve());
  }
});

async function readWords(vector: GPUVector, components = 1): Promise<number[]> {
  const result: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * components * 4);
    result.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length * components));
  }
  return result;
}
