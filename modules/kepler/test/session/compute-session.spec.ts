// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {materializeKeplerSelection} from '../../src/session/selection-export';

test('Arrow session reuses uploaded columns, inspects GPU selections and retires replaced revisions after fences', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for the Kepler session test');
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 8 * 1024 * 1024});
  const first = new arrow.Table({
    value: arrow.vectorFromArray([1, 2, 3, 4], new arrow.Int16()),
    category: arrow.vectorFromArray(['a', 'b', 'a', 'b'], new arrow.Utf8())
  });
  const second = new arrow.Table({
    value: arrow.vectorFromArray([5, 6, 7], new arrow.Int16()),
    category: arrow.vectorFromArray(['b', 'c', 'a'], new arrow.Utf8())
  });
  const source = first.concat(second);
  const dataset = session.importArrowDataset({id: 'observations', table: source});
  const sourceBuffers = Object.values(dataset.value.columns).flatMap(vector =>
    vector.data.map(data => data.buffer)
  );
  const explorer = session.openExplorer('observations', {
    workingMemoryBudgetBytes: 1024 * 1024,
    maximumReadbackBytes: 64 * 1024,
    dimensions: [
      {id: 'value', kind: 'range', column: 'value'},
      {id: 'category', kind: 'category', column: 'category', capacity: 3}
    ],
    histograms: [{id: 'values', column: 'value', dimension: 'value', domain: [0, 8], binCount: 4}]
  });
  try {
    expect(explorer.workingMemoryBudgetBytes).toBe(1024 * 1024);
    expect(explorer.reservedMemoryByteLength).toBeGreaterThanOrEqual(2 * 2 * 64 * 1024);
    expect(explorer.reservedMemoryByteLength).toBeLessThan(explorer.workingMemoryBudgetBytes);
    const reservedMemoryByteLength = explorer.reservedMemoryByteLength;
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(reservedMemoryByteLength);
    expect(dataset.rowCount).toBe(7);
    expect(dataset.value.columns['value'].data.map(chunk => chunk.length)).toEqual([4, 3]);
    expect(dataset.value.uploadedBufferBytes).toBeGreaterThan(0);
    explorer.setRange('value', 2, 6);
    explorer.setCategories('category', [0]);
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    device.submit(encoder.finish());
    const summary = await explorer.readSummary();
    expect(summary?.selectedCount).toBe(1);
    const page = await explorer.readPage(0, 10, ['value', 'category']);
    expect(Array.from(page!.sourceRowIndices)).toEqual([2]);
    expect(Array.from(page!.columns['value'])).toEqual([3]);
    const exported = await materializeKeplerSelection(session, explorer, {
      mode: 'all-selected',
      maximumRows: 10,
      columns: ['value', 'category'],
      maximumReadbackBytes: 1024,
      maximumCpuBytes: 1024
    });
    expect(exported.mode).toBe('all-selected');
    expect(exported.rowCount).toBe(1);
    expect(Array.from(exported.sourceRowIndices)).toEqual([2]);
    expect(Array.from(exported.columns['value'])).toEqual([3]);
    exported.destroy();
    expect(explorer.reservedMemoryByteLength).toBe(reservedMemoryByteLength);
    expect(
      Object.values(dataset.value.columns).flatMap(vector => vector.data.map(data => data.buffer))
    ).toEqual(sourceBuffers);

    const replacement = session.importArrowDataset({id: 'observations', table: second});
    expect(replacement.revision).toBe(dataset.revision + 1);
    expect(explorer.isCurrent).toBe(false);
    expect(await explorer.readSummary()).toBeNull();
    expect(() => explorer.setRange('value', 0, 1)).toThrow(/stale/);
    expect(sourceBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    let resolveFence!: () => void;
    const fence = new Promise<void>(resolve => {
      resolveFence = resolve;
    });
    const closing = explorer.close(fence);
    await Promise.resolve();
    expect(explorer.reservedMemoryByteLength).toBe(reservedMemoryByteLength);
    expect(sourceBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    resolveFence();
    await closing;
    expect(explorer.reservedMemoryByteLength).toBe(0);
    expect(sourceBuffers.every(buffer => buffer.destroyed)).toBe(true);
    await session.removeDataset('observations', Promise.resolve());
    expect(session.registry.memoryUsage.totalBytes).toBe(0);
  } finally {
    await session.destroy(Promise.resolve());
  }
});
