// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../../src/session/compute-session';
import {materializeKeplerSelection} from '../../src/session/selection-export';

function fixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'data',
    table: new arrow.Table({value: arrow.vectorFromArray([10, null, 30, 40], new arrow.Float32())})
  });
  const selected = [0, 1, 3];
  const inspector = {
    dataset,
    currentRevision: 4,
    isCurrent: true,
    maximumInspectionPageSize: 2,
    maximumInspectionReadbackBytes: 28,
    readSelectedCount: vi.fn(async () => ({
      revision: 4,
      datasetRevision: dataset.revision,
      selectedCount: 3
    })),
    readPage: vi.fn(async (offset: number, limit: number) => {
      const indices = selected.slice(offset, offset + limit);
      return {
        revision: 4,
        datasetRevision: dataset.revision,
        sourceRowIndices: new Uint32Array(indices),
        columns: {value: new Float32Array(indices.map(index => [10, 0, 30, 40][index]))},
        validity: {value: new Uint32Array(indices.map(index => (index === 1 ? 0 : 1)))}
      };
    })
  };
  return {
    session,
    inspector,
    explorer: inspector as unknown as KeplerSessionExplorer,
    cleanup: async () => {
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}

test('full selected export is paged, preserves source identity/nulls and owns an explicit reservation', async () => {
  const context = fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  const result = await materializeKeplerSelection(context.session, context.explorer, {
    mode: 'all-selected',
    maximumRows: 3,
    columns: ['value'],
    maximumReadbackBytes: 48,
    maximumCpuBytes: 84
  });
  expect(result.mode).toBe('all-selected');
  expect(Array.from(result.sourceRowIndices)).toEqual([0, 1, 3]);
  expect(Array.from(result.columns['value'])).toEqual([10, 0, 40]);
  expect(Array.from(result.validity['value'])).toEqual([1, 0, 1]);
  expect(result.readbackBytes).toBe(48);
  expect(result.retainedBytes).toBe(36);
  expect(context.inspector.readPage.mock.calls.map(call => call.slice(0, 2))).toEqual([
    [0, 2],
    [2, 1]
  ]);
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline + 84);
  result.destroy();
  result.destroy();
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  expect(result.sourceRowIndices).toHaveLength(0);
  await context.cleanup();
});

test('page export remains explicitly labeled as a subset with its selected offset and total count', async () => {
  const context = fixture();
  const result = await materializeKeplerSelection(context.session, context.explorer, {
    mode: 'page',
    offset: 1,
    limit: 1,
    columns: ['value'],
    maximumReadbackBytes: 20,
    maximumCpuBytes: 36
  });
  expect(result.mode).toBe('page');
  expect(result.offset).toBe(1);
  expect(result.selectedCount).toBe(3);
  expect(result.rowCount).toBe(1);
  expect(Array.from(result.sourceRowIndices)).toEqual([1]);
  result.destroy();
  await context.cleanup();
});

test.each([
  'row',
  'readback',
  'cpu'
] as const)('export %s budget rejection happens before any row readback', async bound => {
  const context = fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  await expect(
    materializeKeplerSelection(context.session, context.explorer, {
      mode: 'all-selected',
      maximumRows: bound === 'row' ? 2 : 3,
      columns: ['value'],
      maximumReadbackBytes: bound === 'readback' ? 47 : 48,
      maximumCpuBytes: bound === 'cpu' ? 83 : 84
    })
  ).rejects.toThrow(/budget/);
  expect(context.inspector.readPage).not.toHaveBeenCalled();
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  await context.cleanup();
});

test.each([
  'selection',
  'dataset',
  'cancel'
] as const)('export discards partial materialization after %s changes', async change => {
  const context = fixture();
  const controller = new AbortController();
  const read = context.inspector.readPage.getMockImplementation()!;
  context.inspector.readPage.mockImplementationOnce(async (...arguments_) => {
    const page = await read(...arguments_);
    if (change === 'selection') context.inspector.currentRevision++;
    else if (change === 'dataset')
      context.session.importArrowDataset({
        id: 'data',
        table: new arrow.Table({value: arrow.makeVector(new Float32Array([99]))})
      });
    else controller.abort();
    return page;
  });
  await expect(
    materializeKeplerSelection(context.session, context.explorer, {
      mode: 'all-selected',
      maximumRows: 3,
      columns: ['value'],
      maximumReadbackBytes: 48,
      maximumCpuBytes: 84,
      signal: controller.signal
    })
  ).rejects.toThrow();
  expect(context.inspector.readPage).toHaveBeenCalledTimes(1);
  expect(context.session.registry.memoryUsage.reservedBytes.cpu).toBe(0);
  await context.cleanup();
});

test('already cancelled export performs no GPU count or row reads', async () => {
  const context = fixture();
  const controller = new AbortController();
  controller.abort();
  await expect(
    materializeKeplerSelection(context.session, context.explorer, {
      mode: 'page',
      offset: 0,
      limit: 1,
      columns: ['value'],
      maximumReadbackBytes: 20,
      maximumCpuBytes: 36,
      signal: controller.signal
    })
  ).rejects.toThrow();
  expect(context.inspector.readSelectedCount).not.toHaveBeenCalled();
  await context.cleanup();
});
