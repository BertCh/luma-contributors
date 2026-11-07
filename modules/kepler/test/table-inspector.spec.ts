// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerTableInspector} from '../src/exploration/gpu-table-inspector';

async function fixture() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    value: arrow.vectorFromArray([3, null, 1], new arrow.Float32()),
    group: arrow.makeVector(new Uint32Array([1, 0, 1]))
  });
  const second = new arrow.Table({
    value: arrow.vectorFromArray([3, -2, NaN], new arrow.Float32()),
    group: arrow.makeVector(new Uint32Array([1, 0, 1]))
  });
  const dataset = session.importArrowDataset({id: 'table', table: first.concat(second)});
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [{id: 'group', kind: 'category', column: 'group', capacity: 2}],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const inspector = new GPUKeplerTableInspector({
    session,
    explorer,
    sortMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const render = () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    device.submit(encoder.finish());
  };
  render();
  return {
    device,
    session,
    dataset,
    explorer,
    inspector,
    render,
    async destroy() {
      await inspector.destroy();
      const fence = device.createFence();
      try {
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  };
}

test('native table sorts across source batches with stable ties and bounded page transfers', async () => {
  const context = await fixture();
  try {
    const {inspector} = context;
    const request = {
      offset: 0,
      limit: 2,
      columns: ['value', 'group'],
      population: 'all' as const,
      sort: {column: 'value', direction: 'ascending' as const}
    };
    const first = await inspector.readPage(request);
    expect(first?.totalCount).toBe(6);
    expect(Array.from(first!.sourceRowIndices)).toEqual([4, 2]);
    expect(Array.from(first!.columns['value'])).toEqual([-2, 1]);
    const second = await inspector.readPage({...request, offset: 2});
    expect(Array.from(second!.sourceRowIndices)).toEqual([0, 3]);
    const last = await inspector.readPage({...request, offset: 4});
    expect(Array.from(last!.sourceRowIndices)).toEqual([5, 1]);
    expect(last!.validity['value'][1]).toBe(0);
    expect(inspector.readbackByteLength).toBe(6 * 4);
    expect(context.dataset.value.columns['value'].data.map(chunk => chunk.length)).toEqual([3, 3]);
    const descending = await inspector.readPage({
      ...request,
      limit: 6,
      sort: {column: 'value', direction: 'descending'}
    });
    expect(Array.from(descending!.sourceRowIndices)).toEqual([0, 3, 2, 4, 5, 1]);
    const nonNullable = await inspector.readPage({
      ...request,
      limit: 6,
      sort: {column: 'group', direction: 'ascending'}
    });
    expect(Array.from(nonNullable!.sourceRowIndices)).toEqual([1, 4, 0, 2, 3, 5]);
  } finally {
    await context.destroy();
  }
});

test('native table selection, pagination and source rows agree after filter changes', async () => {
  const context = await fixture();
  try {
    context.explorer.setCategories('group', [1]);
    context.render();
    const request = {
      offset: 0,
      limit: 2,
      columns: ['value'],
      population: 'selected' as const,
      sort: {column: 'value', direction: 'descending' as const}
    };
    const page = await context.inspector.readPage(request);
    expect(page?.totalCount).toBe(4);
    expect(Array.from(page!.sourceRowIndices)).toEqual([0, 3]);
    const next = await context.inspector.readPage({...request, offset: 2});
    expect(Array.from(next!.sourceRowIndices)).toEqual([2, 5]);
    context.explorer.setCategories('group', [0]);
    context.render();
    const changed = await context.inspector.readPage(request);
    expect(Array.from(changed!.sourceRowIndices)).toEqual([4, 1]);
    context.explorer.setEmpty('group');
    context.render();
    const empty = await context.inspector.readPage(request);
    expect(empty?.totalCount).toBe(0);
    expect(empty?.sourceRowIndices).toHaveLength(0);
  } finally {
    await context.destroy();
  }
});

test('native table supersedes queued requests and releases borrowed sources after inspection', async () => {
  const context = await fixture();
  try {
    const request = {offset: 0, limit: 2, columns: ['value'], population: 'all' as const};
    const first = context.inspector.readPage(request);
    const second = context.inspector.readPage({...request, offset: 2});
    expect(await first).toBeNull();
    expect(Array.from((await second)!.sourceRowIndices)).toEqual([2, 3]);
    const pending = context.inspector.readPage(request);
    context.inspector.cancel();
    expect(await pending).toBeNull();
    await context.inspector.destroy();
    expect(await context.explorer.readRows([0], ['value'])).not.toBeNull();
  } finally {
    await context.destroy();
  }
});

test('derived table columns preserve source identity, nullable populations and revision guards', async () => {
  const context = await fixture();
  let current = true;
  const inspector = new GPUKeplerTableInspector({
    session: context.session,
    explorer: context.explorer,
    sortMemoryBudgetBytes: 8 * 1024 * 1024,
    getDerivedColumns: () => ({
      revision: String(context.explorer.currentRevision),
      columns: {computed: context.dataset.value.columns['value']},
      validity: {computed: context.explorer.selectionMask},
      isCurrent: () => current
    })
  });
  try {
    context.explorer.setCategories('group', [1]);
    context.render();
    const request = {
      offset: 0,
      limit: 6,
      columns: ['computed'],
      population: 'all' as const,
      sort: {column: 'computed', direction: 'ascending' as const}
    };
    const page = await inspector.readPage(request);
    expect(Array.from(page!.sourceRowIndices)).toEqual([2, 0, 3, 5, 1, 4]);
    expect(Array.from(page!.validity['computed'])).toEqual([1, 1, 1, 1, 0, 0]);
    expect(Array.from(page!.columns['computed']).slice(0, 3)).toEqual([1, 3, 3]);
    expect(inspector.readbackByteLength).toBe(6 * 4 * 3);
    current = false;
    expect(await inspector.readPage(request)).toBeNull();
  } finally {
    await inspector.destroy();
    await context.destroy();
  }
});
