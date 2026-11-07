// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ResidentScalarChartRuntime} from '../../../examples/deck/kepler-compute/src/resident-chart-runtime';
import type {ResidentScalarChartPlan} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

async function fixture() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = new arrow.Table({
    value: arrow.vectorFromArray([3, null, 1, 3, -2, NaN], new arrow.Float32()),
    group: arrow.makeVector(new Uint32Array([1, 0, 1, 1, 0, 1]))
  });
  const dataset = session.importArrowDataset({
    id: 'charts',
    table: source.slice(0, 3).concat(source.slice(3))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [{id: 'group', kind: 'category', column: 'group', capacity: 2}],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const readers: ResidentScalarChartRuntime[] = [];
  const create = (
    operation: ResidentScalarChartPlan['operation'],
    population: ResidentScalarChartPlan['population'] = 'selected',
    budget = 8 * 1024 * 1024
  ) => {
    const reader = new ResidentScalarChartRuntime({
      session,
      explorer,
      workingMemoryBudgetBytes: budget,
      plan: {
        kind: 'scalar',
        chartId: `chart-${readers.length}`,
        datasetId: dataset.id,
        population,
        operation,
        ...(operation === 'count' ? {} : {column: 'value'})
      }
    });
    readers.push(reader);
    return reader;
  };
  return {
    device,
    session,
    dataset,
    explorer,
    create,
    async destroy() {
      await Promise.all(readers.map(reader => reader.destroy()));
      const fence = device.createFence();
      try {
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  };
}

test('native scalar chart measures respect nulls, batches, selected population and bounded output', async () => {
  const context = await fixture();
  const sourceBuffers = Object.values(context.dataset.value.columns).flatMap(vector =>
    vector.data.map(chunk => chunk.buffer)
  );
  const sourceSchema = context.dataset.value.source.schema;
  try {
    const expectations = {count: 6, sum: 5, mean: 1.25, min: -2, max: 3};
    for (const [operation, expected] of Object.entries(expectations)) {
      const reader = context.create(operation as ResidentScalarChartPlan['operation']);
      const result = await reader.read();
      expect(result?.value).toBeCloseTo(expected, 6);
      expect(result?.sourceRevision).toBe(context.dataset.revision);
      expect(reader.readbackByteLength).toBe(operation === 'count' ? 4 : 8);
    }
    context.explorer.setCategories('group', [1]);
    const selected = context.create('mean');
    expect((await selected.read())?.value).toBeCloseTo(7 / 3, 6);
    const count = context.create('count');
    expect((await count.read())?.value).toBe(4);
    const all = context.create('sum', 'all');
    expect(await all.read()).toMatchObject({value: 5, selectionRevision: null});
    expect(context.dataset.value.source.schema).toBe(sourceSchema);
    expect(context.dataset.value.source.batches.map(batch => batch.numRows)).toEqual([3, 3]);
    expect(
      Object.values(context.dataset.value.columns).flatMap(vector =>
        vector.data.map(chunk => chunk.buffer)
      )
    ).toEqual(sourceBuffers);
  } finally {
    await context.destroy();
  }
});

test('empty charts use native zero semantics and canceled or superseded reads cannot publish', async () => {
  const context = await fixture();
  try {
    context.explorer.setEmpty('group');
    const reader = context.create('mean');
    expect(await reader.read()).toMatchObject({value: 0, hasValue: false});
    const pending = reader.read();
    reader.cancel();
    expect(await pending).toBeNull();
    const previous = reader.read();
    context.explorer.clearAll();
    expect(await previous).toBeNull();
    expect((await reader.read())?.value).toBeCloseTo(1.25, 6);
    const afterDelete = reader.read();
    await reader.destroy();
    expect(await afterDelete).toBeNull();
  } finally {
    await context.destroy();
  }
});

test('chart memory admission rejects before GPU allocation and readers release reservations', async () => {
  const context = await fixture();
  try {
    const before = context.session.registry.memoryUsage;
    const allocation = vi.spyOn(context.device, 'createBuffer');
    expect(() => context.create('sum', 'selected', 0)).toThrow(/before allocation/);
    expect(allocation).not.toHaveBeenCalled();
    allocation.mockRestore();
    const reader = context.create('sum');
    await reader.read();
    await reader.destroy();
    expect(context.session.registry.memoryUsage).toEqual(before);
  } finally {
    await context.destroy();
  }
});

test('count charts use source-only row topology without uploading a synthetic numeric field', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 8 * 1024 * 1024});
  const source = new arrow.Table({
    geometry: arrow.vectorFromArray(['first', null, 'third'], new arrow.Utf8())
  });
  const dataset = session.importArrowDataset({
    id: 'geometry',
    table: source.slice(0, 1).concat(source.slice(1)),
    normalization: {columns: []}
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 65536,
    maximumReadbackBytes: 128,
    maximumPageSize: 2
  });
  const reader = new ResidentScalarChartRuntime({
    session,
    explorer,
    workingMemoryBudgetBytes: 1024 * 1024,
    plan: {
      kind: 'scalar',
      chartId: 'geometry-count',
      datasetId: dataset.id,
      operation: 'count',
      population: 'selected'
    }
  });
  try {
    expect(dataset.value.columns).toEqual({});
    expect(dataset.value.uploadedBufferBytes).toBe(0);
    expect(await reader.read()).toMatchObject({value: 3, readbackBytes: 4});
    expect(explorer.selectionMask.data.map(chunk => chunk.length)).toEqual([1, 2]);
    expect(dataset.value.source.schema.fields.map(field => field.name)).toEqual(['geometry']);
  } finally {
    await reader.destroy();
    const fence = device.createFence();
    try {
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('scalar charts compute exact median and deviation over the selected population', async () => {
  const context = await fixture();
  try {
    // Values 3, null, 1, 3, -2, NaN: null and NaN are skipped, leaving -2, 1, 3, 3.
    const median = await context.create('median', 'all').read();
    expect(median).toMatchObject({value: 2, hasValue: true, selectionRevision: null});
    const deviation = await context.create('standardDeviation', 'all').read();
    expect(deviation?.value).toBeCloseTo(Math.sqrt(16.75 / 3), 5);
    context.explorer.setCategories('group', [1]);
    // Group one keeps 3, 1, 3 and the NaN row.
    expect((await context.create('median').read())?.value).toBe(3);
    context.explorer.setEmpty('group');
    expect(await context.create('median').read()).toMatchObject({value: 0, hasValue: false});
  } finally {
    await context.destroy();
  }
});
