// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import {ResidentNumericChartRuntime} from '../../../examples/deck/kepler-compute/src/resident-chart-numeric-runtime';
import type {ResidentNumericChartPlan} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import type {ResidentChartSelection} from '../../../examples/deck/kepler-compute/src/resident-chart-group-runtime';

test('numeric chart reductions preserve nullable source batches, empty bins and native selected domains', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const table = new arrow.Table({
    value: arrow.vectorFromArray([0, null, 1, 2, NaN, 3, 4, Infinity], new arrow.Float32()),
    amount: arrow.vectorFromArray([10, 99, null, 30, 99, 40, 50, 99], new arrow.Float32()),
    owned: arrow.makeVector(new Uint32Array([0, 0, 0, 1, 0, 1, 1, 0])),
    other: arrow.makeVector(new Uint32Array([0, 1, 1, 1, 1, 1, 1, 1]))
  });
  const dataset = session.importArrowDataset({
    id: 'numeric',
    table: table.slice(0, 3).concat(table.slice(3))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const mask = (name: string) =>
    new GPUVector({
      name,
      type: 'data',
      format: 'uint32',
      data: dataset.value.columns[name].data,
      ownsData: false
    });
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: dataset.value.columns['value'],
    scalarFilters: [
      {
        filterId: 'owned',
        mask: mask('owned'),
        nativeTopology: false,
        nativeFieldName: 'value',
        dynamicDomain: true
      },
      {
        filterId: 'other',
        mask: mask('other'),
        nativeTopology: true,
        nativeFieldName: 'rank',
        dynamicDomain: false
      }
    ]
  });
  let blocked = false;
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    get canConsume() {
      return !blocked;
    },
    isLayerRegistrationPending: false,
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => {
      scopes.encode(encoder);
    }
  };
  const readers: ResidentNumericChartRuntime[] = [];
  const create = (
    operation: ResidentNumericChartPlan['operation'],
    population: 'all' | 'selected' = 'all',
    exclude = false,
    budget = 8 * 1024 * 1024
  ) => {
    const reader = new ResidentNumericChartRuntime({
      session,
      explorer,
      selection,
      workingMemoryBudgetBytes: budget,
      plan: {
        kind: 'numeric',
        chartId: `numeric-${readers.length}`,
        datasetId: dataset.id,
        operation,
        population,
        ...(operation === 'count' ? {} : {column: 'amount'}),
        binColumn: 'value',
        binCount: 4,
        binFormat: 'float32',
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: exclude ? ['value'] : []
      }
    });
    readers.push(reader);
    return reader;
  };
  const memory = session.registry.memoryUsage;
  const buffers = Object.values(dataset.value.columns).flatMap(vector =>
    vector.data.map(chunk => chunk.buffer)
  );
  try {
    const expectations = {
      count: [1, 1, 1, 2],
      sum: [10, 0, 30, 90],
      mean: [10, 0, 30, 45],
      min: [10, 0, 30, 40],
      max: [10, 0, 30, 50]
    };
    for (const operation of ['count', 'sum', 'mean', 'min', 'max'] as const) {
      const result = await create(operation).read();
      expect(result?.boundaries).toEqual([0, 1, 2, 3, 4]);
      expect(result?.summaries.map(summary => summary.value)).toEqual(expectations[operation]);
      expect(result?.summaries.map(summary => summary.count)).toEqual([1, 1, 1, 2]);
      expect(result?.readbackBytes).toBe(12 + 4 * (operation === 'count' ? 12 : 16));
    }
    const selected = await create('count', 'selected').read();
    expect(selected?.boundaries).toEqual([2, 2.5, 3, 3.5, 4]);
    expect(selected?.summaries.map(summary => summary.count)).toEqual([1, 0, 1, 1]);
    const excluding = await create('count', 'selected', true).read();
    expect(excluding?.boundaries).toEqual([1, 1.75, 2.5, 3.25, 4]);
    expect(excluding?.summaries.map(summary => summary.count)).toEqual([1, 1, 1, 1]);
    const guarded = create('count', 'selected');
    blocked = true;
    expect(await guarded.read()).toBeNull();
    expect(guarded.readbackByteLength).toBe(0);
    blocked = false;
    const changingSelection = create('count', 'selected');
    const staleSelection = changingSelection.read();
    // Invalidate after extent encoding but before its readback completes: phases must not mix revisions.
    await Promise.resolve();
    scopes.invalidate();
    expect(await staleSelection).toBeNull();
    expect((await changingSelection.read())?.boundaries).toEqual([2, 2.5, 3, 3.5, 4]);
    const interrupted = create('count');
    const pending = interrupted.read();
    await Promise.resolve();
    interrupted.cancel();
    expect(await pending).toBeNull();
    expect((await interrupted.read())?.boundaries).toEqual([0, 1, 2, 3, 4]);
    expect(() => create('count', 'all', false, 1)).toThrow('before allocation');
    expect(
      Object.values(dataset.value.columns).flatMap(vector => vector.data.map(chunk => chunk.buffer))
    ).toEqual(buffers);
    expect(dataset.value.source.batches.map(batch => batch.numRows)).toEqual([3, 5]);
  } finally {
    await Promise.all(readers.map(reader => reader.destroy()));
    expect(session.registry.memoryUsage).toEqual(memory);
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('numeric chart classification retains subnormal and extreme integer edges without float arithmetic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const minimumSubnormal = new Float32Array(new Uint32Array([1]).buffer)[0];
  const fixtures = [
    {
      format: 'float32' as const,
      values: [-minimumSubnormal, 0, minimumSubnormal],
      type: new arrow.Float32()
    },
    {format: 'float32' as const, values: [1, 1 + 2 ** -23], type: new arrow.Float32()},
    {format: 'float32' as const, values: [7, 7, 7], type: new arrow.Float32()},
    {format: 'float32' as const, values: [NaN, Infinity, -Infinity], type: new arrow.Float32()},
    {format: 'sint32' as const, values: [-2147483648, -1, 0, 2147483647], type: new arrow.Int32()},
    {
      format: 'uint32' as const,
      values: [4294967293, 4294967294, 4294967295],
      type: new arrow.Uint32()
    }
  ];
  for (const fixture of fixtures) {
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
    const source = new arrow.Table({value: arrow.vectorFromArray(fixture.values, fixture.type)});
    const dataset = session.importArrowDataset({
      id: 'numeric',
      table: source.slice(0, 1).concat(source.slice(1))
    });
    const explorer = session.openExplorer(dataset.id, {
      dimensions: [],
      workingMemoryBudgetBytes: 8 * 1024 * 1024
    });
    const reader = new ResidentNumericChartRuntime({
      session,
      explorer,
      workingMemoryBudgetBytes: 8 * 1024 * 1024,
      plan: {
        kind: 'numeric',
        chartId: 'edge',
        datasetId: dataset.id,
        population: 'all',
        operation: 'count',
        binColumn: 'value',
        binFormat: fixture.format,
        binCount: 3,
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: []
      }
    });
    try {
      const result = await reader.read();
      const values = fixture.values.filter(Number.isFinite);
      if (!values.length) {
        expect(result).toMatchObject({boundaries: [], summaries: [], readbackBytes: 12});
        continue;
      }
      expect(result?.boundaries[0]).toBe(Math.min(...values));
      expect(result?.boundaries.at(-1)).toBe(Math.max(...values));
      const expected = [0, 0, 0];
      for (const value of values)
        expected[result!.boundaries.slice(1, -1).filter(threshold => value >= threshold).length]++;
      expect(result?.summaries.map(summary => summary.count)).toEqual(expected);
    } finally {
      await reader.destroy();
      const fence = device.createFence();
      try {
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  }
});
