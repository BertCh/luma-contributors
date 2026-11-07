// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import {
  ResidentCategoryChartRuntime,
  type ResidentChartSelection
} from '../../../examples/deck/kepler-compute/src/resident-chart-group-runtime';
import {widenResidentChartMeasure} from '../../../examples/deck/kepler-compute/src/resident-chart-measure-widening';
import type {ResidentCategoryChartPlan} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

test('category charts preserve chunks, validity, exact first rows and self-excluding native scopes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = new arrow.Table({
    category: arrow.vectorFromArray([
      'Beta',
      'Alpha',
      'Gamma',
      'Beta',
      'Alpha',
      'Gamma',
      '',
      null,
      'Delta'
    ]),
    amount: arrow.vectorFromArray([4, null, -2, 2, 3, 3, 100, 100, null], new arrow.Float32()),
    owned: arrow.makeVector(new Uint32Array([1, 0, 0, 1, 0, 0, 0, 0, 0])),
    other: arrow.makeVector(new Uint32Array([0, 1, 1, 1, 1, 1, 1, 1, 1]))
  });
  const dataset = session.importArrowDataset({
    id: 'categories',
    table: source.slice(0, 3).concat(source.slice(3))
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
    rows: dataset.value.columns['category'],
    scalarFilters: [
      {
        filterId: 'owned',
        mask: mask('owned'),
        nativeTopology: true,
        nativeFieldName: 'category',
        dynamicDomain: false
      },
      {
        filterId: 'other',
        mask: mask('other'),
        nativeTopology: true,
        nativeFieldName: 'category',
        dynamicDomain: false
      }
    ]
  });
  let blockConsumption = false;
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    isLayerRegistrationPending: false,
    get canConsume() {
      return !blockConsumption;
    },
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => {
      scopes.encode(encoder);
    }
  };
  const readers: ResidentCategoryChartRuntime[] = [];
  const create = (
    operation: ResidentCategoryChartPlan['operation'],
    population: 'all' | 'selected' = 'all',
    exclude = false,
    workingMemoryBudgetBytes = 8 * 1024 * 1024
  ) => {
    const reader = new ResidentCategoryChartRuntime({
      session,
      explorer,
      selection,
      workingMemoryBudgetBytes,
      plan: {
        kind: 'category',
        chartId: `test-${readers.length}`,
        datasetId: dataset.id,
        operation,
        population,
        categoryColumn: 'category',
        categoryCount: dataset.value.prepared.dictionaries['category'].values.length,
        ...(operation === 'count' ? {} : {column: 'amount'}),
        excludedCpuFilterIds: exclude ? ['owned'] : [],
        excludedGpuFieldNames: exclude ? ['category'] : []
      }
    });
    readers.push(reader);
    return reader;
  };
  const originalBuffers = Object.values(dataset.value.columns).flatMap(vector =>
    vector.data.map(chunk => chunk.buffer)
  );
  const originalMemoryUsage = session.registry.memoryUsage;
  try {
    const expectations = {
      count: [2, 2, 2, 1],
      sum: [6, 3, 1, 0],
      mean: [3, 3, 0.5, 0],
      min: [2, 3, -2, 0],
      max: [4, 3, 3, 0]
    };
    for (const operation of ['count', 'sum', 'mean', 'min', 'max'] as const) {
      const reader = create(operation);
      const result = await reader.read();
      const summaries = ['Beta', 'Alpha', 'Gamma', 'Delta'].map(category =>
        result?.summaries.find(summary => summary.category === category)
      );
      expect(summaries.map(summary => summary?.value)).toEqual(expectations[operation]);
      expect(summaries.map(summary => summary?.firstRow)).toEqual([0, 1, 2, 8]);
      expect(summaries.map(summary => summary?.count)).toEqual([2, 2, 2, 1]);
      expect(result?.readbackBytes).toBe(
        dataset.value.prepared.dictionaries['category'].values.length *
          (operation === 'count' ? 12 : 16)
      );
    }
    const selected = await create('count', 'selected').read();
    expect(selected?.summaries.filter(summary => summary.count)).toEqual([
      {category: 'Beta', value: 1, count: 1, firstRow: 3}
    ]);
    const excluding = await create('mean', 'selected', true).read();
    expect(excluding?.summaries.find(summary => summary.category === 'Beta')).toEqual({
      category: 'Beta',
      value: 2,
      count: 1,
      firstRow: 3
    });
    expect(excluding?.summaries.find(summary => summary.category === 'Alpha')).toEqual({
      category: 'Alpha',
      value: 3,
      count: 2,
      firstRow: 1
    });
    const guarded = create('count', 'selected');
    blockConsumption = true;
    expect(await guarded.read()).toBeNull();
    expect(guarded.readbackByteLength).toBe(0);
    blockConsumption = false;
    expect(
      (await guarded.read())?.summaries.find(summary => summary.category === 'Beta')?.count
    ).toBe(1);
    const cancelled = create('count');
    const pending = cancelled.read();
    cancelled.cancel();
    expect(await pending).toBeNull();
    expect(() => create('mean', 'all', false, 1)).toThrow('before allocation');
    expect(
      Object.values(dataset.value.columns).flatMap(vector => vector.data.map(chunk => chunk.buffer))
    ).toEqual(originalBuffers);
    expect(dataset.value.source.batches.map(batch => batch.numRows)).toEqual([3, 6]);
  } finally {
    await Promise.all(readers.map(reader => reader.destroy()));
    expect(session.registry.memoryUsage).toEqual(originalMemoryUsage);
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('category charts reduce integer measures exactly and offer median and deviation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = new arrow.Table({
    category: arrow.vectorFromArray(['Beta', 'Alpha', 'Gamma', 'Beta', 'Alpha', 'Gamma', 'Delta']),
    amount: arrow.vectorFromArray([4, null, -2, 2, 3, 3, null], new arrow.Float32()),
    whole: arrow.vectorFromArray([4, null, -2, 2, 3, 3, null], new arrow.Int32()),
    large: arrow.vectorFromArray([1, 2, 3, 4, 5, 6, 2 ** 24 + 1], new arrow.Int32())
  });
  const dataset = session.importArrowDataset({
    id: 'integer-measures',
    table: source.slice(0, 3).concat(source.slice(3))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const readers: ResidentCategoryChartRuntime[] = [];
  const widened: {destroy(): void}[] = [];
  const read = async (operation: ResidentCategoryChartPlan['operation'], measureColumn: string) => {
    const vector = dataset.value.columns[measureColumn];
    const measure =
      vector.format === 'float32'
        ? undefined
        : await widenResidentChartMeasure({
            session,
            vector: vector as GPUVector<'sint32'> | GPUVector<'uint32'>
          });
    if (measure) widened.push(measure);
    const reader = new ResidentCategoryChartRuntime({
      session,
      explorer,
      workingMemoryBudgetBytes: 8 * 1024 * 1024,
      ...(measure ? {measure: measure.vector} : {}),
      plan: {
        kind: 'category',
        chartId: `integer-${readers.length}`,
        datasetId: dataset.id,
        operation,
        population: 'all',
        categoryColumn: 'category',
        categoryCount: dataset.value.prepared.dictionaries['category'].values.length,
        column: measureColumn,
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: []
      }
    });
    readers.push(reader);
    const result = await reader.read();
    return ['Beta', 'Alpha', 'Gamma', 'Delta'].map(
      category => result?.summaries.find(summary => summary.category === category)?.value
    );
  };
  try {
    expect(await read('sum', 'whole')).toEqual([6, 3, 1, 0]);
    expect(await read('mean', 'whole')).toEqual([3, 3, 0.5, 0]);
    expect(await read('max', 'whole')).toEqual([4, 3, 3, 0]);
    // Alpha and Delta have fewer than two valid values, so their deviation is undefined (zero).
    expect(await read('median', 'amount')).toEqual([3, 3, 0.5, 0]);
    expect(await read('median', 'whole')).toEqual([3, 3, 0.5, 0]);
    const deviation = await read('standardDeviation', 'whole');
    expect(deviation[0]).toBeCloseTo(Math.SQRT2, 5);
    expect(deviation[1]).toBe(0);
    expect(deviation[2]).toBeCloseTo(Math.sqrt(12.5), 5);
    expect(deviation[3]).toBe(0);
    // Values beyond the exact Float32 integer range are rejected rather than rounded.
    await expect(read('sum', 'large')).rejects.toThrow('2^24');
  } finally {
    await Promise.all(readers.map(reader => reader.destroy()));
    widened.forEach(measure => measure.destroy());
    const fence = device.createFence();
    try {
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
