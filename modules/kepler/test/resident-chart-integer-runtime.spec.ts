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

test('integer auto switches between intervals and selected unique groups without replacing source buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = new arrow.Table({
    value: arrow.vectorFromArray(
      [4294967295, 4294967294, 4294967293, 4294967295, null],
      new arrow.Uint32()
    ),
    amount: arrow.vectorFromArray([10, null, 30, 50, 99], new arrow.Float32()),
    subset: arrow.makeVector(new Uint32Array([1, 1, 0, 1, 0]))
  });
  const dataset = session.importArrowDataset({
    id: 'integers',
    table: source.slice(0, 2).concat(source.slice(2))
  });
  const explorer = session.openExplorer(dataset.id, {
    dimensions: [],
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const scopes = new GPUKeplerSelectionScopes({device, rows: dataset.value.columns['value']});
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    canConsume: true,
    isLayerRegistrationPending: false,
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => {
      scopes.encode(encoder);
    }
  };
  const readers: ResidentNumericChartRuntime[] = [];
  const create = (
    integerGroupLimit: number,
    operation: ResidentNumericChartPlan['operation'] = 'count'
  ) => {
    const reader = new ResidentNumericChartRuntime({
      session,
      explorer,
      selection,
      workingMemoryBudgetBytes: 8 * 1024 * 1024,
      plan: {
        kind: 'numeric',
        chartId: `integer-${readers.length}`,
        datasetId: dataset.id,
        population: 'selected',
        operation,
        ...(operation !== 'count' ? {column: 'amount'} : {}),
        binColumn: 'value',
        binFormat: 'uint32',
        binCount: 2,
        integerGroupLimit,
        excludedCpuFilterIds: [],
        excludedGpuFieldNames: []
      }
    });
    readers.push(reader);
    return reader;
  };
  const memory = session.registry.memoryUsage;
  const buffers = dataset.value.columns['value'].data.map(chunk => chunk.buffer);
  try {
    const reader = create(2);
    const intervals = await reader.read();
    expect(intervals?.categoryValues).toBeUndefined();
    expect(intervals?.boundaries).toEqual([4294967293, 4294967294, 4294967295]);
    expect(intervals?.summaries.slice(0, 2).map(summary => summary.count)).toEqual([1, 3]);
    const mask = new GPUVector({
      name: 'subset',
      type: 'data',
      format: 'uint32',
      data: dataset.value.columns['subset'].data,
      ownsData: false
    });
    scopes.setFilters(
      [
        {
          filterId: 'subset',
          mask,
          nativeTopology: true,
          nativeFieldName: 'subset',
          dynamicDomain: true
        }
      ],
      []
    );
    const unique = await reader.read();
    expect(unique?.categoryValues).toEqual([4294967294, 4294967295]);
    expect(unique?.boundaries).toEqual([]);
    expect(unique?.summaries.slice(0, 2).map(summary => [summary.count, summary.firstRow])).toEqual(
      [
        [1, 1],
        [2, 0]
      ]
    );
    const mean = await create(2, 'mean').read();
    expect(mean?.categoryValues).toEqual([4294967294, 4294967295]);
    expect(mean?.summaries.slice(0, 2).map(summary => summary.value)).toEqual([0, 30]);
    scopes.setFilters([], []);
    const unlimited = await create(0).read();
    expect(unlimited?.categoryValues).toEqual([4294967293, 4294967294, 4294967295]);
    expect(unlimited?.summaries.slice(0, 3).map(summary => summary.count)).toEqual([1, 1, 2]);
    const pending = reader.read();
    await Promise.resolve();
    scopes.invalidate();
    expect(await pending).toBeNull();
    expect((await reader.read())?.categoryValues).toBeUndefined();
    expect(dataset.value.source.batches.map(batch => batch.numRows)).toEqual([2, 3]);
    expect(dataset.value.columns['value'].data.map(chunk => chunk.buffer)).toEqual(buffers);
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

test('integer auto preserves canonical nonfinite categories, signed extrema and bounded cardinality overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixtures = [
    {
      format: 'float32' as const,
      type: new arrow.Float32(),
      values: [NaN, NaN, -0, 0, Infinity, -Infinity, 3, null]
    },
    {
      format: 'sint32' as const,
      type: new arrow.Int32(),
      values: [-2147483648, 2147483647, -1, 2147483647, null]
    },
    {
      format: 'uint32' as const,
      type: new arrow.Uint32(),
      values: [...Array.from({length: 512}, (_, index) => index), 4294967295]
    },
    {
      format: 'uint32' as const,
      type: new arrow.Uint32(),
      values: Array.from({length: 1537}, (_, index) => index)
    }
  ];
  for (const fixture of fixtures) {
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
    const source = new arrow.Table({value: arrow.vectorFromArray(fixture.values, fixture.type)});
    const dataset = session.importArrowDataset({
      id: 'integer-edge',
      table: source.slice(0, 2).concat(source.slice(2))
    });
    const explorer = session.openExplorer(dataset.id, {
      dimensions: [],
      workingMemoryBudgetBytes: 8 * 1024 * 1024
    });
    const readers: ResidentNumericChartRuntime[] = [];
    const create = (integerGroupLimit: number) => {
      const reader = new ResidentNumericChartRuntime({
        session,
        explorer,
        workingMemoryBudgetBytes: 8 * 1024 * 1024,
        plan: {
          kind: 'numeric',
          chartId: `edge-${readers.length}`,
          datasetId: dataset.id,
          population: 'all',
          operation: 'count',
          binColumn: 'value',
          binFormat: fixture.format,
          binCount: 3,
          integerGroupLimit,
          excludedCpuFilterIds: [],
          excludedGpuFieldNames: []
        }
      });
      readers.push(reader);
      return reader;
    };
    const memory = session.registry.memoryUsage;
    try {
      const reader = create(0);
      if (fixture.values.length > 1000) {
        await expect(reader.read()).rejects.toThrow('bounded summary budget');
        const fallback = await create(3).read();
        expect(fallback?.categoryValues).toBeUndefined();
        expect(fallback?.summaries.reduce((sum, group) => sum + group.count, 0)).toBe(
          fixture.values.length
        );
      } else {
        const result = await reader.read();
        const expected = new Map<string, {count: number; firstRow: number}>();
        fixture.values.forEach((value, row) => {
          if (value === null) return;
          const key = String(value);
          const group = expected.get(key) ?? {count: 0, firstRow: row};
          group.count++;
          expected.set(key, group);
        });
        expect(result?.categoryValues?.length).toBe(expected.size);
        for (const [index, value] of result!.categoryValues!.entries())
          expect(result?.summaries[index]).toMatchObject(expected.get(String(value))!);
        expect(result?.summaries.reduce((sum, group) => sum + group.count, 0)).toBe(
          fixture.values.filter(value => value !== null).length
        );
      }
    } finally {
      await Promise.all(readers.map(reader => reader.destroy()));
      expect(session.registry.memoryUsage).toEqual(memory);
      const fence = device.createFence();
      try {
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  }
});
