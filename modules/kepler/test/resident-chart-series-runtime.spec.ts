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
import {ResidentNumericChartRuntime} from '../../../examples/deck/kepler-compute/src/resident-chart-numeric-runtime';
import type {ResidentCategoryChartPlan} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

test('GPU series preserve missing-series counts, chunk boundaries, native sum-of-means, scopes and ownership', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const table = new arrow.Table({
    category: arrow.vectorFromArray([
      'Beta',
      'Alpha',
      'Beta',
      'Alpha',
      'Beta',
      'Alpha',
      'Beta',
      'Alpha',
      'Gamma',
      'Delta',
      '',
      null
    ]),
    series: arrow.vectorFromArray([
      null,
      'Two',
      'One',
      'One',
      'One',
      'Two',
      'Two',
      '',
      'Two',
      null,
      'One',
      'Two'
    ]),
    amount: arrow.vectorFromArray(
      [100, 4, 2, 8, 6, null, 10, 200, null, 400, 600, 800],
      new arrow.Float32()
    ),
    numeric: arrow.vectorFromArray(
      [0, 2, 0, 2, 0, 2, 0, 2, null, null, null, null],
      new arrow.Int32()
    ),
    owned: arrow.makeVector(new Uint32Array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0])),
    other: arrow.makeVector(new Uint32Array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
  });
  const dataset = session.importArrowDataset({
    id: 'series',
    table: table.slice(0, 3).concat(table.slice(3, 7), table.slice(7))
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
        nativeFieldName: 'amount',
        dynamicDomain: false
      }
    ]
  });
  const selection: ResidentChartSelection = {
    get currentRevision() {
      return scopes.currentRevision;
    },
    isLayerRegistrationPending: false,
    canConsume: true,
    getConsumerMask: request => scopes.getConsumerMask(request),
    removeConsumerMask: id => scopes.removeConsumerMask(id),
    encode: encoder => {
      scopes.encode(encoder);
    }
  };
  const readers: (ResidentCategoryChartRuntime | ResidentNumericChartRuntime)[] = [];
  const baseline = session.registry.memoryUsage;
  const buffers = Object.values(dataset.value.columns).flatMap(vector =>
    vector.data.map(chunk => chunk.buffer)
  );
  const base: ResidentCategoryChartPlan = {
    kind: 'category',
    chartId: 'series',
    datasetId: dataset.id,
    operation: 'mean',
    population: 'all',
    categoryColumn: 'category',
    categoryCount: dataset.value.prepared.dictionaries['category'].values.length,
    column: 'amount',
    series: {
      column: 'series',
      categoryCount: dataset.value.prepared.dictionaries['series'].values.length
    },
    excludedCpuFilterIds: [],
    excludedGpuFieldNames: []
  };
  const create = (changes: Partial<ResidentCategoryChartPlan> = {}) => {
    const reader = new ResidentCategoryChartRuntime({
      session,
      explorer,
      selection,
      plan: {...base, ...changes, chartId: `series-${readers.length}`},
      workingMemoryBudgetBytes: 8 * 1024 * 1024
    });
    readers.push(reader);
    return reader;
  };
  try {
    for (const operation of ['count', 'sum', 'mean', 'min', 'max'] as const) {
      const result = await create({
        operation,
        column: operation === 'count' ? undefined : 'amount'
      }).read();
      const expected = {count: [3, 3], sum: [18, 12], mean: [14, 12], min: [12, 12], max: [16, 12]}[
        operation
      ];
      expect(
        ['Beta', 'Alpha'].map(
          category => result?.summaries.find(summary => summary.category === category)?.value
        )
      ).toEqual(expected);
      expect(result?.summaries.find(summary => summary.category === 'Beta')).toMatchObject({
        count: 4,
        firstRow: 0,
        series: [
          {category: 'One', count: 2, firstRow: 2},
          {category: 'Two', count: 1, firstRow: 6}
        ]
      });
      expect(result?.summaries.find(summary => summary.category === 'Alpha')).toMatchObject({
        count: 4,
        firstRow: 1,
        series: [
          {category: 'Two', count: 2, firstRow: 1},
          {category: 'One', count: 1, firstRow: 3}
        ]
      });
      expect(result?.summaries.find(summary => summary.category === 'Delta')).toMatchObject({
        count: 1,
        value: 0,
        series: []
      });
      expect(result?.readbackBytes).toBe(
        base.categoryCount * (base.series!.categoryCount + 1) * (operation === 'count' ? 12 : 16)
      );
    }
    const selected = await create({
      population: 'selected',
      excludedGpuFieldNames: ['category'],
      excludedCpuFilterIds: ['owned']
    }).read();
    expect(selected?.summaries.find(summary => summary.category === 'Beta')).toMatchObject({
      value: 16,
      count: 2,
      firstRow: 4,
      series: [
        {category: 'One', value: 6},
        {category: 'Two', value: 10}
      ]
    });
    const sameField = await create({
      series: {column: 'category', categoryCount: base.categoryCount}
    }).read();
    expect(sameField?.summaries.find(summary => summary.category === 'Beta')).toMatchObject({
      value: 29.5,
      count: 4,
      series: [{category: 'Beta', value: 29.5, count: 4}]
    });
    for (const integerGroupLimit of [undefined, 4]) {
      const reader = new ResidentNumericChartRuntime({
        session,
        explorer,
        selection,
        workingMemoryBudgetBytes: 8 * 1024 * 1024,
        plan: {
          kind: 'numeric',
          chartId: `numeric-${readers.length}`,
          datasetId: dataset.id,
          population: 'all',
          operation: 'mean',
          column: 'amount',
          binColumn: 'numeric',
          binFormat: 'sint32',
          binCount: 4,
          integerGroupLimit,
          series: base.series,
          excludedCpuFilterIds: [],
          excludedGpuFieldNames: []
        }
      });
      readers.push(reader);
      const result = await reader.read();
      expect(result?.summaries[0]).toMatchObject({
        value: 14,
        count: 4,
        series: [
          {category: 'One', value: 4},
          {category: 'Two', value: 10}
        ]
      });
      expect(result?.summaries[integerGroupLimit === undefined ? 3 : 1]).toMatchObject({
        value: 12,
        count: 4
      });
      expect(result?.summaries).toHaveLength(4);
    }
    const cancelled = create();
    const pending = cancelled.read();
    cancelled.cancel();
    expect(await pending).toBeNull();
    expect(() => create({series: {column: 'series', categoryCount: 999}})).toThrow('dictionary');
    expect(dataset.value.source.batches.map(batch => batch.numRows)).toEqual([3, 4, 5]);
    expect(
      Object.values(dataset.value.columns).flatMap(vector => vector.data.map(chunk => chunk.buffer))
    ).toEqual(buffers);
  } finally {
    await Promise.all(readers.map(reader => reader.destroy()));
    expect(session.registry.memoryUsage).toEqual(baseline);
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
