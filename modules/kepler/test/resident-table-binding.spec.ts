// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {ResidentTableBinding} from '../../../examples/deck/kepler-compute/src/resident-table-binding';
import {
  installKeplerComputeTableBridge,
  type KeplerComputeFilter
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

test('native point polygons and dynamic domains share GPU data while retaining independent selection scope', async () => {
  const availableDevice = await getWebGPUTestDevice();
  if (!availableDevice)
    throw new Error('WebGPU is required for native Kepler provider verification');
  const device = availableDevice;
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 32 * 1024 * 1024});
  const first = arrow.tableFromArrays({
    value: new Float32Array([10, 20]),
    category: new Uint32Array([0, 1])
  });
  const second = arrow.tableFromArrays({
    value: new Float32Array([30, 40, 50]),
    category: new Uint32Array([0, 1, 0])
  });
  session.importArrowDataset({id: 'observations', table: first.concat(second)});
  const positions = new GPUVector({
    name: 'resident-provider-points',
    type: 'data',
    format: 'float32x2',
    ownsData: true,
    data: [new Float32Array([0, 0, 1, 1]), new Float32Array([2, 2, 3, 3, 5, 5])].map(
      values =>
        new GPUData({
          buffer: device.createBuffer({data: values, usage: Buffer.STORAGE | Buffer.COPY_DST}),
          format: 'float32x2',
          length: values.length / 2,
          ownsBuffer: true
        })
    )
  });
  const errors: string[] = [];
  const provider = new ResidentTableBinding({
    device,
    session,
    datasetId: 'observations',
    positions,
    layerId: 'points',
    fields: {
      value: {dimension: 'value', kind: 'range', domain: [10, 50]},
      category: {dimension: 'category', kind: 'category', domain: [0, 1], categories: [0, 1]}
    },
    onUpdate() {},
    onUnsupported: reason => errors.push(reason),
    domainWorkingMemoryBudgetBytes: 8 * 1024 * 1024,
    spatialWorkingMemoryBudgetBytes: 2 * 1024 * 1024
  });
  const explorer = session.openExplorer('observations', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [
      {id: 'value', kind: 'range', column: 'value'},
      {id: 'category', kind: 'category', column: 'category', capacity: 2},
      provider.externalDimension
    ]
  });
  const table = installKeplerComputeTableBridge(
    {
      id: 'observations',
      fields: [
        {name: 'value', type: 'real'},
        {name: 'category', type: 'integer'}
      ],
      allIndexes: [0, 1, 2, 3, 4],
      filteredIndex: [0, 1, 2, 3, 4],
      filteredIndexForDomain: [0, 1, 2, 3, 4],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers: readonly unknown[]) {
        return this;
      },
      getColumnLayerDomain(_field: {name: string}, _scale: string): readonly unknown[] | null {
        return null;
      }
    },
    provider.bind(explorer)
  );
  const layers = [{id: 'points', type: 'point', config: {dataId: 'observations'}}];
  const numeric: KeplerComputeFilter = {
    id: 'numeric',
    dataId: ['observations'],
    name: ['value'],
    type: 'range',
    value: [20, 40],
    fixedDomain: false
  };
  const category: KeplerComputeFilter = {
    id: 'category',
    dataId: ['observations'],
    name: ['category'],
    type: 'multiSelect',
    value: [0],
    fixedDomain: true
  };
  const polygon: KeplerComputeFilter = {
    id: 'drawn',
    type: 'polygon',
    dataId: ['observations'],
    layerId: ['points'],
    fixedDomain: true,
    value: {
      type: 'Feature',
      id: 'drawn',
      properties: {filterId: 'drawn'},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-1, -1],
            [3, -1],
            [3, 3],
            [-1, 3],
            [-1, -1]
          ],
          [
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.5],
            [0.5, 1.5],
            [0.5, 0.5]
          ]
        ]
      }
    }
  };
  async function execute(filters: readonly KeplerComputeFilter[]): Promise<Uint32Array> {
    table.filterTable(filters, layers);
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    device.submit(encoder.finish());
    expect(await provider.readSummary()).toBe(true);
    const page = await explorer.readPage(0, 5, ['value']);
    return page!.sourceRowIndices;
  }
  try {
    expect(Array.from(await execute([numeric, category, polygon]))).toEqual([2]);
    // Fixed categorical display filter and layer-specific polygon do not affect the domain scope.
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([20, 40]);
    expect(table.getColumnLayerDomain({name: 'category'}, 'ordinal')).toEqual([0, 1]);
    expect(explorer.selectionMask.data.map(chunk => chunk.length)).toEqual([2, 3]);

    // Native Turf within excludes outer/hole boundaries and holes; separate filters intersect.
    expect(Array.from(await execute([polygon]))).toEqual([0, 2]);
    const movedPolygon: KeplerComputeFilter = {
      ...polygon,
      value: {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [4, 0],
              [4, 4],
              [0, 4],
              [0, 0]
            ],
            [
              [1.5, 1.5],
              [2.5, 1.5],
              [2.5, 2.5],
              [1.5, 2.5],
              [1.5, 1.5]
            ]
          ]
        }
      }
    };
    expect(Array.from(await execute([movedPolygon]))).toEqual([1, 3]);
    expect(Array.from(await execute([{...polygon, inverted: true}]))).toEqual([1, 3, 4]);
    const secondPolygon: KeplerComputeFilter = {
      ...polygon,
      id: 'second',
      value: {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [1.8, 1.8],
              [4, 1.8],
              [4, 4],
              [1.8, 4]
            ]
          ]
        }
      }
    };
    expect(Array.from(await execute([polygon, secondPolygon]))).toEqual([2]);

    // A new topology replaces the compiled containment input without changing the explorer's mask.
    const stableMask = explorer.selectionMask;
    const union: KeplerComputeFilter = {
      ...polygon,
      value: {
        type: 'Feature',
        geometry: {
          type: 'MultiPolygon',
          coordinates: [
            [
              [
                [-1, -1],
                [0.5, -1],
                [0.5, 0.5],
                [-1, 0.5]
              ]
            ],
            [
              [
                [4, 4],
                [6, 4],
                [6, 6],
                [4, 6]
              ]
            ]
          ]
        }
      }
    };
    expect(Array.from(await execute([union]))).toEqual([0, 4]);
    expect(explorer.selectionMask).toBe(stableMask);
    expect(
      Array.from(await execute([numeric, {...numeric, id: 'disjoint', value: [45, 50]}]))
    ).toEqual([]);
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toBeNull();
    expect(Array.from(await execute([]))).toEqual([0, 1, 2, 3, 4]);
    expect(errors).toEqual([]);
  } finally {
    const fence = device.createFence();
    try {
      await provider.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
      positions.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
