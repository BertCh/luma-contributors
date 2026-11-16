// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  getTiledKeyUnionNodes,
  validateTiledKeyUnionRowCount
} from '../../src/analysis-core/kernels/tiled-key-union';
import {GPUCellAggregation} from '../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {GPUGroupStatistics} from '../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources
} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from './kernel-audit/kernel-audit-wgsl';

function createFixture(tileCount: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const tiles = Array.from({length: tileCount}, (_, index) => {
    const backing = createTransientView(graph, `keys-${index}`, 'uint32x2', 8);
    const keys = graph.createDataView(backing.buffer, {
      format: 'uint32x2',
      length: 4,
      byteOffset: 4 * (index + 1)
    });
    return {
      keys,
      counts: createTransientView(graph, `counts-${index}`, 'uint32', 4),
      count: createTransientView(graph, `count-${index}`, 'uint32', 1)
    };
  });
  const output = {
    keys: createTransientView(graph, 'keys', 'uint32x2', 4),
    counts: createTransientView(graph, 'counts', 'uint32', 4)
  };
  const status = {
    count: createTransientView(graph, 'count', 'uint32', 1),
    overflow: createTransientView(graph, 'overflow', 'uint32', 1),
    requiredCount: createTransientView(graph, 'required', 'uint32', 1)
  };
  return {device, graph, tiles, output, status};
}

it('reuses shaders and pipelines with changing tile count and binding prefixes', () => {
  const keys = [2, 3, 5].map(tileCount => {
    const fixture = createFixture(tileCount);
    try {
      const nodes = getTiledKeyUnionNodes(
        fixture.graph,
        'union',
        fixture.tiles,
        fixture.output,
        fixture.status
      ).nodes;
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
      return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    } finally {
      fixture.device.destroy();
    }
  });
  expect([...keys[0]].sort()).toEqual([...keys[1]].sort());
  expect([...keys[0]].sort()).toEqual([...keys[2]].sort());
});

it('prechecks aliases, padded widths and uniform limits before any declaration', () => {
  for (const failure of ['alias', 'width', 'uniform'] as const) {
    const {device, graph, tiles, output, status} = createFixture(2);
    try {
      if (failure === 'alias') output.counts = tiles[0].counts;
      if (failure === 'width') {
        tiles[1] = {
          keys: createTransientView(graph, 'short-keys', 'uint32x2', 2),
          counts: createTransientView(graph, 'short-counts', 'uint32', 2),
          count: tiles[1].count
        };
      }
      if (failure === 'uniform') device.limits.maxUniformBufferBindingSize = 128;
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      expect(() => getTiledKeyUnionNodes(graph, 'union', tiles, output, status)).toThrow();
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('bounds global prefix totals and checks derived scratch allocations before declaration', () => {
  expect(() => validateTiledKeyUnionRowCount('union', 0xffffffff)).not.toThrow();
  expect(() => validateTiledKeyUnionRowCount('union', 0x100000000)).toThrow(/padded tile rows/);
  const {device, graph, output, status} = createFixture(0);
  try {
    const tiles = [
      {
        keys: createTransientView(graph, 'wide-keys', 'uint32x2', 128),
        counts: createTransientView(graph, 'wide-counts', 'uint32', 128),
        count: createTransientView(graph, 'wide-count', 'uint32', 1)
      }
    ];
    // Uniforms fit, and storage bindings fit; the derived 512-byte arrays cannot be allocated.
    device.limits.maxBufferSize = 256;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    expect(() => getTiledKeyUnionNodes(graph, 'union', tiles, output, status)).toThrow(
      /scratch byteLength/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('allows separate columns in one alignment window while rejecting co-bound status aliases', () => {
  const {device, graph, tiles, status} = createFixture(2);
  try {
    const buffer = graph.createTransientBuffer({id: 'columns', byteLength: 256, usage: 128});
    const output = {
      keys: graph.createDataView(buffer, {format: 'uint32x2', length: 4}),
      counts: graph.createDataView(buffer, {format: 'uint32', length: 4, byteOffset: 32})
    };
    expect(() => getTiledKeyUnionNodes(graph, 'union', tiles, output, status)).not.toThrow();
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const overlapping = graph.createDataView(buffer, {format: 'uint32', length: 4, byteOffset: 4});
    expect(() =>
      getTiledKeyUnionNodes(graph, 'union', tiles, {...output, counts: overlapping}, status)
    ).toThrow(/logical/);
    const coBoundStatus = {
      count: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 64}),
      overflow: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 68})
    };
    expect(() => getTiledKeyUnionNodes(graph, 'union', tiles, output, coBoundStatus)).toThrow(
      /aligned binding/
    );
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

for (const family of ['cells', 'groups'] as const) {
  it(`uses the core union in ${family} and preflights its uniform before tile allocation`, () => {
    for (const failUniform of [false, true]) {
      const {device, graph, output, status} = createFixture(0);
      try {
        const contributor =
          family === 'cells'
            ? new GPUCellAggregation({
                cells: createTransientView(graph, 'input-cells', 'uint32x2', 49),
                family: 'quadbin',
                resolution: 26,
                scratchTileSize: 16,
                output: {cells: output.keys, counts: output.counts, status}
              })
            : new GPUGroupStatistics({
                keys: createTransientView(graph, 'input-keys', 'uint32', 49),
                columns: [],
                scratchTileSize: 16,
                output: {
                  keys: createTransientView(graph, 'group-keys', 'uint32', 4),
                  counts: output.counts,
                  status
                }
              });
        expect(contributor.usesTiledScratch).toBe(true);
        if (failUniform) device.limits.maxUniformBufferBindingSize = 128;
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        if (failUniform) {
          expect(() => contributor.getCommandNodes(graph)).toThrow(/uniform/);
          expect(allocate).not.toHaveBeenCalled();
        } else {
          const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
          expect(kernels.some(({source}) => source.includes('fn getUnionCapacity()'))).toBe(true);
          for (const kernel of kernels) {
            expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
          }
        }
      } finally {
        device.destroy();
      }
    }
  });
}
