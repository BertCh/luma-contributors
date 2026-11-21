// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getGridCandidateNodes} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('region grid reduces polygon vertices and scans candidate row lengths in parallel', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const candidates = getGridCandidateNodes(graph, {
      id: 'parallel',
      region: {
        kind: 'polygon',
        vertices: createTransientView(graph, 'vertices', 'float32x2', 8193),
        vertexCount: createTransientView(graph, 'vertex-count', 'uint32', 1)
      },
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      rowCount: 3,
      regionOverflow: createTransientView(graph, 'region-overflow', 'uint32', 1),
      spatialIndex: {
        kind: 'grid',
        candidateCapacity: 2,
        index: {
          gridSize: [1, 8193],
          bounds: [0, 0, 1, 8193],
          cellOffsets: createTransientView(graph, 'offsets', 'uint32', 8194),
          objectIds: createTransientView(graph, 'ids', 'uint32', 3),
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    });
    const ids = candidates.nodes.map(node => node.id);
    expect(ids).toContain('parallel-polygon-vertex-bounds');
    expect(ids.some(id => id.startsWith('parallel-range-scan'))).toBe(true);
    const kernels = getGPUCommandNodeKernelSources(candidates.nodes);
    const blocks = kernels.find(kernel => kernel.id === 'parallel-polygon-vertex-bounds-blocks')!;
    const merge = kernels.find(kernel => kernel.id === 'parallel-polygon-vertex-bounds')!;
    expect(blocks).toBeDefined();
    const workgroupSize = Number(/@workgroup_size\((\d+)\)/.exec(blocks.source)![1]);
    expect(workgroupSize).toBe(256);
    // 8193 vertices × two virtual endpoints / 4096 items per block => five workgroups.
    expect(blocks.constants['INVOCATION_COUNT'] / workgroupSize).toBe(5);
    expect(merge.constants['INVOCATION_COUNT'] / workgroupSize).toBe(1);
    expect(merge.source).toContain('const PARTIAL_COUNT: u32 = 5u;');
    expect(merge.source).toContain('partials[base]');
    const blockNode = candidates.nodes.find(node => node.id === blocks.id)!;
    const mergeNode = candidates.nodes.find(node => node.id === merge.id)!;
    const partialWrite = blockNode.resources?.find(
      resource => resource.usage === 'storage-read-write'
    );
    const partialRead = mergeNode.resources?.find(resource => resource.usage === 'storage-read');
    expect(
      partialWrite &&
        'buffer' in partialWrite &&
        partialRead &&
        'buffer' in partialRead &&
        partialWrite.buffer === partialRead.buffer
    ).toBe(true);
    expect(candidates.nodes.indexOf(blockNode)).toBeLessThan(candidates.nodes.indexOf(mergeNode));
    const ranges = kernels.find(kernel => kernel.id === 'parallel-cell-ranges')!;
    expect(ranges.source).not.toContain('row < HEIGHT');
    expect(ranges.source).toContain('rangeLength');
  } finally {
    device.destroy();
  }
});
