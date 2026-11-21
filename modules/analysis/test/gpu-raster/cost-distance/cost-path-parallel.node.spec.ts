// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUCostDistancePath,
  MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const cellCount of [1, 256, 257, 1025]) {
  it(`certifies ${cellCount} cells with parallel jumps and gathers a compact prefix`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    expect(MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS).toBe(256);
    recordWGSLKernelNodes(graph);
    const contributor = new GPUCostDistancePath({
      grid: makeGPURasterMetadata({width: cellCount, height: 1}),
      backLinks: createTransientView(graph, 'links', 'uint32', cellCount),
      target: createTransientView(graph, 'target', 'uint32', 1),
      output: {
        path: {
          ids: createTransientView(graph, 'ids', 'uint32', 3),
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          }
        }
      }
    });
    contributor.getCommandNodes(graph);
    if (cellCount <= MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS) {
      expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-walk')?.invocationCount).toBe(1);
      expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-initialize')).toBeUndefined();
      return;
    }
    expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-walk') === undefined).toBe(true);
    expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-initialize')?.invocationCount).toBe(
      cellCount
    );
    for (let round = 0; round < Math.ceil(Math.log2(cellCount)); round++) {
      expect(
        getWGSLKernelNodeRecord(graph, `cost-distance-path-jump-${round}`)?.invocationCount
      ).toBe(cellCount);
    }
    expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-gather-0')?.invocationCount).toBe(1);
    expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-gather-1')?.invocationCount).toBe(1);
    expect(getWGSLKernelNodeRecord(graph, 'cost-distance-path-gather-2')).toBeUndefined();
  });
}

it('checks parallel route bindings before allocating any scratch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUCostDistancePath({
    grid: makeGPURasterMetadata({width: 257, height: 1}),
    backLinks: createTransientView(graph, 'links', 'uint32', 257),
    target: createTransientView(graph, 'target', 'uint32', 1),
    output: {
      path: {
        ids: createTransientView(graph, 'ids', 'uint32', 3),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }
  });
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxStorageBufferBindingSize: 1027}
  });
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    // Caller links and every scratch plane need 257 * 4 = 1028 bytes per binding.
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 1028 bytes/);
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    graph.device.destroy();
  }
});
