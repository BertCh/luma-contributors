// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryThinning} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('computes independent track boundaries with empty tracks and a clamped compact capacity', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const trackOffsets = createTransientView(graph, 'tracks', 'uint32', 7);
    const outOffsets = createTransientView(graph, 'out-offsets', 'uint32', 7);
    recordWGSLKernelNodes(graph);
    const nodes = new GPUTrajectoryThinning({
      id: 'thin',
      mode: 'minimum-time',
      positions: createTransientView(graph, 'positions', 'float32x2', 10),
      timestamps: createTransientView(graph, 'times', 'float32', 10),
      trackOffsets,
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {
        rowIds: createTransientView(graph, 'ids', 'uint32', 3),
        trackOffsets: outOffsets,
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }).getCommandNodes(graph);
    const node = nodes.find(node => node.id === 'thin-offsets')!;
    const record = getWGSLKernelNodeRecord(graph, node.id)!;
    expect(record.invocationCount).toBe(7);
    expect(record.body).not.toContain('for (var track');
    const memory = createCPUGraphMemory();
    memory.write(trackOffsets, new Uint32Array([0, 0, 1, 5, 5, 9, 10]));
    memory.write(
      record.bindings.find(binding => binding.name === 'keptRows')!.view,
      new Uint32Array([0, 1, 3, 4, 5, 7, 8, 9, 0, 0])
    );
    memory.write(
      record.bindings.find(binding => binding.name === 'keptTotal')!.view,
      new Uint32Array([8])
    );
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(outOffsets))).toEqual([0, 0, 1, 3, 3, 3, 3]);
  } finally {
    device.destroy();
  }
});
