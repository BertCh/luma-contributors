// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('publishes the executed component round count when requested alone', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const iterationCount = createTransientView(graph, 'iteration-count', 'uint32', 1);
    memory.write(iterationCount, Uint32Array.of(99));
    const contributor = new GPURedcapSingleLinkage({
      componentIterations: 3,
      values: createTransientView(graph, 'values', 'float32', 2),
      parameters: createTransientView(graph, 'parameters', 'uint32', 1),
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
        weights: createTransientView(graph, 'weights', 'float32', 2)
      },
      output: {labels: createTransientView(graph, 'labels', 'uint32', 2), status: {iterationCount}}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    // The component loop has no dispatch gates: every scheduled round executes, even after
    // labels stabilize. Exercise its scalar publication without interpreting the upstream MST.
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.endsWith('-iteration-count')),
      memory
    );
    expect(Array.from(memory.readUint32(iterationCount))).toEqual([3]);
  } finally {
    device.destroy();
  }
});

it('consumes weights loss flags and suppresses incomplete region labels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    recordWGSLKernelNodes(graph);
    const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
    const labels = createTransientView(graph, 'labels', 'uint32', 3);
    const offsets = createTransientView(graph, 'offsets', 'uint32', 4);
    memory.write(overflow, Uint32Array.of(1));
    memory.write(offsets, Uint32Array.of(0, 1, 2, 2));
    memory.write(labels, Uint32Array.of(0, 0, 2));
    const nodes = new GPURedcapSingleLinkage({
      componentIterations: 3,
      values: createTransientView(graph, 'values', 'float32', 3),
      parameters: createTransientView(graph, 'parameters', 'uint32', 1),
      weights: {
        offsets,
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
        weights: createTransientView(graph, 'weights', 'float32', 4),
        status: {overflow}
      },
      output: {labels}
    }).getCommandNodes(graph);
    expect(
      nodes.some(node =>
        getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(binding => binding.view === overflow)
      )
    ).toBe(true);
    expect(
      nodes.some(node => {
        const record = getWGSLKernelNodeRecord(graph, node.id);
        return (
          record?.operation === 'GPUSpatialWeightsMinimumSpanningTree' &&
          record.bindings.some(binding => binding.view === offsets)
        );
      })
    ).toBe(true);
    // Interpret only the result guard. The headless companion executes the complete MST.
    const guards = nodes.filter(node => node.id.includes('-suppress-'));
    executeCommandNodesOnCPU(guards, memory);
    expect(Array.from(memory.readUint32(labels))).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    memory.write(overflow, Uint32Array.of(0));
    memory.write(offsets, Uint32Array.of(0, 1, 3, 4));
    memory.write(labels, Uint32Array.of(0, 0, 0));
    executeCommandNodesOnCPU(guards, memory);
    expect(Array.from(memory.readUint32(labels))).toEqual([0, 0, 0]);
  } finally {
    device.destroy();
  }
});
