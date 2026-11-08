// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {addSparseResistanceSystemNodes} from '../../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-system';
import {addSparseResistanceSolveNodes} from '../../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-solver';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function countNodes(
  vertexCount: number,
  maximumIterations: number,
  connectivityIterations: number,
  edgeCount = 4
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const edges = createTransientView(graph, 'edges', 'uint32x2', edgeCount);
    const conductances = createTransientView(graph, 'conductances', 'float32', edgeCount);
    const system = addSparseResistanceSystemNodes(graph, {
      id: 'sparse',
      operation: 'sparseResistanceTest',
      edges,
      conductances,
      vertexCount,
      connectivityIterations
    });
    addSparseResistanceSolveNodes(graph, {
      id: 'sparse',
      operation: 'sparseResistanceTest',
      system,
      vertexCount,
      source: 0,
      sink: vertexCount - 1,
      maximumIterations,
      tolerance: 1e-4
    });
    return (graph as unknown as {nodes: unknown[]}).nodes;
  } finally {
    device.destroy();
  }
}

it('sparse resistance kernels parse, audit and stay within portable binding limits', () => {
  const kernels = getGPUCommandNodeKernelSources(countNodes(5, 2, 2));
  expect(kernels.length).toBeGreaterThan(20);
  for (const kernel of kernels) {
    expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
  }
});

it('sparse resistance solve adds five nodes per iteration', () => {
  const one = countNodes(5, 1, 2).length;
  const three = countNodes(5, 3, 2).length;
  expect(three - one).toBe(10);
});

it('sparse resistance node counts stay under the graph cap', () => {
  // Default configuration: 1,024 iterations, 128 connectivity rounds.
  const defaultCount = countNodes(1024, 1024, 128).length;
  // Maximal configuration: 2^20 vertices, 2^21 edges, 4,096 iterations, 1,024 rounds.
  const maximalCount = countNodes(1048576, 4096, 1024, 2097152).length;
  expect(defaultCount).toBeLessThanOrEqual(5500);
  expect(maximalCount).toBeLessThanOrEqual(22600);
});
