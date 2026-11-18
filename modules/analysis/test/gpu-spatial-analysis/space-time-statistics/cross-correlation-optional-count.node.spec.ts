// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it.each([
  false,
  true
])('writes testedCount and clears it when all correlations become untestable (adjusted=%s)', adjusted => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const first = createTransientView(graph, 'first', 'float32', 4);
    const second = createTransientView(graph, 'second', 'float32', 4);
    const testedCount = createTransientView(graph, 'tested-count', 'uint32', 1);
    memory.write(first, Float32Array.of(1, 2, 3, 4));
    memory.write(second, Float32Array.of(1, 2, 3, 4));
    memory.write(testedCount, Uint32Array.of(99));
    const nodes = new GPUSpaceTimeCrossCorrelation({
      first,
      second,
      cellCount: 1,
      periodCount: 4,
      lags: [0],
      layout: 'cell-major',
      multipleTesting: 'none',
      output: {
        correlations: createTransientView(graph, 'correlations', 'float32', 1),
        pValues: createTransientView(graph, 'p-values', 'float32', 1),
        adjustedPValues: adjusted
          ? createTransientView(graph, 'adjusted', 'float32', 1)
          : undefined,
        testedCount
      }
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(testedCount))).toEqual([1]);
    memory.write(first, Float32Array.of(2, 2, 2, 2));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(testedCount))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it('builds adjustment and count publication when raw p-values and testedCount share storage', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    recordWGSLKernelNodes(graph);
    const storage = createTransientView(graph, 'p-values-and-count', 'float32', 65);
    const pValues = graph.createDataView(storage.buffer, {format: 'float32', length: 1});
    const testedCount = graph.createDataView(storage.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    const contributor = new GPUSpaceTimeCrossCorrelation({
      first: createTransientView(graph, 'first', 'float32', 4),
      second: createTransientView(graph, 'second', 'float32', 4),
      cellCount: 1,
      periodCount: 4,
      lags: [0],
      layout: 'cell-major',
      multipleTesting: 'none',
      output: {
        correlations: createTransientView(graph, 'correlations', 'float32', 1),
        pValues,
        adjustedPValues: createTransientView(graph, 'adjusted', 'float32', 1),
        testedCount
      }
    });
    // Graph construction itself failed before any GPU submission in the lead's Metal run.
    const nodes = contributor.getCommandNodes(graph);
    const publication = nodes.find(node => node.id.endsWith('-publish-tested-count'));
    expect(publication).toBeDefined();
    const record = getWGSLKernelNodeRecord(graph, publication!.id)!;
    const source = record.bindings.find(binding => binding.name === 'count')!;
    const target = record.bindings.find(binding => binding.name === 'testedCount')!;
    expect(source.view.buffer).not.toBe(pValues.buffer);
    expect(target.view).toBe(testedCount);
  } finally {
    device.destroy();
  }
});
