// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GPUSegmentedReduction} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  createSegmentSumNode,
  getSortedSegmentSumNodes
} from '../../src/analysis-core/kernels/sorted-segment-sums';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('review B3 recording facts describe the returned reduction node', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const node = createSegmentSumNode(graph, {
      id: 'recorded',
      operation: 'RecordedSum',
      segmentCount: 2,
      input: createTransientView(graph, 'input', 'float32', 3),
      segmentOffsets: createTransientView(graph, 'offsets', 'uint32', 6),
      output: createTransientView(graph, 'output', 'float32', 5)
    });
    const record = getWGSLKernelNodeRecord(graph, node.id);
    expect(record).toBeDefined();
    const resources = node.resources as readonly {buffer: {length: number}}[];
    expect(record!.bindings.map(binding => binding.view.length)).toEqual(
      resources.map(resource => resource.buffer.length)
    );
    expect(record!.operation).toBe(node.workload?.operation);
    expect(record!.invocationCount).toBe(node.workload?.maximumInvocationCount);
  } finally {
    device.destroy();
  }
});

it.each([
  [16776960, true, 4294901760],
  [16776961, false, 4294902016],
  [16777215, false, 4294967040]
] as const)('review B4 %i segments respect the padded invocation ceiling', (segmentCount, delegates, maximumInvocationCount) => {
  const device = createNullWebGPUDevice();
  const delegated = vi.spyOn(GPUSegmentedReduction.prototype, 'getCommandNodes');
  try {
    const graph = new GPUCommandGraph(device);
    const node = createSegmentSumNode(graph, {
      id: 'boundary',
      operation: 'BoundarySum',
      segmentCount,
      input: createTransientView(graph, 'input', 'float32', 1),
      segmentOffsets: createTransientView(graph, 'offsets', 'uint32', segmentCount + 1),
      output: createTransientView(graph, 'output', 'float32', segmentCount)
    });
    expect(delegated).toHaveBeenCalledTimes(delegates ? 1 : 0);
    expect(node.workload?.maximumInvocationCount).toBe(maximumInvocationCount);
    expect(node.workload?.maximumInvocationCount).toBeLessThanOrEqual(0xffffffff);
  } finally {
    delegated.mockRestore();
    device.destroy();
  }
});

it.each([
  'input',
  'segmentOffsets',
  'output'
] as const)('review B5 rejects nonpacked direct %s', subject => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      id: 'strided-direct',
      operation: 'DirectSum',
      segmentCount: 1,
      input: createTransientView(graph, 'input', 'float32', 3),
      segmentOffsets: createTransientView(graph, 'offsets', 'uint32', 3),
      output: createTransientView(graph, 'output', 'float32', 3)
    };
    const storage = props[subject];
    const strided = graph.createDataView(storage.buffer, {
      format: storage.format,
      length: 2,
      byteStride: 8
    });
    expect(() => createSegmentSumNode(graph, {...props, [subject]: strided})).toThrow(
      new RegExp(`strided-direct ${subject}.*packed`)
    );
  } finally {
    device.destroy();
  }
});

it.each([
  -1,
  0.5,
  NaN
])('review earlier count %s refuses before scratch and allows same-ID retry', segmentCount => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const segmentKeys = createTransientView(graph, 'keys', 'uint32', 1);
    const buffers = Reflect.get(graph, 'buffers') as Map<string, unknown>;
    const before = Array.from(buffers.keys());
    expect(() =>
      getSortedSegmentSumNodes(graph, {id: 'retry', operation: 'Count', segmentCount, segmentKeys})
    ).toThrow(/retry segmentCount/);
    expect(Array.from(buffers.keys())).toEqual(before);
    expect(
      getSortedSegmentSumNodes(graph, {
        id: 'retry',
        operation: 'Count',
        segmentCount: 0,
        segmentKeys
      }).length
    ).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it.each([
  'sums',
  'weightSums'
] as const)('review earlier %s requires contributions before scratch', outputName => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const segmentKeys = createTransientView(graph, 'keys', 'uint32', 1);
    const output = createTransientView(graph, 'output', 'float32', 1);
    const buffers = Reflect.get(graph, 'buffers') as Map<string, unknown>;
    const before = Array.from(buffers.keys());
    expect(() =>
      getSortedSegmentSumNodes(graph, {
        id: 'missing',
        operation: 'Missing',
        segmentCount: 1,
        segmentKeys,
        [outputName]: output
      })
    ).toThrow(/missing.*Contributions/);
    expect(Array.from(buffers.keys())).toEqual(before);
  } finally {
    device.destroy();
  }
});

it('LUMA-115 delegates active packed prefixes and the ordinary route after gather', () => {
  const device = createNullWebGPUDevice();
  const delegated = vi.spyOn(GPUSegmentedReduction.prototype, 'getCommandNodes');
  try {
    const graph = new GPUCommandGraph(device);
    const inputStorage = createTransientView(graph, 'input', 'float32', 8);
    const offsetStorage = createTransientView(graph, 'offsets', 'uint32', 8);
    const outputStorage = createTransientView(graph, 'output', 'float32', 8);
    const input = graph.createDataView(inputStorage.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 4
    });
    const segmentOffsets = graph.createDataView(offsetStorage.buffer, {
      format: 'uint32',
      length: 5,
      byteOffset: 4
    });
    const output = graph.createDataView(outputStorage.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 4
    });
    const node = createSegmentSumNode(graph, {
      id: 'prefix',
      operation: 'AnalysisSum',
      segmentCount: 2,
      input,
      segmentOffsets,
      output
    });
    expect(delegated).toHaveBeenCalledTimes(1);
    const reducer = delegated.mock.contexts[0] as GPUSegmentedReduction<'float32'>;
    expect(reducer.input).toBe(input);
    expect(reducer.segmentOffsets.length).toBe(3);
    expect(reducer.output.length).toBe(2);
    expect(reducer.segmentOffsets.byteOffset).toBe(4);
    expect(reducer.output.byteOffset).toBe(4);
    expect(node.id).toBe('prefix');
    expect(node.workload?.operation).toBe('AnalysisSum');
    getSortedSegmentSumNodes(graph, {
      id: 'sorted',
      operation: 'AnalysisSum',
      segmentCount: 2,
      segmentKeys: createTransientView(graph, 'keys', 'uint32', 4),
      reductions: [{name: 'total', contributions: input, output}]
    });
    expect(delegated).toHaveBeenCalledTimes(2);
  } finally {
    delegated.mockRestore();
    device.destroy();
  }
});

it('LUMA-115 retains zero-segment, gated, extra-resource, alias and min routes', () => {
  const device = createNullWebGPUDevice();
  const delegated = vi.spyOn(GPUSegmentedReduction.prototype, 'getCommandNodes');
  try {
    const graph = new GPUCommandGraph(device);
    const input = createTransientView(graph, 'input', 'float32', 4);
    const output = createTransientView(graph, 'output', 'float32', 2);
    const segmentOffsets = createTransientView(graph, 'offsets', 'uint32', 3);
    const props = {
      id: 'local',
      operation: 'AnalysisSum',
      segmentCount: 2,
      input,
      segmentOffsets,
      output
    };
    for (const options of [
      {segmentCount: 0},
      {condition: {id: 'gate', source: 'cpu' as const, evaluate: () => true}},
      {extraResources: []},
      {output: input},
      {reduction: 'min' as const}
    ]) {
      expect(createSegmentSumNode(graph, {...props, ...options}).id).toBe('local');
    }
    expect(delegated).not.toHaveBeenCalled();
  } finally {
    delegated.mockRestore();
    device.destroy();
  }
});
