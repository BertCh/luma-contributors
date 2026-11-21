// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {maskWGSLComments} from '@luma.gl/shaders/wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowOutput,
  type GPUTerrainFlowProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function hasWriter(
  graph: GPUCommandGraph,
  nodes: readonly GPUCommandNode[],
  output: GraphDataView
): boolean {
  return nodes.some(node => {
    const record = getWGSLKernelNodeRecord(graph, node.id);
    return (
      record?.bindings.some(binding => {
        const view = binding.view;
        if (
          binding.access === 'read' ||
          view.buffer !== output.buffer ||
          view.byteOffset !== output.byteOffset ||
          view.length * view.rowByteLength !== output.length * output.rowByteLength
        )
          return false;
        // Binding access alone proves no write. Require a store or a mutating atomic in the module.
        // Numerical GPU cases below the host matrix qualify actual destination contents and reuse.
        const store = new RegExp(`\\b${binding.name}\\s*\\[[^\\]]*\\]\\s*=(?!=)`);
        const atomic = new RegExp(
          `atomic(?:Store|Add|Sub|Max|Min|And|Or|Xor|Exchange|CompareExchangeWeak)\\s*\\(\\s*&${binding.name}\\s*\\[`
        );
        const executableSource = maskWGSLComments(record.source);
        return store.test(executableSource) || atomic.test(executableSource);
      }) ?? false
    );
  });
}

it('WRT-09 rejects a writer to the sibling word on the same backing', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const backing = createTransientView(graph, 'two-words', 'uint32', 2);
  const requested = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  const sibling = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 0,
    length: 1
  });
  try {
    const wrongSlice = createFillNode(graph, {
      id: 'wrong-slice',
      operation: 'test',
      view: sibling,
      type: 'u32',
      value: '0u'
    });
    expect(hasWriter(graph, [wrongSlice], requested)).toBe(false);
    const correctSlice = createFillNode(graph, {
      id: 'correct-slice',
      operation: 'test',
      view: requested,
      type: 'u32',
      value: '0u'
    });
    expect(hasWriter(graph, [correctSlice], requested)).toBe(true);
  } finally {
    device.destroy();
  }
});

it('WRT-09 rejects a writable binding with no store', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const requested = createTransientView(graph, 'requested', 'uint32', 1);
  try {
    const noStore = createWGSLKernelNode(graph, {
      id: 'no-store',
      operation: 'test',
      bindings: [{name: 'output', view: requested, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      body: 'return;'
    });
    expect(hasWriter(graph, [noStore], requested)).toBe(false);
  } finally {
    device.destroy();
  }
});

function findMissingWriters(
  createContributor: (props: GPUTerrainFlowProps) => GPUTerrainFlow = props =>
    new GPUTerrainFlow(props)
) {
  const device = createNullWebGPUDevice();
  const missing: string[] = [];
  const names = [
    'filledElevation',
    'flowDirections',
    'cellClasses',
    'accumulation',
    'streams'
  ] as const;
  try {
    for (let combination = 1; combination < 34; combination++) {
      // All 31 subsets must construct. The last two cases request each flat word alone.
      const mask = combination < 32 ? combination : 1;
      const flatStatus =
        combination === 32 ? 'converged' : combination === 33 ? 'iterations' : 'both';
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const grid = makeGPURasterMetadata({width: 1, height: 1});
      const requested = new Map<string, GraphDataView>();
      const view = <Format extends 'uint32' | 'float32'>(name: string, format: Format) => {
        const result = createTransientView(graph, name, format, 1);
        requested.set(name, result);
        return result;
      };
      const selected = names.filter((_, index) => (mask & (1 << index)) !== 0);
      const output = {
        filledElevation: selected.includes('filledElevation')
          ? view('filledElevation', 'float32')
          : undefined,
        flowDirections: selected.includes('flowDirections')
          ? view('flowDirections', 'uint32')
          : undefined,
        cellClasses: selected.includes('cellClasses') ? view('cellClasses', 'uint32') : undefined,
        accumulation: selected.includes('accumulation')
          ? view('accumulation', 'float32')
          : undefined,
        streams: selected.includes('streams') ? view('streams', 'uint32') : undefined,
        status: {
          fillConverged: view('status.fillConverged', 'uint32'),
          fillIterationCount: view('status.fillIterationCount', 'uint32'),
          flatsConverged:
            flatStatus !== 'iterations' ? view('status.flatsConverged', 'uint32') : undefined,
          flatsIterationCount:
            flatStatus !== 'converged' ? view('status.flatsIterationCount', 'uint32') : undefined,
          ...(selected.includes('accumulation') || selected.includes('streams')
            ? {
                accumulationConverged: view('status.accumulationConverged', 'uint32'),
                accumulationIterationCount: view('status.accumulationIterationCount', 'uint32')
              }
            : {})
        }
      };
      // selected is nonempty in every matrix case; the loop cannot express that union statically.
      const selectedOutput = output as GPUTerrainFlowOutput;
      const flow = createContributor({
        grid,
        elevation: makeGPURasterBand(createTransientView(graph, 'elevation', 'float32', 1), grid),
        parameters: createTransientView(graph, 'parameters', 'float32', 8),
        fillDepressions: true,
        resolveFlats: true,
        maximumFillIterations: 2,
        maximumFlatIterations: 2,
        maximumAccumulationIterations: 2,
        output: selectedOutput
      });
      const nodes = flow.getCommandNodes(graph);
      for (const [name, outputView] of requested) {
        const written = hasWriter(graph, nodes, outputView);
        if (!written) missing.push(`${selected.join('+')} (${flatStatus}): ${name}`);
      }
    }
    return missing;
  } finally {
    device.destroy();
  }
}

it('does not swallow an unexpected flow constructor failure (S1)', () => {
  expect(() =>
    findMissingWriters(() => {
      throw new Error('unexpected fixture failure');
    })
  ).toThrow('unexpected fixture failure');
});

it('declares a writer for every accepted output in all 31 flow subsets and individual flat words (RT-122, S1)', () => {
  expect(findMissingWriters()).toEqual([]);
});

it('B5 excludes comments from executable writer evidence', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const output = createTransientView(graph, 'comment-output', 'uint32', 1);
  try {
    for (const body of [
      '// output[outputOffset + index] = 4u;\nreturn;',
      '/* outer /* nested */ output[outputOffset + index] = 4u; */ return;'
    ]) {
      const node = createWGSLKernelNode(graph, {
        id: 'comment-only',
        operation: 'test',
        bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body
      });
      expect(hasWriter(graph, [node], output)).toBe(false);
    }
  } finally {
    device.destroy();
  }
});

it('B5 refuses to infer a second-word store from a wider binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const backing = createTransientView(graph, 'wide-output', 'uint32', 2);
  const second = graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 4, length: 1});
  try {
    const node = createWGSLKernelNode(graph, {
      id: 'first-only',
      operation: 'test',
      bindings: [{name: 'output', view: backing, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      body: 'output[outputOffset] = 4u;'
    });
    expect(hasWriter(graph, [node], second)).toBe(false);
    const memory = createCPUGraphMemory();
    memory.write(backing, Uint32Array.of(17, 99));
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(backing))).toEqual([4, 99]);
    const correct = createWGSLKernelNode(graph, {
      id: 'second-only',
      operation: 'test',
      bindings: [{name: 'output', view: backing, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      body: 'output[outputOffset + 1u] = 0u;'
    });
    // Wider bindings are inconclusive in the host matcher, even with executable store syntax.
    expect(hasWriter(graph, [correct], second)).toBe(false);
    memory.write(backing, Uint32Array.of(17, 99));
    executeCommandNodesOnCPU([correct], memory);
    expect(Array.from(memory.readUint32(backing))).toEqual([17, 0]);
  } finally {
    device.destroy();
  }
});
