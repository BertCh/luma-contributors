// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  createTransientView,
  createGPUComputeCommandNode,
  GPUCommandGraph,
  GPUGraph,
  gpuGraph
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {GPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';
import {GPUOffsetCurveComponents} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
import {createOffsetComponentAssemblyNodes} from '../../../src/gpu-spatial-analysis/geometry-refinement/offset-component-assembly';
import {createOffsetAssemblySequence} from '../../../src/gpu-spatial-analysis/geometry-refinement/offset-assembly-sequence';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {OFFSET_ASSEMBLY_SCENES, packOffsetAssemblyFixture} from './offset-assembly-fixtures';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createOffsetSequenceBudgetFixture} from './offset-sequence-budget-fixture';

useCPUExecutorTestHooks();

function createAssemblyPreflightDevice() {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => ({id: props.id ?? 'offset-preflight', hash: '', destroy: () => {}}) as never
  );
  return device;
}

it.each([
  17, 25
])('public offset operation prepares pass boundaries at segment capacity %i', segmentCapacity => {
  const device = createAssemblyPreflightDevice();
  const graph = new GPUGraph({id: 'offset-test'});
  graph.addInput('original', gpuGraph.vector('float32x2', {length: 5}));
  graph.addInput('raw', gpuGraph.vector('float32x2', {length: 5}));
  graph.addInput('offsets', gpuGraph.vector('uint32', {length: 2}));
  const operation = new GPUOffsetCurveComponentsOperation({
    lineCount: 1,
    original: {positions: 'original', lineOffsets: 'offsets'},
    raw: {positions: 'raw', lineOffsets: 'offsets'},
    distance: 1,
    tolerance: 0,
    intersectionCapacity: 8,
    segmentCapacity,
    vertexCapacity: 32,
    componentCapacity: 8,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  });
  graph.add(operation);
  const compiled = graph.compile(device, {
    outputs: ['status'],
    lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
  });
  const input = (
    name: string,
    format: 'float32x2' | 'uint32',
    values: Float32Array | Uint32Array
  ) =>
    new GPUVector({
      type: 'buffer',
      name,
      buffer: device.createBuffer({data: values, usage: Buffer.STORAGE}),
      format,
      length: format === 'float32x2' ? values.length / 2 : values.length,
      ownsBuffer: true
    });
  const inputs = {
    original: input('original', 'float32x2', Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0)),
    raw: input('raw', 'float32x2', Float32Array.of(0, 1, 1, 1, 2, 1, 3, 1, 4, 1)),
    offsets: input('offsets', 'uint32', Uint32Array.of(0, 5))
  };
  try {
    const prepared = compiled.prepare({inputs});
    prepared.destroy();
    for (const output of Object.values(prepared.outputs)) output.destroy();
  } finally {
    compiled.destroy();
    operation.destroy();
    for (const vector of Object.values(inputs)) vector.destroy();
    device.destroy();
  }
});

it.each([
  {capacity: 416, dispatches: 2920},
  {capacity: 9728, dispatches: 685},
  {capacity: 12800, dispatches: 901},
  {capacity: 18944, dispatches: 1333},
  {capacity: 31232, dispatches: 2197},
  {capacity: 34304, dispatches: 2413}
])('compiled offset passes stay within budget for $capacity fragment slots', ({
  capacity,
  dispatches
}) => {
  const device = createAssemblyPreflightDevice();
  try {
    const graph = new GPUCommandGraph(device);
    graph.add(
      createOffsetComponentAssemblyNodes(graph, {
        id: 'parity',
        capacity,
        fragments: createTransientView(graph, 'fragments', 'float32x4', capacity),
        metadata: createTransientView(graph, 'metadata', 'uint32', capacity * 4),
        output: {
          positions: createTransientView(graph, 'positions', 'float32x2', 1024),
          components: createTransientView(graph, 'components', 'uint32x4', 128),
          sourceEdges: createTransientView(graph, 'sources', 'uint32', 1024),
          status: createTransientView(graph, 'status', 'uint32x4', 2)
        }
      })
    );
    const compiled = graph.compile();
    try {
      let dispatchesInPass = 0;
      let assemblyDispatches = 0;
      for (const node of compiled.preflight.nodes) {
        if (node.type !== 'compute') {
          dispatchesInPass = 0;
          continue;
        }
        dispatchesInPass += node.commandCount;
        expect(dispatchesInPass, node.id).toBeLessThanOrEqual(2048);
        if (node.variant === 'bounded-assembly') assemblyDispatches += node.commandCount;
      }
      expect(assemblyDispatches).toBe(dispatches);
    } finally {
      compiled.destroy();
    }
  } finally {
    device.destroy();
  }
});

it('forward F9 keeps every repetition and command order across sequence batches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const result = createTransientView(graph, 'result', 'uint32', 1);
    const fixture = createOffsetSequenceBudgetFixture(graph, result);
    expect(fixture.batches.map(node => node.workload?.maximumInvocationCount)).toEqual([
      4000000000, 2000000000
    ]);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    // Append digits 1,2 in each of three rounds, including across the batch boundary.
    expect(Array.from(memory.readUint32(result))).toEqual([121212]);
  } finally {
    device.destroy();
  }
});

it('forward F9 bounds each offset assembly command without truncating its schedule', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const original = {
    positions: createTransientView(graph, 'original', 'float32x2', 32),
    lineOffsets: createTransientView(graph, 'original-offsets', 'uint32', 3)
  };
  try {
    const nodes = new GPUOffsetCurveComponents({
      original,
      raw: {
        positions: createTransientView(graph, 'raw', 'float32x2', 544),
        lineOffsets: createTransientView(graph, 'raw-offsets', 'uint32', 3)
      },
      rowsPerVertex: 17,
      distance: 1,
      tolerance: 0.01,
      segmentCapacity: 1024,
      intersectionCapacity: 512,
      output: {
        positions: createTransientView(graph, 'positions', 'float32x2', 2048),
        sourceEdges: createTransientView(graph, 'source-edges', 'uint32', 2048),
        components: createTransientView(graph, 'components', 'uint32x4', 1024),
        status: createTransientView(graph, 'status', 'uint32x4', 2)
      }
    }).getCommandNodes(graph);
    const cancellation = nodes.filter(
      node =>
        node.type === 'compute' &&
        node.id.startsWith('offset-curve-components-assembly-cancellation')
    );
    const traversal = nodes.filter(
      node =>
        node.type === 'compute' &&
        node.id.startsWith('offset-curve-components-assembly-traversal') &&
        !node.id.endsWith('-init')
    );
    for (const node of [...cancellation, ...traversal]) {
      expect(node.workload?.maximumInvocationCount).toBeLessThanOrEqual(4294967295);
      expect(node.workload?.commandCount).toBeLessThanOrEqual(2048);
    }
    expect(nodes.some(node => node.type === 'copy' && node.id.includes('-pass-boundary-'))).toBe(
      true
    );
    // 1024 * (32*6+1) = 197632 fragment slots. Each cancellation round runs
    // one serial lane and two fragment passes; traversal adds two serial lanes.
    expect(
      cancellation.reduce((sum, node) => sum + (node.workload?.maximumInvocationCount ?? 0), 0)
    ).toBe(78117012480); // 197632 * (1 + 2*197632)
    expect(
      traversal.reduce((sum, node) => sum + (node.workload?.maximumInvocationCount ?? 0), 0)
    ).toBe(78117605378); // (197632+1) * (2 + 2*197632)
  } finally {
    device.destroy();
  }
});

it('large continuation estimates do not impose an accidental safe-integer input limit', () => {
  const command = createGPUComputeCommandNode({
    id: 'estimate',
    resources: [],
    workload: {
      operation: 'GPUOffsetCurveComponents',
      commandCount: 1,
      maximumInvocationCount: 1000000000,
      maximumWorkgroupCount: 1000000000
    },
    compile: () => ({encode: () => {}})
  });
  const sequence = createOffsetAssemblySequence('large-estimate', [
    {nodes: [command], repetitions: 1000000000}
  ]);
  expect(sequence.workload?.commandCount).toBe(1000000000);
  // Optional aggregate estimates cannot be represented exactly above Number.MAX_SAFE_INTEGER.
  expect(sequence.workload?.maximumInvocationCount).toBeUndefined();
  expect(sequence.workload?.maximumWorkgroupCount).toBeUndefined();
});

it('incoming census includes a predecessor beyond its first 256-row dispatch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const capacity = 257;
  const fragments = createTransientView(graph, 'cycle-fragments', 'float32x4', capacity);
  const metadata = createTransientView(graph, 'cycle-metadata', 'uint32', capacity * 4);
  const coordinates = Float32Array.from(
    Array.from({length: capacity}, (_, index) => [
      index,
      0,
      index + 1 === capacity ? 0 : index + 1,
      0
    ]).flat()
  );
  memory.write(fragments, coordinates);
  memory.write(
    metadata,
    Uint32Array.from(Array.from({length: capacity}, (_, index) => [0, index, 1, 0]).flat())
  );
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', capacity + 1),
    components: createTransientView(graph, 'components', 'uint32x4', 1),
    sourceEdges: createTransientView(graph, 'sources', 'uint32', capacity + 1),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  const views: import('@luma.gl/graph/gpu-core').GraphDataView[] = [];
  const create = graph.createDataView.bind(graph);
  const observe = vi.spyOn(graph, 'createDataView').mockImplementation((...args) => {
    const result = create(...args);
    views.push(result);
    return result;
  });
  const nodes = createOffsetComponentAssemblyNodes(graph, {
    id: 'census',
    capacity,
    fragments,
    metadata,
    output
  });
  observe.mockRestore();
  try {
    executeCommandNodesOnCPU(
      nodes.filter(
        node => !/-cancel-init$|-cancellation$|-traversal-init$|-traversal$|-status$/.test(node.id)
      ),
      memory
    );
    const incoming = views.find(view => view.buffer.id === 'census-incoming');
    if (!incoming) throw new Error('missing census view');
    // A 257-edge cycle has exactly one predecessor per edge; edge zero's is row 256.
    expect(Array.from(memory.readUint32(incoming))).toEqual(Array(capacity).fill(1));
  } finally {
    device.destroy();
  }
});
it('large fragment tables do not use the serial component kernel', () => {
  const device = createAssemblyPreflightDevice();
  const graph = new GPUCommandGraph(device);
  const lines = {
    positions: createTransientView(graph, 'input', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2)
  };
  try {
    const nodes = new GPUOffsetCurveComponents({
      original: lines,
      raw: lines,
      distance: 1,
      tolerance: 0,
      intersectionCapacity: 1,
      segmentCapacity: 32,
      output: {
        positions: createTransientView(graph, 'positions', 'float32x2', 8),
        components: createTransientView(graph, 'components', 'uint32x4', 4),
        sourceEdges: createTransientView(graph, 'sources', 'uint32', 8),
        status: createTransientView(graph, 'status', 'uint32x4', 2)
      }
    }).getCommandNodes(graph);
    expect(nodes.some(node => node.id === 'offset-curve-components-components')).toBe(false);
    graph.add(nodes);
    const compiled = graph.compile();
    compiled.destroy();
  } finally {
    device.destroy();
  }
});

it.each(OFFSET_ASSEMBLY_SCENES)('parallel assembly preserves $name', fixture => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const packed = packOffsetAssemblyFixture(fixture);
  const capacity = packed.capacity;
  const fragments = createTransientView(graph, 'fragments', 'float32x4', capacity);
  const metadata = createTransientView(graph, 'metadata', 'uint32', capacity * 4);
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 16),
    components: createTransientView(graph, 'components', 'uint32x4', 8),
    sourceEdges: createTransientView(graph, 'sources', 'uint32', 16),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  memory.write(fragments, packed.fragments);
  memory.write(metadata, packed.metadata);
  memory.write(output.components, new Uint32Array(32).fill(99));
  const nodes = createOffsetComponentAssemblyNodes(graph, {
    id: 'assembly',
    capacity,
    fragments,
    metadata,
    output
  });
  try {
    expect(getGPUCommandNodeKernelSources(nodes).length).toBeGreaterThan(0);
    executeCommandNodesOnCPU(nodes, memory);
    const componentCount = fixture.components.length / 4;
    const vertexCount = fixture.expected.length / 2;
    expect(Array.from(memory.readUint32(output.status))).toEqual([
      componentCount,
      vertexCount,
      componentCount,
      vertexCount,
      0,
      0,
      0,
      0
    ]);
    expect(
      Array.from(memory.readFloat32(output.positions).slice(0, fixture.expected.length))
    ).toEqual(fixture.expected);
    expect(Array.from(memory.readUint32(output.sourceEdges).slice(0, vertexCount))).toEqual(
      fixture.sources
    );
    expect(Array.from(memory.readUint32(output.components))).toEqual([
      ...fixture.components,
      ...Array(32 - fixture.components.length).fill(0)
    ]);
    memory.write(metadata, new Uint32Array(capacity * 4));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.status))).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});
