// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkStructure,
  type GPUNetworkStructureProps
} from '../../../src/gpu-network/network-structure/gpu-network-structure';
import {
  computeNetworkStructureOnCPU,
  type CPUNetworkStructureInput
} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUCommandNodeKernelSources,
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {createComponentCSR, createRandomComponentEdges} from './strong-components-fixtures';
import {
  createSplitGridStructureCSR,
  createUndirectedStructureCSR
} from './network-structure-parallel-fixtures';

async function runStructure(
  input: CPUNetworkStructureInput,
  mode: 'auto' | 'parallel' | 'exact' | undefined,
  options: {
    strong?: boolean;
    resilience?: boolean;
    maximumIterations?: number;
    repeats?: number;
    status?: boolean;
    validateShaders?: boolean;
    resilienceOutputs?: readonly ('bridges' | 'articulationPoints' | 'biconnectedComponents')[];
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const prefix = 3;
  const suffix = 2;
  const canary = 0xabcdef01;
  const importInput = (name: string, values: Uint32Array) => {
    const padded = new Uint32Array(prefix + values.length + suffix).fill(canary);
    padded.set(values, prefix);
    const buffer = createInputBuffer(device, padded);
    buffers.push(buffer);
    const whole = importGraphBuffer(graph, name, buffer, 'uint32');
    return graph.createDataView(whole.buffer, {
      format: 'uint32',
      byteOffset: prefix * 4,
      length: values.length
    });
  };
  const outputs = new Map<string, {buffer: Buffer; length: number}>();
  const importOutput = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, prefix + length + suffix);
    buffer.write(new Uint32Array(prefix + length + suffix).fill(canary));
    buffers.push(buffer);
    outputs.set(name, {buffer, length});
    const whole = importGraphBuffer(graph, name, buffer, 'uint32');
    return graph.createDataView(whole.buffer, {format: 'uint32', byteOffset: prefix * 4, length});
  };
  const nodeCount = input.offsets.length - 1;
  const edgeCount = input.neighbors.length;
  const strong = options.strong ?? true;
  const resilience = options.resilience ?? Boolean(input.reverseEdgeSlots);
  const output: GPUNetworkStructureProps['output'] = {};
  if (strong) output.strongComponents = importOutput('strongComponents', nodeCount);
  if (resilience) {
    for (const name of options.resilienceOutputs ?? [
      'bridges',
      'articulationPoints',
      'biconnectedComponents'
    ]) {
      output[name] = importOutput(name, name === 'articulationPoints' ? nodeCount : edgeCount);
    }
  }
  const contributor = new GPUNetworkStructure({
    id: `structure-${mode}`,
    ...(mode === undefined ? {} : {mode}),
    ...(options.maximumIterations === undefined
      ? {}
      : {strongComponentsMaximumIterations: options.maximumIterations}),
    ...(strong && options.status !== false
      ? {
          strongComponentsStatus: {
            converged: importOutput('converged', 1),
            iterationCount: importOutput('iterationCount', 1),
            iterationLimitReached: importOutput('iterationLimitReached', 1)
          }
        }
      : {}),
    topology: {
      offsets: importInput('offsets', input.offsets),
      neighbors: importInput('neighbors', input.neighbors)
    },
    reverseEdgeSlots: input.reverseEdgeSlots
      ? importInput('reverse', input.reverseEdgeSlots)
      : undefined,
    output
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  if (options.validateShaders) {
    const sources = new Map(
      getGPUCommandNodeKernelSources(nodes).map(kernel => [kernel.source, kernel.id])
    );
    for (const [source, shaderId] of sources) {
      const shader = device.createShader({
        id: shaderId,
        source,
        language: 'wgsl',
        stage: 'compute'
      });
      try {
        const diagnostics = await shader.getCompilationInfo();
        expect(
          diagnostics.filter(message => message.type === 'error'),
          shaderId
        ).toEqual([]);
      } finally {
        shader.destroy();
      }
    }
  }
  for (const node of nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    let result: Record<string, number[]> = {};
    for (let repeat = 0; repeat < (options.repeats ?? 1); repeat++) {
      submitGraph(device, compiled, undefined);
      result = {};
      for (const [name, {buffer, length}] of outputs) {
        const values = await readUint32(buffer);
        expect(values.slice(0, prefix)).toEqual(new Array(prefix).fill(canary));
        expect(values.slice(prefix + length)).toEqual(new Array(suffix).fill(canary));
        result[name] = values.slice(prefix, prefix + length);
      }
    }
    return result;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const seed of [1, 11, 21, 29]) {
  it(`parallel structure equals bounded exact kernels and CPU Tarjan on multigraph seed ${seed}`, async () => {
    // Four isolated vertices; random sampling deliberately produces loops and duplicate pairs.
    const edges = createRandomComponentEdges(60, 180, seed);
    edges.push([0, 0], [0, 1], [0, 1], [3, 4], [4, 5], [5, 3]);
    const input = createUndirectedStructureCSR(64, edges, 3);
    const parallel = await runStructure(input, 'parallel', {repeats: 2});
    const exact = await runStructure(input, 'exact');
    const expected = computeNetworkStructureOnCPU(input);
    for (const name of [
      'strongComponents',
      'bridges',
      'articulationPoints',
      'biconnectedComponents'
    ] as const) {
      expect(parallel[name]).toEqual(exact[name]);
      expect(parallel[name]).toEqual(Array.from(expected[name]));
    }
    expect(parallel['converged']).toEqual([1]);
    expect(parallel['iterationLimitReached']).toEqual([0]);
  }, 120000);
}

it('ranks a depth-1023 tree in logarithmic rounds and retains loop singleton blocks', async () => {
  const edges: [number, number][] = Array.from({length: 1023}, (_, vertex) => [vertex, vertex + 1]);
  edges.push([0, 0], [512, 512], [512, 512]);
  const input = createUndirectedStructureCSR(1024, edges);
  const parallel = await runStructure(input, 'parallel', {strong: false});
  const exact = await runStructure(input, 'exact', {strong: false});
  expect(parallel).toEqual(exact);
  const expected = computeNetworkStructureOnCPU(input);
  for (const name of ['bridges', 'articulationPoints', 'biconnectedComponents'] as const) {
    expect(parallel[name]).toEqual(Array.from(expected[name]));
  }
}, 120000);

it('handles empty adjacency, a singleton loop and ignored invalid/unowned slots', async () => {
  const invalid = {
    offsets: new Uint32Array([0, 1, 1]),
    neighbors: new Uint32Array([99, 0]),
    reverseEdgeSlots: new Uint32Array([0xffffffff, 0xffffffff])
  };
  for (const input of [
    createUndirectedStructureCSR(1, []),
    createUndirectedStructureCSR(1, [[0, 0]]),
    invalid
  ]) {
    const parallel = await runStructure(input, 'parallel', {repeats: 2});
    const exact = await runStructure(input, 'exact');
    const expected = computeNetworkStructureOnCPU(input);
    for (const name of [
      'strongComponents',
      'bridges',
      'articulationPoints',
      'biconnectedComponents'
    ] as const) {
      expect(parallel[name]).toEqual(exact[name]);
      expect(parallel[name]).toEqual(Array.from(expected[name]));
    }
    expect(parallel['converged']).toEqual([1]);
  }
}, 120000);

it('composes linear-scratch SCC beyond 1024 vertices and reports budget exhaustion', async () => {
  const nodeCount = 5000;
  const edges: [number, number][] = Array.from({length: nodeCount}, (_, vertex) => {
    const first = Math.floor(vertex / 3) * 3;
    const size = Math.min(3, nodeCount - first);
    return [vertex, first + ((vertex - first + 1) % size)];
  });
  const result = await runStructure(createComponentCSR(nodeCount, edges), 'parallel');
  expect(result['strongComponents']).toEqual(
    Array.from({length: nodeCount}, (_, vertex) => Math.floor(vertex / 3) * 3)
  );
  expect(result['converged']).toEqual([1]);
  expect(result['iterationLimitReached']).toEqual([0]);
  const exhausted = await runStructure(
    createComponentCSR(
      33,
      Array.from({length: 32}, (_, vertex) => [vertex, (vertex + 1) % 32])
    ),
    'parallel',
    {maximumIterations: 1, repeats: 2}
  );
  expect(exhausted['strongComponents']).toEqual(new Array(33).fill(0xffffffff));
  expect(exhausted['converged']).toEqual([0]);
  expect(exhausted['iterationLimitReached']).toEqual([1]);
}, 120000);

it('finds the only bridge and two articulation points in a 100k-vertex split grid', async () => {
  const input = createSplitGridStructureCSR();
  const result = await runStructure(input, 'parallel', {strong: false});
  const bridgeVertices = [input.boundary - 1, input.boundary];
  const expectedArticulation = new Array(100000).fill(0);
  for (const vertex of bridgeVertices) expectedArticulation[vertex] = 1;
  expect(result['articulationPoints']).toEqual(expectedArticulation);
  const expectedBridges = new Array(input.neighbors.length).fill(0);
  const expectedBlocks = new Array(input.neighbors.length).fill(0);
  const minimumSlots = [0xffffffff, 0xffffffff];
  let bridgeSlot = 0xffffffff;
  for (let vertex = 0; vertex < 100000; vertex++) {
    const side = Number(vertex % input.width >= input.boundary);
    for (let slot = input.offsets[vertex]; slot < input.offsets[vertex + 1]; slot++) {
      const target = input.neighbors[slot];
      if (side !== Number(target % input.width >= input.boundary)) {
        bridgeSlot = Math.min(bridgeSlot, slot);
        expectedBridges[slot] = 1;
      } else minimumSlots[side] = Math.min(minimumSlots[side], slot);
    }
  }
  for (let vertex = 0; vertex < 100000; vertex++) {
    const side = Number(vertex % input.width >= input.boundary);
    for (let slot = input.offsets[vertex]; slot < input.offsets[vertex + 1]; slot++) {
      expectedBlocks[slot] = expectedBridges[slot] ? bridgeSlot : minimumSlots[side];
    }
  }
  expect(result['bridges']).toEqual(expectedBridges);
  expect(result['biconnectedComponents']).toEqual(expectedBlocks);
}, 300000);

for (const resilienceOutputs of [
  ['bridges'],
  ['articulationPoints'],
  ['biconnectedComponents'],
  ['bridges', 'articulationPoints', 'biconnectedComponents']
] as const) {
  it(`compiles and executes a two-vertex edge with ${resilienceOutputs.join(', ')}`, async () => {
    const input = createUndirectedStructureCSR(2, [[0, 1]]);
    const expected = computeNetworkStructureOnCPU(input);
    const result = await runStructure(input, 'parallel', {
      strong: false,
      resilienceOutputs,
      validateShaders: true
    });
    for (const name of resilienceOutputs) expect(result[name]).toEqual(Array.from(expected[name]));
  }, 120000);
}

for (const nodeCount of [513, 1024]) {
  it(`default SCC labels on a ${nodeCount}-vertex directed path match CPU through contributor and operation`, async () => {
    const input = createComponentCSR(
      nodeCount,
      Array.from({length: nodeCount - 1}, (_, vertex) => [vertex, vertex + 1])
    );
    const expected = computeNetworkStructureOnCPU(input).strongComponents;
    // Deliberately omit all mode, budget and status props, including through the wrapper.
    const direct = await runStructure(input, undefined, {status: false});
    expect(direct['strongComponents']).toEqual(Array.from(expected));
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const operation = new GPUNetworkStructureOperation({
      nodeCount,
      edgeCount: input.neighbors.length,
      topology: {offsets: 'offsets', neighbors: 'neighbors'},
      output: {strongComponents: 'labels'}
    });
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: input.offsets},
        neighbors: {format: 'uint32', values: input.neighbors}
      },
      operations: [operation],
      outputs: ['labels']
    });
    expect(result['labels']).toEqual(expected);
  }, 120000);
}

for (const nodeCount of [31, 32, 33, 255, 256, 257, 1023, 1024, 1025]) {
  for (const shape of ['path', 'high-root-star', 'cycle', 'parallel-and-loops'] as const) {
    it(`parallel resilience matches CPU and bounded exact outputs: ${shape}, ${nodeCount} vertices`, async () => {
      const edges: [number, number][] = Array.from({length: nodeCount - 1}, (_, vertex) =>
        shape === 'high-root-star' ? [vertex, nodeCount - 1] : [vertex, vertex + 1]
      );
      if (shape === 'cycle') edges.push([nodeCount - 1, 0]);
      if (shape === 'parallel-and-loops')
        edges.push(
          [0, 1],
          [0, 1],
          [0, 0],
          [Math.floor(nodeCount / 2), Math.floor(nodeCount / 2)],
          [nodeCount - 1, nodeCount - 1]
        );
      const input = createUndirectedStructureCSR(nodeCount, edges);
      const expected = computeNetworkStructureOnCPU(input);
      const parallel = await runStructure(input, 'parallel', {strong: false});
      const exact =
        nodeCount <= 1024 ? await runStructure(input, 'exact', {strong: false}) : undefined;
      for (const name of ['bridges', 'articulationPoints', 'biconnectedComponents'] as const) {
        expect(parallel[name]).toEqual(Array.from(expected[name]));
        if (exact) expect(parallel[name]).toEqual(exact[name]);
      }
    }, 120000);
  }
}

it('rejects physically aliased output bindings, including nonoverlapping rows with aligned-prefix overlap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const byteOffset of [0, 4]) {
    const graph = new GPUCommandGraph(device);
    const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
    try {
      const first = importGraphBuffer(graph, 'alias-first', buffer, 'uint32');
      const second = importGraphBuffer(graph, 'alias-second', buffer, 'uint32');
      const strongComponents = graph.createDataView(first.buffer, {format: 'uint32', length: 1});
      const articulationPoints = graph.createDataView(second.buffer, {
        format: 'uint32',
        byteOffset,
        length: 1
      });
      expect(
        () =>
          new GPUNetworkStructure({
            topology: {
              offsets: createTransientView(graph, 'alias-offsets', 'uint32', 2),
              neighbors: createTransientView(graph, 'alias-neighbors', 'uint32', 0)
            },
            reverseEdgeSlots: createTransientView(graph, 'alias-reverse', 'uint32', 0),
            output: {strongComponents, articulationPoints}
          })
      ).toThrow(/aligned binding byte ranges must not overlap/);
    } finally {
      buffer.destroy();
    }
  }
});
