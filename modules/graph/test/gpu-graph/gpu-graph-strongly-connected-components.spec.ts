// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUGraphAlgorithmsData,
  GPUGraphAlgorithmsTopology
} from '@luma-contributors/graph/gpu-graph-algorithms';
// Not yet exported from the barrel: public export needs a lens operator, formula, scenario and docs.
import {GPUGraphAlgorithmsStronglyConnectedComponents} from '../../src/gpu-graph-algorithms/gpu-graph-strongly-connected-components';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';

const INVALID_COMPONENT = 0xffffffff;

type SccResult = {labels: number[]; converged: number};

/** Iterative Tarjan reference labelling each component with its lowest vertex id. */
function calculateTarjanLabels(vertexCount: number, edges: readonly [number, number][]): number[] {
  const adjacency: number[][] = Array.from({length: vertexCount}, () => []);
  for (const [source, target] of edges) adjacency[source].push(target);
  const index = new Array<number>(vertexCount).fill(-1);
  const lowLink = new Array<number>(vertexCount).fill(0);
  const onStack = new Array<boolean>(vertexCount).fill(false);
  const labels = new Array<number>(vertexCount).fill(0);
  const stack: number[] = [];
  let counter = 0;
  for (let root = 0; root < vertexCount; root++) {
    if (index[root] !== -1) continue;
    const work: [number, number][] = [[root, 0]];
    index[root] = lowLink[root] = counter++;
    stack.push(root);
    onStack[root] = true;
    while (work.length > 0) {
      const frame = work[work.length - 1];
      const vertex = frame[0];
      if (frame[1] < adjacency[vertex].length) {
        const next = adjacency[vertex][frame[1]++];
        if (index[next] === -1) {
          index[next] = lowLink[next] = counter++;
          stack.push(next);
          onStack[next] = true;
          work.push([next, 0]);
        } else if (onStack[next]) {
          lowLink[vertex] = Math.min(lowLink[vertex], index[next]);
        }
      } else {
        work.pop();
        if (work.length > 0) {
          const parent = work[work.length - 1][0];
          lowLink[parent] = Math.min(lowLink[parent], lowLink[vertex]);
        }
        if (lowLink[vertex] === index[vertex]) {
          const members: number[] = [];
          let member: number;
          do {
            member = stack.pop()!;
            onStack[member] = false;
            members.push(member);
          } while (member !== vertex);
          const label = Math.min(...members);
          for (const entry of members) labels[entry] = label;
        }
      }
    }
  }
  return labels;
}

function createRandomEdges(
  vertexCount: number,
  edgeCount: number,
  seed: number
): [number, number][] {
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
  return Array.from({length: edgeCount}, () => [next() % vertexCount, next() % vertexCount]);
}

async function runStronglyConnectedComponents(
  device: Device,
  vertexCount: number,
  edges: readonly [number, number][],
  iterations: number,
  invalidateOnNonConvergence = true
): Promise<SccResult> {
  const buffers: Buffer[] = [];
  const vectors: GPUVector[] = [];
  const createInput = (name: string, values: number[]) => {
    const buffer = device.createBuffer({
      id: name,
      data: values.length > 0 ? Uint32Array.from(values) : new Uint32Array(1),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const vector = new GPUVector<'uint32'>({
      type: 'data',
      name,
      format: 'uint32',
      data: [new GPUData<'uint32'>({buffer, format: 'uint32', length: values.length})],
      ownsData: false
    });
    vectors.push(vector);
    return vector;
  };
  const createOutput = (name: string, length: number) => {
    const buffer = device.createBuffer({
      id: name,
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC
    });
    buffers.push(buffer);
    const vector = new GPUVector<'uint32'>({
      type: 'buffer',
      name,
      format: 'uint32',
      buffer,
      length,
      byteOffset: 0,
      ownsBuffer: false
    });
    vectors.push(vector);
    return vector;
  };
  const graph = new GPUGraphAlgorithmsData({
    vertexCount,
    sourceVertices: createInput(
      'sources',
      edges.map(edge => edge[0])
    ),
    targetVertices: createInput(
      'targets',
      edges.map(edge => edge[1])
    ),
    directed: true
  });
  const capacity = edges.length;
  const topology = new GPUGraphAlgorithmsTopology({
    graph,
    forward: {
      offsets: createOutput('offsets', vertexCount + 1),
      neighbors: createOutput('neighbors', capacity),
      edgeIds: createOutput('edge-ids', capacity),
      count: createOutput('count', 1),
      overflow: createOutput('overflow', 1)
    },
    invalidEdgeCount: createOutput('invalid', 1)
  });
  const output = createOutput('labels', vertexCount);
  const converged = createOutput('converged', 1);
  const components = new GPUGraphAlgorithmsStronglyConnectedComponents({
    topology,
    output,
    converged,
    iterations,
    invalidateOnNonConvergence
  });
  const commandGraph = new GPUCommandGraph(device);
  topology.addToGraph(commandGraph);
  components.addToGraph(commandGraph);
  const compiled = commandGraph.compile();
  try {
    const commandEncoder = device.createCommandEncoder({id: 'scc-test'});
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const read = async (vector: GPUVector<'uint32'>) => {
      const bytes = await (vector.data[0].buffer as Buffer).readAsync(0, vector.length * 4);
      return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, vector.length));
    };
    return {labels: await read(output), converged: (await read(converged))[0]};
  } finally {
    compiled.destroy();
    for (const vector of vectors) vector.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

const hexagonScenarios: {name: string; vertexCount: number; edges: [number, number][]}[] = [
  {name: 'empty edge list leaves singleton components', vertexCount: 4, edges: []},
  {
    name: 'cycle, bridge, and tail',
    vertexCount: 7,
    edges: [
      [1, 2],
      [2, 3],
      [3, 1],
      [3, 4],
      [4, 5],
      [5, 4],
      [0, 1],
      [5, 6],
      [6, 6]
    ]
  },
  {
    name: 'two cycles joined by shared vertex ordering traps',
    vertexCount: 6,
    edges: [
      [5, 4],
      [4, 3],
      [3, 5],
      [2, 1],
      [1, 0],
      [0, 2],
      [3, 2]
    ]
  }
];

for (const scenario of hexagonScenarios) {
  it(`GPUGraphAlgorithmsStronglyConnectedComponents matches Tarjan: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const result = await runStronglyConnectedComponents(
      device,
      scenario.vertexCount,
      scenario.edges,
      32
    );
    expect(result.labels).toEqual(calculateTarjanLabels(scenario.vertexCount, scenario.edges));
    expect(result.converged).toBe(1);
  });
}

it('GPUGraphAlgorithmsStronglyConnectedComponents matches Tarjan on large random graphs beyond the 1024-node exact cap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const [vertexCount, edgeCount, seed] of [
    [5000, 4000, 1],
    [5000, 7500, 2],
    [4096, 12000, 3]
  ]) {
    const edges = createRandomEdges(vertexCount, edgeCount, seed);
    const result = await runStronglyConnectedComponents(device, vertexCount, edges, 512);
    expect(result.converged, `converged for ${vertexCount}/${edgeCount}`).toBe(1);
    expect(result.labels).toEqual(calculateTarjanLabels(vertexCount, edges));
  }
});

it('GPUGraphAlgorithmsStronglyConnectedComponents handles adversarial chains and nested cycles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const vertexCount = 300;
  const ascending: [number, number][] = Array.from({length: vertexCount - 1}, (_, vertex) => [
    vertex,
    vertex + 1
  ]);
  const descending = ascending.map(([source, target]): [number, number] => [target, source]);
  const ring: [number, number][] = [...ascending, [vertexCount - 1, 0]];
  const ringWithChords: [number, number][] = [
    ...ring,
    ...createRandomEdges(vertexCount, 50, 9).map(([source, target]): [number, number] => [
      Math.max(source, target),
      Math.min(source, target)
    ])
  ];
  for (const edges of [ascending, descending, ring, ringWithChords]) {
    const result = await runStronglyConnectedComponents(device, vertexCount, edges, 1024);
    expect(result.converged).toBe(1);
    expect(result.labels).toEqual(calculateTarjanLabels(vertexCount, edges));
  }
  // Chains of cycles linked in a line need one colour/mark/finalize cycle per component.
  const blockCount = 12;
  const chainOfCycles: [number, number][] = [];
  for (let block = 0; block < blockCount; block++) {
    const base = block * 3;
    chainOfCycles.push([base, base + 1], [base + 1, base + 2], [base + 2, base]);
    if (block + 1 < blockCount) chainOfCycles.push([base + 2, base + 3]);
  }
  const reversedChainOfCycles = chainOfCycles.map(([source, target]): [number, number] => [
    target,
    source
  ]);
  for (const edges of [chainOfCycles, reversedChainOfCycles]) {
    const result = await runStronglyConnectedComponents(device, blockCount * 3, edges, 1024);
    expect(result.converged).toBe(1);
    expect(result.labels).toEqual(calculateTarjanLabels(blockCount * 3, edges));
  }
});

it('GPUGraphAlgorithmsStronglyConnectedComponents fails closed or keeps exact partial labels when the budget ends', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const vertexCount = 200;
  const edges: [number, number][] = Array.from({length: vertexCount - 1}, (_, vertex) => [
    vertex + 1,
    vertex
  ]);
  const invalidated = await runStronglyConnectedComponents(device, vertexCount, edges, 3);
  expect(invalidated.converged).toBe(0);
  expect(invalidated.labels.every(label => label === INVALID_COMPONENT)).toBe(true);
  const partial = await runStronglyConnectedComponents(device, vertexCount, edges, 3, false);
  expect(partial.converged).toBe(0);
  const reference = calculateTarjanLabels(vertexCount, edges);
  expect(
    partial.labels.every(
      (label, vertex) => label === INVALID_COMPONENT || label === reference[vertex]
    )
  ).toBe(true);
});
