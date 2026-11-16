// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkStructure} from '../../../src/gpu-network/network-structure/gpu-network-structure';
import {computeNetworkStructureOnCPU} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createUndirectedStructureCSR} from './network-structure-parallel-fixtures';
import {createComponentCSR, createRandomComponentEdges} from './strong-components-fixtures';

// Execute the generated WGSL itself with the existing interpreter, including indirect gates.
// The independent oracle is serial CPU Tarjan, rather than a transcription of the new kernels.
for (const fixture of [
  'empty',
  'bow-tie',
  'high-root-star',
  'path',
  'random-1',
  'random-9'
] as const) {
  it(`parallel resilience WGSL matches CPU Tarjan: ${fixture}`, () => {
    const nodeCount = fixture === 'empty' ? 1 : fixture === 'high-root-star' ? 33 : 17;
    let edges: [number, number][];
    if (fixture === 'empty') edges = [];
    else if (fixture === 'bow-tie')
      edges = [
        [0, 1],
        [1, 2],
        [2, 0],
        [2, 3],
        [3, 4],
        [4, 2],
        [4, 5],
        [5, 6],
        [5, 6],
        [2, 2],
        [2, 2]
      ];
    else if (fixture === 'high-root-star')
      edges = Array.from({length: nodeCount - 1}, (_, vertex) => [vertex, nodeCount - 1]);
    else if (fixture === 'path')
      edges = Array.from({length: nodeCount - 1}, (_, vertex) => [vertex, vertex + 1]);
    else edges = createRandomComponentEdges(nodeCount - 2, 30, fixture === 'random-1' ? 1 : 9);
    const input = createUndirectedStructureCSR(nodeCount, edges, fixture === 'empty' ? 0 : 2);
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = (length: number) =>
      createTransientView(graph, `interpreted-${serial++}`, 'uint32', length);
    const topology = {offsets: view(input.offsets.length), neighbors: view(input.neighbors.length)};
    const reverseEdgeSlots = view(input.reverseEdgeSlots.length);
    const output = {
      bridges: view(input.neighbors.length),
      articulationPoints: view(nodeCount),
      biconnectedComponents: view(input.neighbors.length)
    };
    const contributor = new GPUNetworkStructure({topology, reverseEdgeSlots, output});
    const nodes = contributor.getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(topology.offsets, input.offsets);
    memory.write(topology.neighbors, input.neighbors);
    memory.write(reverseEdgeSlots, input.reverseEdgeSlots);
    const expected = computeNetworkStructureOnCPU(input);
    for (let repeat = 0; repeat < 2; repeat++) {
      for (const node of nodes) {
        try {
          executeCommandNodesOnCPU([node], memory);
        } catch (error) {
          throw new Error(`WGSL node ${node.id}: ${String(error)}`, {cause: error});
        }
      }
      for (const name of ['bridges', 'articulationPoints', 'biconnectedComponents'] as const) {
        expect(Array.from(memory.readUint32(output[name]))).toEqual(Array.from(expected[name]));
      }
    }
  }, 120000);
}

it('composed SCC kernels match CPU labels and publish exhaustion status', () => {
  const input = createComponentCSR(7, [
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 5]
  ]);
  for (const maximumIterations of [1, 32]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = (length: number) =>
      createTransientView(graph, `scc-interpreted-${serial++}`, 'uint32', length);
    const topology = {offsets: view(input.offsets.length), neighbors: view(input.neighbors.length)};
    const output = {strongComponents: view(7)};
    const strongComponentsStatus = {
      converged: view(1),
      iterationCount: view(1),
      iterationLimitReached: view(1)
    };
    const contributor = new GPUNetworkStructure({
      mode: 'parallel',
      topology,
      output,
      strongComponentsStatus,
      strongComponentsMaximumIterations: maximumIterations
    });
    const memory = createCPUGraphMemory();
    memory.write(topology.offsets, input.offsets);
    memory.write(topology.neighbors, input.neighbors);
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    const finished = maximumIterations > 1;
    expect(Array.from(memory.readUint32(output.strongComponents))).toEqual(
      finished
        ? Array.from(computeNetworkStructureOnCPU(input).strongComponents)
        : new Array(7).fill(0xffffffff)
    );
    expect(Array.from(memory.readUint32(strongComponentsStatus.converged))).toEqual([
      Number(finished)
    ]);
    expect(Array.from(memory.readUint32(strongComponentsStatus.iterationLimitReached))).toEqual([
      Number(!finished)
    ]);
  }
});
