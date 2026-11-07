// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkStructure} from '../../../src/gpu-network/network-structure/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

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
    return state >>> 8;
  };
  return Array.from({length: edgeCount}, () => [next() % vertexCount, next() % vertexCount]);
}

async function runNetworkStrongComponents(vertexCount: number, edges: readonly [number, number][]) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const offsets = new Array<number>(vertexCount + 1).fill(0);
  for (const [source] of edges) offsets[source + 1]++;
  for (let vertex = 0; vertex < vertexCount; vertex++) offsets[vertex + 1] += offsets[vertex];
  const cursor = offsets.slice(0, vertexCount);
  const neighbors = new Array<number>(Math.max(edges.length, 1)).fill(0);
  for (const [source, target] of edges) neighbors[cursor[source]++] = target;
  const buffers = [
    createInputBuffer(device, Uint32Array.from(offsets)),
    createInputBuffer(device, Uint32Array.from(neighbors)),
    createOutputBuffer(device, vertexCount)
  ];
  const [offsetsBuffer, neighborsBuffer, labelsBuffer] = buffers;
  const structure = new GPUNetworkStructure({
    topology: {
      offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', offsets.length),
      neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', neighbors.length)
    },
    output: {
      strongComponents: importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', vertexCount)
    }
  });
  graph.add(structure);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {labels: await readUint32(labelsBuffer, vertexCount)};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('GPUNetworkStructure reachability strong components match Tarjan on bounded graphs', async () => {
  for (const [vertexCount, edgeCount, seed] of [
    [200, 260, 21],
    [1000, 1500, 11]
  ]) {
    const edges = createRandomEdges(vertexCount, edgeCount, seed);
    const result = await runNetworkStrongComponents(vertexCount, edges);
    expect(result.labels).toEqual(calculateTarjanLabels(vertexCount, edges));
  }
});
