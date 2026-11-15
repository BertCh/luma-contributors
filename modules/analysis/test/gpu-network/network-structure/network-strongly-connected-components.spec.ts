// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  GPUNetworkStronglyConnectedComponents,
  computeNetworkStructureOnCPU
} from '../../../src/gpu-network/network-structure/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createComponentCSR, createRandomComponentEdges} from './strong-components-fixtures';

const INVALID_COMPONENT = 0xffffffff;
const CANARY = 0xabcdef01;

async function runComponents(
  nodeCount: number,
  edges: readonly (readonly [number, number])[],
  maximumIterations = 512,
  padding = 0,
  repeats = 1
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const input = createComponentCSR(nodeCount, edges, padding);
  const buffers: Buffer[] = [];
  // Exercise nonzero element offsets and rounded-up dispatch lanes on every port.
  const prefix = 3;
  const suffix = 2;
  const importInput = (name: string, values: Uint32Array) => {
    const padded = new Uint32Array(prefix + values.length + suffix).fill(CANARY);
    padded.set(values, prefix);
    const buffer = createInputBuffer(device, padded);
    buffers.push(buffer);
    const whole = importGraphBuffer(graph, name, buffer, 'uint32');
    return graph.createDataView(whole.buffer, {
      format: 'uint32',
      length: values.length,
      byteOffset: prefix * 4
    });
  };
  const importOutput = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, prefix + length + suffix);
    buffer.write(new Uint32Array(prefix + length + suffix).fill(CANARY));
    buffers.push(buffer);
    const whole = importGraphBuffer(graph, name, buffer, 'uint32');
    return {
      buffer,
      length,
      view: graph.createDataView(whole.buffer, {
        format: 'uint32',
        length,
        byteOffset: prefix * 4
      })
    };
  };
  const topology = {
    offsets: importInput('offsets', input.offsets),
    neighbors: importInput('neighbors', input.neighbors)
  };
  const labels = importOutput('labels', nodeCount);
  const converged = importOutput('converged', 1);
  const iterationCount = importOutput('iteration-count', 1);
  const iterationLimitReached = importOutput('iteration-limit-reached', 1);
  graph.add(
    new GPUNetworkStronglyConnectedComponents({
      topology,
      maximumIterations,
      output: {
        strongComponents: labels.view,
        status: {
          converged: converged.view,
          iterationCount: iterationCount.view,
          iterationLimitReached: iterationLimitReached.view
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    const readOutput = async (output: ReturnType<typeof importOutput>) => {
      const values = await readUint32(output.buffer);
      expect(values.slice(0, prefix)).toEqual(new Array(prefix).fill(CANARY));
      expect(values.slice(prefix + output.length)).toEqual(new Array(suffix).fill(CANARY));
      return values.slice(prefix, prefix + output.length);
    };
    let result = {
      labels: [] as number[],
      converged: 0,
      iterationCount: 0,
      iterationLimitReached: 0
    };
    for (let repeat = 0; repeat < repeats; repeat++) {
      submitGraph(device, compiled, undefined);
      result = {
        labels: await readOutput(labels),
        converged: (await readOutput(converged))[0],
        iterationCount: (await readOutput(iterationCount))[0],
        iterationLimitReached: (await readOutput(iterationLimitReached))[0]
      };
      expect(result.iterationCount).toBeLessThanOrEqual(maximumIterations);
      expect(result.iterationCount).toBeGreaterThan(0);
    }
    return {result, reference: Array.from(computeNetworkStructureOnCPU(input).strongComponents)};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const [nodeCount, edgeCount, seed] of [
  [1024, 2600, 21],
  [4096, 12000, 3],
  [5000, 4000, 1],
  [5000, 7500, 2],
  // Also exceed the old 65536-slot bound.
  [5000, 70000, 9]
]) {
  it(`linear-scratch SCC matches CPU Tarjan on ${nodeCount} vertices/${edgeCount} random edges`, async () => {
    const {result, reference} = await runComponents(
      nodeCount,
      createRandomComponentEdges(nodeCount, edgeCount, seed)
    );
    expect(result.labels).toEqual(reference);
    expect(result.converged).toBe(1);
    expect(result.iterationLimitReached).toBe(0);
  }, 120000);
}

it('handles DAGs in both orders and a single large cycle', async () => {
  const nodeCount = 1024;
  const ascending: [number, number][] = Array.from({length: nodeCount - 1}, (_, vertex) => [
    vertex,
    vertex + 1
  ]);
  const descending = ascending.map(([source, target]): [number, number] => [target, source]);
  for (const edges of [
    ascending,
    descending,
    [...ascending, [nodeCount - 1, 0] as [number, number]]
  ]) {
    const {result, reference} = await runComponents(nodeCount, edges, 3072);
    expect(result.labels).toEqual(reference);
    expect(result.converged).toBe(1);
    expect(result.iterationLimitReached).toBe(0);
  }
}, 180000);

it('handles self-loops, isolates, parallel edges, invalid targets and unowned CSR padding', async () => {
  const edges: [number, number][] = [
    [0, 0],
    [2, 1],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 6],
    [6, 5],
    [6, 2],
    [4, 0xffffffff],
    [5, 8]
  ];
  const {result, reference} = await runComponents(8, edges, 64, 7, 2);
  expect(reference).toEqual([0, 1, 1, 3, 3, 5, 5, 7]);
  expect(result.labels).toEqual(reference);
  expect(result.converged).toBe(1);
  expect(result.iterationLimitReached).toBe(0);
});

it('resolves empty adjacency and singleton self-loops at the exact one-step boundary', async () => {
  for (const [nodeCount, edges] of [
    [257, []],
    [1, [[0, 0]]]
  ] as const) {
    const {result, reference} = await runComponents(nodeCount, edges, 1);
    expect(result.labels).toEqual(reference);
    expect(result.converged).toBe(1);
    expect(result.iterationCount).toBe(1);
    expect(result.iterationLimitReached).toBe(0);
  }
});

it('invalidates resolved and unresolved labels when the step budget is exhausted, on every run', async () => {
  const edges: [number, number][] = Array.from({length: 32}, (_, vertex) => [
    vertex,
    (vertex + 1) % 32
  ]);
  // Vertex 32 is trimmed, but its otherwise valid label must also be invalidated.
  const {result} = await runComponents(33, edges, 1, 3, 2);
  expect(result.labels).toEqual(new Array(33).fill(INVALID_COMPONENT));
  expect(result.converged).toBe(0);
  expect(result.iterationCount).toBe(1);
  expect(result.iterationLimitReached).toBe(1);
});

it('separates chains of SCCs in either direction after repeated colour/finalize phases', async () => {
  const edges: [number, number][] = [];
  const blockCount = 12;
  for (let block = 0; block < blockCount; block++) {
    const base = block * 3;
    edges.push([base, base + 1], [base + 1, base + 2], [base + 2, base]);
    if (block + 1 < blockCount) edges.push([base + 2, base + 3]);
  }
  for (const directed of [
    edges,
    edges.map(([source, target]): [number, number] => [target, source])
  ]) {
    const {result, reference} = await runComponents(blockCount * 3, directed, 512);
    expect(result.labels).toEqual(reference);
    expect(result.converged).toBe(1);
    expect(result.iterationLimitReached).toBe(0);
  }
}, 120000);
