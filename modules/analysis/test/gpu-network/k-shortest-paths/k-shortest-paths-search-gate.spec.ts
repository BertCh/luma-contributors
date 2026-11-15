// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUKShortestPaths} from '../../../src/gpu-network/k-shortest-paths/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('inactive and early-finished batches dispatch zero search passes, including after graph reuse', async () => {
  const testDevice = await getWebGPUTestDevice();
  requireTestDevice(testDevice);
  const device = testDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let nextView = 0;
  const input = <Format extends 'uint32' | 'float32'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `gate-input-${nextView++}`, buffer, format, values.length)
    };
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `gate-output-${nextView++}`, buffer, format, length)
    };
  };
  const nodeCount = 100000;
  const routeCount = 32;
  const pathLength = 8;
  // The only route is 0 -> 1. Remaining vertices are isolated; the search exhausts in round 1.
  const offsets = new Uint32Array(nodeCount + 1).fill(1);
  offsets[0] = 0;
  const sources = input(Uint32Array.of(nodeCount), 'uint32');
  const counts = output(1, 'uint32');
  const iterationCount = output(1, 'uint32');
  const converged = output(1, 'uint32');
  const passes = output(routeCount, 'uint32');
  const contributor = new GPUKShortestPaths({
    topology: {
      offsets: input(offsets, 'uint32').view,
      neighbors: input(Uint32Array.of(1), 'uint32').view,
      weights: input(Float32Array.of(1), 'float32').view
    },
    sources: sources.view,
    targets: input(Uint32Array.of(1), 'uint32').view,
    routeCount,
    maximumPathLength: pathLength,
    maximumIterations: 2,
    output: {
      vertices: output(routeCount * pathLength, 'uint32').view,
      edges: output(routeCount * pathLength, 'uint32').view,
      lengths: output(routeCount, 'uint32').view,
      costs: output(routeCount, 'float32').view,
      counts: counts.view,
      status: {
        converged: converged.view,
        iterationLimitReached: output(1, 'uint32').view,
        candidateOverflow: output(1, 'uint32').view,
        iterationCount: iterationCount.view
      }
    }
  });
  graph.add(
    createFillNode(graph, {
      id: 'clear-search-pass-probes',
      operation: 'KShortestPathsGateProbe',
      view: passes.view,
      type: 'u32',
      value: '0u'
    })
  );
  for (const node of contributor.getCommandNodes(graph)) {
    graph.add(node);
    if (node.workload?.operation !== 'GPUNetworkReachability') continue;
    const round = Number(/^k-shortest-paths-(\d+)-shortest-/.exec(node.id)![1]);
    // A one-word atomic probe immediately follows each actual search pass, with the same GPU
    // condition and dispatch triple. A nonzero count proves that search work was dispatched;
    // absence of a condition deliberately leaves the probe unconditional (the pre-fix failure).
    graph.add(
      createWGSLKernelNode(graph, {
        id: `${node.id}-probe`,
        operation: 'KShortestPathsGateProbe',
        invocationCount: 1,
        condition: node.condition,
        extraResources:
          node.condition?.source === 'gpu'
            ? [{buffer: node.condition.buffer, usage: 'indirect'}]
            : [],
        bindings: [{name: 'passes', view: passes.view, type: 'atomic<u32>', access: 'read_write'}],
        body: `atomicAdd(&passes[passesOffset + ${round}u], 1u);`
      })
    );
  }
  const compiled = graph.compile();
  try {
    for (const source of [nodeCount, 0, nodeCount, 0]) {
      sources.buffer.write(Uint32Array.of(source));
      await submitGraph(device, compiled, undefined);
      const actualPasses = Array.from(await readUint32(passes.buffer, routeCount));
      expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
      if (source === nodeCount) {
        expect(Array.from(await readUint32(counts.buffer, 1))).toEqual([0]);
        expect(Array.from(await readUint32(iterationCount.buffer, 1))).toEqual([0]);
        expect(actualPasses).toEqual(new Array(routeCount).fill(0));
      } else {
        expect(Array.from(await readUint32(counts.buffer, 1))).toEqual([1]);
        expect(Array.from(await readUint32(iterationCount.buffer, 1))).toEqual([2]);
        expect(actualPasses[0]).toBeGreaterThan(0);
        expect(actualPasses[1]).toBeGreaterThan(0);
        expect(actualPasses.slice(2)).toEqual(new Array(routeCount - 2).fill(0));
      }
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
