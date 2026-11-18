// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkPathExtraction} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createRandomNetwork,
  F1_EDGES,
  NONE,
  type NetworkCSR
} from '../network-reachability/network-reachability-oracle';
import {extractPathsOracle} from './network-path-extraction-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {formatNetworkPathReadout} from '../../../../../examples/deck/spatial-analysis-shared/network-path-readout';

type HarnessOptions = {
  nodeCount: number;
  csr: NetworkCSR;
  /** Run `GPUNetworkReachability` from this source into the predecessor and cost buffers. */
  reachabilitySource?: number;
  /** Directly written predecessors and costs when no reachability runs. */
  predecessors?: number[];
  costs?: number[];
  targets: number[];
  targetCount?: number;
  capacity: number;
  edgeCapacity?: number;
  edgeIds?: number[];
  nodeIds?: number[];
  maxPathLength?: number;
  maxIterations?: number;
};

function createHarness(device: Device, options: HarnessOptions) {
  const {nodeCount, csr} = options;
  const graph = new GPUCommandGraph(device, {id: 'path-extraction'});
  const buffers: Buffer[] = [];
  const parameters: GPUParameterBuffer[] = [];
  const track = <B extends Buffer>(buffer: B): B => {
    buffers.push(buffer);
    return buffer;
  };
  const edgeCount = csr.neighbors.length;
  const offsetsBuffer = track(createInputBuffer(device, csr.offsets));
  const neighborsBuffer = track(
    createInputBuffer(device, edgeCount ? csr.neighbors : new Uint32Array(1))
  );
  const weightsBuffer = track(
    createInputBuffer(device, edgeCount ? csr.weights : new Float32Array(1))
  );
  const predecessorsBuffer = track(createOutputBuffer(device, nodeCount));
  const costsBuffer = track(createOutputBuffer(device, nodeCount));
  if (options.predecessors) {
    predecessorsBuffer.write(Uint32Array.from(options.predecessors));
  }
  if (options.costs) {
    costsBuffer.write(Float32Array.from(options.costs));
  }
  const offsets = importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', nodeCount + 1);
  const neighbors = importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', edgeCount);
  const weights = importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', edgeCount);
  const predecessors = importGraphBuffer(
    graph,
    'predecessors',
    predecessorsBuffer,
    'uint32',
    nodeCount
  );
  const costs = importGraphBuffer(graph, 'costs', costsBuffer, 'float32', nodeCount);
  function createParameter(
    name: string,
    format: 'uint32',
    values: number[]
  ): GPUParameterBuffer<'uint32'>;
  function createParameter(
    name: string,
    format: 'float32',
    values: number[]
  ): GPUParameterBuffer<'float32'>;
  function createParameter(
    name: string,
    format: 'uint32' | 'float32',
    values: number[]
  ): GPUParameterBuffer<'uint32'> | GPUParameterBuffer<'float32'> {
    const parameter =
      format === 'uint32'
        ? new GPUParameterBuffer(device, {
            id: `path-${name}`,
            format,
            length: values.length,
            values: Uint32Array.from(values)
          })
        : new GPUParameterBuffer(device, {
            id: `path-${name}`,
            format,
            length: values.length,
            values: Float32Array.from(values)
          });
    parameters.push(parameter);
    return parameter;
  }
  if (options.reachabilitySource !== undefined) {
    const sources = createParameter('sources', 'uint32', [options.reachabilitySource]);
    graph.add(
      new GPUNetworkReachability({
        id: 'reach',
        topology: {offsets, neighbors, weights},
        startNodes: sources.importToGraph(graph),
        output: {costs, predecessors},
        maximumIterations: options.maxIterations ?? 256
      })
    );
  }
  const targetCapacity = options.targets.length;
  const targetsParameter = createParameter('targets', 'uint32', options.targets);
  const targetCountParameter =
    options.targetCount !== undefined
      ? createParameter('target-count', 'uint32', [options.targetCount])
      : undefined;
  const nodeIdsParameter = options.nodeIds
    ? createParameter('node-ids', 'uint32', options.nodeIds)
    : undefined;
  const edgeIdsParameter = options.edgeIds
    ? createParameter('edge-ids', 'uint32', options.edgeIds)
    : undefined;
  const withEdges = options.edgeCapacity !== undefined;
  const outputIds = track(createOutputBuffer(device, options.capacity));
  const outputCount = track(createOutputBuffer(device, 1));
  const outputOverflow = track(createOutputBuffer(device, 1));
  const outputCandidate = track(createOutputBuffer(device, 1));
  const outputTotal = track(createOutputBuffer(device, 1));
  const pathOffsets = track(createOutputBuffer(device, targetCapacity + 1));
  const pathCosts = track(createOutputBuffer(device, targetCapacity));
  const pathFound = track(createOutputBuffer(device, targetCapacity));
  const edgeIdsOut = track(createOutputBuffer(device, options.edgeCapacity ?? 1));
  const edgeCountOut = track(createOutputBuffer(device, 1));
  const edgeOverflowOut = track(createOutputBuffer(device, 1));
  const edgeCandidate = track(createOutputBuffer(device, 1));
  const edgeTotalOut = track(createOutputBuffer(device, 1));
  const edgePathOffsets = track(createOutputBuffer(device, targetCapacity + 1));
  const contributor = new GPUNetworkPathExtraction({
    id: 'path',
    predecessors,
    costs,
    targets: targetsParameter.importToGraph(graph),
    targetCount: targetCountParameter?.importToGraph(graph),
    maximumPathLength: options.maxPathLength,
    nodeIds: nodeIdsParameter?.importToGraph(graph),
    output: {
      ids: importGraphBuffer(graph, 'out-ids', outputIds, 'uint32', options.capacity),
      status: {
        count: importGraphBuffer(graph, 'out-count', outputCount, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'out-overflow', outputOverflow, 'uint32', 1),
        candidateOverflow: importGraphBuffer(graph, 'out-candidate', outputCandidate, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'out-total', outputTotal, 'uint32', 1)
      },
      lineOffsets: importGraphBuffer(
        graph,
        'path-offsets',
        pathOffsets,
        'uint32',
        targetCapacity + 1
      ),
      pathCosts: importGraphBuffer(graph, 'path-costs', pathCosts, 'float32', targetCapacity),
      pathFound: importGraphBuffer(graph, 'path-found', pathFound, 'uint32', targetCapacity)
    },
    edges: withEdges
      ? {
          topology: {offsets, neighbors, weights},
          edgeIds: edgeIdsParameter?.importToGraph(graph),
          output: {
            ids: importGraphBuffer(graph, 'edge-ids', edgeIdsOut, 'uint32', options.edgeCapacity),
            status: {
              count: importGraphBuffer(graph, 'edge-count', edgeCountOut, 'uint32', 1),
              overflow: importGraphBuffer(graph, 'edge-overflow', edgeOverflowOut, 'uint32', 1),
              candidateOverflow: importGraphBuffer(
                graph,
                'edge-candidate',
                edgeCandidate,
                'uint32',
                1
              ),
              requiredCount: importGraphBuffer(graph, 'edge-total', edgeTotalOut, 'uint32', 1)
            },
            lineOffsets: importGraphBuffer(
              graph,
              'edge-path-offsets',
              edgePathOffsets,
              'uint32',
              targetCapacity + 1
            )
          }
        }
      : undefined
  });
  graph.add(contributor);
  const compiled: CompiledGPUCommandGraph<void> = graph.compile();
  const state = {
    targets: [...options.targets],
    targetCount: options.targetCount as number | undefined
  };
  return {
    state,
    compiled,
    setTargets(values: number[]) {
      state.targets = [...values];
      targetsParameter.write(Uint32Array.from(values));
    },
    setTargetCount(value: number) {
      state.targetCount = value;
      targetCountParameter!.write(Uint32Array.from([value]));
    },
    /** Encodes once and compares every output with the oracle; returns the oracle result. */
    async run() {
      submitGraph(device, compiled, undefined);
      const predecessorValues = await readUint32(predecessorsBuffer, nodeCount);
      const costValues = await readFloat32(costsBuffer, nodeCount);
      const expected = extractPathsOracle({
        predecessors: predecessorValues,
        costs: costValues,
        targets: state.targets,
        targetCount: state.targetCount,
        maxPathLength: options.maxPathLength ?? 1024,
        nodeIds: options.nodeIds,
        capacity: options.capacity,
        csr: withEdges ? csr : undefined,
        edgeIds: options.edgeIds,
        edgeCapacity: options.edgeCapacity
      });
      const [count] = await readUint32(outputCount, 1);
      expect(count).toBe(expected.count);
      expect(await readUint32(outputIds, count)).toEqual(expected.ids);
      expect(await readUint32(outputTotal, 1)).toEqual([expected.requiredCount]);
      expect(await readUint32(outputOverflow, 1)).toEqual([expected.overflow]);
      expect(await readUint32(outputCandidate, 1)).toEqual([expected.candidateOverflow]);
      expect(await readUint32(pathOffsets, targetCapacity + 1)).toEqual(expected.pathOffsets);
      expect(await readFloat32(pathCosts, targetCapacity)).toEqual(expected.pathCosts);
      expect(await readUint32(pathFound, targetCapacity)).toEqual(expected.found);
      if (withEdges) {
        const [edgeCountValue] = await readUint32(edgeCountOut, 1);
        expect(edgeCountValue).toBe(expected.edgeCount);
        expect(await readUint32(edgeIdsOut, edgeCountValue)).toEqual(expected.edgeIds);
        expect(await readUint32(edgeTotalOut, 1)).toEqual([expected.edgeTotalCount]);
        expect(await readUint32(edgeOverflowOut, 1)).toEqual([expected.edgeOverflow]);
        expect(await readUint32(edgeCandidate, 1)).toEqual([expected.candidateOverflow]);
        expect(await readUint32(edgePathOffsets, targetCapacity + 1)).toEqual(
          expected.edgePathOffsets
        );
      }
      return expected;
    },
    destroy() {
      compiled.destroy();
      for (const parameter of parameters) parameter.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

const F1 = buildCSR(8, F1_EDGES);

it('GPUNetworkPathExtraction extracts nodes and edges after GPUNetworkReachability', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [4, 6, 0, 99],
    capacity: 16,
    edgeCapacity: 16,
    maxPathLength: 16,
    maxIterations: 16
  });
  const result = await harness.run();
  expect(result.ids).toEqual([0, 2, 1, 3, 4, 0]);
  expect(result.found).toEqual([1, 0, 1, 0]);
  expect(result.pathOffsets).toEqual([0, 5, 5, 6, 6]);
  expect(result.pathCosts).toEqual([7, Infinity, 0, Infinity]);
  expect(result.edgeIds).toEqual([1, 3, 2, 5]);
  expect(result.overflow).toBe(0);
  harness.destroy();

  const mapped = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [4, 6, 0, 99],
    capacity: 16,
    edgeCapacity: 16,
    maxPathLength: 16,
    maxIterations: 16,
    nodeIds: [100, 101, 102, 103, 104, 105, 106, 107],
    edgeIds: [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
  });
  const mappedResult = await mapped.run();
  expect(mappedResult.ids).toEqual([100, 102, 101, 103, 104, 100]);
  expect(mappedResult.edgeIds).toEqual([201, 203, 202, 205]);
  mapped.destroy();
});

it('GPUNetworkPathExtraction matches the oracle on a random network', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 2000;
  const csr = buildCSR(nodeCount, createRandomNetwork(11, nodeCount, 8000));
  let state = 12345;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const targets = Array.from({length: 64}, () => Math.floor(next() * nodeCount));
  const harness = createHarness(device, {
    nodeCount,
    csr,
    reachabilitySource: 0,
    targets,
    capacity: 20000,
    edgeCapacity: 20000,
    maxPathLength: 512,
    maxIterations: 256,
    nodeIds: Array.from({length: nodeCount}, (_, node) => node * 3 + 7),
    edgeIds: Array.from({length: csr.neighbors.length}, (_, edge) => edge * 5 + 1)
  });
  const result = await harness.run();
  expect(result.found.filter(Boolean).length).toBeGreaterThan(8);
  expect(result.requiredCount).toBeGreaterThan(64);
  expect(result.edgeIds.every(edge => edge !== NONE)).toBe(true);
  harness.destroy();
});

it('path readouts distinguish a three-node chain beyond a two-node walk budget from a disconnected target', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reachable of [true, false]) {
    const harness = createHarness(device, {
      nodeCount: 3,
      csr: buildCSR(
        3,
        reachable
          ? [
              [0, 1, 1],
              [1, 2, 1]
            ]
          : [[0, 1, 1]]
      ),
      reachabilitySource: 0,
      targets: [2],
      capacity: 3,
      edgeCapacity: 2,
      maxPathLength: 2,
      maxIterations: 4
    });
    try {
      const result = await harness.run();
      expect(result.found).toEqual([0]);
      expect([result.count, result.overflow, result.candidateOverflow]).toEqual([
        0,
        0,
        Number(reachable)
      ]);
      expect(
        formatNetworkPathReadout(
          {
            found: result.found[0] === 1,
            converged: 1,
            overflow: result.overflow,
            candidateOverflow: result.candidateOverflow
          },
          'found'
        )
      ).toBe(reachable ? 'incomplete batch (path walk budget exhausted)' : 'no route found');
    } finally {
      harness.destroy();
    }
  }
});

it('mixed path batch preserves the found target cost when another walk exhausts its budget', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    nodeCount: 3,
    csr: buildCSR(3, [
      [0, 1, 60],
      [1, 2, 60]
    ]),
    reachabilitySource: 0,
    targets: [1, 2],
    capacity: 6,
    edgeCapacity: 4,
    maxPathLength: 2,
    maxIterations: 4
  });
  try {
    const result = await harness.run();
    expect(result.found).toEqual([1, 0]);
    expect(result.pathCosts).toEqual([60, Infinity]);
    expect([result.count, result.overflow, result.candidateOverflow]).toEqual([2, 0, 1]);
    expect(
      result.found.map((found, index) =>
        formatNetworkPathReadout(
          {
            found: found === 1,
            converged: 1,
            overflow: result.overflow,
            candidateOverflow: result.candidateOverflow
          },
          `${result.pathCosts[index] / 60} min`
        )
      )
    ).toEqual(['1 min', 'incomplete batch (path walk budget exhausted)']);
  } finally {
    harness.destroy();
  }
});

it('GPUNetworkPathExtraction terminates on garbage predecessors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    // Cycle 1 -> 2 -> 3 -> 1, self-loop at 4, out-of-range predecessor at 5, chain 6 -> 7.
    predecessors: [NONE, 3, 1, 2, 4, 77, NONE, 6],
    costs: [1, 1, 1, 1, 1, 1, 1, 1],
    targets: [1, 4, 5, 6, 7, 0],
    capacity: 32,
    edgeCapacity: 32,
    maxPathLength: 8
  });
  const result = await harness.run();
  expect(result.found).toEqual([0, 0, 0, 1, 1, 1]);
  expect(result.overflow).toBe(0);
  expect(result.candidateOverflow).toBe(1);
  expect(result.ids).toEqual([6, 6, 7, 0]);
  harness.setTargets([5, 6, 7, 0, 5, 6]);
  const clean = await harness.run();
  expect(clean.overflow).toBe(0);
  expect(clean.candidateOverflow).toBe(0);
  expect(clean.found).toEqual([0, 1, 1, 1, 0, 1]);
  harness.destroy();
});

it('GPUNetworkPathExtraction separates final capacity overflow from bounded-path loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [4, 0],
    capacity: 3,
    edgeCapacity: 2,
    maxPathLength: 16,
    maxIterations: 16
  });
  const result = await harness.run();
  expect(result.count).toBe(3);
  expect(result.requiredCount).toBe(6);
  expect(result.overflow).toBe(1);
  expect(result.edgeCount).toBe(2);
  expect(result.edgeTotalCount).toBe(4);
  expect(result.edgeOverflow).toBe(1);
  expect(result.candidateOverflow).toBe(0);
  harness.destroy();

  const tight = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [4],
    capacity: 5,
    edgeCapacity: 4,
    maxPathLength: 5,
    maxIterations: 16
  });
  const exact = await tight.run();
  expect(exact.overflow).toBe(0);
  expect(exact.edgeOverflow).toBe(0);
  expect(exact.candidateOverflow).toBe(0);
  expect(exact.count).toBe(5);
  tight.destroy();

  const truncated = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [4, 0],
    capacity: 16,
    edgeCapacity: 16,
    maxPathLength: 4,
    maxIterations: 16
  });
  const truncatedResult = await truncated.run();
  expect(truncatedResult.found).toEqual([0, 1]);
  expect(truncatedResult.ids).toEqual([0]);
  expect([truncatedResult.count, truncatedResult.requiredCount, truncatedResult.edgeCount]).toEqual(
    [1, 1, 0]
  );
  expect(truncatedResult.overflow).toBe(0);
  expect(truncatedResult.edgeOverflow).toBe(0);
  expect(truncatedResult.candidateOverflow).toBe(1);
  // Reuse the same compiled graph with only root targets; both candidate words must clear.
  truncated.setTargets([0, 0]);
  const clean = await truncated.run();
  expect(clean.found).toEqual([1, 1]);
  expect(clean.ids).toEqual([0, 0]);
  expect([clean.overflow, clean.edgeOverflow, clean.candidateOverflow]).toEqual([0, 0, 0]);
  truncated.destroy();
});

it('GPUNetworkPathExtraction re-encodes after target and count changes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [4, 0, 3, 1],
    targetCount: 2,
    capacity: 16,
    edgeCapacity: 16,
    maxPathLength: 16,
    maxIterations: 16
  });
  expect((await harness.run()).requiredCount).toBe(6);
  harness.setTargetCount(4);
  expect((await harness.run()).requiredCount).toBe(6 + 4 + 3);
  harness.setTargets([1, 3, 4, 0]);
  expect((await harness.run()).ids.slice(0, 3)).toEqual([0, 2, 1]);
  harness.setTargetCount(0);
  const empty = await harness.run();
  expect(empty.count).toBe(0);
  expect(empty.found).toEqual([0, 0, 0, 0]);
  harness.setTargetCount(1);
  expect((await harness.run()).count).toBe(3);
  harness.destroy();
});

it('GPUNetworkPathExtraction publishes zero counts without targets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    nodeCount: 8,
    csr: F1,
    reachabilitySource: 0,
    targets: [],
    capacity: 4,
    edgeCapacity: 4,
    maxPathLength: 16,
    maxIterations: 16
  });
  const result = await harness.run();
  expect(result.count).toBe(0);
  expect(result.pathOffsets).toEqual([0]);
  harness.destroy();

  const single = createHarness(device, {
    nodeCount: 1,
    csr: buildCSR(1, []),
    predecessors: [NONE],
    costs: [0],
    targets: [0, 0],
    capacity: 2,
    edgeCapacity: 2,
    maxPathLength: 1
  });
  const singleResult = await single.run();
  expect(singleResult.ids).toEqual([0, 0]);
  expect(singleResult.overflow).toBe(0);
  expect(singleResult.edgeTotalCount).toBe(0);
  single.destroy();
});

it('GPUNetworkPathExtraction extracts full routes across zero-weight edges and a zero-weight cycle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 0 -2-> 1 -0-> 2 <-0-> 3 -0-> 4 -1-> 5 -0-> 6, with the source on 0 and a zero-weight cycle 2 <-> 3.
  const csr = buildCSR(8, [
    [0, 1, 2],
    [1, 2, 0],
    [2, 3, 0],
    [3, 2, 0],
    [3, 4, 0],
    [4, 5, 1],
    [5, 6, 0]
  ]);
  const harness = createHarness(device, {
    nodeCount: 8,
    csr,
    reachabilitySource: 0,
    targets: [6, 4, 3, 0, 7],
    capacity: 32,
    edgeCapacity: 32,
    maxPathLength: 16,
    maxIterations: 16
  });
  const result = await harness.run();
  expect(result.found).toEqual([1, 1, 1, 1, 0]);
  expect(result.ids).toEqual([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0]);
  expect(result.pathCosts).toEqual([3, 2, 2, 0, Infinity]);
  expect(result.edgeIds).toEqual([0, 1, 2, 4, 5, 6, 0, 1, 2, 4, 0, 1, 2]);
  expect(result.overflow).toBe(0);
  harness.destroy();
});
