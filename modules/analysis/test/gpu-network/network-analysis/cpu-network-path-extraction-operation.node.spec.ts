// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkPathExtractionOnCPU,
  CPUNetworkPathExtractionOperation
} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {
  GPUNetworkPathExtractionOperation,
  type NetworkPathExtractionOperationProps
} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  buildCSR,
  dijkstra,
  F1_EDGES,
  predecessorOracle
} from '../network-reachability/network-reachability-oracle';
import {extractPathsOracle} from './network-path-extraction-oracle';

const NONE = 0xffffffff;
const INPUT = {
  predecessors: new Uint32Array([NONE, 0, 1, 0, NONE]),
  costs: new Float32Array([0, 2, 5, 7, Infinity]),
  targets: new Uint32Array([2, 3, 0, 4, 99]),
  nodeIds: new Uint32Array([10, 11, 12, 13, 14]),
  capacity: 4,
  edges: {
    offsets: new Uint32Array([0, 3, 4, 4, 4, 4]),
    neighbors: new Uint32Array([1, 1, 3, 2]),
    weights: new Float32Array([9, 2, 7, 3]),
    edgeIds: new Uint32Array([90, 20, 70, 30]),
    capacity: 2
  }
};
const PROPS: NetworkPathExtractionOperationProps = {
  id: 'path-test',
  nodeCount: 5,
  targetCapacity: 5,
  predecessors: 'predecessors',
  costs: 'costs',
  targets: 'targets',
  nodeIds: 'nodeIds',
  targetCount: 'active',
  predecessorsComplete: 'complete',
  output: {
    capacity: 4,
    ids: 'ids',
    lineOffsets: 'lines',
    pathCosts: 'pathCosts',
    pathFound: 'found',
    status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
  },
  edges: {
    edgeCount: 4,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    edgeIds: 'edgeIds',
    output: {
      capacity: 2,
      ids: 'edgeOutput',
      lineOffsets: 'edgeLines',
      status: {count: 'edgeCount', requiredCount: 'edgeRequired', overflow: 'edgeOverflow'}
    }
  }
};

it('packs a hand-derived forest with stable IDs, independent node/edge overflow and exact-slot preference', () => {
  // Paths are [0,1,2], [0,3], [0], [], []; edge slots are [1,3], [2], [], [], [].
  const result = computeNetworkPathExtractionOnCPU(INPUT);
  expect(Array.from(result.ids)).toEqual([10, 11, 12, 10]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 3, 5, 6, 6, 6]);
  expect(Array.from(result.pathFound)).toEqual([1, 1, 1, 0, 0]);
  expect(Array.from(result.pathCosts)).toEqual([5, 7, 0, Infinity, Infinity]);
  expect([result.count[0], result.requiredCount[0], result.overflow[0]]).toEqual([4, 6, 1]);
  expect(Array.from(result.edges!.ids)).toEqual([20, 30]);
  expect(Array.from(result.edges!.lineOffsets)).toEqual([0, 2, 3, 3, 3, 3]);
  expect([
    result.edges!.count[0],
    result.edges!.requiredCount[0],
    result.edges!.overflow[0]
  ]).toEqual([2, 3, 1]);
  const fallback = computeNetworkPathExtractionOnCPU({
    ...INPUT,
    targets: new Uint32Array([2]),
    edges: {...INPUT.edges, weights: new Float32Array([9, 8, 7, 3])}
  });
  expect(Array.from(fallback.edges!.ids)).toEqual([90, 30]);
  const missing = computeNetworkPathExtractionOnCPU({
    ...INPUT,
    targets: new Uint32Array([2]),
    edges: {...INPUT.edges, neighbors: new Uint32Array([99, 99, 3, 2])}
  });
  expect(Array.from(missing.edges!.ids)).toEqual([NONE, 30]);
});

it('honors active-count and completion flags, and rejects broken/cyclic chains without truncation', () => {
  const active = computeNetworkPathExtractionOnCPU({...INPUT, targetCount: new Uint32Array([1])});
  expect(Array.from(active.lineOffsets)).toEqual([0, 3, 3, 3, 3, 3]);
  expect(Array.from(active.pathFound)).toEqual([1, 0, 0, 0, 0]);
  const incomplete = computeNetworkPathExtractionOnCPU({
    ...INPUT,
    predecessorsComplete: new Uint32Array([0])
  });
  expect(incomplete.requiredCount[0]).toBe(0);
  expect(incomplete.pathFound.every(value => value === 0)).toBe(true);
  expect(incomplete.pathCosts.every(value => value === Infinity)).toBe(true);
  for (const predecessors of [new Uint32Array([1, 0]), new Uint32Array([99, 0])]) {
    const result = computeNetworkPathExtractionOnCPU({
      predecessors,
      costs: new Float32Array([0, 1]),
      targets: new Uint32Array([1]),
      capacity: 2
    });
    expect([
      result.count[0],
      result.requiredCount[0],
      result.overflow[0],
      result.pathFound[0]
    ]).toEqual([0, 0, 0, 0]);
  }
  const empty = computeNetworkPathExtractionOnCPU({
    ...INPUT,
    targets: new Uint32Array(0),
    capacity: 0
  });
  expect(Array.from(empty.lineOffsets)).toEqual([0]);
  expect(empty.count[0]).toBe(0);
  expect(Array.from(empty.edges!.ids)).toEqual([0, 0]);
});

it('honors an explicit walk budget and reports candidate loss with ample final capacity', () => {
  const nodeCount = 65537;
  const result = computeNetworkPathExtractionOnCPU({
    predecessors: Uint32Array.from({length: nodeCount}, (_, node) =>
      node === 0 ? NONE : node - 1
    ),
    costs: Float32Array.from({length: nodeCount}, (_, node) => node),
    targets: new Uint32Array([nodeCount - 1]),
    maximumPathLength: 1,
    capacity: 1
  });
  expect(result.requiredCount[0]).toBe(0);
  expect(result.pathFound[0]).toBe(0);
  expect(result.overflow[0]).toBe(0);
  expect(result.candidateOverflow[0]).toBe(1);
  expect(result.ids[0]).toBe(0);
});

it('separates bounded-path loss from final capacity on the F1 node and edge outputs', () => {
  const topology = buildCSR(8, F1_EDGES);
  const costs = dijkstra(topology, 8, [{node: 0, cost: 0}]);
  const predecessors = predecessorOracle(topology, costs, [{node: 0, cost: 0}]);
  const input = {
    predecessors,
    costs,
    targets: new Uint32Array([4, 0]),
    maximumPathLength: 4,
    capacity: 16,
    edges: {...topology, capacity: 16}
  };
  const result = computeNetworkPathExtractionOnCPU(input);
  // 0 -> 2 -> 1 -> 3 -> 4 needs five nodes, so only the root target survives the budget.
  expect(Array.from(result.pathFound)).toEqual([0, 1]);
  expect([result.count[0], result.requiredCount[0], result.edges!.count[0]]).toEqual([1, 1, 0]);
  expect(Array.from(result.ids.slice(0, result.count[0]))).toEqual([0]);
  expect([
    result.overflow[0],
    result.candidateOverflow[0],
    result.edges!.overflow[0],
    result.edges!.candidateOverflow[0]
  ]).toEqual([0, 1, 0, 1]);
  const reference = extractPathsOracle({
    predecessors,
    costs,
    targets: input.targets,
    maxPathLength: 4,
    capacity: 16,
    csr: topology,
    edgeCapacity: 16
  });
  expect([reference.ids, reference.edgeIds, reference.found]).toEqual([[0], [], [0, 1]]);
  expect([reference.overflow, reference.candidateOverflow, reference.edgeOverflow]).toEqual([
    0, 1, 0
  ]);
});

it('uses exact float64 edge consistency before choosing the smallest slot', () => {
  // At 2^24 the shader rounds +1 back to the same cost. Only the zero-weight slot is
  // mathematically consistent with the supplied equal labels, so the CPU selects slot 1.
  const result = computeNetworkPathExtractionOnCPU({
    predecessors: new Uint32Array([NONE, 0]),
    costs: new Float32Array([2 ** 24, 2 ** 24]),
    targets: new Uint32Array([1]),
    capacity: 2,
    edges: {
      offsets: new Uint32Array([0, 2, 2]),
      neighbors: new Uint32Array([1, 1]),
      weights: new Float32Array([1, 0]),
      capacity: 1
    }
  });
  expect(result.edges!.ids[0]).toBe(1);
});

it('rejects inconsistent arrays and validates both operation constructions', () => {
  for (const [argument, values] of Object.entries({
    costs: new Float32Array(1),
    nodeIds: new Uint32Array(1),
    targetCount: new Uint32Array(2),
    predecessorsComplete: new Uint32Array(2)
  }))
    expect(() => computeNetworkPathExtractionOnCPU({...INPUT, [argument]: values})).toThrow(
      new RegExp(argument)
    );
  expect(() =>
    computeNetworkPathExtractionOnCPU({
      ...INPUT,
      edges: {...INPUT.edges, weights: new Float32Array(1)}
    })
  ).toThrow(/edges.weights/);
  expect(() => new CPUNetworkPathExtractionOperation({...PROPS, maximumPathLength: 0})).toThrow(
    /path-test.*maximumPathLength/
  );
  expect(() => new CPUNetworkPathExtractionOperation({...PROPS, targetCount: 6})).toThrow(
    /path-test.*targetCount/
  );
  const shapes = {
    predecessors: {format: 'uint32' as const, length: 5},
    costs: {format: 'float32' as const, length: 5},
    targets: {format: 'uint32' as const, length: 5},
    nodeIds: {format: 'uint32' as const, length: 5},
    targetCount: {format: 'uint32' as const, length: 1},
    predecessorsComplete: {format: 'uint32' as const, length: 1},
    offsets: {format: 'uint32' as const, length: 6},
    neighbors: {format: 'uint32' as const, length: 4},
    weights: {format: 'float32' as const, length: 4},
    edgeIds: {format: 'uint32' as const, length: 4}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkPathExtractionOperation({...PROPS, maximumPathLength: 0}),
      shapes
    )
  ).toThrow(/path-test.*maximumPathLength/);
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkPathExtractionOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUNetworkPathExtractionOperation, CPUNetworkPathExtractionOperation])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, ids: 'targets'}})).toThrow(
      /path-test/
    );
  expect(
    () => new CPUNetworkPathExtractionOperation({...PROPS, maximumPathLength: 65537})
  ).not.toThrow();
});
