// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkReachabilityOnCPU} from '../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {computeNetworkPathExtractionOnCPU} from '../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {computeKShortestPathsOnCPU} from '../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';
import {computeNetworkFlowOnCPU} from '../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPU_NETWORK_FLOW_WORD} from '../../src/gpu-network/network-optimization/gpu-network-flow';

function createChain(nodeCount: number, weight = 1) {
  return {
    offsets: Uint32Array.from({length: nodeCount + 1}, (_, row) => Math.min(row, nodeCount - 1)),
    neighbors: Uint32Array.from({length: nodeCount - 1}, (_, row) => row + 1),
    weights: new Float32Array(nodeCount - 1).fill(weight)
  };
}

it('network:F5 zero active reachability rounds suppress the downstream path', () => {
  const result = computeNetworkReachabilityOnCPU({
    topology: createChain(3),
    startNodes: Uint32Array.of(0),
    activeIterations: Uint32Array.of(0)
  });
  expect(result.costs).toEqual(Float32Array.of(0, Infinity, Infinity));
  expect(result.converged).toEqual(Uint32Array.of(0));
  expect(result.iterationCount).toEqual(Uint32Array.of(0));
  expect(result.iterationLimitReached).toEqual(Uint32Array.of(1));
  const path = computeNetworkPathExtractionOnCPU({
    predecessors: result.predecessors,
    costs: result.costs,
    predecessorsComplete: result.converged,
    targets: Uint32Array.of(2),
    capacity: 3
  });
  expect(path.pathFound).toEqual(Uint32Array.of(0));
  expect(path.count).toEqual(Uint32Array.of(0));
});

it('network:F5 reachability binds positive relaxation and zero-cost tie budgets', () => {
  const input = {topology: createChain(3), startNodes: Uint32Array.of(0)};
  const options = {maximumIterations: 1, localIterations: 1};
  const limited = computeNetworkReachabilityOnCPU(input, options);
  expect(limited.costs).toEqual(Float32Array.of(0, 1, Infinity));
  expect(limited.converged[0]).toBe(0);
  const complete = computeNetworkReachabilityOnCPU(input, {...options, maximumIterations: 3});
  expect(complete.costs).toEqual(Float32Array.of(0, 1, 2));
  expect(complete.converged[0]).toBe(1);
  const ties = computeNetworkReachabilityOnCPU(
    {
      ...input,
      topology: createChain(3, 0)
    },
    {maximumIterations: 4, localIterations: 1, maximumTieIterations: 1}
  );
  expect(ties.converged[0]).toBe(0);
  expect(ties.unresolvedCount[0]).toBe(1);
});

it('network:F5 one shortest-search round cannot certify the 300-vertex K-shortest route', () => {
  const input = {
    topology: createChain(300),
    sources: Uint32Array.of(0),
    targets: Uint32Array.of(299)
  };
  const options = {routeCount: 2, maximumPathLength: 300, maximumIterations: 1};
  const result = computeKShortestPathsOnCPU(input, options);
  expect(result.counts).toEqual(Uint32Array.of(0));
  expect(result.converged).toEqual(Uint32Array.of(0));
  expect(result.iterationLimitReached).toEqual(Uint32Array.of(1));
  expect(computeKShortestPathsOnCPU(input, {...options, maximumIterations: 8}).counts[0]).toBe(1);
});

it('network:F5 zero augmentations retains feasible zero flow without certifying completion', () => {
  const input = {
    edges: Uint32Array.of(0, 1),
    capacities: Uint32Array.of(1),
    parameters: Uint32Array.of(0, 1, 1)
  };
  const options = {nodeCount: 2, maximumAugmentations: 0};
  const result = computeNetworkFlowOnCPU(input, options);
  expect(result.flow).toEqual(Uint32Array.of(0));
  expect(result.summary[GPU_NETWORK_FLOW_WORD.complete]).toBe(0);
  expect(result.summary[GPU_NETWORK_FLOW_WORD.unserved]).toBe(1);
  expect(result.summary[GPU_NETWORK_FLOW_WORD.augmentations]).toBe(0);
  expect(result.status.iterationLimitReached[0]).toBe(1);
  expect(
    computeNetworkFlowOnCPU(input, {...options, maximumAugmentations: 1}).summary[
      GPU_NETWORK_FLOW_WORD.complete
    ]
  ).toBe(1);
});

it('network:F5 bounded frontier diagnostics separate repeated visits from final reached edges', () => {
  const result = computeNetworkReachabilityOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 2, 3, 4, 4),
        neighbors: Uint32Array.of(1, 2, 3, 1),
        weights: Float32Array.of(5, 1, 1, 1)
      },
      startNodes: Uint32Array.of(0)
    },
    {maximumIterations: 1, localIterations: 32}
  );
  // Node 1 is first reached at 5, then improved to 2. Its outgoing edge is visited twice.
  expect(result.costs).toEqual(Float32Array.of(0, 2, 1, 3));
  expect(result.relaxationCount).toEqual(Uint32Array.of(5, 0));
  expect(result.reachedEdgeCount).toEqual(Uint32Array.of(4, 0));
  expect(result.converged[0]).toBe(1);
});

it('network:F10 mapped identities retain separate rows for positional node and edge gathers', () => {
  const result = computeNetworkPathExtractionOnCPU({
    predecessors: Uint32Array.of(0xffffffff, 0, 1),
    predecessorEdges: Uint32Array.of(0xffffffff, 0, 1),
    costs: Float32Array.of(0, 1, 2),
    targets: Uint32Array.of(2),
    capacity: 3,
    nodeIds: Uint32Array.of(100, 200, 300),
    edges: {...createChain(3), edgeIds: Uint32Array.of(900, 800), capacity: 2}
  });
  expect(result.ids).toEqual(Uint32Array.of(100, 200, 300));
  expect(result).toHaveProperty('nodeRows', Uint32Array.of(0, 1, 2));
  expect(result.edges?.ids).toEqual(Uint32Array.of(900, 800));
  expect(result.edges).toHaveProperty('edgeRows', Uint32Array.of(0, 1));
  const positions = [
    [10, 20],
    [30, 40],
    [50, 60]
  ];
  expect(
    Array.from((result as typeof result & {nodeRows: Uint32Array}).nodeRows, row => positions[row])
  ).toEqual(positions);
});
