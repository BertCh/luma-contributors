// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {NetworkCentralityOperationProps} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {expect} from 'vitest';
import {expectOracleRows} from '../alternative-routes/output-oracle-harness';

export function expectCentralityOracleOutputs(
  selected: string,
  result: Record<string, ArrayLike<number>>
) {
  const requested = selected === 'all' ? Object.keys(centralityOracleExpected) : [selected];
  expect(Object.keys(result).sort()).toEqual([...requested].sort());
  for (const port of requested) {
    const expected = centralityOracleExpected[port as keyof typeof centralityOracleExpected];
    expectOracleRows(result[port], expected, port === 'reachedCounts' ? 0 : 2e-5);
  }
}

// Unit diamond 0->{1,2}->3, plus isolated node 4. From pivot 0, distances are 0,1,1,2.
// Each middle vertex carries half of the two shortest 0->3 paths; scale 2 gives 1 each.
// Edge dependencies are 1.5,1.5,.5,.5 before scaling. Closeness=3/4; harmonic=1+1+1/2.
// Pivot 3 reaches only itself; invalid pivot 99 reaches nothing. Lane count 2 tests the batch tail.
export const centralityOracleInput = {
  topology: {
    offsets: Uint32Array.of(0, 2, 3, 4, 4, 4),
    neighbors: Uint32Array.of(1, 2, 3, 3),
    weights: Float32Array.of(1, 1, 1, 1)
  },
  reverseTopology: {
    offsets: Uint32Array.of(0, 0, 1, 2, 4, 4),
    neighbors: Uint32Array.of(0, 0, 1, 2),
    weights: Float32Array.of(1, 1, 1, 1)
  },
  pivotNodes: Uint32Array.of(0, 3, 99)
};
export const centralityOracleExpected = {
  betweenness: [0, 1, 1, 0, 0],
  edgeBetweenness: [3, 3, 1, 1],
  closeness: [0.75, 0, 0],
  harmonicCloseness: [2.5, 0, 0],
  reachedCounts: [4, 1, 0]
};
export function createCentralityOracleProps(): NetworkCentralityOperationProps {
  return {
    nodeCount: 5,
    edgeCount: 4,
    reverseEdgeCount: 4,
    pivotCount: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    reverseTopology: {
      offsets: 'reverseOffsets',
      neighbors: 'reverseNeighbors',
      weights: 'reverseWeights'
    },
    pivotNodes: 'pivots',
    laneCount: 2,
    maximumIterations: 8,
    betweennessScale: 2,
    sampling: 'sampled',
    output: {}
  };
}
export const centralityOracleGraphInputs: Record<string, OperationGraphInput> = {
  offsets: {format: 'uint32', values: centralityOracleInput.topology.offsets},
  neighbors: {format: 'uint32', values: centralityOracleInput.topology.neighbors},
  weights: {format: 'float32', values: centralityOracleInput.topology.weights},
  reverseOffsets: {format: 'uint32', values: centralityOracleInput.reverseTopology.offsets},
  reverseNeighbors: {format: 'uint32', values: centralityOracleInput.reverseTopology.neighbors},
  reverseWeights: {format: 'float32', values: centralityOracleInput.reverseTopology.weights},
  pivots: {format: 'uint32', values: centralityOracleInput.pivotNodes}
};

export function createCentralityBudgetOracle(betweenness: boolean, maximumIterations = 1) {
  // Unit chain 0->1->2, pivot 0: one search hop reaches only 0 and 1. Thus r=2,
  // closeness=(r-1)/1=1 and harmonic=1. Dependency phases spend one edge per round,
  // independently of search localIterations; with two rounds they leave the root/sink queued.
  const props = createCentralityOracleProps();
  Object.assign(props, {
    nodeCount: 3,
    edgeCount: 2,
    reverseEdgeCount: 2,
    pivotCount: 1,
    laneCount: 1,
    maximumIterations,
    localIterations: betweenness ? 3 : 1,
    betweennessScale: 1
  });
  props.output = {
    closeness: 'closeness',
    harmonicCloseness: 'harmonic',
    reachedCounts: 'reached',
    ...(betweenness ? {betweenness: 'between', edgeBetweenness: 'edges'} : {}),
    status: {
      converged: 'converged',
      iterationLimitReached: 'limited',
      ...(betweenness ? {zeroWeightCycle: 'cycle'} : {})
    }
  };
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
    neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
    weights: {format: 'float32', values: Float32Array.of(1, 1)},
    reverseOffsets: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 2)},
    reverseNeighbors: {format: 'uint32', values: Uint32Array.of(0, 1)},
    reverseWeights: {format: 'float32', values: Float32Array.of(1, 1)},
    pivots: {format: 'uint32', values: Uint32Array.of(0)}
  };
  const expected: Record<string, number[]> = betweenness
    ? {
        closeness: [2 / 3],
        harmonic: [1.5],
        reached: [3],
        between: maximumIterations < 3 ? [0, 0, 0] : [0, 1, 0],
        edges: maximumIterations === 1 ? [0, 0] : maximumIterations === 2 ? [1, 0] : [2, 1],
        converged: [maximumIterations >= 3 ? 1 : 0],
        limited: [maximumIterations >= 3 ? 0 : 1],
        cycle: [0]
      }
    : {closeness: [1], harmonic: [1], reached: [2], converged: [0], limited: [1]};
  return {props, inputs, expected};
}
