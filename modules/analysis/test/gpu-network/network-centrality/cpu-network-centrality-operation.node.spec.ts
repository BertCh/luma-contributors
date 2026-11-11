// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkCentralityOnCPU,
  CPUNetworkCentralityOperation
} from '../../../src/gpu-network/network-centrality/cpu-network-centrality-operation';
import {
  GPUNetworkCentralityOperation,
  type NetworkCentralityOperationProps
} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {buildCentralityCSRs, CENTRALITY_REFERENCE} from './network-centrality-oracle';

const PROPS: NetworkCentralityOperationProps = {
  id: 'centrality-test',
  nodeCount: 4,
  edgeCount: 4,
  reverseEdgeCount: 4,
  pivotCount: 2,
  topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  reverseTopology: {
    offsets: 'reverseOffsets',
    neighbors: 'reverseNeighbors',
    weights: 'reverseWeights'
  },
  pivotNodes: 'pivots',
  output: {
    betweenness: 'between',
    edgeBetweenness: 'edges',
    closeness: 'close',
    harmonicCloseness: 'harmonic',
    reachedCounts: 'reached',
    status: {converged: 'converged', zeroWeightCycle: 'cycle'}
  }
};

function expectRelative(actual: ArrayLike<number>, expected: ArrayLike<number>): void {
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < expected.length; row++)
    expect(Math.abs(actual[row] - expected[row])).toBeLessThanOrEqual(
      2e-6 * Math.abs(expected[row])
    );
}

it('matches committed networkx fixtures, including sampled pivots and inclusive radii', () => {
  for (const scene of Object.values(CENTRALITY_REFERENCE.scenes)) {
    const {forward, reverse} = buildCentralityCSRs(scene.nodeCount, scene.edges);
    for (const bounded of [false, true]) {
      for (const sampled of [false, true]) {
        const pivots = sampled
          ? scene.sampledPivots
          : Array.from({length: scene.nodeCount}, (_, row) => row);
        const result = computeNetworkCentralityOnCPU(
          {
            topology: forward,
            reverseTopology: reverse,
            pivotNodes: Uint32Array.from(pivots),
            costLimit: bounded ? Float32Array.of(scene.radius) : undefined
          },
          {betweennessScale: sampled ? scene.sampledScale : 1}
        );
        const expected = bounded
          ? sampled
            ? scene.betweenness.radiusSampled
            : scene.betweenness.radius
          : sampled
            ? scene.betweenness.sampled
            : scene.betweenness.exact;
        expectRelative(
          result.betweenness,
          expected.map(value => value * (sampled ? scene.sampledScale : 1))
        );
        const reference = bounded ? scene.closeness.radius : scene.closeness.unbounded;
        expectRelative(
          result.closeness,
          pivots.map(pivot => reference.closeness[pivot])
        );
        expectRelative(
          result.harmonicCloseness,
          pivots.map(pivot => reference.harmonicCloseness[pivot])
        );
        expect(Array.from(result.reachedCounts)).toEqual(
          pivots.map(pivot => reference.reachedCounts[pivot])
        );
        expect(Array.from(result.converged)).toEqual([1]);
        expect(Array.from(result.zeroWeightCycle)).toEqual([0]);
      }
    }
  }
});

it('counts hand-enumerated diamond paths, duplicate pivots and zero-weight chains', () => {
  const {forward, reverse} = buildCentralityCSRs(5, [
    [0, 1, 1],
    [0, 2, 2],
    [1, 3, 1],
    [2, 3, 0],
    [3, 4, 3]
  ]);
  const result = computeNetworkCentralityOnCPU({
    topology: forward,
    reverseTopology: reverse,
    pivotNodes: Uint32Array.of(0, 0, 99)
  });
  // For each pivot: targets 1 and 2 use one branch, targets 3 and 4 split equally.
  expect(Array.from(result.betweenness)).toEqual([0, 2, 2, 2, 0]);
  expect(Array.from(result.edgeBetweenness)).toEqual([4, 4, 2, 2, 2]);
  expect(Array.from(result.reachedCounts)).toEqual([5, 5, 0]);
  expectRelative(result.closeness, [0.4, 0.4, 0]);
  expectRelative(result.harmonicCloseness, [2.2, 2.2, 0]);
});

it('retains frontier partial results at a zero cycle and ignores impassable slots', () => {
  const {forward, reverse} = buildCentralityCSRs(5, [
    [0, 1, 1],
    [1, 2, 0],
    [2, 1, 0],
    [2, 3, 1],
    [0, 4, -1],
    [0, 4, NaN],
    [0, 4, Infinity],
    [0, 0, 0]
  ]);
  const input = {topology: forward, reverseTopology: reverse, pivotNodes: Uint32Array.of(0)};
  const result = computeNetworkCentralityOnCPU(input);
  expect(Array.from(result.betweenness)).toEqual([0, 0, 0, 0, 0]);
  expect(Array.from(result.edgeBetweenness)).toEqual(new Array(8).fill(0));
  expect(Array.from(result.reachedCounts)).toEqual([4]);
  expect(Array.from(result.converged)).toEqual([0]);
  expect(Array.from(result.zeroWeightCycle)).toEqual([1]);
  expect(
    Array.from(computeNetworkCentralityOnCPU(input, {computeBetweenness: false}).converged)
  ).toEqual([1]);
});

it('does not overflow 2^1100 shortest paths or absorb a small positive edge', () => {
  const nodeCount = 1101;
  const {forward, reverse} = buildCentralityCSRs(
    nodeCount,
    Array.from({length: nodeCount - 1}, (_, source): [number, number, number][] => [
      [source, source + 1, 1],
      [source, source + 1, 1]
    ]).flat()
  );
  const result = computeNetworkCentralityOnCPU({
    topology: forward,
    reverseTopology: reverse,
    pivotNodes: Uint32Array.of(0)
  });
  expect(result.betweenness[1]).toBe(1099);
  expect(result.edgeBetweenness[0]).toBe(550);
  expect(result.edgeBetweenness.every(Number.isFinite)).toBe(true);
  const small = buildCentralityCSRs(3, [
    [0, 1, 2 ** 24],
    [1, 2, 1]
  ]);
  expect(
    Array.from(
      computeNetworkCentralityOnCPU({
        topology: small.forward,
        reverseTopology: small.reverse,
        pivotNodes: Uint32Array.of(0)
      }).betweenness
    )
  ).toEqual([0, 1, 0]);
});

it('applies the cost bound to pivot seeding and every reachable path', () => {
  const {forward, reverse} = buildCentralityCSRs(3, [
    [0, 1, 0],
    [1, 2, 2]
  ]);
  const input = {topology: forward, reverseTopology: reverse, pivotNodes: Uint32Array.of(0)};
  const negative = computeNetworkCentralityOnCPU({...input, costLimit: Float32Array.of(-1)});
  expect(Array.from(negative.reachedCounts)).toEqual([0]);
  expect(Array.from(negative.betweenness)).toEqual([0, 0, 0]);
  expect(Array.from(negative.edgeBetweenness)).toEqual([0, 0]);
  expect(Array.from(negative.closeness)).toEqual([0]);
  expect(Array.from(negative.harmonicCloseness)).toEqual([0]);
  expect(Array.from(negative.converged)).toEqual([1]);
  const zero = computeNetworkCentralityOnCPU({...input, costLimit: Float32Array.of(0)});
  expect(Array.from(zero.reachedCounts)).toEqual([2]);
  expect(Array.from(zero.edgeBetweenness)).toEqual([1, 0]);
  const inclusive = computeNetworkCentralityOnCPU({...input, costLimit: Float32Array.of(2)});
  expect(Array.from(inclusive.reachedCounts)).toEqual([3]);
  expect(Array.from(inclusive.betweenness)).toEqual([0, 1, 0]);
  // The shader compares candidate > NaN, which is false: the CPU retains that behavior.
  expect(
    Array.from(
      computeNetworkCentralityOnCPU({...input, costLimit: Float32Array.of(NaN)}).reachedCounts
    )
  ).toEqual([3]);
});

it('checks array lengths but preserves clamped GPU CSR intervals', () => {
  const {forward, reverse} = buildCentralityCSRs(2, [[0, 1, 1]]);
  const input = {topology: forward, reverseTopology: reverse, pivotNodes: Uint32Array.of(0)};
  expect(() =>
    computeNetworkCentralityOnCPU({...input, topology: {...forward, weights: new Float32Array(0)}})
  ).toThrow(/topology.weights/);
  expect(() =>
    computeNetworkCentralityOnCPU({
      ...input,
      reverseTopology: {...reverse, offsets: new Uint32Array(2)}
    })
  ).toThrow(/reverseTopology.offsets/);
  expect(() => computeNetworkCentralityOnCPU({...input, costLimit: new Float32Array(2)})).toThrow(
    /costLimit/
  );
  const empty = {
    offsets: Uint32Array.of(9, 0, 99),
    neighbors: Uint32Array.of(99),
    weights: Float32Array.of(1)
  };
  expect(
    Array.from(
      computeNetworkCentralityOnCPU({
        topology: empty,
        reverseTopology: empty,
        pivotNodes: Uint32Array.of(0)
      }).reachedCounts
    )
  ).toEqual([1]);
});

it('validates both operation forms at construction, with deferred GPU contributor checks', () => {
  const shapes = {
    'topology.offsets': {format: 'uint32' as const, length: 5},
    'topology.neighbors': {format: 'uint32' as const, length: 4},
    'topology.weights': {format: 'float32' as const, length: 4},
    'reverseTopology.offsets': {format: 'uint32' as const, length: 5},
    'reverseTopology.neighbors': {format: 'uint32' as const, length: 4},
    'reverseTopology.weights': {format: 'float32' as const, length: 4},
    pivotNodes: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkCentralityOperation(PROPS), shapes)
  ).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkCentralityOperation({...PROPS, maximumIterations: 0}),
      shapes
    )
  ).toThrow(/centrality-test.*maximumIterations/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkCentralityOperation({...PROPS, betweennessScale: Infinity}),
      shapes
    )
  ).toThrow(/centrality-test.*betweennessScale/);
  expect(() => new CPUNetworkCentralityOperation({...PROPS, nodeCount: 0})).toThrow(
    /centrality-test.*nodeCount/
  );
  expect(() => new CPUNetworkCentralityOperation({...PROPS, betweennessScale: Infinity})).toThrow(
    /centrality-test.*betweennessScale/
  );
  expect(() => new CPUNetworkCentralityOperation({...PROPS, output: {}})).toThrow(
    /centrality-test.*output/
  );
  expect(
    () =>
      new CPUNetworkCentralityOperation({...PROPS, maximumIterations: 100000, laneCount: 100000})
  ).not.toThrow();
  for (const Operation of [GPUNetworkCentralityOperation, CPUNetworkCentralityOperation])
    expect(() => new Operation({...PROPS, output: {betweenness: 'weights'}})).toThrow(
      /centrality-test/
    );
});
