// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkCostTableOnCPU} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {computeMapMatchingOnCPU} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {getGPUMapMatchingParameterValues} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {searchRouteMirror} from './route-search-mirror';
import {getMatchedEndpointNode} from './map-matching-operation-fixture';

it('matches the two-fix chain with exhausted, complete and absent tables without breaks', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 8),
    neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
    weights: new Float32Array(8).fill(1),
    nodePositions: Float32Array.from({length: 18}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
  };
  const costLimit = Float32Array.of(100);
  const exhausted = computeNetworkCostTableOnCPU(
    {topology, costLimit},
    {capacity: 81, maximumIterations: 1, localIterations: 1}
  );
  const complete = computeNetworkCostTableOnCPU(
    {topology, costLimit},
    {capacity: 81, maximumIterations: 16, localIterations: 1}
  );
  expect(exhausted.requiredCount[0]).toBe(17);
  expect(exhausted.complete[0]).toBe(0);
  expect(exhausted.targets.slice(exhausted.offsets[1], exhausted.offsets[2])).toEqual(
    Uint32Array.of(1, 2)
  );
  expect(complete.requiredCount[0]).toBe(45);
  expect(complete.complete[0]).toBe(1);
  for (const table of [undefined, exhausted, complete]) {
    const result = computeMapMatchingOnCPU(
      {
        topology,
        positions: Float32Array.of(0.5, 0, 3.5, 0),
        trackOffsets: Uint32Array.of(0, 2),
        parameters: getGPUMapMatchingParameterValues({
          sigma: 0.1,
          beta: 1,
          searchRadius: 0.1,
          routeFactor: 3,
          routeSlack: 0
        }),
        routeTable: table ? {...table, costLimit} : undefined
      },
      {candidateCount: 1, cellSize: 1}
    );
    expect(result.matchedEdges).toEqual(Uint32Array.of(0, 3));
    expect(result.breaks).toEqual(Uint32Array.of(0, 0));
    expect(result.routeOverflow[0]).toBe(0);
    expect(Number.isFinite(result.trackLogLikelihoods[0])).toBe(true);
    expect(result.trackLogLikelihoods[0]).toBeCloseTo(
      -2 * Math.log(0.1 * Math.sqrt(2 * Math.PI)),
      5
    );
  }
});

it('a twelve-node chain distinguishes a complete table from the eight-node A* fallback', () => {
  const result = searchRouteMirror(
    {
      offsets: Array.from({length: 13}, (_, node) => Math.min(node, 11)),
      neighbors: Array.from({length: 11}, (_, node) => node + 1),
      positions: Array.from({length: 24}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
    },
    1,
    40,
    [10],
    8,
    'evict-worst'
  );
  expect(result.targetDistances).toEqual([-1]);
  expect(result.droppedCount).toBeGreaterThan(0);
});

it('distinguishes equivalent junction representations from interior or different-junction matches', () => {
  const topology = {sources: Uint32Array.of(0, 1, 1), neighbors: Uint32Array.of(1, 2, 3)};
  expect(getMatchedEndpointNode(0, 1, topology)).toBe(1);
  expect(getMatchedEndpointNode(1, 0, topology)).toBe(1);
  expect(getMatchedEndpointNode(1, 0.5, topology)).toBeUndefined();
  expect(getMatchedEndpointNode(2, 1, topology)).toBe(3);
  expect(getMatchedEndpointNode(0xffffffff, 0, topology)).toBeUndefined();
});

it('isolates cutoff rounding with binary-exact axis lengths and endpoint fractions', () => {
  const halfUlp = 2 ** -25;
  // At 0.5, half an f32 ULP rounds back to the even significand. A float64 table omits this
  // target. Powers of two keep geometric edge measurement and endpoint projection exact too.
  expect(0.5 + halfUlp).toBeGreaterThan(0.5);
  expect(Math.fround(0.5 + halfUlp)).toBe(0.5);
  const nodes = Float32Array.of(-0.5, 0, 0, 0, 0.5, halfUlp, 0.5, 0, 1, halfUlp);
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 4),
    neighbors: Uint32Array.of(1, 3, 4, 2),
    weights: Float32Array.of(0.5, 0.5, 0.5, halfUlp)
  };
  const sources = [0, 1, 2, 3];
  topology.neighbors.forEach((target, edge) => {
    const source = sources[edge];
    const horizontal = Math.fround(nodes[target * 2] - nodes[source * 2]);
    const vertical = Math.fround(nodes[target * 2 + 1] - nodes[source * 2 + 1]);
    const length = Math.fround(
      Math.sqrt(
        Math.fround(Math.fround(horizontal * horizontal) + Math.fround(vertical * vertical))
      )
    );
    expect(length).toBe(topology.weights[edge]);
  });
  const result = computeNetworkCostTableOnCPU(
    {topology, costLimit: Float32Array.of(0.5)},
    {capacity: 25}
  );
  const row = Array.from(result.targets.slice(result.offsets[1], result.offsets[2]));
  expect(row).toContain(2);
  expect(result.costs[result.offsets[1] + row.indexOf(2)]).toBe(0.5);
  expect(result.complete[0]).toBe(1);
});

it.each([
  {nodeCount: 4, builtLimit: 0, requestedLimit: 10},
  {nodeCount: 5, builtLimit: 1, requestedLimit: 22}
])('retains the built cutoff when the caller changes the build input for a later match ($nodeCount nodes)', ({
  nodeCount,
  builtLimit,
  requestedLimit
}) => {
  const topology = {
    offsets: Uint32Array.from({length: nodeCount + 1}, (_, node) => Math.min(node, nodeCount - 1)),
    neighbors: Uint32Array.from({length: nodeCount - 1}, (_, edge) => edge + 1),
    weights: new Float32Array(nodeCount - 1).fill(1),
    nodePositions: Float32Array.from({length: nodeCount * 2}, (_, component) =>
      component % 2 === 0 ? component / 2 : 0
    )
  };
  const costLimit = Float32Array.of(builtLimit);
  const table = computeNetworkCostTableOnCPU(
    {topology, costLimit},
    {capacity: nodeCount * nodeCount}
  );
  costLimit[0] = requestedLimit;
  const fresh = computeNetworkCostTableOnCPU(
    {topology, costLimit},
    {capacity: nodeCount * nodeCount}
  );
  expect(table.builtCostLimit).toEqual(Float32Array.of(builtLimit));
  for (const retained of [table, fresh]) {
    const result = computeMapMatchingOnCPU(
      {
        topology,
        positions: Float32Array.of(0.5, 0, nodeCount - 1.5, 0),
        trackOffsets: Uint32Array.of(0, 2),
        parameters: getGPUMapMatchingParameterValues({
          sigma: 0.1,
          beta: 1,
          searchRadius: 0.1,
          routeFactor: 3,
          routeSlack: 0
        }),
        routeTable: {...retained, costLimit}
      },
      {candidateCount: 1, cellSize: 1}
    );
    // A straight, connected chain has no break, regardless of the preprocessing cutoff.
    expect(result.breaks).toEqual(Uint32Array.of(0, 0));
  }
});
