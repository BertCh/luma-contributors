// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUNetworkCostTableOperation,
  computeNetworkCostTableOnCPU
} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {
  GPUNetworkCostTableOperation,
  type NetworkCostTableOperationProps
} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {
  enumerateRoutingCosts,
  getRoutingTopology
} from '../network-reachability/network-routing-operation-test-utils';

for (const capacity of [3, 81])
  it(`matches exhaustive paths and clamped hand-assembled CSR at capacity ${capacity}`, () => {
    const topology = getRoutingTopology();
    const cutoff = 5;
    const expectedTargets: number[] = [];
    const expectedCosts: number[] = [];
    const expectedOffsets = [0];
    for (let source = 0; source < 9; source++) {
      const distances = enumerateRoutingCosts(topology, [{node: source, cost: 0}], cutoff);
      for (let target = 0; target < 9; target++)
        if (Number.isFinite(distances[target])) {
          expectedTargets.push(target);
          expectedCosts.push(distances[target]);
        }
      expectedOffsets.push(Math.min(expectedTargets.length, capacity));
    }
    const result = computeNetworkCostTableOnCPU(
      {topology, costLimit: Float32Array.of(cutoff)},
      {capacity}
    );
    expect(Array.from(result.offsets)).toEqual(expectedOffsets);
    expect(
      Array.from(result.targets.subarray(0, Math.min(capacity, expectedTargets.length)))
    ).toEqual(expectedTargets.slice(0, capacity));
    expect(Array.from(result.costs.subarray(0, Math.min(capacity, expectedCosts.length)))).toEqual(
      expectedCosts.slice(0, capacity)
    );
    expect(Array.from(result.requiredCount)).toEqual([expectedTargets.length]);
    expect(Array.from(result.overflow)).toEqual([Number(expectedTargets.length > capacity)]);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(result).toHaveProperty(
      'complete',
      Uint32Array.of(Number(expectedTargets.length <= capacity))
    );
    expect(result.targets.subarray(expectedTargets.length).every(value => value === 0)).toBe(true);
  });

it('honors a negative cutoff and checks arrays and construction-time capacity', () => {
  const topology = getRoutingTopology();
  const empty = computeNetworkCostTableOnCPU(
    {topology, costLimit: Float32Array.of(-1)},
    {capacity: 3}
  );
  expect(Array.from(empty.offsets)).toEqual(Array(10).fill(0));
  expect(empty.requiredCount[0]).toBe(0);
  expect(() =>
    computeNetworkCostTableOnCPU({topology, costLimit: new Float32Array(0)}, {capacity: 3})
  ).toThrow(/costLimit/);
  const props: NetworkCostTableOperationProps = {
    id: 'check-table',
    nodeCount: 9,
    capacity: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    costLimit: 'cutoff',
    output: {offsets: 'outputOffsets', targets: 'targets', costs: 'costs'}
  };
  for (const Operation of [CPUNetworkCostTableOperation, GPUNetworkCostTableOperation])
    expect(() => new Operation({...props, capacity: 0})).toThrow(/^check-table/);
});

it('reports no GPU round exhaustion from the exact CPU cost table, including overflow', () => {
  const result = computeNetworkCostTableOnCPU(
    {topology: getRoutingTopology(), costLimit: Float32Array.of(Infinity)},
    {capacity: 1}
  );
  expect(result.overflow[0]).toBe(1);
  expect(result).toHaveProperty('iterationLimitReached', Uint32Array.of(0));
});

it('an explicit one-round budget publishes the chain incumbents as incomplete', () => {
  const result = computeNetworkCostTableOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 8),
        neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
        weights: new Float32Array(8).fill(1)
      },
      costLimit: Float32Array.of(Infinity)
    },
    {capacity: 81, maximumIterations: 1, localIterations: 1}
  );
  expect(result.requiredCount[0]).toBe(17);
  expect(result.offsets).toEqual(Uint32Array.of(0, 2, 4, 6, 8, 10, 12, 14, 16, 17));
  expect(result.converged[0]).toBe(0);
  expect(result.iterationLimitReached[0]).toBe(1);
  expect(result.complete[0]).toBe(0);
  expect(result.overflow[0]).toBe(0);
});
