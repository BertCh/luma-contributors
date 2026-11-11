// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUNetworkServiceAreasOperation,
  computeNetworkServiceAreasOnCPU
} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {
  GPUNetworkServiceAreasOperation,
  type NetworkServiceAreasOperationProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas-operation';
import {
  enumerateRoutingCosts,
  getOverlappingRoutingTopology,
  getRoutingTopology
} from '../network-reachability/network-routing-operation-test-utils';

it('propagates facility labels through slot flags shared by overlapping CSR rows', () => {
  const result = computeNetworkServiceAreasOnCPU({
    topology: getOverlappingRoutingTopology(),
    facilities: Uint32Array.of(0, 2),
    facilityCosts: Float32Array.of(1, 0)
  });
  // Node 1 costs 0 via facility 1. Row 2 marks slot 0 tight, then row 0 reuses
  // the flag to propagate facility 0. Its group contains costs 1 + 0, the other 0.
  expect(Array.from(result.costs)).toEqual([1, 0, 0]);
  expect(Array.from(result.assignments)).toEqual([0, 0, 1]);
  expect(Array.from(result.facilityNodeCounts)).toEqual([2, 1]);
  expect(Array.from(result.facilityCostSums)).toEqual([1, 0]);
  // Neither incoming edge to node 1 is both tight and within its final label.
  expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  expect(Array.from(result.converged)).toEqual([1]);
});

it('matches independent per-facility exhaustive paths and hand-computed same-label parents', () => {
  const topology = getRoutingTopology();
  const facilities = Uint32Array.of(7, 0, 4, 99);
  const facilityCosts = Float32Array.of(2, 0.5, 1.75, 0);
  const result = computeNetworkServiceAreasOnCPU({
    topology,
    facilities,
    facilityCosts,
    facilityCount: Uint32Array.of(3)
  });
  const individual = Array.from(facilities.subarray(0, 3), (node, facility) =>
    enumerateRoutingCosts(topology, [{node, cost: facilityCosts[facility]}])
  );
  const expectedCosts = Float32Array.from({length: 9}, (_, node) =>
    Math.min(...individual.map(costs => costs[node]))
  );
  const assignments = Uint32Array.from(expectedCosts, (cost, node) =>
    Number.isFinite(cost) ? individual.findIndex(costs => costs[node] === cost) : 0xffffffff
  );
  expect(result.costs).toEqual(expectedCosts);
  expect(result.assignments).toEqual(assignments);
  // Facility 0 wins the positive-cost tie at node 3. Facility 1 wins facility 2's own node
  // through the zero-cost cycle, which must have cycle-safe parents leading to node 0.
  expect(Array.from(result.assignments)).toEqual([1, 1, 1, 0, 1, 0, 0, 0, 0xffffffff]);
  expect(Array.from(result.predecessors)).toEqual([
    0xffffffff, 2, 0, 7, 2, 3, 7, 0xffffffff, 0xffffffff
  ]);
  expect(Array.from(result.facilityNodeCounts)).toEqual([4, 4, 0, 0]);
  expect(Array.from(result.facilityCostSums)).toEqual([14.25, 8, 0, 0]);
  expect(Array.from(result.converged)).toEqual([1]);
});

it('checks inactive facilities, malformed arrays and construction-time shapes', () => {
  const topology = getRoutingTopology();
  const facilities = Uint32Array.of(0, 7);
  const empty = computeNetworkServiceAreasOnCPU({topology, facilities, facilityCount: 0});
  expect(empty.assignments.every(value => value === 0xffffffff)).toBe(true);
  expect(empty.facilityNodeCounts).toEqual(new Uint32Array(2));
  expect(empty.facilityCostSums).toEqual(new Float32Array(2));
  expect(() =>
    computeNetworkServiceAreasOnCPU({topology, facilities, facilityCosts: Float32Array.of(1)})
  ).toThrow(/facilityCosts/);
  expect(() =>
    computeNetworkServiceAreasOnCPU({topology, facilities, facilityCount: new Uint32Array(2)})
  ).toThrow(/facilityCount/);
  const props: NetworkServiceAreasOperationProps = {
    id: 'check-service',
    nodeCount: 9,
    facilityCapacity: 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    facilities: 'facilities',
    output: {assignments: 'assignments'}
  };
  for (const Operation of [CPUNetworkServiceAreasOperation, GPUNetworkServiceAreasOperation])
    expect(() => new Operation({...props, nodeCount: 0})).toThrow(/^check-service/);
  expect(() => new CPUNetworkServiceAreasOperation({...props, facilityCount: 3})).toThrow(
    /^check-service.*facilityCount/
  );
});
