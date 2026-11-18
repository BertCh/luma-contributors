// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUNetworkReachabilityOperation,
  computeNetworkReachabilityOnCPU
} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {
  GPUNetworkReachabilityOperation,
  type NetworkReachabilityOperationProps
} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {
  enumerateRoutingCosts,
  getOverlappingRoutingTopology,
  getRoutingTopology
} from './network-routing-operation-test-utils';

const NONE = 0xffffffff;
it.each([
  false,
  true
])('network:G7 CPU reachability snapshots budgets and keeps activeIterations live (inherited=%s)', async inherited => {
  let maximumIterations = 1;
  const props: NetworkReachabilityOperationProps = Object.assign(
    inherited
      ? Object.create({
          get maximumIterations() {
            return maximumIterations;
          }
        })
      : {},
    {
      nodeCount: 3,
      localIterations: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      startNodes: 'starts',
      activeIterations: 'active',
      output: {costs: 'costs', status: {converged: 'complete'}}
    }
  );
  if (!inherited) props.maximumIterations = maximumIterations;
  const operation = new CPUNetworkReachabilityOperation(props);
  maximumIterations = 3;
  if (!inherited) props.maximumIterations = maximumIterations;
  const active = Uint32Array.of(1);
  const values = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(1, 1),
    startNodes: Uint32Array.of(0),
    activeIterations: active
  };
  const inputs = Object.fromEntries(
    Object.entries(values).map(([port, rows]) => [
      port,
      {
        format: port === 'weights' ? ('float32' as const) : ('uint32' as const),
        length: rows.length,
        chunks: [{values: rows, length: rows.length}],
        pack: () => rows
      }
    ])
  );
  active[0] = 3;
  const result = await operation.execute({inputs});
  expect(result['costs']).toEqual(Float32Array.of(0, 1, Infinity));
  expect(result['complete']).toEqual(Uint32Array.of(0));
  active[0] = 0;
  expect((await operation.execute({inputs}))['costs']).toEqual(
    Float32Array.of(0, Infinity, Infinity)
  );
});

it('reuses slot-indexed tie flags across overlapping CSR rows', () => {
  const result = computeNetworkReachabilityOnCPU({
    topology: getOverlappingRoutingTopology(),
    startNodes: Uint32Array.of(0, 2),
    startCosts: Float32Array.of(1, 0)
  });
  // Row 2 lowers node 1 to 0 and marks slot 0 as a tie. Both roots have level 0;
  // row 0 reuses slot 0, so the lowest level-0 predecessor is 0 despite its cost of 1.
  expect(Array.from(result.costs)).toEqual([1, 0, 0]);
  expect(Array.from(result.predecessors)).toEqual([NONE, 0, NONE]);
  expect(Array.from(result.unresolvedCount)).toEqual([0]);
  expect(Array.from(result.converged)).toEqual([1]);
});

it('publishes one CPU iteration and zero relaxation words for the empty comparison fixture', () => {
  const result = computeNetworkReachabilityOnCPU({
    topology: getRoutingTopology(),
    startNodes: Uint32Array.of(99, 7, 1),
    startCosts: Float32Array.of(0.5, 2, 0),
    startCount: Uint32Array.of(0),
    costLimit: Float32Array.of(6),
    costBreaks: Float32Array.of(1, 2, 3, 6)
  });
  expect(Array.from(result.iterationCount)).toEqual([1]);
  expect(Array.from(result.relaxationCount)).toEqual([0, 0]);
});

it('matches exhaustive simple paths, hand-computed tie parents and inclusive bands', () => {
  const topology = getRoutingTopology();
  const result = computeNetworkReachabilityOnCPU({
    topology,
    startNodes: Uint32Array.of(0, 7, 1),
    startCosts: Float32Array.of(0.5, 2, 0),
    startCount: Uint32Array.of(2),
    costLimit: Float32Array.of(6),
    costBreaks: Float32Array.of(1, 2, 3, 6)
  });
  expect(result.costs).toEqual(
    enumerateRoutingCosts(
      topology,
      [
        {node: 0, cost: 0.5},
        {node: 7, cost: 2}
      ],
      6
    )
  );
  expect(Array.from(result.costs)).toEqual([0.5, 4, 1.75, 4.5, 1.75, 5.25, 2.5, 2, Infinity]);
  expect(Array.from(result.predecessors)).toEqual([NONE, 2, 0, 1, 2, 3, 7, NONE, NONE]);
  expect(Array.from(result.bands)).toEqual([0, 3, 1, 3, 1, 3, 2, 1, NONE]);
  expect(Array.from(result.bandCounts)).toEqual([1, 3, 1, 3]);
  expect(Array.from(result.reachedEdgeCount)).toEqual([13, 0]);
  expect(Array.from(result.relaxationCount)).toEqual([13, 0]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  expect(Array.from(result.unresolvedCount)).toEqual([0]);
});

it('finishes uncapped tie plateaus and keeps separate lanes and one-round output arithmetic', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 4),
    neighbors: Uint32Array.of(1, 2, 3, 4),
    weights: Float32Array.of(16777216, 1, 1, 0)
  };
  const result = computeNetworkReachabilityOnCPU({
    topology,
    startNodes: Uint32Array.of(0)
  });
  expect(Array.from(result.costs)).toEqual([0, 16777216, 16777216, 16777218, 16777218]);
  expect(Array.from(result.predecessors)).toEqual([NONE, 0, 1, 2, 3]);
  const lanes = computeNetworkReachabilityOnCPU(
    {
      topology: getRoutingTopology(),
      startNodes: Uint32Array.of(0, 16),
      startCosts: Float32Array.of(0.5, 2)
    },
    {laneCount: 2}
  );
  expect(lanes.costs.subarray(0, 9)).toEqual(
    enumerateRoutingCosts(getRoutingTopology(), [{node: 0, cost: 0.5}])
  );
  expect(lanes.costs.subarray(9)).toEqual(
    enumerateRoutingCosts(getRoutingTopology(), [{node: 7, cost: 2}])
  );
});

it('preserves defined invalid-row and clamped-offset behavior and rejects inconsistent arrays', () => {
  const topology = getRoutingTopology();
  expect(
    computeNetworkReachabilityOnCPU({
      topology,
      startNodes: Uint32Array.of(99),
      costBreaks: Float32Array.of(1)
    }).costs.every(value => value === Infinity)
  ).toBe(true);
  expect(() =>
    computeNetworkReachabilityOnCPU({
      topology,
      startNodes: Uint32Array.of(0),
      startCosts: new Float32Array(0)
    })
  ).toThrow(/startCosts/);
  expect(() =>
    computeNetworkReachabilityOnCPU({
      topology: {...topology, weights: new Float32Array(1)},
      startNodes: Uint32Array.of(0)
    })
  ).toThrow(/weights/);
  expect(() =>
    computeNetworkReachabilityOnCPU({
      topology,
      startNodes: Uint32Array.of(0),
      costLimit: new Float32Array(2)
    })
  ).toThrow(/costLimit/);
  const clamped = computeNetworkReachabilityOnCPU({
    topology: {
      offsets: Uint32Array.of(8, 0, 1),
      neighbors: Uint32Array.of(0),
      weights: Float32Array.of(1)
    },
    startNodes: Uint32Array.of(0)
  });
  expect(Array.from(clamped.costs)).toEqual([0, Infinity]);
});

it('ignores impassable weights and bad targets while counting their defined scan work', () => {
  const result = computeNetworkReachabilityOnCPU({
    topology: {
      offsets: Uint32Array.of(0, 5, 5, 5),
      neighbors: Uint32Array.of(1, 1, 1, 2, 99),
      weights: Float32Array.of(-1, NaN, Infinity, 2, 0)
    },
    startNodes: Uint32Array.of(0, 1, 2, 1),
    startCosts: Float32Array.of(-0, NaN, -1, Infinity),
    startCount: Uint32Array.of(99),
    costLimit: Float32Array.of(NaN)
  });
  expect(Array.from(result.costs)).toEqual([0, Infinity, 2]);
  expect(Object.is(result.costs[0], -0)).toBe(false);
  expect(Array.from(result.predecessors)).toEqual([NONE, NONE, 0]);
  expect(Array.from(result.relaxationCount)).toEqual([5, 0]);
  expect(Array.from(result.reachedEdgeCount)).toEqual([4, 0]);
});

it('validates CPU props and GPU output shapes at construction', () => {
  const props: NetworkReachabilityOperationProps = {
    id: 'check-reach',
    nodeCount: 9,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {costs: 'costs'}
  };
  expect(
    new GPUNetworkReachabilityOperation({
      ...props,
      topology: {...props.topology, nodePositions: undefined}
    }).inputs
  ).not.toHaveProperty('nodePositions');
  for (const Operation of [CPUNetworkReachabilityOperation, GPUNetworkReachabilityOperation])
    expect(() => new Operation({...props, nodeCount: 0})).toThrow(/^check-reach/);
  expect(
    () =>
      new CPUNetworkReachabilityOperation({
        ...props,
        laneCount: 2,
        output: {...props.output, predecessors: 'parents'}
      })
  ).toThrow(/^check-reach.*laneCount/);
  expect(
    () => new CPUNetworkReachabilityOperation({...props, output: {...props.output, bands: 'bands'}})
  ).toThrow(/^check-reach.*costBreaks/);
});
