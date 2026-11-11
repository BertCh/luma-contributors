// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUNetworkCostMatrixOperation,
  computeNetworkCostMatrixOnCPU
} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {
  GPUNetworkCostMatrixOperation,
  type NetworkCostMatrixOperationProps
} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {
  enumerateRoutingCosts,
  getRoutingTopology
} from '../network-reachability/network-routing-operation-test-utils';

it('matches exhaustive simple paths for multiple weighted multi-start rows and duplicate targets', () => {
  const topology = getRoutingTopology();
  const targetNodes = Uint32Array.of(5, 1, 2, 8, 5, 99);
  const result = computeNetworkCostMatrixOnCPU(
    {
      topology,
      startNodes: Uint32Array.of(0, 7, 4, 2, 99),
      startCosts: Float32Array.of(0.5, 2, 3, 0.75, 0),
      startRows: Uint32Array.of(0, 0, 1, 1, 2),
      targetNodes
    },
    {rowCount: 3}
  );
  const expected = [
    enumerateRoutingCosts(topology, [
      {node: 0, cost: 0.5},
      {node: 7, cost: 2}
    ]),
    enumerateRoutingCosts(topology, [
      {node: 4, cost: 3},
      {node: 2, cost: 0.75}
    ]),
    new Float32Array(9).fill(Infinity)
  ];
  expect(Array.from(result.costs)).toEqual(
    expected.flatMap(costs =>
      Array.from(targetNodes, target => (target < costs.length ? costs[target] : Infinity))
    )
  );
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  const grouped = computeNetworkCostMatrixOnCPU(
    {
      topology,
      startNodes: Uint32Array.of(0, 7, 4, 2),
      startCosts: Float32Array.of(0.5, 2, 3, 0.75)
    },
    {rowCount: 2, startsPerRow: 2}
  );
  expect(Array.from(grouped.costs)).toEqual([...expected[0], ...expected[1]]);
});

it('checks lengths and construction-time output shapes', () => {
  const topology = getRoutingTopology();
  expect(() =>
    computeNetworkCostMatrixOnCPU(
      {topology: {...topology, offsets: Uint32Array.of(0)}, startNodes: Uint32Array.of(0)},
      {rowCount: 2}
    )
  ).toThrow(/topology.offsets/);
  expect(() =>
    computeNetworkCostMatrixOnCPU(
      {topology, startNodes: Uint32Array.of(0), startRows: new Uint32Array(0)},
      {rowCount: 2}
    )
  ).toThrow(/startRows/);
  expect(() =>
    computeNetworkCostMatrixOnCPU(
      {topology, startNodes: Uint32Array.of(0), targetNodes: new Uint32Array(0)},
      {rowCount: 2}
    )
  ).toThrow(/targetNodes/);
  const props: NetworkCostMatrixOperationProps = {
    id: 'check-matrix',
    nodeCount: 9,
    rowCount: 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {costs: 'costs'}
  };
  for (const Operation of [CPUNetworkCostMatrixOperation, GPUNetworkCostMatrixOperation])
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/^check-matrix/);
  expect(() => new CPUNetworkCostMatrixOperation({...props, startsPerRow: 0})).toThrow(
    /^check-matrix.*startsPerRow/
  );
});
