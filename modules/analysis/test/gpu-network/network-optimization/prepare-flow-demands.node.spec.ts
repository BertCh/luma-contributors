// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';
import {computeNetworkFlowOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';

it('selects reference at 126 input vertices and parallel at 127 after adding super vertices', () => {
  for (const nodeCount of [126, 127]) {
    const demands = new Int32Array(nodeCount);
    demands[0] = -1;
    demands[1] = 1;
    const result = prepareGPUNetworkFlowDemands({
      nodeCount,
      demands,
      edges: [[0, 1]],
      capacities: [1]
    });
    expect(result.nodeCount).toBe(nodeCount + 2);
    expect(result.solver).toBe(nodeCount === 126 ? 'reference' : 'parallel');
    expect(computeNetworkFlowOnCPU(result, result).summary[2]).toBe(1);
  }
});

it('selects parallel for expanded capacity totals above the reference integral bound', () => {
  const result = prepareGPUNetworkFlowDemands({
    nodeCount: 2,
    edges: [[0, 1]],
    capacities: [0x1000000],
    demands: [-1, 1]
  });
  expect(result.solver).toBe('parallel');
  expect(result.capacities).toEqual(Uint32Array.of(0x1000000, 1, 1));
});

it('selects against expanded edge counts and retains an explicit parallel option', () => {
  for (const edgeCount of [1022, 1023]) {
    const result = prepareGPUNetworkFlowDemands({
      nodeCount: 2,
      edges: Array.from({length: edgeCount}, () => [0, 1] as const),
      capacities: new Uint32Array(edgeCount),
      demands: [-1, 1]
    });
    expect(result.capacities.length).toBe(edgeCount + 2);
    expect(result.solver).toBe(edgeCount === 1022 ? 'reference' : 'parallel');
  }
  expect(
    prepareGPUNetworkFlowDemands({
      nodeCount: 2,
      edges: [],
      capacities: [],
      demands: [0, 0],
      solver: 'parallel'
    }).solver
  ).toBe('parallel');
});

it('retains explicit reference limits on the expanded shape and rejects unrepresentable inputs', () => {
  expect(() =>
    prepareGPUNetworkFlowDemands({
      nodeCount: 127,
      edges: [],
      capacities: [],
      demands: new Int32Array(127),
      solver: 'reference'
    })
  ).toThrow(/nodeCount.*128/);
  expect(() =>
    prepareGPUNetworkFlowDemands({
      nodeCount: 2,
      edges: [[0, 1]],
      capacities: [0x1000000],
      demands: [-1, 1],
      solver: 'reference'
    })
  ).toThrow(/total capacity/);
  expect(() =>
    prepareGPUNetworkFlowDemands({
      nodeCount: 2,
      edges: [[0, 1]],
      capacities: [0xffffffff],
      demands: [-1, 1]
    })
  ).toThrow(/total capacity/);
  expect(() =>
    prepareGPUNetworkFlowDemands({
      nodeCount: 100000,
      edges: [[0, 1]],
      capacities: [1],
      demands: new Int32Array(100000),
      costs: [32767]
    })
  ).toThrow(/costs/);
});

it('rejects short, sparse and missing edge rows before packing aligned capacities', () => {
  for (const edges of [[[1]], [[0, ,]], [, [0, 1]]]) {
    expect(() =>
      prepareGPUNetworkFlowDemands({
        nodeCount: 2,
        edges: edges as unknown as [number, number][],
        capacities: edges.map(() => 8),
        demands: [0, 0]
      })
    ).toThrow(/edges\[0\].*\[source, target\]/);
  }
  const result = prepareGPUNetworkFlowDemands({
    nodeCount: 2,
    edges: [
      [1, 1],
      [0, 1]
    ],
    capacities: [3, 8],
    costs: [2, 4],
    demands: [0, 0]
  });
  expect(Array.from(result.edges)).toEqual([1, 1, 0, 1]);
  expect(Array.from(result.capacities)).toEqual([3, 8]);
  expect(Array.from(result.costs!)).toEqual([2, 4]);
});
