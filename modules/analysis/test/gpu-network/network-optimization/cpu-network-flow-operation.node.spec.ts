// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkFlowOnCPU,
  CPUNetworkFlowOperation,
  type CPUNetworkFlowInput
} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPU_NETWORK_FLOW_WORD} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: NetworkFlowOperationProps = {
  id: 'flow-test',
  nodeCount: 4,
  edgeCount: 4,
  edges: 'edges',
  capacities: 'capacities',
  costs: 'costs',
  parameters: 'parameters',
  output: {flow: 'flow', cut: 'cut', summary: 'summary', exactCost: 'exact'}
};

function enumerateFeasibleFlows(
  input: CPUNetworkFlowInput,
  nodeCount: number
): {value: number; cost: number} {
  const [source, sink, requested] = input.parameters;
  const bound = requested === 0xffffffff ? Infinity : requested;
  const balance = new Int32Array(nodeCount);
  let bestValue = -1;
  let bestCost = Infinity;
  const enumerateEdges = (edge: number, objective: number) => {
    if (edge === input.capacities.length) {
      const value = 0 - balance[source];
      if (
        value < 0 ||
        value > bound ||
        balance[sink] !== value ||
        balance.some((net, vertex) => vertex !== source && vertex !== sink && net !== 0)
      )
        return;
      if (value > bestValue || (value === bestValue && objective < bestCost)) {
        bestValue = value;
        bestCost = objective;
      }
      return;
    }
    const from = input.edges[edge * 2];
    const to = input.edges[edge * 2 + 1];
    for (let amount = 0; amount <= input.capacities[edge]; amount++) {
      balance[from] -= amount;
      balance[to] += amount;
      enumerateEdges(edge + 1, objective + amount * (input.costs?.[edge] ?? 0));
      balance[from] += amount;
      balance[to] -= amount;
    }
  };
  enumerateEdges(0, 0);
  return {value: bestValue, cost: bestCost};
}

it('matches exhaustive feasible-flow enumeration for parallel, antiparallel and loop edges', () => {
  for (const requested of [0, 1, 2, 4, 0xffffffff])
    for (const withCosts of [false, true]) {
      const input: CPUNetworkFlowInput = {
        edges: Uint32Array.of(0, 1, 0, 1, 1, 2, 2, 1, 1, 3, 2, 3, 0, 0),
        capacities: Uint32Array.of(1, 2, 2, 1, 1, 2, 1),
        costs: withCosts ? Uint32Array.of(3, 1, 0, 1, 4, 2, 1) : undefined,
        parameters: Uint32Array.of(0, 3, requested)
      };
      const expected = enumerateFeasibleFlows(input, 4);
      const result = computeNetworkFlowOnCPU(input, {nodeCount: 4});
      expect(result.summary[GPU_NETWORK_FLOW_WORD.value]).toBe(expected.value);
      expect(BigInt(result.exactCost[0]) + (BigInt(result.exactCost[1]) << 32n)).toBe(
        BigInt(expected.cost)
      );
      expect(result.summary[GPU_NETWORK_FLOW_WORD.unserved]).toBe(
        requested === 0xffffffff ? 0 : requested - expected.value
      );
      expect(result.summary[GPU_NETWORK_FLOW_WORD.complete]).toBe(1);
      expect(result.summary[GPU_NETWORK_FLOW_WORD.valid]).toBe(1);
      const balance = new Int32Array(4);
      for (let edge = 0; edge < input.capacities.length; edge++) {
        expect(result.flow[edge]).toBeLessThanOrEqual(input.capacities[edge]);
        balance[input.edges[edge * 2]] -= result.flow[edge];
        balance[input.edges[edge * 2 + 1]] += result.flow[edge];
      }
      expect(Array.from(balance)).toEqual([
        expected.value === 0 ? 0 : -expected.value,
        0,
        0,
        expected.value
      ]);
      if (requested === 0xffffffff) {
        // Independent min-cut enumeration over the two interior vertices.
        let minimumCut = Infinity;
        for (let mask = 0; mask < 4; mask++) {
          const side = [true, Boolean(mask & 1), Boolean(mask & 2), false];
          let capacity = 0;
          for (let edge = 0; edge < input.capacities.length; edge++)
            if (side[input.edges[edge * 2]] && !side[input.edges[edge * 2 + 1]])
              capacity += input.capacities[edge];
          minimumCut = Math.min(minimumCut, capacity);
        }
        expect(result.summary[GPU_NETWORK_FLOW_WORD.cutCapacity]).toBe(minimumCut);
        expect(result.summary[GPU_NETWORK_FLOW_WORD.cutCertified]).toBe(1);
      }
    }
});

it('uses reverse residual arcs to undo a locally cheap assignment', () => {
  const input = {
    edges: Uint32Array.of(0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5),
    capacities: new Uint32Array(8).fill(1),
    costs: Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0),
    parameters: Uint32Array.of(0, 5, 0xffffffff)
  };
  const result = computeNetworkFlowOnCPU(input, {nodeCount: 6});
  // Match A->Y and B->X for cost 4; the first augmentation had chosen A->X.
  expect(Array.from(result.flow)).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
  expect(Array.from(result.exactCost)).toEqual([4, 0]);
  expect(result.summary[GPU_NETWORK_FLOW_WORD.augmentations]).toBe(2);
  expect(enumerateFeasibleFlows(input, 6)).toEqual({value: 2, cost: 4});
});

it('keeps the two-word objective exact and rounds it directly to float32 only once', () => {
  const result = computeNetworkFlowOnCPU(
    {
      edges: Uint32Array.of(0, 1, 0, 1, 0, 1),
      capacities: Uint32Array.of(2 ** 31, 128, 1),
      costs: Uint32Array.of(0xffffffff, 0xffffffff, 2 ** 31 + 129),
      parameters: Uint32Array.of(0, 1, 0xffffffff)
    },
    {nodeCount: 2}
  );
  const exact = 2n ** 63n + 2n ** 39n + 1n;
  expect(BigInt(result.exactCost[0]) + (BigInt(result.exactCost[1]) << 32n)).toBe(exact);
  // One above the midpoint rounds up; conversion via float64 would lose the +1 and round down.
  expect(result.summary[GPU_NETWORK_FLOW_WORD.cost]).toBe(0x5f000001);
  expect(result.summary[GPU_NETWORK_FLOW_WORD.value]).toBe(2 ** 31 + 129);
  expect(result.summary[GPU_NETWORK_FLOW_WORD.cutCertified]).toBe(1);
});

it('returns completion with unserved demand, accepts no edges, and defines invalid input', () => {
  const empty = computeNetworkFlowOnCPU(
    {
      edges: new Uint32Array(0),
      capacities: new Uint32Array(0),
      parameters: Uint32Array.of(0, 1, 7)
    },
    {nodeCount: 2}
  );
  expect(Array.from(empty.flow)).toEqual([]);
  expect(Array.from(empty.cut)).toEqual([1, 0]);
  expect(Array.from(empty.summary)).toEqual([1, 1, 0, 7, 0, 0, 0, 1]);
  for (const input of [
    {
      edges: Uint32Array.of(0, 2),
      capacities: Uint32Array.of(1),
      parameters: Uint32Array.of(0, 1, 1)
    },
    {
      edges: Uint32Array.of(0, 1),
      capacities: Uint32Array.of(1),
      parameters: Uint32Array.of(0, 0, 1)
    },
    {
      edges: Uint32Array.of(0, 1, 0, 1),
      capacities: Uint32Array.of(0xffffffff, 1),
      parameters: Uint32Array.of(0, 1, 1)
    }
  ]) {
    const result = computeNetworkFlowOnCPU(input, {nodeCount: 2});
    expect(result.summary.every(word => word === 0)).toBe(true);
    expect(result.flow.every(word => word === 0)).toBe(true);
    expect(result.cut.every(word => word === 0)).toBe(true);
    expect(Array.from(result.exactCost)).toEqual([0, 0]);
  }
});

it('checks inconsistent arrays and both operation forms at construction', () => {
  const input = {
    edges: Uint32Array.of(0, 1),
    capacities: Uint32Array.of(1),
    parameters: Uint32Array.of(0, 1, 1)
  };
  expect(() =>
    computeNetworkFlowOnCPU({...input, edges: new Uint32Array(1)}, {nodeCount: 2})
  ).toThrow(/edges/);
  expect(() =>
    computeNetworkFlowOnCPU({...input, costs: new Uint32Array(0)}, {nodeCount: 2})
  ).toThrow(/costs/);
  expect(() =>
    computeNetworkFlowOnCPU({...input, parameters: new Uint32Array(2)}, {nodeCount: 2})
  ).toThrow(/parameters/);
  const shapes = {
    edges: {format: 'uint32x2' as const, length: 4},
    capacities: {format: 'uint32' as const, length: 4},
    costs: {format: 'uint32' as const, length: 4},
    parameters: {format: 'uint32' as const, length: 3}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkFlowOperation(PROPS), shapes)
  ).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(
      new GPUNetworkFlowOperation({...PROPS, maximumAugmentations: -1}),
      shapes
    )
  ).toThrow(/flow-test.*maximumAugmentations/);
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkFlowOperation(PROPS), {
      ...shapes,
      capacities: {...shapes.capacities, length: 3}
    })
  ).toThrow(/flow-test.*capacities/);
  expect(() => new CPUNetworkFlowOperation({...PROPS, nodeCount: 1})).toThrow(
    /flow-test.*nodeCount/
  );
  expect(
    () =>
      new CPUNetworkFlowOperation({
        ...PROPS,
        nodeCount: 129,
        edgeCount: 1025,
        maximumAugmentations: 0
      })
  ).not.toThrow();
  for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, flow: 'capacities'}})).toThrow(
      /flow-test/
    );
});
