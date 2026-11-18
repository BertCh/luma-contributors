// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeKShortestPathsOnCPU} from '../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';
import {
  computeNetworkFlowOnCPU,
  CPUNetworkFlowOperation
} from '../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {constructNetworkOperationContributor} from '../utils/network-operation-test-utils';
import {
  getSpurInterruptionInput,
  getCandidateSpurInterruptionInput
} from './k-shortest-paths/k-shortest-paths-test-utils';

it('network:H2 later spur interruption wins over an earlier overlong path in either query order', () => {
  for (const reverse of [false, true]) {
    const input = getSpurInterruptionInput();
    if (reverse) {
      input.sources.reverse();
      input.targets.reverse();
    }
    const result = computeKShortestPathsOnCPU(input, {
      routeCount: 2,
      maximumPathLength: 3,
      maximumIterations: 1
    });
    expect(result.counts).toEqual(Uint32Array.of(1, 1));
    expect(result.lengths).toEqual(Uint32Array.from(reverse ? [2, 0, 3, 0] : [3, 0, 2, 0]));
    expect(result.costs).toEqual(
      Float32Array.from(reverse ? [1, Infinity, 2, Infinity] : [2, Infinity, 1, Infinity])
    );
    expect(result.converged[0]).toBe(0);
    expect(result.iterationLimitReached[0]).toBe(1);
    expect(result.candidateOverflow[0]).toBe(0);
  }
});

const REFERENCE_OPTIONS = [
  {solver: 'auto' as const, maximumRelabels: 0},
  {solver: 'reference' as const, mode: 'primal-dual' as const, maximumCostPhases: 0}
];

it('network:H2 a candidate-capacity stop cannot hide a later spur interruption', () => {
  for (const reverse of [false, true]) {
    const input = getCandidateSpurInterruptionInput();
    if (reverse) {
      input.sources.reverse();
      input.targets.reverse();
    }
    const result = computeKShortestPathsOnCPU(input, {
      routeCount: 2,
      maximumPathLength: 4,
      maximumIterations: 1,
      candidateCapacity: 1
    });
    expect(result.counts).toEqual(Uint32Array.of(1, 1));
    expect(result.lengths).toEqual(Uint32Array.from(reverse ? [2, 0, 4, 0] : [4, 0, 2, 0]));
    expect(result.iterationLimitReached[0]).toBe(1);
    expect(result.candidateOverflow[0]).toBe(0);
  }
});
for (const options of REFERENCE_OPTIONS) {
  it(`network:H4 pure reference flow refuses parallel-only settings: ${options.solver}`, () => {
    expect(() =>
      computeNetworkFlowOnCPU(
        {
          edges: Uint32Array.of(0, 1),
          capacities: Uint32Array.of(1),
          costs: Uint32Array.of(1),
          parameters: Uint32Array.of(0, 1, 1)
        },
        {nodeCount: 2, ...options}
      )
    ).toThrow(
      /maximumRelaxationRounds, maximumRelabels, mode and maximumCostPhases require solver parallel/
    );
  });
  it(`network:H4 both operation forms refuse incompatible reference settings: ${options.solver}`, () => {
    const props: NetworkFlowOperationProps = {
      id: 'reference-settings',
      nodeCount: 2,
      edgeCount: 1,
      ...options,
      edges: 'edges',
      capacities: 'capacities',
      costs: 'costs',
      parameters: 'parameters',
      output: {flow: 'flow', cut: 'cut', summary: 'summary'}
    };
    const message =
      /reference-settings maximumRelaxationRounds, maximumRelabels, mode and maximumCostPhases require solver parallel/;
    // Before the fix the GPU rejects at lowering while CPU construction accepts the same props.
    expect(() =>
      constructNetworkOperationContributor(new GPUNetworkFlowOperation(props), {
        edges: {format: 'uint32x2', length: 1},
        capacities: {format: 'uint32', length: 1},
        costs: {format: 'uint32', length: 1},
        parameters: {format: 'uint32', length: 3}
      })
    ).toThrow(message);
    expect(() => new CPUNetworkFlowOperation(props)).toThrow(message);
  });
}
