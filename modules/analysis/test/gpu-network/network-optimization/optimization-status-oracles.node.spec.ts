// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkFlowOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {computeNetworkPointRoutingOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {computeNetworkTimeRoutingOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';

for (const scene of ['complete', 'truncated', 'invalid', 'stopped'] as const) {
  it(`enumerates both unit flow paths and every diagnostic (${scene})`, () => {
    // Two parallel unit edges require two augmentations and two decomposed paths.
    // Capacity one stores only the first; source=sink is invalid; a zero solver
    // budget leaves both unit paths unserved. There is no numerical tolerance.
    const result = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.of(0, 1, 0, 1),
        capacities: Uint32Array.of(1, 1),
        parameters: Uint32Array.of(0, scene === 'invalid' ? 0 : 1, 2)
      },
      {
        nodeCount: 2,
        maximumAugmentations: scene === 'stopped' ? 0 : 2,
        pathCapacity: scene === 'truncated' ? 1 : 2
      }
    );
    const status = result.status;
    expect([
      status.converged[0],
      status.iterationLimitReached[0],
      status.invalidCount[0],
      status.iterationCount[0]
    ]).toEqual(
      scene === 'invalid' ? [0, 0, 1, 0] : scene === 'stopped' ? [0, 1, 0, 0] : [1, 0, 0, 2]
    );
    const paths = result.paths!.status;
    expect([
      paths.count[0],
      paths.requiredCount[0],
      paths.overflow[0],
      paths.converged[0],
      paths.iterationLimitReached[0],
      paths.invalidCount[0]
    ]).toEqual(
      {
        complete: [2, 2, 0, 1, 0, 0],
        truncated: [1, 2, 1, 1, 0, 0],
        invalid: [0, 0, 0, 0, 0, 1],
        stopped: [0, 0, 0, 0, 1, 0]
      }[scene]
    );
  });
}

for (const mode of ['reference', 'scalable'] as const)
  for (const maximumRelaxations of [0, 4])
    it(`derives complete invalid and exhausted point-routing words (${mode}, ${maximumRelaxations})`, () => {
      // The only turn is edge 0 -> edge 1: the valid query needs it and costs
      // 1+2=3 when settled. Query 99->1 has an invalid endpoint. Zero budget
      // cannot examine the turn, so the valid query reports exhausted=2.
      const result = computeNetworkPointRoutingOnCPU(
        {
          turns: Uint32Array.of(0, 1),
          costs: Float32Array.of(1, 2),
          queries: Uint32Array.of(0, 1, 99, 1),
          fractions: Float32Array.of(0, 1, 0, 1)
        },
        {mode, maximumRelaxations}
      );
      expect(Array.from(result.status)).toEqual(maximumRelaxations === 0 ? [2, 1] : [0, 1]);
      expect(Array.from(result.costs)).toEqual(
        maximumRelaxations === 0 ? [Infinity, Infinity] : [3, Infinity]
      );
    });

for (const maximumRelaxations of [1, 3])
  it(`derives time-routing completion from a reverse-listed chain (${maximumRelaxations})`, () => {
    // Scan 1->2 before 0->1: sweep 1 reaches only node 1, sweep 2
    // reaches node 2, sweep 3 sees no change. Thus the unresolved item
    // after sweep 1 is the still-unreachable target 2.
    const result = computeNetworkTimeRoutingOnCPU({
      stateVertices: Uint32Array.of(0, 1, 2),
      transitions: Uint32Array.of(1, 2, 0, 1),
      costs: Float32Array.of(1, 0, 1, 0),
      endpoints: Uint32Array.of(0, 2),
      departure: Float32Array.of(0),
      maximumRelaxations
    });
    expect([
      result.status.converged[0],
      result.status.iterationCount[0],
      result.status.iterationLimitReached[0]
    ]).toEqual(maximumRelaxations === 1 ? [0, 1, 1] : [1, 3, 0]);
  });
