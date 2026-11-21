// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkCostTableOnCPU} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';

for (const maximumIterations of [1, 4]) {
  it(`derives cost-table stopping and counts from a three-node chain (${maximumIterations})`, () => {
    // Self pairs (0,0),(1,1),(2,2) plus two edges give 5 after one hop.
    // Pair (0,2) remains unresolved, and completing the chain gives 6 pairs.
    const result = computeNetworkCostTableOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2, 2),
          neighbors: Uint32Array.of(1, 2),
          weights: Float32Array.of(1, 1)
        },
        costLimit: Float32Array.of(Infinity)
      },
      {capacity: 9, maximumIterations, localIterations: 1}
    );
    expect([
      result.complete[0],
      result.converged[0],
      result.iterationLimitReached[0],
      result.overflow[0],
      result.requiredCount[0]
    ]).toEqual(maximumIterations === 1 ? [0, 0, 1, 0, 5] : [1, 1, 0, 0, 6]);
  });
}
