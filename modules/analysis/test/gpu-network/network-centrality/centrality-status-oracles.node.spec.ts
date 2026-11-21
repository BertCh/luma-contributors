// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkCentralityOnCPU} from '../../../src/gpu-network/network-centrality/cpu-network-centrality-operation';

for (const maximumIterations of [1, 16]) {
  it(`derives centrality round exhaustion from the unsettled chain target (${maximumIterations})`, () => {
    // Eight unit edges from 0 to 8. One round of one local hop cannot
    // settle node 8; sixteen rounds settle the chain and certify no changes.
    const result = computeNetworkCentralityOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 8),
          neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
          weights: new Float32Array(8).fill(1)
        },
        reverseTopology: {
          offsets: Uint32Array.of(0, 0, 1, 2, 3, 4, 5, 6, 7, 8),
          neighbors: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
          weights: new Float32Array(8).fill(1)
        },
        pivotNodes: Uint32Array.of(0)
      },
      {maximumIterations, localIterations: 1, computeBetweenness: false}
    );
    expect(Array.from(result.iterationLimitReached)).toEqual([maximumIterations === 1 ? 1 : 0]);
    expect(Array.from(result.converged)).toEqual([maximumIterations === 1 ? 0 : 1]);
    expect(Array.from(result.zeroWeightCycle)).toEqual([0]);
  });
}
