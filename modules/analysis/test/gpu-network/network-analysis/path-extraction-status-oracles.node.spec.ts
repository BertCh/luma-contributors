// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkPathExtractionOnCPU} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';

for (const maximumPathLength of [1, 4]) {
  it(`derives node and edge truncation independently of walk loss (${maximumPathLength})`, () => {
    // Targets 2 and 0 in 0->1->2 need 3+1 node rows and 2+0 edge rows.
    // A walk bound of 1 leaves predecessor 1 of target 2 unresolved, suppressing
    // that path but retaining the one-node root path. Capacities are 2/1.
    const result = computeNetworkPathExtractionOnCPU({
      predecessors: Uint32Array.of(0xffffffff, 0, 1),
      predecessorEdges: Uint32Array.of(0xffffffff, 0, 1),
      costs: Float32Array.of(0, 2, 5),
      targets: Uint32Array.of(2, 0),
      capacity: 2,
      maximumPathLength,
      edges: {
        offsets: Uint32Array.of(0, 1, 2, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(2, 3),
        capacity: 1
      }
    });
    const edge = result.edges!;
    expect([
      result.count[0],
      result.requiredCount[0],
      result.overflow[0],
      result.candidateOverflow[0],
      edge.count[0],
      edge.requiredCount[0],
      edge.overflow[0],
      edge.candidateOverflow[0]
    ]).toEqual(maximumPathLength === 1 ? [1, 1, 0, 1, 0, 0, 0, 1] : [2, 4, 1, 0, 1, 2, 1, 0]);
  });
}
