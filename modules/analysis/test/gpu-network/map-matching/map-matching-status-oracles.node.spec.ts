// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMapMatchingOnCPU} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {getGPUMapMatchingParameterValues} from '../../../src/gpu-network/map-matching/gpu-map-matching';

for (const scene of ['forward', 'reversed', 'unmatched'] as const) {
  it(`derives matching counts and flags on a single directed edge: ${scene}`, () => {
    // Tracks contain two fixes and one singleton. Forward fixes share one edge,
    // so each track needs one traversal. Reversing the first pair cannot travel
    // backward on a directed edge: it restarts, emits [0,NONE,0], and needs 3
    // entries despite capacity 1. Remote fixes are outside the search radius;
    // the second unmatched fix is the only counted break (track starts are not).
    // The target node has no outgoing edges, so no route search can drop a node.
    const result = computeMapMatchingOnCPU(
      {
        positions:
          scene === 'unmatched'
            ? Float32Array.of(10, 10, 10, 10, 10, 10)
            : scene === 'reversed'
              ? Float32Array.of(0.75, 0, 0.25, 0, 0.5, 0)
              : Float32Array.of(0.25, 0, 0.75, 0, 0.5, 0),
        trackOffsets: Uint32Array.of(0, 2, 3),
        parameters: getGPUMapMatchingParameterValues({sigma: 0.1, beta: 1, searchRadius: 0.1}),
        topology: {
          offsets: Uint32Array.of(0, 1, 1),
          neighbors: Uint32Array.of(1),
          nodePositions: Float32Array.of(0, 0, 1, 0)
        }
      },
      {cellSize: 1, candidateCount: 1, routeNodeBudget: 8, traversedEdgeCapacity: 1}
    );
    expect([
      result.matchedCount[0],
      result.breakCount[0],
      result.overflow[0],
      result.candidateOverflow[0],
      result.routeOverflow[0],
      ...result.traversedCounts,
      result.traversedOverflow[0],
      result.traversedCandidateOverflow[0]
    ]).toEqual(
      {
        forward: [3, 0, 0, 0, 0, 1, 1, 0, 0],
        reversed: [3, 1, 0, 0, 0, 3, 1, 1, 0],
        unmatched: [0, 1, 0, 0, 0, 0, 0, 0, 0]
      }[scene]
    );
  });
}
