// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeKShortestPathsOnCPU} from '../../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';

for (const scene of ['complete', 'candidate-limit', 'path-limit', 'invalid'] as const) {
  it(`derives every Yen status word by enumerating three simple paths: ${scene}`, () => {
    // Exactly three paths: 0-1-3 via edge 2 (cost 2), via edge 3 (cost 3),
    // and 0-2-3 (cost 4). In Yen round 2 the root-0 spur proposes cost 4
    // and the root-1 spur proposes cost 3: both must coexist before choosing.
    // Thus capacity 1 stops in round 2 with one accepted route; capacity 2
    // accepts all three in three rounds. A two-vertex path limit refuses
    // the first three-vertex route in round 1. Invalid endpoints attempt no round.
    const input = {
      topology: {
        offsets: Uint32Array.of(0, 2, 4, 5, 5),
        neighbors: Uint32Array.of(1, 2, 3, 3, 3),
        weights: Float32Array.of(1, 2, 1, 2, 2)
      },
      sources: Uint32Array.of(scene === 'invalid' ? 99 : 0),
      targets: Uint32Array.of(3)
    };
    const options = {
      routeCount: 3,
      maximumPathLength: scene === 'path-limit' ? 2 : 3,
      candidateCapacity: scene === 'candidate-limit' ? 1 : 2,
      maximumIterations: 2
    };
    const result = computeKShortestPathsOnCPU(input, options);
    const expected = {
      complete: [3, 1, 0, 0, 3],
      'candidate-limit': [1, 0, 0, 1, 2],
      'path-limit': [0, 0, 1, 0, 1],
      invalid: [0, 1, 0, 0, 0]
    }[scene];
    expect([
      result.counts[0],
      result.converged[0],
      result.iterationLimitReached[0],
      result.candidateOverflow[0],
      result.iterationCount[0]
    ]).toEqual(expected);
    // With every edge impassable, the sole valid source exhausts in round 1.
    input.topology.weights.fill(-1);
    const empty = computeKShortestPathsOnCPU(input, options);
    expect([
      empty.counts[0],
      empty.converged[0],
      empty.iterationLimitReached[0],
      empty.candidateOverflow[0],
      empty.iterationCount[0]
    ]).toEqual(scene === 'invalid' ? [0, 1, 0, 0, 0] : [0, 1, 0, 0, 1]);
  });
}
