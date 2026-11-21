// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkReachabilityOnCPU} from '../../src/gpu-network/network-reachability/cpu-network-reachability-operation';

it('review B2 bounded public search chooses the least passable parallel edge beyond its witness', () => {
  const result = computeNetworkReachabilityOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 2, 4, 5, 5),
        neighbors: Uint32Array.of(1, 2, 3, 3, 1),
        weights: Float32Array.of(5, 0, 1, 5, 1)
      },
      startNodes: Uint32Array.of(0)
    },
    {maximumIterations: 2, localIterations: 1}
  );
  // Round 1: [0,5,0,inf]; round 2: [0,1,0,6]. Final parent 1 witnesses 6
  // through weight 5, but authoritative provenance uses its least passable edge (slot 2).
  expect(Array.from(result.costs)).toEqual([0, 1, 0, 6]);
  expect(result.converged[0]).toBe(0);
  expect(result.predecessors[3]).toBe(1);
  expect(result.predecessorEdges[3]).toBe(2);
});

it('review earlier-fix overlapping rows keep tie witnesses in the source row domain', () => {
  const result = computeNetworkReachabilityOnCPU({
    topology: {
      offsets: Uint32Array.of(0, 2, 2, 0, 2),
      neighbors: Uint32Array.of(1, 1),
      weights: Float32Array.of(0, 1)
    },
    startNodes: Uint32Array.of(0, 3),
    startCosts: Float32Array.of(19, 2)
  });
  // Two roots share CSR slots. Only root 3 at cost 2 can witness node 1 at cost 2;
  // root 0 at cost 19 cannot inherit root 3's zero-weight tie flag.
  expect(result.costs[1]).toBe(2);
  expect(result.converged[0]).toBe(1);
  expect(result.predecessors[1]).toBe(3);
  expect(result.predecessorEdges[1]).toBe(0);
});
