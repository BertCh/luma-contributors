// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkRoutingDistances,
  computeNetworkRoutingPredecessorsWithStatus
} from '../../src/analysis-core/operations/cpu-network-routing';

it('N1 keeps the tie parent specific to its source when CSR rows overlap', () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 1, 1, 0, 1),
      neighbors: Uint32Array.of(1),
      weights: Float32Array.of(0)
    },
    startNodes: Uint32Array.of(0, 3),
    startCosts: Float32Array.of(7, 0)
  };
  const {distances} = computeNetworkRoutingDistances('overlapping-rows', input);
  // Both starts reach node 1 through slot 0. Only 3 -> 1 matches its final cost of 0.
  expect(distances).toEqual(Float64Array.of(7, 0, Infinity, 0));
  expect(computeNetworkRoutingPredecessorsWithStatus(input, distances)).toEqual({
    predecessors: Uint32Array.of(0xffffffff, 3, 0xffffffff, 0xffffffff),
    predecessorEdges: Uint32Array.of(0xffffffff, 0, 0xffffffff, 0xffffffff),
    converged: true
  });
});
