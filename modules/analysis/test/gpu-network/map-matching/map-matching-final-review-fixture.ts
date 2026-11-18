// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUMapMatchingParameterValues} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {computeNetworkCostTableOnCPU} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';

/** Twelve-node chain: an eight-slot fallback cannot settle node 10 from node 1. */
export function createFinalMatchingInput() {
  const topology = {
    offsets: Uint32Array.from({length: 13}, (_, row) => Math.min(row, 11)),
    neighbors: Uint32Array.from({length: 11}, (_, row) => row + 1),
    weights: new Float32Array(11).fill(1),
    nodePositions: Float32Array.from({length: 24}, (_, component) =>
      component % 2 === 0 ? component / 2 : 0
    )
  };
  const costLimit = Float32Array.of(40);
  return {
    topology,
    positions: Float32Array.of(0.5, 0, 10.5, 0),
    trackOffsets: Uint32Array.of(0, 2),
    parameters: getGPUMapMatchingParameterValues({
      sigma: 0.1,
      beta: 1,
      searchRadius: 0.1,
      routeFactor: 3,
      routeSlack: 0
    }),
    routeTable: {...computeNetworkCostTableOnCPU({topology, costLimit}, {capacity: 144}), costLimit}
  };
}

/** A long nearest road beside a three-edge alternative 0.25 units away. */
export function createParallelRoadMatchingInput() {
  return {
    topology: {
      offsets: Uint32Array.from({length: 17}, (_, row) =>
        row <= 11 ? row : Math.min(row - 1, 14)
      ),
      neighbors: Uint32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]),
      nodePositions: Float32Array.from([
        ...Array.from({length: 12}, (_, row) => [row, 0]).flat(),
        0,
        0.25,
        1,
        0.25,
        10,
        0.25,
        11,
        0.25
      ])
    },
    positions: Float32Array.of(0.5, 0, 10.5, 0),
    trackOffsets: Uint32Array.of(0, 2),
    parameters: getGPUMapMatchingParameterValues({
      sigma: 0.2,
      beta: 1,
      searchRadius: 0.5,
      routeFactor: 3,
      routeSlack: 0
    })
  };
}
