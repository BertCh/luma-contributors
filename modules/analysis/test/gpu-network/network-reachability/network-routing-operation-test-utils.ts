// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUNetworkReachabilityInput} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';

/** Rows 0 and 2 share slot 0; row 1 is empty under the GPU's per-row clamp rule. */
export function getOverlappingRoutingTopology(): CPUNetworkReachabilityInput['topology'] {
  return {
    offsets: Uint32Array.of(0, 1, 0, 1),
    neighbors: Uint32Array.of(1),
    weights: Float32Array.of(0)
  };
}

/** Weighted directed fixture: a zero-cost cycle, competing paths, two components and an isolate. */
export function getRoutingTopology(): CPUNetworkReachabilityInput['topology'] {
  return {
    offsets: Uint32Array.of(0, 3, 4, 7, 8, 10, 11, 11, 13, 13),
    neighbors: Uint32Array.of(1, 2, 3, 3, 1, 3, 4, 5, 2, 5, 6, 6, 3),
    weights: Float32Array.of(6, 1.25, 10, 0.5, 2.25, 5, 0, 0.75, 0, 7, 1, 0.5, 2.5)
  };
}

/** Independent exhaustive simple-path oracle, with no shortest-path relaxation algorithm. */
export function enumerateRoutingCosts(
  topology: CPUNetworkReachabilityInput['topology'],
  starts: readonly {node: number; cost: number}[],
  costLimit = Infinity
): Float32Array {
  const nodeCount = topology.offsets.length - 1;
  const minima = new Float64Array(nodeCount).fill(Infinity);
  const visitPath = (node: number, cost: number, visited: readonly number[]) => {
    minima[node] = Math.min(minima[node], cost);
    for (let edge = topology.offsets[node]; edge < topology.offsets[node + 1]; edge++) {
      const target = topology.neighbors[edge];
      const weight = topology.weights[edge];
      if (
        target >= nodeCount ||
        visited.includes(target) ||
        !(weight >= 0) ||
        !Number.isFinite(weight)
      )
        continue;
      const candidate = cost + weight;
      if (candidate <= costLimit) visitPath(target, candidate, [...visited, target]);
    }
  };
  for (const start of starts)
    if (
      start.node < nodeCount &&
      Number.isFinite(start.cost) &&
      start.cost >= 0 &&
      start.cost <= costLimit
    )
      visitPath(start.node, start.cost, [start.node]);
  return Float32Array.from(minima);
}
