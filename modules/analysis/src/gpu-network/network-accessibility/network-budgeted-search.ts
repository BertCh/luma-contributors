// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addNetworkRoutingCosts,
  resolveNetworkRoutingCostPrecision,
  type CPUNetworkRoutingInput
} from '../../analysis-core/operations/cpu-network-routing';

/** Deterministic host scheduling of analysis routing rounds, including the 256-slot local spill.
 * GPU workgroup order can change partial labels with competing discoveries; this schedule visits
 * chunks and CSR slots in order and never spends a spilled entry in the round that spilled it.
 * Inputs and budgets are validated by the caller. @internal
 */
export function computeBudgetedNetworkDistances(
  input: CPUNetworkRoutingInput,
  maximumIterations: number,
  localIterations: number
): {distances: Float64Array; converged: boolean} {
  const {topology, startNodes, startCosts} = input;
  const costPrecision = resolveNetworkRoutingCostPrecision(
    'computeBudgetedNetworkDistances',
    input.costPrecision
  );
  const nodeCount = topology.offsets.length - 1;
  const distances = new Float64Array(nodeCount).fill(Infinity);
  const cutoff = input.costLimit?.[0] ?? Infinity;
  let frontier: number[] = [];
  for (let row = 0; row < startNodes.length; row++) {
    const node = startNodes[row];
    const cost = startCosts?.[row] ?? 0;
    if (
      node < nodeCount &&
      Number.isFinite(cost) &&
      cost >= 0 &&
      !(cost > cutoff) &&
      cost < distances[node]
    ) {
      if (!Number.isFinite(distances[node])) frontier.push(node);
      distances[node] = cost === 0 ? 0 : cost;
    }
  }
  for (let round = 0; round < maximumIterations && frontier.length; round++) {
    const nextFrontier = new Set<number>();
    for (let chunk = 0; chunk < frontier.length; chunk += 256) {
      let localFrontier = frontier.slice(chunk, chunk + 256);
      for (let hop = 0; hop < localIterations && localFrontier.length; hop++) {
        const nextLocalFrontier: number[] = [];
        const sources = localFrontier.map(node => ({node, sourceCost: distances[node]}));
        for (const {node, sourceCost} of sources) {
          for (
            let edge = Math.min(topology.offsets[node], topology.neighbors.length);
            edge < Math.min(topology.offsets[node + 1], topology.neighbors.length);
            edge++
          ) {
            const target = topology.neighbors[edge];
            const weight = topology.weights[edge];
            const candidate = addNetworkRoutingCosts(sourceCost, weight, costPrecision);
            if (
              target < nodeCount &&
              weight >= 0 &&
              Number.isFinite(candidate) &&
              !(candidate > cutoff) &&
              candidate < distances[target]
            ) {
              distances[target] = candidate;
              if (hop + 1 < localIterations && nextLocalFrontier.length < 256)
                nextLocalFrontier.push(target);
              else nextFrontier.add(target);
            }
          }
        }
        localFrontier = nextLocalFrontier;
      }
    }
    frontier = Array.from(nextFrontier);
  }
  return {distances, converged: frontier.length === 0};
}
