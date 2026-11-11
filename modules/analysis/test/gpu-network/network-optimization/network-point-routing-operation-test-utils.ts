// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUNetworkPointRoutingInput} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';

/** Competing turns, a returning cycle and one disconnected edge. */
export function getPointRoutingInput(): CPUNetworkPointRoutingInput {
  return {
    costs: Float32Array.of(8, 4, 6, 2, 3, 5),
    turns: Uint32Array.of(0, 1, 1, 2, 0, 3, 3, 2, 2, 0, 3, 4, 4, 2),
    queries: Uint32Array.of(0, 2, 0, 0, 0, 0, 4, 1, 5, 2),
    fractions: Float32Array.of(0.25, 0.5, 0.75, 0.25, 0.25, 0.75, 0, 0.5, 0, 1)
  };
}

/** Independent exhaustive traversal of simple edge-state paths, plus the final partial turn. */
export function enumeratePointRoutingCosts(input: CPUNetworkPointRoutingInput): Float32Array {
  const result = new Float32Array(input.queries.length / 2).fill(Infinity);
  for (let query = 0; query < result.length; query++) {
    const source = input.queries[query * 2];
    const destination = input.queries[query * 2 + 1];
    const sourceFraction = input.fractions[query * 2];
    const targetFraction = input.fractions[query * 2 + 1];
    let best =
      source === destination && targetFraction >= sourceFraction
        ? (targetFraction - sourceFraction) * input.costs[source]
        : Infinity;
    const visitPath = (incoming: number, cost: number, visited: readonly number[]) => {
      for (let turn = 0; turn < input.turns.length; turn += 2) {
        if (input.turns[turn] !== incoming) continue;
        const outgoing = input.turns[turn + 1];
        if (outgoing === destination)
          best = Math.min(best, cost + targetFraction * input.costs[destination]);
        if (!visited.includes(outgoing))
          visitPath(outgoing, cost + input.costs[outgoing], [...visited, outgoing]);
      }
    };
    visitPath(source, (1 - sourceFraction) * input.costs[source], [source]);
    result[query] = best;
  }
  return result;
}
