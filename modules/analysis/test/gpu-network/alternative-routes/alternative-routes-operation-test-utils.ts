// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CPUAlternativeRoutesInput,
  CPUAlternativeRoutesResult
} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';

/** Parallel edges, zero-cost cycles, tied paths and impassable slots. */
export function getAlternativeRoutesInput(): CPUAlternativeRoutesInput {
  return {
    topology: {
      offsets: Uint32Array.of(0, 6, 8, 10, 11, 11),
      neighbors: Uint32Array.of(1, 1, 2, 3, 99, 4, 2, 3, 1, 3, 4),
      weights: Float32Array.of(1, 1, 1, 4, 0, NaN, 0, 1, 0, 1, 0.5)
    },
    sources: Uint32Array.of(0, 2),
    targets: Uint32Array.of(4, 4)
  };
}

/** Independent oracle: enumerate every simple path by depth-first traversal, then sort destinations. */
export function enumerateAlternativeRoutes(
  input: CPUAlternativeRoutesInput,
  routeCount: number
): CPUAlternativeRoutesResult {
  const nodeCount = input.topology.offsets.length - 1;
  const queryCount = input.sources.length;
  const vertices = new Uint32Array(queryCount * routeCount * nodeCount).fill(0xffffffff);
  const edges = new Uint32Array(vertices.length).fill(0xffffffff);
  const lengths = new Uint32Array(queryCount * routeCount);
  const costs = new Float32Array(lengths.length).fill(Infinity);
  const counts = new Uint32Array(queryCount);
  for (let query = 0; query < queryCount; query++) {
    const source = input.sources[query];
    const destination = input.targets[query];
    if (source >= nodeCount || destination >= nodeCount) continue;
    const paths: {cost: number; vertices: number[]; edges: number[]}[] = [];
    const visitPath = (path: number[], slots: number[], cost: number) => {
      const vertex = path[path.length - 1];
      if (vertex === destination) {
        paths.push({vertices: path, edges: slots, cost});
        return;
      }
      for (
        let slot = input.topology.offsets[vertex];
        slot < input.topology.offsets[vertex + 1];
        slot++
      ) {
        const neighbor = input.topology.neighbors[slot];
        const weight = input.topology.weights[slot];
        if (
          neighbor < nodeCount &&
          !path.includes(neighbor) &&
          Number.isFinite(weight) &&
          weight >= 0
        )
          visitPath([...path, neighbor], [...slots, slot], cost + weight);
      }
    };
    visitPath([source], [], 0);
    paths.sort((first, second) => {
      if (first.cost !== second.cost) return first.cost - second.cost;
      const difference = first.edges.findIndex((slot, index) => slot !== second.edges[index]);
      return difference < 0
        ? first.edges.length - second.edges.length
        : first.edges[difference] - (second.edges[difference] ?? -1);
    });
    counts[query] = Math.min(paths.length, routeCount);
    paths.slice(0, routeCount).forEach((path, route) => {
      const row = query * routeCount + route;
      vertices.set(path.vertices, row * nodeCount);
      edges.set(path.edges, row * nodeCount);
      lengths[row] = path.vertices.length;
      costs[row] = path.cost;
    });
  }
  return {vertices, edges, lengths, costs, counts, complete: new Uint32Array(queryCount).fill(1)};
}

/** Six simple edge-slot paths: direct cost 0.5, then five cost-2 paths (including parallel slots).
 * Query 1 starts at isolated vertex 4 and has no path. All sums are exact binary fractions.
 */
export function getTraversalEnumerationInput(): CPUAlternativeRoutesInput {
  return {
    topology: {
      offsets: Uint32Array.of(0, 4, 6, 7, 7, 7),
      neighbors: Uint32Array.of(1, 1, 2, 3, 2, 3, 3),
      weights: Float32Array.of(1, 1, 1, 0.5, 0, 1, 1)
    },
    sources: Uint32Array.of(0, 4),
    targets: Uint32Array.of(3, 3)
  };
}

/** Read only the certified route prefix, preserving parallel CSR slot identity. */
export function readTraversalRoutes(
  rows: {edges: ArrayLike<number>; lengths: ArrayLike<number>; costs: ArrayLike<number>},
  count: number,
  maximumPathLength = 5
): {edges: number[]; cost: number}[] {
  return Array.from({length: count}, (_, route) => ({
    edges: Array.from(
      {length: rows.lengths[route] - 1},
      (_, slot) => rows.edges[route * maximumPathLength + slot]
    ),
    cost: rows.costs[route]
  }));
}
