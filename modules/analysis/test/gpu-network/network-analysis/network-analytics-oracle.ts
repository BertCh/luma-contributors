// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  buildCSR,
  type NetworkCSR,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';

/**
 * Out-degree of every node from CSR offsets. Only valid for a CSR without dead slots (neighbors
 * `>= nodeCount`): the contributor counts live slots (review network-paths-06), which the spec
 * "degree skips dead slots" pins by hand.
 */
export function degreeOracle(offsets: Uint32Array): Uint32Array {
  return Uint32Array.from(
    {length: offsets.length - 1},
    (_, node) => offsets[node + 1] - offsets[node]
  );
}

/** Transposes a CSR (weights are zero). Edges with out-of-range targets are dropped. */
export function buildReverseCSR(nodeCount: number, csr: NetworkCSR): NetworkCSR {
  const edges: NetworkEdge[] = [];
  for (let node = 0; node < nodeCount; node++) {
    for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
      if (csr.neighbors[edge] < nodeCount) {
        edges.push([csr.neighbors[edge], node, 0]);
      }
    }
  }
  return buildCSR(nodeCount, edges);
}

/**
 * Deduplicated two-way edge list: each unordered pair of distinct nodes appears once in each
 * direction, which is the symmetric forward CSR convention of the undirected mode.
 */
export function createSymmetricEdges(edges: readonly NetworkEdge[]): NetworkEdge[] {
  const pairs = new Set<string>();
  const result: NetworkEdge[] = [];
  for (const [from, to] of edges) {
    if (from === to || pairs.has(`${Math.min(from, to)}:${Math.max(from, to)}`)) {
      continue;
    }
    pairs.add(`${Math.min(from, to)}:${Math.max(from, to)}`);
    result.push([from, to, 1], [to, from, 1]);
  }
  return result;
}

/** Two-way road grid with `width * height` nodes in row-major order. */
export function createGridEdges(width: number, height: number): NetworkEdge[] {
  const edges: NetworkEdge[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const node = row * width + column;
      if (column + 1 < width) {
        edges.push([node, node + 1, 1], [node + 1, node, 1]);
      }
      if (row + 1 < height) {
        edges.push([node, node + width, 1], [node + width, node, 1]);
      }
    }
  }
  return edges;
}

/** Weak connected components labelled by the lowest node index, as `GPUGraphAlgorithmsConnectedComponents`. */
export function componentsOracle(nodeCount: number, csr: NetworkCSR): Uint32Array {
  const parent = Uint32Array.from({length: nodeCount}, (_, node) => node);
  const find = (node: number): number => {
    while (parent[node] !== node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  };
  for (let node = 0; node < nodeCount; node++) {
    for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
      const neighbor = csr.neighbors[edge];
      if (neighbor < nodeCount) {
        const left = find(node);
        const right = find(neighbor);
        parent[Math.max(left, right)] = Math.min(left, right);
      }
    }
  }
  return Uint32Array.from({length: nodeCount}, (_, node) => find(node));
}

/** `(value - min) / (max - min)` in f32, or 0 when the column is constant. */
export function normalizeOracle(values: ArrayLike<number>): Float32Array {
  const lowest = Math.min(...Array.from(values));
  const highest = Math.max(...Array.from(values));
  const range = Math.fround(Math.fround(highest) - Math.fround(lowest));
  return Float32Array.from(values, value =>
    range > 0 ? Math.fround(Math.fround(Math.fround(value) - Math.fround(lowest)) / range) : 0
  );
}

/** Returns `[min, max]` of a column. */
export function extentOracle(values: ArrayLike<number>): [number, number] {
  const array = Array.from(values);
  return [Math.min(...array), Math.max(...array)];
}
