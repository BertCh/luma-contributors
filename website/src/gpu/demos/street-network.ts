// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A seeded street network for the network demos: a jittered lattice of intersections with some
// blocks knocked out and a few diagonals, stored as CSR topology (offsets, neighbors) plus a
// per-slot length and random delay, so congestion can restretch the costs without rebuilding it.

export type StreetNetwork = {
  columns: number;
  nodeCount: number;
  /** Intersection positions, (x, y) pairs. */
  nodes: Float32Array;
  /** Undirected roads as (from, to) pairs. */
  edgePairs: Uint32Array;
  edgeCount: number;
  /** CSR row offsets, `nodeCount + 1` entries. */
  offsets: Uint32Array;
  /** CSR neighbor node per directed slot. */
  neighbors: Uint32Array;
  slotCount: number;
  /** Writes `length * (1 + delay * congestion)` for every slot into `target`. */
  fillWeights(target: Float32Array, congestion: number): void;
  /** Index of the intersection closest to a world position. */
  findNearestNode(x: number, y: number): number;
};

export function createStreetNetwork(
  random: () => number,
  worldWidth: number,
  rows: number
): StreetNetwork {
  const columns = Math.max(16, Math.round(rows * worldWidth));
  const nodeCount = columns * rows;
  const nodes = new Float32Array(nodeCount * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      nodes[index * 2] = ((column + 0.5 + (random() - 0.5) * 0.5) / columns) * worldWidth;
      nodes[index * 2 + 1] = (row + 0.5 + (random() - 0.5) * 0.5) / rows;
    }
  }
  const edgeList: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (column + 1 < columns && random() > 0.1) edgeList.push(index, index + 1);
      if (row + 1 < rows && random() > 0.1) edgeList.push(index, index + columns);
      if (column + 1 < columns && row + 1 < rows && random() > 0.93) {
        edgeList.push(index, index + columns + 1);
      }
    }
  }
  const edgeCount = edgeList.length / 2;

  // Undirected roads become two directed CSR slots.
  const degree = new Uint32Array(nodeCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    degree[edgeList[edge * 2]]++;
    degree[edgeList[edge * 2 + 1]]++;
  }
  const offsets = new Uint32Array(nodeCount + 1);
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] = offsets[node] + degree[node];
  const slotCount = offsets[nodeCount];
  const neighbors = new Uint32Array(slotCount);
  const slotLengths = new Float32Array(slotCount);
  const slotDelays = new Float32Array(slotCount);
  const cursor = Uint32Array.from(offsets.subarray(0, nodeCount));
  for (let edge = 0; edge < edgeCount; edge++) {
    const from = edgeList[edge * 2];
    const to = edgeList[edge * 2 + 1];
    const length = Math.hypot(
      nodes[from * 2] - nodes[to * 2],
      nodes[from * 2 + 1] - nodes[to * 2 + 1]
    );
    const delay = random();
    for (const [source, target] of [
      [from, to],
      [to, from]
    ]) {
      slotLengths[cursor[source]] = length;
      slotDelays[cursor[source]] = delay;
      neighbors[cursor[source]++] = target;
    }
  }

  return {
    columns,
    nodeCount,
    nodes,
    edgePairs: Uint32Array.from(edgeList),
    edgeCount,
    offsets,
    neighbors,
    slotCount,
    fillWeights(target, congestion) {
      for (let slot = 0; slot < slotCount; slot++) {
        target[slot] = slotLengths[slot] * (1 + slotDelays[slot] * congestion);
      }
    },
    findNearestNode(x, y) {
      let best = 0;
      let bestDistance = Infinity;
      for (let node = 0; node < nodeCount; node++) {
        const distance = (nodes[node * 2] - x) ** 2 + (nodes[node * 2 + 1] - y) ** 2;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = node;
        }
      }
      return best;
    }
  };
}
