// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import streets from '../data/sf-market-streets.json';
import {fitLocalMap} from '../data/fit-local-map';

export type StreetNetwork = {
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

export function createStreetNetwork(worldWidth: number, maximumNodes: number): StreetNetwork {
  // The fixture is ordered so that every prefix stays connected, even in the smallest demo.
  const locations = streets.nodes.slice(0, maximumNodes);
  const project = fitLocalMap(locations, {width: worldWidth, height: 1});
  const nodeCount = locations.length;
  const nodes = Float32Array.from(locations.flatMap(([x, y]) => project(x, y)));
  const edgeList = streets.edges.filter(([, to]) => to < nodeCount).flat();
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
    // Reproducible scenario cost. TIGER records road topology, not traffic or travel speeds.
    const delay = (Math.imul(edge + 1, 2654435761) >>> 0) / 4294967296;
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
