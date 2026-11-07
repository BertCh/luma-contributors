// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the network runs: a seeded street grid, evenly spread node picks, and the road
// buffers every picture draws underneath its output.

import type {RunBuffer, RunBuilder, SegmentsRunLayer} from '../../../define-run';
import {createNetwork, uploadNetworkTopology, type NetworkInput, type World} from '../../../inputs';

/** A street grid with per-slot congestion-free lengths. */
export function createRoads(random: () => number, world: World, rows = 14): NetworkInput {
  return createNetwork(random, world, rows);
}

/** `count` intersections spread across the world (low-discrepancy sequence, deterministic). */
export function pickNodes(
  network: NetworkInput,
  world: World,
  count: number,
  offset = 0
): Uint32Array {
  return Uint32Array.from({length: count}, (_, index) => {
    const step = index + 1 + offset;
    return network.findNearestNode(
      world.width * ((step * 0.7548776662) % 1),
      (step * 0.5698402909) % 1
    );
  });
}

/** (x, y) of each listed node. */
export function gatherPositions(nodes: Float32Array, nodeIds: Uint32Array): Float32Array {
  const positions = new Float32Array(nodeIds.length * 2);
  nodeIds.forEach((node, index) => {
    positions[index * 2] = nodes[node * 2];
    positions[index * 2 + 1] = nodes[node * 2 + 1];
  });
  return positions;
}

/** Road geometry buffers for `draw`. */
export type RoadBuffers = {nodes: RunBuffer<'float32x2'>; edges: RunBuffer<'uint32x2'>};

export function uploadRoads(builder: RunBuilder, network: NetworkInput): RoadBuffers {
  return {
    nodes: builder.upload('road-nodes', 'float32x2', network.nodes),
    edges: builder.upload('road-edges', 'uint32x2', network.edgePairs)
  };
}

/** Muted roads drawn as input context. */
export function getRoadContext(roads: RoadBuffers): SegmentsRunLayer {
  return {layer: 'segments', context: true, nodes: roads.nodes, edges: roads.edges, radius: 1};
}

/** Roads colored by a per-node value buffer (costs, labels, scores). */
export function getRoadValues(
  roads: RoadBuffers,
  values: RunBuffer,
  look: Partial<SegmentsRunLayer> = {}
): SegmentsRunLayer {
  return {
    layer: 'segments',
    nodes: roads.nodes,
    edges: roads.edges,
    values,
    radius: 2,
    ...look
  } as SegmentsRunLayer;
}

// ---------------------------------------------------------------------------------------------
// Closures and per-slot drawing

/** A road network with some roads closed: CSR plus reverse slots and the source node per slot. */
export type RoadNetwork = NetworkInput & {
  /** For each directed slot, the slot of the opposite direction. */
  reverseSlots: Uint32Array;
  /** Source node of each directed slot. */
  slotSources: Uint32Array;
};

function hashFraction(index: number): number {
  return (Math.imul(index + 1, 2654435761) >>> 0) / 4294967296;
}

/** Rebuilds a network without the `closureFraction` of roads with the smallest hash (monotone). */
export function createRoadNetwork(base: NetworkInput, closureFraction: number): RoadNetwork {
  const kept: number[] = [];
  for (let edge = 0; edge < base.edgeCount; edge++) {
    if (hashFraction(edge) >= closureFraction) kept.push(edge);
  }
  const degree = new Uint32Array(base.nodeCount);
  for (const edge of kept) {
    degree[base.edgePairs[edge * 2]]++;
    degree[base.edgePairs[edge * 2 + 1]]++;
  }
  const offsets = new Uint32Array(base.nodeCount + 1);
  for (let node = 0; node < base.nodeCount; node++)
    offsets[node + 1] = offsets[node] + degree[node];
  const slotCount = offsets[base.nodeCount];
  const neighbors = new Uint32Array(slotCount);
  const lengths = new Float32Array(slotCount);
  const delays = new Float32Array(slotCount);
  const reverseSlots = new Uint32Array(slotCount);
  const slotSources = new Uint32Array(slotCount);
  const cursor = Uint32Array.from(offsets.subarray(0, base.nodeCount));
  const edgePairs = new Uint32Array(kept.length * 2);
  kept.forEach((edge, index) => {
    const from = base.edgePairs[edge * 2];
    const to = base.edgePairs[edge * 2 + 1];
    edgePairs[index * 2] = from;
    edgePairs[index * 2 + 1] = to;
    const length = Math.hypot(
      base.nodes[from * 2] - base.nodes[to * 2],
      base.nodes[from * 2 + 1] - base.nodes[to * 2 + 1]
    );
    const forward = cursor[from]++;
    const backward = cursor[to]++;
    neighbors[forward] = to;
    neighbors[backward] = from;
    slotSources[forward] = from;
    slotSources[backward] = to;
    reverseSlots[forward] = backward;
    reverseSlots[backward] = forward;
    lengths[forward] = lengths[backward] = length;
    delays[forward] = delays[backward] = hashFraction(edge + 7919);
  });
  return {
    ...base,
    edgePairs,
    edgeCount: kept.length,
    offsets,
    neighbors,
    slotCount,
    lengths,
    reverseSlots,
    slotSources,
    fillWeights(target, congestion) {
      for (let slot = 0; slot < slotCount; slot++) {
        target[slot] = lengths[slot] * (1 + delays[slot] * congestion);
      }
    }
  };
}

/** Directed slots as drawable lines: virtual node `slot` is the source, `slotCount + slot` the target. */
export type SlotLines = {nodes: RunBuffer<'float32x2'>; edges: RunBuffer<'uint32x2'>};

export function uploadSlotLines(
  builder: RunBuilder,
  network: RoadNetwork,
  laneOffset = 0
): SlotLines {
  const nodes = new Float32Array(network.slotCount * 4);
  const edges = new Uint32Array(network.slotCount * 2);
  for (let slot = 0; slot < network.slotCount; slot++) {
    const source = network.slotSources[slot];
    const target = network.neighbors[slot];
    // Shift each direction to its own side so opposite slots of one road do not overpaint.
    const directionX = network.nodes[target * 2] - network.nodes[source * 2];
    const directionY = network.nodes[target * 2 + 1] - network.nodes[source * 2 + 1];
    const length = Math.hypot(directionX, directionY) || 1;
    const shiftX = (directionY / length) * laneOffset;
    const shiftY = (-directionX / length) * laneOffset;
    nodes[slot * 2] = network.nodes[source * 2] + shiftX;
    nodes[slot * 2 + 1] = network.nodes[source * 2 + 1] + shiftY;
    nodes[(network.slotCount + slot) * 2] = network.nodes[target * 2] + shiftX;
    nodes[(network.slotCount + slot) * 2 + 1] = network.nodes[target * 2 + 1] + shiftY;
    edges[slot * 2] = slot;
    edges[slot * 2 + 1] = network.slotCount + slot;
  }
  return {
    nodes: builder.upload('slot-nodes', 'float32x2', nodes),
    edges: builder.upload('slot-edges', 'uint32x2', edges)
  };
}

/** Roads colored by a per-slot uint32 value (masks, labels); the first endpoint's value is used. */
export function getSlotValues(
  lines: SlotLines,
  values: RunBuffer,
  look: Partial<SegmentsRunLayer> = {}
): SegmentsRunLayer {
  return {
    layer: 'segments',
    nodes: lines.nodes,
    edges: lines.edges,
    values,
    radius: 2,
    ...look
  } as SegmentsRunLayer;
}

/** Upload helper for a closed network's topology. */
export function uploadRoadTopology(builder: RunBuilder, network: RoadNetwork) {
  return uploadNetworkTopology(builder, network, network.lengths);
}

/** A small full grid of roads (columns x rows intersections) for operations with tight limits. */
export function createGridRoadNetwork(
  random: () => number,
  world: World,
  columns: number,
  rows: number
): RoadNetwork {
  const nodeCount = columns * rows;
  const nodes = new Float32Array(nodeCount * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      nodes[index * 2] = ((column + 0.5 + (random() - 0.5) * 0.3) / columns) * world.width;
      nodes[index * 2 + 1] = (row + 0.5 + (random() - 0.5) * 0.3) / rows;
    }
  }
  const edgeList: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (column + 1 < columns) edgeList.push(index, index + 1);
      if (row + 1 < rows) edgeList.push(index, index + columns);
    }
  }
  const base = {
    columns,
    nodeCount,
    nodes,
    edgePairs: Uint32Array.from(edgeList),
    edgeCount: edgeList.length / 2,
    offsets: new Uint32Array(nodeCount + 1),
    neighbors: new Uint32Array(0),
    slotCount: 0,
    lengths: new Float32Array(0),
    fillWeights() {},
    findNearestNode(x: number, y: number) {
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
  return createRoadNetwork(base, 0);
}
