// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the network runs: San Francisco street topology, evenly spread node picks, and the road
// buffers every picture draws underneath its output.

import type {DemoTheme} from '../../../../demos/kit';
import type {DotsRunLayer, RunBuffer, RunBuilder, SegmentsRunLayer} from '../../../define-run';
import {createNetwork, uploadNetworkTopology, type NetworkInput, type World} from '../../../inputs';
import {createPointIndex, type PointIndex} from '../../../pick';

/** Real intersections with per-slot congestion-free lengths. */
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

/** A connected neighborhood within the operation's node limit. */
export function createSmallRoadNetwork(
  random: () => number,
  world: World,
  nodeLimit: number
): RoadNetwork {
  return createRoadNetwork(createNetwork(random, world, 1, nodeLimit), 0);
}

// ---------------------------------------------------------------------------------------------
// Pointer helpers

/** Nearest-intersection lookup for place and drag gestures. */
export function createNodeIndex(network: NetworkInput): PointIndex {
  return createPointIndex(network.nodes, network.nodeCount);
}

/** (x0, y0, x1, y1) of every directed slot, in CSR order. */
export function createSlotSegments(network: NetworkInput): Float32Array {
  const segments = new Float32Array(network.slotCount * 4);
  for (let node = 0; node < network.nodeCount; node++) {
    for (let slot = network.offsets[node]; slot < network.offsets[node + 1]; slot++) {
      const target = network.neighbors[slot];
      segments[slot * 4] = network.nodes[node * 2];
      segments[slot * 4 + 1] = network.nodes[node * 2 + 1];
      segments[slot * 4 + 2] = network.nodes[target * 2];
      segments[slot * 4 + 3] = network.nodes[target * 2 + 1];
    }
  }
  return segments;
}

/**
 * The directed road under a position and the fraction along it. A position on the right of the
 * direction of travel prefers that direction, so the two slots of one road are told apart.
 */
export function findNearestSlot(
  segments: Float32Array,
  x: number,
  y: number
): {slot: number; fraction: number; point: [number, number]} {
  const count = segments.length >> 2;
  let bestSlot = 0;
  let bestScore = Infinity;
  let bestFraction = 0;
  let bestPoint: [number, number] = [0, 0];
  for (let slot = 0; slot < count; slot++) {
    const x0 = segments[slot * 4];
    const y0 = segments[slot * 4 + 1];
    const deltaX = segments[slot * 4 + 2] - x0;
    const deltaY = segments[slot * 4 + 3] - y0;
    const lengthSquared = deltaX * deltaX + deltaY * deltaY;
    const fraction =
      lengthSquared > 0
        ? Math.min(1, Math.max(0, ((x - x0) * deltaX + (y - y0) * deltaY) / lengthSquared))
        : 0;
    const pointX = x0 + deltaX * fraction;
    const pointY = y0 + deltaY * fraction;
    const distance = Math.hypot(x - pointX, y - pointY);
    const isLeft = deltaX * (y - y0) - deltaY * (x - x0) > 0;
    const score = distance + (isLeft ? 1e-3 : 0);
    if (score < bestScore) {
      bestScore = score;
      bestSlot = slot;
      bestFraction = fraction;
      bestPoint = [pointX, pointY];
    }
  }
  return {slot: bestSlot, fraction: bestFraction, point: bestPoint};
}

/** Pins for origins, sources, sinks and starts: an ink halo under a filled dot. */
export function getEndpointMarkers(
  positions: RunBuffer,
  theme: DemoTheme,
  options: {count?: number; colorIndex?: number; radius?: number} = {}
): DotsRunLayer[] {
  const radius = options.radius ?? 7;
  const color = theme.categories[(options.colorIndex ?? 1) % theme.categories.length];
  return [
    {
      layer: 'dots',
      positions,
      count: options.count,
      radius: radius + 3,
      color: [...theme.ink, 1]
    },
    {layer: 'dots', positions, count: options.count, radius, color: [...color, 1]}
  ];
}

/**
 * Look for a uint32 mask: 0 is muted (dropped, closed), 1 is bright. Pass `danger` to draw the
 * zero state in the alert color instead (closed roads).
 */
export function getMaskLook(
  theme: DemoTheme,
  options: {danger?: boolean; invert?: boolean} = {}
): Pick<SegmentsRunLayer, 'ramp' | 'categories'> {
  const muted = [...theme.muted, 0.28] as const;
  const bright = [...theme.categories[0], 1] as const;
  const danger = [...theme.categories[2], 1] as const;
  const zero = options.danger ? danger : muted;
  const one = options.danger ? muted : bright;
  return {ramp: 'categorical', categories: options.invert ? [one, zero] : [zero, one]};
}

/**
 * A world position where the nearest anchor is `distance` away: the rest position of a reach
 * gesture. Scans a grid and keeps the best fit, preferring the middle of the picture.
 */
export function findReachPoint(
  anchors: ArrayLike<number>,
  distance: number,
  world: World
): [number, number] {
  let best: [number, number] = [world.width / 2, 0.5];
  let bestError = Infinity;
  const count = anchors.length >> 1;
  for (let row = 0; row <= 36; row++) {
    for (let column = 0; column <= 64; column++) {
      const x = world.width * (0.04 + (0.92 * column) / 64);
      const y = 0.04 + (0.92 * row) / 36;
      let nearest = Infinity;
      for (let anchor = 0; anchor < count; anchor++) {
        nearest = Math.min(
          nearest,
          Math.hypot(anchors[anchor * 2] - x, anchors[anchor * 2 + 1] - y)
        );
      }
      const error = Math.abs(nearest - distance) + 0.01 * Math.hypot(x - world.width / 2, y - 0.5);
      if (error < bestError) {
        bestError = error;
        best = [x, y];
      }
    }
  }
  return best;
}

/** A world position far from every anchor: the rest position of a probe that starts unhovered. */
export function findEmptyPoint(anchors: ArrayLike<number>, world: World): [number, number] {
  let best: [number, number] = [world.width / 2, 0.5];
  let bestDistance = -1;
  const count = anchors.length >> 1;
  for (let row = 0; row <= 18; row++) {
    for (let column = 0; column <= 32; column++) {
      const x = world.width * (0.04 + (0.92 * column) / 32);
      const y = 0.04 + (0.92 * row) / 18;
      let nearest = Infinity;
      for (let anchor = 0; anchor < count; anchor++) {
        nearest = Math.min(
          nearest,
          Math.hypot(anchors[anchor * 2] - x, anchors[anchor * 2 + 1] - y)
        );
      }
      if (nearest > bestDistance) {
        bestDistance = nearest;
        best = [x, y];
      }
    }
  }
  return best;
}

/** A small copy of the road map drawn in the left half of the picture, beside a chart. */
export type InsetMap = {
  scale: number;
  transform(positions: Float32Array): Float32Array;
  toMap(x: number, y: number): [number, number];
};

export function createInsetMap(world: World): InsetMap {
  const scale = 0.46;
  const offsetX = world.width * 0.02;
  const offsetY = 0.27;
  return {
    scale,
    toMap: (x, y) => [offsetX + x * scale, offsetY + y * scale],
    transform: positions =>
      Float32Array.from(positions, (value, index) =>
        index % 2 === 0 ? offsetX + value * scale : offsetY + value * scale
      )
  };
}
