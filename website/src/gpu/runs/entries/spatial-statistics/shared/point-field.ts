// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs of the spatial statistics runs: points with a spatial field, k-nearest-neighbor
// weights, and helpers to mix noise into the field and to draw the neighbor graph.

import type {Random, World} from '../../../inputs';
import {
  createDistanceBandWeights,
  createKnnWeights,
  createUniformPoints,
  type CsrWeights
} from '../../../inputs';

/** Share of the world height (from the bottom) reserved for a chart under the map. */
export const CHART_BAND = 0.36;

export type PointField = {
  count: number;
  /** Width of the world the points were generated in. */
  worldWidth: number;
  positions: Float32Array;
  /** Smooth field in [0, 1]. */
  field: Float32Array;
  /** Independent noise in [0, 1], mixed in by `mixNoise`. */
  noise: Float32Array;
};

export type NeighborGraph = {
  weights: CsrWeights;
  /** Neighbor graph as node pairs (uint32x2), one per weights slot. */
  edges: Uint32Array;
  edgeCount: number;
};

/** k-nearest-neighbor weights of `positions` plus the graph edges for drawing. */
export function createNeighborGraph(positions: Float32Array, neighborCount: number): NeighborGraph {
  const weights = createKnnWeights(positions, neighborCount);
  const edges = createWeightEdges(weights);
  return {weights, edges, edgeCount: edges.length / 2};
}

/** Distance-band weights (pairs closer than `radius`) plus the graph edges; sparse areas become islands. */
export function createBandGraph(positions: Float32Array, radius: number): NeighborGraph {
  const weights = createDistanceBandWeights(positions, radius);
  const edges = createWeightEdges(weights);
  return {weights, edges, edgeCount: edges.length / 2};
}

/** Pairs (row, neighbor) for every slot of the weights, for drawing the neighbor graph. */
export function createWeightEdges(weights: CsrWeights): Uint32Array {
  const edges = new Uint32Array(weights.neighbors.length * 2);
  for (let row = 0; row < weights.rowCount; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      edges[slot * 2] = row;
      edges[slot * 2 + 1] = weights.neighbors[slot];
    }
  }
  return edges;
}

/**
 * Points over the upper part of the world (the lower `CHART_BAND` is left for a chart), valued by
 * a smooth field.
 */
export function createPointField(
  random: Random,
  world: World,
  count = 700,
  reserveChart = true
): PointField {
  const mapHeight = reserveChart ? 1 - CHART_BAND : world.height;
  const points = createUniformPoints(random, {width: world.width, height: mapHeight}, count, 0);
  const offset = world.height - mapHeight;
  const positions = Float32Array.from(points.positions, (value, index) =>
    index % 2 === 1 ? value + offset : value
  );
  const noise = Float32Array.from({length: count}, () => random());
  return {count, worldWidth: world.width, positions, field: points.values, noise};
}

/** `field` blended with `noise`: share 0 is the smooth field, 1 is pure noise. */
export function mixNoise(field: Float32Array, noise: Float32Array, share: number): Float32Array {
  return Float32Array.from(field, (value, index) => value * (1 - share) + noise[index] * share);
}

export type SpaceTimeEvents = {
  count: number;
  worldWidth: number;
  positions: Float32Array;
  /** Event time in days (0 to 120) when none of the events are space-time clustered. */
  backgroundTimes: Float32Array;
  /** Time of the event's local cluster center, for the clustered share of events. */
  clusterTimes: Float32Array;
  /** Position of each event in [0, 1] used to pick which events are clustered. */
  order: Float32Array;
};

/**
 * Events in the upper map area: four space-time clusters (a place and a time each) plus uniform
 * background. `getEventTimes` blends the clustered and background times.
 */
export function createSpaceTimeEvents(random: Random, world: World, count = 300): SpaceTimeEvents {
  const mapHeight = 1 - CHART_BAND;
  const offset = world.height - mapHeight;
  const centers = Array.from({length: 4}, (_, index) => [
    (0.15 + 0.7 * random()) * world.width,
    offset + (0.15 + 0.7 * random()) * mapHeight,
    index * 30 + 10
  ]);
  const positions = new Float32Array(count * 2);
  const backgroundTimes = new Float32Array(count);
  const clusterTimes = new Float32Array(count);
  const order = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const [centerX, centerY, centerTime] = centers[index % 4];
    const clustered = index % 3 !== 2;
    const spread = clustered ? 0.04 : 1;
    positions[index * 2] = clustered
      ? Math.min(Math.max(centerX + (random() - 0.5) * 2 * spread, 0), world.width)
      : random() * world.width;
    positions[index * 2 + 1] = clustered
      ? Math.min(Math.max(centerY + (random() - 0.5) * 2 * spread, offset), world.height)
      : offset + random() * mapHeight;
    backgroundTimes[index] = random() * 120;
    clusterTimes[index] = centerTime + (random() - 0.5) * 8;
    order[index] = random();
  }
  return {count, worldWidth: world.width, positions, backgroundTimes, clusterTimes, order};
}

/** Event times where a `clustering` share of the events keep their cluster's time. */
export function getEventTimes(events: SpaceTimeEvents, clustering: number): Float32Array {
  return Float32Array.from(events.order, (position, index) =>
    position < clustering ? events.clusterTimes[index] : events.backgroundTimes[index]
  );
}
