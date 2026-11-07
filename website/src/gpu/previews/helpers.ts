// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared building blocks for preview builders: the build context and small geometry and layer
// helpers. Used by the per-kind builders (builders.ts) and the per-entry builders (entries/*.ts).

import {clamp, getDistance, type Random, type RoadNetwork} from './data.ts';
import type {GpuSceneParameter} from '../types.ts';
import type {PreviewParameterValues} from './parameters.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type PathItem,
  type Point,
  type PreviewLayer,
  type Role,
  type SpriteItem
} from './types.ts';

export type BuildContext = {
  random: Random;
  /** An independent stream per label, so a parameter only changes the data it controls. */
  fork(label: string): Random;
  /** Parameter values, defaults filled in. */
  p: PreviewParameterValues;
  /** Pointer in scene units, when the kind reacts to it and the pointer is over the canvas. */
  pointer: Point | null;
};

export type Builder = (context: BuildContext) => PreviewLayer[];

export function createGridLayer(): PreviewLayer {
  const items: PathItem[] = [];
  for (let x = 20; x < SCENE_WIDTH; x += 20) {
    items.push({
      points: [
        [x, 0],
        [x, SCENE_HEIGHT]
      ],
      role: 'dim',
      width: 0.18,
      alpha: 0.14
    });
  }
  for (let y = 20; y < SCENE_HEIGHT; y += 20) {
    items.push({
      points: [
        [0, y],
        [SCENE_WIDTH, y]
      ],
      role: 'dim',
      width: 0.18,
      alpha: 0.14
    });
  }
  return {type: 'strokes', items, animation: 'static'};
}

export function toSprites(
  points: Point[],
  role: Role,
  radius: number,
  random?: Random,
  alpha = 0.9
): SpriteItem[] {
  return points.map(([x, y]) => ({
    x,
    y,
    radius: radius * (random ? random.range(0.7, 1.3) : 1),
    role,
    alpha,
    phase: random ? random.next() : 0
  }));
}

export function toPaths(
  paths: Point[][],
  role: Role,
  width: number,
  alpha = 0.9,
  random?: Random
): PathItem[] {
  return paths.map(points => ({points, role, width, alpha, phase: random ? random.next() : 0}));
}

export function clampPoint([x, y]: Point, margin = 6): Point {
  return [clamp(x, margin, SCENE_WIDTH - margin), clamp(y, margin, SCENE_HEIGHT - margin)];
}

export const CLUSTER_CENTERS: Point[] = [
  [42, 36],
  [100, 62],
  [122, 28],
  [60, 74]
];

export function getClusterCenters(random: Random): Point[] {
  return CLUSTER_CENTERS.map(([x, y]) => [x + random.range(-8, 8), y + random.range(-6, 6)]);
}

export function getRoleCycle(index: number): Role {
  return (['primary', 'warm', 'hot', 'cool', 'violet'] as const)[index % 5];
}

export function buildNetworkLayers(
  network: RoadNetwork,
  options: {
    closed?: Set<number>;
    source?: number;
    hops?: number;
    pulse?: number;
  } = {}
): PreviewLayer[] {
  const {closed = new Set<number>(), source = -1, hops = -1, pulse = 0.5} = options;
  const open = network.edges.filter((_, index) => !closed.has(index));
  const reach = hops >= 0 ? getHopCounts(network, open, source) : [];
  const isReached = (node: number) => reach[node] !== undefined && reach[node] <= hops;
  const quiet: Point[][] = [];
  const active: Point[][] = [];
  for (const [first, second] of open) {
    const segment = [network.nodes[first], network.nodes[second]];
    if (hops >= 0 && isReached(first) && isReached(second)) active.push(segment);
    else quiet.push(segment);
  }
  const center = network.nodes[Math.floor(network.nodes.length / 2)];
  const nodes = network.nodes.map(
    ([x, y], index): SpriteItem =>
      hops >= 0 && isReached(index)
        ? {
            x,
            y,
            radius: index === source ? 4.2 : 2.3,
            role: 'warm',
            alpha: 0.95 - (reach[index] / (hops + 1)) * 0.5,
            phase: reach[index] / (hops + 1)
          }
        : {
            x,
            y,
            radius: 1.6,
            role: 'primary',
            alpha: hops >= 0 ? 0.55 : 0.9,
            phase: clamp(getDistance([x, y], center) / 90, 0, 1)
          }
  );
  const layers: PreviewLayer[] = [
    {type: 'strokes', items: toPaths(quiet, 'dim', 0.5, 0.55), animation: 'static'}
  ];
  if (closed.size > 0) {
    layers.push({
      type: 'strokes',
      items: toPaths(
        [...closed].map(index => [
          network.nodes[network.edges[index][0]],
          network.nodes[network.edges[index][1]]
        ]),
        'hot',
        0.7,
        0.45
      ),
      animation: 'pulse',
      speed: 0.4,
      blend: 'add'
    });
  }
  if (active.length > 0) {
    layers.push({
      type: 'strokes',
      items: toPaths(active, 'warm', 1.1, 0.9),
      animation: 'flow',
      speed: 0.3,
      blend: 'add'
    });
  }
  layers.push({type: 'sprites', items: nodes, pulse});
  return layers;
}

/** Breadth first hop count from a node; unreachable nodes stay undefined. */
export function getHopCounts(
  network: RoadNetwork,
  edges: [number, number][],
  source: number
): number[] {
  const adjacency: number[][] = Array.from({length: network.nodes.length}, () => []);
  for (const [first, second] of edges) {
    adjacency[first].push(second);
    adjacency[second].push(first);
  }
  const counts: number[] = [];
  counts[source] = 0;
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    for (const neighbor of adjacency[queue[head]]) {
      if (counts[neighbor] !== undefined) continue;
      counts[neighbor] = counts[queue[head]] + 1;
      queue.push(neighbor);
    }
  }
  return counts;
}

export function findNearestIndex(points: Point[], target: Point): number {
  let best = 0;
  let bestDistance = Infinity;
  points.forEach((point, index) => {
    const distance = getDistance(point, target);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  });
  return best;
}

/** Edge indices to close: a stable prefix of a seeded shuffle, so sliding adds one at a time. */
export function pickClosedEdges(random: Random, edgeCount: number, closures: number): Set<number> {
  const order = Array.from({length: edgeCount}, (_, index) => ({index, key: random.next()}))
    .sort((first, second) => first.key - second.key)
    .map(entry => entry.index);
  return new Set(order.slice(0, Math.round(closures)));
}

/** Chaikin corner cutting; keeps the ends, so wide ribbons do not spike at sharp corners. */
export function smoothPath(points: Point[], iterations: number): Point[] {
  let current = points;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const next: Point[] = [current[0]];
    for (let index = 0; index + 1 < current.length; index++) {
      const [ax, ay] = current[index];
      const [bx, by] = current[index + 1];
      next.push([ax * 0.75 + bx * 0.25, ay * 0.75 + by * 0.25]);
      next.push([ax * 0.25 + bx * 0.75, ay * 0.25 + by * 0.75]);
    }
    next.push(current[current.length - 1]);
    current = next;
  }
  return current;
}

export const BUFFER_FIELD_RANGE = 40;

export function getSegmentDistance(point: Point, start: Point, end: Point): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy || 1;
  const amount = clamp(
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared,
    0,
    1
  );
  return Math.hypot(point[0] - start[0] - dx * amount, point[1] - start[1] - dy * amount);
}

export const CLASS_ROLES: Role[] = ['cool', 'primary', 'violet', 'warm', 'hot'];

export function getClassRole(bin: number, classes: number): Role {
  return CLASS_ROLES[classes <= 1 ? 0 : Math.round((bin / (classes - 1)) * 4)];
}

/** Compass azimuth in degrees (0 north, clockwise) to the field shader's light angle (y down). */
export function getLightAngle(azimuthDegrees: number): number {
  const azimuth = (azimuthDegrees * Math.PI) / 180;
  return Math.atan2(-Math.cos(azimuth), Math.sin(azimuth));
}

export function getTrailingMean(values: number[], windowSize: number): number[] {
  return values.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1);
    let sum = 0;
    for (let cursor = start; cursor <= index; cursor++) sum += values[cursor];
    return sum / (index - start + 1);
  });
}

/**
 * A function-specific preview for one catalog entry. It replaces the generic scene of the entry's
 * preview kind on cards and on the function page, in both the live canvas and the SVG fallback.
 */
export type EntryPreview = {
  build: Builder;
  /**
   * Slider parameters read by `build` through `p`. Omit to keep the preview kind's parameters.
   * Defaults are what cards and the static fallback show.
   */
  parameters?: GpuSceneParameter[];
  /**
   * How the picture reacts to the pointer (scene units in `pointer`): 'hover' follows it, 'sticky'
   * keeps the last position after it leaves. Omit to follow the preview kind.
   */
  pointer?: 'none' | 'hover' | 'sticky';
};
