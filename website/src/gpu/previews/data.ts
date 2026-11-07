// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded synthetic data helpers. Everything is deterministic for a given seed.

import {SCENE_HEIGHT, SCENE_WIDTH, type Point} from './types.ts';

export const FIELD_COLUMNS = 128;
export const FIELD_ROWS = 80;

export type Random = {
  next(): number;
  range(minimum: number, maximum: number): number;
  gaussian(): number;
  integer(count: number): number;
};

/** Mulberry32 pseudo random generator. */
export function createRandom(seed: number): Random {
  let state = (seed >>> 0) + 0x6d2b79f5;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    range: (minimum, maximum) => minimum + (maximum - minimum) * next(),
    gaussian: () => {
      const first = Math.max(next(), 1e-9);
      return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * next());
    },
    integer: count => Math.floor(next() * count)
  };
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function mixNumbers(first: number, second: number, amount: number): number {
  return first + (second - first) * amount;
}

export function getDistance(first: Point, second: Point): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

/** Gaussian blobs of points around the given centers. */
export function createClusterPoints(
  random: Random,
  centers: Point[],
  countPerCenter: number,
  spread: number
): Point[] {
  const points: Point[] = [];
  for (const center of centers) {
    for (let index = 0; index < countPerCenter; index++) {
      points.push([
        clamp(center[0] + random.gaussian() * spread, 6, SCENE_WIDTH - 6),
        clamp(center[1] + random.gaussian() * spread * 0.8, 6, SCENE_HEIGHT - 6)
      ]);
    }
  }
  return points;
}

export function createScatterPoints(random: Random, count: number, margin = 8): Point[] {
  return Array.from({length: count}, () => [
    random.range(margin, SCENE_WIDTH - margin),
    random.range(margin, SCENE_HEIGHT - margin)
  ]);
}

/** Smooth random field in 0..1 built from random plane waves. */
export function createSmoothField(
  random: Random,
  waveCount = 7,
  baseFrequency = 1.6,
  columns = FIELD_COLUMNS,
  rows = FIELD_ROWS
): Float32Array {
  const waves = Array.from({length: waveCount}, (_, index) => {
    const angle = random.range(0, Math.PI * 2);
    const frequency = baseFrequency * (1 + index * 0.45) * random.range(0.8, 1.2);
    return {
      kx: Math.cos(angle) * frequency,
      ky: Math.sin(angle) * frequency,
      phase: random.range(0, Math.PI * 2),
      amplitude: 1 / (1 + index * 0.7)
    };
  });
  return normalizeValues(
    evaluateField(columns, rows, (u, v) => {
      let sum = 0;
      for (const wave of waves) {
        sum +=
          wave.amplitude * Math.sin((wave.kx * u + wave.ky * v * 0.625) * Math.PI * 2 + wave.phase);
      }
      return sum;
    })
  );
}

export function evaluateField(
  columns: number,
  rows: number,
  evaluate: (u: number, v: number, column: number, row: number) => number
): Float32Array {
  const values = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      values[row * columns + column] = evaluate(
        (column + 0.5) / columns,
        (row + 0.5) / rows,
        column,
        row
      );
    }
  }
  return values;
}

export function normalizeValues(values: Float32Array): Float32Array {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  const range = maximum - minimum || 1;
  return values.map(value => (value - minimum) / range);
}

/** Gaussian kernel density of points on the field grid. */
export function createDensityField(points: Point[], bandwidth: number): Float32Array {
  const scale = 1 / (2 * bandwidth * bandwidth);
  return normalizeValues(
    evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
      const x = u * SCENE_WIDTH;
      const y = v * SCENE_HEIGHT;
      let sum = 0;
      for (const point of points) {
        const dx = point[0] - x;
        const dy = point[1] - y;
        sum += Math.exp(-(dx * dx + dy * dy) * scale);
      }
      return sum;
    })
  );
}

export function sampleField(
  values: Float32Array,
  columns: number,
  rows: number,
  u: number,
  v: number
): number {
  const x = clamp(u * columns - 0.5, 0, columns - 1);
  const y = clamp(v * rows - 0.5, 0, rows - 1);
  const column = Math.min(Math.floor(x), columns - 2);
  const row = Math.min(Math.floor(y), rows - 2);
  const fx = x - column;
  const fy = y - row;
  const at = (c: number, r: number) => values[r * columns + c];
  return mixNumbers(
    mixNumbers(at(column, row), at(column + 1, row), fx),
    mixNumbers(at(column, row + 1), at(column + 1, row + 1), fx),
    fy
  );
}

/** Star-shaped blob around a center. */
export function createBlob(
  random: Random,
  centerX: number,
  centerY: number,
  radius: number,
  vertexCount = 14,
  roughness = 0.28,
  squash = 1
): Point[] {
  const phase = random.range(0, Math.PI * 2);
  const lobes = [random.range(0, 6), random.range(0, 6)];
  return Array.from({length: vertexCount}, (_, index) => {
    const angle = (index / vertexCount) * Math.PI * 2;
    const wobble =
      1 +
      roughness * 0.6 * Math.sin(angle * 2 + phase + lobes[0]) +
      roughness * 0.4 * Math.sin(angle * 3 + lobes[1]) +
      roughness * 0.3 * (random.next() - 0.5);
    return [
      centerX + Math.cos(angle) * radius * wobble,
      centerY + Math.sin(angle) * radius * wobble * squash
    ];
  });
}

export function createCircle(
  centerX: number,
  centerY: number,
  radius: number,
  segments = 40
): Point[] {
  return Array.from({length: segments}, (_, index) => {
    const angle = (index / segments) * Math.PI * 2;
    return [centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius];
  });
}

export function createRegularPolygon(
  centerX: number,
  centerY: number,
  radius: number,
  sides: number,
  rotation = 0
): Point[] {
  return Array.from({length: sides}, (_, index) => {
    const angle = rotation + (index / sides) * Math.PI * 2;
    return [centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius];
  });
}

/** Quadratic arc from one point to another, bent sideways by `bend` (fraction of length). */
export function createArc(from: Point, to: Point, bend: number, segments = 28): Point[] {
  const midX = (from[0] + to[0]) / 2 - (to[1] - from[1]) * bend;
  const midY = (from[1] + to[1]) / 2 + (to[0] - from[0]) * bend;
  return Array.from({length: segments + 1}, (_, index) => {
    const t = index / segments;
    const inverse = 1 - t;
    return [
      inverse * inverse * from[0] + 2 * inverse * t * midX + t * t * to[0],
      inverse * inverse * from[1] + 2 * inverse * t * midY + t * t * to[1]
    ];
  });
}

/** Smooth wandering path through the box. */
export function createWanderPath(
  random: Random,
  start: Point,
  heading: number,
  stepLength: number,
  steps: number,
  turn = 0.35
): Point[] {
  const points: Point[] = [start];
  let angle = heading;
  let drift = 0;
  for (let index = 0; index < steps; index++) {
    drift = drift * 0.8 + random.gaussian() * turn * 0.4;
    angle += drift;
    const last = points[points.length - 1];
    points.push([last[0] + Math.cos(angle) * stepLength, last[1] + Math.sin(angle) * stepLength]);
  }
  return points;
}

export type RoadNetwork = {
  nodes: Point[];
  edges: [number, number][];
  columns: number;
  rows: number;
};

/** Jittered grid road network with a few missing links. */
export function createRoadNetwork(random: Random, columns = 9, rows = 6): RoadNetwork {
  const nodes: Point[] = [];
  const stepX = (SCENE_WIDTH - 24) / (columns - 1);
  const stepY = (SCENE_HEIGHT - 24) / (rows - 1);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      nodes.push([
        12 + column * stepX + random.range(-stepX, stepX) * 0.22,
        12 + row * stepY + random.range(-stepY, stepY) * 0.22
      ]);
    }
  }
  const edges: [number, number][] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (column < columns - 1 && random.next() > 0.1) edges.push([index, index + 1]);
      if (row < rows - 1 && random.next() > 0.14) edges.push([index, index + columns]);
    }
  }
  return {nodes, edges, columns, rows};
}

/** Dijkstra over the network. Returns distances and predecessor indices. */
export function findShortestPaths(network: RoadNetwork, source: number) {
  const count = network.nodes.length;
  const distances = new Array<number>(count).fill(Infinity);
  const previous = new Array<number>(count).fill(-1);
  const visited = new Array<boolean>(count).fill(false);
  const adjacency: [number, number][][] = Array.from({length: count}, () => []);
  for (const [first, second] of network.edges) {
    const length = getDistance(network.nodes[first], network.nodes[second]);
    adjacency[first].push([second, length]);
    adjacency[second].push([first, length]);
  }
  distances[source] = 0;
  for (let round = 0; round < count; round++) {
    let current = -1;
    for (let index = 0; index < count; index++) {
      if (!visited[index] && (current < 0 || distances[index] < distances[current]))
        current = index;
    }
    if (current < 0 || distances[current] === Infinity) break;
    visited[current] = true;
    for (const [neighbor, length] of adjacency[current]) {
      if (distances[current] + length < distances[neighbor]) {
        distances[neighbor] = distances[current] + length;
        previous[neighbor] = current;
      }
    }
  }
  return {distances, previous};
}

export function getPathTo(previous: number[], target: number): number[] {
  const path: number[] = [];
  for (let node = target; node >= 0; node = previous[node]) path.unshift(node);
  return path;
}
