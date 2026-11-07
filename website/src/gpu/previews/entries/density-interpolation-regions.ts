// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/density-interpolation-regions.ts, keyed by entry id.

import {
  clamp,
  createBlob,
  createCircle,
  createClusterPoints,
  createScatterPoints,
  createSmoothField,
  createWanderPath,
  evaluateField,
  getDistance,
  sampleField,
  type Random
} from '../data.ts';
import {
  getClassRole,
  getClusterCenters,
  getRoleCycle,
  smoothPath,
  toPaths,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewLayer,
  type Role,
  type SpriteItem
} from '../types.ts';

const WIDTH = SCENE_WIDTH;
const HEIGHT = SCENE_HEIGHT;

// ---------------------------------------------------------------------------------------------
// Local helpers

function getValueRole(value: number): Role {
  return getClassRole(Math.round(clamp(value, 0, 1) * 4), 5);
}

function createRectRing(x0: number, y0: number, x1: number, y1: number): Point[] {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1]
  ];
}

function isInsideRing([x, y]: Point, ring: Point[]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[index];
    const [bx, by] = ring[previous];
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
  }
  return inside;
}

function getRingBounds(ring: Point[]): [number, number, number, number] {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of ring) {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  }
  return [x0, y0, x1, y1];
}

function getRingCenter(ring: Point[]): Point {
  const total = ring.reduce<Point>((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0]);
  return [total[0] / ring.length, total[1] / ring.length];
}

function getStaticField(
  columns: number,
  rows: number,
  values: Float32Array,
  ramp: 'heat' | 'ice' | 'terrain' | 'spectral',
  options: Partial<Extract<PreviewLayer, {type: 'field'}>> = {}
): PreviewLayer {
  return {type: 'field', columns, rows, values, ramp, style: 'ramp', strength: 0.05, ...options};
}

/** Points spread out by best-candidate sampling. */
function createSpreadPoints(random: Random, count: number, margin = 10): Point[] {
  const points: Point[] = [];
  for (let index = 0; index < count; index++) {
    let best: Point = [0, 0];
    let bestDistance = -1;
    for (let attempt = 0; attempt < 8; attempt++) {
      const candidate: Point = [
        random.range(margin, WIDTH - margin),
        random.range(margin, HEIGHT - margin)
      ];
      const distance = points.reduce(
        (minimum, point) => Math.min(minimum, getDistance(point, candidate)),
        1e9
      );
      if (distance > bestDistance) {
        bestDistance = distance;
        best = candidate;
      }
    }
    points.push(best);
  }
  return points;
}

type Sample = {x: number; y: number; value: number};

function createSamples(random: Random, count: number, margin = 10): Sample[] {
  const field = createSmoothField(random, 6, 1.1, 32, 20);
  return createSpreadPoints(random, count, margin).map(([x, y]) => ({
    x,
    y,
    value: sampleField(field, 32, 20, x / WIDTH, y / HEIGHT)
  }));
}

function invertMatrix(matrix: number[][]): number[][] {
  const size = matrix.length;
  const work = matrix.map((row, index) => [
    ...row,
    ...Array.from({length: size}, (_, column) => (column === index ? 1 : 0))
  ]);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(work[row][column]) > Math.abs(work[pivot][column])) pivot = row;
    }
    [work[column], work[pivot]] = [work[pivot], work[column]];
    const divisor = work[column][column] || 1e-9;
    for (let index = 0; index < size * 2; index++) work[column][index] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = work[row][column];
      if (factor === 0) continue;
      for (let index = 0; index < size * 2; index++) {
        work[row][index] -= factor * work[column][index];
      }
    }
  }
  return work.map(row => row.slice(size));
}

/** Simple kriging with an exponential covariance: prediction and normalized variance at a point. */
function createKriging(samples: Sample[], range: number, nugget = 0.03) {
  const covariance = (distance: number) => Math.exp(-distance / range);
  const matrix = samples.map((first, row) =>
    samples.map((second, column) => {
      const distance = Math.hypot(first.x - second.x, first.y - second.y);
      return covariance(distance) + (row === column ? nugget : 0);
    })
  );
  const inverse = invertMatrix(matrix);
  const mean = samples.reduce((sum, sample) => sum + sample.value, 0) / samples.length;
  const vector = new Float64Array(samples.length);
  return (x: number, y: number) => {
    for (let index = 0; index < samples.length; index++) {
      vector[index] = covariance(Math.hypot(samples[index].x - x, samples[index].y - y));
    }
    let prediction = mean;
    let reduction = 0;
    for (let row = 0; row < samples.length; row++) {
      let weight = 0;
      for (let column = 0; column < samples.length; column++) {
        weight += inverse[row][column] * vector[column];
      }
      prediction += weight * (samples[row].value - mean);
      reduction += weight * vector[row];
    }
    return {prediction, variance: clamp(1 - reduction, 0, 1)};
  };
}

function createSampleSprites(samples: Sample[], radius = 2): SpriteItem[] {
  return samples.map(sample => ({
    x: sample.x,
    y: sample.y,
    radius,
    role: getValueRole(sample.value),
    alpha: 1,
    phase: sample.value
  }));
}

type Tracts = {
  columns: number;
  rows: number;
  corners: Point[][];
  rings: Point[][];
  centers: Point[];
  values: number[];
  /** Rook (edge) neighbors. */
  neighbors: number[][];
};

/** A jittered lattice of contiguous tracts with a smooth attribute value each. */
function createTracts(random: Random, columns: number, rows: number, jitter = 0.28): Tracts {
  const cellWidth = (WIDTH - 16) / columns;
  const cellHeight = (HEIGHT - 12) / rows;
  const corners: Point[][] = [];
  for (let row = 0; row <= rows; row++) {
    const line: Point[] = [];
    for (let column = 0; column <= columns; column++) {
      const jitterX = column > 0 && column < columns ? random.range(-1, 1) * cellWidth * jitter : 0;
      const jitterY = row > 0 && row < rows ? random.range(-1, 1) * cellHeight * jitter : 0;
      line.push([8 + column * cellWidth + jitterX, 6 + row * cellHeight + jitterY]);
    }
    corners.push(line);
  }
  const rings: Point[][] = [];
  const neighbors: number[][] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      rings.push([
        corners[row][column],
        corners[row][column + 1],
        corners[row + 1][column + 1],
        corners[row + 1][column]
      ]);
      const near: number[] = [];
      if (column > 0) near.push(row * columns + column - 1);
      if (column < columns - 1) near.push(row * columns + column + 1);
      if (row > 0) near.push((row - 1) * columns + column);
      if (row < rows - 1) near.push((row + 1) * columns + column);
      neighbors.push(near);
    }
  }
  const field = createSmoothField(random, 5, 1.1, columns, rows);
  return {
    columns,
    rows,
    corners,
    rings,
    centers: rings.map(getRingCenter),
    values: Array.from(field),
    neighbors
  };
}

function getBorderEdges(tracts: Tracts, assignment: number[]): Point[][] {
  const edges: Point[][] = [];
  const {columns, rows, corners} = tracts;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (column < columns - 1 && assignment[index] !== assignment[index + 1]) {
        edges.push([corners[row][column + 1], corners[row + 1][column + 1]]);
      }
      if (row < rows - 1 && assignment[index] !== assignment[index + columns]) {
        edges.push([corners[row + 1][column], corners[row + 1][column + 1]]);
      }
    }
  }
  return edges;
}

function buildRegionLayers(
  tracts: Tracts,
  assignment: number[],
  options: {alpha?: number; animation?: 'static' | 'pulse'} = {}
): PreviewLayer[] {
  const {alpha = 0.5, animation = 'static'} = options;
  return [
    {
      type: 'fills',
      items: tracts.rings.map(
        (ring, index): PolygonItem => ({
          ring,
          role: getRoleCycle(assignment[index]),
          alpha,
          phase: (index % 7) / 7
        })
      ),
      animation,
      blend: 'add'
    },
    {
      type: 'strokes',
      items: toPaths(
        tracts.rings.map(ring => [...ring, ring[0]]),
        'dim',
        0.3,
        0.3
      )
    },
    {
      type: 'strokes',
      items: toPaths(getBorderEdges(tracts, assignment), 'ink', 1.1, 0.9),
      animation: 'static',
      blend: 'add'
    }
  ];
}

function getTractAssignmentConnected(
  tracts: Tracts,
  assignment: number[],
  region: number
): boolean {
  const members = assignment.flatMap((value, index) => (value === region ? [index] : []));
  if (members.length <= 1) return true;
  const seen = new Set<number>([members[0]]);
  const queue = [members[0]];
  for (let head = 0; head < queue.length; head++) {
    for (const neighbor of tracts.neighbors[queue[head]]) {
      if (assignment[neighbor] === region && !seen.has(neighbor)) {
        seen.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  return seen.size === members.length;
}

function getRegionSums(values: number[], assignment: number[], count: number) {
  const sums = new Array<number>(count).fill(0);
  const sizes = new Array<number>(count).fill(0);
  assignment.forEach((region, index) => {
    sums[region] += values[index];
    sizes[region] += 1;
  });
  return {sums, sizes};
}

function getTotalVariance(values: number[], assignment: number[], count: number): number {
  const {sums, sizes} = getRegionSums(values, assignment, count);
  return values.reduce((total, value, index) => {
    const mean = sums[assignment[index]] / sizes[assignment[index]];
    return total + (value - mean) ** 2;
  }, 0);
}

/** Row-normalized transition matrix with persistence on the diagonal and drift toward higher classes. */
function createTransitionMatrix(size: number, persistence: number, drift = 0): number[][] {
  return Array.from({length: size}, (_, row) => {
    const weights = Array.from({length: size}, (_, column) => {
      if (column === row) return persistence;
      const distance = Math.abs(column - row);
      const direction = column > row ? 1 : -1;
      return ((1 - persistence) / distance ** 2) * (1 + drift * direction * 3);
    });
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return weights.map(weight => weight / total);
  });
}

const FIELD_COLUMNS_LOW = 64;
const FIELD_ROWS_LOW = 40;

function evaluateSceneField(
  columns: number,
  rows: number,
  evaluate: (x: number, y: number, column: number, row: number) => number
): Float32Array {
  return evaluateField(columns, rows, (u, v, column, row) =>
    evaluate(u * WIDTH, v * HEIGHT, column, row)
  );
}

export const DENSITY_INTERPOLATION_REGIONS_PREVIEWS: Record<string, EntryPreview> = {
  // ------------------------------------------------------------------------- density surfaces
  'point-density': {
    parameters: [
      {id: 'cell', label: 'Cell size', min: 8, max: 24, step: 1, default: 12, unit: 'px'},
      {
        id: 'binning',
        label: 'Binning',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['grid', 'hexagon']
      }
    ],
    build: ({fork, p}) => {
      const random = fork('points');
      const points = [
        ...createClusterPoints(random, getClusterCenters(random), 36, 9),
        ...createScatterPoints(random, 24)
      ];
      const cell = p.cell;
      const hexagon = p.binning >= 1;
      const rowHeight = cell * 0.866;
      const bins = new Map<string, {center: Point; count: number}>();
      for (const [x, y] of points) {
        let center: Point;
        if (hexagon) {
          const baseRow = Math.round(y / rowHeight);
          let best: Point = [0, 0];
          let bestDistance = Infinity;
          for (let row = baseRow - 1; row <= baseRow + 1; row++) {
            const offset = (row & 1) * cell * 0.5;
            const column = Math.round((x - offset) / cell);
            const candidate: Point = [column * cell + offset, row * rowHeight];
            const distance = getDistance(candidate, [x, y]);
            if (distance < bestDistance) {
              bestDistance = distance;
              best = candidate;
            }
          }
          center = best;
        } else {
          center = [(Math.floor(x / cell) + 0.5) * cell, (Math.floor(y / cell) + 0.5) * cell];
        }
        const key = `${center[0].toFixed(1)},${center[1].toFixed(1)}`;
        const bin = bins.get(key) ?? {center, count: 0};
        bin.count += 1;
        bins.set(key, bin);
      }
      const maximum = Math.max(...[...bins.values()].map(bin => bin.count));
      const items: PolygonItem[] = [...bins.values()].map(({center, count}, index) => {
        const amount = count / maximum;
        const ring = hexagon
          ? createCircle(center[0], center[1], (cell / Math.sqrt(3)) * 0.96, 6).map(
              ([x, y]): Point => [
                center[0] +
                  (x - center[0]) * Math.cos(Math.PI / 6) -
                  (y - center[1]) * Math.sin(Math.PI / 6),
                center[1] +
                  (x - center[0]) * Math.sin(Math.PI / 6) +
                  (y - center[1]) * Math.cos(Math.PI / 6)
              ]
            )
          : createRectRing(
              center[0] - cell / 2 + 0.4,
              center[1] - cell / 2 + 0.4,
              center[0] + cell / 2 - 0.4,
              center[1] + cell / 2 - 0.4
            );
        return {
          ring,
          role: getValueRole(amount),
          alpha: 0.2 + amount * 0.7,
          phase: (index % 9) / 9
        };
      });
      return [
        {type: 'fills', items, animation: 'pulse', blend: 'add'},
        {type: 'sprites', items: toSprites(points, 'ink', 0.6, random, 0.55)}
      ];
    }
  },

  'kernel-density': {
    parameters: [
      {id: 'bandwidth', label: 'Bandwidth', min: 3, max: 22, step: 0.5, default: 10, unit: 'px'},
      {
        id: 'kernel',
        label: 'Kernel',
        min: 0,
        max: 2,
        step: 1,
        default: 0,
        options: ['gaussian', 'quartic', 'epanechnikov']
      },
      {id: 'events', label: 'Events', min: 8, max: 70, step: 2, default: 34}
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('events');
      const centers = getClusterCenters(random);
      const events = createClusterPoints(random, centers, Math.ceil(p.events / 4), 10).slice(
        0,
        p.events
      );
      const kernel = Math.round(p.kernel);
      const bandwidth = p.bandwidth;
      const support = bandwidth * 2.5;
      const weigh = (distance: number): number => {
        if (kernel === 0) return Math.exp(-(distance * distance) / (2 * bandwidth * bandwidth));
        const u = distance / support;
        if (u >= 1) return 0;
        return kernel === 1 ? (1 - u * u) ** 2 : 1 - u * u;
      };
      const values = evaluateSceneField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, (x, y) => {
        let sum = 0;
        for (const [ex, ey] of events) sum += weigh(Math.hypot(ex - x, ey - y));
        return sum;
      });
      let maximum = 1e-6;
      for (const value of values) maximum = Math.max(maximum, value);
      for (let index = 0; index < values.length; index++)
        values[index] = Math.min(1, values[index] / maximum);
      const probe = pointer ?? centers[1];
      const profile: Point[] = Array.from({length: 33}, (_, index) => {
        const distance = ((index - 16) / 16) * support;
        return [136 + (distance / support) * 17, 90 - weigh(Math.abs(distance)) * 18];
      });
      return [
        getStaticField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, values, 'heat', {strength: 0.07}),
        {type: 'sprites', items: toSprites(events, 'ink', 0.7, random, 0.45)},
        {
          type: 'strokes',
          items: toPaths([createCircle(probe[0], probe[1], support, 48)], 'warm', 0.6, 0.75),
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: [
            ...toPaths(
              [
                [
                  [117, 90],
                  [155, 90]
                ]
              ],
              'dim',
              0.4,
              0.5
            ),
            ...toPaths([profile], 'warm', 1.1, 0.95)
          ],
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [{x: probe[0], y: probe[1], radius: 2.4, role: 'warm', alpha: 1}],
          pulse: 0.3
        }
      ];
    }
  },

  'line-density': {
    parameters: [
      {id: 'lines', label: 'Lines', min: 3, max: 14, step: 1, default: 9},
      {id: 'radius', label: 'Search radius', min: 4, max: 14, step: 1, default: 8, unit: 'px'}
    ],
    build: ({fork, p}) => {
      const random = fork('roads');
      // Lines converge on a busy hub, so the density surface has a clear core and shoulders.
      const hub: Point = [86, 52];
      const roads: Point[][] = Array.from({length: p.lines}, () => {
        const angle = random.range(0, Math.PI * 2);
        const start: Point = [
          clamp(hub[0] + Math.cos(angle) * random.range(40, 70), 4, WIDTH - 4),
          clamp(hub[1] + Math.sin(angle) * random.range(30, 48), 4, HEIGHT - 4)
        ];
        const heading = Math.atan2(hub[1] - start[1], hub[0] - start[0]);
        return smoothPath(createWanderPath(random, start, heading, 8, 14, 0.35), 1).map(
          point => [clamp(point[0], 2, WIDTH - 2), clamp(point[1], 2, HEIGHT - 2)] as Point
        );
      });
      const columns = 96;
      const rows = 60;
      const cellWidth = WIDTH / columns;
      const cellHeight = HEIGHT / rows;
      const radius = p.radius;
      const density = new Float32Array(columns * rows);
      // Each line is sampled every unit; every sample splats a smooth kernel inside the radius.
      for (const road of roads) {
        for (let index = 0; index + 1 < road.length; index++) {
          const [ax, ay] = road[index];
          const [bx, by] = road[index + 1];
          const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 1.5));
          for (let step = 0; step < steps; step++) {
            const x = ax + ((bx - ax) * (step + 0.5)) / steps;
            const y = ay + ((by - ay) * (step + 0.5)) / steps;
            const column0 = Math.max(0, Math.floor((x - radius) / cellWidth));
            const column1 = Math.min(columns - 1, Math.ceil((x + radius) / cellWidth));
            const row0 = Math.max(0, Math.floor((y - radius) / cellHeight));
            const row1 = Math.min(rows - 1, Math.ceil((y + radius) / cellHeight));
            for (let row = row0; row <= row1; row++) {
              for (let column = column0; column <= column1; column++) {
                const distance = Math.hypot(
                  (column + 0.5) * cellWidth - x,
                  (row + 0.5) * cellHeight - y
                );
                if (distance >= radius) continue;
                const weight = 1 - (distance / radius) ** 2;
                density[row * columns + column] += weight * weight;
              }
            }
          }
        }
      }
      let maximum = 1e-6;
      for (const value of density) maximum = Math.max(maximum, value);
      const normalized = density.map(value => Math.pow(value / maximum, 0.7));
      // Search discs sweep along the longest line; staggered pulse phases read as a moving window.
      const sweepRoad = roads.reduce((best, road) => (road.length > best.length ? road : best));
      const discs: PathItem[] = [];
      const centers: SpriteItem[] = [];
      for (let index = 0; index < 7; index++) {
        const [x, y] = sweepRoad[Math.round((index / 6) * (sweepRoad.length - 1))];
        discs.push({
          points: [...createCircle(x, y, radius, 28), createCircle(x, y, radius, 28)[0]],
          role: 'ink',
          width: 0.5,
          alpha: 0.9,
          phase: index / 7
        });
        centers.push({x, y, radius: 1.6, role: 'ink', alpha: 0.95, phase: index / 7});
      }
      return [
        getStaticField(columns, rows, normalized, 'heat', {strength: 0.1}),
        {
          type: 'strokes',
          items: toPaths(roads, 'cool', 0.8, 0.9),
          animation: 'flow',
          speed: 0.25,
          trail: 1.5,
          blend: 'add'
        },
        {type: 'strokes', items: discs, animation: 'pulse', speed: 0.35, blend: 'add'},
        {type: 'sprites', items: centers, pulse: 0.6}
      ];
    }
  },

  'line-length-per-polygon': {
    parameters: [
      {id: 'size', label: 'Polygon size', min: 10, max: 26, step: 1, default: 18, unit: 'px'},
      {id: 'lines', label: 'Lines', min: 3, max: 14, step: 1, default: 8}
    ],
    build: ({fork, p}) => {
      const random = fork('polygons');
      const centers: Point[] = [
        [34, 32],
        [86, 28],
        [128, 40],
        [52, 72],
        [108, 74]
      ];
      const polygons = centers.map(([x, y]) => createBlob(random, x, y, p.size, 14, 0.25, 0.85));
      const roadRandom = fork('roads');
      const roads: Point[][] = Array.from({length: p.lines}, () =>
        smoothPath(
          createWanderPath(
            roadRandom,
            [roadRandom.range(6, 60), roadRandom.range(10, 90)],
            roadRandom.range(-0.5, 0.5),
            9,
            13,
            0.4
          ),
          1
        ).map(point => [clamp(point[0], 1, WIDTH - 1), clamp(point[1], 1, HEIGHT - 1)] as Point)
      );
      const lengths = new Array<number>(polygons.length).fill(0);
      const inside: Point[][] = [];
      const outside: Point[][] = [];
      for (const road of roads) {
        const dense: Point[] = [];
        for (let index = 0; index + 1 < road.length; index++) {
          const [ax, ay] = road[index];
          const [bx, by] = road[index + 1];
          const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 1.2));
          for (let step = 0; step < steps; step++) {
            const t = step / steps;
            dense.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
          }
        }
        let current: Point[] = [];
        let currentZone = -2;
        const flush = () => {
          if (current.length > 1) (currentZone >= 0 ? inside : outside).push(current);
        };
        for (const point of dense) {
          const zone = polygons.findIndex(ring => isInsideRing(point, ring));
          if (zone !== currentZone) {
            const last = current[current.length - 1];
            flush();
            current = last ? [last] : [];
            currentZone = zone;
          }
          current.push(point);
          if (zone >= 0) lengths[zone] += 1.2;
        }
        flush();
      }
      const maximum = Math.max(1, ...lengths);
      return [
        {
          type: 'fills',
          items: polygons.map(
            (ring, index): PolygonItem => ({
              ring,
              role: getValueRole(lengths[index] / maximum),
              alpha: 0.12 + (lengths[index] / maximum) * 0.45,
              phase: index / 5
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            polygons.map(ring => [...ring, ring[0]]),
            'dim',
            0.6,
            0.7
          )
        },
        {type: 'strokes', items: toPaths(outside, 'dim', 0.7, 0.55)},
        {
          type: 'strokes',
          items: toPaths(inside, 'warm', 1.3, 0.95),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        }
      ];
    }
  },

  'dot-density': {
    parameters: [
      {id: 'dots', label: 'Dots', min: 60, max: 420, step: 10, default: 220},
      {id: 'mixture', label: 'Category mix', min: 0, max: 1, step: 0.05, default: 0.6}
    ],
    build: ({fork, p}) => {
      const tractRandom = fork('tracts');
      const tracts = createTracts(tractRandom, 4, 3, 0.2);
      const mixField = createSmoothField(fork('mix'), 4, 1.0, 4, 3);
      const total = tracts.values.reduce((sum, value) => sum + 0.2 + value, 0);
      const dotRandom = fork('dots');
      const categories: SpriteItem[][] = [[], [], []];
      const roles: Role[] = ['cool', 'warm', 'violet'];
      tracts.rings.forEach((ring, index) => {
        const count = Math.round((p.dots * (0.2 + tracts.values[index])) / total);
        const [x0, y0, x1, y1] = getRingBounds(ring);
        const lean = (mixField[index] - 0.5) * p.mixture * 2;
        const shares = [0.34 - lean * 0.4, 0.33, 0.33 + lean * 0.4];
        for (let dot = 0; dot < count; dot++) {
          let placed: Point | null = null;
          for (let attempt = 0; attempt < 30 && !placed; attempt++) {
            const candidate: Point = [dotRandom.range(x0, x1), dotRandom.range(y0, y1)];
            if (isInsideRing(candidate, ring)) placed = candidate;
          }
          if (!placed) continue;
          const pick = dotRandom.next();
          const category = pick < shares[0] ? 0 : pick < shares[0] + shares[1] ? 1 : 2;
          categories[category].push({
            x: placed[0],
            y: placed[1],
            radius: 0.95,
            role: roles[category],
            alpha: 0.9,
            phase: dotRandom.next()
          });
        }
      });
      return [
        {
          type: 'strokes',
          items: toPaths(
            tracts.rings.map(ring => [...ring, ring[0]]),
            'dim',
            0.4,
            0.45
          )
        },
        ...categories.map((items): PreviewLayer => ({type: 'sprites', items, pulse: 0.3}))
      ];
    }
  },

  'random-points-in-polygon': {
    parameters: [
      {id: 'count', label: 'Points per polygon', min: 4, max: 60, step: 2, default: 24},
      {
        id: 'rejected',
        label: 'Show rejected',
        min: 0,
        max: 1,
        step: 1,
        default: 1,
        options: ['hide', 'show']
      }
    ],
    build: ({fork, p}) => {
      const random = fork('shapes');
      const shapes = [
        createBlob(random, 36, 50, 24, 12, 0.3, 0.9),
        createBlob(random, 88, 36, 20, 12, 0.35, 0.9),
        createBlob(random, 128, 70, 19, 12, 0.3, 0.9)
      ];
      const sampler = fork('sampling');
      const accepted: Point[] = [];
      const rejected: Point[] = [];
      for (const ring of shapes) {
        const [x0, y0, x1, y1] = getRingBounds(ring);
        let found = 0;
        for (let attempt = 0; found < p.count && attempt < p.count * 6; attempt++) {
          const candidate: Point = [sampler.range(x0, x1), sampler.range(y0, y1)];
          if (isInsideRing(candidate, ring)) {
            accepted.push(candidate);
            found += 1;
          } else if (rejected.length < 90) rejected.push(candidate);
        }
      }
      const layers: PreviewLayer[] = [
        {
          type: 'strokes',
          items: toPaths(
            shapes.map(ring => {
              const [x0, y0, x1, y1] = getRingBounds(ring);
              return [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0]
              ] as Point[];
            }),
            'dim',
            0.3,
            0.35
          )
        },
        {
          type: 'fills',
          items: shapes.map((ring): PolygonItem => ({ring, role: 'primary', alpha: 0.1})),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            shapes.map(ring => [...ring, ring[0]]),
            'primary',
            0.8,
            0.85
          ),
          animation: 'pulse',
          speed: 0.3,
          blend: 'add'
        }
      ];
      if (p.rejected >= 1)
        layers.push({type: 'sprites', items: toSprites(rejected, 'ink', 0.8, sampler, 0.45)});
      layers.push({
        type: 'sprites',
        items: toSprites(accepted, 'warm', 1.3, sampler, 1),
        pulse: 0.4
      });
      return layers;
    }
  },

  'random-points-on-line': {
    parameters: [
      {id: 'count', label: 'Points per line', min: 2, max: 30, step: 1, default: 10},
      {id: 'lines', label: 'Lines', min: 1, max: 5, step: 1, default: 4}
    ],
    build: ({fork, p}) => {
      const random = fork('lines');
      const starts: Point[] = [
        [10, 22],
        [14, 52],
        [10, 78],
        [60, 12],
        [70, 88]
      ];
      const roads: Point[][] = starts
        .slice(0, p.lines)
        .map((start, index) =>
          smoothPath(
            createWanderPath(random, start, index % 2 ? -0.15 : 0.12, 9 + index * 1.5, 14, 0.5),
            2
          ).map(point => [clamp(point[0], 3, WIDTH - 3), clamp(point[1], 3, HEIGHT - 3)] as Point)
        );
      const sampler = fork('positions');
      const sprites: SpriteItem[] = [];
      const ticks: PathItem[] = [];
      roads.forEach((road, index) => {
        const cumulative = [0];
        for (let i = 1; i < road.length; i++)
          cumulative.push(cumulative[i - 1] + getDistance(road[i - 1], road[i]));
        const length = cumulative[cumulative.length - 1];
        const targets = Array.from({length: p.count}, () => sampler.next() * length);
        for (const target of targets) {
          let segment = 1;
          while (segment < cumulative.length - 1 && cumulative[segment] < target) segment++;
          const t =
            (target - cumulative[segment - 1]) /
            (cumulative[segment] - cumulative[segment - 1] || 1);
          const [ax, ay] = road[segment - 1];
          const [bx, by] = road[segment];
          const x = ax + (bx - ax) * t;
          const y = ay + (by - ay) * t;
          const normal: Point = [-(by - ay), bx - ax];
          const size = Math.hypot(normal[0], normal[1]) || 1;
          ticks.push({
            points: [
              [x - (normal[0] / size) * 2.2, y - (normal[1] / size) * 2.2],
              [x + (normal[0] / size) * 2.2, y + (normal[1] / size) * 2.2]
            ],
            role: 'warm',
            width: 0.4,
            alpha: 0.5
          });
          sprites.push({
            x,
            y,
            radius: 1.5,
            role: index % 2 ? 'hot' : 'warm',
            alpha: 1,
            phase: sampler.next()
          });
        }
      });
      return [
        {
          type: 'strokes',
          items: toPaths(roads, 'cool', 1, 0.7),
          animation: 'flow',
          speed: 0.18,
          blend: 'add'
        },
        {type: 'strokes', items: ticks, animation: 'static', blend: 'add'},
        {type: 'sprites', items: sprites, pulse: 0.4}
      ];
    }
  },

  // ------------------------------------------------------------------------- interpolation
  'inverse-distance-weighting': {
    parameters: [
      {id: 'power', label: 'Power', min: 1, max: 6, step: 0.1, default: 2.5},
      {id: 'samples', label: 'Samples', min: 5, max: 22, step: 1, default: 12}
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const samples = createSamples(fork('samples'), p.samples);
      const power = p.power;
      const weigh = (distance: number) => 1 / (distance * distance + 1e-3) ** (power / 2);
      const values = evaluateSceneField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, (x, y) => {
        let sum = 0;
        let total = 0;
        for (const sample of samples) {
          const weight = weigh(Math.hypot(sample.x - x, sample.y - y));
          sum += weight * sample.value;
          total += weight;
        }
        return sum / total;
      });
      const probe = pointer ?? ([88, 46] as Point);
      const weights = samples.map(sample =>
        weigh(Math.hypot(sample.x - probe[0], sample.y - probe[1]))
      );
      const total = weights.reduce((sum, weight) => sum + weight, 0);
      const spokes: PathItem[] = samples.map((sample, index) => ({
        points: [probe, [sample.x, sample.y]],
        role: 'warm',
        width: 0.3 + (weights[index] / total) * 3.2,
        alpha: 0.25 + (weights[index] / total) * 0.7
      }));
      return [
        {
          ...getStaticField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, values, 'heat', {
            style: 'contours',
            levels: 14,
            strength: 0.04
          })
        },
        {type: 'strokes', items: spokes, animation: 'flow', speed: 0.3, blend: 'add'},
        {type: 'sprites', items: createSampleSprites(samples, 2.2), pulse: 0.2},
        {
          type: 'sprites',
          items: [{x: probe[0], y: probe[1], radius: 2.6, role: 'ink', alpha: 1}],
          pulse: 0.3
        }
      ];
    }
  },

  kriging: {
    parameters: [
      {id: 'range', label: 'Variogram range', min: 12, max: 90, step: 2, default: 40, unit: 'px'},
      {id: 'samples', label: 'Samples', min: 6, max: 20, step: 1, default: 12}
    ],
    build: ({fork, p}) => {
      const samples = createSamples(fork('samples'), p.samples);
      const krige = createKriging(samples, p.range);
      const values = evaluateSceneField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, (x, y) =>
        clamp(krige(x, y).prediction, 0, 1)
      );
      const candidates: {x: number; y: number; uncertainty: number}[] = [];
      for (let row = 0; row < 10; row++) {
        for (let column = 0; column < 16; column++) {
          const x = 8 + column * 9.6;
          const y = 8 + row * 9.4;
          candidates.push({x, y, uncertainty: krige(x, y).variance});
        }
      }
      const variance: SpriteItem[] = [];
      for (const candidate of candidates.sort(
        (first, second) => second.uncertainty - first.uncertainty
      )) {
        if (variance.length >= 7 || candidate.uncertainty < 0.25) break;
        if (variance.some(item => Math.hypot(item.x - candidate.x, item.y - candidate.y) < 22))
          continue;
        variance.push({
          x: candidate.x,
          y: candidate.y,
          radius: 5 + candidate.uncertainty * 9,
          role: 'violet',
          alpha: 0.28,
          phase: variance.length / 7
        });
      }
      return [
        getStaticField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, values, 'heat', {strength: 0.05}),
        {type: 'sprites', items: variance, pulse: 0.55},
        {
          type: 'sprites',
          items: createSampleSprites(samples, 2.2).map(item => ({...item, role: 'ink' as const}))
        }
      ];
    }
  },

  'local-kriging': {
    parameters: [
      {id: 'neighbors', label: 'Neighbors', min: 3, max: 10, step: 1, default: 6},
      {id: 'range', label: 'Variogram range', min: 12, max: 90, step: 2, default: 40, unit: 'px'}
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const samples = createSamples(fork('samples'), 24, 8);
      const queries: {center: Point; block: boolean}[] = [
        {center: [40, 30], block: false},
        {center: [78, 72], block: true},
        {center: [122, 34], block: true},
        {center: [128, 78], block: false}
      ];
      if (pointer) queries.push({center: pointer, block: false});
      else queries.push({center: [82, 38], block: false});
      const neighborCount = Math.round(p.neighbors);
      const strokes: PathItem[] = [];
      const negative: PathItem[] = [];
      const rings: Point[][] = [];
      const blocks: PolygonItem[] = [];
      const predictions: SpriteItem[] = [];
      for (const {center, block} of queries) {
        const near = samples
          .map(sample => ({
            sample,
            distance: Math.hypot(sample.x - center[0], sample.y - center[1])
          }))
          .sort((first, second) => first.distance - second.distance)
          .slice(0, neighborCount);
        const local = near.map(entry => entry.sample);
        const krige = createKriging(local, p.range);
        const vector = local.map(sample =>
          Math.exp(-Math.hypot(sample.x - center[0], sample.y - center[1]) / p.range)
        );
        const matrix = local.map((first, row) =>
          local.map(
            (second, column) =>
              Math.exp(-Math.hypot(first.x - second.x, first.y - second.y) / p.range) +
              (row === column ? 0.03 : 0)
          )
        );
        const inverse = invertMatrix(matrix);
        local.forEach((sample, row) => {
          const weight = inverse[row].reduce(
            (sum, value, column) => sum + value * vector[column],
            0
          );
          (weight >= 0 ? strokes : negative).push({
            points: [center, [sample.x, sample.y]],
            role: weight >= 0 ? 'primary' : 'violet',
            width: 0.3 + Math.min(1.8, Math.abs(weight) * 3),
            alpha: 0.5
          });
        });
        rings.push(createCircle(center[0], center[1], near[near.length - 1].distance, 36));
        predictions.push({
          x: center[0],
          y: center[1],
          radius: block ? 3.4 : 3,
          role: getValueRole(krige(center[0], center[1]).prediction),
          alpha: 1
        });
        if (block)
          blocks.push({
            ring: createRectRing(center[0] - 6, center[1] - 6, center[0] + 6, center[1] + 6),
            role: 'warm',
            alpha: 0.2
          });
      }
      const neighborhoods: PathItem[] = rings.map(ring => ({
        points: [...ring, ring[0]],
        role: 'dim',
        width: 0.4,
        alpha: 0.45
      }));
      return [
        {type: 'strokes', items: neighborhoods, animation: 'static'},
        {type: 'fills', items: blocks, animation: 'pulse', blend: 'add'},
        {
          type: 'strokes',
          items: [...strokes, ...negative],
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: samples.map(
            (sample): SpriteItem => ({
              x: sample.x,
              y: sample.y,
              radius: 1.4,
              role: 'ink',
              alpha: 0.6
            })
          )
        },
        {type: 'sprites', items: predictions, pulse: 0.3}
      ];
    }
  },

  'conditional-gaussian-simulation': {
    parameters: [
      {id: 'realization', label: 'Realization', min: 1, max: 8, step: 1, default: 1},
      {id: 'samples', label: 'Samples', min: 5, max: 16, step: 1, default: 10}
    ],
    build: ({fork, p}) => {
      const samples = createSamples(fork('samples'), p.samples);
      const krige = createKriging(samples, 36);
      const simulation = createSmoothField(
        fork(`realization-${p.realization}`),
        8,
        1.9,
        FIELD_COLUMNS_LOW,
        FIELD_ROWS_LOW
      );
      const simulationSamples = createKriging(
        samples.map(sample => ({
          ...sample,
          value: sampleField(
            simulation,
            FIELD_COLUMNS_LOW,
            FIELD_ROWS_LOW,
            sample.x / WIDTH,
            sample.y / HEIGHT
          )
        })),
        36
      );
      const values = evaluateSceneField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, (x, y, column, row) => {
        const noise = simulation[row * FIELD_COLUMNS_LOW + column];
        return clamp(krige(x, y).prediction + noise - simulationSamples(x, y).prediction, 0, 1);
      });
      return [
        getStaticField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, values, 'spectral', {strength: 0.06}),
        {
          type: 'strokes',
          items: toPaths(
            samples.map(sample => createCircle(sample.x, sample.y, 4.5, 20)),
            'ink',
            0.5,
            0.8
          ),
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: createSampleSprites(samples, 1.6).map(item => ({...item, role: 'ink' as const}))
        }
      ];
    }
  },

  'local-rbf': {
    parameters: [
      {id: 'support', label: 'Support radius', min: 14, max: 56, step: 1, default: 30, unit: 'px'},
      {id: 'samples', label: 'Samples', min: 6, max: 22, step: 1, default: 14}
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const samples = createSamples(fork('samples'), p.samples);
      const support = p.support;
      const weigh = (distance: number) => {
        const u = distance / support;
        return u >= 1 ? 0 : (1 - u) ** 4 * (4 * u + 1);
      };
      const values = evaluateSceneField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, (x, y) => {
        let sum = 0;
        let total = 0;
        for (const sample of samples) {
          const weight = weigh(Math.hypot(sample.x - x, sample.y - y));
          sum += weight * sample.value;
          total += weight;
        }
        return total < 0.02 ? 0 : 0.1 + 0.9 * (sum / total) * Math.min(1, total * 2);
      });
      const probe = pointer ?? ([84, 48] as Point);
      const covering = samples.filter(
        sample => Math.hypot(sample.x - probe[0], sample.y - probe[1]) < support
      );
      return [
        getStaticField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, values, 'ice', {strength: 0.06}),
        {
          type: 'strokes',
          items: toPaths(
            samples.map(sample => createCircle(sample.x, sample.y, support, 36)),
            'dim',
            0.3,
            0.35
          )
        },
        {
          type: 'strokes',
          items: covering.map(
            (sample): PathItem => ({
              points: [probe, [sample.x, sample.y]],
              role: 'warm',
              width: 0.5 + weigh(Math.hypot(sample.x - probe[0], sample.y - probe[1])) * 2,
              alpha: 0.9
            })
          ),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: createSampleSprites(samples, 1.9).map(item => ({
            ...item,
            role: 'ink' as const,
            alpha: 0.9
          }))
        },
        {
          type: 'sprites',
          items: [{x: probe[0], y: probe[1], radius: 2.8, role: 'warm', alpha: 1}],
          pulse: 0.3
        }
      ];
    }
  },

  'natural-neighbor': {
    parameters: [{id: 'samples', label: 'Samples', min: 5, max: 16, step: 1, default: 9}],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const samples = createSamples(fork('samples'), p.samples, 12);
      const query = pointer ?? ([82, 50] as Point);
      const columns = 128;
      const rows = 80;
      const stolen = new Array<number>(samples.length).fill(0);
      let stolenTotal = 0;
      const values = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const x = ((column + 0.5) / columns) * WIDTH;
          const y = ((row + 0.5) / rows) * HEIGHT;
          let best = 0;
          let bestDistance = Infinity;
          for (let index = 0; index < samples.length; index++) {
            const distance = (samples[index].x - x) ** 2 + (samples[index].y - y) ** 2;
            if (distance < bestDistance) {
              bestDistance = distance;
              best = index;
            }
          }
          const queryDistance = (query[0] - x) ** 2 + (query[1] - y) ** 2;
          if (queryDistance < bestDistance) {
            stolen[best] += 1;
            stolenTotal += 1;
            values[row * columns + column] = 0.66 + 0.34 * samples[best].value;
          } else {
            values[row * columns + column] = 0.06 + 0.42 * samples[best].value;
          }
        }
      }
      const predicted = samples.reduce(
        (sum, sample, index) => sum + (stolen[index] / Math.max(1, stolenTotal)) * sample.value,
        0
      );
      return [
        getStaticField(columns, rows, values, 'ice', {filter: 'nearest', strength: 0.03}),
        {
          type: 'strokes',
          items: samples.flatMap((sample, index): PathItem[] =>
            stolen[index] > 0
              ? [
                  {
                    points: [query, [sample.x, sample.y]],
                    role: 'warm',
                    width: 0.4 + (stolen[index] / Math.max(1, stolenTotal)) * 4.5,
                    alpha: 0.9
                  }
                ]
              : []
          ),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: createSampleSprites(samples, 2).map(item => ({...item, role: 'ink' as const}))
        },
        {
          type: 'sprites',
          items: [{x: query[0], y: query[1], radius: 3, role: getValueRole(predicted), alpha: 1}],
          pulse: 0.35
        }
      ];
    }
  },

  // -------------------------------------------------------------------------- rasters and zones
  'focal-statistics': {
    parameters: [
      {id: 'radius', label: 'Window radius', min: 1, max: 5, step: 1, default: 2, unit: 'cells'},
      {
        id: 'statistic',
        label: 'Statistic',
        min: 0,
        max: 2,
        step: 1,
        default: 0,
        options: ['mean', 'max', 'range']
      }
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const columns = 32;
      const rows = 20;
      const random = fork('raster');
      const base = createSmoothField(random, 6, 1.3, columns, rows);
      const raw = base.map(value => clamp(value + random.gaussian() * 0.12, 0, 1));
      const radius = Math.round(p.radius);
      const statistic = Math.round(p.statistic);
      const focus = pointer ?? ([84, 52] as Point);
      const windowColumn = clamp(Math.floor((focus[0] / WIDTH) * columns), 0, columns - 1);
      const windowRow = clamp(Math.floor((focus[1] / HEIGHT) * rows), 0, rows - 1);
      const values = Float32Array.from(raw);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < windowColumn; column++) {
          let sum = 0;
          let count = 0;
          let maximum = 0;
          let minimum = 1;
          for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
              if (dx * dx + dy * dy > radius * radius + 0.5) continue;
              const x = column + dx;
              const y = row + dy;
              if (x < 0 || y < 0 || x >= columns || y >= rows) continue;
              const value = raw[y * columns + x];
              sum += value;
              count += 1;
              maximum = Math.max(maximum, value);
              minimum = Math.min(minimum, value);
            }
          }
          values[row * columns + column] =
            statistic === 0 ? sum / count : statistic === 1 ? maximum : (maximum - minimum) * 1.6;
        }
      }
      const cellWidth = WIDTH / columns;
      const cellHeight = HEIGHT / rows;
      const center: Point = [(windowColumn + 0.5) * cellWidth, (windowRow + 0.5) * cellHeight];
      return [
        getStaticField(columns, rows, values, 'ice', {filter: 'nearest', strength: 0.04}),
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [(windowColumn + 0.5) * cellWidth, 0],
                [(windowColumn + 0.5) * cellWidth, HEIGHT]
              ]
            ],
            'warm',
            0.4,
            0.55
          ),
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            [createCircle(center[0], center[1], (radius + 0.5) * cellWidth, 40)],
            'warm',
            1,
            1
          ),
          animation: 'pulse',
          speed: 0.6,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [{x: center[0], y: center[1], radius: 2, role: 'warm', alpha: 1}],
          pulse: 0.3
        }
      ];
    }
  },

  'areal-interpolation': {
    parameters: [
      {id: 'shift', label: 'Zone offset', min: 0, max: 30, step: 1, default: 14, unit: 'px'},
      {
        id: 'mode',
        label: 'Variable',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['extensive', 'intensive']
      }
    ],
    build: ({fork, p}) => {
      const random = fork('zones');
      const sourceSplitX = 64 + random.range(-6, 6);
      const sourceSplitY = 50 + random.range(-6, 6);
      const sources = [
        createRectRing(8, 6, sourceSplitX, sourceSplitY),
        createRectRing(sourceSplitX, 6, 152, sourceSplitY),
        createRectRing(8, sourceSplitY, sourceSplitX, 94),
        createRectRing(sourceSplitX, sourceSplitY, 152, 94)
      ];
      const sourceValues = [0.25, 0.9, 0.6, 0.35].map(value =>
        clamp(value + random.range(-0.1, 0.1), 0.05, 1)
      );
      const shift = p.shift;
      const xSplits = [8, 54 + shift * 0.4, 106 - shift * 0.3, 152];
      const ySplits = [6, 44 + shift * 0.5, 94];
      const targets: Point[][] = [];
      for (let row = 0; row < 2; row++) {
        for (let column = 0; column < 3; column++) {
          targets.push(
            createRectRing(xSplits[column], ySplits[row], xSplits[column + 1], ySplits[row + 1])
          );
        }
      }
      const overlap = targets.map(() => new Array<number>(sources.length).fill(0));
      const sourceArea = new Array<number>(sources.length).fill(0);
      const targetArea = new Array<number>(targets.length).fill(0);
      for (let sy = 0; sy < 44; sy++) {
        for (let sx = 0; sx < 72; sx++) {
          const point: Point = [8 + (sx + 0.5) * 2, 6 + (sy + 0.5) * 2];
          const source = sources.findIndex(ring => isInsideRing(point, ring));
          const target = targets.findIndex(ring => isInsideRing(point, ring));
          if (source < 0 || target < 0) continue;
          overlap[target][source] += 1;
          sourceArea[source] += 1;
          targetArea[target] += 1;
        }
      }
      const extensive = p.mode < 1;
      const targetValues = targets.map((_, target) =>
        sources.reduce((sum, _unused, source) => {
          const share = extensive
            ? overlap[target][source] / (sourceArea[source] || 1)
            : overlap[target][source] / (targetArea[target] || 1);
          return sum + share * sourceValues[source];
        }, 0)
      );
      const maximum = Math.max(...targetValues);
      // Intersection pieces: every source-target overlap, shaded by its source value and shrunk a
      // little so the pieces read as separate parts, then linked to the target they feed.
      const pieces: PolygonItem[] = [];
      const links: PathItem[] = [];
      const pieceCenters: SpriteItem[] = [];
      targets.forEach((targetRing, target) => {
        const [tx0, ty0, tx1, ty1] = getRingBounds(targetRing);
        const targetCenter = getRingCenter(targetRing);
        sources.forEach((sourceRing, source) => {
          const [sx0, sy0, sx1, sy1] = getRingBounds(sourceRing);
          const x0 = Math.max(tx0, sx0);
          const y0 = Math.max(ty0, sy0);
          const x1 = Math.min(tx1, sx1);
          const y1 = Math.min(ty1, sy1);
          if (x1 - x0 < 1 || y1 - y0 < 1) return;
          const inset = 1.1;
          pieces.push({
            ring: createRectRing(x0 + inset, y0 + inset, x1 - inset, y1 - inset),
            role: getValueRole(sourceValues[source]),
            alpha: 0.7,
            phase: (target + source * 0.25) / 6
          });
          const center: Point = [(x0 + x1) / 2, (y0 + y1) / 2];
          pieceCenters.push({
            x: center[0],
            y: center[1],
            radius: 0.9,
            role: 'ink',
            alpha: 0.9,
            phase: target / 6
          });
          links.push({points: [center, targetCenter], role: 'warm', width: 0.5, alpha: 0.75});
        });
      });
      return [
        {
          type: 'fills',
          items: sources.map(
            (ring, index): PolygonItem => ({
              ring,
              role: getValueRole(sourceValues[index]),
              alpha: 0.16
            })
          ),
          animation: 'static',
          blend: 'add'
        },
        {type: 'fills', items: pieces, animation: 'pulse', blend: 'add'},
        {
          type: 'strokes',
          items: toPaths(
            targets.map(ring => [...ring, ring[0]]),
            'cool',
            0.9,
            0.9
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            sources.map(ring => [...ring, ring[0]]),
            'violet',
            1.4,
            0.95
          ),
          animation: 'pulse',
          speed: 0.2,
          blend: 'add'
        },
        {type: 'strokes', items: links, animation: 'flow', speed: 0.3, blend: 'add'},
        {type: 'sprites', items: pieceCenters, pulse: 0.3},
        {
          type: 'sprites',
          items: targets.map((ring, index): SpriteItem => {
            const [x, y] = getRingCenter(ring);
            const value = targetValues[index] / maximum;
            return {
              x,
              y,
              radius: 2.5 + value * 5,
              role: getValueRole(value),
              alpha: 0.95,
              phase: index / 6
            };
          }),
          pulse: 0.35
        }
      ];
    }
  },

  pycnophylactic: {
    parameters: [
      {id: 'smoothing', label: 'Iterations', min: 0, max: 40, step: 1, default: 14},
      {id: 'contrast', label: 'Zone contrast', min: 0.3, max: 1, step: 0.05, default: 0.8}
    ],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('zones'), 3, 2, 0.25);
      const columns = FIELD_COLUMNS_LOW;
      const rows = FIELD_ROWS_LOW;
      const zone = new Int8Array(columns * rows).fill(-1);
      const target = tracts.values.map(value => 0.15 + value * p.contrast * 0.85);
      let values = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const point: Point = [((column + 0.5) / columns) * WIDTH, ((row + 0.5) / rows) * HEIGHT];
          const index = tracts.rings.findIndex(ring => isInsideRing(point, ring));
          zone[row * columns + column] = index;
          values[row * columns + column] = index >= 0 ? target[index] : 0;
        }
      }
      for (let iteration = 0; iteration < p.smoothing; iteration++) {
        const next = new Float32Array(values.length);
        for (let row = 0; row < rows; row++) {
          for (let column = 0; column < columns; column++) {
            const at = (c: number, r: number) =>
              values[clamp(r, 0, rows - 1) * columns + clamp(c, 0, columns - 1)];
            next[row * columns + column] =
              (at(column, row) * 2 +
                at(column - 1, row) +
                at(column + 1, row) +
                at(column, row - 1) +
                at(column, row + 1)) /
              6;
          }
        }
        const sums = new Array<number>(tracts.rings.length).fill(0);
        const counts = new Array<number>(tracts.rings.length).fill(0);
        next.forEach((value, index) => {
          if (zone[index] >= 0) {
            sums[zone[index]] += value;
            counts[zone[index]] += 1;
          }
        });
        next.forEach((value, index) => {
          const owner = zone[index];
          if (owner >= 0)
            next[index] = Math.max(0, value + target[owner] - sums[owner] / counts[owner]);
        });
        values = next;
      }
      return [
        getStaticField(columns, rows, values, 'heat', {style: 'ramp', strength: 0.04}),
        {
          type: 'strokes',
          items: toPaths(
            tracts.rings.map(ring => [...ring, ring[0]]),
            'ink',
            0.8,
            0.8
          ),
          animation: 'pulse',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: tracts.centers.map(
            (center, index): SpriteItem => ({
              x: center[0],
              y: center[1],
              radius: 1 + target[index] * 3,
              role: 'ink',
              alpha: 0.85,
              phase: index / 6
            })
          ),
          pulse: 0.3
        }
      ];
    }
  },

  // ------------------------------------------------------------------------- point patterns
  'inhomogeneous-point-pattern': {
    parameters: [
      {id: 'radius', label: 'Distance r', min: 6, max: 30, step: 1, default: 14, unit: 'px'},
      {id: 'events', label: 'Events', min: 40, max: 200, step: 10, default: 110}
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const intensity = createSmoothField(
        fork('intensity'),
        4,
        0.9,
        FIELD_COLUMNS_LOW,
        FIELD_ROWS_LOW
      ).map(value => value ** 1.6);
      const random = fork('events');
      const points: Point[] = [];
      for (let attempt = 0; attempt < 4000 && points.length < p.events; attempt++) {
        const candidate: Point = [random.range(6, WIDTH - 6), random.range(6, HEIGHT - 6)];
        if (
          random.next() <
          sampleField(
            intensity,
            FIELD_COLUMNS_LOW,
            FIELD_ROWS_LOW,
            candidate[0] / WIDTH,
            candidate[1] / HEIGHT
          )
        ) {
          points.push(candidate);
        }
      }
      const target = pointer ?? ([70, 50] as Point);
      let focus = points[0] ?? target;
      for (const point of points)
        if (getDistance(point, target) < getDistance(focus, target)) focus = point;
      const near = points.filter(point => point !== focus && getDistance(point, focus) <= p.radius);
      return [
        getStaticField(FIELD_COLUMNS_LOW, FIELD_ROWS_LOW, intensity, 'ice', {
          style: 'contours',
          levels: 8,
          strength: 0.03
        }),
        {
          type: 'sprites',
          items: toSprites(
            points.filter(point => !near.includes(point)),
            'primary',
            1,
            random,
            0.75
          ),
          pulse: 0.2
        },
        {
          type: 'strokes',
          items: toPaths(
            near.map(point => [focus, point]),
            'warm',
            0.6,
            0.85
          ),
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        },
        {type: 'sprites', items: toSprites(near, 'warm', 1.5, random, 1), pulse: 0.3},
        {
          type: 'strokes',
          items: toPaths([createCircle(focus[0], focus[1], p.radius, 48)], 'warm', 0.7, 0.9),
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [{x: focus[0], y: focus[1], radius: 2.4, role: 'hot', alpha: 1}],
          pulse: 0.3
        }
      ];
    }
  },

  'point-pattern-envelope': {
    parameters: [
      {id: 'simulations', label: 'Simulations', min: 9, max: 99, step: 5, default: 39},
      {id: 'clustering', label: 'Clustering', min: 0, max: 1, step: 0.05, default: 0.75}
    ],
    build: ({fork, p}) => {
      const steps = 40;
      const x = (index: number) => 16 + (index / (steps - 1)) * 130;
      const y = (value: number) => 58 - value * 36;
      const random = fork('simulations');
      const simulated: number[][] = Array.from({length: Math.round(p.simulations)}, () => {
        let walk = 0;
        return Array.from({length: steps}, (_, index) => {
          walk = walk * 0.8 + random.gaussian() * 0.07;
          return walk * Math.sqrt(index / steps) * 1.3;
        });
      });
      const lower = Array.from({length: steps}, (_, index) =>
        Math.min(...simulated.map(curve => curve[index]))
      );
      const upper = Array.from({length: steps}, (_, index) =>
        Math.max(...simulated.map(curve => curve[index]))
      );
      const observedRandom = fork('observed');
      let wobble = 0;
      const observed = Array.from({length: steps}, (_, index) => {
        wobble = wobble * 0.7 + observedRandom.gaussian() * 0.03;
        const t = index / (steps - 1);
        return (
          p.clustering * 1.05 * Math.sin(Math.PI * Math.min(1, t * 1.25)) * (1 - 0.25 * t) +
          wobble * (t + 0.2)
        );
      });
      const band: Point[] = [
        ...upper.map((value, index): Point => [x(index), y(value)]),
        ...lower.map((value, index): Point => [x(index), y(value)]).reverse()
      ];
      const outside: SpriteItem[] = observed.flatMap((value, index): SpriteItem[] =>
        value > upper[index] || value < lower[index]
          ? [{x: x(index), y: y(value), radius: 1.5, role: 'hot', alpha: 1, phase: index / steps}]
          : []
      );
      return [
        {
          type: 'strokes',
          items: [
            {
              points: [
                [14, 24],
                [14, 92]
              ] as Point[],
              role: 'dim',
              width: 0.4,
              alpha: 0.5
            },
            {
              points: [
                [14, 58],
                [150, 58]
              ] as Point[],
              role: 'dim',
              width: 0.4,
              alpha: 0.5
            }
          ],
          animation: 'static'
        },
        {
          type: 'fills',
          items: [{ring: band, role: 'cool', alpha: 0.28}],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: simulated.slice(0, 40).map(
            (curve): PathItem => ({
              points: curve.map((value, index): Point => [x(index), y(value)]),
              role: 'cool',
              width: 0.35,
              alpha: 0.28
            })
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            [observed.map((value, index): Point => [x(index), y(value)])],
            'warm',
            1.6,
            1
          ),
          animation: 'draw',
          speed: 0.25,
          blend: 'add'
        },
        {type: 'sprites', items: outside, pulse: 0.5}
      ];
    }
  },

  // -------------------------------------------------------------------------------- sampling
  'continuous-spatial-sample': {
    parameters: [{id: 'count', label: 'Sites', min: 4, max: 40, step: 1, default: 16}],
    build: ({fork, p}) => {
      const order = 3;
      const size = 1 << order;
      const cellWidth = (WIDTH - 20) / size;
      const cellHeight = (HEIGHT - 16) / size;
      const getCurvePoint = (distance: number): [number, number] => {
        let x = 0;
        let y = 0;
        let t = distance;
        for (let scale = 1; scale < size; scale *= 2) {
          const rx = 1 & (t / 2);
          const ry = 1 & (t ^ rx);
          if (ry === 0) {
            if (rx === 1) {
              x = scale - 1 - x;
              y = scale - 1 - y;
            }
            [x, y] = [y, x];
          }
          x += scale * rx;
          y += scale * ry;
          t = Math.floor(t / 4);
        }
        return [x, y];
      };
      const cells: Point[] = Array.from({length: size * size}, (_, index) => {
        const [column, row] = getCurvePoint(index);
        return [10 + (column + 0.5) * cellWidth, 8 + (row + 0.5) * cellHeight];
      });
      const random = fork('sites');
      const offset = random.next();
      const sites: Point[] = Array.from({length: Math.round(p.count)}, (_, index) => {
        const cell = Math.min(
          cells.length - 1,
          Math.floor(((index + offset) / p.count) * cells.length)
        );
        return [
          cells[cell][0] + random.range(-0.35, 0.35) * cellWidth,
          cells[cell][1] + random.range(-0.35, 0.35) * cellHeight
        ];
      });
      return [
        {
          type: 'strokes',
          items: toPaths([cells], 'cool', 0.6, 0.55),
          animation: 'flow',
          speed: 0.12,
          trail: 2,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: sites.map(
            (site, index): SpriteItem => ({
              x: site[0],
              y: site[1],
              radius: 5.5,
              role: 'warm',
              alpha: 0.13,
              phase: index / sites.length
            })
          ),
          pulse: 0.5
        },
        {type: 'sprites', items: toSprites(sites, 'warm', 1.7, random, 1), pulse: 0.3}
      ];
    }
  },

  'spatial-balanced-sample': {
    parameters: [{id: 'count', label: 'Sample size', min: 4, max: 32, step: 2, default: 14}],
    build: ({fork, p}) => {
      const random = fork('frame');
      const frame: Point[] = [
        ...createClusterPoints(
          random,
          [
            [36, 40],
            [58, 74],
            [112, 36],
            [124, 70]
          ],
          36,
          11
        ),
        ...createScatterPoints(random, 30)
      ];
      const leaves: {bounds: [number, number, number, number]; points: Point[]}[] = [];
      const split = (points: Point[], count: number, bounds: [number, number, number, number]) => {
        if (count <= 1 || points.length <= 1) {
          leaves.push({bounds, points});
          return;
        }
        const horizontal = bounds[2] - bounds[0] >= bounds[3] - bounds[1];
        const axis = horizontal ? 0 : 1;
        const sorted = [...points].sort((first, second) => first[axis] - second[axis]);
        const leftCount = Math.floor(count / 2);
        const cut = Math.round((sorted.length * leftCount) / count);
        const edge = (sorted[cut - 1][axis] + sorted[cut][axis]) / 2;
        const first: [number, number, number, number] = horizontal
          ? [bounds[0], bounds[1], edge, bounds[3]]
          : [bounds[0], bounds[1], bounds[2], edge];
        const second: [number, number, number, number] = horizontal
          ? [edge, bounds[1], bounds[2], bounds[3]]
          : [bounds[0], edge, bounds[2], bounds[3]];
        split(sorted.slice(0, cut), leftCount, first);
        split(sorted.slice(cut), count - leftCount, second);
      };
      const half = Math.round(p.count / 2);
      split(
        frame.filter(point => point[0] < 80),
        half,
        [0, 0, 80, HEIGHT]
      );
      split(
        frame.filter(point => point[0] >= 80),
        Math.round(p.count) - half,
        [80, 0, WIDTH, HEIGHT]
      );
      const chooser = fork('choice');
      const selected = leaves
        .filter(leaf => leaf.points.length > 0)
        .map(leaf => leaf.points[chooser.integer(leaf.points.length)]);
      return [
        {
          type: 'strokes',
          items: leaves.map((leaf): PathItem => {
            const [x0, y0, x1, y1] = leaf.bounds;
            return {
              points: [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0]
              ],
              role: 'dim',
              width: 0.35,
              alpha: 0.5
            };
          }),
          animation: 'static'
        },
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [80, 0],
                [80, HEIGHT]
              ]
            ],
            'violet',
            0.9,
            0.8
          ),
          animation: 'pulse',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: toSprites(
            frame.filter(point => point[0] < 80),
            'cool',
            0.8,
            random,
            0.5
          )
        },
        {
          type: 'sprites',
          items: toSprites(
            frame.filter(point => point[0] >= 80),
            'primary',
            0.8,
            random,
            0.5
          )
        },
        {
          type: 'sprites',
          items: selected.map(
            (point, index): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: 4.2,
              role: 'warm',
              alpha: 0.22,
              phase: index / selected.length
            })
          ),
          pulse: 0.5
        },
        {type: 'sprites', items: toSprites(selected, 'warm', 1.8, chooser, 1), pulse: 0.3}
      ];
    }
  },

  'geographic-distribution': {
    parameters: [
      {id: 'angle', label: 'Orientation', min: 0, max: 180, step: 5, default: 30, unit: '°'},
      {id: 'elongation', label: 'Elongation', min: 1, max: 4, step: 0.1, default: 2.6}
    ],
    build: ({fork, p}) => {
      const layers: PreviewLayer[] = [];
      const groups: {center: Point; angle: number; role: Role}[] = [
        {center: [48, 50], angle: (p.angle * Math.PI) / 180, role: 'primary'},
        {center: [114, 50], angle: ((p.angle + 75) * Math.PI) / 180, role: 'violet'}
      ];
      groups.forEach(({center, angle, role}, groupIndex) => {
        const random = fork(`group-${groupIndex}`);
        const points: Point[] = Array.from({length: 44}, () => {
          const along = random.gaussian() * 6.5 * p.elongation * 0.6;
          const across = random.gaussian() * 6.5 * 0.6;
          return [
            center[0] + Math.cos(angle) * along - Math.sin(angle) * across,
            center[1] + Math.sin(angle) * along + Math.cos(angle) * across
          ];
        });
        const count = points.length;
        const mean = points.reduce<Point>(
          (sum, point) => [sum[0] + point[0] / count, sum[1] + point[1] / count],
          [0, 0]
        );
        const sortedX = points.map(point => point[0]).sort((first, second) => first - second);
        const sortedY = points.map(point => point[1]).sort((first, second) => first - second);
        const median: Point = [sortedX[count >> 1], sortedY[count >> 1]];
        let sxx = 0;
        let syy = 0;
        let sxy = 0;
        for (const point of points) {
          sxx += (point[0] - mean[0]) ** 2 / count;
          syy += (point[1] - mean[1]) ** 2 / count;
          sxy += ((point[0] - mean[0]) * (point[1] - mean[1])) / count;
        }
        const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        const trace = sxx + syy;
        const determinant = Math.sqrt(Math.max(0, ((sxx - syy) / 2) ** 2 + sxy * sxy));
        const major = Math.sqrt(trace / 2 + determinant) * 1.6;
        const minor = Math.sqrt(Math.max(0.01, trace / 2 - determinant)) * 1.6;
        const ellipse: Point[] = Array.from({length: 49}, (_, index) => {
          const t = (index / 48) * Math.PI * 2;
          return [
            mean[0] + Math.cos(theta) * major * Math.cos(t) - Math.sin(theta) * minor * Math.sin(t),
            mean[1] + Math.sin(theta) * major * Math.cos(t) + Math.cos(theta) * minor * Math.sin(t)
          ];
        });
        layers.push(
          {type: 'sprites', items: toSprites(points, role, 1, random, 0.7), pulse: 0.2},
          {
            type: 'strokes',
            items: toPaths(
              [createCircle(mean[0], mean[1], Math.sqrt(trace) * 1.4, 44)],
              'dim',
              0.6,
              0.7
            ),
            animation: 'static'
          },
          {
            type: 'strokes',
            items: toPaths([ellipse], role === 'primary' ? 'warm' : 'hot', 1.2, 1),
            animation: 'flow',
            speed: 0.25,
            blend: 'add'
          },
          {
            type: 'sprites',
            items: [
              {x: mean[0], y: mean[1], radius: 2.8, role: 'ink', alpha: 1},
              {x: median[0], y: median[1], radius: 1.7, role: 'warm', alpha: 1}
            ],
            pulse: 0.3
          }
        );
      });
      return layers;
    }
  },

  'rose-statistic': {
    parameters: [
      {
        id: 'orientation',
        label: 'Grid orientation',
        min: 0,
        max: 90,
        step: 5,
        default: 25,
        unit: '°'
      },
      {id: 'bins', label: 'Bins', min: 8, max: 36, step: 2, default: 18}
    ],
    build: ({fork, p}) => {
      const random = fork('segments');
      const bearings = Array.from({length: 240}, () => {
        const mode = random.next() < 0.7 ? 0 : 90;
        const jitter = random.next() < 0.18 ? random.range(-90, 90) : random.gaussian() * 7;
        return (((p.orientation + mode + jitter) % 180) + 180) % 180;
      });
      const bins = Math.round(p.bins);
      const counts = new Array<number>(bins).fill(0);
      for (const bearing of bearings)
        counts[Math.min(bins - 1, Math.floor((bearing / 180) * bins))] += 1;
      const maximum = Math.max(...counts);
      const center: Point = [114, 50];
      const radius = 36;
      const toPoint = (degrees: number, distance: number): Point => [
        center[0] + Math.sin((degrees * Math.PI) / 180) * distance,
        center[1] - Math.cos((degrees * Math.PI) / 180) * distance
      ];
      const petals: PolygonItem[] = [];
      for (let half = 0; half < 2; half++) {
        counts.forEach((count, index) => {
          const amount = Math.sqrt(count / maximum);
          const start = half * 180 + (index / bins) * 180;
          const end = half * 180 + ((index + 1) / bins) * 180;
          const ring: Point[] = [center, toPoint(start, amount * radius)];
          for (let step = 1; step < 4; step++)
            ring.push(toPoint(start + ((end - start) * step) / 4, amount * radius));
          ring.push(toPoint(end, amount * radius));
          petals.push({
            ring,
            role: getValueRole(amount),
            alpha: 0.35 + amount * 0.5,
            phase: index / bins
          });
        });
      }
      let sine = 0;
      let cosine = 0;
      for (const bearing of bearings) {
        sine += Math.sin((bearing * 2 * Math.PI) / 180);
        cosine += Math.cos((bearing * 2 * Math.PI) / 180);
      }
      const meanDirection = (Math.atan2(sine, cosine) * 90) / Math.PI;
      const streets: PathItem[] = bearings.slice(0, 46).map((bearing, index) => {
        const x = 8 + ((index * 37) % 60) + random.range(-2, 2);
        const y = 10 + ((index * 53) % 80) + random.range(-2, 2);
        const [dx, dy] = [
          Math.sin((bearing * Math.PI) / 180) * 5,
          -Math.cos((bearing * Math.PI) / 180) * 5
        ];
        return {
          points: [
            [x - dx, y - dy],
            [x + dx, y + dy]
          ],
          role: getValueRole((bearing / 180) % 1),
          width: 0.7,
          alpha: 0.7
        };
      });
      return [
        {type: 'strokes', items: streets, animation: 'pulse', speed: 0.2, blend: 'add'},
        {
          type: 'strokes',
          items: toPaths(
            [
              createCircle(center[0], center[1], radius, 48),
              createCircle(center[0], center[1], radius / 2, 36)
            ],
            'dim',
            0.4,
            0.5
          )
        },
        {type: 'fills', items: petals, animation: 'pulse', blend: 'add'},
        {
          type: 'strokes',
          items: toPaths(
            [[toPoint(meanDirection, -radius - 3), toPoint(meanDirection, radius + 3)]],
            'warm',
            1.2,
            1
          ),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        }
      ];
    }
  },

  // ------------------------------------------------------------------------- classification
  'class-assignment': {
    parameters: [
      {id: 'classes', label: 'Classes', min: 3, max: 7, step: 1, default: 5},
      {id: 'drift', label: 'Drift per period', min: 0, max: 0.3, step: 0.01, default: 0.14}
    ],
    build: ({fork, p}) => {
      const random = fork('values');
      const classes = Math.round(p.classes);
      const rows = [22, 50, 78];
      const periods = rows.map((_, period) =>
        Array.from({length: 40}, () =>
          clamp(0.26 + period * p.drift + random.gaussian() * 0.14, 0.01, 0.99)
        )
      );
      const pooled = periods.flat().sort((first, second) => first - second);
      const breaks = Array.from(
        {length: classes - 1},
        (_, index) => pooled[Math.floor(((index + 1) * pooled.length) / classes)]
      );
      const getClass = (value: number) => breaks.filter(boundary => value >= boundary).length;
      const edges = [0, ...breaks, 1];
      const bands = edges.slice(0, -1).map(
        (start, index): PolygonItem => ({
          ring: createRectRing(10 + start * 140, 6, 10 + edges[index + 1] * 140, 94),
          role: getClassRole(index, classes),
          alpha: 0.09
        })
      );
      const dots = periods.flatMap((values, period) =>
        values.map(
          (value): SpriteItem => ({
            x: 10 + value * 140,
            y: rows[period] + random.range(-7, 7),
            radius: 1.5,
            role: getClassRole(getClass(value), classes),
            alpha: 0.95,
            phase: random.next()
          })
        )
      );
      return [
        {type: 'fills', items: bands, animation: 'static', blend: 'add'},
        {
          type: 'strokes',
          items: toPaths(
            breaks.map(
              boundary =>
                [
                  [10 + boundary * 140, 6],
                  [10 + boundary * 140, 94]
                ] as Point[]
            ),
            'ink',
            0.7,
            0.85
          ),
          animation: 'pulse',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            rows.map(
              y =>
                [
                  [8, y + 11],
                  [152, y + 11]
                ] as Point[]
            ),
            'dim',
            0.25,
            0.35
          )
        },
        {type: 'sprites', items: dots, pulse: 0.3}
      ];
    }
  },

  'classification-fit': {
    parameters: [
      {id: 'classes', label: 'Classes', min: 2, max: 8, step: 1, default: 5},
      {id: 'skew', label: 'Skew', min: 0.2, max: 1.2, step: 0.05, default: 0.7}
    ],
    build: ({fork, p}) => {
      const random = fork('values');
      const values = Array.from({length: 240}, () => Math.exp(random.gaussian() * p.skew)).sort(
        (first, second) => first - second
      );
      const upperLimit = values[Math.floor(values.length * 0.985)];
      const clipped = values.map(value => Math.min(value, upperLimit) / upperLimit);
      const classes = Math.round(p.classes);
      const bars = 28;
      const counts = new Array<number>(bars).fill(0);
      for (const value of clipped) counts[Math.min(bars - 1, Math.floor(value * bars))] += 1;
      const maximum = Math.max(...counts);
      const getClass = (value: number) => Math.min(classes - 1, Math.floor(value * classes));
      const median = (items: number[]) => (items.length ? items[items.length >> 1] : 0);
      const overallMedian = median(clipped);
      const classMedians = Array.from({length: classes}, (_, index) =>
        median(clipped.filter(value => getClass(value) === index))
      );
      const baseline = clipped.reduce((sum, value) => sum + Math.abs(value - overallMedian), 0);
      const residual = clipped.reduce(
        (sum, value) => sum + Math.abs(value - classMedians[getClass(value)]),
        0
      );
      const gadf = clamp(1 - residual / baseline, 0, 1);
      const barWidth = 140 / bars;
      return [
        {
          type: 'fills',
          items: counts.map(
            (count, index): PolygonItem => ({
              ring: createRectRing(
                10 + index * barWidth + 0.5,
                66 - (count / maximum) * 50,
                10 + (index + 1) * barWidth - 0.5,
                66
              ),
              role: getClassRole(getClass((index + 0.5) / bars), classes),
              alpha: 0.75,
              phase: index / bars
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: classMedians.map(
            (value, index): PathItem => ({
              points: [
                [10 + value * 140, 10],
                [10 + value * 140, 70]
              ],
              role: getClassRole(index, classes),
              width: 0.9,
              alpha: 0.9
            })
          ),
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: classMedians.map(
            (value, index): PathItem => ({
              points: [
                [10 + (index / classes) * 140, 77],
                [10 + ((index + 1) / classes) * 140, 77]
              ],
              role: getClassRole(index, classes),
              width: 1,
              alpha: 0.85
            })
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'sprites',
          items: classMedians.map(
            (value, index): SpriteItem => ({
              x: 10 + value * 140,
              y: 77,
              radius: 1.9,
              role: 'ink',
              alpha: 1,
              phase: index / classes
            })
          ),
          pulse: 0.3
        },
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [10, 90],
                [150, 90]
              ]
            ],
            'dim',
            2.4,
            0.4
          )
        },
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [10, 90],
                [10 + gadf * 140, 90]
              ]
            ],
            gadf > 0.5 ? 'warm' : 'cool',
            2.4,
            1
          ),
          animation: 'draw',
          speed: 0.2,
          blend: 'add'
        }
      ];
    }
  },

  // ------------------------------------------------------------------------- Markov dynamics
  'transition-matrix': {
    parameters: [
      {id: 'persistence', label: 'Persistence', min: 0.3, max: 0.9, step: 0.05, default: 0.6},
      {id: 'classes', label: 'Classes', min: 3, max: 5, step: 1, default: 4}
    ],
    build: ({p}) => {
      const classes = Math.round(p.classes);
      const matrix = createTransitionMatrix(classes, p.persistence, 0.1);
      const spacing = 80 / (classes - 1);
      const row = (index: number) => 10 + index * spacing;
      const strokes: PathItem[] = [];
      matrix.forEach((probabilities, from) => {
        probabilities.forEach((probability, to) => {
          if (probability < 0.04) return;
          const points: Point[] = Array.from({length: 14}, (_, step) => {
            const t = step / 13;
            const eased = t * t * (3 - 2 * t);
            return [38 + t * 84, row(from) + (row(to) - row(from)) * eased];
          });
          strokes.push({
            points,
            role: getClassRole(from, classes),
            width: 0.4 + probability * 5,
            alpha: 0.35 + probability * 0.6
          });
        });
      });
      const nodes: SpriteItem[] = [38, 122].flatMap(x =>
        Array.from(
          {length: classes},
          (_, index): SpriteItem => ({
            x,
            y: row(index),
            radius: 3.2,
            role: getClassRole(index, classes),
            alpha: 1,
            phase: index / classes
          })
        )
      );
      return [
        {type: 'strokes', items: strokes, animation: 'flow', speed: 0.25, trail: 1.5, blend: 'add'},
        {type: 'sprites', items: nodes, pulse: 0.25}
      ];
    }
  },

  'spatial-markov': {
    parameters: [
      {id: 'persistence', label: 'Persistence', min: 0.3, max: 0.85, step: 0.05, default: 0.5},
      {id: 'dependence', label: 'Spatial dependence', min: 0, max: 0.4, step: 0.02, default: 0.24}
    ],
    build: ({fork, p}) => {
      const random = fork('neighbors');
      const items: PolygonItem[] = [];
      const lines: PathItem[] = [];
      const neighbors: SpriteItem[] = [];
      const lagRoles: Role[] = ['cool', 'primary', 'hot'];
      [0, 1, 2].forEach(lag => {
        const matrix = createTransitionMatrix(3, p.persistence, (lag - 1) * p.dependence);
        const left = 12 + lag * 50;
        matrix.forEach((probabilities, row) => {
          probabilities.forEach((probability, column) => {
            items.push({
              ring: createRectRing(
                left + column * 11 + 0.5,
                28 + row * 11 + 0.5,
                left + column * 11 + 10.5,
                28 + row * 11 + 10.5
              ),
              role: getValueRole(probability + 0.1),
              alpha: 0.12 + probability * 0.85,
              phase: (row + column + lag) / 8
            });
          });
        });
        lines.push({
          points: [
            [left, 18],
            [left + 33, 18]
          ],
          role: lagRoles[lag],
          width: 2.2,
          alpha: 0.95
        });
        const center: Point = [left + 16.5, 80];
        neighbors.push({x: center[0], y: center[1], radius: 2.4, role: 'ink', alpha: 1});
        for (let ring = 0; ring < 6; ring++) {
          const angle = (ring / 6) * Math.PI * 2 + random.range(-0.2, 0.2);
          neighbors.push({
            x: center[0] + Math.cos(angle) * 9,
            y: center[1] + Math.sin(angle) * 9,
            radius: 1.7,
            role: lagRoles[lag],
            alpha: 0.9,
            phase: ring / 6
          });
        }
      });
      return [
        {type: 'fills', items, animation: 'pulse', blend: 'add'},
        {type: 'strokes', items: lines, animation: 'pulse', speed: 0.3, blend: 'add'},
        {type: 'sprites', items: neighbors, pulse: 0.3}
      ];
    }
  },

  'lisa-markov': {
    parameters: [
      {id: 'persistence', label: 'Persistence', min: 0.2, max: 1, step: 0.05, default: 0.6},
      {id: 'autocorrelation', label: 'Moran slope', min: 0.1, max: 0.9, step: 0.05, default: 0.55}
    ],
    build: ({fork, p}) => {
      const random = fork('places');
      const quadrantRole = (z: number, lag: number): Role =>
        z >= 0 ? (lag >= 0 ? 'hot' : 'warm') : lag >= 0 ? 'violet' : 'cool';
      const toScene = (z: number, lag: number): Point => [
        80 + clamp(z, -1.2, 1.2) * 62,
        50 - clamp(lag, -1.2, 1.2) * 40
      ];
      const start: SpriteItem[] = [];
      const end: SpriteItem[] = [];
      const moves: PathItem[] = [];
      const changed: PathItem[] = [];
      for (let index = 0; index < 40; index++) {
        const z = random.gaussian() * 0.5;
        const lag = z * p.autocorrelation + random.gaussian() * 0.3;
        const nextZ = z * p.persistence + random.gaussian() * 0.32 * (1.15 - p.persistence);
        const nextLag = nextZ * p.autocorrelation + lag * 0.4 + random.gaussian() * 0.2;
        const from = toScene(z, lag);
        const to = toScene(nextZ, nextLag);
        start.push({x: from[0], y: from[1], radius: 1.1, role: quadrantRole(z, lag), alpha: 0.4});
        end.push({
          x: to[0],
          y: to[1],
          radius: 1.6,
          role: quadrantRole(nextZ, nextLag),
          alpha: 1,
          phase: random.next()
        });
        const switched = quadrantRole(z, lag) !== quadrantRole(nextZ, nextLag);
        (switched ? changed : moves).push({
          points: [from, to],
          role: quadrantRole(nextZ, nextLag),
          width: switched ? 0.9 : 0.4,
          alpha: switched ? 0.95 : 0.3
        });
      }
      return [
        {
          type: 'strokes',
          items: [
            {
              points: [
                [10, 50],
                [150, 50]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.7
            },
            {
              points: [
                [80, 6],
                [80, 94]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.7
            }
          ],
          animation: 'static'
        },
        {type: 'strokes', items: moves, animation: 'static', blend: 'add'},
        {type: 'strokes', items: changed, animation: 'flow', speed: 0.3, trail: 1.4, blend: 'add'},
        {type: 'sprites', items: start},
        {type: 'sprites', items: end, pulse: 0.3}
      ];
    }
  },

  'rank-markov': {
    parameters: [
      {id: 'volatility', label: 'Mobility', min: 0.05, max: 1.2, step: 0.05, default: 0.45},
      {id: 'units', label: 'Places', min: 5, max: 14, step: 1, default: 10}
    ],
    build: ({fork, p}) => {
      const random = fork('scores');
      const periods = 5;
      const units = Math.round(p.units);
      let scores = Array.from({length: units}, () => random.gaussian());
      const ranks: number[][] = [];
      for (let period = 0; period < periods; period++) {
        const order = scores
          .map((score, index) => ({score, index}))
          .sort((first, second) => second.score - first.score);
        const rankOf = new Array<number>(units);
        order.forEach((entry, rank) => (rankOf[entry.index] = rank));
        ranks.push(rankOf);
        scores = scores.map(score => score + random.gaussian() * p.volatility);
      }
      const x = (period: number) => 16 + period * 32;
      const y = (rank: number) => 12 + (rank / (units - 1)) * 76;
      const tracks = Array.from({length: units}, (_, unit): Point[] =>
        smoothPath(
          ranks.map((rankOf, period): Point => [x(period), y(rankOf[unit])]),
          2
        )
      );
      const highlighted = [0, 1];
      const roles: Role[] = ['hot', 'cool'];
      return [
        {
          type: 'strokes',
          items: Array.from(
            {length: periods},
            (_, period): PathItem => ({
              points: [
                [x(period), 6],
                [x(period), 94]
              ],
              role: 'dim',
              width: 0.3,
              alpha: 0.35
            })
          ),
          animation: 'static'
        },
        {
          type: 'strokes',
          items: tracks
            .filter((_, unit) => !highlighted.includes(unit))
            .map((track): PathItem => ({points: track, role: 'primary', width: 0.7, alpha: 0.45})),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: highlighted.map(
            (unit, index): PathItem => ({
              points: tracks[unit],
              role: roles[index],
              width: 1.6,
              alpha: 1,
              phase: index * 0.5
            })
          ),
          animation: 'flow',
          speed: 0.25,
          trail: 1.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: ranks.flatMap((rankOf, period) =>
            rankOf.map(
              (rank, unit): SpriteItem => ({
                x: x(period),
                y: y(rank),
                radius: highlighted.includes(unit) ? 2.2 : 1.2,
                role: highlighted.includes(unit) ? roles[highlighted.indexOf(unit)] : 'primary',
                alpha: highlighted.includes(unit) ? 1 : 0.7,
                phase: period / periods
              })
            )
          ),
          pulse: 0.25
        }
      ];
    }
  },

  'transition-diagnostics': {
    parameters: [
      {id: 'persistence', label: 'Persistence', min: 0.3, max: 0.9, step: 0.05, default: 0.6},
      {id: 'states', label: 'States', min: 3, max: 5, step: 1, default: 4}
    ],
    build: ({p}) => {
      const states = Math.round(p.states);
      const matrix = createTransitionMatrix(states, p.persistence, 0.15);
      const steps = 9;
      let distribution = Array.from({length: states}, (_, index) =>
        index === 0 ? 0.75 : 0.25 / (states - 1)
      );
      const history: number[][] = [distribution];
      const advance = (current: number[]) =>
        current.map((_, column) =>
          current.reduce((sum, value, row) => sum + value * matrix[row][column], 0)
        );
      for (let step = 0; step < steps; step++) {
        distribution = advance(distribution);
        history.push(distribution);
      }
      let steady = distribution;
      for (let step = 0; step < 200; step++) steady = advance(steady);
      const x = (step: number) => 12 + (step / steps) * 100;
      const y = (cumulative: number) => 90 - cumulative * 78;
      const bands: PolygonItem[] = [];
      const edges: PathItem[] = [];
      for (let state = 0; state < states; state++) {
        const lower = (distributionAt: number[]) =>
          distributionAt.slice(0, state).reduce((sum, value) => sum + value, 0);
        const top: Point[] = history.map(
          (entry, step): Point => [x(step), y(lower(entry) + entry[state])]
        );
        const bottom: Point[] = history
          .map((entry, step): Point => [x(step), y(lower(entry))])
          .reverse();
        const steadyLower = lower(steady);
        bands.push({
          ring: [...top, [150, y(steadyLower + steady[state])], [150, y(steadyLower)], ...bottom],
          role: getClassRole(state, states),
          alpha: 0.5,
          phase: state / states
        });
        edges.push({
          points: [
            [112, y(steadyLower + steady[state])],
            [150, y(steadyLower + steady[state])]
          ],
          role: 'ink',
          width: 0.6,
          alpha: 0.8
        });
      }
      return [
        {type: 'fills', items: bands, animation: 'pulse', blend: 'add'},
        {type: 'strokes', items: edges, animation: 'flow', speed: 0.25, blend: 'add'},
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [112, 8],
                [112, 92]
              ]
            ],
            'dim',
            0.4,
            0.6
          )
        }
      ];
    }
  },

  // ------------------------------------------------------------------------ region statistics
  'region-statistics': {
    parameters: [
      {id: 'radius', label: 'Region size', min: 8, max: 36, step: 1, default: 20, unit: 'px'},
      {
        id: 'shape',
        label: 'Shape',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['radius', 'rectangle']
      }
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const field = createSmoothField(random, 5, 1.2, 32, 20);
      const points: Point[] = Array.from({length: 130}, () => [
        random.range(8, 152),
        random.range(6, 62)
      ]);
      const getPointValue = (point: Point) =>
        sampleField(field, 32, 20, point[0] / WIDTH, point[1] / (HEIGHT * 0.65));
      const center: Point = pointer ? [pointer[0], clamp(pointer[1], 6, 62)] : [66, 34];
      const rectangle = p.shape >= 1;
      const isInside = (point: Point) =>
        rectangle
          ? Math.abs(point[0] - center[0]) <= p.radius * 1.2 &&
            Math.abs(point[1] - center[1]) <= p.radius * 0.8
          : getDistance(point, center) <= p.radius;
      const inside = points.filter(isInside);
      const outside = points.filter(point => !isInside(point));
      const insideValues = inside.map(getPointValue);
      const bins = 14;
      const counts = new Array<number>(bins).fill(0);
      for (const value of insideValues) counts[Math.min(bins - 1, Math.floor(value * bins))] += 1;
      const maximum = Math.max(1, ...counts);
      const mean = insideValues.length
        ? insideValues.reduce((sum, value) => sum + value, 0) / insideValues.length
        : 0.5;
      const extremes = inside.length
        ? [
            inside[insideValues.indexOf(Math.max(...insideValues))],
            inside[insideValues.indexOf(Math.min(...insideValues))]
          ]
        : [];
      const outline = rectangle
        ? ([
            [center[0] - p.radius * 1.2, center[1] - p.radius * 0.8],
            [center[0] + p.radius * 1.2, center[1] - p.radius * 0.8],
            [center[0] + p.radius * 1.2, center[1] + p.radius * 0.8],
            [center[0] - p.radius * 1.2, center[1] + p.radius * 0.8],
            [center[0] - p.radius * 1.2, center[1] - p.radius * 0.8]
          ] as Point[])
        : createCircle(center[0], center[1], p.radius, 48);
      return [
        {
          type: 'sprites',
          items: outside.map(
            (point): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: 1 + getPointValue(point),
              role: 'dim',
              alpha: 0.6
            })
          )
        },
        {
          type: 'strokes',
          items: toPaths([outline], 'warm', 0.9, 0.9),
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: inside.map(
            (point, index): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: 1.1 + insideValues[index] * 1.2,
              role: getValueRole(insideValues[index]),
              alpha: 1,
              phase: insideValues[index]
            })
          ),
          pulse: 0.3
        },
        {
          type: 'sprites',
          items: extremes.map(
            (point): SpriteItem => ({x: point[0], y: point[1], radius: 4, role: 'ink', alpha: 0.35})
          ),
          pulse: 0.4
        },
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [10, 92],
                [150, 92]
              ]
            ],
            'dim',
            0.4,
            0.6
          )
        },
        {
          type: 'fills',
          items: counts.map(
            (count, index): PolygonItem => ({
              ring: createRectRing(
                10 + index * 10 + 0.8,
                92 - (count / maximum) * 20,
                10 + (index + 1) * 10 - 0.8,
                92
              ),
              role: getClassRole(index, bins),
              alpha: 0.75,
              phase: index / bins
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [10 + mean * 140, 66],
                [10 + mean * 140, 94]
              ]
            ],
            'ink',
            0.9,
            0.9
          ),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        }
      ];
    }
  },

  'region-statistics-readback': {
    parameters: [
      {id: 'slots', label: 'Ring slots', min: 2, max: 6, step: 1, default: 4},
      {id: 'speed', label: 'Speed', min: 0.1, max: 1, step: 0.05, default: 0.35}
    ],
    build: ({fork, p}) => {
      const random = fork('points');
      const slots = Math.round(p.slots);
      const gpu = createRectRing(8, 22, 54, 78);
      const points = Array.from(
        {length: 40},
        (): Point => [random.range(12, 50), random.range(26, 74)]
      );
      const regionCenter: Point = [30, 50];
      const slotWidth = 10;
      const slotGap = 3;
      const ringWidth = slots * slotWidth + (slots - 1) * slotGap;
      const ringLeft = 80 - ringWidth / 2;
      const slotRings = Array.from({length: slots}, (_, index) =>
        createRectRing(
          ringLeft + index * (slotWidth + slotGap),
          43,
          ringLeft + index * (slotWidth + slotGap) + slotWidth,
          57
        )
      );
      const inside = points.filter(point => getDistance(point, regionCenter) < 13);
      return [
        {
          type: 'fills',
          items: [{ring: gpu, role: 'primary', alpha: 0.08}],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            [[...gpu, gpu[0]], [...createRectRing(122, 30, 152, 70), [122, 30]] as Point[]],
            'dim',
            0.7,
            0.7
          )
        },
        {
          type: 'sprites',
          items: toSprites(
            points.filter(point => !inside.includes(point)),
            'dim',
            1,
            random,
            0.6
          )
        },
        {type: 'sprites', items: toSprites(inside, 'warm', 1.4, random, 1), pulse: 0.3},
        {
          type: 'strokes',
          items: toPaths(
            [createCircle(regionCenter[0], regionCenter[1], 13, 36)],
            'warm',
            0.8,
            0.9
          ),
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'fills',
          items: slotRings.map(
            (ring, index): PolygonItem => ({
              ring,
              role: index === 0 ? 'hot' : index === slots - 1 ? 'cool' : 'primary',
              alpha: index === 0 || index === slots - 1 ? 0.8 : 0.35,
              phase: index / slots
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            slotRings.map(ring => [...ring, ring[0]]),
            'ink',
            0.5,
            0.7
          )
        },
        {
          type: 'strokes',
          items: [
            {
              points: [
                [56, 50],
                [ringLeft - 3, 50]
              ] as Point[],
              role: 'hot' as const,
              width: 1.2,
              alpha: 0.9
            },
            {
              points: [
                [ringLeft + ringWidth + 3, 50],
                [120, 50]
              ] as Point[],
              role: 'cool' as const,
              width: 1.2,
              alpha: 0.9
            }
          ],
          animation: 'flow',
          speed: p.speed,
          blend: 'add'
        },
        {
          type: 'fills',
          items: Array.from(
            {length: 5},
            (_, index): PolygonItem => ({
              ring: createRectRing(
                126 + index * 5,
                66 - (index === 2 ? 28 : 12 + index * 3),
                129.5 + index * 5,
                66
              ),
              role: 'cool',
              alpha: 0.7,
              phase: index / 5
            })
          ),
          animation: 'pulse',
          blend: 'add'
        }
      ];
    }
  },

  'region-mask': {
    parameters: [
      {id: 'size', label: 'Shape size', min: 12, max: 36, step: 1, default: 24, unit: 'px'},
      {id: 'roughness', label: 'Roughness', min: 0, max: 0.6, step: 0.05, default: 0.3}
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const points: Point[] = Array.from({length: 48}, () => [
        random.range(10, 150),
        random.range(8, 66)
      ]);
      const center: Point = pointer ? [pointer[0], clamp(pointer[1], 8, 66)] : [76, 36];
      const shape = createBlob(fork('shape'), center[0], center[1], p.size, 11, p.roughness, 0.8);
      const mask = points.map(point => isInsideRing(point, shape));
      const cell = 140 / points.length;
      return [
        {
          type: 'fills',
          items: [{ring: shape, role: 'warm', alpha: 0.12}],
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths([[...shape, shape[0]]], 'warm', 0.9, 0.9),
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: points.map(
            (point, index): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: mask[index] ? 2 : 1.2,
              role: mask[index] ? 'warm' : 'dim',
              alpha: mask[index] ? 1 : 0.6,
              phase: index / points.length
            })
          ),
          pulse: 0.3
        },
        {
          type: 'fills',
          items: mask.map(
            (on, index): PolygonItem => ({
              ring: createRectRing(
                10 + index * cell + 0.3,
                on ? 78 : 84,
                10 + (index + 1) * cell - 0.3,
                92
              ),
              role: on ? 'warm' : 'dim',
              alpha: on ? 0.9 : 0.35,
              phase: index / mask.length
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: mask.flatMap((on, index): PathItem[] =>
            on
              ? [
                  {
                    points: [points[index], [10 + (index + 0.5) * cell, 77]],
                    role: 'warm',
                    width: 0.3,
                    alpha: 0.35
                  }
                ]
              : []
          ),
          animation: 'static',
          blend: 'add'
        }
      ];
    }
  },

  'pick-region-mask': {
    parameters: [{id: 'objects', label: 'Objects', min: 4, max: 14, step: 1, default: 9}],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('objects');
      const count = Math.round(p.objects);
      const centers = createSpreadPoints(random, count, 14).map(
        ([x, y]): Point => [x, clamp(y, 12, 62)]
      );
      const objects = centers.map(([x, y]) =>
        createBlob(random, x, y, random.range(6, 10), 7, 0.3, 0.9)
      );
      const anchor: Point = [30, 14];
      const corner: Point = pointer ? [pointer[0], clamp(pointer[1], 8, 70)] : [98, 52];
      const [x0, x1] = [Math.min(anchor[0], corner[0]), Math.max(anchor[0], corner[0])];
      const [y0, y1] = [Math.min(anchor[1], corner[1]), Math.max(anchor[1], corner[1])];
      const picked = objects.map(ring => {
        const [bx0, by0, bx1, by1] = getRingBounds(ring);
        return bx1 >= x0 && bx0 <= x1 && by1 >= y0 && by0 <= y1;
      });
      const cell = Math.min(10, 140 / count);
      return [
        {
          type: 'fills',
          items: objects.map(
            (ring, index): PolygonItem => ({
              ring,
              role: picked[index] ? 'warm' : 'primary',
              alpha: picked[index] ? 0.7 : 0.22,
              phase: index / count
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            objects.map(ring => [...ring, ring[0]]),
            'primary',
            0.5,
            0.6
          )
        },
        {
          type: 'fills',
          items: [{ring: createRectRing(x0, y0, x1, y1), role: 'warm', alpha: 0.07}],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0]
              ]
            ],
            'warm',
            1,
            1
          ),
          animation: 'flow',
          speed: 0.4,
          trail: 3,
          blend: 'add'
        },
        {
          type: 'fills',
          items: picked.map(
            (on, index): PolygonItem => ({
              ring: createRectRing(
                80 - (count * cell) / 2 + index * cell + 0.6,
                on ? 80 : 86,
                80 - (count * cell) / 2 + (index + 1) * cell - 0.6,
                92
              ),
              role: on ? 'warm' : 'dim',
              alpha: on ? 0.9 : 0.35
            })
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [{x: corner[0], y: corner[1], radius: 2, role: 'ink', alpha: 1}],
          pulse: 0.3
        }
      ];
    }
  },

  'zonal-statistics': {
    parameters: [{id: 'resolution', label: 'Cells across', min: 16, max: 48, step: 8, default: 32}],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const tracts = createTracts(fork('zones'), 4, 3, 0.22);
      const columns = Math.round(p.resolution);
      const rows = Math.round((columns * HEIGHT) / WIDTH);
      const rasterNoise = fork('noise');
      const raster = createSmoothField(fork('raster'), 6, 1.4, columns, rows).map(value =>
        clamp(value + rasterNoise.gaussian() * 0.07, 0, 1)
      );
      const zones = new Int8Array(columns * rows);
      const sums = new Array<number>(tracts.rings.length).fill(0);
      const counts = new Array<number>(tracts.rings.length).fill(0);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const point: Point = [((column + 0.5) / columns) * WIDTH, ((row + 0.5) / rows) * HEIGHT];
          const zone = tracts.rings.findIndex(ring => isInsideRing(point, ring));
          zones[row * columns + column] = zone;
          if (zone >= 0) {
            sums[zone] += raster[row * columns + column];
            counts[zone] += 1;
          }
        }
      }
      const target = pointer ?? tracts.centers[5];
      const active = tracts.rings.findIndex(ring => isInsideRing(target, ring));
      const values = raster.map((value, index) =>
        zones[index] === active ? 0.1 + value * 0.9 : (0.08 + value * 0.6) * 0.6
      );
      return [
        getStaticField(columns, rows, values, 'ice', {filter: 'nearest', strength: 0.03}),
        {
          type: 'strokes',
          items: toPaths(
            tracts.rings.map(ring => [...ring, ring[0]]),
            'dim',
            0.7,
            0.7
          )
        },
        {
          type: 'strokes',
          items:
            active >= 0
              ? toPaths([[...tracts.rings[active], tracts.rings[active][0]]], 'warm', 1.3, 1)
              : [],
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: tracts.centers.map((center, index): SpriteItem => {
            const mean = counts[index] ? sums[index] / counts[index] : 0;
            return {
              x: center[0],
              y: center[1],
              radius: 1.2 + mean * 3.6,
              role: index === active ? 'warm' : getValueRole(mean),
              alpha: 1,
              phase: mean
            };
          }),
          pulse: 0.3
        }
      ];
    }
  },

  // -------------------------------------------------------------------------- regionalization
  'spatial-weights-minimum-spanning-tree': {
    parameters: [
      {id: 'noise', label: 'Attribute noise', min: 0, max: 1, step: 0.05, default: 0.25},
      {
        id: 'queen',
        label: 'Contiguity',
        min: 0,
        max: 1,
        step: 1,
        default: 1,
        options: ['rook', 'queen']
      }
    ],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('tracts'), 8, 5, 0.12);
      const noise = fork('noise');
      tracts.values = tracts.values.map(value =>
        clamp(value + noise.gaussian() * 0.25 * p.noise, 0, 1)
      );
      const edges = createContiguityEdges(tracts, p.queen >= 1);
      const tree = buildSpanningTree(tracts.centers.length, edges);
      const inTree = new Set(tree.map(edge => `${edge.a}-${edge.b}`));
      return [
        {
          type: 'strokes',
          items: edges
            .filter(edge => !inTree.has(`${edge.a}-${edge.b}`))
            .map(
              (edge): PathItem => ({
                points: [tracts.centers[edge.a], tracts.centers[edge.b]],
                role: 'dim',
                width: 0.4,
                alpha: 0.5
              })
            ),
          animation: 'static'
        },
        {
          type: 'strokes',
          items: tree.map(
            (edge): PathItem => ({
              points: [tracts.centers[edge.a], tracts.centers[edge.b]],
              role: 'warm',
              width: 1.4,
              alpha: 0.95
            })
          ),
          animation: 'flow',
          speed: 0.3,
          trail: 1.2,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: tracts.centers.map(
            (center, index): SpriteItem => ({
              x: center[0],
              y: center[1],
              radius: 2,
              role: getValueRole(tracts.values[index]),
              alpha: 1,
              phase: tracts.values[index]
            })
          ),
          pulse: 0.25
        }
      ];
    }
  },

  'skater-regions': {
    parameters: [{id: 'regions', label: 'Regions', min: 2, max: 10, step: 1, default: 5}],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('tracts'), 8, 5, 0.12);
      const tree = buildSpanningTree(tracts.centers.length, createContiguityEdges(tracts, false));
      let remaining = [...tree];
      const cuts: TreeEdge[] = [];
      let assignment = getForestComponents(tracts.centers.length, remaining);
      for (let cut = 0; cut < Math.round(p.regions) - 1; cut++) {
        let best = -1;
        let bestVariance = Infinity;
        remaining.forEach((edge, index) => {
          const trial = getForestComponents(
            tracts.centers.length,
            remaining.filter((_, other) => other !== index)
          );
          const count = Math.max(...trial) + 1;
          const sizes = getRegionSums(tracts.values, trial, count).sizes;
          if (
            Math.min(
              trial[edge.a] === trial[edge.b] ? 99 : sizes[trial[edge.a]],
              sizes[trial[edge.b]]
            ) < 3
          )
            return;
          const variance = getTotalVariance(tracts.values, trial, count);
          if (variance < bestVariance) {
            bestVariance = variance;
            best = index;
          }
        });
        if (best < 0) break;
        cuts.push(remaining[best]);
        remaining = remaining.filter((_, index) => index !== best);
        assignment = getForestComponents(tracts.centers.length, remaining);
      }
      return [
        ...buildRegionLayers(tracts, assignment, {alpha: 0.42}),
        {
          type: 'strokes',
          items: remaining.map(
            (edge): PathItem => ({
              points: [tracts.centers[edge.a], tracts.centers[edge.b]],
              role: 'ink',
              width: 0.6,
              alpha: 0.65
            })
          ),
          animation: 'flow',
          speed: 0.2,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: cuts.map(
            (edge): PathItem => ({
              points: [tracts.centers[edge.a], tracts.centers[edge.b]],
              role: 'hot',
              width: 1.8,
              alpha: 1
            })
          ),
          animation: 'pulse',
          speed: 0.7,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: tracts.centers.map(
            (center): SpriteItem => ({
              x: center[0],
              y: center[1],
              radius: 1,
              role: 'ink',
              alpha: 0.7
            })
          )
        }
      ];
    }
  },

  'ward-regions': {
    parameters: [{id: 'regions', label: 'Regions', min: 2, max: 14, step: 1, default: 7}],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('tracts'), 8, 5, 0.12);
      const count = tracts.centers.length;
      const assignment = Array.from({length: count}, (_, index) => index);
      const pairCost = (first: number, second: number) => {
        const a = getRegionSums(tracts.values, assignment, count);
        const meanA = a.sums[first] / a.sizes[first];
        const meanB = a.sums[second] / a.sizes[second];
        return (
          ((a.sizes[first] * a.sizes[second]) / (a.sizes[first] + a.sizes[second])) *
          (meanA - meanB) ** 2
        );
      };
      const findNextMerge = (): [number, number] => {
        let best: [number, number] = [0, 1];
        let bestCost = Infinity;
        tracts.neighbors.forEach((near, tract) => {
          for (const other of near) {
            if (assignment[tract] >= assignment[other]) continue;
            const cost = pairCost(assignment[tract], assignment[other]);
            if (cost < bestCost) {
              bestCost = cost;
              best = [assignment[tract], assignment[other]];
            }
          }
        });
        return best;
      };
      let regions = count;
      while (regions > Math.round(p.regions)) {
        const [keep, drop] = findNextMerge();
        for (let index = 0; index < count; index++)
          if (assignment[index] === drop) assignment[index] = keep;
        regions -= 1;
      }
      const [first, second] = findNextMerge();
      const centroid = (region: number): Point => {
        const members = assignment.flatMap((value, index) =>
          value === region ? [tracts.centers[index]] : []
        );
        return [
          members.reduce((sum, point) => sum + point[0], 0) / members.length,
          members.reduce((sum, point) => sum + point[1], 0) / members.length
        ];
      };
      const sizes = getRegionSums(tracts.values, assignment, count).sizes;
      const regionIds = [...new Set(assignment)];
      const rolesByRegion = new Map(regionIds.map((region, index) => [region, index]));
      const remapped = assignment.map(region => rolesByRegion.get(region) ?? 0);
      return [
        ...buildRegionLayers(tracts, remapped, {alpha: 0.45}),
        {
          type: 'strokes',
          items: [
            {
              points: [centroid(first), centroid(second)] as Point[],
              role: 'ink' as const,
              width: 1.6,
              alpha: 1
            }
          ],
          animation: 'pulse',
          speed: 0.7,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: regionIds.map((region): SpriteItem => {
            const [x, y] = centroid(region);
            return {
              x,
              y,
              radius: 1.2 + Math.sqrt(sizes[region]) * 0.9,
              role: 'ink',
              alpha: region === first || region === second ? 1 : 0.7,
              phase: region / count
            };
          }),
          pulse: 0.25
        }
      ];
    }
  },

  'azp-regions': {
    parameters: [{id: 'moves', label: 'Moves applied', min: 0, max: 14, step: 1, default: 5}],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('tracts'), 8, 5, 0.12);
      const regionCount = 4;
      const seedRandom = fork('seeds');
      const seeds = createSpreadPoints(seedRandom, regionCount, 12);
      const assignment = tracts.centers.map(center => {
        let best = 0;
        let bestDistance = Infinity;
        seeds.forEach((seed, index) => {
          const distance = getDistance(seed, center);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = index;
          }
        });
        return best;
      });
      const findMoves = () => {
        const base = getTotalVariance(tracts.values, assignment, regionCount);
        const moves: {tract: number; to: number; gain: number; target: number}[] = [];
        tracts.neighbors.forEach((near, tract) => {
          const from = assignment[tract];
          for (const other of near) {
            const to = assignment[other];
            if (to === from) continue;
            assignment[tract] = to;
            const valid =
              getTractAssignmentConnected(tracts, assignment, from) && assignment.includes(from);
            const gain = base - getTotalVariance(tracts.values, assignment, regionCount);
            assignment[tract] = from;
            if (valid && gain > 1e-6) moves.push({tract, to, gain, target: other});
          }
        });
        return moves.sort((first, second) => second.gain - first.gain);
      };
      for (let move = 0; move < Math.round(p.moves); move++) {
        const best = findMoves()[0];
        if (!best) break;
        assignment[best.tract] = best.to;
      }
      const pending = findMoves().slice(0, 6);
      return [
        ...buildRegionLayers(tracts, assignment, {alpha: 0.45}),
        {
          type: 'strokes',
          items: pending.map(
            (move): PathItem => ({
              points: [tracts.centers[move.tract], tracts.centers[move.target]],
              role: 'ink',
              width: 1.5,
              alpha: 1
            })
          ),
          animation: 'flow',
          speed: 0.5,
          trail: 1.5,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: pending.map(
            (move): SpriteItem => ({
              x: tracts.centers[move.tract][0],
              y: tracts.centers[move.tract][1],
              radius: 2.6,
              role: 'ink',
              alpha: 0.9
            })
          ),
          pulse: 0.5
        }
      ].concat([]) as PreviewLayer[];
    }
  },

  'max-p-regions': {
    parameters: [
      {id: 'threshold', label: 'Minimum total', min: 40, max: 220, step: 10, default: 90}
    ],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('tracts'), 8, 5, 0.12);
      const count = tracts.centers.length;
      const population = tracts.values.map(value => 6 + value * 26);
      const random = fork('growth');
      const order = Array.from({length: count}, (_, index) => ({index, key: random.next()}))
        .sort((first, second) => first.key - second.key)
        .map(entry => entry.index);
      const assignment = new Array<number>(count).fill(-1);
      const tried = new Set<number>();
      let regions = 0;
      for (const seed of order) {
        if (assignment[seed] >= 0 || tried.has(seed)) continue;
        const members = [seed];
        let total = population[seed];
        assignment[seed] = regions;
        while (total < p.threshold) {
          const frontier = members
            .flatMap(member => tracts.neighbors[member])
            .filter(tract => assignment[tract] < 0);
          if (frontier.length === 0) break;
          const next = frontier[0];
          assignment[next] = regions;
          members.push(next);
          total += population[next];
        }
        if (total >= p.threshold) regions += 1;
        else {
          members.forEach(member => (assignment[member] = -1));
          tried.add(seed);
        }
      }
      if (regions === 0) {
        assignment.fill(0);
        regions = 1;
      }
      for (let pass = 0; pass < count && assignment.includes(-1); pass++) {
        assignment.forEach((region, tract) => {
          if (region >= 0) return;
          const neighbor = tracts.neighbors[tract].find(other => assignment[other] >= 0);
          if (neighbor !== undefined) assignment[tract] = assignment[neighbor];
        });
      }
      const totals = new Array<number>(regions).fill(0);
      const centers = new Array<Point>(regions).fill([0, 0]).map((): Point => [0, 0]);
      const sizes = new Array<number>(regions).fill(0);
      assignment.forEach((region, tract) => {
        totals[region] += population[tract];
        centers[region][0] += tracts.centers[tract][0];
        centers[region][1] += tracts.centers[tract][1];
        sizes[region] += 1;
      });
      const ringRadius = 4.5;
      return [
        ...buildRegionLayers(tracts, assignment, {alpha: 0.42}),
        {
          type: 'strokes',
          items: toPaths(
            centers.map((center, region) =>
              createCircle(center[0] / sizes[region], center[1] / sizes[region], ringRadius, 24)
            ),
            'ink',
            0.5,
            0.7
          )
        },
        {
          type: 'sprites',
          items: centers.map(
            (center, region): SpriteItem => ({
              x: center[0] / sizes[region],
              y: center[1] / sizes[region],
              radius: ringRadius * Math.sqrt(totals[region] / p.threshold),
              role: 'ink',
              alpha: 0.5,
              phase: region / regions
            })
          ),
          pulse: 0.35
        }
      ];
    }
  },

  'region-partition-evaluation': {
    parameters: [
      {id: 'noise', label: 'Scramble', min: 0, max: 0.4, step: 0.02, default: 0.12},
      {id: 'regions', label: 'Regions', min: 3, max: 6, step: 1, default: 4}
    ],
    build: ({fork, p}) => {
      const tracts = createTracts(fork('tracts'), 8, 5, 0.12);
      const regionCount = Math.round(p.regions);
      const seeds = createSpreadPoints(fork('seeds'), regionCount, 14);
      const scramble = fork('scramble');
      const assignment = tracts.centers.map(center => {
        if (scramble.next() < p.noise) return scramble.integer(regionCount);
        let best = 0;
        let bestDistance = Infinity;
        seeds.forEach((seed, index) => {
          const distance = getDistance(seed, center);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = index;
          }
        });
        return best;
      });
      const shrink = (point: Point): Point => [4 + point[0] * 0.5, 14 + point[1] * 0.7];
      const small: Tracts = {
        ...tracts,
        rings: tracts.rings.map(ring => ring.map(shrink)),
        corners: tracts.corners.map(line => line.map(shrink)),
        centers: tracts.centers.map(shrink)
      };
      const {sums, sizes} = getRegionSums(tracts.values, assignment, regionCount);
      const deviation = new Array<number>(regionCount).fill(0);
      assignment.forEach((region, tract) => {
        deviation[region] += (tracts.values[tract] - sums[region] / sizes[region]) ** 2;
      });
      const maximumSize = Math.max(...sizes, 1);
      const maximumDeviation = Math.max(...deviation, 1e-6);
      const slot = 62 / regionCount;
      const connected = Array.from({length: regionCount}, (_, region) =>
        getTractAssignmentConnected(tracts, assignment, region)
      );
      return [
        ...buildRegionLayers(small, assignment, {alpha: 0.5}),
        {
          type: 'strokes',
          items: toPaths(
            [
              [
                [88, 82],
                [152, 82]
              ]
            ],
            'dim',
            0.5,
            0.7
          )
        },
        {
          type: 'fills',
          items: sizes.map(
            (size, region): PolygonItem => ({
              ring: createRectRing(
                90 + region * slot,
                82 - (size / maximumSize) * 52,
                90 + region * slot + slot * 0.5,
                82
              ),
              role: getRoleCycle(region),
              alpha: 0.8,
              phase: region / regionCount
            })
          ),
          animation: 'pulse',
          blend: 'add'
        },
        {
          type: 'fills',
          items: deviation.map(
            (value, region): PolygonItem => ({
              ring: createRectRing(
                90 + region * slot + slot * 0.55,
                82 - (value / maximumDeviation) * 30,
                90 + region * slot + slot * 0.85,
                82
              ),
              role: 'ink',
              alpha: 0.5
            })
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'sprites',
          items: connected.map(
            (ok, region): SpriteItem => ({
              x: 90 + region * slot + slot * 0.4,
              y: 90,
              radius: ok ? 1.6 : 2.4,
              role: ok ? 'cool' : 'hot',
              alpha: 1,
              phase: region / regionCount
            })
          ),
          pulse: 0.5
        }
      ];
    }
  }
};

// ---------------------------------------------------------------------------------------------
// Regionalization helpers

type TreeEdge = {a: number; b: number; weight: number};

function createContiguityEdges(tracts: Tracts, queen: boolean): TreeEdge[] {
  const edges: TreeEdge[] = [];
  const {columns, rows, values} = tracts;
  const add = (a: number, b: number) =>
    edges.push({a, b, weight: Math.abs(values[a] - values[b]) + 0.001});
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (column + 1 < columns) add(index, index + 1);
      if (row + 1 < rows) add(index, index + columns);
      if (queen && row + 1 < rows) {
        if (column + 1 < columns) add(index, index + columns + 1);
        if (column > 0) add(index, index + columns - 1);
      }
    }
  }
  return edges;
}

function buildSpanningTree(count: number, edges: TreeEdge[]): TreeEdge[] {
  const parent = Array.from({length: count}, (_, index) => index);
  const find = (node: number): number => {
    if (parent[node] !== node) parent[node] = find(parent[node]);
    return parent[node];
  };
  const tree: TreeEdge[] = [];
  for (const edge of [...edges].sort((first, second) => first.weight - second.weight)) {
    const rootA = find(edge.a);
    const rootB = find(edge.b);
    if (rootA === rootB) continue;
    parent[rootA] = rootB;
    tree.push(edge);
  }
  return tree;
}

/** Component index per node of a forest. */
function getForestComponents(count: number, edges: TreeEdge[]): number[] {
  const parent = Array.from({length: count}, (_, index) => index);
  const find = (node: number): number => {
    if (parent[node] !== node) parent[node] = find(parent[node]);
    return parent[node];
  };
  for (const edge of edges) parent[find(edge.a)] = find(edge.b);
  const labels = new Map<number, number>();
  return parent.map((_, node) => {
    const root = find(node);
    if (!labels.has(root)) labels.set(root, labels.size);
    return labels.get(root) ?? 0;
  });
}
