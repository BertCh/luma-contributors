// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/recipes-trajectories.ts, keyed by entry id.

import type {GpuSceneParameter} from '../../types.ts';
import {
  FIELD_COLUMNS,
  FIELD_ROWS,
  clamp,
  createArc,
  createBlob,
  createCircle,
  createClusterPoints,
  createScatterPoints,
  createSmoothField,
  createWanderPath,
  evaluateField,
  getDistance,
  mixNumbers,
  sampleField,
  type Random
} from '../data.ts';
import {
  clampPoint,
  getClassRole,
  getRoleCycle,
  toPaths,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type FillLayer,
  type PathItem,
  type PolygonItem,
  type Point,
  type PreviewLayer,
  type Role,
  type SpriteItem,
  type StrokeLayer
} from '../types.ts';

// ---------------------------------------------------------------------------------- local helpers

function slider(
  id: string,
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  unit?: string
): GpuSceneParameter {
  return {
    id,
    label,
    min,
    max,
    step,
    default: value,
    ...(unit ? {unit} : {})
  };
}

function strokeLayer(
  items: PathItem[],
  animation: StrokeLayer['animation'] = 'static',
  speed = 0.3,
  trail = 1,
  blend: 'add' | 'alpha' = 'add'
): StrokeLayer {
  return {type: 'strokes', items, animation, speed, trail, blend};
}

function fillLayer(
  items: PolygonItem[],
  animation: FillLayer['animation'] = 'static',
  blend: 'add' | 'alpha' = 'add'
): FillLayer {
  return {type: 'fills', items, animation, blend};
}

function spriteLayer(items: SpriteItem[], pulse = 0.4): PreviewLayer {
  return {type: 'sprites', items, pulse};
}

function createDot(
  x: number,
  y: number,
  radius: number,
  role: Role,
  alpha = 0.9,
  phase = 0
): SpriteItem {
  return {x, y, radius, role, alpha, phase};
}

function createLine(points: Point[], role: Role, width: number, alpha = 0.9, phase = 0): PathItem {
  return {points, role, width, alpha, phase};
}

function closeRing(ring: Point[]): Point[] {
  return [...ring, ring[0]];
}

/** Shaft plus head of an arrow, as two layers: a flowing shaft and a static head. */
function createArrowLayers(from: Point, to: Point, role: Role = 'warm', size = 3): PreviewLayer[] {
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const wing = (offset: number): Point => [
    to[0] - Math.cos(angle + offset) * size,
    to[1] - Math.sin(angle + offset) * size
  ];
  return [
    strokeLayer([createLine([from, to], role, 0.8, 0.8)], 'flow', 0.35, 0.8),
    strokeLayer([createLine([wing(0.5), to, wing(-0.5)], role, 0.8, 0.9)])
  ];
}

function createCellRing(x: number, y: number, size: number): Point[] {
  const gap = size * 0.07;
  return [
    [x + gap, y + gap],
    [x + size - gap, y + gap],
    [x + size - gap, y + size - gap],
    [x + gap, y + size - gap]
  ];
}

function createLatticeLines(
  x: number,
  y: number,
  columns: number,
  rows: number,
  size: number,
  alpha = 0.16
): PathItem[] {
  const items: PathItem[] = [];
  for (let column = 0; column <= columns; column++) {
    items.push(
      createLine(
        [
          [x + column * size, y],
          [x + column * size, y + rows * size]
        ],
        'dim',
        0.2,
        alpha
      )
    );
  }
  for (let row = 0; row <= rows; row++) {
    items.push(
      createLine(
        [
          [x, y + row * size],
          [x + columns * size, y + row * size]
        ],
        'dim',
        0.2,
        alpha
      )
    );
  }
  return items;
}

/** Quad zones sharing jittered corners; ring order is top-left, top-right, bottom-right, bottom-left. */
function createZoneQuads(
  random: Random,
  columns: number,
  rows: number,
  x: number,
  y: number,
  width: number,
  height: number,
  jitter: number
): Point[][] {
  const corners: Point[][] = [];
  for (let row = 0; row <= rows; row++) {
    const line: Point[] = [];
    for (let column = 0; column <= columns; column++) {
      const shiftX = random.range(-1, 1) * jitter * (width / columns);
      const shiftY = random.range(-1, 1) * jitter * (height / rows);
      const isEdge = column === 0 || column === columns || row === 0 || row === rows;
      line.push([
        x + (column / columns) * width + (isEdge ? 0 : shiftX),
        y + (row / rows) * height + (isEdge ? 0 : shiftY)
      ]);
    }
    corners.push(line);
  }
  const zones: Point[][] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      zones.push([
        corners[row][column],
        corners[row][column + 1],
        corners[row + 1][column + 1],
        corners[row + 1][column]
      ]);
    }
  }
  return zones;
}

function getCentroid(ring: Point[]): Point {
  let x = 0;
  let y = 0;
  for (const point of ring) {
    x += point[0];
    y += point[1];
  }
  return [x / ring.length, y / ring.length];
}

function isInsideRing(point: Point, ring: Point[]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[index];
    const [bx, by] = ring[previous];
    if (
      ay > point[1] !== by > point[1] &&
      point[0] < ((bx - ax) * (point[1] - ay)) / (by - ay) + ax
    ) {
      inside = !inside;
    }
  }
  return inside;
}

/** Bilinear point inside a quad given as [top-left, top-right, bottom-right, bottom-left]. */
function getQuadPoint(quad: Point[], u: number, v: number): Point {
  const top: Point = [mixNumbers(quad[0][0], quad[1][0], u), mixNumbers(quad[0][1], quad[1][1], u)];
  const bottom: Point = [
    mixNumbers(quad[3][0], quad[2][0], u),
    mixNumbers(quad[3][1], quad[2][1], u)
  ];
  return [mixNumbers(top[0], bottom[0], v), mixNumbers(top[1], bottom[1], v)];
}

function getConvexHull(points: Point[]): Point[] {
  const sorted = [...points].sort((first, second) => first[0] - second[0] || first[1] - second[1]);
  if (sorted.length < 3) return sorted;
  const cross = (origin: Point, first: Point, second: Point) =>
    (first[0] - origin[0]) * (second[1] - origin[1]) -
    (first[1] - origin[1]) * (second[0] - origin[0]);
  const lower: Point[] = [];
  for (const point of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0
    ) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper: Point[] = [];
  for (const point of [...sorted].reverse()) {
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0
    ) {
      upper.pop();
    }
    upper.push(point);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function getStandardScores(values: number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const deviation =
    Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length) || 1;
  return values.map(value => (value - mean) / deviation);
}

/** Mean of the values of the neighbors (not the cell itself) of a cell on a lattice. */
function getLatticeLag(
  values: number[],
  columns: number,
  rows: number,
  column: number,
  row: number,
  order = 1
): number {
  let sum = 0;
  let count = 0;
  for (let rowOffset = -order; rowOffset <= order; rowOffset++) {
    for (let columnOffset = -order; columnOffset <= order; columnOffset++) {
      const neighborColumn = column + columnOffset;
      const neighborRow = row + rowOffset;
      if (rowOffset === 0 && columnOffset === 0) continue;
      if (
        neighborColumn < 0 ||
        neighborRow < 0 ||
        neighborColumn >= columns ||
        neighborRow >= rows
      ) {
        continue;
      }
      sum += values[neighborRow * columns + neighborColumn];
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

function getBoxBlur(
  values: Float32Array,
  columns: number,
  rows: number,
  radius: number
): Float32Array {
  if (radius < 1) return values;
  const horizontal = new Float32Array(values.length);
  const result = new Float32Array(values.length);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      let sum = 0;
      let count = 0;
      for (let offset = -radius; offset <= radius; offset++) {
        const sampleColumn = column + offset;
        if (sampleColumn < 0 || sampleColumn >= columns) continue;
        sum += values[row * columns + sampleColumn];
        count++;
      }
      horizontal[row * columns + column] = sum / count;
    }
  }
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      let sum = 0;
      let count = 0;
      for (let offset = -radius; offset <= radius; offset++) {
        const sampleRow = row + offset;
        if (sampleRow < 0 || sampleRow >= rows) continue;
        sum += horizontal[sampleRow * columns + column];
        count++;
      }
      result[row * columns + column] = sum / count;
    }
  }
  return result;
}

/** A smooth wavy track across the scene, u in 0..1. Different phases give different tracks. */
function getCurvePoint(u: number, phase: number, amplitude = 24, baseline = 44): Point {
  return [
    12 + 136 * u,
    baseline +
      amplitude * Math.sin(u * 5.6 + phase) +
      amplitude * 0.25 * Math.sin(u * 13 + phase * 2)
  ];
}

function getCumulative(points: Point[]): number[] {
  const lengths = [0];
  for (let index = 1; index < points.length; index++) {
    lengths.push(lengths[index - 1] + getDistance(points[index - 1], points[index]));
  }
  return lengths;
}

/** The point at a fraction of the arc length of a polyline. */
function getPointAtFraction(points: Point[], lengths: number[], fraction: number): Point {
  const target = clamp(fraction, 0, 1) * lengths[lengths.length - 1];
  let index = 1;
  while (index < lengths.length - 1 && lengths[index] < target) index++;
  const span = lengths[index] - lengths[index - 1] || 1;
  const amount = (target - lengths[index - 1]) / span;
  return [
    mixNumbers(points[index - 1][0], points[index][0], amount),
    mixNumbers(points[index - 1][1], points[index][1], amount)
  ];
}

function createCrossingTrack(
  random: Random,
  steps: number,
  wiggle: number,
  xStart = 6,
  xEnd = 154
): Point[] {
  const yStart = random.range(14, 86);
  const yEnd = random.range(14, 86);
  const frequency = random.range(1.5, 3.5);
  const phase = random.range(0, Math.PI * 2);
  const amplitude = random.range(4, 10) * wiggle;
  return Array.from({length: steps + 1}, (_, index): Point => {
    const u = index / steps;
    return [
      mixNumbers(xStart, xEnd, u),
      clamp(
        mixNumbers(yStart, yEnd, u) + amplitude * Math.sin(u * Math.PI * frequency + phase),
        6,
        94
      )
    ];
  });
}

function createBars(
  originX: number,
  baseline: number,
  heights: number[],
  barWidth: number,
  spacing: number,
  role: Role,
  alpha = 0.85
): PathItem[] {
  return heights.map((height, index) =>
    createLine(
      [
        [originX + index * spacing, baseline],
        [originX + index * spacing, baseline - Math.max(height, 0.4)]
      ],
      role,
      barWidth,
      alpha,
      index / heights.length
    )
  );
}

/** A gently curving track that starts anywhere and heads through the middle of the scene. */
function createMeetingTrack(random: Random, steps: number): Point[] {
  const angle = random.range(0, Math.PI * 2);
  const reach = random.range(46, 70);
  const phase = random.range(0, 6);
  const bend = random.range(-9, 9);
  const center: Point = [80 + random.range(-22, 22), 50 + random.range(-16, 16)];
  return Array.from({length: steps + 1}, (_, index): Point => {
    const u = (index / steps) * 2 - 1;
    const sideways = bend * Math.sin(u * 2 + phase);
    return clampPoint(
      [
        center[0] + Math.cos(angle) * reach * u - Math.sin(angle) * sideways,
        center[1] + Math.sin(angle) * reach * u * 0.7 + Math.cos(angle) * sideways
      ],
      5
    );
  });
}

export const RECIPES_TRAJECTORIES_PREVIEWS: Record<string, EntryPreview> = {
  // ------------------------------------------------------------ hot spots, rates and regression

  'hot-spot-analysis-recipe': {
    parameters: [
      slider('events', 'Events', 40, 260, 10, 150),
      slider('threshold', 'Significance', 0.3, 2.5, 0.1, 1, 'σ')
    ],
    build: ({fork, p}) => {
      const columns = 8;
      const rows = 11;
      const size = 7;
      const leftX = 9;
      const rightX = 96;
      const top = 11;
      const random = fork('events');
      const hotCenter: Point = [leftX + size * 5.4, top + size * 3.6];
      const coldCenter: Point = [leftX + size * 2.2, top + size * 8];
      const points: Point[] = [];
      const total = Math.round(p.events);
      while (points.length < total * 0.55) {
        const point: Point = [
          random.range(leftX, leftX + columns * size),
          random.range(top, top + rows * size)
        ];
        if (getDistance(point, coldCenter) < 17 && random.next() > 0.12) continue;
        points.push(point);
      }
      while (points.length < total) {
        points.push([
          clamp(hotCenter[0] + random.gaussian() * 7, leftX + 0.5, leftX + columns * size - 0.5),
          clamp(hotCenter[1] + random.gaussian() * 7, top + 0.5, top + rows * size - 0.5)
        ]);
      }
      const counts = new Array<number>(columns * rows).fill(0);
      for (const [x, y] of points) {
        const column = clamp(Math.floor((x - leftX) / size), 0, columns - 1);
        const row = clamp(Math.floor((y - top) / size), 0, rows - 1);
        counts[row * columns + column]++;
      }
      const mean = total / counts.length;
      const deviation =
        Math.sqrt(counts.reduce((sum, count) => sum + (count - mean) ** 2, 0) / counts.length) || 1;
      const hot: PolygonItem[] = [];
      const cool: PolygonItem[] = [];
      const quiet: PolygonItem[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          let sum = 0;
          let count = 0;
          for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
              const neighborColumn = column + columnOffset;
              const neighborRow = row + rowOffset;
              if (
                neighborColumn < 0 ||
                neighborRow < 0 ||
                neighborColumn >= columns ||
                neighborRow >= rows
              ) {
                continue;
              }
              sum += counts[neighborRow * columns + neighborColumn];
              count++;
            }
          }
          const score = (sum - count * mean) / (deviation * Math.sqrt(count));
          const ring = createCellRing(rightX + column * size, top + row * size, size);
          if (score > p.threshold) {
            hot.push({
              ring,
              role: 'hot',
              alpha: clamp(0.35 + 0.22 * score, 0.35, 0.9),
              phase: column / columns
            });
          } else if (score < -p.threshold) {
            cool.push({
              ring,
              role: 'cool',
              alpha: clamp(0.3 + 0.2 * -score, 0.3, 0.8)
            });
          } else {
            quiet.push({ring, role: 'dim', alpha: 0.2});
          }
        }
      }
      return [
        strokeLayer(createLatticeLines(leftX, top, columns, rows, size)),
        spriteLayer(toSprites(points, 'primary', 1, fork('size'), 0.85), 0.3),
        ...createArrowLayers([77, 50], [92, 50]),
        fillLayer(quiet),
        fillLayer(cool),
        fillLayer(hot, 'pulse')
      ];
    }
  },

  'space-time-hot-spots-recipe': {
    parameters: [
      slider('slices', 'Time slices', 3, 6, 1, 5),
      slider('hot', 'Hot cells', 1, 5, 1, 3)
    ],
    build: ({fork, p}) => {
      const columns = 6;
      const rows = 5;
      const size = 6;
      const sliceCount = Math.round(p.slices);
      const stepY = Math.min(17, 66 / Math.max(sliceCount - 1, 1));
      const orderRandom = fork('order');
      const order = Array.from({length: columns * rows}, (_, index) => ({
        index,
        key: orderRandom.next()
      }));
      order.sort((first, second) => first.key - second.key);
      const hotCount = Math.round(p.hot);
      const categoryOf = new Map<number, number>();
      order.slice(0, hotCount * 4).forEach((entry, position) => {
        categoryOf.set(entry.index, Math.floor(position / hotCount));
      });
      const flicker = fork('flicker');
      const project = (column: number, row: number, slice: number): Point => [
        20 + column * size + row * size * 0.5 + slice * 2,
        84 - slice * stepY + row * size * 0.42
      ];
      const roles: Role[] = ['hot', 'warm', 'violet', 'cool'];
      const slices: PreviewLayer[] = [];
      const fillsByRole: PolygonItem[] = [];
      const quiet: PolygonItem[] = [];
      const frames: PathItem[] = [];
      const spine: PathItem[] = [];
      for (let slice = 0; slice < sliceCount; slice++) {
        frames.push(
          createLine(
            closeRing([
              project(0, 0, slice),
              project(columns, 0, slice),
              project(columns, rows, slice),
              project(0, rows, slice)
            ]),
            'dim',
            0.4,
            0.5
          )
        );
        for (let row = 0; row < rows; row++) {
          for (let column = 0; column < columns; column++) {
            const index = row * columns + column;
            const category = categoryOf.get(index);
            const randomHot = flicker.next() > 0.5;
            const isHot =
              category === 0 ||
              (category === 1 && slice >= sliceCount - 2) ||
              (category === 2 && randomHot) ||
              (category === 3 && slice < sliceCount / 2);
            const ring = [
              project(column, row, slice),
              project(column + 1, row, slice),
              project(column + 1, row + 1, slice),
              project(column, row + 1, slice)
            ];
            if (isHot && category !== undefined) {
              fillsByRole.push({
                ring,
                role: roles[category],
                alpha: 0.75,
                phase: slice / sliceCount
              });
            } else {
              quiet.push({ring, role: 'dim', alpha: 0.1});
            }
          }
        }
      }
      categoryOf.forEach((category, index) => {
        if (category !== 0) return;
        const column = index % columns;
        const row = Math.floor(index / columns);
        spine.push(
          createLine(
            [project(column + 0.5, row + 0.5, 0), project(column + 0.5, row + 0.5, sliceCount - 1)],
            'hot',
            0.7,
            0.8
          )
        );
      });
      slices.push(strokeLayer(frames), fillLayer(quiet), fillLayer(fillsByRole, 'pulse'));
      slices.push(strokeLayer(spine, 'flow', 0.35, 1.2));
      // Flat result map: one emerging class per cell.
      const result: PolygonItem[] = [];
      for (let index = 0; index < columns * rows; index++) {
        const category = categoryOf.get(index);
        result.push({
          ring: createCellRing(
            118 + (index % columns) * 5.6,
            34 + Math.floor(index / columns) * 5.6,
            5.6
          ),
          role: category === undefined ? 'dim' : roles[category],
          alpha: category === undefined ? 0.15 : 0.85
        });
      }
      return [...slices, ...createArrowLayers([94, 50], [112, 50]), fillLayer(result, 'pulse')];
    }
  },

  'rate-cluster-map-recipe': {
    parameters: [
      slider('noise', 'Small-area noise', 0, 0.5, 0.02, 0.28),
      slider('shrinkage', 'Smoothing', 0, 1, 0.05, 0.65)
    ],
    build: ({fork, p}) => {
      const columns = 5;
      const rows = 4;
      const zonesLeft = createZoneQuads(fork('zones'), columns, rows, 6, 14, 62, 64, 0.12);
      const zonesRight = zonesLeft.map(ring => ring.map(([x, y]): Point => [x + 88, y]));
      const truth = createSmoothField(fork('truth'), 4, 0.9, columns, rows);
      const noise = fork('noise');
      const raw = Array.from(truth, value => value + noise.gaussian() * p.noise);
      const mean = raw.reduce((sum, value) => sum + value, 0) / raw.length;
      const smoothed = raw.map((value, index) => {
        const lag = getLatticeLag(raw, columns, rows, index % columns, Math.floor(index / columns));
        return mixNumbers(value, mixNumbers(lag, mean, 0.3), p.shrinkage);
      });
      const scores = getStandardScores(smoothed);
      const quadrants: PolygonItem[][] = [[], [], [], [], []];
      const rawFills: PolygonItem[] = zonesLeft.map((ring, index) => ({
        ring,
        role: 'primary',
        alpha: clamp(0.1 + 0.8 * (raw[index] - mean + 0.5), 0.08, 0.9)
      }));
      zonesRight.forEach((ring, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const lag = getLatticeLag(scores, columns, rows, column, row);
        const score = scores[index];
        const significant = Math.abs(score * lag) > 0.18;
        let group = 4;
        if (significant) group = score > 0 ? (lag > 0 ? 0 : 2) : lag < 0 ? 1 : 3;
        const roleByGroup: Role[] = ['hot', 'cool', 'warm', 'violet', 'dim'];
        quadrants[group].push({
          ring,
          role: roleByGroup[group],
          alpha: group === 4 ? 0.14 : 0.7,
          phase: index / 20
        });
      });
      return [
        fillLayer(rawFills),
        strokeLayer(toPaths([...zonesLeft, ...zonesRight].map(closeRing), 'dim', 0.35, 0.7)),
        ...createArrowLayers([70, 46], [86, 46]),
        fillLayer(quadrants[4]),
        fillLayer([...quadrants[1], ...quadrants[2], ...quadrants[3]]),
        fillLayer(quadrants[0], 'pulse')
      ];
    }
  },

  'excess-risk-recipe': {
    parameters: [
      slider('excess', 'Excess', 0, 2, 0.1, 1),
      slider('noise', 'Noise', 0, 0.4, 0.02, 0.15)
    ],
    build: ({fork, p}) => {
      const columns = 4;
      const rows = 3;
      const zones = createZoneQuads(fork('zones'), columns, rows, 8, 8, 144, 84, 0.1);
      const field = createSmoothField(fork('risk'), 4, 0.9, columns, rows);
      const noise = fork('noise');
      const fills: PolygonItem[] = [];
      const disks: SpriteItem[] = [];
      const rings: PathItem[] = [];
      zones.forEach((ring, index) => {
        const ratio = Math.max(
          0.25,
          1 + (field[index] - 0.5) * 1.3 * p.excess + noise.gaussian() * p.noise
        );
        const role: Role = ratio > 1.15 ? 'hot' : ratio < 0.87 ? 'cool' : 'primary';
        const [x, y] = getCentroid(ring);
        const expected = 8;
        fills.push({
          ring,
          role,
          alpha: role === 'primary' ? 0.07 : clamp(Math.abs(ratio - 1) * 0.4, 0.08, 0.3)
        });
        rings.push(createLine(closeRing(createCircle(x, y, expected, 36)), 'ink', 0.5, 0.6));
        disks.push(createDot(x, y, expected * Math.sqrt(ratio), role, 0.8, index / zones.length));
      });
      return [
        fillLayer(fills),
        strokeLayer(toPaths(zones.map(closeRing), 'dim', 0.35, 0.6)),
        spriteLayer(disks, 0.25),
        strokeLayer(rings, 'pulse', 0.25)
      ];
    }
  },

  'spatial-rate-recipe': {
    parameters: [
      slider('noise', 'Noise', 0.05, 0.7, 0.05, 0.4),
      slider('order', 'Ring', 1, 2, 1, 1)
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const columns = 7;
      const rows = 5;
      const zones = createZoneQuads(fork('zones'), columns, rows, 6, 8, 148, 84, 0.1);
      const truth = createSmoothField(fork('truth'), 4, 0.9, columns, rows);
      const noise = fork('noise');
      const population = zones.map(() => noise.range(0.5, 1.6));
      const rates = zones.map((_, index) =>
        Math.max(0.02, truth[index] + noise.gaussian() * p.noise)
      );
      const order = Math.round(p.order);
      const pooled = zones.map((_, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        let events = 0;
        let people = 0;
        for (let rowOffset = -order; rowOffset <= order; rowOffset++) {
          for (let columnOffset = -order; columnOffset <= order; columnOffset++) {
            const c = column + columnOffset;
            const r = row + rowOffset;
            if (c < 0 || r < 0 || c >= columns || r >= rows) continue;
            events += rates[r * columns + c] * population[r * columns + c];
            people += population[r * columns + c];
          }
        }
        return events / people;
      });
      const low = Math.min(...pooled);
      const range = Math.max(...pooled) - low || 1;
      let focus = Math.floor(rows / 2) * columns + Math.floor(columns / 2);
      if (pointer) {
        const found = zones.findIndex(ring => isInsideRing(pointer, ring));
        if (found >= 0) focus = found;
      }
      const focusColumn = focus % columns;
      const focusRow = Math.floor(focus / columns);
      const centroids = zones.map(getCentroid);
      const links: PathItem[] = [];
      const outlines: PathItem[] = [];
      zones.forEach((ring, index) => {
        const isNeighbor =
          index !== focus &&
          Math.abs((index % columns) - focusColumn) <= order &&
          Math.abs(Math.floor(index / columns) - focusRow) <= order;
        if (isNeighbor) {
          links.push(
            createLine([centroids[index], centroids[focus]], 'warm', 0.8, 0.8, index / zones.length)
          );
          outlines.push(createLine(closeRing(ring), 'warm', 0.6, 0.8));
        }
      });
      return [
        fillLayer(
          zones.map((ring, index) => ({
            ring,
            role: getClassRole(Math.min(4, Math.floor(((pooled[index] - low) / range) * 5)), 5),
            alpha: 0.4
          }))
        ),
        strokeLayer(toPaths(zones.map(closeRing), 'dim', 0.35, 0.6)),
        strokeLayer(outlines),
        strokeLayer(links, 'flow', 0.4, 0.9),
        strokeLayer([createLine(closeRing(zones[focus]), 'ink', 1.1, 0.95)], 'pulse', 0.5),
        spriteLayer([createDot(centroids[focus][0], centroids[focus][1], 2.6, 'warm', 1)], 0.5)
      ];
    }
  },

  'local-variability-recipe': {
    parameters: [
      slider('erratic', 'Erratic area', 0, 1.2, 0.05, 0.8),
      slider('threshold', 'LOSH cutoff', 0.3, 1.6, 0.05, 0.8)
    ],
    build: ({fork, p}) => {
      const columns = 10;
      const rows = 8;
      const size = 6.2;
      const smooth = createSmoothField(fork('smooth'), 4, 0.8, columns, rows);
      const chaos = fork('chaos');
      const values = Array.from(smooth, (value, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const erratic = Math.hypot(column - 7, row - 3.5) < 2.4 ? chaos.gaussian() * p.erratic : 0;
        return value * 1.5 + erratic;
      });
      const scores = getStandardScores(values);
      const lags = scores.map((_, index) =>
        getLatticeLag(scores, columns, rows, index % columns, Math.floor(index / columns))
      );
      const dispersion = scores.map((score, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        let sum = (score - lags[index]) ** 2;
        let count = 1;
        for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
          for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
            const c = column + columnOffset;
            const r = row + rowOffset;
            if ((!rowOffset && !columnOffset) || c < 0 || r < 0 || c >= columns || r >= rows)
              continue;
            sum += (scores[r * columns + c] - lags[index]) ** 2;
            count++;
          }
        }
        return sum / count;
      });
      const left: PolygonItem[] = [];
      const right: PolygonItem[] = [];
      const masked: Point[][] = [];
      scores.forEach((score, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const masks = column === 0 && row >= 5;
        const ringLeft = createCellRing(8 + column * size, 24 + row * size, size);
        const ringRight = createCellRing(90 + column * size, 24 + row * size, size);
        if (masks) {
          masked.push(closeRing(ringLeft), closeRing(ringRight));
          return;
        }
        const association = score * lags[index];
        left.push({
          ring: ringLeft,
          role: association > 0.2 ? (score > 0 ? 'hot' : 'cool') : 'dim',
          alpha: association > 0.2 ? 0.75 : 0.14,
          phase: column / columns
        });
        const loud = dispersion[index] > p.threshold;
        right.push({
          ring: ringRight,
          role: loud ? 'violet' : 'dim',
          alpha: loud ? clamp(0.3 + dispersion[index] * 0.4, 0.4, 0.95) : 0.14,
          phase: row / rows
        });
      });
      return [
        fillLayer(left.filter(item => item.role === 'dim')),
        fillLayer(
          left.filter(item => item.role !== 'dim'),
          'pulse'
        ),
        fillLayer(right.filter(item => item.role === 'dim')),
        fillLayer(
          right.filter(item => item.role !== 'dim'),
          'pulse'
        ),
        strokeLayer(toPaths(masked, 'dim', 0.3, 0.5)),
        spriteLayer(
          [
            createDot(80, 50, 1.4, 'ink', 0.7),
            createDot(75, 50, 0.9, 'dim', 0.6),
            createDot(85, 50, 0.9, 'dim', 0.6)
          ],
          0.3
        )
      ];
    }
  },

  'quantile-lisa': {
    parameters: [
      slider('quantiles', 'Quantiles', 3, 8, 1, 5),
      slider('target', 'Class', 1, 8, 1, 5)
    ],
    build: ({fork, p}) => {
      const columns = 16;
      const rows = 10;
      const size = 8;
      const values = createSmoothField(fork('values'), 7, 1.3, columns, rows);
      const ranks = Array.from(values, (value, index) => ({
        value,
        index
      })).sort((first, second) => first.value - second.value);
      const quantiles = Math.round(p.quantiles);
      const target = Math.min(Math.round(p.target), quantiles) - 1;
      const classOf = new Array<number>(values.length).fill(0);
      ranks.forEach(
        (entry, position) =>
          (classOf[entry.index] = Math.floor((position / ranks.length) * quantiles))
      );
      const background: PolygonItem[] = [];
      const clustered: PolygonItem[] = [];
      const isolated: PolygonItem[] = [];
      const joins: PathItem[] = [];
      const center = (index: number): Point => [
        16 + (index % columns) * size + size / 2,
        10 + Math.floor(index / columns) * size + size / 2
      ];
      for (let index = 0; index < values.length; index++) {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const ring = createCellRing(16 + column * size, 10 + row * size, size);
        if (classOf[index] !== target) {
          background.push({
            ring,
            role: 'cool',
            alpha: 0.06 + 0.1 * (classOf[index] / quantiles)
          });
          continue;
        }
        let count = 0;
        const neighbors: number[] = [];
        if (column > 0) neighbors.push(index - 1);
        if (column < columns - 1) neighbors.push(index + 1);
        if (row > 0) neighbors.push(index - columns);
        if (row < rows - 1) neighbors.push(index + columns);
        for (const neighbor of neighbors) {
          if (classOf[neighbor] !== target) continue;
          count++;
          if (neighbor > index)
            joins.push(
              createLine([center(index), center(neighbor)], 'hot', 0.9, 0.9, index / values.length)
            );
        }
        (count >= 2 ? clustered : isolated).push({
          ring,
          role: count >= 2 ? 'hot' : 'warm',
          alpha: count >= 2 ? 0.7 : 0.28,
          phase: index / values.length
        });
      }
      return [
        fillLayer(background),
        fillLayer(isolated),
        fillLayer(clustered, 'pulse'),
        strokeLayer(joins, 'flow', 0.35, 0.8)
      ];
    }
  },

  'spatial-regression-recipe': {
    parameters: [
      slider('dependence', 'Residual pattern', 0, 1, 0.05, 0.85),
      slider('noise', 'Noise', 4, 20, 0.5, 11, 'px')
    ],
    build: ({fork, p}) => {
      const columns = 6;
      const rows = 6;
      const smooth = createSmoothField(fork('pattern'), 3, 0.8, columns, rows);
      const noise = fork('noise');
      const residuals = Array.from(smooth, value => {
        const pattern = (value - 0.5) * 2;
        return p.dependence * pattern + (1 - p.dependence) * noise.gaussian() * 0.6;
      });
      const xs = residuals.map(() => noise.range(12, 72));
      const ys = residuals.map(
        (residual, index) => 70 - (xs[index] - 12) * 0.6 - residual * p.noise
      );
      const count = xs.length;
      const meanX = xs.reduce((sum, value) => sum + value, 0) / count;
      const meanY = ys.reduce((sum, value) => sum + value, 0) / count;
      let numerator = 0;
      let denominator = 0;
      xs.forEach((x, index) => {
        numerator += (x - meanX) * (ys[index] - meanY);
        denominator += (x - meanX) ** 2;
      });
      const slope = numerator / denominator;
      const intercept = meanY - slope * meanX;
      const drops: PathItem[] = [];
      const dots: SpriteItem[] = [];
      const cells: PolygonItem[] = [];
      xs.forEach((x, index) => {
        const fitted = intercept + slope * x;
        const residual = fitted - ys[index];
        const role: Role = residual > 1.5 ? 'hot' : residual < -1.5 ? 'cool' : 'dim';
        drops.push(
          createLine(
            [
              [x, ys[index]],
              [x, fitted]
            ],
            role === 'dim' ? 'violet' : role,
            0.5,
            0.65
          )
        );
        dots.push(
          createDot(x, ys[index], 1.6, role === 'dim' ? 'primary' : role, 0.95, index / count)
        );
        cells.push({
          ring: createCellRing(98 + (index % columns) * 8, 26 + Math.floor(index / columns) * 8, 8),
          role: role === 'dim' ? 'dim' : role,
          alpha: role === 'dim' ? 0.2 : clamp(0.35 + Math.abs(residual) * 0.04, 0.4, 0.9),
          phase: (index % columns) / columns
        });
      });
      return [
        strokeLayer([
          createLine(
            [
              [10, 90],
              [76, 90]
            ],
            'dim',
            0.4,
            0.5
          ),
          createLine(
            [
              [10, 90],
              [10, 12]
            ],
            'dim',
            0.4,
            0.5
          )
        ]),
        strokeLayer(drops),
        strokeLayer(
          [
            createLine(
              [
                [12, intercept + slope * 12],
                [72, intercept + slope * 72]
              ],
              'warm',
              1.1,
              0.95
            )
          ],
          'flow',
          0.3,
          1.2
        ),
        spriteLayer(dots, 0.3),
        ...createArrowLayers([79, 50], [94, 50]),
        fillLayer(cells.filter(cell => cell.role === 'dim')),
        fillLayer(
          cells.filter(cell => cell.role !== 'dim'),
          'pulse'
        )
      ];
    }
  },

  'cluster-and-outline-recipe': {
    parameters: [
      slider('epsilon', 'Epsilon', 3, 12, 0.5, 6.5, 'px'),
      slider('noise', 'Noise points', 0, 60, 2, 26)
    ],
    build: ({fork, p}) => {
      const centers: Point[] = [
        [42, 38],
        [108, 34],
        [86, 74]
      ];
      const points = [
        ...centers.flatMap((center, index) =>
          createClusterPoints(fork(`cluster-${index}`), [center], 24, 6)
        ),
        ...createScatterPoints(fork('noise'), Math.round(p.noise), 8)
      ];
      const labels = new Array<number>(points.length).fill(-2);
      let clusterCount = 0;
      const neighborsOf = (index: number) => {
        const found: number[] = [];
        points.forEach((other, otherIndex) => {
          if (getDistance(points[index], other) <= p.epsilon) found.push(otherIndex);
        });
        return found;
      };
      for (let index = 0; index < points.length; index++) {
        if (labels[index] !== -2) continue;
        const seeds = neighborsOf(index);
        if (seeds.length < 4) {
          labels[index] = -1;
          continue;
        }
        labels[index] = clusterCount;
        for (let cursor = 0; cursor < seeds.length; cursor++) {
          const member = seeds[cursor];
          if (labels[member] === -1) labels[member] = clusterCount;
          if (labels[member] !== -2) continue;
          labels[member] = clusterCount;
          const next = neighborsOf(member);
          if (next.length >= 4) seeds.push(...next);
        }
        clusterCount++;
      }
      const hullFills: PolygonItem[] = [];
      const hulls: PathItem[] = [];
      const centroids: SpriteItem[] = [];
      const clusterRoles: Role[] = ['primary', 'cool', 'violet', 'warm', 'hot'];
      for (let cluster = 0; cluster < clusterCount; cluster++) {
        const members = points.filter((_, index) => labels[index] === cluster);
        const hull = getConvexHull(members);
        if (hull.length < 3) continue;
        const role = clusterRoles[cluster % clusterRoles.length];
        hullFills.push({ring: hull, role, alpha: 0.12});
        hulls.push(createLine(closeRing(hull), role, 0.9, 0.9, cluster / 4));
        const [x, y] = getCentroid(members.map(point => point));
        centroids.push(createDot(x, y, 2.4, 'ink', 1, cluster / 4));
      }
      const sprites = points.map(([x, y], index) =>
        createDot(
          x,
          y,
          labels[index] >= 0 ? 1.4 : 0.9,
          labels[index] >= 0 ? clusterRoles[labels[index] % clusterRoles.length] : 'dim',
          labels[index] >= 0 ? 0.95 : 0.5,
          (index % 9) / 9
        )
      );
      return [
        fillLayer(hullFills),
        spriteLayer(sprites, 0.25),
        strokeLayer(hulls, 'flow', 0.3, 1.5),
        spriteLayer(centroids, 0.5)
      ];
    }
  },

  'points-in-polygons-choropleth-recipe': {
    parameters: [
      slider('classes', 'Classes', 2, 6, 1, 5),
      slider('density', 'Points', 0.3, 1.6, 0.1, 1)
    ],
    build: ({fork, p}) => {
      const zones = createZoneQuads(fork('zones'), 4, 3, 8, 8, 144, 84, 0.12);
      const intensity = createSmoothField(fork('intensity'), 4, 0.9, 4, 3);
      const placement = fork('points');
      const counts = zones.map((_, index) => Math.round((3 + intensity[index] * 26) * p.density));
      const sprites: SpriteItem[] = [];
      zones.forEach((quad, index) => {
        for (let point = 0; point < counts[index]; point++) {
          const [x, y] = getQuadPoint(
            quad,
            placement.range(0.08, 0.92),
            placement.range(0.08, 0.92)
          );
          sprites.push(createDot(x, y, 1, 'ink', 0.85, placement.next()));
        }
      });
      const classes = Math.round(p.classes);
      const sorted = [...counts].sort((first, second) => first - second);
      const fills: PolygonItem[] = zones.map((ring, index) => {
        const rank = sorted.indexOf(counts[index]);
        const bin = Math.min(classes - 1, Math.floor((rank / zones.length) * classes));
        return {
          ring,
          role: getClassRole(bin, classes),
          alpha: 0.4,
          phase: index / zones.length
        };
      });
      return [
        fillLayer(fills, 'pulse'),
        strokeLayer(toPaths(zones.map(closeRing), 'dim', 0.4, 0.7)),
        spriteLayer(sprites, 0.35)
      ];
    }
  },

  'period-comparison-recipe': {
    parameters: [
      slider('shift', 'Shift', 0, 4, 0.1, 2.2, 'cells'),
      slider('threshold', 'Change cutoff', 0.05, 0.5, 0.01, 0.2)
    ],
    build: ({fork, p}) => {
      const columns = 6;
      const rows = 6;
      const size = 6.5;
      const noise = fork('noise');
      const jitter = Array.from({length: columns * rows}, () => noise.range(-0.04, 0.04));
      const blob = (cx: number, cy: number, spread: number, index: number) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        return clamp(
          Math.exp(-((column - cx) ** 2 + (row - cy) ** 2) / (2 * spread * spread)) + jitter[index],
          0,
          1
        );
      };
      const first = jitter.map((_, index) => blob(1.7, 1.8, 1.2, index));
      const second = jitter.map((_, index) => blob(1.7 + p.shift, 1.8 + p.shift * 0.7, 1.5, index));
      const panels = [8, 60.5, 113];
      const panelA: PolygonItem[] = [];
      const panelB: PolygonItem[] = [];
      const gains: PolygonItem[] = [];
      const losses: PolygonItem[] = [];
      const steady: PolygonItem[] = [];
      first.forEach((value, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const ring = (panel: number) =>
          createCellRing(panels[panel] + column * size, 30 + row * size, size);
        panelA.push({
          ring: ring(0),
          role: 'violet',
          alpha: 0.08 + 0.8 * value
        });
        panelB.push({
          ring: ring(1),
          role: 'primary',
          alpha: 0.08 + 0.8 * second[index]
        });
        const change = second[index] - value;
        const target = change > p.threshold ? gains : change < -p.threshold ? losses : steady;
        target.push({
          ring: ring(2),
          role: target === gains ? 'hot' : target === losses ? 'cool' : 'dim',
          alpha: target === steady ? 0.15 : clamp(0.3 + Math.abs(change), 0.4, 0.95),
          phase: column / columns
        });
      });
      return [
        fillLayer(panelA),
        fillLayer(panelB),
        strokeLayer([
          createLine(
            [
              [50, 49],
              [57, 49]
            ],
            'dim',
            0.9,
            0.8
          )
        ]),
        ...createArrowLayers([102, 49], [111, 49]),
        fillLayer(steady),
        fillLayer(losses, 'pulse'),
        fillLayer(gains, 'pulse')
      ];
    }
  },

  'cell-flow-topology-recipe': {
    parameters: [
      slider('flows', 'Flows', 6, 40, 1, 22),
      slider('far', 'Long trips', 0, 1, 0.05, 0.3)
    ],
    build: ({fork, p}) => {
      const radius = 9.5;
      const stepX = Math.sqrt(3) * radius;
      const columns = 7;
      const rows = 5;
      const centers: Point[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          centers.push([
            (SCENE_WIDTH - stepX * (columns + 0.5)) / 2 +
              stepX * 0.75 +
              column * stepX +
              (row % 2) * (stepX / 2),
            14 + radius + row * radius * 1.5
          ]);
        }
      }
      const random = fork('flows');
      const totals = new Array<number>(centers.length).fill(0);
      const near: PathItem[] = [];
      const far: PathItem[] = [];
      for (let flow = 0; flow < Math.round(p.flows); flow++) {
        const origin = random.integer(centers.length);
        const neighbors = centers
          .map((_, index) => index)
          .filter(
            index => index !== origin && getDistance(centers[origin], centers[index]) < stepX * 1.05
          );
        const wantsFar = random.next() < p.far;
        let destination = neighbors[random.integer(neighbors.length)];
        if (wantsFar) destination = random.integer(centers.length);
        if (destination === origin) destination = neighbors[0];
        totals[origin]++;
        totals[destination]++;
        const arc = createArc(centers[origin], centers[destination], 0.18, 14);
        const isNeighbor = getDistance(centers[origin], centers[destination]) < stepX * 1.05;
        (isNeighbor ? near : far).push(
          createLine(
            arc,
            isNeighbor ? 'warm' : 'cool',
            isNeighbor ? 1 : 0.5,
            isNeighbor ? 0.9 : 0.35,
            random.next()
          )
        );
      }
      const hexes = centers.map(([x, y]) =>
        createLine(
          closeRing(
            createCircle(x, y, radius * 0.97, 6).map((_, index): Point => {
              const angle = Math.PI / 6 + (index / 6) * Math.PI * 2;
              return [x + Math.cos(angle) * radius * 0.96, y + Math.sin(angle) * radius * 0.96];
            })
          ),
          'dim',
          0.4,
          0.55
        )
      );
      const sprites = centers.map(([x, y], index) =>
        createDot(x, y, 1 + Math.sqrt(totals[index]) * 1.3, 'primary', 0.75, index / centers.length)
      );
      return [
        strokeLayer(hexes),
        strokeLayer(far),
        strokeLayer(near, 'flow', 0.35, 1),
        spriteLayer(sprites, 0.35)
      ];
    }
  },

  'change-of-support-recipe': {
    parameters: [
      slider('mismatch', 'Misalignment', 0, 0.45, 0.01, 0.25),
      slider('contrast', 'Contrast', 0.2, 1, 0.05, 0.8)
    ],
    build: ({fork, p}) => {
      const width = 52;
      const height = 72;
      const source = createZoneQuads(fork('source'), 3, 2, 0, 0, width, height, 0.1);
      const target = createZoneQuads(fork('target'), 2, 3, 0, 0, width, height, p.mismatch);
      const valueRandom = fork('values');
      const values = source.map(() => 0.15 + valueRandom.next() * p.contrast);
      const transfer = new Array<number>(target.length).fill(0);
      const weights = source.map(() => new Array<number>(target.length).fill(0));
      const areas = new Array<number>(source.length).fill(0);
      const columns = 20;
      const rows = 28;
      const owners: number[][] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const point: Point = [((column + 0.5) / columns) * width, ((row + 0.5) / rows) * height];
          const sourceIndex = source.findIndex(ring => isInsideRing(point, ring));
          const targetIndex = target.findIndex(ring => isInsideRing(point, ring));
          if (sourceIndex < 0 || targetIndex < 0) continue;
          owners.push([sourceIndex, targetIndex]);
          areas[sourceIndex]++;
        }
      }
      for (const [sourceIndex, targetIndex] of owners) {
        weights[sourceIndex][targetIndex] += 1 / areas[sourceIndex];
        transfer[targetIndex] += values[sourceIndex] / areas[sourceIndex];
      }
      const maximum = Math.max(...transfer) || 1;
      const move = (ring: Point[], dx: number, dy: number) =>
        ring.map(([x, y]): Point => [x + dx, y + dy]);
      const left = source.map(ring => move(ring, 8, 14));
      const right = target.map(ring => move(ring, 100, 14));
      const links: PathItem[] = [];
      source.forEach((ring, sourceIndex) => {
        target.forEach((_, targetIndex) => {
          const weight = weights[sourceIndex][targetIndex];
          if (weight < 0.06) return;
          const from = getCentroid(left[sourceIndex]);
          const to = getCentroid(right[targetIndex]);
          links.push(createLine(createArc(from, to, 0, 10), 'warm', 0.3 + weight * 2, 0.5, weight));
        });
      });
      return [
        fillLayer(
          left.map((ring, index) => ({
            ring,
            role: 'cool',
            alpha: 0.12 + 0.6 * values[index]
          }))
        ),
        fillLayer(
          right.map((ring, index) => ({
            ring,
            role: 'hot',
            alpha: 0.12 + 0.7 * (transfer[index] / maximum)
          })),
          'pulse'
        ),
        strokeLayer(toPaths([...left, ...right].map(closeRing), 'dim', 0.45, 0.8)),
        strokeLayer(links, 'flow', 0.3, 1.1)
      ];
    }
  },

  'dasymetric-redistribution-recipe': {
    parameters: [
      slider('weight', 'Ancillary weight', 0, 1, 0.05, 0.85),
      slider('total', 'People per zone', 30, 160, 5, 80)
    ],
    build: ({fork, p}) => {
      const ancillary = createSmoothField(fork('ancillary'), 6, 1.4);
      const split = (y: number) => 80 + 7 * Math.sin(y / 11);
      const random = fork('people');
      const dots: SpriteItem[] = [];
      for (let zone = 0; zone < 2; zone++) {
        let placed = 0;
        let attempts = 0;
        while (placed < p.total && attempts < 4000) {
          attempts++;
          const y = random.range(4, 96);
          const x = zone === 0 ? random.range(4, split(y) - 2) : random.range(split(y) + 2, 156);
          const density = sampleField(
            ancillary,
            FIELD_COLUMNS,
            FIELD_ROWS,
            x / SCENE_WIDTH,
            y / SCENE_HEIGHT
          );
          if (random.next() > 1 - p.weight + p.weight * density ** 2) continue;
          dots.push(createDot(x, y, 1.15, 'warm', 0.95, random.next()));
          placed++;
        }
      }
      const boundary: Point[] = Array.from(
        {length: 21},
        (_, index): Point => [split(index * 5), index * 5]
      );
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: ancillary,
          ramp: 'ice',
          style: 'ramp',
          strength: 0.05
        },
        strokeLayer([createLine(boundary, 'violet', 1.1, 0.95)], 'pulse', 0.2),
        strokeLayer([
          createLine(
            closeRing([
              [3, 3],
              [157, 3],
              [157, 97],
              [3, 97]
            ]),
            'violet',
            0.7,
            0.7
          )
        ]),
        spriteLayer(dots, 0.4)
      ];
    }
  },

  'zonal-distribution-recipe': {
    parameters: [
      slider('bins', 'Bins', 5, 14, 1, 9),
      slider('spread', 'Spread', 0.1, 0.4, 0.01, 0.2)
    ],
    build: ({fork, p}) => {
      const zones = createZoneQuads(fork('zones'), 3, 2, 5, 6, 150, 88, 0.05);
      const random = fork('moments');
      const bins = Math.round(p.bins);
      const bars: PathItem[] = [];
      const medians: SpriteItem[] = [];
      const modes: SpriteItem[] = [];
      const ticks: PathItem[] = [];
      zones.forEach(ring => {
        const [cx, cy] = getCentroid(ring);
        const mean = random.range(0.3, 0.7);
        const spread = p.spread * random.range(0.6, 1.3);
        const left = cx - 19;
        const spacing = 38 / (bins - 1);
        const baseline = cy + 14;
        const density = Array.from(
          {length: bins},
          (_, bin) =>
            Math.exp(-((bin / (bins - 1) - mean) ** 2) / (2 * spread * spread)) *
            random.range(0.8, 1.15)
        );
        const peak = Math.max(...density);
        bars.push(
          ...createBars(
            left,
            baseline,
            density.map(value => (value / peak) * 28),
            Math.min(3.6, spacing * 0.7),
            spacing,
            'primary',
            0.8
          )
        );
        const total = density.reduce((sum, value) => sum + value, 0);
        let running = 0;
        const quantile = (fraction: number) => {
          let accumulated = 0;
          for (let bin = 0; bin < bins; bin++) {
            accumulated += density[bin] / total;
            if (accumulated >= fraction) return bin;
          }
          return bins - 1;
        };
        running = quantile(0.5);
        medians.push(createDot(left + running * spacing, baseline + 4.5, 1.9, 'warm', 1));
        modes.push(
          createDot(left + density.indexOf(peak) * spacing, baseline - 30, 1.5, 'hot', 0.95)
        );
        ticks.push(
          createLine(
            [
              [left + quantile(0.1) * spacing, baseline + 4.5],
              [left + quantile(0.9) * spacing, baseline + 4.5]
            ],
            'cool',
            0.7,
            0.8
          )
        );
      });
      return [
        fillLayer(zones.map(ring => ({ring, role: 'dim', alpha: 0.07}))),
        strokeLayer(toPaths(zones.map(closeRing), 'dim', 0.4, 0.7)),
        strokeLayer(bars, 'pulse', 0.4),
        strokeLayer(ticks),
        spriteLayer(medians, 0.5),
        spriteLayer(modes, 0.4)
      ];
    }
  },

  'raster-zonal-distribution-recipe': {
    parameters: [
      slider('nodata', 'Nodata cells', 0, 50, 2, 16),
      slider('roughness', 'Roughness', 0.7, 2.4, 0.1, 1.2)
    ],
    build: ({fork, p}) => {
      const columns = 40;
      const rows = 25;
      const values = createSmoothField(fork('band'), 7, p.roughness, columns, rows);
      const zones = createZoneQuads(fork('zones'), 3, 2, 3, 3, 154, 94, 0.08);
      const random = fork('nodata');
      const missing = new Set<number>();
      while (missing.size < Math.round(p.nodata)) missing.add(random.integer(values.length));
      const holes: PolygonItem[] = [...missing].map(index => ({
        ring: createCellRing((index % columns) * 4, Math.floor(index / columns) * 4, 4),
        role: 'dim',
        alpha: 0.85
      }));
      const whiskers: PathItem[] = [];
      const boxes: PolygonItem[] = [];
      const medians: SpriteItem[] = [];
      zones.forEach(ring => {
        const found: number[] = [];
        for (let index = 0; index < values.length; index++) {
          if (missing.has(index)) continue;
          if (isInsideRing([(index % columns) * 4 + 2, Math.floor(index / columns) * 4 + 2], ring))
            found.push(values[index]);
        }
        found.sort((first, second) => first - second);
        const pick = (fraction: number) =>
          found[Math.min(found.length - 1, Math.floor(found.length * fraction))] ?? 0.5;
        const [cx, cy] = getCentroid(ring);
        const level = (value: number) => cy + 15 - value * 30;
        whiskers.push(
          createLine(
            [
              [cx, level(pick(0.05))],
              [cx, level(pick(0.95))]
            ],
            'ink',
            0.7,
            0.95
          )
        );
        boxes.push({
          ring: [
            [cx - 4.5, level(pick(0.75))],
            [cx + 4.5, level(pick(0.75))],
            [cx + 4.5, level(pick(0.25))],
            [cx - 4.5, level(pick(0.25))]
          ],
          role: 'warm',
          alpha: 0.7
        });
        medians.push(createDot(cx, level(pick(0.5)), 2, 'hot', 1));
      });
      return [
        {
          type: 'field',
          columns,
          rows,
          values,
          ramp: 'terrain',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.04
        },
        fillLayer(holes),
        strokeLayer(toPaths(zones.map(closeRing), 'ink', 0.5, 0.75)),
        fillLayer(boxes, 'pulse'),
        strokeLayer(whiskers),
        spriteLayer(medians, 0.4)
      ];
    }
  },

  'exposure-by-zone-recipe': {
    parameters: [
      slider('threshold', 'Hazard cutoff', 0.3, 0.9, 0.02, 0.6),
      slider('people', 'People', 30, 200, 10, 100)
    ],
    build: ({fork, p}) => {
      const hazard = createSmoothField(fork('hazard'), 5, 1);
      const shown = hazard.map(value => (value >= p.threshold ? value : value * 0.28));
      const zones = createZoneQuads(fork('zones'), 3, 2, 3, 3, 154, 94, 0.06);
      const unobserved: Point[][] = [
        [
          [26, 62],
          [48, 62],
          [48, 80],
          [26, 80]
        ],
        [
          [104, 14],
          [124, 14],
          [124, 32],
          [104, 32]
        ]
      ];
      const random = fork('people');
      const exposedPerZone = new Array<number>(zones.length).fill(0);
      const dots: SpriteItem[] = [];
      for (let person = 0; person < Math.round(p.people); person++) {
        const x = random.range(6, 154);
        const y = random.range(6, 94);
        const level = sampleField(
          hazard,
          FIELD_COLUMNS,
          FIELD_ROWS,
          x / SCENE_WIDTH,
          y / SCENE_HEIGHT
        );
        const observed = !unobserved.some(ring => isInsideRing([x, y], ring));
        const exposed = observed && level >= p.threshold;
        const zone = zones.findIndex(ring => isInsideRing([x, y], ring));
        if (exposed && zone >= 0) exposedPerZone[zone]++;
        dots.push(
          createDot(
            x,
            y,
            exposed ? 1.5 : 1,
            exposed ? 'ink' : 'cool',
            exposed ? 1 : 0.5,
            random.next()
          )
        );
      }
      const bars: PathItem[] = zones.map((ring, index) => {
        const bottom = Math.max(...ring.map(point => point[1])) - 3.5;
        const left = Math.min(...ring.map(point => point[0])) + 4;
        return createLine(
          [
            [left, bottom],
            [left + exposedPerZone[index] * 2.2, bottom]
          ],
          'hot',
          1.8,
          0.95,
          index / zones.length
        );
      });
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: shown,
          ramp: 'heat',
          style: 'ramp',
          strength: 0.05
        },
        fillLayer(
          unobserved.map(ring => ({ring, role: 'dim', alpha: 0.75})),
          'static',
          'alpha'
        ),
        strokeLayer(toPaths(unobserved.map(closeRing), 'dim', 0.5, 0.8)),
        strokeLayer(toPaths(zones.map(closeRing), 'ink', 0.5, 0.6)),
        spriteLayer(dots, 0.35),
        strokeLayer(bars, 'pulse', 0.3)
      ];
    }
  },

  'regional-raster-change-recipe': {
    parameters: [
      slider('change', 'Change', 0.2, 1.5, 0.05, 0.8),
      slider('support', 'Common support', 0, 10, 1, 4, 'cols')
    ],
    build: ({fork, p}) => {
      const columns = 32;
      const rows = 20;
      const before = createSmoothField(fork('before'), 6, 1.1, columns, rows);
      const after = createSmoothField(fork('after'), 5, 1.0, columns, rows);
      const support = Math.round(p.support);
      const difference = before.map((value, index) =>
        index % columns < support ? 0 : (after[index] - value) * p.change
      );
      const values = Float32Array.from(difference, value => clamp(0.5 + value * 0.9, 0, 1));
      const zones = createZoneQuads(fork('zones'), 3, 2, 3, 3, 154, 94, 0.06);
      const sums = zones.map(() => 0);
      for (let index = 0; index < difference.length; index++) {
        const point: Point = [
          ((index % columns) + 0.5) * 5,
          (Math.floor(index / columns) + 0.5) * 5
        ];
        const zone = zones.findIndex(ring => isInsideRing(point, ring));
        if (zone >= 0) sums[zone] += difference[index];
      }
      const marks = zones.map((ring, index): SpriteItem => {
        const [x, y] = getCentroid(ring);
        return createDot(
          x,
          y,
          3 + Math.min(9, Math.abs(sums[index]) * 0.35),
          sums[index] >= 0 ? 'hot' : 'cool',
          0.95,
          index / zones.length
        );
      });
      const halos = marks.map(mark => ({
        ...mark,
        radius: mark.radius + 1.6,
        role: 'ink' as Role,
        alpha: 0.9
      }));
      return [
        {
          type: 'field',
          columns,
          rows,
          values,
          ramp: 'spectral',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.03
        },
        fillLayer(
          support > 0
            ? [
                {
                  ring: [
                    [0, 0],
                    [support * 5, 0],
                    [support * 5, 100],
                    [0, 100]
                  ],
                  role: 'dim',
                  alpha: 0.8
                }
              ]
            : [],
          'static',
          'alpha'
        ),
        strokeLayer(toPaths(zones.map(closeRing), 'ink', 0.5, 0.7)),
        spriteLayer(halos, 0.4),
        spriteLayer(marks, 0.4)
      ];
    }
  },

  'surface-preparation-recipe': {
    parameters: [
      slider('gap', 'Gap size', 5, 17, 1, 11),
      slider('filter', 'Rank filter', 0, 3, 1, 1)
    ],
    build: ({fork, p}) => {
      const raw = createSmoothField(fork('surface'), 7, 1.3);
      const surface = getBoxBlur(raw, FIELD_COLUMNS, FIELD_ROWS, Math.round(p.filter));
      const filled = getBoxBlur(surface, FIELD_COLUMNS, FIELD_ROWS, 7);
      const gaps: Point[] = [
        [44, 36],
        [102, 62],
        [118, 24]
      ];
      const values = Float32Array.from(surface, (value, index) => {
        const x = ((index % FIELD_COLUMNS) + 0.5) * (SCENE_WIDTH / FIELD_COLUMNS);
        const y = (Math.floor(index / FIELD_COLUMNS) + 0.5) * (SCENE_HEIGHT / FIELD_ROWS);
        const nearest = Math.min(...gaps.map(gap => getDistance(gap, [x, y]))) / p.gap;
        const inside = clamp((1.15 - nearest) / 0.3, 0, 1);
        return mixNumbers(value, filled[index], inside);
      });
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'terrain',
          style: 'contours',
          levels: 14,
          strength: 0.04
        },
        fillLayer(
          gaps.map(([x, y]) => ({
            ring: createCircle(x, y, p.gap, 36),
            role: 'warm' as Role,
            alpha: 0.1
          })),
          'pulse'
        ),
        strokeLayer(
          gaps.map(([x, y], index) =>
            createLine(closeRing(createCircle(x, y, p.gap, 36)), 'warm', 0.8, 0.9, index / 3)
          ),
          'pulse',
          0.4
        )
      ];
    }
  },

  'interpolation-diagnostics-recipe': {
    parameters: [
      slider('samples', 'Samples', 6, 26, 1, 14),
      slider('power', 'Range', 1, 6, 0.1, 2)
    ],
    build: ({fork, p}) => {
      const pool = createScatterPoints(fork('samples'), 26, 12);
      const valueRandom = fork('values');
      const poolValues = pool.map(
        ([x, y]) => 0.5 + 0.4 * Math.sin(x / 22) * Math.cos(y / 17) + valueRandom.gaussian() * 0.12
      );
      const count = Math.round(p.samples);
      const samples = pool.slice(0, count);
      const sampleValues = poolValues.slice(0, count);
      const predict = (x: number, y: number, skip = -1) => {
        let weightSum = 0;
        let sum = 0;
        for (let index = 0; index < count; index++) {
          if (index === skip) continue;
          const weight = 1 / (getDistance(samples[index], [x, y]) ** p.power + 0.5);
          weightSum += weight;
          sum += weight * sampleValues[index];
        }
        return sum / weightSum;
      };
      const predicted = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) =>
        predict(u * SCENE_WIDTH, v * SCENE_HEIGHT)
      );
      const low = Math.min(...predicted);
      const range = Math.max(...predicted) - low || 1;
      const surface = predicted.map(value => (value - low) / range);
      const residuals = samples.map(([x, y], index) => sampleValues[index] - predict(x, y, index));
      const deviation =
        Math.sqrt(residuals.reduce((sum, value) => sum + value * value, 0) / count) || 1;
      const standardizedScores = residuals.map(value => value / deviation);
      const rings: PathItem[] = samples.map(([x, y], index) =>
        createLine(
          closeRing(
            createCircle(x, y, 2.5 + Math.min(Math.abs(standardizedScores[index]), 2.2) * 2.6, 28)
          ),
          standardizedScores[index] > 0 ? 'hot' : 'cool',
          1,
          0.95,
          index / count
        )
      );
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: surface,
          ramp: 'spectral',
          style: 'ramp',
          strength: 0.05
        },
        strokeLayer(rings, 'pulse', 0.4),
        spriteLayer(toSprites(samples, 'ink', 1.3, undefined, 0.95), 0)
      ];
    }
  },

  // ------------------------------------------------------------ hydrology, catchments, landscape

  'drainage-network-recipe': {
    parameters: [
      slider('threshold', 'Stream threshold', 3, 60, 1, 8, 'cells'),
      slider('roughness', 'Roughness', 0.7, 2.2, 0.1, 1.2)
    ],
    build: ({fork, p}) => {
      const columns = 40;
      const rows = 25;
      const noise = createSmoothField(fork('terrain'), 7, p.roughness, columns, rows);
      const elevation = noise.map((value, index) => {
        const u = (index % columns) / columns;
        const v = Math.floor(index / columns) / rows;
        return value * 0.3 + (1 - (0.75 * u + 0.25 * v)) * 0.7;
      });
      const downstream = new Array<number>(elevation.length).fill(-1);
      for (let index = 0; index < elevation.length; index++) {
        const column = index % columns;
        const row = Math.floor(index / columns);
        let best = 0;
        for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
          for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
            const c = column + columnOffset;
            const r = row + rowOffset;
            if ((!rowOffset && !columnOffset) || c < 0 || r < 0 || c >= columns || r >= rows)
              continue;
            const drop =
              (elevation[index] - elevation[r * columns + c]) / Math.hypot(rowOffset, columnOffset);
            if (drop > best) {
              best = drop;
              downstream[index] = r * columns + c;
            }
          }
        }
      }
      const accumulation = new Array<number>(elevation.length).fill(1);
      Array.from(elevation, (value, index) => ({value, index}))
        .sort((first, second) => second.value - first.value)
        .forEach(({index}) => {
          if (downstream[index] >= 0) accumulation[downstream[index]] += accumulation[index];
        });
      const cell = SCENE_WIDTH / columns;
      const center = (index: number): Point => [
        ((index % columns) + 0.5) * cell,
        (Math.floor(index / columns) + 0.5) * cell
      ];
      const small: PathItem[] = [];
      const large: PathItem[] = [];
      const outlets: SpriteItem[] = [];
      accumulation.forEach((amount, index) => {
        if (amount < p.threshold) return;
        const next = downstream[index];
        const width = clamp(0.35 + Math.sqrt(amount) * 0.12, 0.4, 2.6);
        if (next >= 0) {
          (amount > p.threshold * 4 ? large : small).push(
            createLine(
              [center(index), center(next)],
              amount > p.threshold * 4 ? 'primary' : 'cool',
              width,
              0.95,
              (index % 17) / 17
            )
          );
        } else if (amount > p.threshold * 2) {
          const [x, y] = center(index);
          outlets.push(createDot(x, y, 2.4, 'warm', 1, index / elevation.length));
        }
      });
      const elevationValues = Float32Array.from(elevation);
      const smoothElevation = Float32Array.from({length: FIELD_COLUMNS * FIELD_ROWS}, (_, index) =>
        sampleField(
          elevationValues,
          columns,
          rows,
          ((index % FIELD_COLUMNS) + 0.5) / FIELD_COLUMNS,
          (Math.floor(index / FIELD_COLUMNS) + 0.5) / FIELD_ROWS
        )
      );
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: smoothElevation,
          ramp: 'terrain',
          style: 'hillshade',
          strength: 0.5,
          light: -2.4
        },
        strokeLayer(small, 'flow', 0.3, 1.2),
        strokeLayer(large, 'flow', 0.35, 1.6),
        spriteLayer(outlets, 0.5)
      ];
    }
  },

  'straight-line-catchments-recipe': {
    parameters: [slider('facilities', 'Facilities', 2, 7, 1, 5)],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const pool = createScatterPoints(fork('facilities'), 7, 16);
      const facilities = pool.slice(0, Math.round(p.facilities));
      if (pointer) facilities[0] = pointer;
      const columns = 32;
      const rows = 20;
      const cell = 5;
      const fills: PolygonItem[] = [];
      const areas = new Array<number>(facilities.length).fill(0);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const x = (column + 0.5) * cell;
          const y = (row + 0.5) * cell;
          let nearest = 0;
          let nearestDistance = Infinity;
          facilities.forEach((facility, index) => {
            const distance = getDistance(facility, [x, y]);
            if (distance < nearestDistance) {
              nearestDistance = distance;
              nearest = index;
            }
          });
          areas[nearest]++;
          fills.push({
            ring: createCellRing(column * cell, row * cell, cell),
            role: getRoleCycle(nearest),
            alpha: clamp(0.62 - nearestDistance / 120, 0.1, 0.6)
          });
        }
      }
      return [
        fillLayer(fills),
        spriteLayer(
          facilities.map(([x, y], index) =>
            createDot(
              x,
              y,
              3 + Math.sqrt(areas[index]) * 0.35,
              getRoleCycle(index),
              0.35,
              index / 7
            )
          ),
          0.4
        ),
        spriteLayer(toSprites(facilities, 'ink', 2.1, undefined, 1), 0)
      ];
    }
  },

  'landscape-metrics-recipe': {
    parameters: [
      slider('classes', 'Classes', 2, 5, 1, 4),
      slider('fragmentation', 'Fragmentation', 0.8, 3, 0.1, 1.6)
    ],
    build: ({fork, p}) => {
      const columns = 32;
      const rows = 20;
      const cell = 5;
      const classes = Math.round(p.classes);
      const values = createSmoothField(fork('cover'), 6, p.fragmentation, columns, rows);
      const ranks = Array.from(values, (value, index) => ({
        value,
        index
      })).sort((first, second) => first.value - second.value);
      const classOf = new Array<number>(values.length).fill(0);
      ranks.forEach(
        (entry, position) =>
          (classOf[entry.index] = Math.floor((position / ranks.length) * classes))
      );
      const fills: PolygonItem[] = [];
      const edges: PathItem[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const index = row * columns + column;
          fills.push({
            ring: createCellRing(column * cell, row * cell, cell),
            role: getClassRole(classOf[index], classes),
            alpha: 0.38
          });
          if (column < columns - 1 && classOf[index + 1] !== classOf[index]) {
            edges.push(
              createLine(
                [
                  [(column + 1) * cell, row * cell],
                  [(column + 1) * cell, (row + 1) * cell]
                ],
                'ink',
                0.7,
                0.9,
                row / rows
              )
            );
          }
          if (row < rows - 1 && classOf[index + columns] !== classOf[index]) {
            edges.push(
              createLine(
                [
                  [column * cell, (row + 1) * cell],
                  [(column + 1) * cell, (row + 1) * cell]
                ],
                'ink',
                0.7,
                0.9,
                column / columns
              )
            );
          }
        }
      }
      return [fillLayer(fills), strokeLayer(edges, 'pulse', 0.3)];
    }
  },

  'patch-proximity-recipe': {
    parameters: [
      slider('depth', 'Edge depth', 0, 8, 0.5, 3, 'px'),
      slider('patches', 'Patches', 4, 9, 1, 8)
    ],
    build: ({fork, p}) => {
      const random = fork('patches');
      const patches = Array.from({length: 9}, (_, index) => {
        const radius = random.range(7, 11.5);
        const center: Point = [
          22 + (index % 3) * 58 + random.range(-8, 8),
          18 + Math.floor(index / 3) * 32 + random.range(-4, 4)
        ];
        return {
          center,
          radius,
          habitat: index % 4 !== 3,
          ring: createBlob(random, center[0], center[1], radius, 16, 0.3)
        };
      }).slice(0, Math.round(p.patches));
      const habitats = patches.filter(patch => patch.habitat);
      const links: PathItem[] = [];
      habitats.forEach(patch => {
        let nearest = patch;
        let nearestDistance = Infinity;
        habitats.forEach(other => {
          if (other === patch) return;
          const distance = getDistance(patch.center, other.center);
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearest = other;
          }
        });
        if (nearest === patch) return;
        const angle = Math.atan2(
          nearest.center[1] - patch.center[1],
          nearest.center[0] - patch.center[0]
        );
        links.push(
          createLine(
            [
              [
                patch.center[0] + Math.cos(angle) * patch.radius,
                patch.center[1] + Math.sin(angle) * patch.radius
              ],
              [
                nearest.center[0] - Math.cos(angle) * nearest.radius,
                nearest.center[1] - Math.sin(angle) * nearest.radius
              ]
            ],
            'cool',
            0.9,
            0.9,
            patch.radius / 11.5
          )
        );
      });
      const cores: PolygonItem[] = habitats.map(patch => {
        const scale = Math.max(0.12, (patch.radius - p.depth) / patch.radius);
        return {
          ring: patch.ring.map(
            ([x, y]): Point => [
              patch.center[0] + (x - patch.center[0]) * scale,
              patch.center[1] + (y - patch.center[1]) * scale
            ]
          ),
          role: 'warm',
          alpha: 0.75
        };
      });
      return [
        fillLayer(
          patches.map(patch => ({
            ring: patch.ring,
            role: (patch.habitat ? 'primary' : 'violet') as Role,
            alpha: patch.habitat ? 0.2 : 0.14
          }))
        ),
        strokeLayer(
          patches.map(patch =>
            createLine(closeRing(patch.ring), patch.habitat ? 'primary' : 'violet', 0.7, 0.8)
          )
        ),
        fillLayer(cores, 'pulse'),
        strokeLayer(links, 'flow', 0.35, 0.8)
      ];
    }
  },

  'resistance-connectivity-recipe': {
    parameters: [
      slider('barrier', 'Barrier', 0, 1, 0.05, 0.7),
      slider('jitter', 'Layout', 0, 1, 0.1, 0.5)
    ],
    build: ({fork, p}) => {
      const columns = 5;
      const rows = 3;
      const random = fork('layout');
      const nodes: Point[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          nodes.push([
            20 + column * 30 + random.range(-4, 4) * p.jitter,
            24 + row * 26 + random.range(-6, 6) * p.jitter
          ]);
        }
      }
      const edges: {from: number; to: number; conductance: number}[] = [];
      const barrierEdge = [1 * columns + 2, 1 * columns + 3];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const index = row * columns + column;
          if (column < columns - 1) {
            const isBarrier = index === barrierEdge[0];
            edges.push({
              from: index,
              to: index + 1,
              conductance: isBarrier ? 1 - 0.97 * p.barrier : 1
            });
          }
          if (row < rows - 1) edges.push({from: index, to: index + columns, conductance: 1});
          if (row < rows - 1 && column < columns - 1 && (column + row) % 2 === 0) {
            edges.push({
              from: index,
              to: index + columns + 1,
              conductance: 0.6
            });
          }
        }
      }
      const source = 1 * columns;
      const target = 1 * columns + columns - 1;
      const size = nodes.length;
      const matrix = Array.from({length: size}, () => new Array<number>(size + 1).fill(0));
      for (const edge of edges) {
        matrix[edge.from][edge.from] += edge.conductance;
        matrix[edge.to][edge.to] += edge.conductance;
        matrix[edge.from][edge.to] -= edge.conductance;
        matrix[edge.to][edge.from] -= edge.conductance;
      }
      for (const [fixed, voltage] of [
        [source, 1],
        [target, 0]
      ]) {
        matrix[fixed].fill(0);
        matrix[fixed][fixed] = 1;
        matrix[fixed][size] = voltage;
      }
      for (let pivot = 0; pivot < size; pivot++) {
        let best = pivot;
        for (let row = pivot + 1; row < size; row++)
          if (Math.abs(matrix[row][pivot]) > Math.abs(matrix[best][pivot])) best = row;
        [matrix[pivot], matrix[best]] = [matrix[best], matrix[pivot]];
        for (let row = 0; row < size; row++) {
          if (row === pivot) continue;
          const factor = matrix[row][pivot] / matrix[pivot][pivot];
          for (let column = pivot; column <= size; column++)
            matrix[row][column] -= factor * matrix[pivot][column];
        }
      }
      const voltage = matrix.map((row, index) => row[size] / row[index]);
      const currents = edges.map(
        edge => edge.conductance * (voltage[edge.from] - voltage[edge.to])
      );
      const maximum = Math.max(...currents.map(Math.abs)) || 1;
      const flows: PathItem[] = edges.map((edge, index) => {
        const forward = currents[index] >= 0;
        const points = forward
          ? [nodes[edge.from], nodes[edge.to]]
          : [nodes[edge.to], nodes[edge.from]];
        const level = Math.abs(currents[index]) / maximum;
        return createLine(
          points,
          'warm',
          0.3 + level * 3.2,
          0.25 + level * 0.7,
          index / edges.length
        );
      });
      return [
        strokeLayer(
          edges.map(edge => createLine([nodes[edge.from], nodes[edge.to]], 'dim', 0.35, 0.55))
        ),
        strokeLayer(flows, 'flow', 0.4, 1.2),
        spriteLayer(
          nodes.map(([x, y], index) =>
            index === source
              ? createDot(x, y, 4, 'hot', 1)
              : index === target
                ? createDot(x, y, 4, 'cool', 1)
                : createDot(x, y, 1.6, 'primary', 0.8, index / size)
          ),
          0.4
        )
      ];
    }
  },

  // ------------------------------------------------------------ segregation and alignment

  'segregation-decomposition-recipe': {
    parameters: [
      slider('spatial', 'Spatial change', 0, 0.8, 0.05, 0.5),
      slider('composition', 'Composition change', 0, 0.4, 0.02, 0.15)
    ],
    build: ({fork, p}) => {
      const keys = fork('keys');
      const noise = Array.from({length: 40}, () => keys.next());
      const jitter = fork('slots');
      const slots: Point[] = Array.from(
        {length: 40},
        (_, index): Point => [
          (index % 8) * 6.4 + jitter.range(-1, 1),
          Math.floor(index / 8) * 6.4 + jitter.range(-1, 1)
        ]
      );
      const makePanel = (originX: number, originY: number, share: number, segregation: number) => {
        const order = slots
          .map((slot, index) => ({
            index,
            key: segregation * 2.4 * (slot[0] / 45) + (1 - segregation) * noise[index]
          }))
          .sort((first, second) => first.key - second.key);
        const count = Math.round(share * 40);
        const items: SpriteItem[] = [];
        order.forEach((entry, rank) => {
          items.push(
            createDot(
              originX + slots[entry.index][0],
              originY + slots[entry.index][1],
              1.5,
              rank < count ? 'hot' : 'cool',
              0.9,
              entry.index / 40
            )
          );
        });
        return items;
      };
      const before = makePanel(8, 8, 0.4, 0.12);
      const after = makePanel(102, 8, 0.4 + p.composition, 0.12 + p.spatial);
      const unit = 90;
      const compositionLength = p.composition * unit * 1.5;
      const spatialLength = p.spatial * unit * 0.9;
      const start = 30;
      return [
        strokeLayer([
          createLine(
            closeRing([
              [4, 4],
              [58, 4],
              [58, 46],
              [4, 46]
            ]),
            'dim',
            0.3,
            0.4
          ),
          createLine(
            closeRing([
              [98, 4],
              [152, 4],
              [152, 46],
              [98, 46]
            ]),
            'dim',
            0.3,
            0.4
          )
        ]),
        spriteLayer(before, 0.25),
        spriteLayer(after, 0.25),
        ...createArrowLayers([64, 26], [93, 26], 'ink'),
        strokeLayer(
          [
            createLine(
              [
                [start, 72],
                [start + compositionLength, 72]
              ],
              'cool',
              3.4,
              0.95
            ),
            createLine(
              [
                [start + compositionLength + 1, 72],
                [start + compositionLength + 1 + spatialLength, 72]
              ],
              'warm',
              3.4,
              0.95
            )
          ],
          'pulse',
          0.3
        ),
        strokeLayer([
          createLine(
            [
              [start, 80],
              [start + compositionLength + 1 + spatialLength, 80]
            ],
            'dim',
            0.5,
            0.7
          )
        ])
      ];
    }
  },

  'segregation-counterfactual-recipe': {
    parameters: [
      slider('spatial', 'Spatial change', 0, 0.8, 0.05, 0.5),
      slider('composition', 'Composition change', 0, 0.4, 0.02, 0.15)
    ],
    build: ({fork, p}) => {
      const keys = fork('keys');
      const noise = Array.from({length: 40}, () => keys.next());
      const slots: Point[] = Array.from(
        {length: 40},
        (_, index): Point => [(index % 8) * 5, Math.floor(index / 8) * 5]
      );
      const makePanel = (originX: number, originY: number, share: number, segregation: number) => {
        const order = slots
          .map((slot, index) => ({
            index,
            key: segregation * 2.4 * (slot[0] / 35) + (1 - segregation) * noise[index]
          }))
          .sort((first, second) => first.key - second.key);
        const count = Math.round(share * 40);
        return order.map((entry, rank) =>
          createDot(
            originX + slots[entry.index][0],
            originY + slots[entry.index][1],
            1.25,
            rank < count ? 'hot' : 'cool',
            0.9,
            entry.index / 40
          )
        );
      };
      const shareFirst = 0.4;
      const shareSecond = 0.4 + p.composition;
      const segregationFirst = 0.12;
      const segregationSecond = 0.12 + p.spatial;
      const columnLeft = 22;
      const columnRight = 104;
      const rowTop = 6;
      const rowBottom = 62;
      const frame = (x: number, y: number) =>
        createLine(
          closeRing([
            [x - 4, y - 4],
            [x + 39, y - 4],
            [x + 39, y + 24],
            [x - 4, y + 24]
          ]),
          'dim',
          0.3,
          0.45
        );
      const sprites = [
        ...makePanel(columnLeft, rowTop, shareFirst, segregationFirst),
        ...makePanel(columnRight, rowTop, shareSecond, segregationSecond),
        ...makePanel(columnLeft, rowBottom, shareSecond, segregationFirst),
        ...makePanel(columnRight, rowBottom, shareFirst, segregationSecond)
      ];
      const downLeft: Point[] = [
        [columnLeft + 17, 34],
        [columnLeft + 17, 55]
      ];
      const downRight: Point[] = [
        [columnRight + 17, 34],
        [columnRight + 17, 55]
      ];
      return [
        strokeLayer([
          frame(columnLeft, rowTop),
          frame(columnRight, rowTop),
          frame(columnLeft, rowBottom),
          frame(columnRight, rowBottom)
        ]),
        spriteLayer(sprites, 0.25),
        strokeLayer(
          [
            createLine(downLeft, 'cool', 0.8, 0.8),
            createLine(downRight, 'warm', 0.8, 0.8),
            createLine(
              [
                [columnLeft + 40, 30],
                [columnRight - 6, 58]
              ],
              'cool',
              0.8,
              0.7
            ),
            createLine(
              [
                [columnRight - 4, 30],
                [columnLeft + 42, 58]
              ],
              'warm',
              0.8,
              0.7
            )
          ],
          'flow',
          0.35,
          1.2
        )
      ];
    }
  },

  'population-support-alignment-recipe': {
    parameters: [
      slider('missing', 'Missing IDs', 0, 2, 1, 0),
      slider('duplicates', 'Duplicate IDs', 0, 2, 1, 0)
    ],
    build: ({fork, p}) => {
      const units = createZoneQuads(fork('units'), 3, 2, 100, 14, 52, 72, 0.08);
      const unitRoles: Role[] = ['primary', 'warm', 'violet', 'cool', 'hot', 'ink'];
      const random = fork('order');
      const shuffled = units
        .map((_, index) => ({index, key: random.next()}))
        .sort((first, second) => first.key - second.key)
        .map(entry => entry.index);
      const missing = Math.round(p.missing);
      const duplicates = Math.round(p.duplicates);
      const ids = shuffled.slice(0, units.length - missing);
      for (let duplicate = 0; duplicate < duplicates; duplicate++) ids.push(ids[duplicate]);
      const spacing = 70 / Math.max(ids.length, 1);
      const centroids = units.map(getCentroid);
      const rowItems: PathItem[] = [];
      const links: PathItem[] = [];
      const faults: PathItem[] = [];
      const seen = new Set<number>();
      ids.forEach((id, row) => {
        const y = 16 + row * spacing + spacing / 2;
        const isDuplicate = seen.has(id) || ids.filter(other => other === id).length > 1;
        seen.add(id);
        const role: Role = isDuplicate ? 'hot' : unitRoles[id];
        rowItems.push(
          createLine(
            [
              [8, y],
              [42, y]
            ],
            role,
            Math.min(5, spacing * 0.55),
            0.7,
            row / ids.length
          )
        );
        rowItems.push(
          createLine(
            [
              [8, y],
              [14, y]
            ],
            'ink',
            Math.min(5, spacing * 0.55),
            0.95
          )
        );
        const bend = (centroids[id][0] - 48) * 0.5;
        links.push(
          createLine(
            [[44, y], [44 + bend, y], [centroids[id][0] - bend, centroids[id][1]], centroids[id]],
            role,
            0.8,
            0.8,
            row / ids.length
          )
        );
      });
      units.forEach((ring, index) => {
        if (!ids.includes(index)) faults.push(createLine(closeRing(ring), 'hot', 1.4, 0.95));
      });
      return [
        fillLayer(
          units.map((ring, index) => ({
            ring,
            role: unitRoles[index],
            alpha: 0.22
          }))
        ),
        strokeLayer(toPaths(units.map(closeRing), 'dim', 0.4, 0.7)),
        strokeLayer(faults, 'pulse', 0.6),
        strokeLayer(rowItems),
        strokeLayer(links, 'flow', 0.4, 1)
      ];
    }
  },

  // ------------------------------------------------------------ trajectories

  'trajectory-resample': {
    parameters: [slider('samples', 'Samples', 4, 32, 1, 14)],
    build: ({fork, p}) => {
      const phase = fork('shape').range(0, 1.5);
      const dense = Array.from(
        {length: 160},
        (_, index): Point => getCurvePoint(index / 159, phase, 22, 50)
      );
      const lengths = getCumulative(dense);
      const lane = (point: Point, offset: number): Point => [
        point[0],
        50 + (point[1] - 50) * 0.45 + offset
      ];
      const random = fork('gaps');
      const gaps = Array.from({length: 17}, (_, index) =>
        index % 5 === 2 ? random.range(2.5, 4) : random.range(0.25, 1.1)
      );
      const total = gaps.reduce((sum, gap) => sum + gap, 0);
      const rawFractions = [0];
      for (const gap of gaps) {
        rawFractions.push(rawFractions[rawFractions.length - 1] + gap / total);
      }
      const count = Math.round(p.samples);
      const resampled = Array.from({length: count}, (_, index) => index / (count - 1));
      const place = (fraction: number, offset: number) =>
        lane(getPointAtFraction(dense, lengths, fraction), offset);
      return [
        strokeLayer([
          createLine(
            dense.map(point => lane(point, -22)),
            'cool',
            0.5,
            0.35
          ),
          createLine(
            dense.map(point => lane(point, 22)),
            'warm',
            0.5,
            0.3
          )
        ]),
        strokeLayer(
          resampled.map((fraction, index) =>
            createLine([place(fraction, -22), place(fraction, 22)], 'dim', 0.3, 0.25, index / count)
          ),
          'pulse',
          0.2
        ),
        strokeLayer(
          [
            createLine(
              dense.map(point => lane(point, 22)),
              'warm',
              1.1,
              0.9
            )
          ],
          'flow',
          0.3,
          1.5
        ),
        spriteLayer(
          rawFractions.map(fraction => {
            const [x, y] = place(fraction, -22);
            return createDot(x, y, 1.5, 'cool', 0.95, fraction);
          }),
          0.3
        ),
        spriteLayer(
          resampled.map((fraction, index) => {
            const [x, y] = place(fraction, 22);
            return createDot(x, y, 1.7, 'warm', 0.95, index / count);
          }),
          0.3
        )
      ];
    }
  },

  'trajectory-playhead': {
    parameters: [
      slider('time', 'Time', 0, 1, 0.01, 0.55),
      slider('vehicles', 'Vehicles', 3, 14, 1, 8)
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const time = pointer ? clamp((pointer[0] - 14) / 132, 0, 1) : p.time;
      const random = fork('tracks');
      const roles: Role[] = ['primary', 'warm', 'cool', 'violet', 'hot'];
      const past: PathItem[] = [];
      const future: PathItem[] = [];
      const vehicles: SpriteItem[] = [];
      for (let vehicle = 0; vehicle < Math.round(p.vehicles); vehicle++) {
        const track = createWanderPath(
          random,
          [random.range(14, 40), random.range(14, 86)],
          random.range(-0.4, 0.4),
          6,
          22,
          0.4
        ).map(point => clampPoint(point, 8));
        const times = [0];
        for (let index = 1; index < track.length; index++)
          times.push(times[index - 1] + random.range(0.4, 1.6));
        const scale = times[times.length - 1];
        const normalized = times.map(value => value / scale);
        let cursor = 1;
        while (cursor < normalized.length - 1 && normalized[cursor] < time) cursor++;
        const amount = clamp(
          (time - normalized[cursor - 1]) / (normalized[cursor] - normalized[cursor - 1]),
          0,
          1
        );
        const here: Point = [
          mixNumbers(track[cursor - 1][0], track[cursor][0], amount),
          mixNumbers(track[cursor - 1][1], track[cursor][1], amount)
        ];
        const role = roles[vehicle % roles.length];
        past.push(createLine([...track.slice(0, cursor), here], role, 1.1, 0.85, vehicle / 14));
        future.push(createLine([here, ...track.slice(cursor)], 'dim', 0.5, 0.45));
        vehicles.push(createDot(here[0], here[1], 2.4, 'ink', 1, vehicle / 14));
      }
      const barY = 94;
      const x = 14 + time * 132;
      const ticks: SpriteItem[] = Array.from({length: 12}, (_, index) =>
        createDot(14 + (index / 11) * 132, barY, 0.7, 'dim', 0.8)
      );
      return [
        strokeLayer(future),
        strokeLayer(past, 'flow', 0.25, 1.4),
        spriteLayer(vehicles, 0.35),
        strokeLayer([
          createLine(
            [
              [14, barY],
              [146, barY]
            ],
            'dim',
            0.5,
            0.6
          ),
          createLine(
            [
              [14, barY],
              [x, barY]
            ],
            'warm',
            1.2,
            0.9
          )
        ]),
        spriteLayer(ticks, 0),
        spriteLayer([createDot(x, barY, 3, 'hot', 1)], 0.4)
      ];
    }
  },

  'trajectory-metrics': {
    parameters: [slider('stop', 'Stop speed', 0.04, 0.5, 0.02, 0.18)],
    build: ({fork, p}) => {
      const random = fork('profile');
      const phase = fork('shape').range(0, 1.2);
      const segments = 44;
      const speeds = Array.from({length: segments}, (_, index) => {
        const stop = (index >= 13 && index <= 17) || (index >= 31 && index <= 34);
        return stop
          ? random.range(0.02, 0.07)
          : 0.6 + 0.3 * Math.sin(index / 5) + random.range(0, 0.1);
      });
      const top = Math.max(...speeds);
      const cumulative = [0];
      speeds.forEach(speed =>
        cumulative.push(cumulative[cumulative.length - 1] + Math.max(speed, 0.03))
      );
      const total = cumulative[cumulative.length - 1];
      const vertices = cumulative.map(
        (value): Point => getCurvePoint(value / total, phase, 24, 42)
      );
      const pieces: PathItem[] = [];
      const stops: Point[] = [];
      let inStop = false;
      speeds.forEach((speed, index) => {
        const level = speed / top;
        const role: Role =
          level < p.stop ? 'warm' : level < 0.6 ? 'cool' : level < 0.82 ? 'primary' : 'hot';
        pieces.push(
          createLine([vertices[index], vertices[index + 1]], role, 1.5, 0.95, index / segments)
        );
        if (level < p.stop && !inStop) stops.push(vertices[index]);
        inStop = level < p.stop;
      });
      const profile: Point[] = speeds.map(
        (speed, index): Point => [12 + (index / (segments - 1)) * 136, 94 - (speed / top) * 12]
      );
      const threshold = 94 - p.stop * 12;
      return [
        strokeLayer(pieces, 'flow', 0.25, 2),
        spriteLayer(
          vertices.map(([x, y], index) =>
            createDot(x, y, 0.8, 'ink', 0.55, index / vertices.length)
          ),
          0.2
        ),
        strokeLayer(
          stops.map(([x, y], index) =>
            createLine(closeRing(createCircle(x, y, 6, 24)), 'warm', 0.9, 0.9, index / 3)
          ),
          'pulse',
          0.5
        ),
        strokeLayer(
          [
            createLine(
              [
                [12, threshold],
                [148, threshold]
              ],
              'dim',
              0.4,
              0.7
            ),
            createLine(profile, 'primary', 0.9, 0.95)
          ],
          'static'
        )
      ];
    }
  },

  'trajectory-kalman': {
    parameters: [
      slider('noise', 'GPS noise', 1, 9, 0.5, 4.5, 'px'),
      slider('gain', 'Filter gain', 0.1, 0.9, 0.05, 0.3),
      {
        ...slider('mode', 'Pass', 0, 1, 1, 1),
        options: ['Filter', 'Smoother']
      }
    ],
    build: ({fork, p}) => {
      const phase = fork('shape').range(0, 1.2);
      const random = fork('noise');
      const count = 44;
      const truth = Array.from(
        {length: count},
        (_, index): Point => getCurvePoint(index / (count - 1), phase, 22, 50)
      );
      const measured = truth.map(
        ([x, y]): Point => [x + random.gaussian() * p.noise, y + random.gaussian() * p.noise]
      );
      const forward: Point[] = [];
      measured.forEach((point, index) => {
        const previous = forward[index - 1] ?? point;
        forward.push([
          previous[0] + p.gain * (point[0] - previous[0]),
          previous[1] + p.gain * (point[1] - previous[1])
        ]);
      });
      const backward: Point[] = new Array<Point>(count);
      for (let index = count - 1; index >= 0; index--) {
        const previous = backward[index + 1] ?? measured[index];
        backward[index] = [
          previous[0] + p.gain * (measured[index][0] - previous[0]),
          previous[1] + p.gain * (measured[index][1] - previous[1])
        ];
      }
      const smooth =
        p.mode >= 1
          ? forward.map(
              (point, index): Point => [
                (point[0] + backward[index][0]) / 2,
                (point[1] + backward[index][1]) / 2
              ]
            )
          : forward;
      return [
        strokeLayer([createLine(truth, 'dim', 0.5, 0.5)]),
        strokeLayer([createLine(measured, 'cool', 0.4, 0.4)]),
        spriteLayer(
          smooth
            .filter((_, index) => index % 3 === 0)
            .map(([x, y], index) => createDot(x, y, 4 + p.noise * 0.5, 'warm', 0.12, index / 15)),
          0.3
        ),
        spriteLayer(
          measured.map(([x, y], index) => createDot(x, y, 1.2, 'cool', 0.9, index / count)),
          0.25
        ),
        strokeLayer([createLine(smooth, 'warm', 1.5, 0.95)], 'flow', 0.3, 1.8)
      ];
    }
  },

  'trajectory-restriction': {
    parameters: [
      slider('start', 'Window start', 0, 0.8, 0.02, 0.25),
      slider('width', 'Window length', 0.1, 1, 0.02, 0.45)
    ],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const center = pointer ?? ([80, 46] as Point);
      const box = {
        left: center[0] - 26,
        right: center[0] + 26,
        top: center[1] - 19,
        bottom: center[1] + 19
      };
      const random = fork('tracks');
      const steps = 120;
      const outside: PathItem[] = [];
      const inside: PathItem[] = [];
      const crossings: SpriteItem[] = [];
      for (let track = 0; track < 7; track++) {
        const points = createCrossingTrack(random, steps, 1.3, 6, 154);
        let run: Point[] = [];
        let rest: Point[] = [];
        const flush = (
          target: PathItem[],
          buffer: Point[],
          role: Role,
          width: number,
          alpha: number
        ) => {
          if (buffer.length > 1) target.push(createLine(buffer, role, width, alpha, track / 7));
        };
        let wasInside = false;
        points.forEach((point, index) => {
          const time = index / steps;
          const isInside =
            point[0] >= box.left &&
            point[0] <= box.right &&
            point[1] >= box.top &&
            point[1] <= box.bottom &&
            time >= p.start &&
            time <= p.start + p.width;
          if (isInside) {
            if (!wasInside) {
              crossings.push(createDot(point[0], point[1], 2, 'hot', 1));
              flush(outside, rest, 'dim', 0.5, 0.5);
              rest = [];
              run = [points[Math.max(0, index - 1)]];
            }
            run.push(point);
          } else {
            if (wasInside) {
              crossings.push(createDot(point[0], point[1], 2, 'cool', 1));
              run.push(point);
              flush(inside, run, 'warm', 1.5, 0.95);
              run = [];
              rest = [point];
            }
            rest.push(point);
          }
          wasInside = isInside;
        });
        flush(outside, rest, 'dim', 0.5, 0.5);
        flush(inside, run, 'warm', 1.5, 0.95);
      }
      const barX = (fraction: number) => 14 + fraction * 132;
      return [
        strokeLayer(outside),
        fillLayer([
          {
            ring: [
              [box.left, box.top],
              [box.right, box.top],
              [box.right, box.bottom],
              [box.left, box.bottom]
            ],
            role: 'violet',
            alpha: 0.12
          }
        ]),
        strokeLayer([
          createLine(
            closeRing([
              [box.left, box.top],
              [box.right, box.top],
              [box.right, box.bottom],
              [box.left, box.bottom]
            ]),
            'violet',
            0.9,
            0.9
          )
        ]),
        strokeLayer(inside, 'flow', 0.35, 1),
        spriteLayer(crossings, 0.4),
        strokeLayer([
          createLine(
            [
              [14, 95],
              [146, 95]
            ],
            'dim',
            0.5,
            0.6
          ),
          createLine(
            [
              [barX(p.start), 95],
              [barX(Math.min(1, p.start + p.width)), 95]
            ],
            'warm',
            1.8,
            0.95
          )
        ])
      ];
    }
  },

  'zone-events': {
    parameters: [
      slider('tracks', 'Tracks', 3, 12, 1, 7),
      slider('zone', 'Zone size', 0.6, 1.4, 0.05, 1)
    ],
    build: ({fork, p}) => {
      const shapeRandom = fork('zones');
      const zones = [
        createBlob(shapeRandom, 50, 50, 18 * p.zone, 14, 0.2),
        createBlob(shapeRandom, 114, 46, 15 * p.zone, 14, 0.2)
      ];
      const random = fork('tracks');
      const steps = 110;
      const inside: PathItem[] = [];
      const outside: PathItem[] = [];
      const enters: SpriteItem[] = [];
      const exits: SpriteItem[] = [];
      for (let track = 0; track < Math.round(p.tracks); track++) {
        const points = createCrossingTrack(random, steps, 0.9);
        let state = -1;
        let buffer: Point[] = [];
        points.forEach((point, index) => {
          const zone = zones.findIndex(ring => isInsideRing(point, ring));
          if (index > 0 && zone !== state) {
            const edge: Point = [
              (point[0] + points[index - 1][0]) / 2,
              (point[1] + points[index - 1][1]) / 2
            ];
            (zone >= 0 ? enters : exits).push(
              createDot(edge[0], edge[1], 2.1, zone >= 0 ? 'hot' : 'cool', 1, track / 12)
            );
            buffer.push(edge);
            (state >= 0 ? inside : outside).push(
              createLine(
                buffer,
                state >= 0 ? 'primary' : 'dim',
                state >= 0 ? 1.3 : 0.5,
                state >= 0 ? 0.95 : 0.5,
                track / 12
              )
            );
            buffer = [edge];
          }
          buffer.push(point);
          state = zone;
        });
        (state >= 0 ? inside : outside).push(
          createLine(
            buffer,
            state >= 0 ? 'primary' : 'dim',
            state >= 0 ? 1.3 : 0.5,
            state >= 0 ? 0.95 : 0.5
          )
        );
      }
      return [
        fillLayer(
          zones.map(ring => ({ring, role: 'violet' as Role, alpha: 0.12})),
          'pulse'
        ),
        strokeLayer(zones.map(ring => createLine(closeRing(ring), 'violet', 0.9, 0.85))),
        strokeLayer(outside),
        strokeLayer(inside, 'flow', 0.35, 1),
        spriteLayer(enters, 0.5),
        spriteLayer(exits, 0.5)
      ];
    }
  },

  'fleet-dwell-recipe': {
    parameters: [
      slider('vehicles', 'Vehicles', 3, 16, 1, 12),
      slider('minimum', 'Min dwell', 0, 5, 0.5, 1.5)
    ],
    build: ({fork, p}) => {
      const rectangle = (x: number, y: number, width: number, height: number): Point[] => [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
      ];
      const zones = [
        rectangle(16, 14, 30, 24),
        rectangle(106, 12, 32, 24),
        rectangle(26, 62, 30, 26),
        rectangle(100, 60, 34, 26)
      ];
      const centers = zones.map(getCentroid);
      const random = fork('routes');
      const totals = zones.map(() => 0);
      const routes: PathItem[] = [];
      const stops: SpriteItem[] = [];
      for (let vehicle = 0; vehicle < Math.round(p.vehicles); vehicle++) {
        const visits = [random.integer(4), random.integer(4), random.integer(4)].filter(
          (zone, index, all) => index === 0 || zone !== all[index - 1]
        );
        const places: Point[] = visits.map(zone => [
          centers[zone][0] + random.gaussian() * 4,
          centers[zone][1] + random.gaussian() * 3
        ]);
        for (let leg = 0; leg + 1 < places.length; leg++)
          routes.push(
            createLine(
              createArc(places[leg], places[leg + 1], random.range(-0.2, 0.2), 16),
              'primary',
              1,
              0.85,
              random.next()
            )
          );
        places.forEach((place, index) => {
          const dwell = random.range(0.5, 5.5);
          const kept = dwell >= p.minimum;
          if (kept) totals[visits[index]] += dwell;
          stops.push(
            createDot(
              place[0],
              place[1],
              0.8 + Math.sqrt(dwell) * 0.8,
              kept ? 'warm' : 'dim',
              kept ? 0.85 : 0.4,
              random.next()
            )
          );
        });
      }
      const peak = Math.max(...totals, 1);
      return [
        fillLayer(
          zones.map((ring, index) => ({
            ring,
            role: 'hot' as Role,
            alpha: 0.05 + 0.35 * (totals[index] / peak)
          })),
          'pulse'
        ),
        strokeLayer(zones.map(ring => createLine(closeRing(ring), 'hot', 0.7, 0.4 + 0.5 * 0))),
        strokeLayer(routes, 'flow', 0.3, 1.2),
        spriteLayer(stops, 0.4)
      ];
    }
  },

  'fleet-dwell-zone-events-recipe': {
    parameters: [
      slider('tracks', 'Tracks', 3, 8, 1, 6),
      slider('wiggle', 'Wiggle', 0.2, 1.6, 0.1, 0.8)
    ],
    build: ({fork, p}) => {
      const zones = [
        {left: 10, right: 50},
        {left: 60, right: 100},
        {left: 110, right: 150}
      ];
      const top = 12;
      const bottom = 62;
      const random = fork('tracks');
      const steps = 100;
      const dim: PathItem[] = [];
      const chords: PathItem[] = [];
      const enters: SpriteItem[] = [];
      const exits: SpriteItem[] = [];
      const stacks: PathItem[] = [];
      const trackCount = Math.round(p.tracks);
      const lengths = zones.map(() => [] as number[]);
      const roles: Role[] = ['primary', 'warm', 'cool', 'violet', 'hot'];
      for (let track = 0; track < trackCount; track++) {
        const yStart = random.range(18, 56);
        const phase = random.range(0, 6);
        const points = Array.from(
          {length: steps + 1},
          (_, index): Point => [
            4 + (index / steps) * 152,
            clamp(
              yStart + 10 * p.wiggle * Math.sin(index / 9 + phase) + random.range(-0.3, 0.3),
              4,
              74
            )
          ]
        );
        dim.push(createLine(points, 'dim', 0.5, 0.5));
        zones.forEach((zone, zoneIndex) => {
          const inside = points.filter(
            point =>
              point[0] >= zone.left &&
              point[0] <= zone.right &&
              point[1] >= top &&
              point[1] <= bottom
          );
          lengths[zoneIndex].push(inside.length > 1 ? (getCumulative(inside).pop() ?? 0) : 0);
          if (inside.length > 1) {
            chords.push(createLine(inside, roles[track % roles.length], 1.5, 0.95, track / 8));
            enters.push(createDot(inside[0][0], inside[0][1], 1.7, 'hot', 1));
            exits.push(
              createDot(inside[inside.length - 1][0], inside[inside.length - 1][1], 1.7, 'cool', 1)
            );
          }
        });
      }
      zones.forEach((zone, zoneIndex) => {
        let cursor = zone.left;
        lengths[zoneIndex].forEach((length, track) => {
          const span = length * 0.14;
          if (span < 0.3) return;
          stacks.push(
            createLine(
              [
                [cursor, 84],
                [cursor + span, 84]
              ],
              roles[track % roles.length],
              4,
              0.9
            )
          );
          cursor += span + 0.7;
        });
      });
      return [
        fillLayer(
          zones.map(zone => ({
            ring: [
              [zone.left, top],
              [zone.right, top],
              [zone.right, bottom],
              [zone.left, bottom]
            ] as Point[],
            role: 'violet' as Role,
            alpha: 0.1
          }))
        ),
        strokeLayer(
          zones.map(zone =>
            createLine(
              closeRing([
                [zone.left, top],
                [zone.right, top],
                [zone.right, bottom],
                [zone.left, bottom]
              ]),
              'violet',
              0.8,
              0.8
            )
          )
        ),
        strokeLayer(dim),
        strokeLayer(chords, 'flow', 0.3, 1.4),
        spriteLayer(enters, 0.4),
        spriteLayer(exits, 0.4),
        strokeLayer(
          zones.map(zone =>
            createLine(
              [
                [zone.left, 92],
                [zone.right, 92]
              ],
              'dim',
              0.4,
              0.5
            )
          )
        ),
        strokeLayer(stacks, 'pulse', 0.3)
      ];
    }
  },

  'trajectory-encounters': {
    parameters: [
      slider('distance', 'Distance', 2, 14, 0.5, 3.5, 'px'),
      slider('tracks', 'Tracks', 6, 20, 1, 10)
    ],
    build: ({fork, p}) => {
      const random = fork('tracks');
      const count = Math.round(p.tracks);
      const steps = 36;
      const tracks = Array.from({length: count}, () => createMeetingTrack(random, steps));
      const links: PathItem[] = [];
      const rings: SpriteItem[] = [];
      for (let first = 0; first < count; first++) {
        for (let second = first + 1; second < count; second++) {
          for (let step = 0; step <= steps; step++) {
            if (getDistance(tracks[first][step], tracks[second][step]) > p.distance) continue;
            const x = (tracks[first][step][0] + tracks[second][step][0]) / 2;
            const y = (tracks[first][step][1] + tracks[second][step][1]) / 2;
            links.push(createLine([tracks[first][step], tracks[second][step]], 'hot', 0.9, 0.9));
            rings.push(createDot(x, y, p.distance * 0.7 + 1.5, 'hot', 0.4, step / steps));
            break;
          }
        }
      }
      return [
        strokeLayer(
          tracks.map((track, index) =>
            createLine(track, index % 2 ? 'primary' : 'violet', 0.8, 0.7, index / count)
          ),
          'flow',
          0.2,
          1.5
        ),
        spriteLayer(rings, 0.5),
        strokeLayer(links, 'pulse', 0.5)
      ];
    }
  },

  'clock-encounters': {
    parameters: [
      slider('ticks', 'Clock ticks', 6, 20, 1, 12),
      slider('distance', 'Distance', 3, 16, 0.5, 9, 'px')
    ],
    build: ({fork, p}) => {
      const mapTracks = (u: number): Point[] => [
        [12 + 136 * u, 90 - 50 * u],
        [12 + 136 * u, 40 + 50 * u + 3 * Math.sin(u * 9)],
        [12 + 136 * u, 70 + 10 * Math.sin(u * 7)]
      ];
      const rowY = [9, 18, 27];
      const random = fork('raw');
      const raw: SpriteItem[] = [];
      rowY.forEach((y, row) => {
        let u = random.range(0, 0.06);
        while (u < 1) {
          raw.push(createDot(12 + 136 * u, y, 1.2, 'cool', 0.7, row / 3));
          u += random.range(0.04, 0.2);
        }
      });
      const ticks = Math.round(p.ticks);
      const clock: PathItem[] = [];
      const aligned: SpriteItem[] = [];
      const mapDots: SpriteItem[] = [];
      const meets: PathItem[] = [];
      const rings: SpriteItem[] = [];
      for (let tick = 0; tick < ticks; tick++) {
        const u = tick / (ticks - 1);
        const x = 12 + 136 * u;
        clock.push(
          createLine(
            [
              [x, 4],
              [x, 32]
            ],
            'dim',
            0.3,
            0.35
          )
        );
        const positions = mapTracks(u);
        positions.forEach((position, row) => {
          aligned.push(createDot(x, rowY[row], 1.3, 'warm', 0.95, u));
          mapDots.push(createDot(position[0], position[1], 1.2, 'warm', 0.7, u));
        });
        for (let first = 0; first < 3; first++) {
          for (let second = first + 1; second < 3; second++) {
            if (getDistance(positions[first], positions[second]) > p.distance) continue;
            meets.push(
              createLine(
                [
                  [x, rowY[first]],
                  [x, rowY[second]]
                ],
                'hot',
                1.1,
                0.95
              )
            );
            rings.push(
              createDot(
                (positions[first][0] + positions[second][0]) / 2,
                (positions[first][1] + positions[second][1]) / 2,
                5,
                'hot',
                0.5,
                u
              )
            );
          }
        }
      }
      const mapPaths = [0, 1, 2].map(row =>
        createLine(
          Array.from({length: 40}, (_, index): Point => mapTracks(index / 39)[row]),
          (['primary', 'violet', 'cool'] as Role[])[row],
          0.8,
          0.6
        )
      );
      return [
        strokeLayer(clock),
        strokeLayer(
          rowY.map(y =>
            createLine(
              [
                [12, y],
                [148, y]
              ],
              'dim',
              0.3,
              0.4
            )
          )
        ),
        spriteLayer(raw, 0),
        strokeLayer(mapPaths, 'flow', 0.2, 1.5),
        spriteLayer(aligned, 0.2),
        spriteLayer(mapDots, 0.2),
        strokeLayer(meets, 'pulse', 0.5),
        spriteLayer(rings, 0.6)
      ];
    }
  },

  'trajectory-closest-approach': {
    parameters: [
      slider('threshold', 'Threshold', 4, 24, 1, 12, 'px'),
      slider('separation', 'Separation', -12, 30, 1, 6, 'px')
    ],
    build: ({p}) => {
      const first: {time: number; point: Point}[] = [
        {time: 0, point: [12, 72]},
        {time: 0.2, point: [48, 56]},
        {time: 0.47, point: [84, 46]},
        {time: 0.72, point: [116, 52]},
        {time: 1, point: [148, 30]}
      ];
      const second: {time: number; point: Point}[] = [
        {time: 0, point: [14, 20]},
        {time: 0.28, point: [52, 34]},
        {time: 0.52, point: [86, 52 + p.separation]},
        {time: 0.82, point: [120, 72]},
        {time: 1, point: [150, 82]}
      ];
      const at = (route: typeof first, time: number): Point => {
        let index = 1;
        while (index < route.length - 1 && route[index].time < time) index++;
        const amount = clamp(
          (time - route[index - 1].time) / (route[index].time - route[index - 1].time),
          0,
          1
        );
        return [
          mixNumbers(route[index - 1].point[0], route[index].point[0], amount),
          mixNumbers(route[index - 1].point[1], route[index].point[1], amount)
        ];
      };
      const samples = 200;
      let closest = 0;
      let closestDistance = Infinity;
      const near: number[] = [];
      for (let step = 0; step <= samples; step++) {
        const distance = getDistance(at(first, step / samples), at(second, step / samples));
        if (distance < closestDistance) {
          closestDistance = distance;
          closest = step;
        }
        if (distance <= p.threshold) near.push(step);
      }
      const slice = (route: typeof first) =>
        near.length > 1 ? near.map(step => at(route, step / samples)) : [];
      const connectors: PathItem[] = Array.from({length: 9}, (_, index) => {
        const time = (index + 0.5) / 9;
        return createLine([at(first, time), at(second, time)], 'dim', 0.3, 0.3, index / 9);
      });
      const a = at(first, closest / samples);
      const b = at(second, closest / samples);
      return [
        strokeLayer(connectors),
        strokeLayer(
          [
            createLine(
              first.map(entry => entry.point),
              'primary',
              0.8,
              0.8
            ),
            createLine(
              second.map(entry => entry.point),
              'violet',
              0.8,
              0.8
            )
          ],
          'flow',
          0.2,
          1.5
        ),
        spriteLayer(
          [...first, ...second].map(entry =>
            createDot(entry.point[0], entry.point[1], 1.1, 'ink', 0.5)
          ),
          0
        ),
        strokeLayer(
          [
            createLine(slice(first), 'warm', 1.8, 0.95),
            createLine(slice(second), 'warm', 1.8, 0.95)
          ].filter(item => item.points.length > 1),
          'pulse',
          0.4
        ),
        strokeLayer([createLine([a, b], 'hot', 1.2, 1)]),
        spriteLayer(
          [
            createDot(a[0], a[1], 2.6, 'hot', 1),
            createDot(b[0], b[1], 2.6, 'hot', 1),
            createDot((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 6, 'hot', 0.35)
          ],
          0.5
        )
      ];
    }
  },

  'track-similarity': {
    parameters: [
      slider('deviation', 'Deviation', 0, 22, 1, 14, 'px'),
      slider('links', 'Alignment links', 4, 24, 1, 12)
    ],
    build: ({fork, p}) => {
      const phase = fork('shape').range(0, 1);
      const count = 48;
      const first = Array.from(
        {length: count},
        (_, index): Point => getCurvePoint(index / (count - 1), phase, 20, 44)
      );
      const second = Array.from({length: count}, (_, index): Point => {
        const u = Math.pow(index / (count - 1), 1.18);
        const [x, y] = getCurvePoint(u, phase, 20, 44);
        return [x, y + p.deviation * Math.sin(u * 7 + 1) + p.deviation * 0.5];
      });
      const links: PathItem[] = [];
      const step = Math.max(1, Math.floor(count / Math.round(p.links)));
      for (let index = 0; index < count; index += step)
        links.push(createLine([first[index], second[index]], 'violet', 0.5, 0.55, index / count));
      let worst = 0;
      let worstFrom: Point = first[0];
      let worstTo: Point = second[0];
      const search = (from: Point[], to: Point[], flip: boolean) => {
        from.forEach(point => {
          let nearest = Infinity;
          let nearestPoint: Point = to[0];
          for (let index = 1; index < to.length; index++) {
            const distance = getSegmentDistanceLocal(point, to[index - 1], to[index]);
            if (distance.value < nearest) {
              nearest = distance.value;
              nearestPoint = distance.point;
            }
          }
          if (nearest > worst) {
            worst = nearest;
            worstFrom = flip ? nearestPoint : point;
            worstTo = flip ? point : nearestPoint;
          }
        });
      };
      search(first, second, false);
      search(second, first, true);
      return [
        strokeLayer(links, 'flow', 0.3, 0.8),
        strokeLayer(
          [createLine(first, 'cool', 1.3, 0.95), createLine(second, 'warm', 1.3, 0.95)],
          'flow',
          0.18,
          1.5
        ),
        strokeLayer([createLine([worstFrom, worstTo], 'hot', 1.4, 1)], 'pulse', 0.5),
        spriteLayer(
          [
            createDot(worstFrom[0], worstFrom[1], 2.4, 'hot', 1),
            createDot(worstTo[0], worstTo[1], 2.4, 'hot', 1)
          ],
          0.5
        )
      ];
    }
  },

  'track-similarity-search': {
    parameters: [
      slider('threshold', 'Threshold', 2, 30, 1, 11, 'px'),
      slider('top', 'Top k', 1, 6, 1, 3)
    ],
    build: ({fork, p}) => {
      const count = 24;
      const query = Array.from(
        {length: count},
        (_, index): Point => getCurvePoint(index / (count - 1), 0.4, 14, 50)
      );
      const random = fork('routes');
      const routes = Array.from({length: 20}, (_, route) => {
        const far = route % 5 === 4;
        const amplitude = 1 + route * 0.95;
        const offset: Point = far
          ? [random.range(-4, 4), random.range(0, 1) > 0.5 ? 38 : -34]
          : [random.range(-1, 1) * amplitude, random.range(-1, 1) * amplitude];
        const phase = random.range(0, 6);
        const points = query.map(
          ([x, y], index): Point => [
            x + offset[0],
            clampPoint([x, y + offset[1] + Math.sin(index / 3 + phase) * amplitude * 0.25], 3)[1]
          ]
        );
        const mean =
          points.reduce((sum, point, index) => sum + getDistance(point, query[index]), 0) / count;
        const ys = points.map(point => point[1]);
        const minimumY = Math.min(...ys);
        const maximumY = Math.max(...ys);
        const gap = Math.max(0, minimumY - 66, 34 - maximumY);
        return {points, mean, pruned: gap > p.threshold, minimumY, maximumY};
      });
      const survivors = routes
        .filter(route => !route.pruned && route.mean <= p.threshold)
        .sort((first, second) => first.mean - second.mean);
      const best = new Set(survivors.slice(0, Math.round(p.top)));
      const pruned = routes.filter(route => route.pruned);
      return [
        strokeLayer(
          pruned.map(route =>
            createLine(
              closeRing([
                [12, route.minimumY],
                [148, route.minimumY],
                [148, route.maximumY],
                [12, route.maximumY]
              ]),
              'dim',
              0.3,
              0.3
            )
          )
        ),
        strokeLayer(
          routes
            .filter(route => !route.pruned && !survivors.includes(route))
            .map(route => createLine(route.points, 'dim', 0.4, 0.35))
        ),
        strokeLayer(pruned.map(route => createLine(route.points, 'dim', 0.3, 0.2))),
        strokeLayer(
          survivors
            .filter(route => !best.has(route))
            .map(route => createLine(route.points, 'primary', 0.8, 0.7)),
          'flow',
          0.2,
          1.2
        ),
        strokeLayer(
          survivors
            .filter(route => best.has(route))
            .map(route => createLine(route.points, 'hot', 1.3, 0.95)),
          'flow',
          0.3,
          1.4
        ),
        strokeLayer([createLine(query, 'warm', 1.6, 1)], 'pulse', 0.4)
      ];
    }
  }
};

function getSegmentDistanceLocal(
  point: Point,
  start: Point,
  end: Point
): {value: number; point: Point} {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const amount = clamp(
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy || 1),
    0,
    1
  );
  const nearest: Point = [start[0] + dx * amount, start[1] + dy * amount];
  return {value: getDistance(point, nearest), point: nearest};
}
