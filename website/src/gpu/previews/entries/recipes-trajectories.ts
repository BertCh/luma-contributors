// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/recipes-trajectories.ts, keyed by entry id.

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
  closeRing,
  getConvexHull,
  getNearestOnSegment,
  getVertexMean,
  isInsideRing
} from '../geometry.ts';
import {
  createDot,
  createFillLayer,
  createPath,
  createSlider,
  createSpriteLayer,
  createStrokeLayer,
  type FillLayerOptions,
  type StrokeLayerOptions
} from '../layers.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type PathItem,
  type PolygonItem,
  type Point,
  type PreviewLayer,
  type Role,
  type SpriteItem
} from '../types.ts';

const STROKE_DEFAULTS: StrokeLayerOptions = {
  animation: 'static',
  speed: 0.3,
  trail: 1,
  blend: 'add'
};

const FILL_DEFAULTS: FillLayerOptions = {animation: 'static', blend: 'add'};

/** Shaft plus head of an arrow, as two layers: a flowing shaft and a static head. */
function createArrowLayers(from: Point, to: Point, role: Role = 'warm', size = 3): PreviewLayer[] {
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const wing = (offset: number): Point => [
    to[0] - Math.cos(angle + offset) * size,
    to[1] - Math.sin(angle + offset) * size
  ];
  return [
    createStrokeLayer([createPath([from, to], role, 0.8, 0.8)], {
      ...STROKE_DEFAULTS,
      animation: 'flow',
      speed: 0.35,
      trail: 0.8
    }),
    createStrokeLayer([createPath([wing(0.5), to, wing(-0.5)], role, 0.8, 0.9)], STROKE_DEFAULTS)
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
      createPath(
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
      createPath(
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

/** Bilinear point inside a quad given as [top-left, top-right, bottom-right, bottom-left]. */
function getQuadPoint(quad: Point[], fractionX: number, fractionY: number): Point {
  const top: Point = [
    mixNumbers(quad[0][0], quad[1][0], fractionX),
    mixNumbers(quad[0][1], quad[1][1], fractionX)
  ];
  const bottom: Point = [
    mixNumbers(quad[3][0], quad[2][0], fractionX),
    mixNumbers(quad[3][1], quad[2][1], fractionX)
  ];
  return [mixNumbers(top[0], bottom[0], fractionY), mixNumbers(top[1], bottom[1], fractionY)];
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
    createPath(
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

type CircuitEdge = {from: number; to: number; conductance: number};

/** Dim wires, animated current along each wire (width by current) and the source and target. */
function createCircuitLayers(
  nodes: Point[],
  edges: CircuitEdge[],
  currents: number[],
  source: number,
  target: number,
  options: {
    minimumWidth: number;
    widthRange: number;
    levelExponent: number;
    nodeRadius: number;
    terminalRadius: number;
    /** Nodes to draw; all nodes when omitted. */
    drawnNodes?: Set<number>;
  }
): PreviewLayer[] {
  const maximum = Math.max(...currents.map(Math.abs)) || 1;
  const flows: PathItem[] = edges.map((edge, index) => {
    const forward = currents[index] >= 0;
    const points = forward
      ? [nodes[edge.from], nodes[edge.to]]
      : [nodes[edge.to], nodes[edge.from]];
    const level = (Math.abs(currents[index]) / maximum) ** options.levelExponent;
    return createPath(
      points,
      'warm',
      options.minimumWidth + level * options.widthRange,
      0.25 + level * 0.7,
      index / edges.length
    );
  });
  return [
    createStrokeLayer(
      edges.map(edge => createPath([nodes[edge.from], nodes[edge.to]], 'dim', 0.35, 0.55)),
      STROKE_DEFAULTS
    ),
    createStrokeLayer(flows, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.4, trail: 1.2}),
    createSpriteLayer(
      nodes.flatMap((node, index) => {
        if (options.drawnNodes && !options.drawnNodes.has(index)) return [];
        if (index === source) return [createDot(node, 'hot', options.terminalRadius, 1)];
        if (index === target) return [createDot(node, 'cool', options.terminalRadius, 1)];
        return [createDot(node, 'primary', options.nodeRadius, 0.8, index / nodes.length)];
      }),
      {pulse: 0.4}
    )
  ];
}

export const RECIPES_TRAJECTORIES_PREVIEWS: Record<string, EntryPreview> = {
  // ------------------------------------------------------------ hot spots, rates and regression

  'hot-spot-analysis-recipe': {
    parameters: [
      createSlider('events', 'Events', 40, 260, 10, 150),
      createSlider('threshold', 'Significance', 0.3, 2.5, 0.1, 1, 'σ')
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
        createStrokeLayer(createLatticeLines(leftX, top, columns, rows, size), STROKE_DEFAULTS),
        createSpriteLayer(toSprites(points, 'primary', 1, fork('size'), 0.85), {pulse: 0.3}),
        ...createArrowLayers([77, 50], [92, 50]),
        createFillLayer(quiet, FILL_DEFAULTS),
        createFillLayer(cool, FILL_DEFAULTS),
        createFillLayer(hot, {...FILL_DEFAULTS, animation: 'pulse'})
      ];
    }
  },

  'space-time-hot-spots-recipe': {
    parameters: [
      createSlider('slices', 'Time slices', 3, 6, 1, 5),
      createSlider('hot', 'Hot cells', 1, 5, 1, 3)
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
          createPath(
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
          createPath(
            [project(column + 0.5, row + 0.5, 0), project(column + 0.5, row + 0.5, sliceCount - 1)],
            'hot',
            0.7,
            0.8
          )
        );
      });
      slices.push(
        createStrokeLayer(frames, STROKE_DEFAULTS),
        createFillLayer(quiet, FILL_DEFAULTS),
        createFillLayer(fillsByRole, {...FILL_DEFAULTS, animation: 'pulse'})
      );
      slices.push(
        createStrokeLayer(spine, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 1.2})
      );
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
      return [
        ...slices,
        ...createArrowLayers([94, 50], [112, 50]),
        createFillLayer(result, {...FILL_DEFAULTS, animation: 'pulse'})
      ];
    }
  },

  'rate-cluster-map-recipe': {
    parameters: [
      createSlider('noise', 'Small-area noise', 0, 0.5, 0.02, 0.28),
      createSlider('shrinkage', 'Smoothing', 0, 1, 0.05, 0.65)
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
        createFillLayer(rawFills, FILL_DEFAULTS),
        createStrokeLayer(
          toPaths([...zonesLeft, ...zonesRight].map(closeRing), 'dim', 0.35, 0.7),
          STROKE_DEFAULTS
        ),
        ...createArrowLayers([70, 46], [86, 46]),
        createFillLayer(quadrants[4], FILL_DEFAULTS),
        createFillLayer([...quadrants[1], ...quadrants[2], ...quadrants[3]], FILL_DEFAULTS),
        createFillLayer(quadrants[0], {...FILL_DEFAULTS, animation: 'pulse'})
      ];
    }
  },

  'excess-risk-recipe': {
    parameters: [
      createSlider('excess', 'Excess', 0, 2, 0.1, 1),
      createSlider('noise', 'Noise', 0, 0.4, 0.02, 0.15)
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
        const [x, y] = getVertexMean(ring);
        const expected = 8;
        fills.push({
          ring,
          role,
          alpha: role === 'primary' ? 0.07 : clamp(Math.abs(ratio - 1) * 0.4, 0.08, 0.3)
        });
        rings.push(createPath(closeRing(createCircle(x, y, expected, 36)), 'ink', 0.5, 0.6));
        disks.push(createDot([x, y], role, expected * Math.sqrt(ratio), 0.8, index / zones.length));
      });
      return [
        createFillLayer(fills, FILL_DEFAULTS),
        createStrokeLayer(toPaths(zones.map(closeRing), 'dim', 0.35, 0.6), STROKE_DEFAULTS),
        createSpriteLayer(disks, {pulse: 0.25}),
        createStrokeLayer(rings, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.25})
      ];
    }
  },

  'spatial-rate-recipe': {
    parameters: [
      createSlider('noise', 'Noise', 0.05, 0.7, 0.05, 0.4),
      createSlider('order', 'Ring', 1, 2, 1, 1)
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
            const neighborColumn = column + columnOffset;
            const neighborRow = row + rowOffset;
            if (
              neighborColumn < 0 ||
              neighborRow < 0 ||
              neighborColumn >= columns ||
              neighborRow >= rows
            )
              continue;
            events +=
              rates[neighborRow * columns + neighborColumn] *
              population[neighborRow * columns + neighborColumn];
            people += population[neighborRow * columns + neighborColumn];
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
      const centroids = zones.map(getVertexMean);
      const links: PathItem[] = [];
      const outlines: PathItem[] = [];
      zones.forEach((ring, index) => {
        const isNeighbor =
          index !== focus &&
          Math.abs((index % columns) - focusColumn) <= order &&
          Math.abs(Math.floor(index / columns) - focusRow) <= order;
        if (isNeighbor) {
          links.push(
            createPath([centroids[index], centroids[focus]], 'warm', 0.8, 0.8, index / zones.length)
          );
          outlines.push(createPath(closeRing(ring), 'warm', 0.6, 0.8));
        }
      });
      return [
        createFillLayer(
          zones.map((ring, index) => ({
            ring,
            role: getClassRole(Math.min(4, Math.floor(((pooled[index] - low) / range) * 5)), 5),
            alpha: 0.4
          })),
          FILL_DEFAULTS
        ),
        createStrokeLayer(toPaths(zones.map(closeRing), 'dim', 0.35, 0.6), STROKE_DEFAULTS),
        createStrokeLayer(outlines, STROKE_DEFAULTS),
        createStrokeLayer(links, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.4, trail: 0.9}),
        createStrokeLayer([createPath(closeRing(zones[focus]), 'ink', 1.1, 0.95)], {
          ...STROKE_DEFAULTS,
          animation: 'pulse',
          speed: 0.5
        }),
        createSpriteLayer([createDot([centroids[focus][0], centroids[focus][1]], 'warm', 2.6, 1)], {
          pulse: 0.5
        })
      ];
    }
  },

  'local-variability-recipe': {
    parameters: [
      createSlider('erratic', 'Erratic area', 0, 1.2, 0.05, 0.8),
      createSlider('threshold', 'LOSH cutoff', 0.3, 1.6, 0.05, 0.8)
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
            const neighborColumn = column + columnOffset;
            const neighborRow = row + rowOffset;
            if (
              (!rowOffset && !columnOffset) ||
              neighborColumn < 0 ||
              neighborRow < 0 ||
              neighborColumn >= columns ||
              neighborRow >= rows
            )
              continue;
            sum += (scores[neighborRow * columns + neighborColumn] - lags[index]) ** 2;
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
        createFillLayer(
          left.filter(item => item.role === 'dim'),
          FILL_DEFAULTS
        ),
        createFillLayer(
          left.filter(item => item.role !== 'dim'),
          {...FILL_DEFAULTS, animation: 'pulse'}
        ),
        createFillLayer(
          right.filter(item => item.role === 'dim'),
          FILL_DEFAULTS
        ),
        createFillLayer(
          right.filter(item => item.role !== 'dim'),
          {...FILL_DEFAULTS, animation: 'pulse'}
        ),
        createStrokeLayer(toPaths(masked, 'dim', 0.3, 0.5), STROKE_DEFAULTS),
        createSpriteLayer(
          [
            createDot([80, 50], 'ink', 1.4, 0.7),
            createDot([75, 50], 'dim', 0.9, 0.6),
            createDot([85, 50], 'dim', 0.9, 0.6)
          ],
          {pulse: 0.3}
        )
      ];
    }
  },

  'quantile-lisa': {
    parameters: [
      createSlider('quantiles', 'Quantiles', 3, 8, 1, 5),
      createSlider('target', 'Class', 1, 8, 1, 5)
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
              createPath([center(index), center(neighbor)], 'hot', 0.9, 0.9, index / values.length)
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
        createFillLayer(background, FILL_DEFAULTS),
        createFillLayer(isolated, FILL_DEFAULTS),
        createFillLayer(clustered, {...FILL_DEFAULTS, animation: 'pulse'}),
        createStrokeLayer(joins, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 0.8})
      ];
    }
  },

  'spatial-regression-recipe': {
    parameters: [
      createSlider('dependence', 'Residual pattern', 0, 1, 0.05, 0.85),
      createSlider('noise', 'Noise', 4, 20, 0.5, 11, 'px')
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
          createPath(
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
          createDot([x, ys[index]], role === 'dim' ? 'primary' : role, 1.6, 0.95, index / count)
        );
        cells.push({
          ring: createCellRing(98 + (index % columns) * 8, 26 + Math.floor(index / columns) * 8, 8),
          role: role === 'dim' ? 'dim' : role,
          alpha: role === 'dim' ? 0.2 : clamp(0.35 + Math.abs(residual) * 0.04, 0.4, 0.9),
          phase: (index % columns) / columns
        });
      });
      return [
        createStrokeLayer(
          [
            createPath(
              [
                [10, 90],
                [76, 90]
              ],
              'dim',
              0.4,
              0.5
            ),
            createPath(
              [
                [10, 90],
                [10, 12]
              ],
              'dim',
              0.4,
              0.5
            )
          ],
          STROKE_DEFAULTS
        ),
        createStrokeLayer(drops, STROKE_DEFAULTS),
        createStrokeLayer(
          [
            createPath(
              [
                [12, intercept + slope * 12],
                [72, intercept + slope * 72]
              ],
              'warm',
              1.1,
              0.95
            )
          ],
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.2}
        ),
        createSpriteLayer(dots, {pulse: 0.3}),
        ...createArrowLayers([79, 50], [94, 50]),
        createFillLayer(
          cells.filter(cell => cell.role === 'dim'),
          FILL_DEFAULTS
        ),
        createFillLayer(
          cells.filter(cell => cell.role !== 'dim'),
          {...FILL_DEFAULTS, animation: 'pulse'}
        )
      ];
    }
  },

  'cluster-and-outline-recipe': {
    parameters: [
      createSlider('epsilon', 'Epsilon', 3, 12, 0.5, 6.5, 'px'),
      createSlider('noise', 'Noise points', 0, 60, 2, 26)
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
        hulls.push(createPath(closeRing(hull), role, 0.9, 0.9, cluster / 4));
        const [x, y] = getVertexMean(members.map(point => point));
        centroids.push(createDot([x, y], 'ink', 2.4, 1, cluster / 4));
      }
      const sprites = points.map(([x, y], index) =>
        createDot(
          [x, y],
          labels[index] >= 0 ? clusterRoles[labels[index] % clusterRoles.length] : 'dim',
          labels[index] >= 0 ? 1.4 : 0.9,
          labels[index] >= 0 ? 0.95 : 0.5,
          (index % 9) / 9
        )
      );
      return [
        createFillLayer(hullFills, FILL_DEFAULTS),
        createSpriteLayer(sprites, {pulse: 0.25}),
        createStrokeLayer(hulls, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.5}),
        createSpriteLayer(centroids, {pulse: 0.5})
      ];
    }
  },

  'points-in-polygons-choropleth-recipe': {
    parameters: [
      createSlider('classes', 'Classes', 2, 6, 1, 5),
      createSlider('density', 'Points', 0.3, 1.6, 0.1, 1)
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
          sprites.push(createDot([x, y], 'ink', 1, 0.85, placement.next()));
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
        createFillLayer(fills, {...FILL_DEFAULTS, animation: 'pulse'}),
        createStrokeLayer(toPaths(zones.map(closeRing), 'dim', 0.4, 0.7), STROKE_DEFAULTS),
        createSpriteLayer(sprites, {pulse: 0.35})
      ];
    }
  },

  'period-comparison-recipe': {
    parameters: [
      createSlider('shift', 'Shift', 0, 4, 0.1, 2.2, 'cells'),
      createSlider('threshold', 'Change cutoff', 0.05, 0.5, 0.01, 0.2)
    ],
    build: ({fork, p}) => {
      const columns = 6;
      const rows = 6;
      const size = 6.5;
      const noise = fork('noise');
      const jitter = Array.from({length: columns * rows}, () => noise.range(-0.04, 0.04));
      const blob = (centerColumn: number, centerRow: number, spread: number, index: number) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        return clamp(
          Math.exp(
            -((column - centerColumn) ** 2 + (row - centerRow) ** 2) / (2 * spread * spread)
          ) + jitter[index],
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
        createFillLayer(panelA, FILL_DEFAULTS),
        createFillLayer(panelB, FILL_DEFAULTS),
        createStrokeLayer(
          [
            createPath(
              [
                [50, 49],
                [57, 49]
              ],
              'dim',
              0.9,
              0.8
            )
          ],
          STROKE_DEFAULTS
        ),
        ...createArrowLayers([102, 49], [111, 49]),
        createFillLayer(steady, FILL_DEFAULTS),
        createFillLayer(losses, {...FILL_DEFAULTS, animation: 'pulse'}),
        createFillLayer(gains, {...FILL_DEFAULTS, animation: 'pulse'})
      ];
    }
  },

  'cell-flow-topology-recipe': {
    parameters: [
      createSlider('flows', 'Flows', 6, 40, 1, 22),
      createSlider('far', 'Long trips', 0, 1, 0.05, 0.3)
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
          createPath(
            arc,
            isNeighbor ? 'warm' : 'cool',
            isNeighbor ? 1 : 0.5,
            isNeighbor ? 0.9 : 0.35,
            random.next()
          )
        );
      }
      const hexes = centers.map(([x, y]) =>
        createPath(
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
        createDot(
          [x, y],
          'primary',
          1 + Math.sqrt(totals[index]) * 1.3,
          0.75,
          index / centers.length
        )
      );
      return [
        createStrokeLayer(hexes, STROKE_DEFAULTS),
        createStrokeLayer(far, STROKE_DEFAULTS),
        createStrokeLayer(near, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 1}),
        createSpriteLayer(sprites, {pulse: 0.35})
      ];
    }
  },

  'change-of-support-recipe': {
    parameters: [
      createSlider('mismatch', 'Misalignment', 0, 0.45, 0.01, 0.25),
      createSlider('contrast', 'Contrast', 0.2, 1, 0.05, 0.8)
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
          const from = getVertexMean(left[sourceIndex]);
          const to = getVertexMean(right[targetIndex]);
          links.push(createPath(createArc(from, to, 0, 10), 'warm', 0.3 + weight * 2, 0.5, weight));
        });
      });
      return [
        createFillLayer(
          left.map((ring, index) => ({
            ring,
            role: 'cool',
            alpha: 0.12 + 0.6 * values[index]
          })),
          FILL_DEFAULTS
        ),
        createFillLayer(
          right.map((ring, index) => ({
            ring,
            role: 'hot',
            alpha: 0.12 + 0.7 * (transfer[index] / maximum)
          })),
          {...FILL_DEFAULTS, animation: 'pulse'}
        ),
        createStrokeLayer(
          toPaths([...left, ...right].map(closeRing), 'dim', 0.45, 0.8),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(links, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.1})
      ];
    }
  },

  'dasymetric-redistribution-recipe': {
    parameters: [
      createSlider('weight', 'Ancillary weight', 0, 1, 0.05, 0.85),
      createSlider('total', 'People per zone', 30, 160, 5, 80)
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
          dots.push(createDot([x, y], 'warm', 1.15, 0.95, random.next()));
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
        createStrokeLayer([createPath(boundary, 'violet', 1.1, 0.95)], {
          ...STROKE_DEFAULTS,
          animation: 'pulse',
          speed: 0.2
        }),
        createStrokeLayer(
          [
            createPath(
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
          ],
          STROKE_DEFAULTS
        ),
        createSpriteLayer(dots, {pulse: 0.4})
      ];
    }
  },

  'zonal-distribution-recipe': {
    parameters: [
      createSlider('bins', 'Bins', 5, 14, 1, 9),
      createSlider('spread', 'Spread', 0.1, 0.4, 0.01, 0.2)
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
        const [centerX, centerY] = getVertexMean(ring);
        const mean = random.range(0.3, 0.7);
        const spread = p.spread * random.range(0.6, 1.3);
        const left = centerX - 19;
        const spacing = 38 / (bins - 1);
        const baseline = centerY + 14;
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
        medians.push(createDot([left + running * spacing, baseline + 4.5], 'warm', 1.9, 1));
        modes.push(
          createDot([left + density.indexOf(peak) * spacing, baseline - 30], 'hot', 1.5, 0.95)
        );
        ticks.push(
          createPath(
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
        createFillLayer(
          zones.map(ring => ({ring, role: 'dim', alpha: 0.07})),
          FILL_DEFAULTS
        ),
        createStrokeLayer(toPaths(zones.map(closeRing), 'dim', 0.4, 0.7), STROKE_DEFAULTS),
        createStrokeLayer(bars, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.4}),
        createStrokeLayer(ticks, STROKE_DEFAULTS),
        createSpriteLayer(medians, {pulse: 0.5}),
        createSpriteLayer(modes, {pulse: 0.4})
      ];
    }
  },

  'raster-zonal-distribution-recipe': {
    parameters: [
      createSlider('nodata', 'Nodata cells', 0, 50, 2, 16),
      createSlider('roughness', 'Roughness', 0.7, 2.4, 0.1, 1.2)
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
        const [centerX, centerY] = getVertexMean(ring);
        const level = (value: number) => centerY + 15 - value * 30;
        whiskers.push(
          createPath(
            [
              [centerX, level(pick(0.05))],
              [centerX, level(pick(0.95))]
            ],
            'ink',
            0.7,
            0.95
          )
        );
        boxes.push({
          ring: [
            [centerX - 4.5, level(pick(0.75))],
            [centerX + 4.5, level(pick(0.75))],
            [centerX + 4.5, level(pick(0.25))],
            [centerX - 4.5, level(pick(0.25))]
          ],
          role: 'warm',
          alpha: 0.7
        });
        medians.push(createDot([centerX, level(pick(0.5))], 'hot', 2, 1));
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
        createFillLayer(holes, FILL_DEFAULTS),
        createStrokeLayer(toPaths(zones.map(closeRing), 'ink', 0.5, 0.75), STROKE_DEFAULTS),
        createFillLayer(boxes, {...FILL_DEFAULTS, animation: 'pulse'}),
        createStrokeLayer(whiskers, STROKE_DEFAULTS),
        createSpriteLayer(medians, {pulse: 0.4})
      ];
    }
  },

  'exposure-by-zone-recipe': {
    parameters: [
      createSlider('threshold', 'Hazard cutoff', 0.3, 0.9, 0.02, 0.6),
      createSlider('people', 'People', 30, 200, 10, 100)
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
            [x, y],
            exposed ? 'ink' : 'cool',
            exposed ? 1.5 : 1,
            exposed ? 1 : 0.5,
            random.next()
          )
        );
      }
      const bars: PathItem[] = zones.map((ring, index) => {
        const bottom = Math.max(...ring.map(point => point[1])) - 3.5;
        const left = Math.min(...ring.map(point => point[0])) + 4;
        return createPath(
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
        createFillLayer(
          unobserved.map(ring => ({ring, role: 'dim', alpha: 0.75})),
          {...FILL_DEFAULTS, animation: 'static', blend: 'alpha'}
        ),
        createStrokeLayer(toPaths(unobserved.map(closeRing), 'dim', 0.5, 0.8), STROKE_DEFAULTS),
        createStrokeLayer(toPaths(zones.map(closeRing), 'ink', 0.5, 0.6), STROKE_DEFAULTS),
        createSpriteLayer(dots, {pulse: 0.35}),
        createStrokeLayer(bars, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.3})
      ];
    }
  },

  'regional-raster-change-recipe': {
    parameters: [
      createSlider('change', 'Change', 0.2, 1.5, 0.05, 0.8),
      createSlider('support', 'Common support', 0, 10, 1, 4, 'cols')
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
        const [x, y] = getVertexMean(ring);
        return createDot(
          [x, y],
          sums[index] >= 0 ? 'hot' : 'cool',
          3 + Math.min(9, Math.abs(sums[index]) * 0.35),
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
        createFillLayer(
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
          {...FILL_DEFAULTS, animation: 'static', blend: 'alpha'}
        ),
        createStrokeLayer(toPaths(zones.map(closeRing), 'ink', 0.5, 0.7), STROKE_DEFAULTS),
        createSpriteLayer(halos, {pulse: 0.4}),
        createSpriteLayer(marks, {pulse: 0.4})
      ];
    }
  },

  'surface-preparation-recipe': {
    parameters: [
      createSlider('gap', 'Gap size', 5, 17, 1, 11),
      createSlider('filter', 'Rank filter', 0, 3, 1, 1)
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
        createFillLayer(
          gaps.map(([x, y]) => ({
            ring: createCircle(x, y, p.gap, 36),
            role: 'warm' as Role,
            alpha: 0.1
          })),
          {...FILL_DEFAULTS, animation: 'pulse'}
        ),
        createStrokeLayer(
          gaps.map(([x, y], index) =>
            createPath(closeRing(createCircle(x, y, p.gap, 36)), 'warm', 0.8, 0.9, index / 3)
          ),
          {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.4}
        )
      ];
    }
  },

  'interpolation-diagnostics-recipe': {
    parameters: [
      createSlider('samples', 'Samples', 6, 26, 1, 14),
      createSlider('power', 'Range', 1, 6, 0.1, 2)
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
      const predicted = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (fractionX, fractionY) =>
        predict(fractionX * SCENE_WIDTH, fractionY * SCENE_HEIGHT)
      );
      const low = Math.min(...predicted);
      const range = Math.max(...predicted) - low || 1;
      const surface = predicted.map(value => (value - low) / range);
      const residuals = samples.map(([x, y], index) => sampleValues[index] - predict(x, y, index));
      const deviation =
        Math.sqrt(residuals.reduce((sum, value) => sum + value * value, 0) / count) || 1;
      const standardizedScores = residuals.map(value => value / deviation);
      const rings: PathItem[] = samples.map(([x, y], index) =>
        createPath(
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
        createStrokeLayer(rings, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.4}),
        createSpriteLayer(toSprites(samples, 'ink', 1.3, undefined, 0.95), {pulse: 0})
      ];
    }
  },

  // ------------------------------------------------------------ hydrology, catchments, landscape

  'drainage-network-recipe': {
    parameters: [
      createSlider('threshold', 'Stream threshold', 3, 60, 1, 8, 'cells'),
      createSlider('roughness', 'Roughness', 0.7, 2.2, 0.1, 1.2)
    ],
    build: ({fork, p}) => {
      const columns = 40;
      const rows = 25;
      const noise = createSmoothField(fork('terrain'), 7, p.roughness, columns, rows);
      const elevation = noise.map((value, index) => {
        const fractionX = (index % columns) / columns;
        const fractionY = Math.floor(index / columns) / rows;
        return value * 0.3 + (1 - (0.75 * fractionX + 0.25 * fractionY)) * 0.7;
      });
      const downstream = new Array<number>(elevation.length).fill(-1);
      for (let index = 0; index < elevation.length; index++) {
        const column = index % columns;
        const row = Math.floor(index / columns);
        let best = 0;
        for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
          for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
            const neighborColumn = column + columnOffset;
            const neighborRow = row + rowOffset;
            if (
              (!rowOffset && !columnOffset) ||
              neighborColumn < 0 ||
              neighborRow < 0 ||
              neighborColumn >= columns ||
              neighborRow >= rows
            )
              continue;
            const drop =
              (elevation[index] - elevation[neighborRow * columns + neighborColumn]) /
              Math.hypot(rowOffset, columnOffset);
            if (drop > best) {
              best = drop;
              downstream[index] = neighborRow * columns + neighborColumn;
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
            createPath(
              [center(index), center(next)],
              amount > p.threshold * 4 ? 'primary' : 'cool',
              width,
              0.95,
              (index % 17) / 17
            )
          );
        } else if (amount > p.threshold * 2) {
          const [x, y] = center(index);
          outlets.push(createDot([x, y], 'warm', 2.4, 1, index / elevation.length));
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
        createStrokeLayer(small, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.2}),
        createStrokeLayer(large, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 1.6}),
        createSpriteLayer(outlets, {pulse: 0.5})
      ];
    }
  },

  'straight-line-catchments-recipe': {
    parameters: [createSlider('facilities', 'Facilities', 2, 7, 1, 5)],
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
        createFillLayer(fills, FILL_DEFAULTS),
        createSpriteLayer(
          facilities.map(([x, y], index) =>
            createDot(
              [x, y],
              getRoleCycle(index),
              3 + Math.sqrt(areas[index]) * 0.35,
              0.35,
              index / 7
            )
          ),
          {pulse: 0.4}
        ),
        createSpriteLayer(toSprites(facilities, 'ink', 2.1, undefined, 1), {pulse: 0})
      ];
    }
  },

  'landscape-metrics-recipe': {
    parameters: [
      createSlider('classes', 'Classes', 2, 5, 1, 4),
      createSlider('fragmentation', 'Fragmentation', 0.8, 3, 0.1, 1.6)
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
              createPath(
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
              createPath(
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
      return [
        createFillLayer(fills, FILL_DEFAULTS),
        createStrokeLayer(edges, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.3})
      ];
    }
  },

  'patch-proximity-recipe': {
    parameters: [
      createSlider('depth', 'Edge depth', 0, 8, 0.5, 3, 'px'),
      createSlider('patches', 'Patches', 4, 9, 1, 8)
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
          createPath(
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
        createFillLayer(
          patches.map(patch => ({
            ring: patch.ring,
            role: (patch.habitat ? 'primary' : 'violet') as Role,
            alpha: patch.habitat ? 0.2 : 0.14
          })),
          FILL_DEFAULTS
        ),
        createStrokeLayer(
          patches.map(patch =>
            createPath(closeRing(patch.ring), patch.habitat ? 'primary' : 'violet', 0.7, 0.8)
          ),
          STROKE_DEFAULTS
        ),
        createFillLayer(cores, {...FILL_DEFAULTS, animation: 'pulse'}),
        createStrokeLayer(links, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 0.8})
      ];
    }
  },

  'resistance-connectivity-recipe': {
    parameters: [
      createSlider('barrier', 'Barrier', 0, 1, 0.05, 0.7),
      createSlider('jitter', 'Layout', 0, 1, 0.1, 0.5)
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
      return createCircuitLayers(nodes, edges, currents, source, target, {
        minimumWidth: 0.3,
        widthRange: 3.2,
        levelExponent: 1,
        nodeRadius: 1.6,
        terminalRadius: 4
      });
    }
  },

  'sparse-resistance-connectivity-recipe': {
    parameters: [
      createSlider('barrier', 'Barrier', 0, 1, 0.05, 0.7),
      createSlider('jitter', 'Layout', 0, 1, 0.1, 0.5)
    ],
    build: ({fork, p}) => {
      const columns = 26;
      const rows = 14;
      const random = fork('layout');
      const nodes: Point[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          nodes.push([
            10 + column * 5.8 + random.range(-1.3, 1.3) * p.jitter,
            12 + row * 6.1 + random.range(-1.3, 1.3) * p.jitter
          ]);
        }
      }
      // A fence of weak links across the middle with a few narrow gaps, over a thinned grid.
      const fenceColumn = Math.floor(columns / 2) - 1;
      const gapRows = new Set([2, 7, 11]);
      const links: CircuitEdge[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const index = row * columns + column;
          const isFence = column === fenceColumn && !gapRows.has(row);
          const sideConductance = isFence ? 1 - 0.97 * p.barrier : 1;
          if (column < columns - 1 && random.next() < 0.78) {
            links.push({from: index, to: index + 1, conductance: sideConductance});
          }
          if (row < rows - 1 && random.next() < 0.72) {
            links.push({from: index, to: index + columns, conductance: 1});
          }
        }
      }
      const source = Math.floor(rows / 2) * columns;
      const target = source + columns - 1;
      // Keep only the wires connected to the source, so every node takes part in the circuit.
      const neighbors: number[][] = nodes.map(() => []);
      for (const link of links) {
        neighbors[link.from].push(link.to);
        neighbors[link.to].push(link.from);
      }
      const reached = new Set([source]);
      const queue = [source];
      for (let cursor = 0; cursor < queue.length; cursor++) {
        for (const neighbor of neighbors[queue[cursor]]) {
          if (!reached.has(neighbor)) {
            reached.add(neighbor);
            queue.push(neighbor);
          }
        }
      }
      if (!reached.has(target)) {
        for (let column = 0; column < columns - 1; column++) {
          const index = source + column;
          links.push({from: index, to: index + 1, conductance: 1});
          reached.add(index).add(index + 1);
        }
      }
      const edges = links.filter(link => reached.has(link.from) && reached.has(link.to));

      // Conjugate gradient on the free nodes, with the source held at 1 and the target at 0.
      const voltage = new Float64Array(nodes.length);
      voltage[source] = 1;
      const isFree = (node: number) => reached.has(node) && node !== source && node !== target;
      const applyLaplacian = (input: Float64Array): Float64Array => {
        const output = new Float64Array(nodes.length);
        for (const edge of edges) {
          const difference = edge.conductance * (input[edge.from] - input[edge.to]);
          output[edge.from] += difference;
          output[edge.to] -= difference;
        }
        for (let node = 0; node < nodes.length; node++) if (!isFree(node)) output[node] = 0;
        return output;
      };
      const residual = applyLaplacian(voltage).map(value => -value);
      const direction = Float64Array.from(residual);
      let residualEnergy = residual.reduce((sum, value) => sum + value * value, 0);
      for (let iteration = 0; iteration < 160 && residualEnergy > 1e-12; iteration++) {
        const response = applyLaplacian(direction);
        let curvature = 0;
        for (let node = 0; node < nodes.length; node++)
          curvature += direction[node] * response[node];
        const step = residualEnergy / (curvature || 1);
        let nextEnergy = 0;
        for (let node = 0; node < nodes.length; node++) {
          voltage[node] += step * direction[node];
          residual[node] -= step * response[node];
          nextEnergy += residual[node] * residual[node];
        }
        const ratio = nextEnergy / residualEnergy;
        for (let node = 0; node < nodes.length; node++) {
          direction[node] = residual[node] + ratio * direction[node];
        }
        residualEnergy = nextEnergy;
      }
      const currents = edges.map(
        edge => edge.conductance * (voltage[edge.from] - voltage[edge.to])
      );
      // Square-root scaling keeps the many faint side paths visible next to the main channels.
      // The terminals get their own small layer: the static fallback draws large sprite layers as
      // uniform dots, which would hide them among hundreds of nodes.
      return [
        ...createCircuitLayers(nodes, edges, currents, source, target, {
          minimumWidth: 0.25,
          widthRange: 1.9,
          levelExponent: 0.5,
          nodeRadius: 0.9,
          terminalRadius: 5.5,
          drawnNodes: reached
        }),
        createSpriteLayer(
          [createDot(nodes[source], 'hot', 5.5, 1), createDot(nodes[target], 'cool', 5.5, 1)],
          {pulse: 0.4}
        )
      ];
    }
  },

  // ------------------------------------------------------------ segregation and alignment

  'segregation-decomposition-recipe': {
    parameters: [
      createSlider('spatial', 'Spatial change', 0, 0.8, 0.05, 0.5),
      createSlider('composition', 'Composition change', 0, 0.4, 0.02, 0.15)
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
              [originX + slots[entry.index][0], originY + slots[entry.index][1]],
              rank < count ? 'hot' : 'cool',
              1.5,
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
        createStrokeLayer(
          [
            createPath(
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
            createPath(
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
          ],
          STROKE_DEFAULTS
        ),
        createSpriteLayer(before, {pulse: 0.25}),
        createSpriteLayer(after, {pulse: 0.25}),
        ...createArrowLayers([64, 26], [93, 26], 'ink'),
        createStrokeLayer(
          [
            createPath(
              [
                [start, 72],
                [start + compositionLength, 72]
              ],
              'cool',
              3.4,
              0.95
            ),
            createPath(
              [
                [start + compositionLength + 1, 72],
                [start + compositionLength + 1 + spatialLength, 72]
              ],
              'warm',
              3.4,
              0.95
            )
          ],
          {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.3}
        ),
        createStrokeLayer(
          [
            createPath(
              [
                [start, 80],
                [start + compositionLength + 1 + spatialLength, 80]
              ],
              'dim',
              0.5,
              0.7
            )
          ],
          STROKE_DEFAULTS
        )
      ];
    }
  },

  'segregation-counterfactual-recipe': {
    parameters: [
      createSlider('spatial', 'Spatial change', 0, 0.8, 0.05, 0.5),
      createSlider('composition', 'Composition change', 0, 0.4, 0.02, 0.15)
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
            [originX + slots[entry.index][0], originY + slots[entry.index][1]],
            rank < count ? 'hot' : 'cool',
            1.25,
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
        createPath(
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
        createStrokeLayer(
          [
            frame(columnLeft, rowTop),
            frame(columnRight, rowTop),
            frame(columnLeft, rowBottom),
            frame(columnRight, rowBottom)
          ],
          STROKE_DEFAULTS
        ),
        createSpriteLayer(sprites, {pulse: 0.25}),
        createStrokeLayer(
          [
            createPath(downLeft, 'cool', 0.8, 0.8),
            createPath(downRight, 'warm', 0.8, 0.8),
            createPath(
              [
                [columnLeft + 40, 30],
                [columnRight - 6, 58]
              ],
              'cool',
              0.8,
              0.7
            ),
            createPath(
              [
                [columnRight - 4, 30],
                [columnLeft + 42, 58]
              ],
              'warm',
              0.8,
              0.7
            )
          ],
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 1.2}
        )
      ];
    }
  },

  'population-support-alignment-recipe': {
    parameters: [
      createSlider('missing', 'Missing IDs', 0, 2, 1, 0),
      createSlider('duplicates', 'Duplicate IDs', 0, 2, 1, 0)
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
      const centroids = units.map(getVertexMean);
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
          createPath(
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
          createPath(
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
          createPath(
            [[44, y], [44 + bend, y], [centroids[id][0] - bend, centroids[id][1]], centroids[id]],
            role,
            0.8,
            0.8,
            row / ids.length
          )
        );
      });
      units.forEach((ring, index) => {
        if (!ids.includes(index)) faults.push(createPath(closeRing(ring), 'hot', 1.4, 0.95));
      });
      return [
        createFillLayer(
          units.map((ring, index) => ({
            ring,
            role: unitRoles[index],
            alpha: 0.22
          })),
          FILL_DEFAULTS
        ),
        createStrokeLayer(toPaths(units.map(closeRing), 'dim', 0.4, 0.7), STROKE_DEFAULTS),
        createStrokeLayer(faults, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.6}),
        createStrokeLayer(rowItems, STROKE_DEFAULTS),
        createStrokeLayer(links, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.4, trail: 1})
      ];
    }
  },

  // ------------------------------------------------------------ trajectories

  'trajectory-resample': {
    parameters: [createSlider('samples', 'Samples', 4, 32, 1, 14)],
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
        createStrokeLayer(
          [
            createPath(
              dense.map(point => lane(point, -22)),
              'cool',
              0.5,
              0.35
            ),
            createPath(
              dense.map(point => lane(point, 22)),
              'warm',
              0.5,
              0.3
            )
          ],
          STROKE_DEFAULTS
        ),
        createStrokeLayer(
          resampled.map((fraction, index) =>
            createPath([place(fraction, -22), place(fraction, 22)], 'dim', 0.3, 0.25, index / count)
          ),
          {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.2}
        ),
        createStrokeLayer(
          [
            createPath(
              dense.map(point => lane(point, 22)),
              'warm',
              1.1,
              0.9
            )
          ],
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.5}
        ),
        createSpriteLayer(
          rawFractions.map(fraction => {
            const [x, y] = place(fraction, -22);
            return createDot([x, y], 'cool', 1.5, 0.95, fraction);
          }),
          {pulse: 0.3}
        ),
        createSpriteLayer(
          resampled.map((fraction, index) => {
            const [x, y] = place(fraction, 22);
            return createDot([x, y], 'warm', 1.7, 0.95, index / count);
          }),
          {pulse: 0.3}
        )
      ];
    }
  },

  'trajectory-playhead': {
    parameters: [
      createSlider('time', 'Time', 0, 1, 0.01, 0.55),
      createSlider('vehicles', 'Vehicles', 3, 14, 1, 8)
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
        past.push(createPath([...track.slice(0, cursor), here], role, 1.1, 0.85, vehicle / 14));
        future.push(createPath([here, ...track.slice(cursor)], 'dim', 0.5, 0.45));
        vehicles.push(createDot(here, 'ink', 2.4, 1, vehicle / 14));
      }
      const barY = 94;
      const x = 14 + time * 132;
      const ticks: SpriteItem[] = Array.from({length: 12}, (_, index) =>
        createDot([14 + (index / 11) * 132, barY], 'dim', 0.7, 0.8)
      );
      return [
        createStrokeLayer(future, STROKE_DEFAULTS),
        createStrokeLayer(past, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.25, trail: 1.4}),
        createSpriteLayer(vehicles, {pulse: 0.35}),
        createStrokeLayer(
          [
            createPath(
              [
                [14, barY],
                [146, barY]
              ],
              'dim',
              0.5,
              0.6
            ),
            createPath(
              [
                [14, barY],
                [x, barY]
              ],
              'warm',
              1.2,
              0.9
            )
          ],
          STROKE_DEFAULTS
        ),
        createSpriteLayer(ticks, {pulse: 0}),
        createSpriteLayer([createDot([x, barY], 'hot', 3, 1)], {pulse: 0.4})
      ];
    }
  },

  'trajectory-metrics': {
    parameters: [createSlider('stop', 'Stop speed', 0.04, 0.5, 0.02, 0.18)],
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
          createPath([vertices[index], vertices[index + 1]], role, 1.5, 0.95, index / segments)
        );
        if (level < p.stop && !inStop) stops.push(vertices[index]);
        inStop = level < p.stop;
      });
      const profile: Point[] = speeds.map(
        (speed, index): Point => [12 + (index / (segments - 1)) * 136, 94 - (speed / top) * 12]
      );
      const threshold = 94 - p.stop * 12;
      return [
        createStrokeLayer(pieces, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.25, trail: 2}),
        createSpriteLayer(
          vertices.map(([x, y], index) =>
            createDot([x, y], 'ink', 0.8, 0.55, index / vertices.length)
          ),
          {pulse: 0.2}
        ),
        createStrokeLayer(
          stops.map(([x, y], index) =>
            createPath(closeRing(createCircle(x, y, 6, 24)), 'warm', 0.9, 0.9, index / 3)
          ),
          {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.5}
        ),
        createStrokeLayer(
          [
            createPath(
              [
                [12, threshold],
                [148, threshold]
              ],
              'dim',
              0.4,
              0.7
            ),
            createPath(profile, 'primary', 0.9, 0.95)
          ],
          {...STROKE_DEFAULTS, animation: 'static'}
        )
      ];
    }
  },

  'trajectory-kalman': {
    parameters: [
      createSlider('noise', 'GPS noise', 1, 9, 0.5, 4.5, 'px'),
      createSlider('gain', 'Filter gain', 0.1, 0.9, 0.05, 0.3),
      {
        ...createSlider('mode', 'Pass', 0, 1, 1, 1),
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
        createStrokeLayer([createPath(truth, 'dim', 0.5, 0.5)], STROKE_DEFAULTS),
        createStrokeLayer([createPath(measured, 'cool', 0.4, 0.4)], STROKE_DEFAULTS),
        createSpriteLayer(
          smooth
            .filter((_, index) => index % 3 === 0)
            .map(([x, y], index) => createDot([x, y], 'warm', 4 + p.noise * 0.5, 0.12, index / 15)),
          {pulse: 0.3}
        ),
        createSpriteLayer(
          measured.map(([x, y], index) => createDot([x, y], 'cool', 1.2, 0.9, index / count)),
          {pulse: 0.25}
        ),
        createStrokeLayer([createPath(smooth, 'warm', 1.5, 0.95)], {
          ...STROKE_DEFAULTS,
          animation: 'flow',
          speed: 0.3,
          trail: 1.8
        })
      ];
    }
  },

  'trajectory-restriction': {
    parameters: [
      createSlider('start', 'Window start', 0, 0.8, 0.02, 0.25),
      createSlider('width', 'Window length', 0.1, 1, 0.02, 0.45)
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
          if (buffer.length > 1) target.push(createPath(buffer, role, width, alpha, track / 7));
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
              crossings.push(createDot(point, 'hot', 2, 1));
              flush(outside, rest, 'dim', 0.5, 0.5);
              rest = [];
              run = [points[Math.max(0, index - 1)]];
            }
            run.push(point);
          } else {
            if (wasInside) {
              crossings.push(createDot(point, 'cool', 2, 1));
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
        createStrokeLayer(outside, STROKE_DEFAULTS),
        createFillLayer(
          [
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
          ],
          FILL_DEFAULTS
        ),
        createStrokeLayer(
          [
            createPath(
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
          ],
          STROKE_DEFAULTS
        ),
        createStrokeLayer(inside, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 1}),
        createSpriteLayer(crossings, {pulse: 0.4}),
        createStrokeLayer(
          [
            createPath(
              [
                [14, 95],
                [146, 95]
              ],
              'dim',
              0.5,
              0.6
            ),
            createPath(
              [
                [barX(p.start), 95],
                [barX(Math.min(1, p.start + p.width)), 95]
              ],
              'warm',
              1.8,
              0.95
            )
          ],
          STROKE_DEFAULTS
        )
      ];
    }
  },

  'zone-events': {
    parameters: [
      createSlider('tracks', 'Tracks', 3, 12, 1, 7),
      createSlider('zone', 'Zone size', 0.6, 1.4, 0.05, 1)
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
              createDot(edge, zone >= 0 ? 'hot' : 'cool', 2.1, 1, track / 12)
            );
            buffer.push(edge);
            (state >= 0 ? inside : outside).push(
              createPath(
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
          createPath(
            buffer,
            state >= 0 ? 'primary' : 'dim',
            state >= 0 ? 1.3 : 0.5,
            state >= 0 ? 0.95 : 0.5
          )
        );
      }
      return [
        createFillLayer(
          zones.map(ring => ({ring, role: 'violet' as Role, alpha: 0.12})),
          {...FILL_DEFAULTS, animation: 'pulse'}
        ),
        createStrokeLayer(
          zones.map(ring => createPath(closeRing(ring), 'violet', 0.9, 0.85)),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(outside, STROKE_DEFAULTS),
        createStrokeLayer(inside, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.35, trail: 1}),
        createSpriteLayer(enters, {pulse: 0.5}),
        createSpriteLayer(exits, {pulse: 0.5})
      ];
    }
  },

  'fleet-dwell-recipe': {
    parameters: [
      createSlider('vehicles', 'Vehicles', 3, 16, 1, 12),
      createSlider('minimum', 'Min dwell', 0, 5, 0.5, 1.5)
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
      const centers = zones.map(getVertexMean);
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
            createPath(
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
              [place[0], place[1]],
              kept ? 'warm' : 'dim',
              0.8 + Math.sqrt(dwell) * 0.8,
              kept ? 0.85 : 0.4,
              random.next()
            )
          );
        });
      }
      const peak = Math.max(...totals, 1);
      return [
        createFillLayer(
          zones.map((ring, index) => ({
            ring,
            role: 'hot' as Role,
            alpha: 0.05 + 0.35 * (totals[index] / peak)
          })),
          {...FILL_DEFAULTS, animation: 'pulse'}
        ),
        createStrokeLayer(
          zones.map(ring => createPath(closeRing(ring), 'hot', 0.7, 0.4 + 0.5 * 0)),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(routes, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.2}),
        createSpriteLayer(stops, {pulse: 0.4})
      ];
    }
  },

  'fleet-dwell-zone-events-recipe': {
    parameters: [
      createSlider('tracks', 'Tracks', 3, 8, 1, 6),
      createSlider('wiggle', 'Wiggle', 0.2, 1.6, 0.1, 0.8)
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
      const dimPaths: PathItem[] = [];
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
        dimPaths.push(createPath(points, 'dim', 0.5, 0.5));
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
            chords.push(createPath(inside, roles[track % roles.length], 1.5, 0.95, track / 8));
            enters.push(createDot(inside[0], 'hot', 1.7, 1));
            exits.push(
              createDot(
                [inside[inside.length - 1][0], inside[inside.length - 1][1]],
                'cool',
                1.7,
                1
              )
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
            createPath(
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
        createFillLayer(
          zones.map(zone => ({
            ring: [
              [zone.left, top],
              [zone.right, top],
              [zone.right, bottom],
              [zone.left, bottom]
            ] as Point[],
            role: 'violet' as Role,
            alpha: 0.1
          })),
          FILL_DEFAULTS
        ),
        createStrokeLayer(
          zones.map(zone =>
            createPath(
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
          ),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(dimPaths, STROKE_DEFAULTS),
        createStrokeLayer(chords, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.4}),
        createSpriteLayer(enters, {pulse: 0.4}),
        createSpriteLayer(exits, {pulse: 0.4}),
        createStrokeLayer(
          zones.map(zone =>
            createPath(
              [
                [zone.left, 92],
                [zone.right, 92]
              ],
              'dim',
              0.4,
              0.5
            )
          ),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(stacks, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.3})
      ];
    }
  },

  'trajectory-encounters': {
    parameters: [
      createSlider('distance', 'Distance', 2, 14, 0.5, 3.5, 'px'),
      createSlider('tracks', 'Tracks', 6, 20, 1, 10)
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
            links.push(createPath([tracks[first][step], tracks[second][step]], 'hot', 0.9, 0.9));
            rings.push(createDot([x, y], 'hot', p.distance * 0.7 + 1.5, 0.4, step / steps));
            break;
          }
        }
      }
      return [
        createStrokeLayer(
          tracks.map((track, index) =>
            createPath(track, index % 2 ? 'primary' : 'violet', 0.8, 0.7, index / count)
          ),
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.2, trail: 1.5}
        ),
        createSpriteLayer(rings, {pulse: 0.5}),
        createStrokeLayer(links, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.5})
      ];
    }
  },

  'clock-encounters': {
    parameters: [
      createSlider('ticks', 'Clock ticks', 6, 20, 1, 12),
      createSlider('distance', 'Distance', 3, 16, 0.5, 9, 'px')
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
          raw.push(createDot([12 + 136 * u, y], 'cool', 1.2, 0.7, row / 3));
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
          createPath(
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
          aligned.push(createDot([x, rowY[row]], 'warm', 1.3, 0.95, u));
          mapDots.push(createDot(position, 'warm', 1.2, 0.7, u));
        });
        for (let first = 0; first < 3; first++) {
          for (let second = first + 1; second < 3; second++) {
            if (getDistance(positions[first], positions[second]) > p.distance) continue;
            meets.push(
              createPath(
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
                [
                  (positions[first][0] + positions[second][0]) / 2,
                  (positions[first][1] + positions[second][1]) / 2
                ],
                'hot',
                5,
                0.5,
                u
              )
            );
          }
        }
      }
      const mapPaths = [0, 1, 2].map(row =>
        createPath(
          Array.from({length: 40}, (_, index): Point => mapTracks(index / 39)[row]),
          (['primary', 'violet', 'cool'] as Role[])[row],
          0.8,
          0.6
        )
      );
      return [
        createStrokeLayer(clock, STROKE_DEFAULTS),
        createStrokeLayer(
          rowY.map(y =>
            createPath(
              [
                [12, y],
                [148, y]
              ],
              'dim',
              0.3,
              0.4
            )
          ),
          STROKE_DEFAULTS
        ),
        createSpriteLayer(raw, {pulse: 0}),
        createStrokeLayer(mapPaths, {
          ...STROKE_DEFAULTS,
          animation: 'flow',
          speed: 0.2,
          trail: 1.5
        }),
        createSpriteLayer(aligned, {pulse: 0.2}),
        createSpriteLayer(mapDots, {pulse: 0.2}),
        createStrokeLayer(meets, {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.5}),
        createSpriteLayer(rings, {pulse: 0.6})
      ];
    }
  },

  'trajectory-closest-approach': {
    parameters: [
      createSlider('threshold', 'Threshold', 4, 24, 1, 12, 'px'),
      createSlider('separation', 'Separation', -12, 30, 1, 6, 'px')
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
        return createPath([at(first, time), at(second, time)], 'dim', 0.3, 0.3, index / 9);
      });
      const closestStart = at(first, closest / samples);
      const closestEnd = at(second, closest / samples);
      return [
        createStrokeLayer(connectors, STROKE_DEFAULTS),
        createStrokeLayer(
          [
            createPath(
              first.map(entry => entry.point),
              'primary',
              0.8,
              0.8
            ),
            createPath(
              second.map(entry => entry.point),
              'violet',
              0.8,
              0.8
            )
          ],
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.2, trail: 1.5}
        ),
        createSpriteLayer(
          [...first, ...second].map(entry =>
            createDot([entry.point[0], entry.point[1]], 'ink', 1.1, 0.5)
          ),
          {pulse: 0}
        ),
        createStrokeLayer(
          [
            createPath(slice(first), 'warm', 1.8, 0.95),
            createPath(slice(second), 'warm', 1.8, 0.95)
          ].filter(item => item.points.length > 1),
          {...STROKE_DEFAULTS, animation: 'pulse', speed: 0.4}
        ),
        createStrokeLayer([createPath([closestStart, closestEnd], 'hot', 1.2, 1)], STROKE_DEFAULTS),
        createSpriteLayer(
          [
            createDot(closestStart, 'hot', 2.6, 1),
            createDot(closestEnd, 'hot', 2.6, 1),
            createDot(
              [(closestStart[0] + closestEnd[0]) / 2, (closestStart[1] + closestEnd[1]) / 2],
              'hot',
              6,
              0.35
            )
          ],
          {pulse: 0.5}
        )
      ];
    }
  },

  'track-similarity': {
    parameters: [
      createSlider('deviation', 'Deviation', 0, 22, 1, 14, 'px'),
      createSlider('links', 'Alignment links', 4, 24, 1, 12)
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
        links.push(createPath([first[index], second[index]], 'violet', 0.5, 0.55, index / count));
      let worst = 0;
      let worstFrom: Point = first[0];
      let worstTo: Point = second[0];
      const search = (from: Point[], to: Point[], flip: boolean) => {
        from.forEach(point => {
          let nearest = Infinity;
          let nearestPoint: Point = to[0];
          for (let index = 1; index < to.length; index++) {
            const segment = getNearestOnSegment(point, to[index - 1], to[index]);
            if (segment.distance < nearest) {
              nearest = segment.distance;
              nearestPoint = segment.point;
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
        createStrokeLayer(links, {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 0.8}),
        createStrokeLayer(
          [createPath(first, 'cool', 1.3, 0.95), createPath(second, 'warm', 1.3, 0.95)],
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.18, trail: 1.5}
        ),
        createStrokeLayer([createPath([worstFrom, worstTo], 'hot', 1.4, 1)], {
          ...STROKE_DEFAULTS,
          animation: 'pulse',
          speed: 0.5
        }),
        createSpriteLayer(
          [createDot(worstFrom, 'hot', 2.4, 1), createDot(worstTo, 'hot', 2.4, 1)],
          {pulse: 0.5}
        )
      ];
    }
  },

  'track-similarity-search': {
    parameters: [
      createSlider('threshold', 'Threshold', 2, 30, 1, 11, 'px'),
      createSlider('top', 'Top k', 1, 6, 1, 3)
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
        createStrokeLayer(
          pruned.map(route =>
            createPath(
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
          ),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(
          routes
            .filter(route => !route.pruned && !survivors.includes(route))
            .map(route => createPath(route.points, 'dim', 0.4, 0.35)),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(
          pruned.map(route => createPath(route.points, 'dim', 0.3, 0.2)),
          STROKE_DEFAULTS
        ),
        createStrokeLayer(
          survivors
            .filter(route => !best.has(route))
            .map(route => createPath(route.points, 'primary', 0.8, 0.7)),
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.2, trail: 1.2}
        ),
        createStrokeLayer(
          survivors
            .filter(route => best.has(route))
            .map(route => createPath(route.points, 'hot', 1.3, 0.95)),
          {...STROKE_DEFAULTS, animation: 'flow', speed: 0.3, trail: 1.4}
        ),
        createStrokeLayer([createPath(query, 'warm', 1.6, 1)], {
          ...STROKE_DEFAULTS,
          animation: 'pulse',
          speed: 0.4
        })
      ];
    }
  }
};
