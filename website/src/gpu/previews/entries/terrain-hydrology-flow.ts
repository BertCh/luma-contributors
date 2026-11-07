// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/terrain-hydrology-flow.ts, keyed by entry id.

import {
  FIELD_COLUMNS,
  FIELD_ROWS,
  clamp,
  createCircle,
  createRandom,
  createScatterPoints,
  createSmoothField,
  evaluateField,
  normalizeValues,
  sampleField,
  type Random
} from '../data.ts';
import {
  getClassRole,
  getLightAngle,
  getSegmentDistance,
  smoothPath,
  toPaths,
  type BuildContext,
  type EntryPreview
} from '../helpers.ts';
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

// ---------------------------------------------------------------------------------------------
// Shared terrain model: a coarse DEM with priority-flood filling, D8 routing and accumulation.
// ---------------------------------------------------------------------------------------------

const GRID_COLUMNS = 40;
const GRID_ROWS = 25;
const GRID_COUNT = GRID_COLUMNS * GRID_ROWS;
const CELL_SIZE = SCENE_WIDTH / GRID_COLUMNS;
const FLOOD_EPSILON = 1e-5;
const SQRT_TWO = Math.SQRT2;

/** East, south-east, south, south-west, west, north-west, north, north-east. */
const NEIGHBOR_OFFSETS: readonly (readonly [number, number, number])[] = [
  [1, 0, 1],
  [1, 1, SQRT_TWO],
  [0, 1, 1],
  [-1, 1, SQRT_TWO],
  [-1, 0, 1],
  [-1, -1, SQRT_TWO],
  [0, -1, 1],
  [1, -1, SQRT_TWO]
];

class MinHeap {
  private keys: number[] = [];
  private items: number[] = [];

  get size(): number {
    return this.keys.length;
  }

  push(key: number, item: number): void {
    let index = this.keys.length;
    this.keys.push(key);
    this.items.push(item);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.keys[parent] <= key) break;
      this.keys[index] = this.keys[parent];
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.keys[index] = key;
    this.items[index] = item;
  }

  pop(): number {
    const top = this.items[0];
    const lastKey = this.keys.pop() as number;
    const lastItem = this.items.pop() as number;
    const count = this.keys.length;
    if (count > 0) {
      let index = 0;
      for (;;) {
        let child = index * 2 + 1;
        if (child >= count) break;
        if (child + 1 < count && this.keys[child + 1] < this.keys[child]) child++;
        if (this.keys[child] >= lastKey) break;
        this.keys[index] = this.keys[child];
        this.items[index] = this.items[child];
        index = child;
      }
      this.keys[index] = lastKey;
      this.items[index] = lastItem;
    }
    return top;
  }
}

function getCellCenter(cell: number): Point {
  return [
    ((cell % GRID_COLUMNS) + 0.5) * CELL_SIZE,
    (Math.floor(cell / GRID_COLUMNS) + 0.5) * CELL_SIZE
  ];
}

function getCellAt(point: Point): number {
  const column = clamp(Math.floor(point[0] / CELL_SIZE), 0, GRID_COLUMNS - 1);
  const row = clamp(Math.floor(point[1] / CELL_SIZE), 0, GRID_ROWS - 1);
  return row * GRID_COLUMNS + column;
}

function getNeighbor(cell: number, index: number): number {
  const column = (cell % GRID_COLUMNS) + NEIGHBOR_OFFSETS[index][0];
  const row = Math.floor(cell / GRID_COLUMNS) + NEIGHBOR_OFFSETS[index][1];
  if (column < 0 || row < 0 || column >= GRID_COLUMNS || row >= GRID_ROWS) return -1;
  return row * GRID_COLUMNS + column;
}

function isBorderCell(cell: number): boolean {
  const column = cell % GRID_COLUMNS;
  const row = Math.floor(cell / GRID_COLUMNS);
  return column === 0 || row === 0 || column === GRID_COLUMNS - 1 || row === GRID_ROWS - 1;
}

function fillDepressions(elevation: Float32Array): Float32Array {
  const filled = new Float32Array(GRID_COUNT).fill(Infinity);
  const closed = new Uint8Array(GRID_COUNT);
  const heap = new MinHeap();
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    if (!isBorderCell(cell)) continue;
    filled[cell] = elevation[cell];
    closed[cell] = 1;
    heap.push(elevation[cell], cell);
  }
  while (heap.size > 0) {
    const cell = heap.pop();
    for (let index = 0; index < 8; index++) {
      const neighbor = getNeighbor(cell, index);
      if (neighbor < 0 || closed[neighbor]) continue;
      closed[neighbor] = 1;
      filled[neighbor] = Math.max(elevation[neighbor], filled[cell] + FLOOD_EPSILON);
      heap.push(filled[neighbor], neighbor);
    }
  }
  return filled;
}

type Terrain = {
  /** Fine DEM, 128 by 80, 0..1. */
  fine: Float32Array;
  elevation: Float32Array;
  filled: Float32Array;
  /** Steepest-descent neighbor, or -1 at outlets and sinks. */
  receiver: Int32Array;
  accumulation: Float32Array;
  /** Cells from highest to lowest filled elevation: upstream first. */
  order: number[];
};

function analyzeTerrain(fine: Float32Array, fill: boolean, walls = false): Terrain {
  const elevation = new Float32Array(GRID_COUNT);
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    const column = Math.floor(((cell % GRID_COLUMNS) + 0.5) * (FIELD_COLUMNS / GRID_COLUMNS));
    const row = Math.floor((Math.floor(cell / GRID_COLUMNS) + 0.5) * (FIELD_ROWS / GRID_ROWS));
    elevation[cell] = fine[row * FIELD_COLUMNS + column];
    if (walls && isBorderCell(cell)) {
      const [x, y] = getCellCenter(cell);
      if (getOutletDistance(x, y) > 9) elevation[cell] += 1;
    }
  }
  return analyzeElevation(fine, elevation, fill);
}

function analyzeElevation(fine: Float32Array, elevation: Float32Array, fill: boolean): Terrain {
  const filled = fill ? fillDepressions(elevation) : elevation;
  const receiver = new Int32Array(GRID_COUNT).fill(-1);
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    if (isBorderCell(cell)) continue;
    let bestSlope = 0;
    for (let index = 0; index < 8; index++) {
      const neighbor = getNeighbor(cell, index);
      const slope = (filled[cell] - filled[neighbor]) / NEIGHBOR_OFFSETS[index][2];
      if (slope > bestSlope) {
        bestSlope = slope;
        receiver[cell] = neighbor;
      }
    }
  }
  const order = Array.from({length: GRID_COUNT}, (_, cell) => cell).sort(
    (first, second) => filled[second] - filled[first]
  );
  const accumulation = new Float32Array(GRID_COUNT).fill(1);
  for (const cell of order) {
    if (receiver[cell] >= 0) accumulation[receiver[cell]] += accumulation[cell];
  }
  return {fine, elevation, filled, receiver, accumulation, order};
}

/** Three outlets on the lower and right edges; every other border cell is a wall. */
const OUTLETS: readonly Point[] = [
  [30, 100],
  [88, 100],
  [160, 56]
];

function getOutletDistance(x: number, y: number): number {
  let best = Infinity;
  for (const [outletX, outletY] of OUTLETS)
    best = Math.min(best, Math.hypot(x - outletX, y - outletY));
  return best;
}

function createTerrain(context: BuildContext, fill = true): Terrain {
  const noise = createSmoothField(context.fork('dem'), 8, 1.7);
  const fine = normalizeValues(
    evaluateField(
      FIELD_COLUMNS,
      FIELD_ROWS,
      (u, v, column, row) =>
        noise[row * FIELD_COLUMNS + column] * 0.55 +
        getOutletDistance(u * SCENE_WIDTH, v * SCENE_HEIGHT) / 170
    )
  );
  const terrain = analyzeTerrain(fine, fill, true);
  return terrain;
}

function getStreamMask(terrain: Terrain, threshold: number): Uint8Array {
  const mask = new Uint8Array(GRID_COUNT);
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    mask[cell] = terrain.accumulation[cell] >= threshold ? 1 : 0;
  }
  return mask;
}

/** Copies each wall cell from its inward neighbor so display layers do not show the border. */
function settleBorder(values: Float32Array): Float32Array {
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    if (!isBorderCell(cell)) continue;
    const column = clamp(cell % GRID_COLUMNS, 1, GRID_COLUMNS - 2);
    const row = clamp(Math.floor(cell / GRID_COLUMNS), 1, GRID_ROWS - 2);
    values[cell] = values[row * GRID_COLUMNS + column];
  }
  return values;
}

function expandCoarse(
  values: Float32Array,
  smooth: boolean,
  columns = GRID_COLUMNS,
  rows = GRID_ROWS
): Float32Array {
  return evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) =>
    smooth
      ? sampleField(values, columns, rows, u, v)
      : values[
          Math.min(rows - 1, Math.floor(v * rows)) * columns +
            Math.min(columns - 1, Math.floor(u * columns))
        ]
  );
}

function createReliefLayer(terrain: Terrain, strength = 3): PreviewLayer {
  return {
    type: 'field',
    columns: FIELD_COLUMNS,
    rows: FIELD_ROWS,
    values: terrain.fine,
    ramp: 'terrain',
    style: 'hillshade',
    strength,
    light: getLightAngle(315)
  };
}

type StreamPath = {points: Point[]; streamClass: number};

/** Walks every stream from its headwater downstream, splitting where the class changes. */
function traceStreamPaths(
  terrain: Terrain,
  mask: Uint8Array,
  getClass: (cell: number) => number
): StreamPath[] {
  const hasInflow = new Uint8Array(GRID_COUNT);
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    const next = terrain.receiver[cell];
    if (mask[cell] && next >= 0 && mask[next]) hasInflow[next] = 1;
  }
  const owned = new Uint8Array(GRID_COUNT);
  const paths: StreamPath[] = [];
  for (const head of terrain.order) {
    if (!mask[head] || hasInflow[head]) continue;
    let current = head;
    let points: Point[] = [getCellCenter(head)];
    let segmentClass = getClass(head);
    owned[head] = 1;
    for (;;) {
      const next = terrain.receiver[current];
      if (next < 0 || !mask[next]) break;
      const currentClass = getClass(current);
      if (currentClass !== segmentClass) {
        paths.push({points, streamClass: segmentClass});
        points = [getCellCenter(current)];
        segmentClass = currentClass;
      }
      points.push(getCellCenter(next));
      if (owned[next]) break;
      owned[next] = 1;
      current = next;
    }
    paths.push({points, streamClass: segmentClass});
  }
  return paths;
}

function createStreamItems(
  paths: StreamPath[],
  getRole: (streamClass: number) => Role,
  getWidth: (streamClass: number) => number,
  alpha = 0.9
): PathItem[] {
  return paths.map((path, index) => ({
    points: smoothPath(path.points, 2),
    role: getRole(path.streamClass),
    width: getWidth(path.streamClass),
    alpha,
    phase: (index * 0.37) % 1
  }));
}

function getAccumulationClass(accumulation: number, threshold: number): number {
  return clamp(Math.floor(Math.log2(accumulation / threshold)), 0, 4);
}

function sprite(point: Point, radius: number, role: Role, alpha = 0.9, phase = 0): SpriteItem {
  return {x: point[0], y: point[1], radius, role, alpha, phase};
}

function getMaxValue(values: ArrayLike<number>): number {
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index++) maximum = Math.max(maximum, values[index]);
  return maximum;
}

/** Follows receivers from a cell until the mask or an outlet. */
function traceDownstream(terrain: Terrain, start: number, mask?: Uint8Array): Point[] {
  const points: Point[] = [getCellCenter(start)];
  let current = start;
  for (let step = 0; step < 90; step++) {
    const next = terrain.receiver[current];
    if (next < 0) break;
    points.push(getCellCenter(next));
    if (mask && mask[next]) break;
    current = next;
  }
  return points;
}

function findNearestCell(mask: Uint8Array, point: Point): number {
  let best = -1;
  let bestDistance = Infinity;
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    if (!mask[cell]) continue;
    const center = getCellCenter(cell);
    const distance = (center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = cell;
    }
  }
  return best;
}

const THRESHOLD_PARAMETER = {
  id: 'threshold',
  label: 'Stream threshold',
  min: 4,
  max: 60,
  step: 2,
  default: 12,
  unit: 'cells'
};

// ---------------------------------------------------------------------------------------------
// Fractional flow: eight neighbour fractions (MFD style) used by several entries.
// ---------------------------------------------------------------------------------------------

function getFlowFractions(terrain: Terrain, cell: number, exponent: number): number[] {
  const fractions = new Array<number>(8).fill(0);
  if (isBorderCell(cell)) return fractions;
  let total = 0;
  for (let index = 0; index < 8; index++) {
    const neighbor = getNeighbor(cell, index);
    const slope = (terrain.filled[cell] - terrain.filled[neighbor]) / NEIGHBOR_OFFSETS[index][2];
    if (slope > 0) {
      fractions[index] = slope ** exponent;
      total += fractions[index];
    }
  }
  if (total > 0) for (let index = 0; index < 8; index++) fractions[index] /= total;
  return fractions;
}

function createFanPaths(
  center: Point,
  fractions: number[],
  length: number,
  role: Role,
  width: number
): PathItem[] {
  const items: PathItem[] = [];
  fractions.forEach((fraction, index) => {
    if (fraction < 0.025) return;
    const [dc, dr, distance] = NEIGHBOR_OFFSETS[index];
    const reach = length * (0.3 + 0.7 * Math.sqrt(fraction));
    items.push({
      points: [center, [center[0] + (dc / distance) * reach, center[1] + (dr / distance) * reach]],
      role,
      width: width * (0.4 + fraction * 1.1),
      alpha: 0.3 + 0.7 * fraction,
      phase: index / 8
    });
  });
  return items;
}

// ---------------------------------------------------------------------------------------------
// Contours: marching squares with stitched polylines.
// ---------------------------------------------------------------------------------------------

type Contour = {points: Point[]; closed: boolean};

/** Edge pairs per marching squares case; edges are 0 top, 1 right, 2 bottom, 3 left. */
const CONTOUR_SEGMENTS: readonly (readonly (readonly [number, number])[])[] = [
  [],
  [[3, 0]],
  [[0, 1]],
  [[3, 1]],
  [[1, 2]],
  [
    [3, 0],
    [1, 2]
  ],
  [[0, 2]],
  [[3, 2]],
  [[3, 2]],
  [[0, 2]],
  [
    [0, 1],
    [3, 2]
  ],
  [[1, 2]],
  [[3, 1]],
  [[0, 1]],
  [[3, 0]],
  []
];

function traceContours(
  values: Float32Array,
  columns: number,
  rows: number,
  level: number,
  closeAtBorder = false
): Contour[] {
  const offset = closeAtBorder ? 1 : 0;
  const gridColumns = columns + offset * 2;
  const gridRows = rows + offset * 2;
  const total = gridColumns * gridRows;
  const getValue = (column: number, row: number) => {
    const sourceColumn = column - offset;
    const sourceRow = row - offset;
    if (sourceColumn < 0 || sourceRow < 0 || sourceColumn >= columns || sourceRow >= rows) {
      return -1;
    }
    return values[sourceRow * columns + sourceColumn];
  };
  const getEdge = (cellColumn: number, cellRow: number, edge: number) => {
    if (edge === 0) return cellRow * gridColumns + cellColumn;
    if (edge === 2) return (cellRow + 1) * gridColumns + cellColumn;
    if (edge === 3) return total + cellRow * gridColumns + cellColumn;
    return total + cellRow * gridColumns + cellColumn + 1;
  };
  const links = new Map<number, number[]>();
  const link = (first: number, second: number) => {
    if (!links.has(first)) links.set(first, []);
    if (!links.has(second)) links.set(second, []);
    (links.get(first) as number[]).push(second);
    (links.get(second) as number[]).push(first);
  };
  for (let row = 0; row < gridRows - 1; row++) {
    for (let column = 0; column < gridColumns - 1; column++) {
      const bits =
        (getValue(column, row) >= level ? 1 : 0) |
        (getValue(column + 1, row) >= level ? 2 : 0) |
        (getValue(column + 1, row + 1) >= level ? 4 : 0) |
        (getValue(column, row + 1) >= level ? 8 : 0);
      for (const [first, second] of CONTOUR_SEGMENTS[bits]) {
        link(getEdge(column, row, first), getEdge(column, row, second));
      }
    }
  }
  const toScene = (x: number, y: number): Point => [
    (x - offset + 0.5) * (SCENE_WIDTH / columns),
    (y - offset + 0.5) * (SCENE_HEIGHT / rows)
  ];
  const getEdgePoint = (id: number): Point => {
    const vertical = id >= total;
    const index = vertical ? id - total : id;
    const column = index % gridColumns;
    const row = Math.floor(index / gridColumns);
    const first = getValue(column, row);
    const second = vertical ? getValue(column, row + 1) : getValue(column + 1, row);
    const amount = (level - first) / (second - first || 1);
    return vertical ? toScene(column, row + amount) : toScene(column + amount, row);
  };
  const visited = new Set<number>();
  const contours: Contour[] = [];
  const walk = (start: number) => {
    const points: Point[] = [];
    let previous = -1;
    let current = start;
    let closed = false;
    for (;;) {
      visited.add(current);
      points.push(getEdgePoint(current));
      const next = (links.get(current) as number[]).find(
        candidate => candidate !== previous && !visited.has(candidate)
      );
      if (next === undefined) {
        closed = (links.get(current) as number[]).includes(start) && points.length > 2;
        break;
      }
      previous = current;
      current = next;
    }
    if (closed) points.push(points[0]);
    contours.push({points, closed});
  };
  for (const [edge, neighbors] of links) {
    if (neighbors.length === 1 && !visited.has(edge)) walk(edge);
  }
  for (const edge of links.keys()) {
    if (!visited.has(edge)) walk(edge);
  }
  return contours.filter(contour => contour.points.length > 2);
}

function isPointInRing(point: Point, ring: Point[]): boolean {
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

const SURFACE_COLUMNS = 64;
const SURFACE_ROWS = 40;

// ---------------------------------------------------------------------------------------------
// Vector fields: uniform wind plus vortices.
// ---------------------------------------------------------------------------------------------

type Vortex = {x: number; y: number; strength: number; radius: number};

const VORTEX_SLOTS: readonly Point[] = [
  [40, 32],
  [112, 28],
  [78, 64],
  [134, 74],
  [24, 76]
];

function createVortices(random: Random, count: number, pointer: Point | null): Vortex[] {
  const vortices = VORTEX_SLOTS.slice(0, count).map(
    ([x, y], index): Vortex => ({
      x: x + random.range(-8, 8),
      y: y + random.range(-6, 6),
      strength: (index % 2 === 0 ? 1 : -1) * random.range(10, 16),
      radius: random.range(9, 14)
    })
  );
  if (pointer && vortices.length > 0) {
    vortices[0] = {x: pointer[0], y: pointer[1], strength: 16, radius: 11};
  }
  return vortices;
}

function evaluateVectorField(vortices: Vortex[], x: number, y: number): Point {
  let velocityX = 0.32;
  let velocityY = 0.04;
  for (const vortex of vortices) {
    const dx = x - vortex.x;
    const dy = y - vortex.y;
    const scale = vortex.strength / (dx * dx + dy * dy + vortex.radius * vortex.radius);
    velocityX += -dy * scale;
    velocityY += dx * scale;
  }
  return [velocityX, velocityY];
}

function traceVectorLine(
  vortices: Vortex[],
  start: Point,
  direction: number,
  step: number,
  maximumSteps: number,
  isBlocked: (point: Point) => boolean
): Point[] {
  const points: Point[] = [];
  let [x, y] = start;
  for (let index = 0; index < maximumSteps; index++) {
    const [firstX, firstY] = evaluateVectorField(vortices, x, y);
    const firstLength = Math.hypot(firstX, firstY) || 1;
    const midX = x + ((direction * firstX) / firstLength) * step * 0.5;
    const midY = y + ((direction * firstY) / firstLength) * step * 0.5;
    const [secondX, secondY] = evaluateVectorField(vortices, midX, midY);
    const secondLength = Math.hypot(secondX, secondY) || 1;
    x += ((direction * secondX) / secondLength) * step;
    y += ((direction * secondY) / secondLength) * step;
    if (x < 0 || y < 0 || x > SCENE_WIDTH || y > SCENE_HEIGHT) break;
    if (isBlocked([x, y])) break;
    points.push([x, y]);
  }
  return points;
}

/** Evenly spaced streamlines by seed and reject, with a spatial hash of accepted samples. */
function createEvenStreamlines(vortices: Vortex[], separation: number): Point[][] {
  const hashSize = Math.max(1, separation);
  const hashColumns = Math.ceil(SCENE_WIDTH / hashSize) + 3;
  const buckets = new Map<number, Point[]>();
  const getKey = (point: Point) =>
    (Math.floor(point[1] / hashSize) + 1) * hashColumns + Math.floor(point[0] / hashSize) + 1;
  const isNear = (point: Point, distance: number) => {
    const column = Math.floor(point[0] / hashSize) + 1;
    const row = Math.floor(point[1] / hashSize) + 1;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const bucket = buckets.get((row + dr) * hashColumns + column + dc);
        if (!bucket) continue;
        for (const sample of bucket) {
          if ((sample[0] - point[0]) ** 2 + (sample[1] - point[1]) ** 2 < distance * distance) {
            return true;
          }
        }
      }
    }
    return false;
  };
  const lines: Point[][] = [];
  const spacing = separation * 1.15;
  for (let y = spacing / 2; y < SCENE_HEIGHT; y += spacing) {
    for (let x = spacing / 2; x < SCENE_WIDTH; x += spacing) {
      const seed: Point = [x, y];
      if (isNear(seed, separation)) continue;
      const blocked = (point: Point) => isNear(point, separation * 0.55);
      const forward = traceVectorLine(vortices, seed, 1, 1.4, 130, blocked);
      const backward = traceVectorLine(vortices, seed, -1, 1.4, 130, blocked);
      const line = [...backward.reverse(), seed, ...forward];
      if (line.length < 8) continue;
      for (const sample of line) {
        const key = getKey(sample);
        if (!buckets.has(key)) buckets.set(key, []);
        (buckets.get(key) as Point[]).push(sample);
      }
      lines.push(line.filter((_, index) => index % 2 === 0));
    }
  }
  return lines;
}

// ---------------------------------------------------------------------------------------------
// Cost surface: friction, Dijkstra accumulation and back-links.
// ---------------------------------------------------------------------------------------------

const ROAD_START: Point = [0, 42];
const ROAD_END: Point = [160, 52];
const BARRIER_LEFT = 88;
const BARRIER_RIGHT = 96;
const BRIDGE_TOP = 58;
const BRIDGE_BOTTOM = 70;
const FRICTION_ZONES = [
  {x: 40, y: 72, rx: 22, ry: 10},
  {x: 124, y: 26, rx: 20, ry: 12},
  {x: 58, y: 18, rx: 14, ry: 8}
];

function getFriction(x: number, y: number, obstacle: number): number {
  if (x >= BARRIER_LEFT && x <= BARRIER_RIGHT) {
    return y >= BRIDGE_TOP && y <= BRIDGE_BOTTOM ? 0.6 : obstacle * 2.4;
  }
  if (getSegmentDistance([x, y], ROAD_START, ROAD_END) < 2.6) return 0.3;
  for (const zone of FRICTION_ZONES) {
    if (((x - zone.x) / zone.rx) ** 2 + ((y - zone.y) / zone.ry) ** 2 < 1) return obstacle;
  }
  return 1;
}

function createFrictionLayer(): PreviewLayer[] {
  const items: PolygonItem[] = FRICTION_ZONES.map(zone => ({
    ring: createCircle(0, 0, 1, 28).map(
      ([x, y]): Point => [zone.x + x * zone.rx, zone.y + y * zone.ry]
    ),
    role: 'violet',
    alpha: 0.2
  }));
  items.push(
    {
      ring: [
        [BARRIER_LEFT, 0],
        [BARRIER_RIGHT, 0],
        [BARRIER_RIGHT, BRIDGE_TOP],
        [BARRIER_LEFT, BRIDGE_TOP]
      ],
      role: 'violet',
      alpha: 0.28
    },
    {
      ring: [
        [BARRIER_LEFT, BRIDGE_BOTTOM],
        [BARRIER_RIGHT, BRIDGE_BOTTOM],
        [BARRIER_RIGHT, SCENE_HEIGHT],
        [BARRIER_LEFT, SCENE_HEIGHT]
      ],
      role: 'violet',
      alpha: 0.28
    }
  );
  return [
    {type: 'fills', items, animation: 'static', blend: 'add'},
    {
      type: 'strokes',
      items: [
        {
          points: [ROAD_START, ROAD_END],
          role: 'cool',
          width: 0.5,
          alpha: 0.55
        }
      ],
      animation: 'static',
      blend: 'add'
    }
  ];
}

function solveCostSurface(obstacle: number, source: number) {
  const friction = new Float32Array(GRID_COUNT);
  for (let cell = 0; cell < GRID_COUNT; cell++) {
    const [x, y] = getCellCenter(cell);
    friction[cell] = getFriction(x, y, obstacle);
  }
  const cost = new Float32Array(GRID_COUNT).fill(Infinity);
  const previous = new Int32Array(GRID_COUNT).fill(-1);
  const heap = new MinHeap();
  cost[source] = 0;
  heap.push(0, source);
  while (heap.size > 0) {
    const cell = heap.pop();
    for (let index = 0; index < 8; index++) {
      const neighbor = getNeighbor(cell, index);
      if (neighbor < 0) continue;
      const next =
        cost[cell] + ((friction[cell] + friction[neighbor]) / 2) * NEIGHBOR_OFFSETS[index][2];
      if (next < cost[neighbor]) {
        cost[neighbor] = next;
        previous[neighbor] = cell;
        heap.push(next, neighbor);
      }
    }
  }
  return {cost, previous};
}

// ---------------------------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------------------------

export const TERRAIN_HYDROLOGY_FLOW_PREVIEWS: Record<string, EntryPreview> = {
  // The whole pipeline: fill, route, accumulate, threshold.
  'terrain-flow': {
    parameters: [
      THRESHOLD_PARAMETER,
      {
        id: 'fill',
        label: 'Depressions',
        min: 0,
        max: 1,
        step: 1,
        default: 1,
        options: ['Raw DEM', 'Filled']
      }
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = createTerrain(context, p.fill >= 1);
      const mask = getStreamMask(terrain, p.threshold);
      const paths = traceStreamPaths(terrain, mask, cell =>
        getAccumulationClass(terrain.accumulation[cell], p.threshold)
      );
      const markers: SpriteItem[] = [];
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        if (p.fill >= 1 && terrain.filled[cell] - terrain.elevation[cell] > 0.01) {
          markers.push(sprite(getCellCenter(cell), 1.3, 'cool', 0.45, (cell % 7) / 7));
        } else if (p.fill < 1 && !isBorderCell(cell) && terrain.receiver[cell] < 0) {
          markers.push(sprite(getCellCenter(cell), 2.6, 'hot', 0.85, (cell % 5) / 5));
        } else if (mask[cell] && terrain.receiver[cell] < 0) {
          markers.push(sprite(getCellCenter(cell), 2.4, 'warm', 0.95, 0));
        }
      }
      return [
        createReliefLayer(terrain, 3),
        {
          type: 'strokes',
          items: createStreamItems(
            paths,
            streamClass => (streamClass < 2 ? 'cool' : 'primary'),
            streamClass => 0.6 + streamClass * 0.5
          ),
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        },
        {type: 'sprites', items: markers, pulse: 0.5}
      ];
    }
  },

  // Basins above a pour point; the pointer moves the pour point.
  'terrain-watersheds': {
    parameters: [THRESHOLD_PARAMETER],
    pointer: 'sticky',
    build: context => {
      const {p, pointer} = context;
      const terrain = createTerrain(context);
      const mask = getStreamMask(terrain, p.threshold);
      let pour = -1;
      if (pointer) {
        pour = findNearestCell(mask, pointer);
      } else {
        const target = getMaxValue(terrain.accumulation) * 0.3;
        let bestGap = Infinity;
        for (let cell = 0; cell < GRID_COUNT; cell++) {
          if (!mask[cell] || isBorderCell(cell)) continue;
          const gap = Math.abs(terrain.accumulation[cell] - target);
          if (gap < bestGap) {
            bestGap = gap;
            pour = cell;
          }
        }
      }
      const basin = new Uint8Array(GRID_COUNT);
      if (pour >= 0) {
        basin[pour] = 1;
        for (let index = GRID_COUNT - 1; index >= 0; index--) {
          const cell = terrain.order[index];
          const next = terrain.receiver[cell];
          if (next >= 0 && basin[next]) basin[cell] = 1;
        }
      }
      const coarse = new Float32Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        coarse[cell] = basin[cell]
          ? 0.58 + 0.4 * terrain.elevation[cell]
          : 0.03 + 0.25 * terrain.elevation[cell];
      }
      const inside = new Uint8Array(GRID_COUNT);
      const outside = new Uint8Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        if (mask[cell]) (basin[cell] ? inside : outside)[cell] = 1;
      }
      const toClass = () => 0;
      const insidePaths = traceStreamPaths(terrain, inside, toClass);
      const outsidePaths = traceStreamPaths(terrain, outside, toClass);
      const layers: PreviewLayer[] = [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: expandCoarse(settleBorder(coarse), false),
          ramp: 'ice',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.08,
          speed: 0.5
        },
        {
          type: 'strokes',
          items: createStreamItems(
            outsidePaths,
            () => 'dim',
            () => 0.6,
            0.5
          ),
          animation: 'static'
        },
        {
          type: 'strokes',
          items: createStreamItems(
            insidePaths,
            () => 'warm',
            () => 1.1
          ),
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        }
      ];
      if (pour >= 0) {
        layers.push({
          type: 'sprites',
          items: [sprite(getCellCenter(pour), 3.6, 'hot', 1, 0)],
          pulse: 0.4
        });
      }
      return layers;
    }
  },

  // Strahler order sets the width and colour of each reach.
  'terrain-stream-order': {
    parameters: [THRESHOLD_PARAMETER],
    pointer: 'none',
    build: context => {
      const terrain = createTerrain(context);
      const mask = getStreamMask(terrain, context.p.threshold);
      const order = new Uint8Array(GRID_COUNT);
      const maxInflow = new Uint8Array(GRID_COUNT);
      const maxCount = new Uint8Array(GRID_COUNT);
      const junctions: SpriteItem[] = [];
      let maximumOrder = 2;
      for (const cell of terrain.order) {
        if (!mask[cell]) continue;
        const value = maxInflow[cell] === 0 ? 1 : maxInflow[cell] + (maxCount[cell] >= 2 ? 1 : 0);
        order[cell] = value;
        maximumOrder = Math.max(maximumOrder, value);
        if (maxCount[cell] >= 2) junctions.push(sprite(getCellCenter(cell), 2.2, 'ink', 0.9, 0));
        const next = terrain.receiver[cell];
        if (next < 0 || !mask[next]) continue;
        if (value > maxInflow[next]) {
          maxInflow[next] = value;
          maxCount[next] = 1;
        } else if (value === maxInflow[next]) {
          maxCount[next]++;
        }
      }
      const paths = traceStreamPaths(terrain, mask, cell => order[cell]);
      return [
        createReliefLayer(terrain, 1.5),
        {
          type: 'strokes',
          items: createStreamItems(
            paths,
            streamClass => getClassRole(streamClass - 1, maximumOrder),
            streamClass => 0.5 + (streamClass - 1) * 1
          ),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {type: 'sprites', items: junctions, pulse: 0.5}
      ];
    }
  },

  // Height above the stream cell each hillslope path reaches first.
  'terrain-height-above-drainage': {
    parameters: [
      THRESHOLD_PARAMETER,
      {
        id: 'cutoff',
        label: 'Floodplain height',
        min: 4,
        max: 60,
        step: 2,
        default: 8,
        unit: '%'
      }
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = createTerrain(context);
      const mask = getStreamMask(terrain, p.threshold);
      const reference = new Float32Array(GRID_COUNT);
      const height = new Float32Array(GRID_COUNT);
      for (let index = GRID_COUNT - 1; index >= 0; index--) {
        const cell = terrain.order[index];
        const next = terrain.receiver[cell];
        reference[cell] = mask[cell] || next < 0 ? terrain.elevation[cell] : reference[next];
        height[cell] = Math.max(0, terrain.elevation[cell] - reference[cell]);
      }
      const maximum = getMaxValue(height) || 1;
      const cutoff = (p.cutoff / 100) * maximum;
      const smooth = expandCoarse(settleBorder(height), true);
      const values = new Float32Array(smooth.length);
      for (let index = 0; index < values.length; index++) {
        const above = smooth[index];
        values[index] =
          above <= cutoff
            ? 0.62 + 0.38 * (1 - above / cutoff)
            : 0.04 + 0.22 * (1 - clamp((above - cutoff) / maximum, 0, 1));
      }
      const random = context.fork('slopes');
      const slopes: Point[][] = [];
      for (let attempt = 0; attempt < 60 && slopes.length < 7; attempt++) {
        const cell = random.integer(GRID_COUNT);
        if (mask[cell] || isBorderCell(cell) || height[cell] < maximum * 0.3) continue;
        slopes.push(smoothPath(traceDownstream(terrain, cell, mask), 1));
      }
      const paths = traceStreamPaths(terrain, mask, () => 0);
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'ice',
          style: 'ramp',
          strength: 0.12,
          speed: 0.5
        },
        {
          type: 'strokes',
          items: createStreamItems(
            paths,
            () => 'primary',
            () => 1
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths(slopes, 'warm', 0.8, 0.9),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: slopes.map(path => sprite(path[0], 1.6, 'warm', 0.95, 0)),
          pulse: 0.4
        }
      ];
    }
  },

  // Wetness, stream power or slope derived from accumulation and slope.
  'terrain-hydrologic-indices': {
    parameters: [
      {
        id: 'index',
        label: 'Index',
        min: 0,
        max: 2,
        step: 1,
        default: 0,
        options: ['Wetness', 'Stream power', 'Slope']
      }
    ],
    pointer: 'none',
    build: context => {
      const terrain = createTerrain(context);
      const slope = new Float32Array(GRID_COUNT);
      const values = new Float32Array(GRID_COUNT);
      const mode = Math.round(context.p.index);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        let steepest = 0;
        for (let index = 0; index < 8; index++) {
          const neighbor = getNeighbor(cell, index);
          if (neighbor < 0) continue;
          steepest = Math.max(
            steepest,
            (terrain.filled[cell] - terrain.filled[neighbor]) / NEIGHBOR_OFFSETS[index][2]
          );
        }
        slope[cell] = Math.max(steepest, 0.004);
        const area = terrain.accumulation[cell];
        values[cell] =
          mode === 0
            ? Math.log(area / slope[cell])
            : mode === 1
              ? Math.log(1 + area * slope[cell] * 40)
              : Math.sqrt(slope[cell]);
      }
      const normalized = normalizeValues(settleBorder(values));
      const mask = getStreamMask(terrain, 10);
      const streams = traceStreamPaths(terrain, mask, () => 0);
      const role: Role = mode === 0 ? 'cool' : mode === 1 ? 'hot' : 'warm';
      const glow: SpriteItem[] = [];
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        if (isBorderCell(cell)) continue;
        const level = normalized[cell];
        const cutoff = mode === 0 ? 0.45 : mode === 1 ? 0.5 : 0.4;
        if (level < cutoff) continue;
        const strength = (level - cutoff) / (1 - cutoff);
        const [x, y] = getCellCenter(cell);
        glow.push(
          sprite([x, y], 1.8 + 3.4 * strength, role, 0.35 + 0.65 * strength, (cell % 9) / 9)
        );
      }
      const layers: PreviewLayer[] = [createReliefLayer(terrain, mode === 2 ? 4 : 2)];
      if (mode !== 2) {
        layers.push({
          type: 'strokes',
          items: streams.map(
            (path, index): PathItem => ({
              points: smoothPath(path.points, 2),
              role: mode === 0 ? 'primary' : 'warm',
              width: mode === 0 ? 1.1 : 0.7,
              alpha: 0.55,
              phase: (index * 0.37) % 1
            })
          ),
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        });
      }
      layers.push({type: 'sprites', items: glow, pulse: mode === 1 ? 0.5 : 0.3});
      return layers;
    }
  },

  // Distance along the D8 path to the stream; a probe shows one path.
  'terrain-flow-distance': {
    parameters: [
      THRESHOLD_PARAMETER,
      {
        id: 'reach',
        label: 'Reach',
        min: 20,
        max: 100,
        step: 5,
        default: 70,
        unit: '%'
      }
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = createTerrain(context);
      const mask = getStreamMask(terrain, p.threshold);
      const distance = new Float32Array(GRID_COUNT);
      for (let index = GRID_COUNT - 1; index >= 0; index--) {
        const cell = terrain.order[index];
        const next = terrain.receiver[cell];
        if (mask[cell] || next < 0) continue;
        const diagonal =
          next % GRID_COLUMNS !== cell % GRID_COLUMNS &&
          Math.floor(next / GRID_COLUMNS) !== Math.floor(cell / GRID_COLUMNS);
        distance[cell] = distance[next] + (diagonal ? SQRT_TWO : 1);
      }
      const maximum = getMaxValue(distance) || 1;
      for (let cell = 0; cell < GRID_COUNT; cell++) distance[cell] /= maximum;
      const probe = getCellAt(pointer ?? [34, 30]);
      const probePath = traceDownstream(terrain, probe, mask);
      const paths = traceStreamPaths(terrain, mask, () => 0);
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: expandCoarse(settleBorder(distance), true),
          ramp: 'spectral',
          style: 'expand',
          levels: 8,
          cutoff: p.reach / 100,
          speed: 0.12
        },
        {
          type: 'strokes',
          items: createStreamItems(
            paths,
            () => 'ink',
            () => 0.7,
            0.7
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: toPaths([smoothPath(probePath, 1)], 'warm', 1.3, 1),
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [sprite(probePath[0], 3, 'warm', 1, 0)],
          pulse: 0.4
        }
      ];
    }
  },

  // One polyline per reach, split at headwaters, confluences and terminals.
  'terrain-stream-network': {
    parameters: [THRESHOLD_PARAMETER],
    pointer: 'none',
    build: context => {
      const terrain = createTerrain(context);
      const mask = getStreamMask(terrain, context.p.threshold);
      const inflow = new Uint8Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        const next = terrain.receiver[cell];
        if (mask[cell] && next >= 0 && mask[next]) inflow[next]++;
      }
      const isNode = (cell: number) =>
        mask[cell] === 1 && (inflow[cell] !== 1 || terrain.receiver[cell] < 0);
      const reaches: Point[][] = [];
      const markers: SpriteItem[] = [];
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        if (!isNode(cell)) continue;
        const center = getCellCenter(cell);
        if (terrain.receiver[cell] < 0 || !mask[terrain.receiver[cell]]) {
          markers.push(sprite(center, 2.8, 'hot', 0.95, 0));
          continue;
        }
        const isHead = inflow[cell] === 0;
        markers.push(
          sprite(center, isHead ? 1.7 : 2.5, isHead ? 'cool' : 'warm', 0.95, (cell % 9) / 9)
        );
        const points: Point[] = [center];
        let current = cell;
        for (let step = 0; step < 90; step++) {
          const next = terrain.receiver[current];
          if (next < 0 || !mask[next]) break;
          points.push(getCellCenter(next));
          if (isNode(next)) break;
          current = next;
        }
        reaches.push(smoothPath(points, 1));
      }
      const roles: Role[] = ['primary', 'cool', 'violet', 'primary'];
      return [
        {
          type: 'strokes',
          items: reaches.map((points, index) => ({
            points,
            role: roles[index % roles.length],
            width: 1.2,
            alpha: 0.9,
            phase: (index * 0.29) % 1
          })),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {type: 'sprites', items: markers, pulse: 0.4}
      ];
    }
  },

  // Wind deflected by relief: streamlines bend around hills.
  'terrain-flow-field': {
    parameters: [
      {
        id: 'direction',
        label: 'Wind direction',
        min: -60,
        max: 60,
        step: 5,
        default: 10,
        unit: '°'
      },
      {
        id: 'deflection',
        label: 'Deflection',
        min: 0,
        max: 2,
        step: 0.1,
        default: 1.2
      }
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const noise = createSmoothField(context.fork('relief'), 6, 1.6);
      const hills: [number, number, number, number][] = [
        [78, 50, 20, 0.9],
        [124, 28, 13, 0.6],
        [38, 70, 11, 0.45]
      ];
      const fine = normalizeValues(
        evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v, column, row) => {
          let height = noise[row * FIELD_COLUMNS + column] * 0.25;
          for (const [hx, hy, radius, amount] of hills) {
            height +=
              amount *
              Math.exp(
                -(((u * SCENE_WIDTH - hx) / radius) ** 2 + ((v * SCENE_HEIGHT - hy) / radius) ** 2)
              );
          }
          return height;
        })
      );
      const getHeight = (x: number, y: number) =>
        sampleField(fine, FIELD_COLUMNS, FIELD_ROWS, x / SCENE_WIDTH, y / SCENE_HEIGHT);
      const angle = (p.direction * Math.PI) / 180;
      const windX = Math.cos(angle);
      const windY = Math.sin(angle);
      const strands: Point[][] = [];
      const seedCount = 20;
      for (let strand = 0; strand < seedCount; strand++) {
        const offset = (strand - (seedCount - 1) / 2) * 6;
        let x = 80 - windX * 110 - windY * offset;
        let y = 50 - windY * 110 + windX * offset;
        const points: Point[] = [];
        for (let step = 0; step < 150; step++) {
          const gx = (getHeight(x + 1.5, y) - getHeight(x - 1.5, y)) / 3;
          const gy = (getHeight(x, y + 1.5) - getHeight(x, y - 1.5)) / 3;
          const gradient = Math.hypot(gx, gy) || 1;
          const into = Math.max(0, windX * (gx / gradient) + windY * (gy / gradient));
          const factor = clamp(gradient * 45, 0, 1) * p.deflection;
          let vx = windX - factor * into * (gx / gradient);
          let vy = windY - factor * into * (gy / gradient);
          const length = Math.hypot(vx, vy) || 1;
          vx /= length;
          vy /= length;
          x += vx * 2;
          y += vy * 2;
          if (x >= 0 && x <= SCENE_WIDTH && y >= 0 && y <= SCENE_HEIGHT) points.push([x, y]);
          else if (points.length > 0) break;
        }
        if (points.length > 3) strands.push(points.filter((_, index) => index % 2 === 0));
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: fine,
          ramp: 'terrain',
          style: 'hillshade',
          strength: 4,
          light: getLightAngle(315)
        },
        {
          type: 'strokes',
          items: strands.map((points, index) => ({
            points,
            role: 'primary' as Role,
            width: 0.8,
            alpha: 0.85,
            phase: (index * 0.13) % 1
          })),
          animation: 'flow',
          speed: 0.28,
          trail: 1.4,
          blend: 'add'
        }
      ];
    }
  },

  // Eight neighbour fractions per cell; the exponent sharpens D8-like or spreads MFD-like.
  'terrain-flow-fractions': {
    parameters: [
      {
        id: 'exponent',
        label: 'Flow exponent',
        min: 1,
        max: 12,
        step: 0.5,
        default: 3
      }
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = createTerrain(context);
      const probe = getCellAt(pointer ?? [80, 50]);
      const probeCenter = getCellCenter(probe);
      const fans: PathItem[] = [];
      const centers: SpriteItem[] = [];
      for (let row = 1; row < GRID_ROWS - 1; row += 3) {
        for (let column = 1; column < GRID_COLUMNS - 1; column += 3) {
          const cell = row * GRID_COLUMNS + column;
          const center = getCellCenter(cell);
          if (Math.hypot(center[0] - probeCenter[0], center[1] - probeCenter[1]) < 17) continue;
          fans.push(
            ...createFanPaths(center, getFlowFractions(terrain, cell, p.exponent), 9, 'cool', 1)
          );
          centers.push(sprite(center, 0.9, 'dim', 0.7, 0));
        }
      }
      return [
        {
          type: 'strokes',
          items: fans,
          animation: 'flow',
          speed: 0.5,
          trail: 0.8,
          blend: 'add'
        },
        {type: 'sprites', items: centers},
        {
          type: 'strokes',
          items: createFanPaths(
            probeCenter,
            getFlowFractions(terrain, probe, p.exponent),
            17,
            'warm',
            2.2
          ),
          animation: 'flow',
          speed: 0.5,
          trail: 1.2,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [sprite(probeCenter, 2.6, 'warm', 1, 0)],
          pulse: 0.4
        }
      ];
    }
  },

  // Mass, expected distance or outlet probability pushed through the fraction DAG.
  'terrain-branching-flow': {
    parameters: [
      {
        id: 'quantity',
        label: 'Propagate',
        min: 0,
        max: 2,
        step: 1,
        default: 0,
        options: ['Mass', 'Flow distance', 'Outlet probability']
      }
    ],
    pointer: 'sticky',
    build: context => {
      const {p, pointer} = context;
      const mode = Math.round(p.quantity);
      // A gently tilted, slightly rippled hillside, so mass from one source fans out downslope.
      const noise = createSmoothField(context.fork('hillside'), 5, 1.2);
      const source = getCellAt(pointer ?? [18, 50]);
      const [sourceX, sourceY] = getCellCenter(source);
      const elevation = new Float32Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        const [x, y] = getCellCenter(cell);
        const column = Math.min(FIELD_COLUMNS - 1, Math.floor((x / SCENE_WIDTH) * FIELD_COLUMNS));
        const row = Math.min(FIELD_ROWS - 1, Math.floor((y / SCENE_HEIGHT) * FIELD_ROWS));
        elevation[cell] =
          1 -
          Math.hypot(x - sourceX, (y - sourceY) * 0.32) / 150 +
          noise[row * FIELD_COLUMNS + column] * 0.02;
      }
      const order = Array.from({length: GRID_COUNT}, (_, cell) => cell).sort(
        (first, second) => elevation[second] - elevation[first]
      );
      const fractions = Array.from({length: GRID_COUNT}, (_, cell) => {
        const result = new Array<number>(8).fill(0);
        if (isBorderCell(cell)) return result;
        let total = 0;
        for (let index = 0; index < 8; index++) {
          const slope =
            (elevation[cell] - elevation[getNeighbor(cell, index)]) / NEIGHBOR_OFFSETS[index][2];
          if (slope > 0) {
            result[index] = slope ** 1.2;
            total += result[index];
          }
        }
        if (total > 0) for (let index = 0; index < 8; index++) result[index] /= total;
        return result;
      });
      const outlet = getCellAt([142, 50]);
      const value = new Float32Array(GRID_COUNT);
      const strokes: PathItem[] = [];
      if (mode === 0) {
        value[source] = 1;
        for (const cell of order) {
          if (value[cell] <= 0) continue;
          fractions[cell].forEach((fraction, index) => {
            if (fraction <= 0) return;
            const neighbor = getNeighbor(cell, index);
            const weight = value[cell] * fraction;
            value[neighbor] += weight;
          });
        }
        for (let cell = 0; cell < GRID_COUNT; cell++) value[cell] = value[cell] ** 0.3;
        // Seeded walkers pick each next cell by the flow fractions, tracing the braided fan.
        const walker = context.fork('walkers');
        for (let index = 0; index < 26; index++) {
          let cell = source;
          const points: Point[] = [getCellCenter(cell)];
          for (let step = 0; step < 120; step++) {
            const choice = walker.next();
            let cumulative = 0;
            let next = -1;
            fractions[cell].forEach((fraction, direction) => {
              cumulative += fraction;
              if (next < 0 && fraction > 0 && choice <= cumulative)
                next = getNeighbor(cell, direction);
            });
            if (next < 0) break;
            cell = next;
            points.push(getCellCenter(cell));
          }
          strokes.push({
            points: smoothPath(points, 2),
            role: index % 3 === 0 ? 'hot' : 'warm',
            width: 0.9,
            alpha: 0.65,
            phase: index / 26
          });
        }
      } else {
        for (let index = GRID_COUNT - 1; index >= 0; index--) {
          const cell = order[index];
          fractions[cell].forEach((fraction, direction) => {
            if (fraction <= 0) return;
            const neighbor = getNeighbor(cell, direction);
            value[cell] +=
              mode === 2
                ? fraction * value[neighbor]
                : fraction * (NEIGHBOR_OFFSETS[direction][2] + value[neighbor]);
          });
          if (mode === 2 && cell === outlet) value[cell] = 1;
        }
        const maximum = getMaxValue(value) || 1;
        for (let cell = 0; cell < GRID_COUNT; cell++) value[cell] /= maximum;
        if (mode === 2) for (let cell = 0; cell < GRID_COUNT; cell++) value[cell] **= 0.5;
      }
      const layers: PreviewLayer[] = [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: expandCoarse(settleBorder(value), true),
          ramp: mode === 0 ? 'heat' : mode === 1 ? 'spectral' : 'ice',
          style: 'ramp',
          strength: 0.12,
          speed: 0.6
        }
      ];
      if (strokes.length > 0) {
        layers.push({
          type: 'strokes',
          items: strokes,
          animation: 'flow',
          speed: 0.4,
          blend: 'add'
        });
      }
      layers.push({
        type: 'sprites',
        items: [
          sprite(
            getCellCenter(mode === 2 ? outlet : source),
            3.4,
            mode === 0 ? 'ink' : 'warm',
            1,
            0
          )
        ],
        pulse: 0.4
      });
      return layers;
    }
  },

  // Tiles resolved in a wave, with streams crossing tile edges through exchanged cells.
  'terrain-tiled-flow': {
    parameters: [
      {
        id: 'tiles',
        label: 'Tiles across',
        min: 2,
        max: 8,
        step: 1,
        default: 4
      },
      THRESHOLD_PARAMETER
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = createTerrain(context);
      const mask = getStreamMask(terrain, p.threshold);
      const tilesX = Math.round(p.tiles);
      const tilesY = Math.max(2, Math.round(tilesX * 0.625));
      const tileOf = (cell: number) =>
        Math.floor(((cell % GRID_COLUMNS) * tilesX) / GRID_COLUMNS) +
        Math.floor((Math.floor(cell / GRID_COLUMNS) * tilesY) / GRID_ROWS) * tilesX;
      const tiles: PolygonItem[] = [];
      const edges: PathItem[] = [];
      const width = SCENE_WIDTH / tilesX;
      const height = SCENE_HEIGHT / tilesY;
      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          const [x, y] = [tx * width, ty * height];
          tiles.push({
            ring: [
              [x + 0.8, y + 0.8],
              [x + width - 0.8, y + 0.8],
              [x + width - 0.8, y + height - 0.8],
              [x + 0.8, y + height - 0.8]
            ],
            role: 'primary',
            alpha: 0.09,
            phase: (tx + ty) / (tilesX + tilesY - 2)
          });
        }
      }
      for (let tx = 1; tx < tilesX; tx++) {
        edges.push({
          points: [
            [tx * width, 0],
            [tx * width, SCENE_HEIGHT]
          ],
          role: 'dim',
          width: 0.4,
          alpha: 0.6
        });
      }
      for (let ty = 1; ty < tilesY; ty++) {
        edges.push({
          points: [
            [0, ty * height],
            [SCENE_WIDTH, ty * height]
          ],
          role: 'dim',
          width: 0.4,
          alpha: 0.6
        });
      }
      const paths = traceStreamPaths(terrain, mask, cell =>
        getAccumulationClass(terrain.accumulation[cell], p.threshold)
      );
      const exchanges: SpriteItem[] = [];
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        const next = terrain.receiver[cell];
        if (!mask[cell] || next < 0 || tileOf(cell) === tileOf(next)) continue;
        const [ax, ay] = getCellCenter(cell);
        const [bx, by] = getCellCenter(next);
        exchanges.push(
          sprite([(ax + bx) / 2, (ay + by) / 2], 2, 'warm', 0.95, (tileOf(cell) % 5) / 5)
        );
      }
      return [
        {type: 'fills', items: tiles, animation: 'pulse', blend: 'add'},
        {type: 'strokes', items: edges, animation: 'static'},
        {
          type: 'strokes',
          items: createStreamItems(
            paths,
            () => 'cool',
            streamClass => 0.7 + streamClass * 0.45
          ),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {type: 'sprites', items: exchanges, pulse: 0.5}
      ];
    }
  },

  // Two walled pits; the cheapest cut opens only if it fits the bound.
  'terrain-breach': {
    parameters: [
      {
        id: 'maxLength',
        label: 'Max cut length',
        min: 10,
        max: 48,
        step: 1,
        default: 28,
        unit: 'px'
      }
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const noise = createSmoothField(context.fork('noise'), 6, 1.6);
      const pits = [
        {x: 42, y: 34, radius: 8, depth: 0.32},
        {x: 90, y: 70, radius: 14, depth: 0.36}
      ].map(pit => {
        const length = pit.radius * 2.6;
        const end: Point = [pit.x + length, pit.y + 2];
        return {...pit, length, end, open: length <= p.maxLength};
      });
      const fine = normalizeValues(
        evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v, column, row) => {
          const x = u * SCENE_WIDTH;
          const y = v * SCENE_HEIGHT;
          let height = 0.75 - 0.5 * u + noise[row * FIELD_COLUMNS + column] * 0.05;
          for (const pit of pits) {
            const distance = Math.hypot(x - pit.x, y - pit.y);
            height -= pit.depth * Math.exp(-((distance / pit.radius) ** 2) * 1.2);
            height +=
              0.24 * Math.exp(-(((distance - pit.radius * 1.35) / (pit.radius * 0.45)) ** 2));
            if (pit.open) {
              const cut = getSegmentDistance([x, y], [pit.x, pit.y], pit.end);
              height -= 0.35 * Math.exp(-((cut / 2.2) ** 2));
            }
          }
          return height;
        })
      );
      const cuts: PathItem[] = [];
      const runoff: PathItem[] = [];
      const markers: SpriteItem[] = [];
      for (const pit of pits) {
        markers.push(sprite([pit.x, pit.y], 3.2, pit.open ? 'cool' : 'hot', 0.95, 0));
        if (!pit.open) continue;
        cuts.push({
          points: [[pit.x, pit.y], pit.end],
          role: 'warm',
          width: 1.8,
          alpha: 1
        });
        runoff.push({
          points: smoothPath(
            [
              [pit.x, pit.y],
              pit.end,
              [pit.end[0] + 30, pit.end[1] + 2],
              [SCENE_WIDTH, pit.end[1] + 4]
            ],
            2
          ),
          role: 'cool',
          width: 1,
          alpha: 0.9
        });
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: fine,
          ramp: 'terrain',
          style: 'hillshade',
          strength: 5,
          light: getLightAngle(315)
        },
        {
          type: 'strokes',
          items: cuts,
          animation: 'draw',
          speed: 0.25,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: runoff,
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {type: 'sprites', items: markers, pulse: 0.5}
      ];
    }
  },

  // Lakes rise with the stage and merge across sills.
  'terrain-depression-hierarchy': {
    parameters: [
      {
        id: 'stage',
        label: 'Stage',
        min: 0,
        max: 100,
        step: 2,
        default: 82,
        unit: '%'
      }
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const noise = createSmoothField(context.fork('noise'), 6, 1.6, GRID_COLUMNS, GRID_ROWS);
      const pits = [
        {x: 40, y: 30, radius: 13, depth: 0.34},
        {x: 62, y: 38, radius: 10, depth: 0.26},
        {x: 104, y: 68, radius: 15, depth: 0.38},
        {x: 126, y: 54, radius: 10, depth: 0.24},
        {x: 32, y: 76, radius: 11, depth: 0.28}
      ];
      const elevation = new Float32Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        const [x, y] = getCellCenter(cell);
        let height =
          0.62 - 0.22 * (x / SCENE_WIDTH) - 0.12 * (y / SCENE_HEIGHT) + noise[cell] * 0.08;
        for (const pit of pits) {
          height -= pit.depth * Math.exp(-((Math.hypot(x - pit.x, y - pit.y) / pit.radius) ** 2));
        }
        elevation[cell] = height;
      }
      const filled = fillDepressions(elevation);
      let low = Infinity;
      let high = -Infinity;
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        low = Math.min(low, elevation[cell]);
        if (filled[cell] - elevation[cell] > 0.003) high = Math.max(high, filled[cell]);
      }
      const level = low + (p.stage / 100) * (high - low);
      const depth = new Float32Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        depth[cell] = Math.max(0, Math.min(level, filled[cell]) - elevation[cell]);
      }
      const smoothDepth = expandCoarse(depth, true);
      const smoothElevation = normalizeValues(expandCoarse(elevation, true));
      const values = new Float32Array(smoothDepth.length);
      for (let index = 0; index < values.length; index++) {
        values[index] =
          smoothDepth[index] > 0.003
            ? 0.6 + 0.4 * clamp(smoothDepth[index] * 4, 0, 1)
            : 0.05 + 0.3 * smoothElevation[index];
      }
      const label = new Int32Array(GRID_COUNT).fill(-1);
      let labelCount = 0;
      for (let seed = 0; seed < GRID_COUNT; seed++) {
        if (depth[seed] <= 0.003 || label[seed] >= 0) continue;
        const stack = [seed];
        label[seed] = labelCount;
        while (stack.length > 0) {
          const cell = stack.pop() as number;
          for (let index = 0; index < 8; index++) {
            const neighbor = getNeighbor(cell, index);
            if (neighbor >= 0 && depth[neighbor] > 0.003 && label[neighbor] < 0) {
              label[neighbor] = labelCount;
              stack.push(neighbor);
            }
          }
        }
        labelCount++;
      }
      const centers = pits.map((pit): [Point, number] => [
        [pit.x, pit.y],
        label[getCellAt([pit.x, pit.y])]
      ]);
      const groups = new Map<number, Point[]>();
      for (const [center, id] of centers) {
        if (id < 0) continue;
        if (!groups.has(id)) groups.set(id, []);
        (groups.get(id) as Point[]).push(center);
      }
      const links: PathItem[] = [];
      for (const members of groups.values()) {
        for (let index = 1; index < members.length; index++) {
          links.push({
            points: [members[0], members[index]],
            role: 'warm',
            width: 1,
            alpha: 0.9
          });
        }
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'ice',
          style: 'ramp',
          strength: 0.1,
          speed: 0.5
        },
        {
          type: 'strokes',
          items: links,
          animation: 'flow',
          speed: 0.4,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: centers.map(([center, id]) =>
            sprite(center, id >= 0 ? 2.4 : 1.6, id >= 0 ? 'warm' : 'cool', 0.95, id >= 0 ? 0 : 0.5)
          ),
          pulse: 0.4
        }
      ];
    }
  },

  // Largest catchments coloured apart, with their hypsometric curves.
  'terrain-catchment-summary': {
    parameters: [{id: 'count', label: 'Catchments', min: 2, max: 6, step: 1, default: 4}],
    pointer: 'none',
    build: context => {
      const count = Math.round(context.p.count);
      const terrain = createTerrain(context);
      const label = new Int32Array(GRID_COUNT).fill(-1);
      const area = new Map<number, number>();
      for (let index = GRID_COUNT - 1; index >= 0; index--) {
        const cell = terrain.order[index];
        const next = terrain.receiver[cell];
        label[cell] = next < 0 ? cell : label[next];
        area.set(label[cell], (area.get(label[cell]) ?? 0) + 1);
      }
      const outlets = [...area.entries()]
        .sort((first, second) => second[1] - first[1])
        .slice(0, count);
      const rankOf = new Map(outlets.map(([outlet], rank) => [outlet, rank]));
      const coarse = new Float32Array(GRID_COUNT);
      for (let cell = 0; cell < GRID_COUNT; cell++) {
        const rank = rankOf.get(label[cell]);
        coarse[cell] =
          rank === undefined ? 0.02 : (rank + 0.15 + 0.75 * terrain.elevation[cell]) / count;
      }
      const curves: PathItem[] = [];
      const outletSprites: SpriteItem[] = [];
      outlets.forEach(([outlet], rank) => {
        const role = getClassRole(rank, count);
        const heights: number[] = [];
        let best = outlet;
        for (let cell = 0; cell < GRID_COUNT; cell++) {
          if (label[cell] !== outlet) continue;
          heights.push(terrain.elevation[cell]);
          if (terrain.accumulation[cell] > terrain.accumulation[best]) best = cell;
        }
        heights.sort((first, second) => first - second);
        const low = heights[0];
        const span = heights[heights.length - 1] - low || 1;
        const points: Point[] = [];
        for (let step = 0; step <= 16; step++) {
          const index = Math.min(
            heights.length - 1,
            Math.round((step / 16) * (heights.length - 1))
          );
          points.push([8 + (1 - step / 16) * 36, 92 - ((heights[index] - low) / span) * 22]);
        }
        curves.push({
          points: points.reverse(),
          role,
          width: 1.1,
          alpha: 0.95,
          phase: rank / count
        });
        outletSprites.push(sprite(getCellCenter(best), 2.4, role, 0.95, rank / count));
      });
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: expandCoarse(settleBorder(coarse), false),
          ramp: 'spectral',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.08,
          speed: 0.5
        },
        {
          type: 'fills',
          items: [
            {
              ring: [
                [4, 66],
                [48, 66],
                [48, 96],
                [4, 96]
              ],
              role: 'dim',
              alpha: 0.08
            }
          ],
          animation: 'static',
          blend: 'alpha'
        },
        {
          type: 'strokes',
          items: [
            {
              points: [
                [6, 68],
                [6, 93],
                [46, 93]
              ],
              role: 'ink',
              width: 0.4,
              alpha: 0.6
            }
          ],
          animation: 'static'
        },
        {
          type: 'strokes',
          items: createStreamItems(
            traceStreamPaths(terrain, getStreamMask(terrain, 14), () => 0),
            () => 'ink',
            () => 0.5,
            0.45
          ),
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: curves,
          animation: 'draw',
          speed: 0.2,
          blend: 'add'
        },
        {type: 'sprites', items: outletSprites, pulse: 0.4}
      ];
    }
  },

  // Reference and moving DEM profiles; robust shift from stable ground, outliers ignored.
  'terrain-dem-coregistration': {
    parameters: [
      {
        id: 'shift',
        label: 'Vertical offset',
        min: -9,
        max: 9,
        step: 0.5,
        default: 6,
        unit: 'm'
      },
      {
        id: 'change',
        label: 'Real change',
        min: 0,
        max: 14,
        step: 1,
        default: 9,
        unit: 'm'
      }
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const random = context.fork('profile');
      // Three slow waves, so the terrain profile reads as a calm ridge and valley line.
      const waves = [90, 55, 36].map((period, index) => ({
        frequency: 1 / period,
        phase: random.range(0, 6.28),
        amplitude: 8 / (index + 1)
      }));
      const sampleCount = 72;
      const xs = Array.from(
        {length: sampleCount},
        (_, index) => 8 + (index / (sampleCount - 1)) * 144
      );
      const reference = xs.map(
        x =>
          30 -
          waves.reduce(
            (sum, wave) => sum + wave.amplitude * Math.sin(x * wave.frequency * 6.28 + wave.phase),
            0
          )
      );
      const unit = 1.2;
      const changeAt = (x: number) =>
        p.change * (Math.exp(-(((x - 52) / 8) ** 2)) - 0.8 * Math.exp(-(((x - 112) / 7) ** 2)));
      const isChanged = (x: number) => Math.abs(changeAt(x)) > 2.5;
      const raw = xs.map((x, index) => reference[index] + (p.shift - changeAt(x)) * unit);
      const corrected = xs.map((x, index) => reference[index] - changeAt(x) * unit);
      const zeroLine = 80;
      const zip = (values: number[]): Point[] => xs.map((x, index) => [x, values[index]]);
      const difference = (values: number[]) =>
        values.map((value, index) => zeroLine + value - reference[index]);
      const links: PathItem[] = [];
      const stable: SpriteItem[] = [];
      const outliers: SpriteItem[] = [];
      xs.forEach((x, index) => {
        if (index % 4 !== 2) return;
        const changed = isChanged(x);
        if (!changed) {
          links.push({
            points: [
              [x, reference[index]],
              [x, raw[index]]
            ],
            role: 'warm',
            width: 0.45,
            alpha: 0.5,
            phase: (index % 9) / 9
          });
        }
        (changed ? outliers : stable).push(
          sprite(
            [x, reference[index]],
            changed ? 2 : 1.5,
            changed ? 'hot' : 'cool',
            0.95,
            (index % 7) / 7
          )
        );
      });
      const ground: Point[] = [[xs[0], 56], ...zip(reference), [xs[sampleCount - 1], 56]];
      return [
        {
          type: 'fills',
          items: [{ring: ground, role: 'primary', alpha: 0.1}],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: [
            {
              points: [
                [8, zeroLine],
                [152, zeroLine]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.8
            }
          ],
          animation: 'static'
        },
        {type: 'strokes', items: links, animation: 'pulse', speed: 0.3, blend: 'add'},
        {
          type: 'strokes',
          items: [{points: zip(raw), role: 'warm', width: 1.2, alpha: 0.85}],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: [{points: zip(reference), role: 'primary', width: 1.6, alpha: 0.95}],
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: [{points: zip(difference(raw)), role: 'warm', width: 0.9, alpha: 0.5}],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'strokes',
          items: [{points: zip(difference(corrected)), role: 'cool', width: 1.4, alpha: 0.95}],
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {type: 'sprites', items: [...stable, ...outliers], pulse: 0.4}
      ];
    }
  },

  // Cheapest travel cost expanding from a source across friction.
  'cost-distance': {
    parameters: [
      {
        id: 'budget',
        label: 'Travel budget',
        min: 10,
        max: 100,
        step: 5,
        default: 60,
        unit: '%'
      },
      {
        id: 'obstacle',
        label: 'Obstacle friction',
        min: 2,
        max: 10,
        step: 0.5,
        default: 5
      }
    ],
    pointer: 'sticky',
    build: context => {
      const {p, pointer} = context;
      const source = getCellAt(pointer ?? [22, 52]);
      const {cost} = solveCostSurface(p.obstacle, source);
      const maximum = getMaxValue(cost);
      for (let cell = 0; cell < GRID_COUNT; cell++) cost[cell] /= maximum;
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: expandCoarse(cost, true),
          ramp: 'heat',
          style: 'expand',
          levels: 8,
          cutoff: p.budget / 100,
          speed: 0.12
        },
        ...createFrictionLayer(),
        {
          type: 'sprites',
          items: [sprite(getCellCenter(source), 3.6, 'ink', 1, 0)],
          pulse: 0.4
        }
      ];
    }
  },

  // Back-links from a target trace the cheapest route over the same surface.
  'cost-distance-path': {
    parameters: [
      {
        id: 'obstacle',
        label: 'Obstacle friction',
        min: 2,
        max: 10,
        step: 0.5,
        default: 5
      }
    ],
    pointer: 'sticky',
    build: context => {
      const {p, pointer} = context;
      const source = getCellAt([14, 78]);
      const target = getCellAt(pointer ?? [146, 20]);
      const {cost, previous} = solveCostSurface(p.obstacle, source);
      const maximum = getMaxValue(cost);
      for (let cell = 0; cell < GRID_COUNT; cell++) cost[cell] /= maximum;
      const route: Point[] = [];
      for (
        let cell = target, guard = 0;
        cell >= 0 && guard < GRID_COUNT;
        cell = previous[cell], guard++
      ) {
        route.push(getCellCenter(cell));
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: expandCoarse(cost, true),
          ramp: 'ice',
          style: 'contours',
          levels: 12,
          strength: 0.1,
          speed: 0.3
        },
        ...createFrictionLayer(),
        {
          type: 'strokes',
          items: toPaths([smoothPath(route.reverse(), 2)], 'hot', 1.7, 1),
          animation: 'flow',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: [
            sprite(getCellCenter(source), 3.4, 'warm', 1, 0),
            sprite(getCellCenter(target), 3.4, 'hot', 1, 0.5)
          ],
          pulse: 0.4
        }
      ];
    }
  },

  // Distance to the nearest seed, with the nearest-seed allocation colouring each zone.
  'distance-field': {
    parameters: [{id: 'seeds', label: 'Seeds', min: 2, max: 12, step: 1, default: 6}],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const seeds = createScatterPoints(context.fork('seeds'), Math.round(p.seeds), 14);
      if (pointer) seeds.push([pointer[0], pointer[1]]);
      const count = seeds.length;
      const label = new Int32Array(FIELD_COLUMNS * FIELD_ROWS);
      const distance = new Float32Array(FIELD_COLUMNS * FIELD_ROWS);
      const zoneMaximum = new Float32Array(count);
      for (let row = 0; row < FIELD_ROWS; row++) {
        for (let column = 0; column < FIELD_COLUMNS; column++) {
          const x = ((column + 0.5) / FIELD_COLUMNS) * SCENE_WIDTH;
          const y = ((row + 0.5) / FIELD_ROWS) * SCENE_HEIGHT;
          let best = 0;
          let bestDistance = Infinity;
          for (let seed = 0; seed < count; seed++) {
            const gap = (seeds[seed][0] - x) ** 2 + (seeds[seed][1] - y) ** 2;
            if (gap < bestDistance) {
              bestDistance = gap;
              best = seed;
            }
          }
          const index = row * FIELD_COLUMNS + column;
          label[index] = best;
          distance[index] = Math.sqrt(bestDistance);
          zoneMaximum[best] = Math.max(zoneMaximum[best], distance[index]);
        }
      }
      const values = new Float32Array(distance.length);
      for (let index = 0; index < values.length; index++) {
        const zone = label[index];
        values[index] =
          (zone + 0.1 + 0.8 * (1 - distance[index] / (zoneMaximum[zone] || 1))) / count;
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'spectral',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.1,
          speed: 0.6
        },
        {
          type: 'strokes',
          items: seeds.map((seed, index) => ({
            points: createCircle(seed[0], seed[1], 7, 24).concat([[seed[0] + 7, seed[1]]]),
            role: 'ink' as Role,
            width: 0.4,
            alpha: 0.5,
            phase: (index * 0.21) % 1
          })),
          animation: 'pulse',
          speed: 0.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: seeds.map(seed => sprite(seed, 2.2, 'ink', 1, 0)),
          pulse: 0.3
        }
      ];
    }
  },

  // Marching-squares contour lines at chosen levels; the pointer picks a level.
  isolines: {
    parameters: [{id: 'levels', label: 'Levels', min: 3, max: 16, step: 1, default: 8}],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const surface = createSmoothField(
        context.fork('surface'),
        7,
        1.3,
        SURFACE_COLUMNS,
        SURFACE_ROWS
      );
      const count = Math.round(p.levels);
      const lines: PathItem[] = [];
      for (let level = 0; level < count; level++) {
        const role = getClassRole(level, count);
        for (const contour of traceContours(
          surface,
          SURFACE_COLUMNS,
          SURFACE_ROWS,
          (level + 1) / (count + 1)
        )) {
          lines.push({
            points: smoothPath(contour.points, 1),
            role,
            width: 0.6,
            alpha: 0.9,
            phase: level / count
          });
        }
      }
      const faint = expandCoarse(surface, true, SURFACE_COLUMNS, SURFACE_ROWS).map(
        value => value * 0.4
      );
      const layers: PreviewLayer[] = [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: faint,
          ramp: 'ice',
          style: 'ramp',
          strength: 0.1,
          speed: 0.4
        },
        {
          type: 'strokes',
          items: lines,
          animation: 'draw',
          speed: 0.15,
          blend: 'add'
        }
      ];
      if (pointer) {
        const level = sampleField(
          surface,
          SURFACE_COLUMNS,
          SURFACE_ROWS,
          pointer[0] / SCENE_WIDTH,
          pointer[1] / SCENE_HEIGHT
        );
        layers.push(
          {
            type: 'strokes',
            items: traceContours(surface, SURFACE_COLUMNS, SURFACE_ROWS, level).map(contour => ({
              points: smoothPath(contour.points, 1),
              role: 'hot' as Role,
              width: 1.4,
              alpha: 1
            })),
            animation: 'flow',
            speed: 0.4,
            blend: 'add'
          },
          {
            type: 'sprites',
            items: [sprite(pointer, 2, 'hot', 1, 0)],
            pulse: 0.3
          }
        );
      }
      return layers;
    }
  },

  // Filled bands between breaks, one class colour each.
  isobands: {
    parameters: [{id: 'bands', label: 'Bands', min: 3, max: 10, step: 1, default: 6}],
    pointer: 'none',
    build: context => {
      const count = Math.round(context.p.bands);
      const surface = createSmoothField(
        context.fork('surface'),
        7,
        1.3,
        SURFACE_COLUMNS,
        SURFACE_ROWS
      );
      const values = expandCoarse(surface, true, SURFACE_COLUMNS, SURFACE_ROWS).map(
        value => (Math.min(count - 1, Math.floor(value * count)) + 0.5) / count
      );
      const breaks: PathItem[] = [];
      for (let level = 1; level < count; level++) {
        for (const contour of traceContours(
          surface,
          SURFACE_COLUMNS,
          SURFACE_ROWS,
          level / count
        )) {
          breaks.push({
            points: smoothPath(contour.points, 1),
            role: 'ink',
            width: 0.35,
            alpha: 0.5
          });
        }
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'spectral',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.12,
          speed: 0.5
        },
        {type: 'strokes', items: breaks, animation: 'static', blend: 'add'}
      ];
    }
  },

  // Closed rings per level: shells filled, holes outlined apart.
  'isoband-rings': {
    parameters: [{id: 'levels', label: 'Rings', min: 2, max: 8, step: 1, default: 4}],
    pointer: 'none',
    build: context => {
      const count = Math.round(context.p.levels);
      const surface = createSmoothField(
        context.fork('surface'),
        7,
        1.3,
        SURFACE_COLUMNS,
        SURFACE_ROWS
      );
      const fills: PolygonItem[] = [];
      const shells: PathItem[] = [];
      const holes: PathItem[] = [];
      const starts: SpriteItem[] = [];
      const clampRing = (ring: Point[]): Point[] =>
        ring.map(([x, y]): Point => [clamp(x, 0, SCENE_WIDTH), clamp(y, 0, SCENE_HEIGHT)]);
      for (let level = 0; level < count; level++) {
        const role = getClassRole(level, count);
        const rings = traceContours(
          surface,
          SURFACE_COLUMNS,
          SURFACE_ROWS,
          (level + 1) / (count + 1),
          true
        )
          .filter(contour => contour.closed)
          .map(contour => clampRing(contour.points));
        rings.forEach((ring, index) => {
          const xs = ring.map(point => point[0]);
          const ys = ring.map(point => point[1]);
          if (Math.max(...xs) - Math.min(...xs) > 150 && Math.max(...ys) - Math.min(...ys) > 90)
            return;
          let depth = 0;
          rings.forEach((other, otherIndex) => {
            if (otherIndex !== index && isPointInRing(ring[0], other)) depth++;
          });
          const isHole = depth % 2 === 1;
          if (isHole) {
            holes.push({points: ring, role: 'ink', width: 0.6, alpha: 0.9});
          } else {
            shells.push({
              points: ring,
              role,
              width: 0.7,
              alpha: 0.95,
              phase: level / count
            });
            fills.push({
              ring: ring.slice(0, -1),
              role,
              alpha: 0.13,
              phase: level / count
            });
          }
          starts.push(sprite(ring[0], 1.1, isHole ? 'ink' : role, 0.95, level / count));
        });
      }
      return [
        {type: 'fills', items: fills, animation: 'pulse', blend: 'add'},
        {
          type: 'strokes',
          items: shells,
          animation: 'draw',
          speed: 0.15,
          blend: 'add'
        },
        {
          type: 'strokes',
          items: holes,
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        },
        {type: 'sprites', items: starts, pulse: 0.3}
      ];
    }
  },

  // Evenly spaced streamlines through a vortex field.
  streamlines: {
    parameters: [
      {
        id: 'separation',
        label: 'Separation',
        min: 4,
        max: 16,
        step: 0.5,
        default: 6,
        unit: 'px'
      }
    ],
    pointer: 'hover',
    build: context => {
      const vortices = createVortices(context.fork('vortices'), 4, context.pointer);
      const lines = createEvenStreamlines(vortices, context.p.separation);
      return [
        {
          type: 'strokes',
          items: lines.map((points, index) => {
            const middle = points[points.length >> 1];
            const [vx, vy] = evaluateVectorField(vortices, middle[0], middle[1]);
            return {
              points,
              role: (Math.hypot(vx, vy) > 0.5 ? 'warm' : 'primary') as Role,
              width: 0.8,
              alpha: 0.9,
              phase: (index * 0.17) % 1
            };
          }),
          animation: 'flow',
          speed: 0.25,
          trail: 1.3,
          blend: 'add'
        },
        {
          type: 'sprites',
          items: vortices.map(vortex => sprite([vortex.x, vortex.y], 1.2, 'dim', 0.6, 0))
        }
      ];
    }
  },

  // Noise smeared along the field's streamlines.
  'line-integral-convolution': {
    parameters: [
      {
        id: 'length',
        label: 'Kernel length',
        min: 2,
        max: 24,
        step: 1,
        default: 14,
        unit: 'steps'
      }
    ],
    pointer: 'hover',
    build: context => {
      const columns = 64;
      const rows = 40;
      const vortices = createVortices(context.fork('vortices'), 4, context.pointer);
      const noiseRandom = context.fork('noise');
      const noise = new Float32Array(columns * rows);
      for (let index = 0; index < noise.length; index++) noise[index] = noiseRandom.next();
      const half = Math.round(context.p.length);
      const values = new Float32Array(columns * rows);
      const gain = 0.25 / (0.2887 / Math.sqrt(half * 2 + 1));
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const start: Point = [
            ((column + 0.5) / columns) * SCENE_WIDTH,
            ((row + 0.5) / rows) * SCENE_HEIGHT
          ];
          let sum = noise[row * columns + column];
          let samples = 1;
          for (const direction of [1, -1]) {
            let [x, y] = start;
            for (let step = 0; step < half; step++) {
              const [vx, vy] = evaluateVectorField(vortices, x, y);
              const length = Math.hypot(vx, vy) || 1;
              x += (direction * vx * 1.6) / length;
              y += (direction * vy * 1.6) / length;
              const sampleColumn = Math.floor((x / SCENE_WIDTH) * columns);
              const sampleRow = Math.floor((y / SCENE_HEIGHT) * rows);
              if (
                sampleColumn < 0 ||
                sampleRow < 0 ||
                sampleColumn >= columns ||
                sampleRow >= rows
              ) {
                break;
              }
              sum += noise[sampleRow * columns + sampleColumn];
              samples++;
            }
          }
          values[row * columns + column] = clamp(0.5 + (sum / samples - 0.5) * gain, 0, 1);
        }
      }
      const guides = createEvenStreamlines(vortices, 24);
      return [
        {
          type: 'field',
          columns,
          rows,
          values,
          ramp: 'ice',
          style: 'ramp',
          strength: 0.1,
          speed: 0.6
        },
        {
          type: 'strokes',
          items: guides.map((points, index) => ({
            points,
            role: 'ink' as Role,
            width: 0.5,
            alpha: 0.45,
            phase: (index * 0.2) % 1
          })),
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        }
      ];
    }
  },

  // Particles stepped through the field with replayable randomness, trailing their last frames.
  'particle-advection': {
    parameters: [
      {id: 'frames', label: 'Frames', min: 1, max: 40, step: 1, default: 14},
      {
        id: 'particles',
        label: 'Particles',
        min: 30,
        max: 260,
        step: 10,
        default: 140
      }
    ],
    pointer: 'hover',
    build: context => {
      const {p} = context;
      const vortices = createVortices(context.fork('vortices'), 4, context.pointer);
      const starts = createScatterPoints(context.fork('particles'), Math.round(p.particles), 4);
      const frames = Math.round(p.frames);
      const trailLength = Math.min(frames, 9);
      const trails: PathItem[] = [];
      const heads: SpriteItem[] = [];
      starts.forEach((start, index) => {
        let [x, y] = start;
        let trail: Point[] = [[x, y]];
        for (let frame = 0; frame < frames; frame++) {
          const [vx, vy] = evaluateVectorField(vortices, x, y);
          x += vx * 3.2;
          y += vy * 3.2;
          if (x < 0 || y < 0 || x > SCENE_WIDTH || y > SCENE_HEIGHT) {
            const respawn = createRandom(index * 7919 + frame * 104729 + 17);
            x = respawn.range(4, SCENE_WIDTH - 4);
            y = respawn.range(4, SCENE_HEIGHT - 4);
            trail = [];
          }
          trail.push([x, y]);
          if (trail.length > trailLength + 1) trail.shift();
        }
        if (trail.length > 1) {
          trails.push({
            points: trail,
            role: 'cool',
            width: 0.6,
            alpha: 0.85,
            phase: (index * 0.13) % 1
          });
        }
        heads.push(sprite([x, y], 0.9, 'warm', 0.95, (index % 9) / 9));
      });
      const arrows: PathItem[] = [];
      for (let y = 8; y < SCENE_HEIGHT; y += 12) {
        for (let x = 8; x < SCENE_WIDTH; x += 12) {
          const [vx, vy] = evaluateVectorField(vortices, x, y);
          const length = Math.hypot(vx, vy) || 1;
          arrows.push({
            points: [
              [x, y],
              [x + (vx / length) * 4, y + (vy / length) * 4]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.35
          });
        }
      }
      return [
        {type: 'strokes', items: arrows, animation: 'static'},
        {
          type: 'strokes',
          items: trails,
          animation: 'flow',
          speed: 0.5,
          trail: 0.7,
          blend: 'add'
        },
        {type: 'sprites', items: heads, pulse: 0.3}
      ];
    }
  }
};
