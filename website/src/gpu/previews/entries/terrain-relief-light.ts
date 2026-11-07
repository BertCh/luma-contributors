// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/terrain-relief-light.ts, keyed by entry id.

import {
  clamp,
  createSmoothField,
  FIELD_COLUMNS,
  FIELD_ROWS,
  mixNumbers,
  normalizeValues
} from '../data.ts';
import {createFillLayer, createSpriteLayer, createStrokeLayer} from '../layers.ts';
import {
  clampPoint,
  createGridLayer,
  getLightAngle,
  smoothPath,
  toPaths,
  toSprites,
  type BuildContext,
  type EntryPreview
} from '../helpers.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type FieldLayer,
  type FieldStyle,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewLayer,
  type RampName,
  type Role,
  type SpriteItem
} from '../types.ts';

const COLUMNS = FIELD_COLUMNS;
const ROWS = FIELD_ROWS;
const FINE_CELL = SCENE_WIDTH / COLUMNS;
const COARSE_COLUMNS = 64;
const COARSE_ROWS = 40;
const COARSE_CELL = SCENE_WIDTH / COARSE_COLUMNS;
/** Relief of the whole synthetic tile, in coarse cells, so sight lines and shadows have a slope. */
const VERTICAL_SCALE = 14;
const DIRECTION_COUNT = 12;
const HORIZON_DISTANCES = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20];
const CENTER: Point = [SCENE_WIDTH / 2, SCENE_HEIGHT / 2];

// ---------------------------------------------------------------------------------------------
// Synthetic terrain shared by the pictures: a smooth field plus a few mountains.
// ---------------------------------------------------------------------------------------------

type Summit = {
  x: number;
  y: number;
  height: number;
  prominence: number;
  keyX: number;
  keyY: number;
};

type Terrain = {
  heights: Float32Array;
  coarse: Float32Array;
  rough: Float32Array;
  summits: Summit[];
  memo: Map<string, unknown>;
};

const TERRAIN_CACHE = new Map<number, Terrain>();

function remember<T>(terrain: Terrain, name: string, compute: () => T): T {
  if (!terrain.memo.has(name)) terrain.memo.set(name, compute());
  return terrain.memo.get(name) as T;
}

function downsample(values: Float32Array): Float32Array {
  const coarse = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
  for (let row = 0; row < COARSE_ROWS; row++) {
    for (let column = 0; column < COARSE_COLUMNS; column++) {
      coarse[row * COARSE_COLUMNS + column] = values[row * 2 * COLUMNS + column * 2];
    }
  }
  return coarse;
}

function getTerrain(context: BuildContext): Terrain {
  const random = context.fork('terrain');
  const key = random.next();
  const cached = TERRAIN_CACHE.get(key);
  if (cached) return cached;
  const smooth = createSmoothField(random, 7, 1.7);
  const mountains = Array.from({length: 4}, () => ({
    u: random.range(0.12, 0.88),
    v: random.range(0.15, 0.85),
    sigma: random.range(0.06, 0.1),
    amplitude: random.range(0.3, 0.7)
  }));
  const raw = new Float32Array(COLUMNS * ROWS);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const u = (column + 0.5) / COLUMNS;
      const v = (row + 0.5) / ROWS;
      let sum = smooth[row * COLUMNS + column] * 0.9;
      for (const mountain of mountains) {
        const du = u - mountain.u;
        const dv = (v - mountain.v) * 0.625;
        sum += mountain.amplitude * Math.exp(-(du * du + dv * dv) / (2 * mountain.sigma ** 2));
      }
      raw[row * COLUMNS + column] = sum;
    }
  }
  const heights = normalizeValues(raw);
  // Rough ground: high frequency detail that is strong in some places and absent in others.
  const detail = createSmoothField(random, 8, 7);
  const mask = createSmoothField(random, 5, 1.2);
  const rough = heights.map((height, index) => {
    const amount = clamp((mask[index] - 0.35) * 2.2, 0, 1);
    return height + (detail[index] - 0.5) * 0.1 * amount;
  });
  const terrain: Terrain = {
    heights,
    coarse: downsample(heights),
    rough,
    summits: [],
    memo: new Map()
  };
  terrain.summits = findSummits(terrain.coarse);
  TERRAIN_CACHE.set(key, terrain);
  return terrain;
}

/** Local maxima of the coarse grid, highest first, each with a line-based prominence estimate. */
function findSummits(coarse: Float32Array): Summit[] {
  const radius = 2;
  const peaks: {column: number; row: number; height: number}[] = [];
  for (let row = radius; row < COARSE_ROWS - radius; row++) {
    for (let column = radius; column < COARSE_COLUMNS - radius; column++) {
      const height = coarse[row * COARSE_COLUMNS + column];
      if (height < 0.3) continue;
      let isMaximum = true;
      for (let dy = -radius; dy <= radius && isMaximum; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (coarse[(row + dy) * COARSE_COLUMNS + column + dx] > height) {
            isMaximum = false;
            break;
          }
        }
      }
      if (isMaximum) peaks.push({column, row, height});
    }
  }
  peaks.sort((first, second) => second.height - first.height);
  const kept = peaks.slice(0, 12);
  return kept.map((peak, index) => {
    const higher = kept.slice(0, index);
    let nearest = -1;
    let nearestDistance = Infinity;
    higher.forEach((other, otherIndex) => {
      const distance = Math.hypot(other.column - peak.column, other.row - peak.row);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = otherIndex;
      }
    });
    let lowest = 0;
    let keyColumn = peak.column;
    let keyRow = peak.row;
    if (nearest >= 0) {
      lowest = peak.height;
      for (let step = 1; step < 40; step++) {
        const amount = step / 40;
        const column = Math.round(mixNumbers(peak.column, kept[nearest].column, amount));
        const row = Math.round(mixNumbers(peak.row, kept[nearest].row, amount));
        const height = coarse[row * COARSE_COLUMNS + column];
        if (height < lowest) {
          lowest = height;
          keyColumn = column;
          keyRow = row;
        }
      }
    }
    return {
      x: (peak.column + 0.5) * COARSE_CELL,
      y: (peak.row + 0.5) * COARSE_CELL,
      height: peak.height,
      prominence: peak.height - lowest,
      keyX: (keyColumn + 0.5) * COARSE_CELL,
      keyY: (keyRow + 0.5) * COARSE_CELL
    };
  });
}

/** Box blur with edge clamping, linear in the cell count. */
function blurField(
  values: Float32Array,
  columns: number,
  rows: number,
  radius: number
): Float32Array {
  const size = Math.round(radius);
  if (size <= 0) return values;
  const width = 2 * size + 1;
  const temporary = new Float32Array(values.length);
  const output = new Float32Array(values.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * columns;
    let sum = 0;
    for (let shift = -size; shift <= size; shift++)
      sum += values[offset + clamp(shift, 0, columns - 1)];
    for (let column = 0; column < columns; column++) {
      temporary[offset + column] = sum / width;
      sum +=
        values[offset + Math.min(column + size + 1, columns - 1)] -
        values[offset + Math.max(column - size, 0)];
    }
  }
  for (let column = 0; column < columns; column++) {
    let sum = 0;
    for (let shift = -size; shift <= size; shift++)
      sum += temporary[clamp(shift, 0, rows - 1) * columns + column];
    for (let row = 0; row < rows; row++) {
      output[row * columns + column] = sum / width;
      sum +=
        temporary[Math.min(row + size + 1, rows - 1) * columns + column] -
        temporary[Math.max(row - size, 0) * columns + column];
    }
  }
  return output;
}

function sampleAt(
  values: Float32Array,
  columns: number,
  rows: number,
  column: number,
  row: number
) {
  return values[
    clamp(Math.round(row), 0, rows - 1) * columns + clamp(Math.round(column), 0, columns - 1)
  ];
}

/** Central difference gradient per cell, in value units per cell. */
function getGradient(
  values: Float32Array,
  columns: number,
  rows: number,
  column: number,
  row: number
): Point {
  return [
    (sampleAt(values, columns, rows, column + 1, row) -
      sampleAt(values, columns, rows, column - 1, row)) /
      2,
    (sampleAt(values, columns, rows, column, row + 1) -
      sampleAt(values, columns, rows, column, row - 1)) /
      2
  ];
}

function getPointerCell(point: Point, columns: number, rows: number): Point {
  return [
    clamp(Math.floor((point[0] / SCENE_WIDTH) * columns), 0, columns - 1),
    clamp(Math.floor((point[1] / SCENE_HEIGHT) * rows), 0, rows - 1)
  ];
}

function sampleHeight(terrain: Terrain, point: Point): number {
  const [column, row] = getPointerCell(point, COLUMNS, ROWS);
  return terrain.heights[row * COLUMNS + column];
}

// ---------------------------------------------------------------------------------------------
// Small drawing helpers.
// ---------------------------------------------------------------------------------------------

function createField(
  values: Float32Array,
  ramp: RampName,
  style: FieldStyle,
  extra: Partial<FieldLayer> = {},
  columns = COLUMNS,
  rows = ROWS
): FieldLayer {
  return {type: 'field', columns, rows, values, ramp, style, ...extra};
}

function createRing(center: Point, radius: number, count = 40, radiusY = radius): Point[] {
  return Array.from({length: count + 1}, (_, index) => {
    const angle = (index / count) * Math.PI * 2;
    return [center[0] + Math.cos(angle) * radius, center[1] + Math.sin(angle) * radiusY] as Point;
  });
}

function createWedge(
  center: Point,
  inner: number,
  outer: number,
  start: number,
  end: number
): Point[] {
  const ring: Point[] = [];
  const steps = 6;
  for (let step = 0; step <= steps; step++) {
    const angle = mixNumbers(start, end, step / steps);
    ring.push([center[0] + Math.cos(angle) * outer, center[1] + Math.sin(angle) * outer]);
  }
  for (let step = steps; step >= 0; step--) {
    const angle = mixNumbers(start, end, step / steps);
    ring.push([center[0] + Math.cos(angle) * inner, center[1] + Math.sin(angle) * inner]);
  }
  return ring;
}

/** Compass azimuth in degrees (0 north, clockwise) to a position on a ring around the scene. */
function getSunPosition(azimuth: number, scale = 1): Point {
  const angle = (azimuth * Math.PI) / 180;
  return [CENTER[0] + Math.sin(angle) * 66 * scale, CENTER[1] - Math.cos(angle) * 40 * scale];
}

function getPointerAzimuth(pointer: Point): number {
  const degrees = (Math.atan2(pointer[0] - CENTER[0], -(pointer[1] - CENTER[1])) * 180) / Math.PI;
  return (degrees + 360) % 360;
}

/** Parallel light rays flowing from a sun towards the middle of the scene. */
function createSunRays(sun: Point, role: Role, count = 5): PreviewLayer {
  const dx = CENTER[0] - sun[0];
  const dy = CENTER[1] - sun[1];
  const length = Math.hypot(dx, dy) || 1;
  const normal: Point = [-dy / length, dx / length];
  const items: PathItem[] = [];
  for (let index = 0; index < count; index++) {
    const offset = (index - (count - 1) / 2) * 5;
    items.push({
      points: [
        [sun[0] + normal[0] * offset, sun[1] + normal[1] * offset],
        [CENTER[0] + normal[0] * offset, CENTER[1] + normal[1] * offset]
      ],
      role,
      width: 0.7,
      alpha: 0.8,
      phase: index / count
    });
  }
  return {type: 'strokes', items, animation: 'flow', speed: 0.35, blend: 'add'};
}

function createSunSprite(position: Point, radius = 4): PreviewLayer {
  return createSpriteLayer(
    [
      {x: position[0], y: position[1], radius: radius * 2.2, role: 'warm', alpha: 0.25},
      {x: position[0], y: position[1], radius, role: 'warm', alpha: 1}
    ],
    {pulse: 0.2}
  );
}

function createSegmentPaths(
  segments: [Point, Point][],
  role: Role,
  width: number,
  alpha = 0.9
): PathItem[] {
  return segments.map((points, index) => ({points, role, width, alpha, phase: (index % 9) / 9}));
}

/** Marching squares on a grid, one level: segments in scene units. */
function traceContour(
  values: Float32Array,
  columns: number,
  rows: number,
  level: number
): [Point, Point][] {
  const segments: [Point, Point][] = [];
  const cellWidth = SCENE_WIDTH / columns;
  const cellHeight = SCENE_HEIGHT / rows;
  const cross = (first: number, second: number) => (level - first) / (second - first || 1);
  for (let row = 0; row < rows - 1; row++) {
    for (let column = 0; column < columns - 1; column++) {
      const topLeft = values[row * columns + column];
      const topRight = values[row * columns + column + 1];
      const bottomRight = values[(row + 1) * columns + column + 1];
      const bottomLeft = values[(row + 1) * columns + column];
      const x = (column + 0.5) * cellWidth;
      const y = (row + 0.5) * cellHeight;
      const points: Point[] = [];
      if (topLeft < level !== topRight < level) {
        points.push([x + cross(topLeft, topRight) * cellWidth, y]);
      }
      if (topRight < level !== bottomRight < level) {
        points.push([x + cellWidth, y + cross(topRight, bottomRight) * cellHeight]);
      }
      if (bottomLeft < level !== bottomRight < level) {
        points.push([x + cross(bottomLeft, bottomRight) * cellWidth, y + cellHeight]);
      }
      if (topLeft < level !== bottomLeft < level) {
        points.push([x, y + cross(topLeft, bottomLeft) * cellHeight]);
      }
      if (points.length === 2) segments.push([points[0], points[1]]);
      else if (points.length === 4) {
        segments.push([points[0], points[1]], [points[2], points[3]]);
      }
    }
  }
  return segments;
}

const HILLSHADE_PARAMETERS = [
  {id: 'azimuth', label: 'Sun azimuth', min: 0, max: 360, step: 5, default: 315, unit: '°'},
  {id: 'exaggeration', label: 'Exaggeration', min: 1, max: 14, step: 0.5, default: 5, unit: '×'}
];

// ---------------------------------------------------------------------------------------------
// Coarse-grid analyses: horizon map, shadows, viewsheds.
// ---------------------------------------------------------------------------------------------

/** Tangent of the horizon angle per coarse cell and compass direction (every 30 degrees). */
function getHorizonMap(terrain: Terrain): Float32Array {
  return remember(terrain, 'horizon', () => {
    const map = new Float32Array(COARSE_COLUMNS * COARSE_ROWS * DIRECTION_COUNT);
    const {coarse} = terrain;
    for (let direction = 0; direction < DIRECTION_COUNT; direction++) {
      const angle = (direction / DIRECTION_COUNT) * Math.PI * 2;
      const dx = Math.sin(angle);
      const dy = -Math.cos(angle);
      for (let row = 0; row < COARSE_ROWS; row++) {
        for (let column = 0; column < COARSE_COLUMNS; column++) {
          const base = coarse[row * COARSE_COLUMNS + column];
          let best = 0;
          for (const distance of HORIZON_DISTANCES) {
            const sampleColumn = Math.round(column + dx * distance);
            const sampleRow = Math.round(row + dy * distance);
            if (
              sampleColumn < 0 ||
              sampleRow < 0 ||
              sampleColumn >= COARSE_COLUMNS ||
              sampleRow >= COARSE_ROWS
            ) {
              break;
            }
            const tangent =
              ((coarse[sampleRow * COARSE_COLUMNS + sampleColumn] - base) * VERTICAL_SCALE) /
              distance;
            if (tangent > best) best = tangent;
          }
          map[(row * COARSE_COLUMNS + column) * DIRECTION_COUNT + direction] = best;
        }
      }
    }
    return map;
  });
}

function getHorizonTangent(map: Float32Array, cell: number, azimuthDegrees: number): number {
  const position = (((azimuthDegrees % 360) + 360) % 360) / (360 / DIRECTION_COUNT);
  const first = Math.floor(position) % DIRECTION_COUNT;
  const second = (first + 1) % DIRECTION_COUNT;
  const amount = position - Math.floor(position);
  return mixNumbers(
    map[cell * DIRECTION_COUNT + first],
    map[cell * DIRECTION_COUNT + second],
    amount
  );
}

function getSkyViewFactor(terrain: Terrain, directions: number): Float32Array {
  return remember(terrain, `sky-view-${directions}`, () => {
    const map = getHorizonMap(terrain);
    const stride = DIRECTION_COUNT / directions;
    const values = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
    for (let cell = 0; cell < values.length; cell++) {
      let sum = 0;
      for (let direction = 0; direction < DIRECTION_COUNT; direction += stride) {
        const tangent = map[cell * DIRECTION_COUNT + direction];
        sum += tangent / Math.sqrt(1 + tangent * tangent);
      }
      values[cell] = 1 - sum / directions;
    }
    return normalizeValues(values);
  });
}

/** Direct shadow test towards one sun; 1 lit, 0 in shadow. */
function computeShadow(
  terrain: Terrain,
  azimuthDegrees: number,
  altitudeDegrees: number
): Float32Array {
  const {coarse} = terrain;
  const angle = (azimuthDegrees * Math.PI) / 180;
  const dx = Math.sin(angle);
  const dy = -Math.cos(angle);
  const tangent = Math.tan((altitudeDegrees * Math.PI) / 180);
  const lit = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
  for (let row = 0; row < COARSE_ROWS; row++) {
    for (let column = 0; column < COARSE_COLUMNS; column++) {
      const base = coarse[row * COARSE_COLUMNS + column] * VERTICAL_SCALE;
      let isLit = 1;
      for (let step = 1; step < 30; step++) {
        const sampleColumn = Math.round(column + dx * step);
        const sampleRow = Math.round(row + dy * step);
        if (
          sampleColumn < 0 ||
          sampleRow < 0 ||
          sampleColumn >= COARSE_COLUMNS ||
          sampleRow >= COARSE_ROWS
        ) {
          break;
        }
        if (
          coarse[sampleRow * COARSE_COLUMNS + sampleColumn] * VERTICAL_SCALE >
          base + step * tangent
        ) {
          isLit = 0;
          break;
        }
      }
      lit[row * COARSE_COLUMNS + column] = isLit;
    }
  }
  return lit;
}

/** 1 where the target cell is visible from the observer cell, 0 where terrain hides it. */
function computeViewshed(
  terrain: Terrain,
  observerColumn: number,
  observerRow: number,
  eyeHeight: number,
  range: number
): Uint8Array {
  const {coarse} = terrain;
  const eye = coarse[observerRow * COARSE_COLUMNS + observerColumn] * VERTICAL_SCALE + eyeHeight;
  const visible = new Uint8Array(COARSE_COLUMNS * COARSE_ROWS);
  for (let row = 0; row < COARSE_ROWS; row++) {
    for (let column = 0; column < COARSE_COLUMNS; column++) {
      const deltaColumn = column - observerColumn;
      const deltaRow = row - observerRow;
      const distance = Math.hypot(deltaColumn, deltaRow);
      if (distance === 0 || distance > range) {
        visible[row * COARSE_COLUMNS + column] = distance === 0 ? 1 : 0;
        continue;
      }
      const targetTangent =
        (coarse[row * COARSE_COLUMNS + column] * VERTICAL_SCALE - eye) / distance;
      const steps = Math.ceil(distance);
      let isVisible = 1;
      for (let step = 1; step < steps; step++) {
        const amount = step / steps;
        const sampleColumn = Math.round(observerColumn + deltaColumn * amount);
        const sampleRow = Math.round(observerRow + deltaRow * amount);
        const tangent =
          (coarse[sampleRow * COARSE_COLUMNS + sampleColumn] * VERTICAL_SCALE - eye) /
          (distance * amount);
        if (tangent > targetTangent) {
          isVisible = 0;
          break;
        }
      }
      visible[row * COARSE_COLUMNS + column] = isVisible;
    }
  }
  return visible;
}

/** Elevation angle tangent along one azimuth from an observer cell. */
function getObserverHorizon(
  terrain: Terrain,
  column: number,
  row: number,
  azimuthDegrees: number
): number {
  const {coarse} = terrain;
  const angle = (azimuthDegrees * Math.PI) / 180;
  const base = coarse[row * COARSE_COLUMNS + column] * VERTICAL_SCALE + 0.5;
  let best = 0;
  for (let step = 1; step < 26; step++) {
    const sampleColumn = Math.round(column + Math.sin(angle) * step);
    const sampleRow = Math.round(row - Math.cos(angle) * step);
    if (
      sampleColumn < 0 ||
      sampleRow < 0 ||
      sampleColumn >= COARSE_COLUMNS ||
      sampleRow >= COARSE_ROWS
    ) {
      break;
    }
    const tangent =
      (coarse[sampleRow * COARSE_COLUMNS + sampleColumn] * VERTICAL_SCALE - base) / step;
    if (tangent > best) best = tangent;
  }
  return best;
}

function toCoarseCell(point: Point): [number, number] {
  return [
    clamp(Math.floor(point[0] / COARSE_CELL), 0, COARSE_COLUMNS - 1),
    clamp(Math.floor(point[1] / COARSE_CELL), 0, COARSE_ROWS - 1)
  ];
}

function createSunArc(): PathItem {
  const points: Point[] = [];
  for (let step = 0; step <= 24; step++) {
    const amount = step / 24;
    points.push([14 + 132 * amount, 22 - 16 * Math.sin(Math.PI * amount)]);
  }
  return {points, role: 'dim', width: 0.5, alpha: 0.8};
}

/** Shades a height grid by weighted light directions (compass degrees), 0..1. */
function shadeHeights(
  values: Float32Array,
  exaggeration: number,
  azimuths: number[]
): Float32Array {
  const lights = azimuths.map(azimuth => {
    const angle = (azimuth * Math.PI) / 180;
    return [Math.sin(angle), -Math.cos(angle)];
  });
  const shaded = new Float32Array(values.length);
  const scale = exaggeration * 14;
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const [gx, gy] = getGradient(values, COLUMNS, ROWS, column, row);
      const nx = -gx * scale;
      const ny = -gy * scale;
      const normalLength = Math.sqrt(nx * nx + ny * ny + 1);
      let sum = 0;
      for (const light of lights) {
        const dot = (nx * light[0] + ny * light[1] + 0.7) / (normalLength * 1.22);
        sum += Math.max(0, dot);
      }
      shaded[row * COLUMNS + column] = sum / lights.length;
    }
  }
  return shaded;
}

// ---------------------------------------------------------------------------------------------
// Landform classes.
// ---------------------------------------------------------------------------------------------

function getCoarseSlope(terrain: Terrain): Float32Array {
  return remember(terrain, 'coarse-slope', () => {
    const slope = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
    for (let row = 0; row < COARSE_ROWS; row++) {
      for (let column = 0; column < COARSE_COLUMNS; column++) {
        const [gx, gy] = getGradient(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, column, row);
        slope[row * COARSE_COLUMNS + column] = Math.hypot(gx, gy) * 12;
      }
    }
    return slope;
  });
}

function getStandardDeviation(values: Float32Array): number {
  let sum = 0;
  for (const value of values) sum += value * value;
  return Math.sqrt(sum / values.length) || 1;
}

// ---------------------------------------------------------------------------------------------
// Entry previews.
// ---------------------------------------------------------------------------------------------

export const TERRAIN_RELIEF_LIGHT_PREVIEWS: Record<string, EntryPreview> = {
  // Horn slope, aspect and hillshade: a movable sun over lit relief, plus the aspect compass.
  'slope-aspect-hillshade': {
    parameters: HILLSHADE_PARAMETERS,
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const azimuth = pointer ? getPointerAzimuth(pointer) : p.azimuth;
      const sun = pointer ?? getSunPosition(azimuth);
      const wheelCenter: Point = [142, 80];
      const wheelRoles: Role[] = [
        'cool',
        'primary',
        'violet',
        'hot',
        'warm',
        'hot',
        'violet',
        'primary'
      ];
      const facing = Math.round(((azimuth + 180) % 360) / 45) % 8;
      const wedges: PolygonItem[] = wheelRoles.map((role, index) => {
        const center = ((index * 45 - 90) * Math.PI) / 180;
        return {
          ring: createWedge(wheelCenter, 5, 12, center - 0.38, center + 0.38),
          role,
          alpha: index === facing ? 0.95 : 0.4
        };
      });
      return [
        createField(terrain.heights, 'terrain', 'hillshade', {
          strength: p.exaggeration,
          light: getLightAngle(azimuth)
        }),
        createFillLayer(wedges, {blend: 'add'}),
        createStrokeLayer(
          [{points: createRing(wheelCenter, 13, 30), role: 'dim', width: 0.4, alpha: 0.8}],
          {animation: 'static'}
        ),
        createSunRays(sun, 'warm'),
        createSunSprite(sun)
      ];
    }
  },

  // Curvature: convex ground hot, concave ground cool, with a profile along a section line.
  'terrain-curvature': {
    parameters: [
      {id: 'scale', label: 'Kernel size', min: 1, max: 8, step: 1, default: 3, unit: 'cells'},
      {id: 'gain', label: 'Gain', min: 1, max: 12, step: 0.5, default: 4}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const smooth = blurField(terrain.heights, COLUMNS, ROWS, p.scale + 2);
      const curvature = terrain.rough.map((height, index) => {
        const value = (height + terrain.heights[index] - 2 * smooth[index]) * p.gain * 1.5;
        return 0.5 + 0.5 * Math.tanh(value * 2);
      });
      const row = pointer ? clamp(Math.round(pointer[1] / FINE_CELL), 2, ROWS - 3) : 36;
      const lineY = (row + 0.5) * FINE_CELL;
      const profile: Point[] = [];
      const convexRuns: Point[][] = [];
      const concaveRuns: Point[][] = [];
      let run: Point[] = [];
      let runConvex = true;
      for (let column = 0; column < COLUMNS; column += 2) {
        const index = row * COLUMNS + column;
        const point: Point = [(column + 0.5) * FINE_CELL, 97 - terrain.heights[index] * 24];
        profile.push(point);
        const isConvex = curvature[index] >= 0.5;
        if (run.length > 0 && isConvex !== runConvex) {
          (runConvex ? convexRuns : concaveRuns).push(run);
          run = [run[run.length - 1]];
        }
        runConvex = isConvex;
        run.push(point);
      }
      (runConvex ? convexRuns : concaveRuns).push(run);
      return [
        createField(curvature, 'spectral', 'ramp', {strength: 0.06, speed: 0.4}),
        createStrokeLayer([{points: smoothPath(profile, 0), role: 'ink', width: 2.2, alpha: 0.3}], {
          animation: 'static'
        }),
        createStrokeLayer(
          [
            {
              points: [
                [0, lineY],
                [SCENE_WIDTH, lineY]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.9
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(
          toPaths(
            convexRuns.filter(path => path.length > 1),
            'hot',
            1.4
          ),
          {animation: 'flow', speed: 0.3, blend: 'add'}
        ),
        createStrokeLayer(
          toPaths(
            concaveRuns.filter(path => path.length > 1),
            'cool',
            1.4
          ),
          {animation: 'flow', speed: 0.3, blend: 'add'}
        )
      ];
    }
  },

  // TPI, TRI, roughness: a 3x3 window magnified at the pointer over a roughness map.
  'terrain-ruggedness': {
    parameters: [{id: 'gain', label: 'Contrast', min: 1, max: 8, step: 0.5, default: 3.5}],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const roughness = remember(terrain, 'tri', () => {
        const values = new Float32Array(COLUMNS * ROWS);
        for (let row = 0; row < ROWS; row++) {
          for (let column = 0; column < COLUMNS; column++) {
            const center = terrain.rough[row * COLUMNS + column];
            let sum = 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                sum += Math.abs(
                  sampleAt(terrain.rough, COLUMNS, ROWS, column + dx, row + dy) - center
                );
              }
            }
            values[row * COLUMNS + column] = sum / 8;
          }
        }
        return values;
      });
      const scaled = roughness.map(value => clamp(value * 26 * (p.gain / 3.5), 0, 1));
      const probe = pointer ?? [58, 40];
      const [column, row] = getPointerCell(probe, COLUMNS, ROWS);
      const center = terrain.rough[row * COLUMNS + column];
      const cellSize = 6;
      const windowOrigin: Point = [
        clamp(probe[0], 12, SCENE_WIDTH - 12),
        clamp(probe[1], 12, SCENE_HEIGHT - 12)
      ];
      const cells: PathItem[] = [];
      const spokes: PathItem[] = [];
      const neighbors: SpriteItem[] = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const x = windowOrigin[0] + dx * cellSize;
          const y = windowOrigin[1] + dy * cellSize;
          cells.push({
            points: [
              [x - 3, y - 3],
              [x + 3, y - 3],
              [x + 3, y + 3],
              [x - 3, y + 3],
              [x - 3, y - 3]
            ],
            role: 'primary',
            width: 0.4,
            alpha: 0.9
          });
          if (dx === 0 && dy === 0) continue;
          const difference = Math.abs(
            sampleAt(terrain.rough, COLUMNS, ROWS, column + dx, row + dy) - center
          );
          neighbors.push({
            x,
            y,
            radius: 0.8 + clamp(difference * 90, 0, 2.4),
            role: 'hot',
            alpha: 0.95,
            phase: (dx + 2 * dy) / 8
          });
          spokes.push({
            points: [windowOrigin, [x, y]],
            role: 'warm',
            width: 0.4,
            alpha: 0.7
          });
        }
      }
      return [
        createField(scaled, 'heat', 'ramp', {strength: 0.05, speed: 0.5}),
        createStrokeLayer(spokes, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createStrokeLayer(cells, {animation: 'static', blend: 'add'}),
        createSpriteLayer(
          [
            ...neighbors,
            {x: windowOrigin[0], y: windowOrigin[1], radius: 2.2, role: 'cool', alpha: 1}
          ],
          {pulse: 0.25}
        )
      ];
    }
  },

  // VRM: surface normals as ticks, parallel where smooth and scattered where rugged.
  'vector-ruggedness': {
    parameters: [
      {id: 'window', label: 'Window', min: 1, max: 5, step: 1, default: 3, unit: 'cells'}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const radius = Math.round(p.window);
      const normalScale = 55;
      const measure = (column: number, row: number): {vrm: number; mean: Point; slope: number} => {
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const [gx, gy] = getGradient(terrain.rough, COLUMNS, ROWS, column + dx, row + dy);
            const length = Math.sqrt(gx * gx * normalScale ** 2 + gy * gy * normalScale ** 2 + 1);
            sx += (-gx * normalScale) / length;
            sy += (-gy * normalScale) / length;
            sz += 1 / length;
            count++;
          }
        }
        const resultant = Math.sqrt(sx * sx + sy * sy + sz * sz);
        return {vrm: 1 - resultant / count, mean: [sx / count, sy / count], slope: 0};
      };
      const background = new Float32Array(32 * 20);
      const calm: PathItem[] = [];
      const rugged: PathItem[] = [];
      for (let row = 0; row < 20; row++) {
        for (let column = 0; column < 32; column++) {
          const result = measure(column * 4 + 2, row * 4 + 2);
          background[row * 32 + column] =
            row === 0 || column === 0 || row === 19 || column === 31
              ? 0
              : clamp(result.vrm * 14, 0, 1);
          const x = (column + 0.5) * 5;
          const y = (row + 0.5) * 5;
          const [gx, gy] = getGradient(terrain.rough, COLUMNS, ROWS, column * 4 + 2, row * 4 + 2);
          const direction = Math.atan2(-gy, -gx);
          const length = 1.2 + clamp(Math.hypot(gx, gy) * 120, 0, 1.6);
          const item: PathItem = {
            points: [
              [x - Math.cos(direction) * length, y - Math.sin(direction) * length],
              [x + Math.cos(direction) * length, y + Math.sin(direction) * length]
            ],
            role: result.vrm > 0.02 ? 'hot' : 'cool',
            width: 0.5,
            alpha: 0.9,
            phase: (column + row) / 52
          };
          (result.vrm > 0.02 ? rugged : calm).push(item);
        }
      }
      return [
        createField(background, 'ice', 'ramp', {filter: 'linear', strength: 0.06}, 32, 20),
        createStrokeLayer(calm, {animation: 'pulse', speed: 0.25, blend: 'add'}),
        createStrokeLayer(rugged, {animation: 'pulse', speed: 0.6, blend: 'add'})
      ];
    }
  },

  // Contours: marching squares. The pointer picks one level and that isoline glows.
  'terrain-contours': {
    parameters: [{id: 'levels', label: 'Levels', min: 3, max: 24, step: 1, default: 12}],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const probe = pointer ?? [92, 46];
      const level = pointer ? sampleHeight(terrain, probe) : 0.45;
      const segments = traceContour(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, level);
      return [
        createField(terrain.heights, 'terrain', 'contours', {
          levels: Math.round(p.levels),
          speed: 0.12
        }),
        createStrokeLayer(createSegmentPaths(segments, 'warm', 1.1), {
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        }),
        createSpriteLayer(
          pointer ? [{x: probe[0], y: probe[1], radius: 2.4, role: 'warm', alpha: 1}] : [],
          {pulse: 0.3}
        )
      ];
    }
  },

  // Geomorphons: ten landform classes from rays in eight directions; high ends hot, low ends cool.
  geomorphons: {
    parameters: [
      {id: 'search', label: 'Search radius', min: 10, max: 44, step: 1, default: 26, unit: 'px'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const classes = remember(terrain, 'geomorphon-classes', () => {
        const local = blurField(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, 3);
        const slope = getCoarseSlope(terrain);
        return terrain.coarse.map((height, index) => {
          const position = (height - local[index]) * 14;
          const steep = slope[index] > 0.2;
          let value = 4;
          if (position < -1) value = 0;
          else if (position < -0.55) value = 1;
          else if (position < -0.3) value = 2;
          else if (position < -0.12) value = 3;
          else if (position <= 0.12) value = steep ? 5 : 4;
          else if (position <= 0.3) value = 6;
          else if (position <= 0.55) value = 7;
          else if (position <= 1) value = 8;
          else value = 9;
          return value / 9;
        });
      });
      const probe = pointer ?? [86, 48];
      const centerHeight = sampleHeight(terrain, probe);
      const rays: PathItem[] = [];
      const higher: SpriteItem[] = [];
      const lower: SpriteItem[] = [];
      for (let direction = 0; direction < 8; direction++) {
        const angle = (direction / 8) * Math.PI * 2;
        const end = clampPoint(
          [probe[0] + Math.cos(angle) * p.search, probe[1] + Math.sin(angle) * p.search * 0.8],
          4
        );
        rays.push({
          points: [probe, end],
          role: 'primary',
          width: 0.6,
          alpha: 0.85,
          phase: direction / 8
        });
        const endHeight = sampleHeight(terrain, end);
        (endHeight > centerHeight ? higher : lower).push({
          x: end[0],
          y: end[1],
          radius: 1.8,
          role: endHeight > centerHeight ? 'hot' : 'cool',
          alpha: 1,
          phase: direction / 8
        });
      }
      return [
        createField(classes, 'spectral', 'ramp', {filter: 'nearest'}, COARSE_COLUMNS, COARSE_ROWS),
        createStrokeLayer(rays, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createSpriteLayer(
          [...higher, ...lower, {x: probe[0], y: probe[1], radius: 2.6, role: 'warm', alpha: 1}],
          {pulse: 0.2}
        )
      ];
    }
  },

  // Weiss landforms: the same position index at a small and a large scale, two rings.
  'weiss-landforms': {
    parameters: [
      {id: 'small', label: 'Small scale', min: 3, max: 10, step: 1, default: 5, unit: 'px'},
      {id: 'large', label: 'Large scale', min: 12, max: 34, step: 1, default: 22, unit: 'px'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const smallCells = Math.max(1, Math.round(p.small / COARSE_CELL));
      const largeCells = Math.max(2, Math.round(p.large / COARSE_CELL));
      const smallBlur = blurField(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, smallCells);
      const smallPosition = terrain.coarse.map((height, index) => height - smallBlur[index]);
      const largeBlur = blurField(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, largeCells);
      const largePosition = terrain.coarse.map((height, index) => height - largeBlur[index]);
      const smallDeviation = getStandardDeviation(smallPosition);
      const largeDeviation = getStandardDeviation(largePosition);
      const slope = getCoarseSlope(terrain);
      const classes = terrain.coarse.map((_, index) => {
        const small = smallPosition[index] / smallDeviation;
        const large = largePosition[index] / largeDeviation;
        const smallBand = small <= -1 ? 0 : small >= 1 ? 2 : 1;
        if (large <= -1) return smallBand / 9;
        if (large >= 1) return (7 + smallBand) / 9;
        if (smallBand === 0) return 3 / 9;
        if (smallBand === 2) return 6 / 9;
        return (slope[index] > 0.18 ? 5 : 4) / 9;
      });
      const probe = pointer ?? [86, 52];
      return [
        createField(classes, 'terrain', 'ramp', {filter: 'nearest'}, COARSE_COLUMNS, COARSE_ROWS),
        createStrokeLayer(
          [
            {points: createRing(probe, p.small, 28), role: 'cool', width: 0.9, alpha: 0.95},
            {
              points: createRing(probe, p.large, 48),
              role: 'warm',
              width: 0.9,
              alpha: 0.95,
              phase: 0.5
            }
          ],
          {animation: 'pulse', speed: 0.35, blend: 'add'}
        ),
        createSpriteLayer([{x: probe[0], y: probe[1], radius: 2, role: 'ink', alpha: 1}], {
          pulse: 0.1
        })
      ];
    }
  },

  // Multiscale topographic position: signed position (ridge hot, valley cool) at nested scales.
  'topographic-position': {
    parameters: [
      {id: 'scales', label: 'Scales', min: 1, max: 4, step: 1, default: 3},
      {id: 'base', label: 'Smallest window', min: 2, max: 8, step: 1, default: 3, unit: 'cells'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const scales = Math.round(p.scales);
      const total = new Float32Array(COLUMNS * ROWS);
      const radii: number[] = [];
      for (let scale = 0; scale < scales; scale++) {
        const radius = p.base * 2 ** scale;
        radii.push(radius);
        const blurred = blurField(terrain.rough, COLUMNS, ROWS, radius);
        for (let index = 0; index < total.length; index++) {
          total[index] += (terrain.rough[index] - blurred[index]) / (1 + scale * 0.35);
        }
      }
      const values = total.map(value => clamp(0.5 + (value / scales) * 11, 0, 1));
      const probe = pointer ?? [90, 48];
      return [
        createField(values, 'spectral', 'ramp', {strength: 0.05, speed: 0.5}),
        createStrokeLayer(
          radii.map((radius, index) => ({
            points: createRing(probe, radius * FINE_CELL, 40),
            role: 'ink' as Role,
            width: 0.8,
            alpha: 0.95,
            phase: index / radii.length
          })),
          {animation: 'flow', speed: 0.3, blend: 'alpha'}
        ),
        createSpriteLayer([{x: probe[0], y: probe[1], radius: 1.8, role: 'warm', alpha: 1}], {
          pulse: 0.3
        })
      ];
    }
  },

  // Critical points: peaks, pits and saddles on a contour map.
  'terrain-critical-points': {
    parameters: [
      {id: 'smoothing', label: 'Smoothing', min: 0, max: 4, step: 1, default: 2, unit: 'cells'}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const smoothed = blurField(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, p.smoothing);
      const peaks: Point[] = [];
      const pits: Point[] = [];
      const saddles: Point[] = [];
      const ring: Point[] = [
        [0, -1],
        [1, -1],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1]
      ];
      const far = (list: Point[], point: Point) =>
        list.every(other => Math.hypot(other[0] - point[0], other[1] - point[1]) > 9);
      for (let row = 3; row < COARSE_ROWS - 3; row++) {
        for (let column = 3; column < COARSE_COLUMNS - 3; column++) {
          const center = smoothed[row * COARSE_COLUMNS + column];
          let changes = 0;
          let higher = 0;
          let previous = 0;
          for (let index = 0; index <= ring.length; index++) {
            const [dx, dy] = ring[index % ring.length];
            const sign = smoothed[(row + dy) * COARSE_COLUMNS + column + dx] > center ? 1 : -1;
            if (index > 0 && sign !== previous) changes++;
            if (index < ring.length && sign > 0) higher++;
            previous = sign;
          }
          const point: Point = [(column + 0.5) * COARSE_CELL, (row + 0.5) * COARSE_CELL];
          if (higher === 0 && far(peaks, point)) peaks.push(point);
          else if (higher === 8 && far(pits, point)) pits.push(point);
          else if (changes >= 4 && far(saddles, point)) saddles.push(point);
        }
      }
      return [
        createField(terrain.heights, 'ice', 'contours', {levels: 12, speed: 0.1}),
        createSpriteLayer(
          [
            ...toSprites(saddles.slice(0, 8), 'violet', 2.4),
            ...toSprites(pits.slice(0, 8), 'cool', 3),
            ...toSprites(peaks.slice(0, 10), 'hot', 3.2)
          ],
          {pulse: 0.3}
        )
      ];
    }
  },

  // Summits: a disc that must stand above its surrounding ring.
  'terrain-summits': {
    parameters: [
      {id: 'radius', label: 'Disc radius', min: 6, max: 30, step: 1, default: 15, unit: 'px'}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const accepted: Summit[] = [];
      const rejected: Summit[] = [];
      for (const summit of terrain.summits) {
        const isHighest = accepted.every(
          other => Math.hypot(other.x - summit.x, other.y - summit.y) > p.radius
        );
        let ringSum = 0;
        for (let step = 0; step < 12; step++) {
          const angle = (step / 12) * Math.PI * 2;
          ringSum += sampleHeight(terrain, [
            clamp(summit.x + Math.cos(angle) * p.radius * 1.8, 1, SCENE_WIDTH - 1),
            clamp(summit.y + Math.sin(angle) * p.radius * 1.8, 1, SCENE_HEIGHT - 1)
          ]);
        }
        const stands = summit.height - ringSum / 12 > 0.08;
        (isHighest && stands ? accepted : rejected).push(summit);
      }
      const shown = accepted.slice(0, 7);
      return [
        createField(terrain.heights, 'terrain', 'hillshade', {
          strength: 5,
          light: getLightAngle(315)
        }),
        createStrokeLayer(
          shown.map((summit, index) => ({
            points: createRing([summit.x, summit.y], p.radius, 36),
            role: 'warm' as Role,
            width: 0.7,
            alpha: 0.9,
            phase: index / shown.length
          })),
          {animation: 'pulse', speed: 0.3, blend: 'add'}
        ),
        createStrokeLayer(
          shown.map(summit => ({
            points: createRing([summit.x, summit.y], p.radius * 1.8, 40),
            role: 'dim' as Role,
            width: 0.4,
            alpha: 0.7
          })),
          {animation: 'static'}
        ),
        createSpriteLayer(
          [
            ...toSprites(
              rejected.map(summit => [summit.x, summit.y] as Point),
              'dim',
              1.2
            ),
            ...toSprites(
              shown.map(summit => [summit.x, summit.y] as Point),
              'hot',
              3
            )
          ],
          {pulse: 0.25}
        )
      ];
    }
  },

  // Prominence: size shows how far a peak rises above its key col, not its height.
  'peak-prominence': {
    parameters: [
      {id: 'minimum', label: 'Minimum prominence', min: 0, max: 0.5, step: 0.02, default: 0.1}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const kept = terrain.summits.filter(summit => summit.prominence >= p.minimum);
      const dropped = terrain.summits.filter(summit => summit.prominence < p.minimum);
      const toKey = kept
        .filter(summit => summit.prominence > 0.02 && summit.keyX !== summit.x)
        .map(
          (summit): PathItem => ({
            points: [
              [summit.x, summit.y],
              [summit.keyX, summit.keyY]
            ],
            role: 'cool',
            width: 0.7,
            alpha: 0.85,
            phase: summit.height
          })
        );
      return [
        createField(terrain.heights, 'ice', 'contours', {levels: 10, speed: 0.1}),
        createStrokeLayer(toKey, {animation: 'flow', speed: 0.3, blend: 'add'}),
        createSpriteLayer(
          [
            ...kept.map(
              (summit, index): SpriteItem => ({
                x: summit.keyX,
                y: summit.keyY,
                radius: 1.3,
                role: 'cool',
                alpha: 0.9,
                phase: index / 9
              })
            ),
            ...dropped.map(
              (summit): SpriteItem => ({
                x: summit.x,
                y: summit.y,
                radius: 1.3,
                role: 'dim',
                alpha: 0.7
              })
            ),
            ...kept.map(
              (summit, index): SpriteItem => ({
                x: summit.x,
                y: summit.y,
                radius: 1.8 + summit.prominence * 16,
                role: summit.prominence > 0.45 ? 'hot' : 'warm',
                alpha: 0.9,
                phase: index / 9
              })
            )
          ],
          {pulse: 0.2}
        )
      ];
    }
  },

  // Peak snap: catalogue points jump to the DEM summit within the search radius.
  'peak-snap': {
    parameters: [
      {id: 'radius', label: 'Search radius', min: 4, max: 30, step: 1, default: 14, unit: 'px'}
    ],
    pointer: 'none',
    build: context => {
      const {p, fork} = context;
      const terrain = getTerrain(context);
      const random = fork('catalogue');
      const chosen: Summit[] = [];
      for (const summit of terrain.summits) {
        if (chosen.every(other => Math.hypot(other.x - summit.x, other.y - summit.y) > 22)) {
          chosen.push(summit);
        }
        if (chosen.length === 6) break;
      }
      const catalogue = chosen.map((summit): Point => {
        const angle = random.range(0, Math.PI * 2);
        const distance = random.range(9, 24);
        return clampPoint(
          [summit.x + Math.cos(angle) * distance, summit.y + Math.sin(angle) * distance],
          4
        );
      });
      const snapped: PathItem[] = [];
      const searchRings: PathItem[] = [];
      const movedPoints: SpriteItem[] = [];
      const stayedPoints: SpriteItem[] = [];
      chosen.forEach((summit, index) => {
        const start = catalogue[index];
        const distance = Math.hypot(start[0] - summit.x, start[1] - summit.y);
        searchRings.push({
          points: createRing(start, p.radius, 30),
          role: 'dim',
          width: 0.4,
          alpha: 0.8
        });
        if (distance <= p.radius) {
          snapped.push({
            points: [start, [summit.x, summit.y]],
            role: 'warm',
            width: 1,
            alpha: 0.95,
            phase: index / 6
          });
          movedPoints.push({
            x: start[0],
            y: start[1],
            radius: 1.6,
            role: 'warm',
            alpha: 0.9,
            phase: index / 6
          });
        } else {
          stayedPoints.push({
            x: start[0],
            y: start[1],
            radius: 1.8,
            role: 'cool',
            alpha: 0.95,
            phase: index / 6
          });
        }
      });
      return [
        createField(terrain.heights, 'terrain', 'hillshade', {
          strength: 4,
          light: getLightAngle(315)
        }),
        createStrokeLayer(searchRings, {animation: 'static'}),
        createStrokeLayer(snapped, {animation: 'flow', speed: 0.45, blend: 'add'}),
        createSpriteLayer(
          [
            ...stayedPoints,
            ...movedPoints,
            ...toSprites(
              chosen.map(summit => [summit.x, summit.y] as Point),
              'hot',
              2.8
            )
          ],
          {pulse: 0.25}
        )
      ];
    }
  },

  // Profile peaks: stacked 1-D skylines, each with prominence-filtered peaks.
  'profile-peaks': {
    parameters: [
      {id: 'prominence', label: 'Min prominence', min: 0.01, max: 0.2, step: 0.01, default: 0.05},
      {id: 'profiles', label: 'Profiles', min: 3, max: 8, step: 1, default: 5}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const count = Math.round(p.profiles);
      const lines: PathItem[] = [];
      const baselines: PathItem[] = [];
      const prominenceBars: PathItem[] = [];
      const peaks: SpriteItem[] = [];
      for (let profile = 0; profile < count; profile++) {
        const baseline = 18 + ((profile + 0.5) / count) * 78;
        const amplitude = Math.min(40, 110 / count);
        const row = Math.round(((profile + 0.5) / count) * (ROWS - 1));
        const series: number[] = [];
        for (let sample = 0; sample < 64; sample++) {
          series.push(terrain.rough[row * COLUMNS + sample * 2]);
        }
        const points = series.map(
          (value, index): Point => [(index + 0.5) * COARSE_CELL, baseline - value * amplitude]
        );
        lines.push({
          points: smoothPath(points, 1),
          role: 'primary',
          width: 0.8,
          alpha: 0.95,
          phase: profile / count
        });
        baselines.push({
          points: [
            [0, baseline],
            [SCENE_WIDTH, baseline]
          ],
          role: 'dim',
          width: 0.3,
          alpha: 0.7
        });
        for (let index = 1; index < series.length - 1; index++) {
          if (series[index] < series[index - 1] || series[index] < series[index + 1]) continue;
          let leftLow = series[index];
          for (let cursor = index - 1; cursor >= 0 && series[cursor] <= series[index]; cursor--) {
            leftLow = Math.min(leftLow, series[cursor]);
          }
          let rightLow = series[index];
          for (
            let cursor = index + 1;
            cursor < series.length && series[cursor] <= series[index];
            cursor++
          ) {
            rightLow = Math.min(rightLow, series[cursor]);
          }
          const prominence = series[index] - Math.max(leftLow, rightLow);
          if (prominence < p.prominence) continue;
          const [x, y] = points[index];
          peaks.push({x, y, radius: 1.9, role: 'hot', alpha: 1, phase: profile / count});
          prominenceBars.push({
            points: [
              [x, y],
              [x, y + prominence * amplitude]
            ],
            role: 'warm',
            width: 0.7,
            alpha: 0.9,
            phase: profile / count
          });
        }
      }
      return [
        createStrokeLayer(baselines, {animation: 'static'}),
        createStrokeLayer(lines, {animation: 'flow', speed: 0.2, blend: 'add'}),
        createStrokeLayer(prominenceBars, {animation: 'pulse', speed: 0.4, blend: 'add'}),
        createSpriteLayer(peaks, {pulse: 0.25})
      ];
    }
  },

  // Viewshed: visible ground bright, hidden ground dark, with sight rays from a movable observer.
  viewshed: {
    parameters: [
      {id: 'height', label: 'Eye height', min: 0, max: 6, step: 0.5, default: 1.5},
      {id: 'range', label: 'Range', min: 15, max: 70, step: 1, default: 50, unit: 'cells'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const summit = terrain.summits[1] ?? terrain.summits[0];
      const observer: Point = pointer ?? [summit.x + 4, summit.y + 4];
      const [column, row] = toCoarseCell(observer);
      const visible = computeViewshed(terrain, column, row, p.height, p.range);
      const values = new Float32Array(visible.length);
      for (let index = 0; index < values.length; index++) {
        values[index] = visible[index]
          ? 0.4 + 0.6 * terrain.coarse[index]
          : 0.03 + 0.08 * terrain.coarse[index];
      }
      const eye = terrain.coarse[row * COARSE_COLUMNS + column] * VERTICAL_SCALE + p.height;
      const rays: PathItem[] = [];
      const blockers: SpriteItem[] = [];
      for (let direction = 0; direction < 14; direction++) {
        const angle = (direction / 14) * Math.PI * 2 + 0.2;
        let maximum = -Infinity;
        let last: Point = observer;
        for (let step = 1; step < Math.min(p.range, 60); step++) {
          const sampleColumn = Math.round(column + Math.cos(angle) * step);
          const sampleRow = Math.round(row + Math.sin(angle) * step);
          if (
            sampleColumn < 0 ||
            sampleRow < 0 ||
            sampleColumn >= COARSE_COLUMNS ||
            sampleRow >= COARSE_ROWS
          )
            break;
          const tangent =
            (terrain.coarse[sampleRow * COARSE_COLUMNS + sampleColumn] * VERTICAL_SCALE - eye) /
            step;
          const sample: Point = [
            (sampleColumn + 0.5) * COARSE_CELL,
            (sampleRow + 0.5) * COARSE_CELL
          ];
          if (tangent >= maximum) {
            maximum = tangent;
            last = sample;
          } else {
            blockers.push({x: last[0], y: last[1], radius: 1.1, role: 'hot', alpha: 0.9});
            break;
          }
        }
        rays.push({
          points: [observer, last],
          role: 'primary',
          width: 0.5,
          alpha: 0.8,
          phase: direction / 14
        });
      }
      return [
        createField(values, 'heat', 'ramp', {filter: 'nearest'}, COARSE_COLUMNS, COARSE_ROWS),
        createStrokeLayer(rays, {animation: 'flow', speed: 0.35, blend: 'add'}),
        createSpriteLayer(
          [
            ...blockers,
            {x: observer[0], y: observer[1], radius: 5, role: 'cool', alpha: 0.3},
            {x: observer[0], y: observer[1], radius: 2.4, role: 'cool', alpha: 1}
          ],
          {pulse: 0.25}
        )
      ];
    }
  },

  // Cumulative viewshed: how many towers see each cell.
  'cumulative-viewshed': {
    parameters: [{id: 'towers', label: 'Observers', min: 1, max: 8, step: 1, default: 5}],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const towers = terrain.summits.slice(0, Math.round(p.towers));
      const counts = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
      towers.forEach((tower, index) => {
        const visible = remember(terrain, `viewshed-${index}`, () => {
          const [column, row] = toCoarseCell([tower.x, tower.y]);
          return computeViewshed(terrain, column, row, 1.5, 60);
        });
        for (let cell = 0; cell < counts.length; cell++) counts[cell] += visible[cell];
      });
      const values = counts.map(count =>
        count === 0 ? 0 : 0.15 + (0.85 * count) / Math.max(1, towers.length)
      );
      return [
        createField(
          values,
          'heat',
          'ramp',
          {filter: 'nearest', strength: 0.05},
          COARSE_COLUMNS,
          COARSE_ROWS
        ),
        createSpriteLayer(
          [
            ...toSprites(
              towers.map(tower => [tower.x, tower.y] as Point),
              'cool',
              6
            ).map(sprite => ({...sprite, alpha: 0.25})),
            ...toSprites(
              towers.map(tower => [tower.x, tower.y] as Point),
              'ink',
              2.2
            )
          ],
          {pulse: 0.3}
        )
      ];
    }
  },

  // Line of sight: observer-target pairs, clear lines glow, blocked ones break at the obstruction.
  'line-of-sight': {
    parameters: [{id: 'pairs', label: 'Pairs', min: 2, max: 8, step: 1, default: 5}],
    pointer: 'hover',
    build: context => {
      const {p, pointer, fork} = context;
      const terrain = getTerrain(context);
      const random = fork('targets');
      const count = Math.round(p.pairs);
      const observers = terrain.summits
        .slice(0, count)
        .map((summit): Point => [summit.x, summit.y]);
      if (pointer) observers[0] = pointer;
      const clear: PathItem[] = [];
      const blocked: PathItem[] = [];
      const hidden: PathItem[] = [];
      const sprites: SpriteItem[] = [];
      observers.forEach((observer, index) => {
        const target: Point = [random.range(10, 150), random.range(10, 90)];
        const [observerColumn, observerRow] = toCoarseCell(observer);
        const [targetColumn, targetRow] = toCoarseCell(target);
        const eye =
          terrain.coarse[observerRow * COARSE_COLUMNS + observerColumn] * VERTICAL_SCALE + 1;
        const targetHeight =
          terrain.coarse[targetRow * COARSE_COLUMNS + targetColumn] * VERTICAL_SCALE + 0.6;
        const distance = Math.hypot(targetColumn - observerColumn, targetRow - observerRow) || 1;
        const steps = Math.ceil(distance);
        let blockPoint: Point | null = null;
        for (let step = 1; step < steps; step++) {
          const amount = step / steps;
          const column = Math.round(mixNumbers(observerColumn, targetColumn, amount));
          const row = Math.round(mixNumbers(observerRow, targetRow, amount));
          const lineHeight = mixNumbers(eye, targetHeight, amount);
          if (terrain.coarse[row * COARSE_COLUMNS + column] * VERTICAL_SCALE > lineHeight) {
            blockPoint = [(column + 0.5) * COARSE_CELL, (row + 0.5) * COARSE_CELL];
            break;
          }
        }
        sprites.push({
          x: observer[0],
          y: observer[1],
          radius: index === 0 && pointer ? 3.4 : 2.4,
          role: 'warm',
          alpha: 1,
          phase: index / count
        });
        if (blockPoint) {
          blocked.push({
            points: [observer, blockPoint],
            role: 'primary',
            width: 0.9,
            alpha: 0.95,
            phase: index / count
          });
          hidden.push({
            points: [blockPoint, target],
            role: 'hot',
            width: 0.5,
            alpha: 0.5,
            phase: index / count
          });
          sprites.push({x: blockPoint[0], y: blockPoint[1], radius: 1.8, role: 'hot', alpha: 1});
          sprites.push({x: target[0], y: target[1], radius: 1.6, role: 'dim', alpha: 0.9});
        } else {
          clear.push({
            points: [observer, target],
            role: 'cool',
            width: 1.1,
            alpha: 0.95,
            phase: index / count
          });
          sprites.push({
            x: target[0],
            y: target[1],
            radius: 2,
            role: 'cool',
            alpha: 1,
            phase: index / count
          });
        }
      });
      return [
        createField(terrain.heights, 'terrain', 'hillshade', {
          strength: 4,
          light: getLightAngle(315)
        }),
        createStrokeLayer(hidden, {animation: 'pulse', speed: 0.3, blend: 'add'}),
        createStrokeLayer(blocked, {animation: 'flow', speed: 0.35, blend: 'add'}),
        createStrokeLayer(clear, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createSpriteLayer(sprites, {pulse: 0.2})
      ];
    }
  },

  // Point horizon profile: the skyline around an observer as a polar curve.
  'point-horizon-profile': {
    parameters: [
      {
        id: 'exaggeration',
        label: 'Angle scale',
        min: 0.5,
        max: 2.5,
        step: 0.1,
        default: 1.3,
        unit: '×'
      }
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const observer: Point = pointer ?? [76, 54];
      const [column, row] = toCoarseCell(observer);
      const inner = 7;
      const curve: Point[] = [];
      for (let step = 0; step <= 72; step++) {
        const azimuth = (step / 72) * 360;
        const tangent = getObserverHorizon(terrain, column, row, azimuth);
        const radius = inner + Math.min(1, Math.atan(tangent) * p.exaggeration * 1.2) * 24;
        const angle = (azimuth * Math.PI) / 180;
        curve.push([
          observer[0] + Math.sin(angle) * radius,
          observer[1] - Math.cos(angle) * radius
        ]);
      }
      const spokes: PathItem[] = Array.from({length: 12}, (_, index) => {
        const angle = (index / 12) * Math.PI * 2;
        return {
          points: [
            [observer[0] + Math.sin(angle) * inner, observer[1] - Math.cos(angle) * inner],
            [
              observer[0] + Math.sin(angle) * (inner + 24),
              observer[1] - Math.cos(angle) * (inner + 24)
            ]
          ] as Point[],
          role: 'dim' as Role,
          width: 0.3,
          alpha: 0.8
        };
      });
      return [
        createField(terrain.heights, 'ice', 'hillshade', {strength: 3, light: getLightAngle(315)}),
        createStrokeLayer(spokes, {animation: 'static'}),
        createStrokeLayer(
          [{points: createRing(observer, inner, 36), role: 'dim', width: 0.5, alpha: 0.9}],
          {animation: 'static'}
        ),
        createFillLayer([{ring: curve, role: 'violet', alpha: 0.28}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createStrokeLayer([{points: curve, role: 'warm', width: 1.4, alpha: 1}], {
          animation: 'flow',
          speed: 0.3,
          trail: 1.5,
          blend: 'add'
        }),
        createSpriteLayer([{x: observer[0], y: observer[1], radius: 2.2, role: 'warm', alpha: 1}], {
          pulse: 0.3
        })
      ];
    }
  },

  // Point horizon visibility: targets above or below the unrolled skyline.
  'point-horizon-visibility': {
    parameters: [
      {
        id: 'observer',
        label: 'Observer position',
        min: 0,
        max: 100,
        step: 2,
        default: 40,
        unit: '%'
      },
      {id: 'targets', label: 'Targets', min: 6, max: 40, step: 1, default: 18}
    ],
    pointer: 'none',
    build: context => {
      const {p, fork} = context;
      const terrain = getTerrain(context);
      const random = fork('targets');
      const observer: Point = [mixNumbers(30, 130, p.observer / 100), 54];
      const [column, row] = toCoarseCell(observer);
      const baseline = 80;
      const scale = 52;
      const horizon: number[] = [];
      const silhouette: Point[] = [];
      for (let step = 0; step <= 72; step++) {
        const tangent = getObserverHorizon(terrain, column, row, (step / 72) * 360);
        const angle = Math.min(1, Math.atan(tangent) * 1.6);
        horizon.push(angle);
        silhouette.push([8 + (step / 72) * 144, baseline - angle * scale]);
      }
      const seen: SpriteItem[] = [];
      const hiddenTargets: SpriteItem[] = [];
      const drops: PathItem[] = [];
      for (let index = 0; index < p.targets; index++) {
        const x = random.range(10, 150);
        const angle = random.range(0.02, 0.85);
        const skyline = horizon[Math.round(((x - 8) / 144) * 72)];
        const y = baseline - angle * scale;
        if (angle > skyline) {
          seen.push({x, y, radius: 2, role: 'warm', alpha: 1, phase: random.next()});
          drops.push({
            points: [
              [x, y],
              [x, baseline - skyline * scale]
            ],
            role: 'warm',
            width: 0.4,
            alpha: 0.6
          });
        } else {
          hiddenTargets.push({x, y, radius: 1.4, role: 'dim', alpha: 0.8});
        }
      }
      return [
        createGridLayer(),
        createFillLayer(
          [
            {
              ring: [...silhouette, [152, baseline + 12], [8, baseline + 12]],
              role: 'cool',
              alpha: 0.35
            }
          ],
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer([{points: silhouette, role: 'primary', width: 1.2, alpha: 1}], {
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        }),
        createStrokeLayer(drops, {animation: 'static', blend: 'add'}),
        createSpriteLayer([...hiddenTargets, ...seen], {pulse: 0.3})
      ];
    }
  },

  // Horizon angles and sky-view factor: a sky-openness map with a dome glyph at the probe.
  'horizon-angles': {
    parameters: [
      {
        id: 'directions',
        label: 'Directions',
        min: 0,
        max: 2,
        step: 1,
        default: 2,
        options: ['4', '6', '12']
      }
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const directions = [4, 6, 12][Math.round(p.directions)];
      const stride = DIRECTION_COUNT / directions;
      const skyView = getSkyViewFactor(terrain, directions);
      const probe: Point = pointer ?? [100, 44];
      const [column, row] = toCoarseCell(probe);
      const cell = row * COARSE_COLUMNS + column;
      const map = getHorizonMap(terrain);
      const dome: Point[] = [];
      for (let direction = 0; direction < DIRECTION_COUNT; direction += stride) {
        const tangent = map[cell * DIRECTION_COUNT + direction];
        const openness = 1 - tangent / Math.sqrt(1 + tangent * tangent);
        const angle = (direction / DIRECTION_COUNT) * Math.PI * 2;
        dome.push([
          probe[0] + Math.sin(angle) * openness * 15,
          probe[1] - Math.cos(angle) * openness * 15
        ]);
      }
      dome.push(dome[0]);
      return [
        createField(
          skyView,
          'ice',
          'ramp',
          {strength: 0.05, speed: 0.4},
          COARSE_COLUMNS,
          COARSE_ROWS
        ),
        createStrokeLayer(
          [{points: createRing(probe, 15, 36), role: 'dim', width: 0.5, alpha: 0.9}],
          {animation: 'static'}
        ),
        createFillLayer([{ring: dome, role: 'warm', alpha: 0.35}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createStrokeLayer([{points: dome, role: 'warm', width: 1.1, alpha: 1}], {
          animation: 'flow',
          speed: 0.35,
          blend: 'add'
        }),
        createSpriteLayer([{x: probe[0], y: probe[1], radius: 2, role: 'warm', alpha: 1}], {
          pulse: 0.3
        })
      ];
    }
  },

  // Terrain cast shadow: a low sun, parallel rays and long dark shadows.
  'terrain-cast-shadow': {
    parameters: [
      {id: 'azimuth', label: 'Sun azimuth', min: 0, max: 360, step: 5, default: 250, unit: '°'},
      {id: 'altitude', label: 'Sun altitude', min: 3, max: 50, step: 1, default: 14, unit: '°'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const azimuth = pointer ? getPointerAzimuth(pointer) : p.azimuth;
      const sun = pointer ?? getSunPosition(azimuth);
      const lit = computeShadow(terrain, azimuth, p.altitude);
      const values = lit.map((value, index) =>
        value ? 0.45 + 0.55 * terrain.coarse[index] : 0.04
      );
      return [
        createField(values, 'ice', 'ramp', {}, COARSE_COLUMNS, COARSE_ROWS),
        createSunRays(sun, 'warm', 6),
        createSunSprite(sun)
      ];
    }
  },

  // Solar shadow mask: one frame of a day, soft penumbra, sun marching along its arc.
  'solar-shadow-mask': {
    parameters: [
      {id: 'time', label: 'Time of day', min: 0.06, max: 0.94, step: 0.02, default: 0.28},
      {id: 'softness', label: 'Softness', min: 0, max: 4, step: 1, default: 1, unit: 'cells'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const time = pointer ? clamp((pointer[0] - 14) / 132, 0.06, 0.94) : p.time;
      const azimuth = 90 + time * 180;
      const altitude = 4 + 38 * Math.sin(Math.PI * time);
      const sunTangent = Math.tan((altitude * Math.PI) / 180);
      const horizon = getHorizonMap(terrain);
      const mask = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
      for (let cell = 0; cell < mask.length; cell++) {
        mask[cell] = getHorizonTangent(horizon, cell, azimuth) < sunTangent ? 1 : 0;
      }
      const soft = blurField(mask, COARSE_COLUMNS, COARSE_ROWS, p.softness);
      const values = soft.map(
        (value, index) => 0.05 + value * (0.4 + 0.55 * terrain.coarse[index])
      );
      const sun: Point = [14 + 132 * time, 22 - 16 * Math.sin(Math.PI * time)];
      const frames = Array.from({length: 7}, (_, index): Point => {
        const amount = (index + 0.5) / 7;
        return [14 + 132 * amount, 22 - 16 * Math.sin(Math.PI * amount)];
      });
      return [
        createField(values, 'ice', 'ramp', {filter: 'linear'}, COARSE_COLUMNS, COARSE_ROWS),
        createStrokeLayer([createSunArc()], {animation: 'static'}),
        createSpriteLayer(toSprites(frames, 'dim', 1.1, undefined, 0.9)),
        createSunSprite(sun, 3.4)
      ];
    }
  },

  // Solar position: day and night across a grid of places, with each place's sun azimuth.
  'solar-position': {
    parameters: [
      {id: 'hour', label: 'UTC hour', min: 0, max: 24, step: 0.5, default: 9},
      {id: 'declination', label: 'Season', min: -23.4, max: 23.4, step: 0.5, default: 12, unit: '°'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const declinationDegrees = pointer
        ? clamp(((50 - pointer[1]) / 38) * 23.4, -23.4, 23.4)
        : p.declination;
      const subsolarLongitude = pointer ? ((pointer[0] - 8) / 144) * 360 - 180 : (12 - p.hour) * 15;
      const declination = (declinationDegrees * Math.PI) / 180;
      const day: SpriteItem[] = [];
      const night: SpriteItem[] = [];
      const ticks: PathItem[] = [];
      for (let latitudeIndex = 0; latitudeIndex < 9; latitudeIndex++) {
        for (let longitudeIndex = 0; longitudeIndex < 17; longitudeIndex++) {
          const longitude = -180 + (longitudeIndex / 16) * 360;
          const latitude = 70 - (latitudeIndex / 8) * 140;
          const x = 8 + (longitudeIndex / 16) * 144;
          const y = 12 + (latitudeIndex / 8) * 76;
          const phi = (latitude * Math.PI) / 180;
          const hourAngle = (((longitude - subsolarLongitude + 540) % 360) - 180) * (Math.PI / 180);
          const sinAltitude =
            Math.sin(phi) * Math.sin(declination) +
            Math.cos(phi) * Math.cos(declination) * Math.cos(hourAngle);
          if (sinAltitude <= 0) {
            night.push({x, y, radius: 1, role: 'dim', alpha: 0.5});
            continue;
          }
          const azimuth =
            Math.atan2(
              Math.sin(hourAngle),
              Math.cos(hourAngle) * Math.sin(phi) - Math.tan(declination) * Math.cos(phi)
            ) + Math.PI;
          day.push({
            x,
            y,
            radius: 0.9 + sinAltitude * 1.6,
            role: sinAltitude > 0.6 ? 'hot' : 'warm',
            alpha: 0.6 + sinAltitude * 0.4,
            phase: longitudeIndex / 17
          });
          ticks.push({
            points: [
              [x, y],
              [x + Math.sin(azimuth) * 5, y - Math.cos(azimuth) * 5]
            ],
            role: 'warm',
            width: 0.4,
            alpha: 0.7
          });
        }
      }
      const terminator: Point[] = [];
      if (Math.abs(declinationDegrees) > 0.5) {
        for (let step = 0; step <= 72; step++) {
          const longitude = -180 + (step / 72) * 360;
          const hourAngle = ((longitude - subsolarLongitude) * Math.PI) / 180;
          const latitude =
            (Math.atan(-Math.cos(hourAngle) / Math.tan(declination)) * 180) / Math.PI;
          terminator.push([8 + (step / 72) * 144, 50 - (latitude / 70) * 38]);
        }
      }
      const subsolarX = 8 + (((subsolarLongitude + 540) % 360) / 360) * 144;
      const subsolar: Point = [subsolarX, 50 - (declinationDegrees / 70) * 38];
      return [
        createGridLayer(),
        createSpriteLayer(night),
        createStrokeLayer(ticks, {animation: 'static', blend: 'add'}),
        createSpriteLayer(day, {pulse: 0.15}),
        ...(terminator.length > 0
          ? [
              {
                type: 'strokes' as const,
                items: [{points: terminator, role: 'violet' as Role, width: 0.9, alpha: 0.9}],
                animation: 'flow' as const,
                speed: 0.2,
                blend: 'add' as const
              }
            ]
          : []),
        createSunSprite(subsolar, 3.4)
      ];
    }
  },

  // Solar irradiance: sun-hours accumulated over a day, with the sun path above.
  'solar-irradiance': {
    parameters: [{id: 'season', label: 'Season', min: 0, max: 1, step: 0.05, default: 0.6}],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const horizon = getHorizonMap(terrain);
      const slope = remember(terrain, 'coarse-gradient', () => {
        const gradients = new Float32Array(COARSE_COLUMNS * COARSE_ROWS * 2);
        for (let row = 0; row < COARSE_ROWS; row++) {
          for (let column = 0; column < COARSE_COLUMNS; column++) {
            const [gx, gy] = getGradient(terrain.coarse, COARSE_COLUMNS, COARSE_ROWS, column, row);
            gradients[(row * COARSE_COLUMNS + column) * 2] = -gx * VERTICAL_SCALE;
            gradients[(row * COARSE_COLUMNS + column) * 2 + 1] = -gy * VERTICAL_SCALE;
          }
        }
        return gradients;
      });
      const positions = 9;
      const maximumAltitude = 18 + 42 * p.season;
      const accumulated = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
      const sunPoints: Point[] = [];
      for (let index = 0; index < positions; index++) {
        const time = (index + 0.5) / positions;
        const azimuth = mixNumbers(110 - 45 * p.season, 250 + 45 * p.season, time);
        const altitude = Math.max(2, maximumAltitude * Math.sin(Math.PI * time));
        const altitudeRadians = (altitude * Math.PI) / 180;
        const sunTangent = Math.tan(altitudeRadians);
        const azimuthRadians = (azimuth * Math.PI) / 180;
        const sx = Math.sin(azimuthRadians) * Math.cos(altitudeRadians);
        const sy = -Math.cos(azimuthRadians) * Math.cos(altitudeRadians);
        const sz = Math.sin(altitudeRadians);
        for (let cell = 0; cell < accumulated.length; cell++) {
          if (getHorizonTangent(horizon, cell, azimuth) >= sunTangent) continue;
          const nx = slope[cell * 2];
          const ny = slope[cell * 2 + 1];
          const dot = (nx * sx + ny * sy + sz) / Math.sqrt(nx * nx + ny * ny + 1);
          accumulated[cell] += Math.max(0, dot);
        }
        sunPoints.push([
          14 + 132 * time,
          22 - 16 * (maximumAltitude / 60) * Math.sin(Math.PI * time) - 2
        ]);
      }
      const normalized = normalizeValues(accumulated);
      return [
        createField(
          normalized,
          'heat',
          'ramp',
          {strength: 0.05, speed: 0.4},
          COARSE_COLUMNS,
          COARSE_ROWS
        ),
        createStrokeLayer([{points: sunPoints, role: 'warm', width: 0.6, alpha: 0.9}], {
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        }),
        createSpriteLayer(toSprites(sunPoints, 'warm', 2.2), {pulse: 0.35})
      ];
    }
  },

  // Relief shading: a ring of lights whose shading is averaged into multidirectional relief.
  'relief-shading': {
    parameters: [
      {id: 'lights', label: 'Light directions', min: 1, max: 8, step: 1, default: 5},
      {id: 'exaggeration', label: 'Exaggeration', min: 1, max: 14, step: 0.5, default: 5, unit: '×'}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const count = Math.round(p.lights);
      const azimuths = Array.from({length: count}, (_, index) =>
        count === 1 ? 315 : 200 + (index / (count - 1)) * 320
      );
      const shaded = shadeHeights(terrain.heights, p.exaggeration, azimuths);
      const values = shaded.map(value => clamp((value - 0.2) * 1.4, 0, 1));
      const roles: Role[] = ['warm', 'hot', 'violet', 'cool', 'primary'];
      const summit: Point = [80, 52];
      const ringCount = 7;
      const lightLayers: PreviewLayer[] = [];
      const sunSprites: SpriteItem[] = [];
      const rayItems: PathItem[] = [];
      const arcItems: PathItem[] = [];
      azimuths.forEach((azimuth, index) => {
        const role = roles[index % roles.length];
        const sun = getSunPosition(azimuth, 0.95);
        const angle = (azimuth * Math.PI) / 180;
        const directionX = Math.sin(angle);
        const directionY = -Math.cos(angle);
        sunSprites.push({x: sun[0], y: sun[1], radius: 3.4, role, alpha: 1, phase: index / count});
        // Three parallel rays travel from the sun towards the summit.
        for (let ray = -1; ray <= 1; ray++) {
          const offsetX = -directionY * ray * 2.6;
          const offsetY = directionX * ray * 2.6;
          rayItems.push({
            points: [
              [sun[0] + offsetX - directionX * 3.5, sun[1] + offsetY - directionY * 3.5],
              [
                mixNumbers(sun[0], summit[0], 0.7) + offsetX,
                mixNumbers(sun[1], summit[1], 0.7) + offsetY
              ]
            ],
            role,
            width: 0.6,
            alpha: 0.75,
            phase: index / count
          });
        }
        // The side of every contour that faces this light is lit in its colour.
        const centerAngle = Math.atan2(directionY, directionX);
        for (let ring = 1; ring <= ringCount; ring++) {
          const radiusX = ring * 4.6;
          const radiusY = ring * 2.9;
          const arc: Point[] = [];
          for (let step = -6; step <= 6; step++) {
            const arcAngle = centerAngle + (step / 6) * 0.75;
            arc.push([
              summit[0] + Math.cos(arcAngle) * radiusX,
              summit[1] + Math.sin(arcAngle) * radiusY
            ]);
          }
          arcItems.push({
            points: arc,
            role,
            width: 0.9,
            alpha: 0.9 - ring * 0.05,
            phase: ring / ringCount
          });
        }
      });
      lightLayers.push(
        createStrokeLayer(
          [
            {
              points: createRing(CENTER, 66 * 0.95, 60, 40 * 0.95),
              role: 'dim',
              width: 0.3,
              alpha: 0.5
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(arcItems, {animation: 'pulse', speed: 0.3, blend: 'add'}),
        createStrokeLayer(rayItems, {animation: 'flow', speed: 0.35, blend: 'add'}),
        createSpriteLayer(toSprites([summit], 'ink', 2.2), {pulse: 0.2}),
        createSpriteLayer(sunSprites, {pulse: 0.3})
      );
      return [createField(values, 'terrain', 'ramp', {filter: 'linear'}), ...lightLayers];
    }
  },

  // Texture shading: fine ridge and valley relief at every scale, lit by a sweeping sun.
  'texture-shading': {
    parameters: [
      {id: 'detail', label: 'Detail', min: 0, max: 1, step: 0.05, default: 0.8},
      {id: 'exaggeration', label: 'Exaggeration', min: 1, max: 14, step: 0.5, default: 7, unit: '×'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer, fork} = context;
      const terrain = getTerrain(context);
      const enhanced = new Float32Array(terrain.rough);
      for (let scale = 0; scale < 4; scale++) {
        const radius = 2 ** scale;
        const blurred = blurField(terrain.rough, COLUMNS, ROWS, radius);
        const weight = p.detail * 1.6 * radius ** -0.4;
        for (let index = 0; index < enhanced.length; index++) {
          enhanced[index] += (terrain.rough[index] - blurred[index]) * weight;
        }
      }
      const sharp = normalizeValues(enhanced);
      const soft = normalizeValues(blurField(terrain.rough, COLUMNS, ROWS, 4));
      // Left half: conventional hillshade of the soft surface. Right half: texture shaded.
      const split = COLUMNS / 2;
      const values = new Float32Array(COLUMNS * ROWS);
      for (let row = 0; row < ROWS; row++) {
        for (let column = 0; column < COLUMNS; column++) {
          const index = row * COLUMNS + column;
          values[index] = column < split ? soft[index] : sharp[index];
        }
      }
      const random = fork('drainage');
      const valleyItems: PathItem[] = [];
      const ridgeItems: PathItem[] = [];
      const trunkStarts = [96, 108, 120, 132, 144, 156];
      trunkStarts.forEach((startX, trunkIndex) => {
        const trunk: Point[] = [];
        const wobble = random.range(0, 6);
        for (let step = 0; step <= 12; step++) {
          const along = step / 12;
          trunk.push([startX - along * 12 + Math.sin(along * 7 + wobble) * 3.2, 96 - along * 88]);
        }
        const smoothTrunk = smoothPath(trunk, 2);
        valleyItems.push({points: smoothTrunk, role: 'cool', width: 1.1, alpha: 0.95});
        // Tributaries branch from the trunk, alternating sides, and ridges run between them.
        for (let branch = 1; branch <= 10; branch++) {
          const base = trunk[Math.ceil(branch / 2) * 2 - 1];
          const side = branch % 2 === 0 ? -1 : 1;
          const length = random.range(6, 10);
          const tributary: Point[] = [];
          for (let step = 0; step <= 4; step++) {
            const along = step / 4;
            tributary.push([base[0] + side * along * length, base[1] - along * length * 1.1]);
          }
          valleyItems.push({
            points: smoothPath(tributary, 1),
            role: 'cool',
            width: 0.6,
            alpha: 0.8,
            phase: branch / 10
          });
        }
        const ridge: Point[] = trunk.map(
          ([x, y], index): Point => [x + 6 + Math.sin(index * 0.9 + trunkIndex) * 1.5, y]
        );
        ridgeItems.push({points: smoothPath(ridge, 2), role: 'warm', width: 0.8, alpha: 0.9});
      });
      // The soft side only has the broad drainage, drawn wide and blurry.
      const softItems: PathItem[] = [
        [
          [22, 94],
          [34, 70],
          [28, 48],
          [44, 24],
          [40, 8]
        ],
        [
          [66, 96],
          [58, 72],
          [68, 50],
          [56, 30],
          [62, 6]
        ]
      ].map(points => ({
        points: smoothPath(points as Point[], 2),
        role: 'dim' as Role,
        width: 4.5,
        alpha: 0.4
      }));
      const probe: Point = pointer ? [Math.max(pointer[0], 96), pointer[1]] : [118, 50];
      return [
        createField(values, 'ice', 'hillshade', {strength: p.exaggeration}),
        createStrokeLayer(softItems, {animation: 'static', blend: 'add'}),
        createStrokeLayer(ridgeItems, {animation: 'flow', speed: 0.2, blend: 'add'}),
        createStrokeLayer(valleyItems, {animation: 'flow', speed: 0.3, blend: 'add'}),
        createStrokeLayer(
          [
            {
              points: [
                [80, 0],
                [80, SCENE_HEIGHT]
              ],
              role: 'ink',
              width: 0.7,
              alpha: 0.9
            }
          ],
          {animation: 'pulse', speed: 0.4, blend: 'add'}
        ),
        createStrokeLayer(
          [1, 2, 4, 8].map((radius, index) => ({
            points: createRing(probe, radius * FINE_CELL * 1.2, 28),
            role: 'violet' as Role,
            width: 0.5,
            alpha: 0.9,
            phase: index / 4
          })),
          {animation: 'pulse', speed: 0.4, blend: 'add'}
        )
      ];
    }
  },

  // Simple local relief: elevation minus the local mean, one window.
  'simple-local-relief': {
    parameters: [
      {id: 'window', label: 'Window', min: 3, max: 24, step: 1, default: 9, unit: 'cells'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const trend = blurField(terrain.rough, COLUMNS, ROWS, p.window);
      const values = terrain.rough.map((height, index) =>
        clamp(0.5 + (height - trend[index]) * 4, 0, 1)
      );
      const probe = pointer ?? [88, 50];
      return [
        createField(values, 'spectral', 'ramp', {strength: 0.05, speed: 0.4}),
        createStrokeLayer(
          [
            {
              points: createRing(probe, p.window * FINE_CELL, 40),
              role: 'ink',
              width: 0.9,
              alpha: 0.95
            },
            {
              points: [
                [probe[0] - 3, probe[1]],
                [probe[0] + 3, probe[1]]
              ],
              role: 'ink',
              width: 0.7,
              alpha: 0.95
            },
            {
              points: [
                [probe[0], probe[1] - 3],
                [probe[0], probe[1] + 3]
              ],
              role: 'ink',
              width: 0.7,
              alpha: 0.95
            }
          ],
          {animation: 'static'}
        )
      ];
    }
  },

  // Multi-scale relief: the band between successive low-pass surfaces, drawn as contours.
  'multi-scale-relief': {
    parameters: [
      {id: 'scales', label: 'Low-pass steps', min: 2, max: 5, step: 1, default: 3},
      {id: 'start', label: 'Smallest window', min: 2, max: 8, step: 1, default: 3, unit: 'cells'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const steps = Math.round(p.scales);
      const radii = Array.from({length: steps}, (_, index) => p.start * 2 ** index);
      let previous = terrain.rough;
      const sum = new Float32Array(COLUMNS * ROWS);
      for (const radius of radii) {
        const blurred = blurField(terrain.rough, COLUMNS, ROWS, radius);
        for (let index = 0; index < sum.length; index++)
          sum[index] += previous[index] - blurred[index];
        previous = blurred;
      }
      const values = sum.map(value => clamp(0.5 + (value / steps) * 7, 0, 1));
      const probe = pointer ?? [88, 50];
      return [
        createField(values, 'ice', 'contours', {levels: 12, speed: 0.15}),
        createStrokeLayer(
          radii.map((radius, index) => ({
            points: createRing(probe, radius * FINE_CELL, 40),
            role: (index % 2 === 0 ? 'warm' : 'hot') as Role,
            width: 0.8,
            alpha: 0.95,
            phase: index / radii.length
          })),
          {animation: 'flow', speed: 0.3, blend: 'add'}
        )
      ];
    }
  },

  // Local dominance: how far the centre rises above rings of terrain at a range of distances.
  'local-dominance': {
    parameters: [
      {id: 'inner', label: 'Inner radius', min: 2, max: 8, step: 1, default: 3, unit: 'cells'},
      {id: 'outer', label: 'Outer radius', min: 6, max: 20, step: 1, default: 12, unit: 'cells'}
    ],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const rings = 3;
      const values = new Float32Array(COARSE_COLUMNS * COARSE_ROWS);
      for (let row = 0; row < COARSE_ROWS; row++) {
        for (let column = 0; column < COARSE_COLUMNS; column++) {
          const center = terrain.coarse[row * COARSE_COLUMNS + column];
          let sum = 0;
          let count = 0;
          for (let ring = 0; ring < rings; ring++) {
            const radius = mixNumbers(p.inner, p.outer, ring / (rings - 1)) / 2;
            for (let direction = 0; direction < 8; direction++) {
              const angle = (direction / 8) * Math.PI * 2;
              const other = sampleAt(
                terrain.coarse,
                COARSE_COLUMNS,
                COARSE_ROWS,
                column + Math.cos(angle) * radius,
                row + Math.sin(angle) * radius
              );
              sum += Math.max(0, center - other);
              count++;
            }
          }
          values[row * COARSE_COLUMNS + column] = clamp((sum / count) * 5, 0, 1);
        }
      }
      const probe = pointer ?? [92, 46];
      const samples: SpriteItem[] = [];
      for (let ring = 0; ring < rings; ring++) {
        const radius = (mixNumbers(p.inner, p.outer, ring / (rings - 1)) / 2) * COARSE_CELL;
        for (let direction = 0; direction < 8; direction++) {
          const angle = (direction / 8) * Math.PI * 2 + ring * 0.2;
          samples.push({
            x: probe[0] + Math.cos(angle) * radius,
            y: probe[1] + Math.sin(angle) * radius,
            radius: 0.9,
            role: 'cool',
            alpha: 0.95,
            phase: ring / rings
          });
        }
      }
      return [
        createField(
          values,
          'heat',
          'ramp',
          {strength: 0.05, speed: 0.4},
          COARSE_COLUMNS,
          COARSE_ROWS
        ),
        createStrokeLayer(
          [p.inner, p.outer].map((radius, index) => ({
            points: createRing(probe, (radius / 2) * COARSE_CELL, 40),
            role: 'cool' as Role,
            width: 0.4,
            alpha: 0.7,
            phase: index
          })),
          {animation: 'static'}
        ),
        createSpriteLayer(
          [...samples, {x: probe[0], y: probe[1], radius: 2.4, role: 'warm', alpha: 1}],
          {pulse: 0.25}
        )
      ];
    }
  },

  // Relief blend: several single-band visualisations merging into one image.
  'relief-blend': {
    parameters: [
      {id: 'shade', label: 'Hillshade weight', min: 0, max: 1, step: 0.05, default: 0.6},
      {id: 'sky', label: 'Sky-view weight', min: 0, max: 1, step: 0.05, default: 0.5}
    ],
    pointer: 'none',
    build: context => {
      const {p} = context;
      const terrain = getTerrain(context);
      const shaded = shadeHeights(terrain.heights, 5, [315]);
      const smooth = blurField(terrain.heights, COLUMNS, ROWS, 6);
      const slope = new Float32Array(COLUMNS * ROWS);
      for (let row = 0; row < ROWS; row++) {
        for (let column = 0; column < COLUMNS; column++) {
          const [gx, gy] = getGradient(terrain.heights, COLUMNS, ROWS, column, row);
          slope[row * COLUMNS + column] = clamp(Math.hypot(gx, gy) * 60, 0, 1);
        }
      }
      const values = shaded.map((shade, index) => {
        const sky = clamp(0.5 + (terrain.rough[index] - smooth[index]) * 5, 0, 1);
        const total = p.shade + p.sky + 0.4 || 1;
        return clamp((shade * p.shade + sky * p.sky + (1 - slope[index]) * 0.4) / total, 0, 1);
      });
      const rails: PathItem[] = (['cool', 'primary', 'violet', 'warm', 'hot'] as const).map(
        (role, index) => ({
          points: smoothPath(
            [
              [4, 8 + index * 7],
              [40, 8 + index * 7],
              [88, 16 + index * 3],
              [112, 28]
            ],
            2
          ),
          role,
          width: 0.9,
          alpha: 0.9,
          phase: index / 5
        })
      );
      return [
        createField(values, 'ice', 'ramp', {strength: 0.05, speed: 0.3}),
        createStrokeLayer(rails, {animation: 'flow', speed: 0.3, trail: 1.4, blend: 'add'}),
        createSpriteLayer([{x: 112, y: 28, radius: 3, role: 'warm', alpha: 1}], {pulse: 0.35})
      ];
    }
  },

  // Terrain-RGB decode: a sweep from the encoded, colour-banded tile to smooth decoded heights.
  'terrain-rgb-decode': {
    parameters: [{id: 'decoded', label: 'Decoded', min: 0, max: 1, step: 0.02, default: 0.5}],
    pointer: 'hover',
    build: context => {
      const {p, pointer} = context;
      const terrain = getTerrain(context);
      const sweep = pointer ? clamp(pointer[0] / SCENE_WIDTH, 0, 1) : p.decoded;
      const boundary = Math.round(sweep * COLUMNS);
      const values = new Float32Array(COLUMNS * ROWS);
      for (let row = 0; row < ROWS; row++) {
        for (let column = 0; column < COLUMNS; column++) {
          const index = row * COLUMNS + column;
          if (column < boundary) {
            values[index] = terrain.heights[index];
          } else {
            const block =
              terrain.heights[Math.floor(row / 5) * 5 * COLUMNS + Math.floor(column / 5) * 5];
            values[index] = (block * 6.3) % 1;
          }
        }
      }
      const x = boundary * FINE_CELL;
      return [
        createField(values, 'spectral', 'ramp', {filter: 'nearest'}),
        createStrokeLayer(
          [
            {
              points: [
                [x, 0],
                [x, SCENE_HEIGHT]
              ],
              role: 'ink',
              width: 1.2,
              alpha: 0.95
            }
          ],
          {animation: 'flow', speed: 0.4, blend: 'alpha'}
        )
      ];
    }
  },

  // Spike repair: +/-1 high-byte errors show as bright specks that get removed.
  'spike-repair': {
    parameters: [
      {id: 'repaired', label: 'Repaired', min: 0, max: 100, step: 5, default: 100, unit: '%'}
    ],
    pointer: 'none',
    build: context => {
      const {p, fork} = context;
      const random = fork('spikes');
      const left = 8;
      const right = 152;
      const step = 2;
      const topBase = 30;
      const bottomBase = 74;
      const getWave = (x: number) => Math.sin(x * 0.06) * 5 + Math.sin(x * 0.17 + 1) * 2.5;
      const spikeCount = 7;
      const spikeAt = new Map<number, number>();
      for (let index = 0; index < spikeCount; index++) {
        const x = Math.round((16 + index * 20.5 + random.range(-3, 3) - left) / step) * step + left;
        const isPit = index % 3 === 1;
        spikeAt.set(x, isPit ? -random.range(11, 14) : random.range(15, 19));
      }
      const spikeXs = [...spikeAt.keys()].sort((first, second) => first - second);
      const repairedCount = Math.round((p.repaired / 100) * spikeCount);
      const repairedXs = new Set(spikeXs.slice(0, repairedCount));
      const before: Point[] = [];
      const after: Point[] = [];
      const median: Point[] = [];
      const tipsBefore: SpriteItem[] = [];
      const clamped: SpriteItem[] = [];
      const leftover: SpriteItem[] = [];
      const connectors: PathItem[] = [];
      for (let x = left; x <= right; x += step) {
        const wave = getWave(x);
        const spike = spikeAt.get(x) ?? 0;
        before.push([x, topBase - wave - spike]);
        median.push([x, topBase - wave]);
        const isRepaired = repairedXs.has(x);
        after.push([x, bottomBase - wave - (isRepaired ? 0 : spike)]);
        if (spike === 0) continue;
        tipsBefore.push({x, y: topBase - wave - spike, radius: 3, role: 'hot', alpha: 1});
        connectors.push({
          points: [
            [x, topBase - wave - spike],
            [x, topBase - wave]
          ],
          role: 'hot',
          width: 0.6,
          alpha: 0.7,
          phase: x / right
        });
        if (isRepaired) {
          clamped.push({
            x,
            y: bottomBase - wave,
            radius: 2.6,
            role: 'cool',
            alpha: 1,
            phase: x / right
          });
          connectors.push({
            points: [
              [x, topBase - wave + 5],
              [x, bottomBase - wave - 8]
            ],
            role: 'warm',
            width: 0.7,
            alpha: 0.85,
            phase: x / right
          });
        } else {
          leftover.push({x, y: bottomBase - wave - spike, radius: 3, role: 'hot', alpha: 1});
        }
      }
      const toBody = (line: Point[], base: number): PolygonItem => ({
        ring: [...line, [right, base + 20], [left, base + 20]],
        role: 'cool',
        alpha: 0.14
      });
      return [
        createFillLayer([toBody(median, topBase), toBody(after, bottomBase)], {blend: 'add'}),
        createStrokeLayer(
          [
            {points: median, role: 'cool', width: 0.5, alpha: 0.6},
            {points: before, role: 'hot', width: 0.9, alpha: 0.95}
          ],
          {animation: 'static', blend: 'add'}
        ),
        createStrokeLayer([{points: after, role: 'primary', width: 1.1, alpha: 1}], {
          animation: 'flow',
          speed: 0.25,
          blend: 'add'
        }),
        createStrokeLayer(connectors, {animation: 'flow', speed: 0.5, blend: 'add'}),
        createSpriteLayer(tipsBefore, {pulse: 0.4}),
        createSpriteLayer(leftover, {pulse: 0.5}),
        createSpriteLayer(clamped, {pulse: 0.25})
      ];
    }
  }
};
