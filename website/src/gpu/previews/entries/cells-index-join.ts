// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/cells-index-join.ts, keyed by entry id.

import {
  FIELD_COLUMNS,
  FIELD_ROWS,
  clamp,
  createBlob,
  createCircle,
  createClusterPoints,
  createRegularPolygon,
  createScatterPoints,
  createSmoothField,
  getDistance,
  mixNumbers,
  type Random
} from '../data.ts';
import {
  clampPoint,
  getClassRole,
  getClusterCenters,
  getRoleCycle,
  getSegmentDistance,
  smoothPath,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
import {
  closeRing,
  createRectangleRing,
  getCross,
  getSignedArea,
  getVertexMean,
  isInsideRing
} from '../geometry.ts';
import {
  getVectorCross,
  getVectorDot,
  projectGlobe as projectGlobeWithDepth,
  toLonLat,
  toVector,
  type GlobeView
} from '../globe.ts';
import {
  createChoice,
  createFillLayer,
  createSlider,
  createSpriteLayer,
  createStrokeLayer
} from '../layers.ts';
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

const SQRT3 = Math.sqrt(3);
const DEGREES = Math.PI / 180;

// ---------------------------------------------------------------------------------------------
// Hexagon lattice (pointy top, odd rows shifted right), the stand-in for H3 cells.
// ---------------------------------------------------------------------------------------------

type HexCell = {row: number; column: number; x: number; y: number};

function getHexCenter(row: number, column: number, radius: number): HexCell {
  const width = SQRT3 * radius;
  return {
    row,
    column,
    x: column * width + (Math.abs(row) % 2 ? width / 2 : 0),
    y: row * radius * 1.5
  };
}

function getHexCell(point: Point, radius: number): HexCell {
  const width = SQRT3 * radius;
  const baseRow = Math.round(point[1] / (radius * 1.5));
  let best = getHexCenter(baseRow, Math.round(point[0] / width), radius);
  let bestDistance = Infinity;
  for (let row = baseRow - 1; row <= baseRow + 1; row++) {
    const shift = Math.abs(row) % 2 ? width / 2 : 0;
    const cell = getHexCenter(row, Math.round((point[0] - shift) / width), radius);
    const distance = Math.hypot(cell.x - point[0], cell.y - point[1]);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = cell;
    }
  }
  return best;
}

function getHexCells(radius: number, extra = 0): HexCell[] {
  const width = SQRT3 * radius;
  const cells: HexCell[] = [];
  const rowCount = Math.ceil(SCENE_HEIGHT / (radius * 1.5)) + 1;
  const columnStart = -1 - Math.ceil(extra / width);
  const columnEnd = Math.ceil(SCENE_WIDTH / width) + 1 + Math.ceil(extra / width);
  for (let row = -1; row < rowCount; row++) {
    for (let column = columnStart; column < columnEnd; column++) {
      cells.push(getHexCenter(row, column, radius));
    }
  }
  return cells;
}

function getHexKey(cell: HexCell): string {
  return `${cell.row},${cell.column}`;
}

function getHexRing(cell: HexCell, radius: number, scale = 0.93): Point[] {
  return createRegularPolygon(cell.x, cell.y, radius * scale, 6, Math.PI / 6);
}

function getHexDistance(first: HexCell, second: HexCell): number {
  const toCube = (cell: HexCell) => {
    const x = cell.column - (cell.row - (cell.row & 1)) / 2;
    return [x, -x - cell.row, cell.row];
  };
  const firstCube = toCube(first);
  const secondCube = toCube(second);
  return Math.max(
    Math.abs(firstCube[0] - secondCube[0]),
    Math.abs(firstCube[1] - secondCube[1]),
    Math.abs(firstCube[2] - secondCube[2])
  );
}

function createHexOutlines(
  cells: HexCell[],
  radius: number,
  alpha = 0.28,
  role: Role = 'dim',
  width = 0.25
): PreviewLayer {
  return createStrokeLayer(
    cells.map(cell => ({
      points: closeRing(getHexRing(cell, radius)),
      role,
      width,
      alpha
    })),
    {animation: 'static'}
  );
}

// ---------------------------------------------------------------------------------------------
// Planar geometry
// ---------------------------------------------------------------------------------------------

function doSegmentsIntersect(
  firstStart: Point,
  firstEnd: Point,
  secondStart: Point,
  secondEnd: Point
): boolean {
  return (
    getCross(secondStart, secondEnd, firstStart) * getCross(secondStart, secondEnd, firstEnd) < 0 &&
    getCross(firstStart, firstEnd, secondStart) * getCross(firstStart, firstEnd, secondEnd) < 0
  );
}

type Box = {minX: number; minY: number; maxX: number; maxY: number};

function getBounds(points: Point[]): Box {
  const box = {minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity};
  for (const [x, y] of points) {
    box.minX = Math.min(box.minX, x);
    box.minY = Math.min(box.minY, y);
    box.maxX = Math.max(box.maxX, x);
    box.maxY = Math.max(box.maxY, y);
  }
  return box;
}

function getBoxRing(box: Box): Point[] {
  return closeRing(createRectangleRing(box.minX, box.minY, box.maxX, box.maxY));
}

function expandBox(box: Box, distance: number): Box {
  return {
    minX: box.minX - distance,
    minY: box.minY - distance,
    maxX: box.maxX + distance,
    maxY: box.maxY + distance
  };
}

function doBoxesOverlap(first: Box, second: Box): boolean {
  return (
    first.minX <= second.maxX &&
    second.minX <= first.maxX &&
    first.minY <= second.maxY &&
    second.minY <= first.maxY
  );
}

function isInsideBox([x, y]: Point, box: Box): boolean {
  return x >= box.minX && x <= box.maxX && y >= box.minY && y <= box.maxY;
}

/** True when any polygon edge crosses the boundary of the box or a polygon vertex lies inside it. */
function doesPolygonTouchBoxEdges(polygon: Point[], box: Box): boolean {
  const ring = getBoxRing(box);
  for (let index = 0; index < polygon.length; index++) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    if (isInsideBox(start, box)) return true;
    for (let side = 0; side < 4; side++) {
      if (doSegmentsIntersect(start, end, ring[side], ring[side + 1])) return true;
    }
  }
  return false;
}

function createSquareRing(x: number, y: number, size: number): Point[] {
  return closeRing(createRectangleRing(x, y, x + size, y + size));
}

function createClusterPointsIn(
  random: Random,
  centers: Point[],
  count: number,
  spread: number,
  width: number,
  height: number
): Point[] {
  const points: Point[] = [];
  for (let index = 0; index < count; index++) {
    const center = centers[index % centers.length];
    points.push([
      clamp(center[0] + random.gaussian() * spread, 1, width - 1),
      clamp(center[1] + random.gaussian() * spread * 0.8, 1, height - 1)
    ]);
  }
  return points;
}

// ---------------------------------------------------------------------------------------------
// Globe (orthographic view) for the S2 and spherical entries
// ---------------------------------------------------------------------------------------------

/** Visible points only: null once the point is within a hair of the horizon. */
function projectVisible(view: GlobeView, longitude: number, latitude: number): Point | null {
  const {point, depth} = projectGlobeWithDepth(view, longitude, latitude);
  return depth < 0.02 ? null : point;
}

function unprojectGlobe(view: GlobeView, point: Point): Point | null {
  const x = (point[0] - view.centerX) / view.radius;
  const y = -(point[1] - view.centerY) / view.radius;
  const rho = Math.hypot(x, y);
  if (rho > 1) return null;
  if (rho < 1e-6) return [view.longitude, view.latitude];
  const angle = Math.asin(rho);
  const phiCenter = view.latitude * DEGREES;
  const latitude = Math.asin(
    Math.cos(angle) * Math.sin(phiCenter) + (y * Math.sin(angle) * Math.cos(phiCenter)) / rho
  );
  const longitude =
    view.longitude +
    Math.atan2(
      x * Math.sin(angle),
      rho * Math.cos(phiCenter) * Math.cos(angle) - y * Math.sin(phiCenter) * Math.sin(angle)
    ) /
      DEGREES;
  return [longitude, latitude / DEGREES];
}

/** Longitude/latitude points to screen runs; hidden points split the run. */
function projectLonLatPath(view: GlobeView, points: Point[]): Point[][] {
  const runs: Point[][] = [];
  let current: Point[] = [];
  for (const [longitude, latitude] of points) {
    const projected = projectVisible(view, longitude, latitude);
    if (projected) current.push(projected);
    else if (current.length > 0) {
      runs.push(current);
      current = [];
    }
  }
  if (current.length > 0) runs.push(current);
  return runs.filter(run => run.length > 1);
}

function projectLonLatRing(view: GlobeView, points: Point[]): Point[] | null {
  const projected = points.map(([longitude, latitude]) =>
    projectVisible(view, longitude, latitude)
  );
  return projected.every(point => point) ? (projected as Point[]) : null;
}

function createGlobeBase(view: GlobeView): PreviewLayer[] {
  const graticule: PathItem[] = [];
  for (let longitude = -180; longitude < 180; longitude += 30) {
    const line: Point[] = [];
    for (let latitude = -84; latitude <= 84; latitude += 6) line.push([longitude, latitude]);
    for (const run of projectLonLatPath(view, line)) {
      graticule.push({points: run, role: 'dim', width: 0.22, alpha: 0.4});
    }
  }
  for (let latitude = -60; latitude <= 60; latitude += 30) {
    const line: Point[] = [];
    for (let longitude = -180; longitude <= 180; longitude += 6) line.push([longitude, latitude]);
    for (const run of projectLonLatPath(view, line)) {
      graticule.push({points: run, role: 'dim', width: 0.22, alpha: 0.4});
    }
  }
  return [
    createFillLayer(
      [{ring: createCircle(view.centerX, view.centerY, view.radius, 56), role: 'dim', alpha: 0.12}],
      {animation: 'static'}
    ),
    createStrokeLayer(graticule, {animation: 'static'}),
    createStrokeLayer(
      [
        {
          points: closeRing(createCircle(view.centerX, view.centerY, view.radius, 56)),
          role: 'cool',
          width: 0.5,
          alpha: 0.7
        }
      ],
      {animation: 'static'}
    )
  ];
}

function getAngularDistance(first: Point, second: Point): number {
  return (
    Math.acos(
      clamp(getVectorDot(toVector(first[0], first[1]), toVector(second[0], second[1])), -1, 1)
    ) / DEGREES
  );
}

/** Great-circle arc between two longitude/latitude points, sampled. */
function sampleArc(first: Point, second: Point, steps: number): Point[] {
  const firstVector = toVector(first[0], first[1]);
  const secondVector = toVector(second[0], second[1]);
  const omega = Math.acos(clamp(getVectorDot(firstVector, secondVector), -1, 1));
  const sine = Math.sin(omega) || 1;
  return Array.from({length: steps + 1}, (_, index) => {
    const amount = index / steps;
    const firstWeight = Math.sin((1 - amount) * omega) / sine;
    const secondWeight = Math.sin(amount * omega) / sine;
    return toLonLat([
      firstWeight * firstVector[0] + secondWeight * secondVector[0],
      firstWeight * firstVector[1] + secondWeight * secondVector[1],
      firstWeight * firstVector[2] + secondWeight * secondVector[2]
    ]);
  });
}

function sampleArcRing(vertices: Point[], steps: number): Point[] {
  const ring: Point[] = [];
  vertices.forEach((vertex, index) => {
    ring.push(...sampleArc(vertex, vertices[(index + 1) % vertices.length], steps).slice(0, -1));
  });
  return ring;
}

function createGlobeView(
  pointer: Point | null,
  centerX: number,
  centerY: number,
  radius: number,
  steer: boolean
): GlobeView {
  const view: GlobeView = {centerX, centerY, radius, longitude: 0, latitude: 20};
  if (steer && pointer) {
    view.longitude = clamp((pointer[0] - centerX) / radius, -1.2, 1.2) * 45;
    view.latitude = 20 - clamp((pointer[1] - centerY) / radius, -1, 1) * 25;
  }
  return view;
}

function clipConvex(subject: Point[], clip: Point[]): Point[] {
  // Sutherland-Hodgman against a convex counter-clockwise (or clockwise) clip polygon.
  const orientation = Math.sign(
    clip.reduce((sum, point, index) => {
      const next = clip[(index + 1) % clip.length];
      return sum + point[0] * next[1] - next[0] * point[1];
    }, 0)
  );
  let output = subject;
  for (let index = 0; index < clip.length && output.length > 0; index++) {
    const start = clip[index];
    const end = clip[(index + 1) % clip.length];
    const input = output;
    output = [];
    const isInside = (point: Point) => getCross(start, end, point) * orientation >= 0;
    for (let cursor = 0; cursor < input.length; cursor++) {
      const current = input[cursor];
      const previous = input[(cursor + input.length - 1) % input.length];
      if (isInside(current) !== isInside(previous)) {
        const amount =
          getCross(start, end, previous) /
          (getCross(start, end, previous) - getCross(start, end, current));
        output.push([
          previous[0] + (current[0] - previous[0]) * amount,
          previous[1] + (current[1] - previous[1]) * amount
        ]);
      }
      if (isInside(current)) output.push(current);
    }
  }
  return output;
}

function createLonLatPolygon(
  random: Random,
  longitude: number,
  latitude: number,
  radius: number,
  count: number,
  jitter: number
): Point[] {
  return Array.from({length: count}, (_, index) => {
    const angle = (index / count) * Math.PI * 2 + 0.4;
    const scale = radius * (1 + random.range(-jitter, jitter));
    return [longitude + Math.cos(angle) * scale, latitude + Math.sin(angle) * scale];
  });
}

// ---------------------------------------------------------------------------------------------
// Hilbert curve
// ---------------------------------------------------------------------------------------------

function getHilbertCell(order: number, index: number): Point {
  const side = 2 ** order;
  let x = 0;
  let y = 0;
  let remainder = index;
  for (let step = 1; step < side; step *= 2) {
    const rx = 1 & Math.floor(remainder / 2);
    const ry = 1 & (remainder ^ rx);
    if (ry === 0) {
      if (rx === 1) {
        x = step - 1 - x;
        y = step - 1 - y;
      }
      const swap = x;
      x = y;
      y = swap;
    }
    x += step * rx;
    y += step * ry;
    remainder = Math.floor(remainder / 4);
  }
  return [x, y];
}

// ---------------------------------------------------------------------------------------------
// Spatial join helpers
// ---------------------------------------------------------------------------------------------

/** Jittered quad lattice standing in for census tracts: shared vertices, no gaps. */
function createTractQuads(random: Random, columns: number, rows: number): Point[][] {
  const left = 6;
  const top = 6;
  const cellWidth = (SCENE_WIDTH - 2 * left) / columns;
  const cellHeight = (SCENE_HEIGHT - 2 * top) / rows;
  const vertices: Point[][] = [];
  for (let row = 0; row <= rows; row++) {
    const line: Point[] = [];
    for (let column = 0; column <= columns; column++) {
      const edge = row === 0 || row === rows || column === 0 || column === columns;
      line.push([
        left + column * cellWidth + (edge ? 0 : random.range(-4, 4)),
        top + row * cellHeight + (edge ? 0 : random.range(-4, 4))
      ]);
    }
    vertices.push(line);
  }
  const quads: Point[][] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      quads.push([
        vertices[row][column],
        vertices[row][column + 1],
        vertices[row + 1][column + 1],
        vertices[row + 1][column]
      ]);
    }
  }
  return quads;
}

function getDistanceToRing(point: Point, ring: Point[]): number {
  if (isInsideRing(point, ring)) return 0;
  let best = Infinity;
  ring.forEach((start, index) => {
    best = Math.min(best, getSegmentDistance(point, start, ring[(index + 1) % ring.length]));
  });
  return best;
}

// ---------------------------------------------------------------------------------------------
// Square map shared by the pyramid and level selection entries
// ---------------------------------------------------------------------------------------------

function createSquareCountLayers(
  points: Point[],
  originX: number,
  originY: number,
  width: number,
  height: number,
  cellSize: number,
  role: Role
): PreviewLayer[] {
  const columns = Math.ceil(width / cellSize);
  const rows = Math.ceil(height / cellSize);
  const counts = new Map<number, number>();
  let maximum = 1;
  for (const [x, y] of points) {
    const key =
      Math.min(rows - 1, Math.floor(y / cellSize)) * columns +
      Math.min(columns - 1, Math.floor(x / cellSize));
    const count = (counts.get(key) ?? 0) + 1;
    counts.set(key, count);
    maximum = Math.max(maximum, count);
  }
  const fills: PolygonItem[] = [];
  const outlines: PathItem[] = [];
  for (const [key, count] of counts) {
    const column = key % columns;
    const row = Math.floor(key / columns);
    const x = originX + column * cellSize;
    const y = originY + row * cellSize;
    const ring: Point[] = [
      [x + 0.3, y + 0.3],
      [Math.min(x + cellSize, originX + width) - 0.3, y + 0.3],
      [
        Math.min(x + cellSize, originX + width) - 0.3,
        Math.min(y + cellSize, originY + height) - 0.3
      ],
      [x + 0.3, Math.min(y + cellSize, originY + height) - 0.3]
    ];
    fills.push({
      ring,
      role,
      alpha: 0.2 + 0.6 * (count / maximum),
      phase: (column + row) / (columns + rows)
    });
    outlines.push({points: closeRing(ring), role, width: 0.3, alpha: 0.6});
  }
  return [
    createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
    createStrokeLayer(outlines, {animation: 'static'})
  ];
}

export const CELLS_INDEX_JOIN_PREVIEWS: Record<string, EntryPreview> = {
  // -------------------------------------------------------------------------------------------
  // H3 and Quadbin cell tables
  // -------------------------------------------------------------------------------------------

  'point-to-cell': {
    parameters: [
      createSlider('size', 'Cell size', 4, 14, 0.5, 8, 'px'),
      createSlider('count', 'Points', 20, 120, 4, 56)
    ],
    build: ({fork, p}) => {
      const radius = p.size;
      const points = createClusterPoints(
        fork('points'),
        getClusterCenters(fork('centers')),
        Math.max(1, Math.round(p.count / 4)),
        9
      );
      const lit = new Map<string, HexCell>();
      const links: PathItem[] = [];
      points.forEach((point, index) => {
        const cell = getHexCell(point, radius);
        lit.set(getHexKey(cell), cell);
        links.push({
          points: [point, [cell.x, cell.y]],
          role: 'warm',
          width: 0.45,
          alpha: 0.8,
          phase: index / points.length
        });
      });
      const cells = [...lit.values()];
      return [
        createHexOutlines(getHexCells(radius), radius),
        createFillLayer(
          cells.map(cell => ({
            ring: getHexRing(cell, radius),
            role: 'primary' as Role,
            alpha: 0.35,
            phase: (cell.x + cell.y) / 260
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(links, {animation: 'flow', speed: 0.35, trail: 0.7, blend: 'add'}),
        createSpriteLayer(
          cells.map(cell => ({x: cell.x, y: cell.y, radius: 1.3, role: 'primary' as Role})),
          {pulse: 0.2}
        ),
        createSpriteLayer(toSprites(points, 'ink', 0.9, undefined, 0.8))
      ];
    }
  },

  'cell-geometry': {
    parameters: [
      createSlider('size', 'Cell size', 8, 18, 0.5, 13, 'px'),
      createSlider('cells', 'Keys', 1, 8, 1, 5)
    ],
    build: ({fork, p}) => {
      const radius = p.size;
      const count = Math.round(p.cells);
      const cells = getHexCells(radius)
        .filter(cell => cell.y > 28 && cell.y < 92)
        .sort(
          (first, second) =>
            getDistance([first.x, first.y], [80, 62]) - getDistance([second.x, second.y], [80, 62])
        )
        .slice(0, count)
        .sort((first, second) => first.x - second.x || first.y - second.y);
      const bits = fork('bits');
      const keyBits: SpriteItem[] = Array.from({length: 24}, (_, index) => {
        const set = bits.next() > 0.5;
        return {
          x: 28 + index * 4.6,
          y: 9,
          radius: 1.1,
          role: set ? 'hot' : 'dim',
          alpha: set ? 0.95 : 0.5,
          phase: index / 24
        };
      });
      const wires: PathItem[] = cells.map((cell, index) => {
        const startX = 28 + ((index + 0.5) / count) * 24 * 4.6;
        return {
          points: [
            [startX, 12],
            [startX, 22],
            [cell.x, cell.y]
          ],
          role: 'primary',
          width: 0.5,
          alpha: 0.8,
          phase: index / count
        };
      });
      const corners: SpriteItem[] = cells.flatMap(cell =>
        getHexRing(cell, radius, 1).map(([x, y]) => ({x, y, radius: 0.9, role: 'ink' as Role}))
      );
      return [
        createHexOutlines(getHexCells(radius), radius, 0.14),
        createFillLayer(
          cells.map((cell, index) => ({
            ring: getHexRing(cell, radius),
            role: getRoleCycle(index),
            alpha: 0.2,
            phase: index / count
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(wires, {animation: 'flow', speed: 0.3, blend: 'add'}),
        createStrokeLayer(
          cells.map((cell, index) => ({
            points: closeRing(getHexRing(cell, radius, 1)),
            role: getRoleCycle(index),
            width: 0.8,
            alpha: 0.95,
            phase: index / count
          })),
          {animation: 'draw', speed: 0.25, blend: 'add'}
        ),
        createSpriteLayer(keyBits, {pulse: 0.2}),
        createSpriteLayer(corners),
        createSpriteLayer(
          cells.map(cell => ({x: cell.x, y: cell.y, radius: 2, role: 'warm' as Role})),
          {pulse: 0.3}
        )
      ];
    }
  },

  'cell-measures': {
    parameters: [
      createSlider('size', 'Cell size', 5, 12, 0.5, 9, 'px'),
      createSlider('classes', 'Area classes', 2, 5, 1, 5)
    ],
    build: ({p}) => {
      const radius = p.size;
      const classes = Math.round(p.classes);
      const warp = ([x, y]: Point): Point => [80 + (x - 80) * (0.62 + (0.7 * y) / SCENE_HEIGHT), y];
      const pentagonCell = getHexCell([102, 38], radius);
      const fills: PolygonItem[] = [];
      for (const cell of getHexCells(radius, 70)) {
        if (cell.row === pentagonCell.row && cell.column === pentagonCell.column) continue;
        const ring = getHexRing(cell, radius).map(warp);
        const value = clamp(cell.y / SCENE_HEIGHT, 0, 1);
        fills.push({
          ring,
          role: getClassRole(Math.min(classes - 1, Math.floor(value * classes)), classes),
          alpha: 0.1 + 0.28 * value,
          phase: cell.x / 160
        });
      }
      const pentagon = createRegularPolygon(
        pentagonCell.x,
        pentagonCell.y,
        radius * 0.95,
        5,
        -Math.PI / 2
      ).map(warp);
      return [
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(
          fills.map(item => ({
            points: closeRing(item.ring),
            role: item.role,
            width: 0.3,
            alpha: 0.3
          })),
          {animation: 'static'}
        ),
        createFillLayer([{ring: pentagon, role: 'violet', alpha: 0.75}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createStrokeLayer([{points: closeRing(pentagon), role: 'violet', width: 1, alpha: 1}], {
          animation: 'flow',
          speed: 0.4,
          blend: 'add'
        })
      ];
    }
  },

  'h3-cell-identities': {
    pointer: 'sticky',
    parameters: [createSlider('size', 'Cell size', 10, 22, 0.5, 15, 'px')],
    build: ({p, pointer}) => {
      const radius = p.size;
      const focus = getHexCell(pointer ?? [80, 50], radius);
      const ring = getHexRing(focus, radius, 1);
      const directed: PathItem[] = ring.map((vertex, index) => ({
        points: [vertex, ring[(index + 1) % 6]],
        role: 'warm',
        width: 1.4,
        alpha: 1,
        phase: index / 6
      }));
      const spokes: PathItem[] = ring.map((vertex, index) => {
        const next = ring[(index + 1) % 6];
        return {
          points: [
            [focus.x, focus.y],
            [(vertex[0] + next[0]) / 2, (vertex[1] + next[1]) / 2]
          ],
          role: 'cool',
          width: 0.4,
          alpha: 0.6
        };
      });
      return [
        createHexOutlines(getHexCells(radius), radius, 0.3),
        createFillLayer([{ring: getHexRing(focus, radius), role: 'primary', alpha: 0.28}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createStrokeLayer(spokes, {animation: 'static'}),
        createStrokeLayer(directed, {animation: 'flow', speed: 0.5, trail: 0.8, blend: 'add'}),
        createSpriteLayer(
          ring.map(([x, y], index) => ({
            x,
            y,
            radius: 1.9,
            role: 'violet' as Role,
            phase: index / 6
          })),
          {pulse: 0.3}
        ),
        createSpriteLayer([{x: focus.x, y: focus.y, radius: 2.4, role: 'hot'}], {pulse: 0.3})
      ];
    }
  },

  'cell-aggregation': {
    parameters: [
      createSlider('size', 'Cell size', 5, 14, 0.5, 8, 'px'),
      createSlider('count', 'Points', 30, 240, 10, 110)
    ],
    build: ({fork, p}) => {
      const radius = p.size;
      const centers = getClusterCenters(fork('centers')).map<Point>(([x, y]) => [x * 0.68 + 8, y]);
      const points = createClusterPoints(fork('points'), centers, Math.round(p.count / 4), 7);
      const counts = new Map<string, {cell: HexCell; count: number}>();
      for (const point of points) {
        const cell = getHexCell(point, radius);
        const entry = counts.get(getHexKey(cell)) ?? {cell, count: 0};
        entry.count += 1;
        counts.set(getHexKey(cell), entry);
      }
      const rows = [...counts.values()].sort((first, second) => second.count - first.count);
      const maximum = rows[0]?.count ?? 1;
      const roleOf = (count: number) =>
        getClassRole(Math.min(4, Math.floor((count / maximum) * 5 - 0.01)), 5);
      const bars = rows.slice(0, 9);
      return [
        createHexOutlines(
          getHexCells(radius).filter(cell => cell.x < 118),
          radius,
          0.2
        ),
        createFillLayer(
          rows.map(({cell, count}) => ({
            ring: getHexRing(cell, radius),
            role: roleOf(count),
            alpha: 0.2 + 0.6 * (count / maximum),
            phase: cell.x / 160
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createSpriteLayer(toSprites(points, 'ink', 0.7, undefined, 0.55)),
        createStrokeLayer(
          [
            {
              points: [
                [124, 6],
                [124, 94]
              ],
              role: 'dim',
              width: 0.3,
              alpha: 0.5
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(
          bars.map(({count}, index) => ({
            points: [
              [128, 12 + index * 9],
              [128 + 26 * (count / maximum), 12 + index * 9]
            ],
            role: roleOf(count),
            width: 3.2,
            alpha: 0.9,
            phase: index / bars.length
          })),
          {animation: 'flow', speed: 0.3, trail: 1.4, blend: 'add'}
        )
      ];
    }
  },

  'cell-rollup': {
    parameters: [createSlider('contrast', 'Contrast', 0.5, 2.5, 0.1, 1)],
    build: ({fork, p}) => {
      const random = fork('values');
      const childSize = 12;
      const parentSize = 30;
      const fills: PolygonItem[] = [];
      const parentFills: PolygonItem[] = [];
      const frames: PathItem[] = [];
      const links: PathItem[] = [];
      for (let unit = 0; unit < 4; unit++) {
        const originX = 6 + (unit % 2) * 78;
        const originY = 6 + Math.floor(unit / 2) * 48;
        let sum = 0;
        for (let child = 0; child < 4; child++) {
          const value = random.next();
          sum += value;
          const x = originX + (child % 2) * childSize;
          const y = originY + 9 + Math.floor(child / 2) * childSize;
          fills.push({
            ring: [
              [x + 0.8, y + 0.8],
              [x + childSize - 0.8, y + 0.8],
              [x + childSize - 0.8, y + childSize - 0.8],
              [x + 0.8, y + childSize - 0.8]
            ],
            role: 'cool',
            alpha: 0.1 + 0.65 * value ** p.contrast,
            phase: (unit + child) / 8
          });
        }
        frames.push({
          points: createSquareRing(originX, originY + 9, 2 * childSize),
          role: 'primary',
          width: 0.6,
          alpha: 0.9
        });
        const parentX = originX + 2 * childSize + 14;
        parentFills.push({
          ring: [
            [parentX, originY + 6],
            [parentX + parentSize, originY + 6],
            [parentX + parentSize, originY + 6 + parentSize],
            [parentX, originY + 6 + parentSize]
          ],
          role: 'hot',
          alpha: 0.1 + 0.7 * (sum / 4) ** p.contrast,
          phase: unit / 4
        });
        frames.push({
          points: createSquareRing(parentX, originY + 6, parentSize),
          role: 'hot',
          width: 0.7,
          alpha: 0.9
        });
        links.push({
          points: [
            [originX + 2 * childSize + 1, originY + 9 + childSize],
            [parentX - 1, originY + 6 + parentSize / 2]
          ],
          role: 'warm',
          width: 0.9,
          alpha: 0.95,
          phase: unit / 4
        });
      }
      return [
        createFillLayer([...fills, ...parentFills], {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(frames, {animation: 'static'}),
        createStrokeLayer(links, {animation: 'flow', speed: 0.35, trail: 1.2, blend: 'add'})
      ];
    }
  },

  'cell-pyramid': {
    parameters: [createSlider('points', 'Points', 30, 220, 10, 100)],
    build: ({fork, p}) => {
      const panelWidth = 72;
      const panelHeight = 44;
      const points = createClusterPointsIn(
        fork('points'),
        [
          [18, 14],
          [48, 28],
          [56, 10]
        ],
        Math.round(p.points),
        7,
        panelWidth,
        panelHeight
      );
      const layers: PreviewLayer[] = [];
      const frames: PathItem[] = [];
      const dots: SpriteItem[] = [];
      for (let level = 0; level < 4; level++) {
        const originX = 5 + (level % 2) * 78;
        const originY = 4 + Math.floor(level / 2) * 48;
        layers.push(
          ...createSquareCountLayers(
            points,
            originX,
            originY,
            panelWidth,
            panelHeight,
            panelWidth / 2 ** (level + 1),
            getClassRole(level, 4)
          )
        );
        frames.push({
          points: closeRing([
            [originX, originY],
            [originX + panelWidth, originY],
            [originX + panelWidth, originY + panelHeight],
            [originX, originY + panelHeight]
          ]),
          role: 'dim',
          width: 0.4,
          alpha: 0.6
        });
        for (const [x, y] of points)
          dots.push({x: originX + x, y: originY + y, radius: 0.5, role: 'ink', alpha: 0.7});
      }
      return [...layers, createStrokeLayer(frames, {animation: 'static'}), createSpriteLayer(dots)];
    }
  },

  'cell-level-selection': {
    parameters: [
      createChoice('level', 'Level', ['L1', 'L2', 'L3', 'L4'], 1),
      createSlider('points', 'Points', 40, 300, 10, 140)
    ],
    build: ({fork, p}) => {
      const width = 104;
      const height = 92;
      const level = Math.round(p.level);
      const points = createClusterPointsIn(
        fork('points'),
        [
          [28, 26],
          [70, 60],
          [74, 22],
          [34, 70]
        ],
        Math.round(p.points),
        9,
        width,
        height
      );
      const layers = createSquareCountLayers(
        points,
        8,
        4,
        width,
        height,
        width / 2 ** (level + 2),
        getClassRole(level, 4)
      );
      const pipSizes = [22, 15, 10, 6];
      const pips: PolygonItem[] = [];
      const pipOutlines: PathItem[] = [];
      let top = 8;
      let selectedCenter: Point = [138, 50];
      pipSizes.forEach((size, index) => {
        const x = 124 + (22 - size) / 2;
        const ring = createSquareRing(x, top, size);
        pipOutlines.push({
          points: ring,
          role: index === level ? 'warm' : 'dim',
          width: index === level ? 0.9 : 0.4,
          alpha: index === level ? 1 : 0.6
        });
        if (index === level) {
          pips.push({ring: ring.slice(0, 4), role: 'warm', alpha: 0.6});
          selectedCenter = [x, top + size / 2];
        }
        top += size + 8;
      });
      return [
        ...layers,
        createStrokeLayer(pipOutlines, {animation: 'static'}),
        createFillLayer(pips, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(
          [
            {
              points: [selectedCenter, [112, selectedCenter[1]]],
              role: 'warm',
              width: 0.8,
              alpha: 0.9
            }
          ],
          {animation: 'flow', speed: 0.5, trail: 0.8, blend: 'add'}
        )
      ];
    }
  },

  'cell-table-compare': {
    parameters: [
      createSlider('size', 'Cell size', 5, 14, 0.5, 7.5, 'px'),
      createSlider('threshold', 'Change threshold', 0.05, 0.4, 0.01, 0.16)
    ],
    build: ({fork, p}) => {
      const radius = p.size;
      const before = createSmoothField(fork('before'), 5, 1.4);
      const other = createSmoothField(fork('after'), 5, 1.4);
      const sample = (values: Float32Array, cell: HexCell) =>
        values[
          clamp(Math.floor((cell.y / SCENE_HEIGHT) * FIELD_ROWS), 0, FIELD_ROWS - 1) *
            FIELD_COLUMNS +
            clamp(Math.floor((cell.x / SCENE_WIDTH) * FIELD_COLUMNS), 0, FIELD_COLUMNS - 1)
        ];
      const fills: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      for (const cell of getHexCells(radius)) {
        const beforeValue = sample(before, cell);
        const afterValue = 0.5 * beforeValue + 0.5 * sample(other, cell);
        // Outer join: a cell missing from one table counts as zero there.
        const presentBefore = beforeValue > 0.42;
        const presentAfter = afterValue > 0.42;
        if (!presentBefore && !presentAfter) continue;
        const difference = (presentAfter ? afterValue : 0) - (presentBefore ? beforeValue : 0);
        const role: Role =
          difference > p.threshold ? 'hot' : difference < -p.threshold ? 'cool' : 'dim';
        const ring = getHexRing(cell, radius);
        fills.push({
          ring,
          role,
          alpha: role === 'dim' ? 0.16 : 0.22 + Math.min(0.5, Math.abs(difference) * 1.4),
          phase: cell.x / 160
        });
        outlines.push({
          points: closeRing(ring),
          role,
          width: 0.3,
          alpha: role === 'dim' ? 0.3 : 0.7
        });
      }
      return [
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(outlines, {animation: 'static'})
      ];
    }
  },

  'cell-topology': {
    pointer: 'hover',
    parameters: [
      createSlider('size', 'Cell size', 5, 12, 0.5, 8, 'px'),
      createSlider('radius', 'Disk radius', 0, 5, 1, 2)
    ],
    build: ({p, pointer}) => {
      const radius = p.size;
      const center = getHexCell(pointer ?? [80, 50], radius);
      const cells = getHexCells(radius);
      const diskRoles: Role[] = ['hot', 'warm', 'violet', 'primary', 'cool', 'cool'];
      const fills: PolygonItem[] = [];
      const rim: PathItem[] = [];
      const quiet: PathItem[] = [];
      for (const cell of cells) {
        const distance = getHexDistance(center, cell);
        const ring = getHexRing(cell, radius);
        if (distance <= p.radius) {
          fills.push({
            ring,
            role: diskRoles[distance],
            alpha: 0.7 - distance * 0.1,
            phase: distance / 6
          });
          if (distance === Math.round(p.radius) && distance > 0) {
            rim.push({
              points: closeRing(getHexRing(cell, radius, 1)),
              role: 'warm',
              width: 0.8,
              alpha: 1,
              phase: Math.atan2(cell.y - center.y, cell.x - center.x) / 6.3
            });
          }
        } else {
          quiet.push({points: closeRing(ring), role: 'dim', width: 0.25, alpha: 0.28});
        }
      }
      return [
        createStrokeLayer(quiet, {animation: 'static'}),
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(rim, {animation: 'flow', speed: 0.4, trail: 0.6, blend: 'add'}),
        createSpriteLayer([{x: center.x, y: center.y, radius: 2.2, role: 'ink'}], {pulse: 0.3})
      ];
    }
  },

  'cell-compaction': {
    parameters: [createSlider('merge', 'Merge levels', 0, 3, 1, 2)],
    build: ({fork, p}) => {
      const fine = 5;
      const columns = 32;
      const rows = 20;
      const region = createBlob(fork('region'), 80, 50, 38, 14, 0.2, 0.9);
      const inside = new Uint8Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          inside[row * columns + column] = isInsideRing(
            [(column + 0.5) * fine, (row + 0.5) * fine],
            region
          )
            ? 1
            : 0;
        }
      }
      const covered = new Uint8Array(columns * rows);
      const fills: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      const emit = (column: number, row: number, span: number, level: number) => {
        const ring = createSquareRing(column * fine + 0.2, row * fine + 0.2, span * fine - 0.4);
        fills.push({
          ring: ring.slice(0, 4),
          role: getClassRole(level, 4),
          alpha: 0.25 + level * 0.1,
          phase: (column + row) / 50
        });
        outlines.push({
          points: ring,
          role: level === 0 ? 'dim' : 'ink',
          width: 0.3 + level * 0.15,
          alpha: 0.4 + level * 0.18
        });
      };
      for (let level = Math.round(p.merge); level >= 0; level--) {
        const span = 2 ** level;
        for (let row = 0; row + span <= rows; row += span) {
          for (let column = 0; column + span <= columns; column += span) {
            let complete = true;
            for (let dy = 0; dy < span && complete; dy++) {
              for (let dx = 0; dx < span; dx++) {
                const index = (row + dy) * columns + column + dx;
                if (!inside[index] || covered[index]) {
                  complete = false;
                  break;
                }
              }
            }
            if (!complete) continue;
            for (let dy = 0; dy < span; dy++)
              for (let dx = 0; dx < span; dx++) covered[(row + dy) * columns + column + dx] = 1;
            emit(column, row, span, level);
          }
        }
      }
      return [
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(outlines, {animation: 'static'}),
        createStrokeLayer([{points: closeRing(region), role: 'warm', width: 0.7, alpha: 0.8}], {
          animation: 'flow',
          speed: 0.2,
          blend: 'add'
        })
      ];
    }
  },

  'cell-grid-path': {
    pointer: 'hover',
    parameters: [createSlider('size', 'Cell size', 6, 14, 0.5, 9, 'px')],
    build: ({p, pointer}) => {
      const radius = p.size;
      const origin = getHexCell([24, 72], radius);
      const destination = getHexCell(pointer ?? [134, 28], radius);
      const steps = Math.max(1, getHexDistance(origin, destination));
      const chain = new Map<string, HexCell>();
      for (let index = 0; index <= steps; index++) {
        const amount = index / steps;
        const cell = getHexCell(
          [
            mixNumbers(origin.x, destination.x, amount),
            mixNumbers(origin.y, destination.y, amount)
          ],
          radius
        );
        chain.set(getHexKey(cell), cell);
      }
      const cells = [...chain.values()];
      return [
        createHexOutlines(getHexCells(radius), radius, 0.24),
        createFillLayer(
          cells.map((cell, index) => ({
            ring: getHexRing(cell, radius),
            role: 'warm' as Role,
            alpha: 0.38,
            phase: index / cells.length
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(
          [
            {
              points: cells.map<Point>(cell => [cell.x, cell.y]),
              role: 'warm',
              width: 1.1,
              alpha: 0.95
            }
          ],
          {animation: 'flow', speed: 0.35, trail: 1.2, blend: 'add'}
        ),
        createSpriteLayer(
          [
            {x: origin.x, y: origin.y, radius: 3.4, role: 'cool'},
            {x: destination.x, y: destination.y, radius: 3.4, role: 'hot', phase: 0.5}
          ],
          {pulse: 0.3}
        )
      ];
    }
  },

  'cell-cover': {
    parameters: [
      createSlider('resolution', 'Resolution', 3, 8, 0.5, 6.5),
      createSlider('vertices', 'Polygon vertices', 5, 20, 1, 11),
      createChoice('family', 'Family', ['H3 hex', 'Quadbin'])
    ],
    build: ({fork, p}) => {
      const polygon = createBlob(fork('polygon'), 80, 50, 34, Math.round(p.vertices), 0.3, 0.85);
      const radius = 22 / 1.35 ** (p.resolution - 3);
      const lit: PolygonItem[] = [];
      const quiet: PathItem[] = [];
      const centers: SpriteItem[] = [];
      const consider = (x: number, y: number, ring: Point[]) => {
        if (isInsideRing([x, y], polygon)) {
          lit.push({ring, role: 'primary', alpha: 0.5, phase: (x + y) / 260});
          centers.push({x, y, radius: 0.9, role: 'warm', phase: (x + y) / 200});
        } else {
          quiet.push({points: closeRing(ring), role: 'dim', width: 0.2, alpha: 0.33});
        }
      };
      if (Math.round(p.family) === 0) {
        for (const cell of getHexCells(radius))
          consider(cell.x, cell.y, getHexRing(cell, radius, 0.95));
      } else {
        const side = radius * 1.7;
        for (let y = 0; y < SCENE_HEIGHT; y += side) {
          for (let x = 0; x < SCENE_WIDTH; x += side) {
            consider(
              x + side / 2,
              y + side / 2,
              createSquareRing(x + 0.2, y + 0.2, side - 0.4).slice(0, 4)
            );
          }
        }
      }
      return [
        createStrokeLayer(quiet, {animation: 'static'}),
        createFillLayer(lit, {animation: 'pulse', blend: 'add'}),
        createSpriteLayer(centers, {pulse: 0.3}),
        createStrokeLayer([{points: closeRing(polygon), role: 'warm', width: 1.1, alpha: 0.95}], {
          animation: 'flow',
          speed: 0.25,
          trail: 1.5,
          blend: 'add'
        })
      ];
    }
  },

  'cell-set-outline': {
    parameters: [
      createSlider('size', 'Cell size', 5, 12, 0.5, 7.5, 'px'),
      createSlider('holes', 'Holes', 0, 3, 1, 1)
    ],
    build: ({fork, p}) => {
      const radius = p.size;
      const region = createBlob(fork('region'), 80, 50, 33, 12, 0.25, 0.9);
      const members = new Map<string, HexCell>();
      for (const cell of getHexCells(radius)) {
        if (isInsideRing([cell.x, cell.y], region)) members.set(getHexKey(cell), cell);
      }
      const width = SQRT3 * radius;
      const neighborKey = (cell: HexCell, side: number) => {
        const angle = (side + 1) * (Math.PI / 3);
        return getHexKey(
          getHexCell([cell.x + width * Math.cos(angle), cell.y + width * Math.sin(angle)], radius)
        );
      };
      const holeRandom = fork('holes');
      const interior = [...members.values()]
        .filter(cell => [0, 1, 2, 3, 4, 5].every(side => members.has(neighborKey(cell, side))))
        .map(cell => ({cell, key: holeRandom.next()}))
        .sort((first, second) => first.key - second.key)
        .slice(0, Math.round(p.holes));
      for (const {cell} of interior) members.delete(getHexKey(cell));
      const shared: PathItem[] = [];
      const boundary: PathItem[] = [];
      for (const cell of members.values()) {
        const ring = getHexRing(cell, radius, 1);
        for (let side = 0; side < 6; side++) {
          const edge: Point[] = [ring[side], ring[(side + 1) % 6]];
          if (members.has(neighborKey(cell, side)))
            shared.push({points: edge, role: 'dim', width: 0.3, alpha: 0.5});
          else
            boundary.push({
              points: edge,
              role: 'warm',
              width: 1.2,
              alpha: 1,
              phase: (cell.x + cell.y) / 200
            });
        }
      }
      return [
        createFillLayer(
          [...members.values()].map(cell => ({
            ring: getHexRing(cell, radius),
            role: 'primary' as Role,
            alpha: 0.14,
            phase: cell.x / 160
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(shared, {animation: 'static'}),
        createStrokeLayer(boundary, {animation: 'flow', speed: 0.3, trail: 1.5, blend: 'add'})
      ];
    }
  },

  // -------------------------------------------------------------------------------------------
  // S2 cells
  // -------------------------------------------------------------------------------------------

  's2-cell-hierarchy': {
    pointer: 'hover',
    parameters: [createSlider('depth', 'Levels', 1, 5, 1, 3)],
    build: ({p, pointer}) => {
      const face = {x: 36, y: 6, size: 88};
      const focus: Point = pointer
        ? [
            clamp(pointer[0], face.x + 1, face.x + face.size - 1),
            clamp(pointer[1], face.y + 1, face.y + face.size - 1)
          ]
        : [face.x + face.size * 0.62, face.y + face.size * 0.38];
      const depth = Math.round(p.depth);
      const fills: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      const links: PathItem[] = [];
      let cellX = face.x;
      let cellY = face.y;
      let size = face.size;
      for (let level = 1; level <= depth; level++) {
        const half = size / 2;
        const role = getClassRole(level - 1, Math.max(depth, 2));
        let nextX = cellX;
        let nextY = cellY;
        for (let child = 0; child < 4; child++) {
          const x = cellX + (child % 2) * half;
          const y = cellY + Math.floor(child / 2) * half;
          const selected =
            focus[0] >= x &&
            focus[0] < x + half + 0.001 &&
            focus[1] >= y &&
            focus[1] < y + half + 0.001;
          const ring = createSquareRing(x, y, half);
          fills.push({
            ring: ring.slice(0, 4),
            role,
            alpha: selected ? 0.34 : 0.1,
            phase: level / 6
          });
          outlines.push({points: ring, role, width: 0.7, alpha: 0.9});
          links.push({
            points: [
              [cellX + half, cellY + half],
              [x + half / 2, y + half / 2]
            ],
            role,
            width: 0.4,
            alpha: 0.35,
            phase: level / 6
          });
          if (selected && x !== nextX + 0.0001) {
            nextX = x;
            nextY = y;
          }
        }
        cellX = nextX;
        cellY = nextY;
        size = half;
      }
      return [
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(outlines, {animation: 'static'}),
        createStrokeLayer(links, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createSpriteLayer([{x: focus[0], y: focus[1], radius: 1.6, role: 'ink'}], {pulse: 0.3})
      ];
    }
  },

  's2-cell-measures': {
    pointer: 'hover',
    parameters: [createSlider('level', 'Level', 1, 3, 1, 2)],
    build: ({p, pointer}) => {
      const view = createGlobeView(pointer, 56, 50, 40, true);
      const span = 40 / 2 ** (Math.round(p.level) - 1);
      const fills: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      for (let latitude = -80; latitude < 80; latitude += span) {
        for (let longitude = -180; longitude < 180; longitude += span) {
          const corners: Point[] = [];
          const steps = 3;
          for (let step = 0; step < steps; step++)
            corners.push([longitude + (span * step) / steps, latitude]);
          for (let step = 0; step < steps; step++)
            corners.push([longitude + span, latitude + (span * step) / steps]);
          for (let step = 0; step < steps; step++)
            corners.push([longitude + span - (span * step) / steps, latitude + span]);
          for (let step = 0; step < steps; step++)
            corners.push([longitude, latitude + span - (span * step) / steps]);
          const ring = projectLonLatRing(view, corners);
          if (!ring) continue;
          const area = Math.cos((latitude + span / 2) * DEGREES);
          fills.push({
            ring,
            role: getClassRole(Math.min(4, Math.floor(area * 5)), 5),
            alpha: 0.06 + 0.3 * area,
            phase: area
          });
          outlines.push({points: closeRing(ring), role: 'ink', width: 0.25, alpha: 0.35});
        }
      }
      const level = Math.round(p.level);
      const areaBars: PathItem[] = [];
      const perimeterBars: PathItem[] = [];
      for (let index = 0; index < 4; index++) {
        const y = 18 + index * 20;
        const selected = index + 1 === level;
        areaBars.push({
          points: [
            [108, y],
            [108 + 40 / 4 ** index, y]
          ],
          role: selected ? 'hot' : 'dim',
          width: 3,
          alpha: selected ? 1 : 0.6,
          phase: index / 4
        });
        perimeterBars.push({
          points: [
            [108, y + 6],
            [108 + 40 / 2 ** index, y + 6]
          ],
          role: selected ? 'cool' : 'dim',
          width: 1.6,
          alpha: selected ? 1 : 0.6,
          phase: index / 4
        });
      }
      return [
        ...createGlobeBase(view),
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(outlines, {animation: 'static'}),
        createStrokeLayer([...areaBars, ...perimeterBars], {
          animation: 'flow',
          speed: 0.3,
          trail: 1.2,
          blend: 'add'
        })
      ];
    }
  },

  's2-cap-covering': {
    pointer: 'hover',
    parameters: [
      createSlider('radius', 'Cap radius', 8, 60, 1, 30, '°'),
      createSlider('level', 'Level', 1, 4, 1, 3),
      createChoice('mode', 'Cells', ['exterior', 'interior'])
    ],
    build: ({p, pointer}) => {
      const view: GlobeView = {centerX: 80, centerY: 50, radius: 44, longitude: 0, latitude: 20};
      const center: Point = (pointer && unprojectGlobe(view, pointer)) || [12, 12];
      const span = 48 / 2 ** (Math.round(p.level) - 1);
      const lit: PolygonItem[] = [];
      const quiet: PathItem[] = [];
      const interior = Math.round(p.mode) === 1;
      const radiusLimit = p.radius + span * 1.6;
      for (let latitude = -90 + (span > 0 ? 0 : 0); latitude < 90; latitude += span) {
        for (let longitude = -180; longitude < 180; longitude += span) {
          const middle: Point = [longitude + span / 2, latitude + span / 2];
          if (getAngularDistance(middle, center) > radiusLimit) continue;
          const corners: Point[] = [
            [longitude, latitude],
            [longitude + span, latitude],
            [longitude + span, latitude + span],
            [longitude, latitude + span]
          ];
          const distances = [middle, ...corners].map(point => getAngularDistance(point, center));
          const touches = Math.min(...distances) <= p.radius;
          const fully = Math.max(...distances.slice(1)) <= p.radius;
          const ring = projectLonLatRing(view, sampleArcRing(corners, 2));
          if (!ring) continue;
          if (interior ? fully : touches)
            lit.push({ring, role: fully ? 'primary' : 'violet', alpha: 0.5, phase: middle[0] / 60});
          else quiet.push({points: closeRing(ring), role: 'dim', width: 0.2, alpha: 0.3});
        }
      }
      const boundary: Point[] = [];
      for (let step = 0; step <= 72; step++) {
        const bearing = (step / 72) * 2 * Math.PI;
        const delta = p.radius * DEGREES;
        const latitude0 = center[1] * DEGREES;
        const latitude = Math.asin(
          Math.sin(latitude0) * Math.cos(delta) +
            Math.cos(latitude0) * Math.sin(delta) * Math.cos(bearing)
        );
        const longitude =
          center[0] * DEGREES +
          Math.atan2(
            Math.sin(bearing) * Math.sin(delta) * Math.cos(latitude0),
            Math.cos(delta) - Math.sin(latitude0) * Math.sin(latitude)
          );
        boundary.push([longitude / DEGREES, latitude / DEGREES]);
      }
      const centerPoint = projectVisible(view, center[0], center[1]);
      return [
        ...createGlobeBase(view),
        createStrokeLayer(quiet, {animation: 'static'}),
        createFillLayer(lit, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(
          projectLonLatPath(view, boundary).map(run => ({
            points: run,
            role: 'warm' as Role,
            width: 1.1,
            alpha: 1
          })),
          {animation: 'flow', speed: 0.3, trail: 1.5, blend: 'add'}
        ),
        createSpriteLayer(
          centerPoint ? [{x: centerPoint[0], y: centerPoint[1], radius: 2.4, role: 'hot'}] : [],
          {pulse: 0.3}
        )
      ];
    }
  },

  's2-polygon-covering': {
    parameters: [
      createSlider('level', 'Max level', 2, 6, 1, 5),
      createChoice('mode', 'Cover', ['exterior', 'interior'])
    ],
    build: ({fork, p}) => {
      const face = {x: 36, y: 6, size: 88};
      const polygon = createBlob(fork('polygon'), 80, 50, 31, 12, 0.35, 0.95);
      const maximum = Math.round(p.level);
      const exterior = Math.round(p.mode) === 0;
      const fills: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      const visit = (x: number, y: number, size: number, level: number) => {
        const box: Box = {minX: x, minY: y, maxX: x + size, maxY: y + size};
        const crossing = doesPolygonTouchBoxEdges(polygon, box);
        if (!crossing) {
          if (!isInsideRing([x + size / 2, y + size / 2], polygon)) return;
        } else if (level < maximum) {
          const half = size / 2;
          visit(x, y, half, level + 1);
          visit(x + half, y, half, level + 1);
          visit(x, y + half, half, level + 1);
          visit(x + half, y + half, half, level + 1);
          return;
        } else if (!exterior) return;
        const ring = createSquareRing(x + 0.2, y + 0.2, size - 0.4);
        fills.push({
          ring: ring.slice(0, 4),
          role: crossing ? 'hot' : getClassRole(Math.min(level, 4), 5),
          alpha: crossing ? 0.5 : 0.28,
          phase: level / 7
        });
        outlines.push({
          points: ring,
          role: crossing ? 'hot' : 'ink',
          width: 0.3,
          alpha: crossing ? 0.8 : 0.5
        });
      };
      visit(face.x, face.y, face.size, 0);
      return [
        createFillLayer(fills, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(outlines, {animation: 'static'}),
        createStrokeLayer([{points: closeRing(polygon), role: 'warm', width: 0.9, alpha: 0.95}], {
          animation: 'flow',
          speed: 0.25,
          trail: 1.5,
          blend: 'add'
        })
      ];
    }
  },

  // -------------------------------------------------------------------------------------------
  // Spherical polygons
  // -------------------------------------------------------------------------------------------

  'spherical-convex-contains': {
    pointer: 'hover',
    parameters: [
      createSlider('points', 'Points', 30, 160, 10, 90),
      createSlider('vertices', 'Vertices', 3, 8, 1, 5),
      createChoice('region', 'Region', ['inside', 'complement'])
    ],
    build: ({fork, p, pointer}) => {
      const view: GlobeView = {centerX: 80, centerY: 50, radius: 44, longitude: 0, latitude: 20};
      const vertices = createLonLatPolygon(fork('shape'), 0, 15, 34, Math.round(p.vertices), 0.12);
      const vectors = vertices.map(([longitude, latitude]) => toVector(longitude, latitude));
      const centroid = toVector(0, 15);
      const orientation =
        Math.sign(getVectorDot(getVectorCross(vectors[0], vectors[1]), centroid)) || 1;
      const contains = (point: Point): boolean => {
        const vector = toVector(point[0], point[1]);
        return vectors.every(
          (start, index) =>
            getVectorDot(getVectorCross(start, vectors[(index + 1) % vectors.length]), vector) *
              orientation >=
            0
        );
      };
      const complement = Math.round(p.region) === 1;
      const random = fork('points');
      const selected: SpriteItem[] = [];
      const others: SpriteItem[] = [];
      for (let index = 0; index < Math.round(p.points); index++) {
        const point: Point = [random.range(-75, 75), random.range(-50, 75)];
        const projected = projectVisible(view, point[0], point[1]);
        if (!projected) continue;
        const item: SpriteItem = {
          x: projected[0],
          y: projected[1],
          radius: 1.3,
          role: 'warm',
          phase: random.next()
        };
        if (contains(point) !== complement) selected.push(item);
        else others.push({...item, role: 'primary', alpha: 0.5, radius: 0.9});
      }
      const probe = pointer ? unprojectGlobe(view, pointer) : null;
      const boundary = projectLonLatPath(view, closeRing(sampleArcRing(vertices, 10)));
      const ring = projectLonLatRing(view, sampleArcRing(vertices, 10));
      return [
        ...createGlobeBase(view),
        createFillLayer(
          ring ? [{ring, role: complement ? 'dim' : 'warm', alpha: complement ? 0.1 : 0.14}] : [],
          {animation: 'pulse', blend: 'add'}
        ),
        createSpriteLayer(others),
        createSpriteLayer(selected, {pulse: 0.3}),
        createStrokeLayer(
          boundary.map(run => ({points: run, role: 'warm' as Role, width: 1, alpha: 0.95})),
          {animation: 'flow', speed: 0.25, trail: 1.5, blend: 'add'}
        ),
        createSpriteLayer(
          probe
            ? [
                {
                  x: pointer![0],
                  y: pointer![1],
                  radius: 3,
                  role: contains(probe) !== complement ? 'hot' : 'ink'
                }
              ]
            : [],
          {pulse: 0.3}
        )
      ];
    }
  },

  'spherical-polygon-contains': {
    parameters: [
      createSlider('points', 'Points', 40, 240, 10, 180),
      createSlider('twist', 'Concavity', 0.3, 0.9, 0.05, 0.5)
    ],
    build: ({fork, p}) => {
      const view: GlobeView = {centerX: 80, centerY: 50, radius: 44, longitude: 0, latitude: 20};
      const spikes = 7;
      const outer: Point[] = Array.from({length: spikes * 2}, (_, index) => {
        const angle = (index / (spikes * 2)) * Math.PI * 2;
        const scale = index % 2 === 0 ? 38 : 38 * p.twist;
        return [Math.cos(angle) * scale, 12 + Math.sin(angle) * scale];
      });
      const hole = createLonLatPolygon(fork('hole'), 4, 10, 8, 8, 0);
      const edges = (ring: Point[]) =>
        ring.map((start, index): [Point, Point] => [start, ring[(index + 1) % ring.length]]);
      const allEdges = [...edges(outer), ...edges(hole)];
      const random = fork('points');
      const inside: SpriteItem[] = [];
      const outside: SpriteItem[] = [];
      const onEdge: SpriteItem[] = [];
      for (let index = 0; index < Math.round(p.points); index++) {
        const point: Point = [random.range(-48, 48), random.range(-38, 62)];
        const projected = projectVisible(view, point[0], point[1]);
        if (!projected) continue;
        const item: SpriteItem = {
          x: projected[0],
          y: projected[1],
          radius: 1.2,
          role: 'warm',
          phase: random.next()
        };
        const edgeDistance = Math.min(
          ...allEdges.map(([start, end]) => getSegmentDistance(point, start, end))
        );
        if (edgeDistance < 2.2) onEdge.push({...item, role: 'violet', radius: 1.8});
        else if (isInsideRing(point, outer) && !isInsideRing(point, hole)) inside.push(item);
        else outside.push({...item, role: 'primary', alpha: 0.45, radius: 0.9});
      }
      const outlines = [outer, hole].flatMap(ring =>
        projectLonLatPath(view, closeRing(sampleArcRing(ring, 4)))
      );
      return [
        ...createGlobeBase(view),
        createSpriteLayer(outside),
        createSpriteLayer(inside, {pulse: 0.3}),
        createStrokeLayer(
          outlines.map(run => ({
            points: run,
            role: 'warm' as Role,
            width: 0.9,
            alpha: 0.95
          })),
          {animation: 'flow', speed: 0.25, trail: 1.5, blend: 'add'}
        ),
        createSpriteLayer(onEdge, {pulse: 0.5})
      ];
    }
  },

  'spherical-convex-intersection': {
    pointer: 'hover',
    parameters: [createSlider('separation', 'Separation', 0, 50, 1, 20, '°')],
    build: ({fork, p, pointer}) => {
      const view: GlobeView = {centerX: 66, centerY: 50, radius: 42, longitude: 0, latitude: 20};
      const pointerCenter = pointer ? unprojectGlobe(view, pointer) : null;
      const secondCenter: Point = pointerCenter ?? [p.separation - 12, 8];
      const first = createLonLatPolygon(fork('first'), -12, 8, 28, 5, 0.1);
      const second = createLonLatPolygon(
        fork('second'),
        secondCenter[0],
        secondCenter[1],
        28,
        6,
        0.1
      );
      const overlap = clipConvex(first, second);
      const firstRing = sampleArcRing(first, 6);
      const secondRing = sampleArcRing(second, 6);
      const overlapRing = overlap.length > 2 ? sampleArcRing(overlap, 4) : [];
      const toFill = (ring: Point[], role: Role, alpha: number): PolygonItem[] => {
        const projected = ring.length > 2 ? projectLonLatRing(view, ring) : null;
        return projected ? [{ring: projected, role, alpha}] : [];
      };
      const intersectionArea = overlap.length > 2 ? Math.abs(getSignedArea(overlap)) : 0;
      const union =
        Math.abs(getSignedArea(first)) + Math.abs(getSignedArea(second)) - intersectionArea;
      const fraction = clamp(intersectionArea / union, 0, 1);
      const vertexSprites: SpriteItem[] = overlap
        .map(([longitude, latitude]) => projectVisible(view, longitude, latitude))
        .filter((point): point is Point => point !== null)
        .map(([x, y]) => ({x, y, radius: 1.6, role: 'ink' as Role}));
      return [
        ...createGlobeBase(view),
        createFillLayer([...toFill(firstRing, 'cool', 0.2), ...toFill(secondRing, 'warm', 0.2)], {
          animation: 'static',
          blend: 'add'
        }),
        createFillLayer(toFill(overlapRing, 'violet', 0.7), {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(
          [
            ...projectLonLatPath(view, closeRing(firstRing)).map(run => ({
              points: run,
              role: 'cool' as Role,
              width: 0.9,
              alpha: 0.95
            })),
            ...projectLonLatPath(view, closeRing(secondRing)).map(run => ({
              points: run,
              role: 'warm' as Role,
              width: 0.9,
              alpha: 0.95
            }))
          ],
          {animation: 'static', blend: 'add'}
        ),
        createSpriteLayer(vertexSprites, {pulse: 0.3}),
        createStrokeLayer(
          [
            {
              points: [
                [124, 88],
                [124, 12]
              ],
              role: 'dim',
              width: 4,
              alpha: 0.35
            },
            {
              points: [
                [124, 88],
                [124, 88 - 76 * fraction - 0.01]
              ],
              role: 'violet',
              width: 4,
              alpha: 0.95
            }
          ],
          {animation: 'static', blend: 'add'}
        )
      ];
    }
  },

  'spherical-polygon-overlay': {
    pointer: 'hover',
    parameters: [
      createChoice('operation', 'Operation', ['intersection', 'union', 'difference']),
      createSlider('separation', 'Separation', 0, 40, 1, 16, '°')
    ],
    build: ({fork, p, pointer}) => {
      const view: GlobeView = {centerX: 80, centerY: 50, radius: 44, longitude: 0, latitude: 20};
      const pointerCenter = pointer ? unprojectGlobe(view, pointer) : null;
      const secondCenter: Point = pointerCenter ?? [p.separation - 8, 4];
      const left = createBlob(fork('left'), -10, 10, 30, 14, 0.5, 1);
      const right = createBlob(fork('right'), secondCenter[0], secondCenter[1], 28, 14, 0.5, 1);
      const operation = Math.round(p.operation);
      const roles: Role[] = ['violet', 'primary', 'hot'];
      const dots: SpriteItem[] = [];
      for (let latitude = -42; latitude <= 62; latitude += 3.2) {
        for (let longitude = -62; longitude <= 62; longitude += 3.2) {
          const inLeft = isInsideRing([longitude, latitude], left);
          const inRight = isInsideRing([longitude, latitude], right);
          const kept =
            operation === 0
              ? inLeft && inRight
              : operation === 1
                ? inLeft || inRight
                : inLeft && !inRight;
          if (!kept) continue;
          const projected = projectVisible(view, longitude, latitude);
          if (projected)
            dots.push({
              x: projected[0],
              y: projected[1],
              radius: 0.95,
              role: roles[operation],
              alpha: 0.85,
              phase: (longitude + latitude) / 120
            });
        }
      }
      return [
        ...createGlobeBase(view),
        createSpriteLayer(dots, {pulse: 0.25}),
        createStrokeLayer(
          [
            ...projectLonLatPath(view, closeRing(sampleArcRing(left, 2))).map(run => ({
              points: run,
              role: 'cool' as Role,
              width: 0.9,
              alpha: 0.95
            })),
            ...projectLonLatPath(view, closeRing(sampleArcRing(right, 2))).map(run => ({
              points: run,
              role: 'warm' as Role,
              width: 0.9,
              alpha: 0.95
            }))
          ],
          {animation: 'flow', speed: 0.2, trail: 1.6, blend: 'add'}
        )
      ];
    }
  },

  'spherical-arc-overlay': {
    pointer: 'hover',
    parameters: [
      createSlider('radius', 'Cap radius', 100, 155, 1, 140, '°'),
      createSlider('separation', 'Separation', 20, 120, 2, 56, '°')
    ],
    build: ({p, pointer}) => {
      const view: GlobeView = {centerX: 80, centerY: 50, radius: 44, longitude: 0, latitude: 20};
      // A cap larger than a hemisphere is the complement of a small disc around its antipode.
      const holeRadius = 180 - p.radius;
      const firstHole: Point = [-p.separation / 2, 10];
      const pointerCenter = pointer ? unprojectGlobe(view, pointer) : null;
      const secondHole: Point = pointerCenter ?? [p.separation / 2, 10];
      const dots: SpriteItem[] = [];
      for (let y = -44; y <= 44; y += 3.4) {
        for (let x = -44; x <= 44; x += 3.4) {
          const lonLat = unprojectGlobe(view, [view.centerX + x, view.centerY + y]);
          if (!lonLat) continue;
          const inFirst = getAngularDistance(lonLat, firstHole) >= holeRadius;
          const inSecond = getAngularDistance(lonLat, secondHole) >= holeRadius;
          if (!inFirst && !inSecond) continue;
          const both = inFirst && inSecond;
          dots.push({
            x: view.centerX + x,
            y: view.centerY + y,
            radius: both ? 1 : 1.2,
            role: both ? 'violet' : inFirst ? 'cool' : 'warm',
            alpha: both ? 0.5 : 0.95,
            phase: (x + y) / 90
          });
        }
      }
      const circle = (center: Point): Point[][] => {
        const boundary: Point[] = [];
        for (let step = 0; step <= 90; step++) {
          const bearing = (step / 90) * 2 * Math.PI;
          const delta = holeRadius * DEGREES;
          const latitude0 = center[1] * DEGREES;
          const latitude = Math.asin(
            Math.sin(latitude0) * Math.cos(delta) +
              Math.cos(latitude0) * Math.sin(delta) * Math.cos(bearing)
          );
          const longitude =
            center[0] * DEGREES +
            Math.atan2(
              Math.sin(bearing) * Math.sin(delta) * Math.cos(latitude0),
              Math.cos(delta) - Math.sin(latitude0) * Math.sin(latitude)
            );
          boundary.push([longitude / DEGREES, latitude / DEGREES]);
        }
        return projectLonLatPath(view, boundary);
      };
      return [
        ...createGlobeBase(view),
        createSpriteLayer(dots, {pulse: 0.2}),
        createStrokeLayer(
          [
            ...circle(firstHole).map(run => ({
              points: run,
              role: 'cool' as Role,
              width: 1,
              alpha: 1
            })),
            ...circle(secondHole).map(run => ({
              points: run,
              role: 'warm' as Role,
              width: 1,
              alpha: 1
            }))
          ],
          {animation: 'flow', speed: 0.25, trail: 1.6, blend: 'add'}
        )
      ];
    }
  },

  'hilbert-keys': {
    pointer: 'hover',
    parameters: [
      createSlider('order', 'Order', 1, 5, 1, 3),
      createSlider('points', 'Points', 10, 80, 2, 28)
    ],
    build: ({fork, p, pointer}) => {
      const order = Math.round(p.order);
      const side = 2 ** order;
      const face = {x: 36, y: 6, size: 88};
      const cell = face.size / side;
      const total = side * side;
      const centers: Point[] = [];
      const keyOf = new Map<number, number>();
      for (let index = 0; index < total; index++) {
        const [column, row] = getHilbertCell(order, index);
        keyOf.set(row * side + column, index);
        centers.push([face.x + (column + 0.5) * cell, face.y + (row + 0.5) * cell]);
      }
      const segments = 5;
      const curve: PathItem[] = [];
      for (let segment = 0; segment < segments; segment++) {
        const start = Math.floor((segment * total) / segments);
        const end = Math.min(total, Math.floor(((segment + 1) * total) / segments) + 1);
        curve.push({
          points: centers.slice(start, end),
          role: getClassRole(segment, segments),
          width: 0.9,
          alpha: 0.9,
          phase: segment / segments
        });
      }
      const cellOf = (point: Point) => {
        const column = clamp(Math.floor((point[0] - face.x) / cell), 0, side - 1);
        const row = clamp(Math.floor((point[1] - face.y) / cell), 0, side - 1);
        return keyOf.get(row * side + column) ?? 0;
      };
      const random = fork('points');
      const points: SpriteItem[] = Array.from({length: Math.round(p.points)}, () => {
        const point: Point = [
          random.range(face.x + 2, face.x + face.size - 2),
          random.range(face.y + 2, face.y + face.size - 2)
        ];
        return {
          x: point[0],
          y: point[1],
          radius: 1.4,
          role: getClassRole(Math.floor((cellOf(point) / total) * segments), segments),
          phase: random.next()
        };
      });
      const probe =
        pointer && pointer[0] > face.x && pointer[0] < face.x + face.size ? pointer : null;
      const probeKey = probe ? cellOf(probe) : -1;
      const grid: PathItem[] = [];
      for (let line = 0; line <= side; line++) {
        grid.push({
          points: [
            [face.x + line * cell, face.y],
            [face.x + line * cell, face.y + face.size]
          ],
          role: 'dim',
          width: 0.18,
          alpha: 0.22
        });
        grid.push({
          points: [
            [face.x, face.y + line * cell],
            [face.x + face.size, face.y + line * cell]
          ],
          role: 'dim',
          width: 0.18,
          alpha: 0.22
        });
      }
      return [
        createStrokeLayer(grid, {animation: 'static'}),
        createStrokeLayer(curve, {animation: 'flow', speed: 0.25, trail: 2.5, blend: 'add'}),
        createSpriteLayer(points, {pulse: 0.2}),
        createSpriteLayer(
          probe
            ? [{x: centers[probeKey][0], y: centers[probeKey][1], radius: 3.2, role: 'warm'}]
            : [],
          {pulse: 0.3}
        )
      ];
    }
  },

  // -------------------------------------------------------------------------------------------
  // Spatial joins
  // -------------------------------------------------------------------------------------------

  'spatial-predicate-join': {
    pointer: 'hover',
    parameters: [
      createChoice('predicate', 'Predicate', ['intersects', 'contains', 'dwithin']),
      createSlider('parks', 'Parks', 1, 5, 1, 3)
    ],
    build: ({fork, p, pointer}) => {
      const tracts = createTractQuads(fork('tracts'), 7, 4);
      const random = fork('parks');
      const spots: Point[] = [
        [34, 30],
        [96, 42],
        [122, 74],
        [52, 78],
        [80, 14]
      ];
      const parks = spots
        .slice(0, Math.round(p.parks))
        .map((spot, index): {center: Point; ring: Point[]; radius: number} => {
          const center: Point =
            index === 0 && pointer
              ? clampPoint(pointer, 12)
              : [spot[0] + random.range(-6, 6), spot[1] + random.range(-4, 4)];
          const radius = 8 + random.range(0, 4);
          return {center, radius, ring: createRegularPolygon(center[0], center[1], radius, 10)};
        });
      const predicate = Math.round(p.predicate);
      const matched = new Set<number>();
      const links: PathItem[] = [];
      parks.forEach((park, parkIndex) => {
        tracts.forEach((tract, tractIndex) => {
          const samples = [park.center, ...park.ring];
          const hit =
            predicate === 0
              ? samples.some(point => isInsideRing(point, tract)) ||
                tract.some(vertex => getDistance(vertex, park.center) < park.radius)
              : predicate === 1
                ? samples.every(point => isInsideRing(point, tract))
                : getDistanceToRing(park.center, tract) < park.radius + 6;
          if (!hit) return;
          matched.add(tractIndex);
          links.push({
            points: [park.center, getVertexMean(tract)],
            role: 'warm',
            width: 0.6,
            alpha: 0.85,
            phase: (parkIndex * 3 + tractIndex) / 20
          });
        });
      });
      return [
        createFillLayer(
          tracts.flatMap((tract, index) =>
            matched.has(index)
              ? [{ring: tract, role: 'primary' as Role, alpha: 0.3, phase: index / 15}]
              : []
          ),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(
          tracts.map(tract => ({
            points: closeRing(tract),
            role: 'dim' as Role,
            width: 0.4,
            alpha: 0.6
          })),
          {animation: 'static'}
        ),
        createFillLayer(
          parks.map((park, index) => ({
            ring: park.ring,
            role: 'warm' as Role,
            alpha: 0.5,
            phase: index / 5
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(links, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createSpriteLayer(
          parks.map(park => ({
            x: park.center[0],
            y: park.center[1],
            radius: 1.6,
            role: 'hot' as Role
          })),
          {pulse: 0.3}
        )
      ];
    }
  },

  'point-in-polygon-join': {
    parameters: [createSlider('points', 'Points', 30, 220, 10, 100)],
    build: ({fork, p}) => {
      const centers: Point[] = [
        [36, 32],
        [84, 26],
        [126, 38],
        [52, 72],
        [108, 72]
      ];
      const random = fork('zones');
      const zones = centers.map(([x, y]) => createBlob(random, x, y, 19, 11, 0.3, 0.9));
      const roles: Role[] = ['primary', 'warm', 'cool', 'violet', 'hot'];
      const points = createScatterPoints(fork('points'), Math.round(p.points), 4);
      const sprites: SpriteItem[] = [];
      const quiet: SpriteItem[] = [];
      const random2 = fork('phase');
      for (const point of points) {
        const zone = zones.findIndex(ring => isInsideRing(point, ring));
        if (zone < 0) quiet.push({x: point[0], y: point[1], radius: 0.8, role: 'dim', alpha: 0.5});
        else
          sprites.push({
            x: point[0],
            y: point[1],
            radius: 1.5,
            role: roles[zone],
            phase: random2.next()
          });
      }
      return [
        createFillLayer(
          zones.map((ring, index) => ({
            ring,
            role: roles[index],
            alpha: 0.14,
            phase: index / 5
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(
          zones.map((ring, index) => ({
            points: closeRing(ring),
            role: roles[index],
            width: 0.7,
            alpha: 0.85
          })),
          {animation: 'static', blend: 'add'}
        ),
        createSpriteLayer(quiet),
        createSpriteLayer(sprites, {pulse: 0.3})
      ];
    }
  },

  'spatial-join-candidates': {
    pointer: 'hover',
    parameters: [createSlider('distance', 'Box expansion', 0, 14, 0.5, 3, 'px')],
    build: ({fork, p, pointer}) => {
      const random = fork('shapes');
      const left = (
        [
          [26, 28, 12],
          [60, 70, 14],
          [112, 24, 12],
          [132, 76, 11]
        ] as number[][]
      ).map(([x, y, radius]) => createBlob(random, x, y, radius, 9, 0.3, 1));
      const rightCenters: Point[] = [
        [44, 50],
        [88, 52],
        [120, 54]
      ];
      const right = rightCenters.map(([x, y], index) => {
        const center: Point = index === 1 && pointer ? clampPoint(pointer, 14) : [x, y];
        return createBlob(random, center[0], center[1], 15, 9, 0.3, 1);
      });
      const leftBoxes = left.map(ring => expandBox(getBounds(ring), p.distance / 2));
      const rightBoxes = right.map(ring => expandBox(getBounds(ring), p.distance / 2));
      const overlaps: PolygonItem[] = [];
      const links: PathItem[] = [];
      leftBoxes.forEach((first, leftIndex) => {
        rightBoxes.forEach((second, rightIndex) => {
          if (!doBoxesOverlap(first, second)) return;
          const box: Box = {
            minX: Math.max(first.minX, second.minX),
            minY: Math.max(first.minY, second.minY),
            maxX: Math.min(first.maxX, second.maxX),
            maxY: Math.min(first.maxY, second.maxY)
          };
          overlaps.push({
            ring: getBoxRing(box).slice(0, 4),
            role: 'violet',
            alpha: 0.6,
            phase: (leftIndex + rightIndex) / 8
          });
          links.push({
            points: [getVertexMean(left[leftIndex]), getVertexMean(right[rightIndex])],
            role: 'violet',
            width: 0.7,
            alpha: 0.9,
            phase: (leftIndex * 3 + rightIndex) / 12
          });
        });
      });
      return [
        createFillLayer(
          [
            ...left.map(ring => ({ring, role: 'cool' as Role, alpha: 0.3})),
            ...right.map(ring => ({ring, role: 'warm' as Role, alpha: 0.3}))
          ],
          {animation: 'static', blend: 'add'}
        ),
        createStrokeLayer(
          [
            ...left.map(ring => ({
              points: closeRing(ring),
              role: 'cool' as Role,
              width: 0.6,
              alpha: 0.9
            })),
            ...right.map(ring => ({
              points: closeRing(ring),
              role: 'warm' as Role,
              width: 0.6,
              alpha: 0.9
            }))
          ],
          {animation: 'static', blend: 'add'}
        ),
        createStrokeLayer(
          [
            ...leftBoxes.map(box => ({
              points: getBoxRing(box),
              role: 'cool' as Role,
              width: 0.35,
              alpha: 0.6
            })),
            ...rightBoxes.map(box => ({
              points: getBoxRing(box),
              role: 'warm' as Role,
              width: 0.35,
              alpha: 0.6
            }))
          ],
          {animation: 'static'}
        ),
        createFillLayer(overlaps, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(links, {animation: 'flow', speed: 0.4, blend: 'add'})
      ];
    }
  },

  'spatial-join-prepared': {
    pointer: 'hover',
    parameters: [
      createSlider('features', 'Features', 6, 26, 1, 14),
      createSlider('leaf', 'Leaf size', 1, 4, 1, 2)
    ],
    build: ({fork, p, pointer}) => {
      const random = fork('features');
      const features = createScatterPoints(random, Math.round(p.features), 14).map(([x, y]) =>
        createBlob(random, x, y, 4.5, 8, 0.3, 1)
      );
      const items = features.map(ring => ({ring, box: getBounds(ring)}));
      const nodes: {box: Box; depth: number}[] = [];
      const build = (members: typeof items, depth: number) => {
        const box = getBounds(members.flatMap(member => member.ring));
        nodes.push({box, depth});
        if (members.length <= Math.round(p.leaf)) return;
        const horizontal = box.maxX - box.minX >= box.maxY - box.minY;
        const sorted = [...members].sort((first, second) =>
          horizontal
            ? first.box.minX + first.box.maxX - second.box.minX - second.box.maxX
            : first.box.minY + first.box.maxY - second.box.minY - second.box.maxY
        );
        const middle = Math.floor(sorted.length / 2);
        build(sorted.slice(0, middle), depth + 1);
        build(sorted.slice(middle), depth + 1);
      };
      build(items, 0);
      const query: Point = pointer ?? [88, 46];
      const depthRoles: Role[] = ['dim', 'cool', 'violet', 'primary', 'primary', 'primary'];
      const quiet: PathItem[] = [];
      const hit: PathItem[] = [];
      for (const node of nodes) {
        const entry: PathItem = {
          points: getBoxRing(node.box),
          role: depthRoles[Math.min(5, node.depth)],
          width: 0.3 + node.depth * 0.06,
          alpha: 0.45
        };
        if (isInsideBox(query, node.box))
          hit.push({...entry, role: 'warm', width: 0.7, alpha: 0.85, phase: node.depth / 6});
        else quiet.push(entry);
      }
      const candidates = items.map(entry => isInsideBox(query, entry.box));
      return [
        createFillLayer(
          features.map((ring, index) => ({
            ring,
            role: candidates[index] ? ('hot' as Role) : ('primary' as Role),
            alpha: candidates[index] ? 0.7 : 0.3,
            phase: index / features.length
          })),
          {animation: 'pulse', blend: 'add'}
        ),
        createStrokeLayer(quiet, {animation: 'static'}),
        createStrokeLayer(hit, {animation: 'flow', speed: 0.4, trail: 1.5, blend: 'add'}),
        createSpriteLayer([{x: query[0], y: query[1], radius: 2.2, role: 'ink'}], {pulse: 0.4})
      ];
    }
  },

  'nearest-feature-join': {
    pointer: 'hover',
    parameters: [
      createSlider('queries', 'Queries', 6, 60, 2, 26),
      createSlider('features', 'Features', 3, 12, 1, 6)
    ],
    build: ({fork, p, pointer}) => {
      const features = createScatterPoints(fork('features'), Math.round(p.features), 14);
      const queries = createScatterPoints(fork('queries'), Math.round(p.queries), 5);
      if (pointer) queries[0] = pointer;
      const values = new Float32Array(FIELD_COLUMNS * FIELD_ROWS);
      for (let row = 0; row < FIELD_ROWS; row++) {
        for (let column = 0; column < FIELD_COLUMNS; column++) {
          const x = ((column + 0.5) / FIELD_COLUMNS) * SCENE_WIDTH;
          const y = ((row + 0.5) / FIELD_ROWS) * SCENE_HEIGHT;
          let nearest = Infinity;
          for (const feature of features)
            nearest = Math.min(nearest, Math.hypot(feature[0] - x, feature[1] - y));
          values[row * FIELD_COLUMNS + column] = 1 - Math.min(nearest / 60, 1);
        }
      }
      const links: PathItem[] = [];
      const sprites: SpriteItem[] = [];
      queries.forEach((query, index) => {
        let best = 0;
        let bestDistance = Infinity;
        features.forEach((feature, featureIndex) => {
          const distance = getDistance(query, feature);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = featureIndex;
          }
        });
        const role = index === 0 && pointer ? 'hot' : getRoleCycle(best);
        links.push({
          points: [query, features[best]],
          role,
          width: 0.5,
          alpha: 0.85,
          phase: index / queries.length
        });
        sprites.push({
          x: query[0],
          y: query[1],
          radius: index === 0 && pointer ? 2.4 : 1.2,
          role,
          phase: index / queries.length
        });
      });
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'ice',
          style: 'contours',
          levels: 9,
          strength: 0.15,
          speed: 0.3
        },
        createStrokeLayer(links, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createSpriteLayer(sprites, {pulse: 0.15}),
        createSpriteLayer(
          features.map(([x, y], index) => ({
            x,
            y,
            radius: 3.4,
            role: getRoleCycle(index),
            phase: index / features.length
          })),
          {pulse: 0.3}
        )
      ];
    }
  },

  'nearest-feature-weights': {
    pointer: 'hover',
    parameters: [
      createSlider('neighbors', 'Neighbors (k)', 1, 8, 1, 3),
      createChoice('scheme', 'Weights', ['binary', 'inverse distance'], 1)
    ],
    build: ({fork, p, pointer}) => {
      const points = createScatterPoints(fork('points'), 38, 10);
      const neighborCount = Math.round(p.neighbors);
      const nearestOf = (index: number) =>
        points
          .map((other, otherIndex) => ({otherIndex, distance: getDistance(points[index], other)}))
          .filter(entry => entry.otherIndex !== index)
          .sort((first, second) => first.distance - second.distance)
          .slice(0, neighborCount);
      let focus = 0;
      let centerDistance = Infinity;
      points.forEach((point, index) => {
        const distance = getDistance(point, [80, 50]);
        if (distance < centerDistance) {
          centerDistance = distance;
          focus = index;
        }
      });
      if (pointer) {
        let bestDistance = Infinity;
        points.forEach((point, index) => {
          const distance = getDistance(point, pointer);
          if (distance < bestDistance) {
            bestDistance = distance;
            focus = index;
          }
        });
      }
      const inverse = Math.round(p.scheme) === 1;
      const background: PathItem[] = [];
      const seen = new Set<number>();
      points.forEach((point, index) => {
        if (index === focus) return;
        for (const entry of nearestOf(index)) {
          const key = Math.min(index, entry.otherIndex) * 100 + Math.max(index, entry.otherIndex);
          if (seen.has(key) || entry.otherIndex === focus) continue;
          seen.add(key);
          background.push({
            points: [point, points[entry.otherIndex]],
            role: 'dim',
            width: 0.3,
            alpha: 0.45
          });
        }
      });
      const neighbors = nearestOf(focus);
      const weightsRaw = neighbors.map(entry => (inverse ? 1 / Math.max(entry.distance, 1) : 1));
      const total = weightsRaw.reduce((sum, value) => sum + value, 0);
      const weights = weightsRaw.map(value => value / total);
      const links: PathItem[] = neighbors.map((entry, index) => ({
        points: [points[focus], points[entry.otherIndex]],
        role: 'warm',
        width: 0.5 + weights[index] * 3.2,
        alpha: 0.95,
        phase: index / neighborCount
      }));
      const bars: PathItem[] = weights.map((weight, index) => ({
        points: [
          [132 + index * 3.4, 94],
          [132 + index * 3.4, 94 - 30 * weight * (inverse ? 1 : 1)]
        ],
        role: 'warm',
        width: 2,
        alpha: 0.95
      }));
      return [
        createStrokeLayer(background, {animation: 'static'}),
        createStrokeLayer(links, {animation: 'flow', speed: 0.5, blend: 'add'}),
        createSpriteLayer(toSprites(points, 'primary', 1.6), {pulse: 0.2}),
        createSpriteLayer(
          [
            {x: points[focus][0], y: points[focus][1], radius: 4.2, role: 'hot' as Role},
            ...neighbors.map((entry, index) => ({
              x: points[entry.otherIndex][0],
              y: points[entry.otherIndex][1],
              radius: 1.8 + weights[index] * 4,
              role: 'warm' as Role,
              phase: index / neighborCount
            }))
          ],
          {pulse: 0.3}
        ),
        createStrokeLayer(bars, {animation: 'static', blend: 'add'})
      ];
    }
  },

  'buffer-selection': {
    pointer: 'hover',
    parameters: [createSlider('distance', 'Distance', 2, 24, 1, 10, 'px')],
    build: ({fork, p, pointer}) => {
      const phase = fork('road').range(0, 3);
      const road: Point[] = [];
      for (let x = 6; x <= 154; x += 6) road.push([x, 54 + 20 * Math.sin(x / 24 + phase)]);
      const feature: Point = pointer ?? [112, 24];
      const points = createScatterPoints(fork('points'), 130, 4);
      const isSelected = (point: Point) =>
        getDistance(point, feature) <= p.distance ||
        road.some(
          (start, index) =>
            index + 1 < road.length &&
            getSegmentDistance(point, start, road[index + 1]) <= p.distance
        );
      const selected: Point[] = [];
      const rest: Point[] = [];
      for (const point of points) (isSelected(point) ? selected : rest).push(point);
      return [
        createStrokeLayer([{points: road, role: 'cool', width: p.distance * 2, alpha: 0.13}], {
          animation: 'static',
          blend: 'add'
        }),
        createFillLayer(
          [{ring: createCircle(feature[0], feature[1], p.distance, 36), role: 'cool', alpha: 0.16}],
          {animation: 'static', blend: 'add'}
        ),
        createSpriteLayer(toSprites(rest, 'dim', 0.9, undefined, 0.6)),
        createStrokeLayer(
          [{points: smoothPath(road, 1), role: 'primary', width: 0.8, alpha: 0.9}],
          {animation: 'flow', speed: 0.2, blend: 'add'}
        ),
        createSpriteLayer(toSprites(selected, 'warm', 1.7), {pulse: 0.3}),
        createSpriteLayer([{x: feature[0], y: feature[1], radius: 2.6, role: 'hot'}], {pulse: 0.3})
      ];
    }
  }
};
