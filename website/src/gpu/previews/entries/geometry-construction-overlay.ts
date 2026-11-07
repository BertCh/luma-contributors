// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/geometry-construction-overlay.ts, keyed by entry id.

import {
  clamp,
  createBlob,
  createCircle,
  createRegularPolygon,
  createWanderPath,
  evaluateField,
  getDistance,
  mixNumbers,
  type Random
} from '../data.ts';
import {
  clipRingByHalfPlane,
  closeRing,
  createRectangleRing,
  getConvexHull,
  getCross,
  getNearestOnSegment,
  getRingBounds,
  getSignedArea,
  getVertexMean,
  isInsideRing
} from '../geometry.ts';
import {
  createGridLayer,
  getRoleCycle,
  getSegmentDistance,
  smoothPath,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
import {
  createChoice,
  createDot,
  createFillLayer,
  createPath,
  createSlider,
  createSpriteLayer,
  createStrokeLayer
} from '../layers.ts';
import type {
  PathItem,
  Point,
  PolygonItem,
  PreviewLayer,
  Role,
  SpriteItem,
  StrokeLayer
} from '../types.ts';
import {SCENE_HEIGHT, SCENE_WIDTH} from '../types.ts';

// ---------------------------------------------------------------------------------------------
// Layer shorthands with this group's defaults
// ---------------------------------------------------------------------------------------------

type Ring = Point[];

/** A closed outline of a ring. */
function createLoop(ring: Ring, role: Role, width: number, alpha: number, phase = 0): PathItem {
  return createPath(closeRing(ring), role, width, alpha, phase);
}

function createPolygon(ring: Ring, role: Role, alpha = 0.3, phase = 0): PolygonItem {
  return {ring, role, alpha, phase};
}

function createStrokes(
  items: PathItem[],
  animation: StrokeLayer['animation'] = 'static',
  speed = 0.2,
  trail?: number
): StrokeLayer {
  return createStrokeLayer(items, {animation, speed, trail, blend: 'add'});
}

/** Additive fills; rings with fewer than three vertices are skipped. */
function createFills(items: PolygonItem[], animation: 'static' | 'pulse' = 'pulse') {
  return createFillLayer(
    items.filter(item => item.ring.length >= 3),
    {animation, blend: 'add'}
  );
}

function createPulseSprites(items: SpriteItem[], pulse = 0.2) {
  return createSpriteLayer(items, {pulse});
}

// ---------------------------------------------------------------------------------------------
// Geometry for this group
// ---------------------------------------------------------------------------------------------

function mixPoints(first: Point, second: Point, amount: number): Point {
  return [mixNumbers(first[0], second[0], amount), mixNumbers(first[1], second[1], amount)];
}

function createEllipse(
  centerX: number,
  centerY: number,
  radiusX: number,
  radiusY: number,
  rotation = 0,
  segments = 28
): Ring {
  const cosine = Math.cos(rotation);
  const sine = Math.sin(rotation);
  return Array.from({length: segments}, (_, index) => {
    const angle = (index / segments) * Math.PI * 2;
    const x = Math.cos(angle) * radiusX;
    const y = Math.sin(angle) * radiusY;
    return [centerX + x * cosine - y * sine, centerY + x * sine + y * cosine];
  });
}

function createRectangle(x: number, y: number, width: number, height: number): Ring {
  return createRectangleRing(x, y, x + width, y + height);
}

function getInwardNormal(clip: Ring, index: number): {origin: Point; normal: Point} {
  const start = clip[index];
  const end = clip[(index + 1) % clip.length];
  const sign = getSignedArea(clip) >= 0 ? 1 : -1;
  return {
    origin: start,
    normal: [-(end[1] - start[1]) * sign, (end[0] - start[0]) * sign]
  };
}

/** Intersection of a subject ring with a convex clip ring. */
function intersectConvex(subject: Ring, clip: Ring): Ring {
  let current = subject;
  for (let index = 0; index < clip.length && current.length >= 3; index++) {
    const {origin, normal} = getInwardNormal(clip, index);
    current = clipRingByHalfPlane(current, origin, normal);
  }
  return current;
}

/** Pieces of subject outside a convex clip ring; each piece is convex when the subject is. */
function subtractConvex(subject: Ring, clip: Ring): Ring[] {
  const pieces: Ring[] = [];
  let current = subject;
  for (let index = 0; index < clip.length && current.length >= 3; index++) {
    const {origin, normal} = getInwardNormal(clip, index);
    const outside = clipRingByHalfPlane(current, origin, [-normal[0], -normal[1]]);
    if (outside.length >= 3) pieces.push(outside);
    current = clipRingByHalfPlane(current, origin, normal);
  }
  return pieces;
}

/** Position along the first segment (0..1) where it crosses the second, or null. */
function getSegmentCrossing(
  first: Point,
  second: Point,
  third: Point,
  fourth: Point
): number | null {
  const firstX = second[0] - first[0];
  const firstY = second[1] - first[1];
  const secondX = fourth[0] - third[0];
  const secondY = fourth[1] - third[1];
  const denominator = firstX * secondY - firstY * secondX;
  if (Math.abs(denominator) < 1e-9) return null;
  const amount = ((third[0] - first[0]) * secondY - (third[1] - first[1]) * secondX) / denominator;
  const otherAmount =
    ((third[0] - first[0]) * firstY - (third[1] - first[1]) * firstX) / denominator;
  return amount >= 0 && amount <= 1 && otherAmount >= 0 && otherAmount <= 1 ? amount : null;
}

/** Splits a polyline into runs inside and outside a ring. */
function clipPathByRing(path: Point[], ring: Ring): {inside: Point[][]; outside: Point[][]} {
  const inside: Point[][] = [];
  const outside: Point[][] = [];
  let lastClass = '';
  for (let index = 0; index + 1 < path.length; index++) {
    const start = path[index];
    const end = path[index + 1];
    const cuts = [0, 1];
    for (let edge = 0; edge < ring.length; edge++) {
      const hit = getSegmentCrossing(start, end, ring[edge], ring[(edge + 1) % ring.length]);
      if (hit !== null) cuts.push(hit);
    }
    cuts.sort((first, second) => first - second);
    for (let cut = 0; cut + 1 < cuts.length; cut++) {
      if (cuts[cut + 1] - cuts[cut] < 1e-6) continue;
      const from = mixPoints(start, end, cuts[cut]);
      const to = mixPoints(start, end, cuts[cut + 1]);
      const isInside = isInsideRing(mixPoints(from, to, 0.5), ring);
      const runs = isInside ? inside : outside;
      const className = isInside ? 'inside' : 'outside';
      const last = runs[runs.length - 1];
      if (lastClass === className && last && getDistance(last[last.length - 1], from) < 1e-6) {
        last.push(to);
      } else {
        runs.push([from, to]);
      }
      lastClass = className;
    }
  }
  return {inside, outside};
}

function getPathLength(path: Point[]): number {
  let length = 0;
  for (let index = 0; index + 1 < path.length; index++)
    length += getDistance(path[index], path[index + 1]);
  return length;
}

/** The part of a polyline between two arc length positions. */
function sliceRange(path: Point[], from: number, to: number): Point[] {
  const result: Point[] = [];
  let walked = 0;
  for (let index = 0; index + 1 < path.length; index++) {
    const start = path[index];
    const end = path[index + 1];
    const length = getDistance(start, end);
    const segmentStart = walked;
    walked += length;
    if (walked <= from || segmentStart >= to || length === 0) continue;
    const first = Math.max(0, (from - segmentStart) / length);
    const last = Math.min(1, (to - segmentStart) / length);
    if (result.length === 0) result.push(mixPoints(start, end, first));
    result.push(mixPoints(start, end, last));
  }
  return result;
}

function getPointAtMeasure(path: Point[], measure: number): {point: Point; angle: number} {
  let walked = 0;
  for (let index = 0; index + 1 < path.length; index++) {
    const length = getDistance(path[index], path[index + 1]);
    if (walked + length >= measure || index + 2 === path.length) {
      const amount = length === 0 ? 0 : clamp((measure - walked) / length, 0, 1);
      return {
        point: mixPoints(path[index], path[index + 1], amount),
        angle: Math.atan2(path[index + 1][1] - path[index][1], path[index + 1][0] - path[index][0])
      };
    }
    walked += length;
  }
  return {point: path[0], angle: 0};
}

function getNearestOnPath(path: Point[], point: Point) {
  let best = {distance: Infinity, point: path[0], measure: 0, side: 1};
  let walked = 0;
  for (let index = 0; index + 1 < path.length; index++) {
    const start = path[index];
    const end = path[index + 1];
    const nearest = getNearestOnSegment(point, start, end);
    const length = getDistance(start, end);
    if (nearest.distance < best.distance) {
      best = {
        distance: nearest.distance,
        point: nearest.point,
        measure: walked + length * nearest.amount,
        side:
          (end[0] - start[0]) * (point[1] - start[1]) -
            (end[1] - start[1]) * (point[0] - start[0]) >=
          0
            ? 1
            : -1
      };
    }
    walked += length;
  }
  return best;
}

function simplifyDouglasPeucker(path: Point[], tolerance: number): Point[] {
  const last = path.length - 1;
  const keep = new Array<boolean>(path.length).fill(false);
  keep[0] = true;
  keep[last] = true;
  const stack: [number, number][] = [[0, last]];
  while (stack.length > 0) {
    const [start, end] = stack.pop()!;
    let best = -1;
    let bestDistance = tolerance;
    for (let index = start + 1; index < end; index++) {
      const distance = getSegmentDistance(path[index], path[start], path[end]);
      if (distance > bestDistance) {
        best = index;
        bestDistance = distance;
      }
    }
    if (best >= 0) {
      keep[best] = true;
      stack.push([start, best], [best, end]);
    }
  }
  return path.filter((_, index) => keep[index]);
}

function simplifyVisvalingam(path: Point[], minimumArea: number): Point[] {
  const points = [...path];
  while (points.length > 2) {
    let best = -1;
    let bestArea = Infinity;
    for (let index = 1; index + 1 < points.length; index++) {
      const area = Math.abs(getCross(points[index - 1], points[index], points[index + 1])) / 2;
      if (area < bestArea) {
        bestArea = area;
        best = index;
      }
    }
    if (bestArea >= minimumArea) break;
    points.splice(best, 1);
  }
  return points;
}

/** Closed Chaikin corner cutting. */
function smoothRing(ring: Ring, iterations: number): Ring {
  let current = ring;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const next: Ring = [];
    for (let index = 0; index < current.length; index++) {
      const start = current[index];
      const end = current[(index + 1) % current.length];
      next.push(mixPoints(start, end, 0.25), mixPoints(start, end, 0.75));
    }
    current = next;
  }
  return current;
}

function offsetPath(path: Point[], distance: number): Point[] {
  return path.map((point, index) => {
    const before = path[Math.max(0, index - 1)];
    const after = path[Math.min(path.length - 1, index + 1)];
    const normals: Point[] = [];
    if (index > 0) normals.push(getRightNormal(before, point));
    if (index + 1 < path.length) normals.push(getRightNormal(point, after));
    const sumX = normals.reduce((sum, normal) => sum + normal[0], 0);
    const sumY = normals.reduce((sum, normal) => sum + normal[1], 0);
    const length = Math.hypot(sumX, sumY) || 1;
    const miter = Math.max(0.5, (sumX * normals[0][0] + sumY * normals[0][1]) / length);
    return [
      point[0] + (sumX / length / miter) * distance,
      point[1] + (sumY / length / miter) * distance
    ];
  });
}

function getRightNormal(start: Point, end: Point): Point {
  const length = getDistance(start, end) || 1;
  return [-(end[1] - start[1]) / length, (end[0] - start[0]) / length];
}

type Triangle = [number, number, number];

function getCircumcircle(first: Point, second: Point, third: Point) {
  const [firstX, firstY] = first;
  const [secondX, secondY] = second;
  const [thirdX, thirdY] = third;
  const determinant =
    2 * (firstX * (secondY - thirdY) + secondX * (thirdY - firstY) + thirdX * (firstY - secondY));
  if (Math.abs(determinant) < 1e-9) return {x: 0, y: 0, radiusSquared: Infinity};
  const firstSquared = firstX * firstX + firstY * firstY;
  const secondSquared = secondX * secondX + secondY * secondY;
  const thirdSquared = thirdX * thirdX + thirdY * thirdY;
  const x =
    (firstSquared * (secondY - thirdY) +
      secondSquared * (thirdY - firstY) +
      thirdSquared * (firstY - secondY)) /
    determinant;
  const y =
    (firstSquared * (thirdX - secondX) +
      secondSquared * (firstX - thirdX) +
      thirdSquared * (secondX - firstX)) /
    determinant;
  return {x, y, radiusSquared: (firstX - x) ** 2 + (firstY - y) ** 2};
}

/** Bowyer-Watson Delaunay triangulation. */
function triangulatePoints(sites: Point[]): Triangle[] {
  const count = sites.length;
  const points: Point[] = [...sites, [-3000, -3000], [3000, -3000], [80, 3500]];
  const createTriangle = (first: number, second: number, third: number) => ({
    first,
    second,
    third,
    circle: getCircumcircle(points[first], points[second], points[third])
  });
  let triangles = [createTriangle(count, count + 1, count + 2)];
  for (let index = 0; index < count; index++) {
    const point = points[index];
    const kept: typeof triangles = [];
    const edges = new Map<string, [number, number, number]>();
    for (const triangle of triangles) {
      const {x, y, radiusSquared} = triangle.circle;
      if ((point[0] - x) ** 2 + (point[1] - y) ** 2 < radiusSquared) {
        for (const [from, to] of [
          [triangle.first, triangle.second],
          [triangle.second, triangle.third],
          [triangle.third, triangle.first]
        ]) {
          const key = from < to ? `${from}:${to}` : `${to}:${from}`;
          const entry = edges.get(key);
          if (entry) entry[2]++;
          else edges.set(key, [from, to, 1]);
        }
      } else {
        kept.push(triangle);
      }
    }
    for (const [from, to, uses] of edges.values()) {
      if (uses === 1) kept.push(createTriangle(from, to, index));
    }
    triangles = kept;
  }
  return triangles
    .filter(triangle => triangle.first < count && triangle.second < count && triangle.third < count)
    .map(triangle => [triangle.first, triangle.second, triangle.third]);
}

function isInsideTriangle(point: Point, first: Point, second: Point, third: Point): boolean {
  const epsilon = -1e-9;
  const areaSign = getCross(first, second, third) >= 0 ? 1 : -1;
  return (
    getCross(first, second, point) * areaSign >= epsilon &&
    getCross(second, third, point) * areaSign >= epsilon &&
    getCross(third, first, point) * areaSign >= epsilon
  );
}

/** Ear clipping of a simple or bridge-merged ring into index triangles. */
function clipEars(ring: Ring): Triangle[] {
  const triangles: Triangle[] = [];
  const indices = ring.map((_, index) => index);
  const orientation = getSignedArea(ring) >= 0 ? 1 : -1;
  let guard = ring.length * ring.length;
  while (indices.length > 3 && guard-- > 0) {
    let earPosition = -1;
    let flattest = 0;
    let flattestCross = Infinity;
    for (let position = 0; position < indices.length && earPosition < 0; position++) {
      const previous = indices[(position + indices.length - 1) % indices.length];
      const current = indices[position];
      const next = indices[(position + 1) % indices.length];
      const cross = getCross(ring[previous], ring[current], ring[next]) * orientation;
      if (Math.abs(cross) < flattestCross) {
        flattestCross = Math.abs(cross);
        flattest = position;
      }
      if (cross <= 1e-6) continue;
      const blocked = indices.some(other => {
        if (other === previous || other === current || other === next) return false;
        const point = ring[other];
        if ([previous, current, next].some(corner => getDistance(ring[corner], point) < 1e-6))
          return false;
        return isInsideTriangle(point, ring[previous], ring[current], ring[next]);
      });
      if (!blocked) earPosition = position;
    }
    if (earPosition < 0) {
      indices.splice(flattest, 1);
      continue;
    }
    const previous = indices[(earPosition + indices.length - 1) % indices.length];
    const next = indices[(earPosition + 1) % indices.length];
    triangles.push([previous, indices[earPosition], next]);
    indices.splice(earPosition, 1);
  }
  if (indices.length === 3) triangles.push([indices[0], indices[1], indices[2]]);
  return triangles;
}

function getTriangleEdges(triangles: Triangle[]): Map<string, [number, number]> {
  const edges = new Map<string, [number, number]>();
  for (const [first, second, third] of triangles) {
    for (const [from, to] of [
      [first, second],
      [second, third],
      [third, first]
    ]) {
      edges.set(from < to ? `${from}:${to}` : `${to}:${from}`, from < to ? [from, to] : [to, from]);
    }
  }
  return edges;
}

/** Delaunay flips of shared edges; boundary edges have a single triangle and never move. */
function flipTowardDelaunay(points: Point[], input: Triangle[], maximumFlips: number) {
  const triangles = input.map(triangle => [...triangle] as Triangle);
  const flipped = new Set<string>();
  let flipCount = 0;
  const getKey = (from: number, to: number) => (from < to ? `${from}:${to}` : `${to}:${from}`);
  let changed = true;
  while (changed && flipCount < maximumFlips) {
    changed = false;
    for (let first = 0; first < triangles.length && !changed; first++) {
      for (let second = first + 1; second < triangles.length && !changed; second++) {
        const shared = triangles[first].filter(vertex => triangles[second].includes(vertex));
        if (shared.length !== 2) continue;
        const apexA = triangles[first].find(vertex => !shared.includes(vertex))!;
        const apexB = triangles[second].find(vertex => !shared.includes(vertex))!;
        const circle = getCircumcircle(points[shared[0]], points[shared[1]], points[apexA]);
        const insideCircle =
          (points[apexB][0] - circle.x) ** 2 + (points[apexB][1] - circle.y) ** 2 <
          circle.radiusSquared - 1e-6;
        const convex =
          getCross(points[apexA], points[apexB], points[shared[0]]) *
            getCross(points[apexA], points[apexB], points[shared[1]]) <
          0;
        if (!insideCircle || !convex) continue;
        triangles[first] = [apexA, apexB, shared[0]];
        triangles[second] = [apexA, apexB, shared[1]];
        flipped.add(getKey(apexA, apexB));
        flipCount++;
        changed = true;
      }
    }
  }
  return {triangles, flipped, flipCount};
}

// ---------------------------------------------------------------------------------------------
// Shared sample data
// ---------------------------------------------------------------------------------------------

function createStreets(random: Random): Point[][] {
  const base: Point[][] = [
    [
      [8, 66],
      [30, 54],
      [56, 60],
      [82, 44],
      [108, 50],
      [132, 34],
      [152, 40]
    ],
    [
      [30, 8],
      [46, 34],
      [62, 48],
      [72, 72],
      [98, 90],
      [120, 92]
    ],
    [
      [118, 8],
      [114, 34],
      [134, 62],
      [152, 86]
    ]
  ];
  return base.map(path =>
    path.map(([x, y]): Point => [x + random.range(-2, 2), y + random.range(-2, 2)])
  );
}

type HexCell = {center: Point; ring: Ring; neighbors: number[]};

/** Pointy-top hexagon coverage; neighbors[edge] is the cell across that edge or -1. */
function createHexCoverage(): HexCell[] {
  const radius = 10;
  const width = Math.sqrt(3) * radius;
  const columns = 8;
  const rows = 6;
  const startX = (SCENE_WIDTH - (columns * width + width / 2)) / 2 + width / 2;
  const startY = (SCENE_HEIGHT - ((rows - 1) * radius * 1.5 + radius * 2)) / 2 + radius;
  const lookup = new Map<string, number>();
  const cells: HexCell[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const center: Point = [
        startX + column * width + (row % 2 ? width / 2 : 0),
        startY + row * radius * 1.5
      ];
      lookup.set(`${column}:${row}`, cells.length);
      cells.push({
        center,
        ring: createRegularPolygon(center[0], center[1], radius, 6, Math.PI / 6),
        neighbors: []
      });
    }
  }
  // Edge k faces SE, SW, W, NW, NE, E in an odd-row-shifted layout.
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const odd = row % 2;
      const offsets = [
        [odd, 1],
        [odd - 1, 1],
        [-1, 0],
        [odd - 1, -1],
        [odd, -1],
        [1, 0]
      ];
      cells[lookup.get(`${column}:${row}`)!].neighbors = offsets.map(
        ([deltaColumn, deltaRow]) => lookup.get(`${column + deltaColumn}:${row + deltaRow}`) ?? -1
      );
    }
  }
  return cells;
}

/** Points scattered in a thick arc, like a smile or a crescent. */
function createCrescentPoints(
  random: Random,
  center: Point,
  radius: number,
  thickness: number,
  startAngle: number,
  endAngle: number,
  count: number
): Point[] {
  return Array.from({length: count}, () => {
    const angle = random.range(startAngle, endAngle);
    const distance = radius - random.next() * thickness;
    return [center[0] + Math.cos(angle) * distance, center[1] + Math.sin(angle) * distance];
  });
}

/** The alpha-shape outline of a crescent; `loosen` opens it toward the convex hull. */
function createCrescentHull(
  center: Point,
  radius: number,
  thickness: number,
  startAngle: number,
  endAngle: number,
  loosen: number
): Ring {
  const pad = 3;
  const capAngle = pad / radius;
  const innerRadius = (radius - thickness - pad) * (1 - loosen);
  const ring: Ring = [];
  const steps = 14;
  for (let step = 0; step <= steps; step++) {
    const angle = mixNumbers(startAngle - capAngle, endAngle + capAngle, step / steps);
    ring.push([
      center[0] + Math.cos(angle) * (radius + pad),
      center[1] + Math.sin(angle) * (radius + pad)
    ]);
  }
  for (let step = steps; step >= 0; step--) {
    const angle = mixNumbers(startAngle - capAngle, endAngle + capAngle, step / steps);
    ring.push([
      center[0] + Math.cos(angle) * innerRadius,
      center[1] + Math.sin(angle) * innerRadius
    ]);
  }
  return ring;
}

function createRotatedCluster(
  random: Random,
  center: Point,
  spreadMajor: number,
  spreadMinor: number,
  rotation: number,
  count: number
): Point[] {
  return Array.from({length: count}, () => {
    const major = random.gaussian() * spreadMajor;
    const minor = random.gaussian() * spreadMinor;
    return [
      clamp(
        center[0] + major * Math.cos(rotation) - minor * Math.sin(rotation),
        5,
        SCENE_WIDTH - 5
      ),
      clamp(
        center[1] + major * Math.sin(rotation) + minor * Math.cos(rotation),
        5,
        SCENE_HEIGHT - 5
      )
    ];
  });
}

const NO_POINTER = 'none' as const;
const HOVER_POINTER = 'hover' as const;

export const GEOMETRY_CONSTRUCTION_OVERLAY_PREVIEWS: Record<string, EntryPreview> = {
  // ===== Buffers and offsets =====
  'buffer-surface': {
    parameters: [createSlider('distance', 'Buffer distance', 2, 30, 1, 12, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const rawLine = createWanderPath(fork('line'), [8, 74], -0.3, 10, 13, 0.3).map(
        ([x, y]): Point => [x, clamp(y, 14, 86)]
      );
      const line = smoothPath(rawLine, 1);
      const parcel = createBlob(fork('parcel'), 118, 28, 13, 9, 0.22);
      const columns = 64;
      const rows = 40;
      const distance = p.distance;
      const values = evaluateField(columns, rows, (unitX, unitY) => {
        const point: Point = [unitX * SCENE_WIDTH, unitY * SCENE_HEIGHT];
        let best = Infinity;
        for (let index = 0; index + 1 < line.length; index++) {
          best = Math.min(best, getSegmentDistance(point, line[index], line[index + 1]));
        }
        if (isInsideRing(point, parcel)) return 1;
        for (let index = 0; index < parcel.length; index++) {
          best = Math.min(
            best,
            getSegmentDistance(point, parcel[index], parcel[(index + 1) % parcel.length])
          );
        }
        // 1 on the source geometry, falling to 0 at the buffer distance and staying dark beyond it.
        return best >= distance ? 0 : 1 - best / distance;
      });
      return [
        {
          type: 'field',
          columns,
          rows,
          values,
          ramp: 'ice',
          style: 'contours',
          levels: 5,
          speed: 0.2
        },
        createGridLayer(),
        createFills([createPolygon(parcel, 'warm', 0.18)], 'static'),
        createStrokes(
          [createPath(line, 'cool', 1.3, 0.95), createLoop(parcel, 'warm', 1, 0.95)],
          'static'
        ),
        createPulseSprites(
          toSprites(
            line.filter((_, index) => index % 3 === 0),
            'ink',
            1.1
          ),
          0.1
        )
      ];
    }
  },

  'offset-curve': {
    parameters: [createSlider('distance', 'Signed offset', -14, 14, 1, 8, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const street = createStreets(fork('streets'))[0];
      const offset = offsetPath(street, p.distance);
      const ghost = offsetPath(street, -p.distance);
      const ticks = street.map((point, index) =>
        createPath([point, offset[index]], 'dim', 0.35, 0.7)
      );
      return [
        createGridLayer(),
        createStrokes([createPath(ghost, 'dim', 0.5, 0.35)], 'static'),
        createStrokes(ticks, 'static'),
        createStrokes([createPath(street, 'primary', 1.2, 0.9)], 'static'),
        createStrokes([createPath(offset, 'warm', 1.5, 0.95)], 'flow', 0.3, 1.3),
        createPulseSprites(toSprites(offset, 'warm', 1.2), 0.2)
      ];
    }
  },

  'outline-geometry': {
    parameters: [createSlider('radius', 'Outline radius', 2, 12, 0.5, 6, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('shapes');
      const line: Point[] = [
        [14, 72],
        [34, 46],
        [58, 58],
        [80, 30],
        [108, 40],
        [134, 18]
      ];
      const ring = createRegularPolygon(112, 72, 16, 3, random.range(0, 1));
      const points: Point[] = [
        [40, 18],
        [64, 80],
        [146, 52]
      ];
      const segments = [
        ...line.slice(0, -1).map((point, index) => [point, line[index + 1]] as Point[]),
        ...ring.map((point, index) => [point, ring[(index + 1) % ring.length]] as Point[])
      ];
      const vertices = [...line, ...ring, ...points];
      return [
        createGridLayer(),
        // Each segment is its own stroke and each vertex its own disc, so overlaps glow brighter.
        createStrokes(
          segments.map(segment => createPath(segment, 'cool', p.radius * 2, 0.22)),
          'pulse',
          0.25
        ),
        createPulseSprites(
          vertices.map((vertex, index) =>
            createDot(vertex, 'cool', p.radius * 1.15, 0.3, index / vertices.length)
          ),
          0.25
        ),
        createStrokes(
          [createPath(line, 'ink', 0.6, 0.9), createLoop(ring, 'ink', 0.6, 0.9)],
          'static'
        ),
        createPulseSprites(toSprites(vertices, 'warm', 1.3), 0.1)
      ];
    }
  },

  // ===== Overlay and coverage =====
  'polygon-overlay': {
    parameters: [
      createChoice('operation', 'Operation', ['Intersect', 'Union', 'Difference', 'Symmetric'], 0)
    ],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const shape = fork('shape');
      const first = createEllipse(66, 50, 38, 27, shape.range(-0.5, 0));
      const secondCenter: Point = pointer
        ? [clamp(pointer[0], 40, 130), clamp(pointer[1], 26, 74)]
        : [98, 54];
      const second = createEllipse(secondCenter[0], secondCenter[1], 34, 29, shape.range(0.2, 0.7));
      const firstBoundary = clipPathByRing([...first, first[0]], second);
      const secondBoundary = clipPathByRing([...second, second[0]], first);
      const firstOnly = subtractConvex(first, second);
      const secondOnly = subtractConvex(second, first);
      const operation = Math.round(p.operation);
      let fills: Ring[] = [];
      let boundary: Point[][] = [];
      if (operation === 0) {
        fills = [intersectConvex(first, second)];
        boundary = [...firstBoundary.inside, ...secondBoundary.inside];
      } else if (operation === 1) {
        fills = [...firstOnly, second];
        boundary = [...firstBoundary.outside, ...secondBoundary.outside];
      } else if (operation === 2) {
        fills = firstOnly;
        boundary = [...firstBoundary.outside, ...secondBoundary.inside];
      } else {
        fills = [...firstOnly, ...secondOnly];
        boundary = [
          ...firstBoundary.outside,
          ...firstBoundary.inside,
          ...secondBoundary.outside,
          ...secondBoundary.inside
        ];
      }
      return [
        createGridLayer(),
        createStrokes([
          createLoop(first, 'primary', 0.6, 0.45),
          createLoop(second, 'violet', 0.6, 0.45)
        ]),
        createFills(fills.map((ring, index) => createPolygon(ring, 'warm', 0.42, index * 0.07))),
        createStrokes(
          boundary.map(path => createPath(path, 'hot', 1.3, 0.95)),
          'flow',
          0.25,
          1.4
        )
      ];
    }
  },

  'coverage-dissolve': {
    parameters: [createSlider('dissolve', 'Dissolve', 0, 100, 5, 100, '%')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const cells = createHexCoverage();
      const random = fork('districts');
      const seeds: Point[] = [
        [random.range(26, 44), random.range(26, 44)],
        [random.range(100, 130), random.range(24, 44)],
        [random.range(60, 100), random.range(68, 82)]
      ];
      const labels = cells.map(cell => {
        let best = 0;
        seeds.forEach((seed, index) => {
          if (getDistance(seed, cell.center) < getDistance(seeds[best], cell.center)) best = index;
        });
        return best;
      });
      const roles: Role[] = ['cool', 'warm', 'violet'];
      const interior: PathItem[] = [];
      const boundary: PathItem[] = [];
      const keep = 1 - p.dissolve / 100;
      cells.forEach((cell, index) => {
        cell.neighbors.forEach((neighbor, edge) => {
          const segment: Point[] = [cell.ring[edge], cell.ring[(edge + 1) % 6]];
          if (neighbor < 0 || labels[neighbor] !== labels[index]) {
            if (neighbor < index && neighbor >= 0) return;
            boundary.push(createPath(segment, 'hot', 1.4, 0.95));
          } else if (neighbor > index) {
            interior.push(createPath(segment, 'ink', 0.5, 0.75 * keep));
          }
        });
      });
      return [
        createFills(
          cells.map((cell, index) =>
            createPolygon(cell.ring, roles[labels[index]], 0.28, labels[index] / 3)
          )
        ),
        createStrokes(interior, 'static'),
        createStrokes(boundary, 'flow', 0.2, 1.2)
      ];
    }
  },

  'coverage-simplification': {
    parameters: [createSlider('tolerance', 'Tolerance', 0, 6, 0.25, 2.5, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const junction: Point = [80, 52];
      const angles = [-Math.PI / 2, Math.PI / 6, (5 * Math.PI) / 6];
      const outer = angles.map(
        (angle): Point => [80 + Math.cos(angle) * 42, 52 + Math.sin(angle) * 42]
      );
      const borders = outer.map((end, index) => {
        const random = fork(`border-${index}`);
        const direction: Point = [end[0] - junction[0], end[1] - junction[1]];
        const length = Math.hypot(...direction);
        const normal: Point = [-direction[1] / length, direction[0] / length];
        return Array.from({length: 26}, (_, step): Point => {
          const amount = step / 25;
          const wobble =
            step === 0 || step === 25
              ? 0
              : Math.sin(step * 0.9 + index) * 4.5 + random.range(-1.1, 1.1);
          return [
            junction[0] + direction[0] * amount + normal[0] * wobble,
            junction[1] + direction[1] * amount + normal[1] * wobble
          ];
        });
      });
      const simplified = borders.map(border => simplifyDouglasPeucker(border, p.tolerance));
      const regions = angles.map((angle, index) => {
        const next = (index + 1) % 3;
        const nextAngle = angles[next] + (next === 0 ? Math.PI * 2 : 0);
        const arc: Ring = Array.from({length: 10}, (_, step): Point => {
          const value = mixNumbers(angle, nextAngle, step / 9);
          return [80 + Math.cos(value) * 42, 52 + Math.sin(value) * 42];
        });
        return [...simplified[index], ...arc, ...[...simplified[next]].reverse()];
      });
      const kept = simplified.flat();
      return [
        createFills(
          regions.map((ring, index) =>
            createPolygon(ring, (['cool', 'violet', 'primary'] as Role[])[index], 0.22, index / 3)
          )
        ),
        createStrokes(borders.map(border => createPath(border, 'dim', 0.5, 0.6))),
        createStrokes(
          simplified.map(border => createPath(border, 'warm', 1.5, 0.95)),
          'flow',
          0.22,
          1.3
        ),
        createPulseSprites(toSprites(kept, 'hot', 1.5), 0.2)
      ];
    }
  },

  'coverage-validity': {
    parameters: [createSlider('size', 'Defect size', 0.5, 2, 0.1, 1)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('lattice');
      const columns = 4;
      const rows = 3;
      // Shared lattice: neighbouring parcels reuse the same corner points, so valid edges coincide.
      const lattice: Point[][] = [];
      for (let row = 0; row <= rows; row++) {
        const line: Point[] = [];
        for (let column = 0; column <= columns; column++) {
          const isBorder = row === 0 || row === rows || column === 0 || column === columns;
          line.push([
            10 + column * 35 + (isBorder ? 0 : random.range(-3, 3)),
            9 + row * 27.3 + (isBorder ? 0 : random.range(-3, 3))
          ]);
        }
        lattice.push(line);
      }
      const corner = (column: number, row: number): Point => lattice[row][column];
      const gapDepth = 4 * p.size;
      const overlapDepth = 5 * p.size;
      const gapMiddle = mixPoints(corner(1, 0), corner(1, 1), 0.5);
      const gapTip: Point = [gapMiddle[0] + gapDepth, gapMiddle[1]];
      const overlapMiddle = mixPoints(corner(3, 1), corner(3, 2), 0.5);
      const overlapTip: Point = [overlapMiddle[0] - overlapDepth, overlapMiddle[1]];
      const mismatchPoint = mixPoints(corner(1, 2), corner(2, 2), 0.45);
      const cells: Ring[] = [];
      const cellRoles: Role[] = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const ring: Ring = [
            corner(column, row),
            corner(column + 1, row),
            corner(column + 1, row + 1),
            corner(column, row + 1)
          ];
          if (column === 1 && row === 0) ring.push(gapTip);
          if (column === 3 && row === 1) ring.push(overlapTip);
          // The upper parcel carries an extra vertex on the shared edge that its neighbour lacks.
          if (column === 1 && row === 1) ring.splice(3, 0, mismatchPoint);
          cells.push(ring);
          cellRoles.push((row + column) % 2 === 0 ? 'cool' : 'primary');
        }
      }
      const gapSliver: Ring = [corner(1, 0), gapTip, corner(1, 1)];
      const overlapSliver: Ring = [corner(3, 1), overlapTip, corner(3, 2)];
      const mismatchEdge: Point[] = [corner(1, 2), mismatchPoint, corner(2, 2)];
      return [
        createFills(
          cells.map((ring, index) => createPolygon(ring, cellRoles[index], 0.1, index / 12)),
          'static'
        ),
        createStrokes(
          cells.map(ring => createLoop(ring, 'cool', 0.55, 0.6)),
          'static'
        ),
        createFills([
          createPolygon(gapSliver, 'hot', 0.9),
          createPolygon(overlapSliver, 'hot', 0.9, 0.33)
        ]),
        createStrokes(
          [
            createPath(mismatchEdge, 'hot', 1.3, 1, 0.66),
            createLoop(gapSliver, 'hot', 0.9, 1),
            createLoop(overlapSliver, 'hot', 0.9, 1, 0.33)
          ],
          'pulse',
          0.6
        ),
        createPulseSprites(
          [
            createDot(mixPoints(gapMiddle, gapTip, 0.5), 'hot', 7, 0.55),
            createDot(mixPoints(overlapMiddle, overlapTip, 0.5), 'hot', 7, 0.55, 0.33),
            createDot(mismatchPoint, 'hot', 6, 0.8, 0.66)
          ],
          0.5
        )
      ];
    }
  },

  'segment-ring-assembly': {
    parameters: [createSlider('assembled', 'Assembled', 0, 100, 5, 70, '%')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('rings');
      const makeRing = (center: Point, radiusX: number, radiusY: number, count: number): Ring => {
        const phase = random.range(0, 1);
        return Array.from({length: count}, (_, index): Point => {
          const angle = ((index + phase) / count) * Math.PI * 2;
          const wobble = 1 + 0.14 * Math.sin(angle * 3 + phase * 6) + random.range(-0.06, 0.06);
          return [
            center[0] + Math.cos(angle) * radiusX * wobble,
            center[1] + Math.sin(angle) * radiusY * wobble
          ];
        });
      };
      const shellA = makeRing([46, 50], 37, 38, 14);
      const holeA = shellA.map((point): Point => mixPoints([46, 50], point, 0.42));
      const shellB = makeRing([114, 28], 32, 20, 11);
      const shellC = makeRing([116, 74], 30, 19, 12);
      const holeC = shellC.map((point): Point => mixPoints([116, 74], point, 0.4));
      const rings: {ring: Ring; role: Role; order: number}[] = [
        {ring: shellA, role: 'warm', order: 0},
        {ring: holeA, role: 'violet', order: 1},
        {ring: shellB, role: 'warm', order: 2},
        {ring: shellC, role: 'warm', order: 3},
        {ring: holeC, role: 'violet', order: 4}
      ];
      const loose: PathItem[] = [];
      const heads: SpriteItem[] = [];
      const chained: PathItem[] = [];
      const joints: SpriteItem[] = [];
      const fills: PolygonItem[] = [];
      const progressRatio = p.assembled / 100;
      for (const {ring, role, order} of rings) {
        const count = ring.length;
        const progress = clamp(progressRatio * 1.5 - order * 0.1, 0, 1);
        const joined = Math.round(progress * count);
        if (joined > 0) {
          const closed = [...ring, ring[0]];
          chained.push(createPath(closed.slice(0, joined + 1), role, 1.5, 0.95, order / 5));
          for (let index = 1; index < joined; index++) {
            joints.push(createDot(ring[index], role, 1.3, 0.9, index / count));
          }
        }
        for (let index = joined; index < count; index++) {
          const start = ring[index];
          const end = ring[(index + 1) % count];
          loose.push(
            createPath(
              [mixPoints(start, end, 0.12), mixPoints(start, end, 0.8)],
              'primary',
              1,
              0.85
            )
          );
          heads.push(createDot(mixPoints(start, end, 0.8), role, 1.4, 0.95, index / count));
          heads.push(createDot(mixPoints(start, end, 0.12), role, 1.4, 0.95, index / count));
        }
      }
      // Shell with its hole: fill the ring between them once both are fully chained.
      if (progressRatio * 1.5 >= 1.1) {
        for (let index = 0; index < shellA.length; index++) {
          const next = (index + 1) % shellA.length;
          fills.push(
            createPolygon([shellA[index], shellA[next], holeA[next], holeA[index]], 'warm', 0.2)
          );
        }
      }
      if (progressRatio * 1.5 >= 1.2) fills.push(createPolygon(shellB, 'warm', 0.18));
      if (progressRatio * 1.5 >= 1.3) {
        for (let index = 0; index < shellC.length; index++) {
          const next = (index + 1) % shellC.length;
          fills.push(
            createPolygon([shellC[index], shellC[next], holeC[next], holeC[index]], 'warm', 0.2)
          );
        }
      }
      // Stray input segments that belong to no ring yet.
      const strays: PathItem[] = [];
      for (let index = 0; index < 16; index++) {
        const start: Point = [random.range(6, 154), random.range(6, 94)];
        const angle = random.range(0, Math.PI);
        const length = random.range(4, 8);
        strays.push(
          createPath(
            [start, [start[0] + Math.cos(angle) * length, start[1] + Math.sin(angle) * length]],
            'dim',
            0.7,
            0.5,
            random.next()
          )
        );
      }
      return [
        createFills(fills, 'static'),
        createStrokes(strays, 'draw', 0.15),
        createStrokes(loose, 'pulse', 0.4),
        createPulseSprites(heads, 0.4),
        createStrokes(chained, 'flow', 0.25, 1.4),
        createPulseSprites(joints, 0.1)
      ];
    }
  },

  // ===== Hulls and group geometry =====
  'concave-hull': {
    parameters: [createSlider('loosen', 'Looseness', 0, 1, 0.05, 0)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const center: Point = [80, 30];
      const points = createCrescentPoints(
        fork('points'),
        center,
        40,
        13,
        0.08 * Math.PI,
        0.92 * Math.PI,
        90
      );
      const convex = getConvexHull(points);
      const concave = createCrescentHull(center, 40, 13, 0.08 * Math.PI, 0.92 * Math.PI, p.loosen);
      return [
        createGridLayer(),
        createStrokes([createLoop(convex, 'dim', 0.6, 0.6)]),
        createFills([createPolygon(concave, 'warm', 0.24)]),
        createStrokes([createLoop(concave, 'hot', 1.3, 0.95)], 'flow', 0.22, 1.4),
        createPulseSprites(toSprites(points, 'cool', 1.3, fork('radius')), 0.15)
      ];
    }
  },

  'group-concave-hull': {
    parameters: [createSlider('loosen', 'Looseness', 0, 1, 0.05, 0)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const groups = [
        {
          center: [40, 30] as Point,
          radius: 24,
          thickness: 9,
          start: 0.1 * Math.PI,
          end: 1.0 * Math.PI,
          role: 'cool' as Role
        },
        {
          center: [112, 36] as Point,
          radius: 26,
          thickness: 9,
          start: 1.05 * Math.PI,
          end: 1.95 * Math.PI,
          role: 'warm' as Role
        },
        {
          center: [86, 78] as Point,
          radius: 17,
          thickness: 7,
          start: -0.7 * Math.PI,
          end: 0.55 * Math.PI,
          role: 'violet' as Role
        }
      ];
      const hulls = groups.map(group =>
        createCrescentHull(
          group.center,
          group.radius,
          group.thickness,
          group.start,
          group.end,
          p.loosen
        )
      );
      const noiseRandom = fork('noise-points');
      const noisePoints = Array.from(
        {length: 8},
        (): Point => [noiseRandom.range(8, 152), noiseRandom.range(8, 92)]
      );
      return [
        createGridLayer(),
        createFills(
          hulls.map((ring, index) => createPolygon(ring, groups[index].role, 0.2, index / 3))
        ),
        createStrokes(
          hulls.map((ring, index) => createLoop(ring, groups[index].role, 1.2, 0.95)),
          'flow',
          0.2,
          1.2
        ),
        ...groups.map((group, index) =>
          createPulseSprites(
            toSprites(
              createCrescentPoints(
                fork(`group-${index}`),
                group.center,
                group.radius,
                group.thickness,
                group.start,
                group.end,
                34
              ),
              group.role,
              1.2
            ),
            0.15
          )
        ),
        createPulseSprites(toSprites(noisePoints, 'dim', 1, undefined, 0.5), 0)
      ];
    }
  },

  'group-convex-hull': {
    parameters: [createSlider('spread', 'Cluster spread', 4, 14, 0.5, 8, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const centers: Point[] = [
        [38, 34],
        [108, 30],
        [124, 72],
        [58, 74]
      ];
      const rotations = [0.5, -0.4, 0.9, -0.1];
      const clusters = centers.map((center, index) =>
        createRotatedCluster(
          fork(`cluster-${index}`),
          center,
          p.spread,
          p.spread * 0.55,
          rotations[index],
          26
        )
      );
      const hulls = clusters.map(getConvexHull);
      const noiseRandom = fork('noise');
      const noise = Array.from(
        {length: 7},
        (): Point => [noiseRandom.range(8, 152), noiseRandom.range(8, 92)]
      );
      return [
        createGridLayer(),
        createFills(
          hulls.map((ring, index) => createPolygon(ring, getRoleCycle(index + 1), 0.16, index / 4))
        ),
        createStrokes(
          hulls.map((ring, index) => createLoop(ring, getRoleCycle(index + 1), 1.1, 0.95)),
          'flow',
          0.2,
          1.3
        ),
        ...clusters.map((points, index) =>
          createPulseSprites(toSprites(points, getRoleCycle(index + 1), 1.1), 0.1)
        ),
        createPulseSprites(
          hulls.flatMap((ring, index) =>
            toSprites(ring, 'ink', 1.7, undefined, 0.9).map(item => ({
              ...item,
              phase: index / 4
            }))
          ),
          0.3
        ),
        createPulseSprites(toSprites(noise, 'dim', 1), 0)
      ];
    }
  },

  'group-geometry': {
    parameters: [createSlider('spread', 'Cluster spread', 5, 14, 0.5, 9, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const specs = [
        {center: [40, 36] as Point, rotation: 0.6, role: 'cool' as Role},
        {center: [108, 34] as Point, rotation: -0.5, role: 'warm' as Role},
        {center: [86, 74] as Point, rotation: 0.1, role: 'violet' as Role}
      ];
      const strokes: PathItem[] = [];
      const ellipses: PathItem[] = [];
      const layers: PreviewLayer[] = [createGridLayer()];
      const centroids: SpriteItem[] = [];
      const medoids: SpriteItem[] = [];
      specs.forEach((spec, index) => {
        const points = createRotatedCluster(
          fork(`group-${index}`),
          spec.center,
          p.spread * 1.3,
          p.spread * 0.55,
          spec.rotation,
          30
        );
        const [minimumX, minimumY, maximumX, maximumY] = getRingBounds(points);
        const [meanX, meanY] = getVertexMean(points);
        let varianceX = 0;
        let varianceY = 0;
        let covariance = 0;
        let medoid = points[0];
        for (const point of points) {
          varianceX += (point[0] - meanX) ** 2 / points.length;
          varianceY += (point[1] - meanY) ** 2 / points.length;
          covariance += ((point[0] - meanX) * (point[1] - meanY)) / points.length;
          if (getDistance(point, [meanX, meanY]) < getDistance(medoid, [meanX, meanY]))
            medoid = point;
        }
        const angle = 0.5 * Math.atan2(2 * covariance, varianceX - varianceY);
        const root = Math.sqrt(((varianceX - varianceY) / 2) ** 2 + covariance ** 2);
        const major = Math.sqrt(Math.max((varianceX + varianceY) / 2 + root, 0.01));
        const minor = Math.sqrt(Math.max((varianceX + varianceY) / 2 - root, 0.01));
        strokes.push(
          createLoop(
            createRectangle(minimumX, minimumY, maximumX - minimumX, maximumY - minimumY),
            'dim',
            0.5,
            0.65
          )
        );
        ellipses.push(
          createLoop(
            createEllipse(meanX, meanY, major * 1.6, minor * 1.6, angle, 32),
            spec.role,
            1.2,
            0.95,
            index / 3
          )
        );
        centroids.push(createDot([meanX, meanY], 'hot', 2.6, 1, index / 3));
        medoids.push(createDot(medoid, 'ink', 1.5, 0.95));
        layers.push(createPulseSprites(toSprites(points, spec.role, 1.1, undefined, 0.75), 0.1));
      });
      layers.push(
        createStrokes(strokes),
        createStrokes(ellipses, 'flow', 0.2, 1.2),
        createPulseSprites(medoids, 0),
        createPulseSprites(centroids, 0.35)
      );
      return layers;
    }
  },

  // ===== Triangulation =====
  'delaunay-tessellation': {
    parameters: [createSlider('sites', 'Sites', 10, 60, 2, 30)],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const random = fork('sites');
      const sites = Array.from(
        {length: Math.round(p.sites)},
        (): Point => [random.range(8, 152), random.range(8, 92)]
      );
      const triangles = triangulatePoints(sites);
      const probe: Point = pointer ?? [86, 46];
      let selected = 0;
      let selectedDistance = Infinity;
      triangles.forEach(([first, second, third], index) => {
        const centroid: Point = [
          (sites[first][0] + sites[second][0] + sites[third][0]) / 3,
          (sites[first][1] + sites[second][1] + sites[third][1]) / 3
        ];
        const distance = isInsideTriangle(probe, sites[first], sites[second], sites[third])
          ? -1
          : getDistance(centroid, probe);
        if (distance < selectedDistance) {
          selectedDistance = distance;
          selected = index;
        }
      });
      const layers: PreviewLayer[] = [
        createFills(
          triangles.map(([first, second, third], index) =>
            createPolygon(
              [sites[first], sites[second], sites[third]],
              getRoleCycle(index),
              0.1,
              (index % 9) / 9
            )
          )
        ),
        createStrokes(
          [...getTriangleEdges(triangles).values()].map(([from, to]) =>
            createPath([sites[from], sites[to]], 'primary', 0.55, 0.65)
          ),
          'static'
        )
      ];
      if (triangles[selected]) {
        const [first, second, third] = triangles[selected];
        const circle = getCircumcircle(sites[first], sites[second], sites[third]);
        const center: Point = [circle.x, circle.y];
        const radius = Math.sqrt(circle.radiusSquared);
        layers.push(
          createFills([createPolygon([sites[first], sites[second], sites[third]], 'warm', 0.4)]),
          createStrokes(
            [createLoop(createCircle(center[0], center[1], radius, 48), 'warm', 0.9, 0.9)],
            'flow',
            0.3,
            1.2
          ),
          createPulseSprites([createDot(center, 'warm', 1.6, 0.9)], 0.3)
        );
      }
      layers.push(createPulseSprites(toSprites(sites, 'ink', 1.4), 0.1));
      return layers;
    }
  },

  'voronoi-diagram': {
    parameters: [createSlider('sites', 'Sites', 8, 60, 2, 26)],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const random = fork('sites');
      const sites = Array.from(
        {length: Math.round(p.sites)},
        (): Point => [random.range(8, 152), random.range(8, 92)]
      );
      const probe: Point = pointer ?? [84, 50];
      const allSites = [...sites, probe];
      const bounds = createRectangle(3, 3, 154, 94);
      const cells = allSites.map(site => {
        let ring = bounds;
        for (const other of allSites) {
          if (other === site) continue;
          ring = clipRingByHalfPlane(ring, mixPoints(site, other, 0.5), [
            site[0] - other[0],
            site[1] - other[1]
          ]);
          if (ring.length < 3) break;
        }
        return ring;
      });
      const probeCell = cells[cells.length - 1];
      return [
        createFills(
          cells
            .slice(0, -1)
            .map((ring, index) => createPolygon(ring, getRoleCycle(index), 0.13, (index % 8) / 8))
        ),
        createFills([createPolygon(probeCell, 'warm', 0.4)]),
        createStrokes(cells.slice(0, -1).map(ring => createLoop(ring, 'primary', 0.6, 0.7))),
        createStrokes([createLoop(probeCell, 'warm', 1.2, 0.95)], 'flow', 0.3, 1.2),
        createPulseSprites(toSprites(sites, 'ink', 1.4), 0.1),
        createPulseSprites([createDot(probe, 'warm', 2.6)], 0.3)
      ];
    }
  },

  'polygon-triangulation': {
    parameters: [createSlider('vertices', 'Outline vertices', 8, 24, 1, 14)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const count = Math.round(p.vertices);
      const outer = createBlob(fork('outline'), 80, 50, 40, count, 0.3, 0.82);
      const hole = createCircle(80, 50, 11, 9);
      // Bridge the hole to the outline along the +X ray; both start at angle zero.
      const merged: Ring = [...outer, outer[0], hole[0], ...hole.slice(1).reverse(), hole[0]];
      const triangles = clipEars(merged);
      const edges = getTriangleEdges(triangles);
      return [
        createFills(
          triangles.map(([first, second, third], index) =>
            createPolygon(
              [merged[first], merged[second], merged[third]],
              getRoleCycle(index),
              0.34,
              index / Math.max(triangles.length, 1)
            )
          )
        ),
        createStrokes(
          [...edges.values()].map(([from, to]) =>
            createPath([merged[from], merged[to]], 'ink', 0.5, 0.75)
          ),
          'draw',
          0.12
        ),
        createStrokes(
          [createLoop(outer, 'hot', 1.1, 0.95), createLoop(hole, 'violet', 1.1, 0.95)],
          'static'
        ),
        createPulseSprites(toSprites(outer, 'ink', 1.2), 0.1)
      ];
    }
  },

  'constrained-delaunay-triangulation': {
    parameters: [createSlider('flips', 'Edge flips', 0, 40, 1, 40)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const outline = createBlob(fork('outline'), 80, 50, 42, 16, 0.34, 0.74);
      const initial = clipEars(outline);
      const before = getTriangleEdges(initial);
      const {triangles, flipped} = flipTowardDelaunay(outline, initial, Math.round(p.flips));
      const after = getTriangleEdges(triangles);
      const removed = [...before.entries()].filter(([key]) => !after.has(key));
      return [
        createFills(
          triangles.map(([first, second, third], index) =>
            createPolygon(
              [outline[first], outline[second], outline[third]],
              getRoleCycle(index),
              0.16,
              index / triangles.length
            )
          )
        ),
        createStrokes(
          removed.map(([, [from, to]]) =>
            createPath([outline[from], outline[to]], 'violet', 0.5, 0.2)
          )
        ),
        createStrokes(
          [...after.entries()]
            .filter(([key]) => !flipped.has(key))
            .map(([, [from, to]]) => createPath([outline[from], outline[to]], 'primary', 0.7, 0.8))
        ),
        createStrokes(
          [...after.entries()]
            .filter(([key]) => flipped.has(key))
            .map(([, [from, to]]) => createPath([outline[from], outline[to]], 'hot', 1.3, 0.95)),
          'pulse',
          0.5
        ),
        createStrokes([createLoop(outline, 'ink', 1, 0.9)]),
        createPulseSprites(toSprites(outline, 'ink', 1.2), 0.1)
      ];
    }
  },

  // ===== Lines =====
  'line-segmentize': {
    parameters: [createSlider('length', 'Maximum length', 4, 30, 1, 12, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const streets = createStreets(fork('streets')).slice(0, 2);
      const original: SpriteItem[] = [];
      const inserted: SpriteItem[] = [];
      for (const street of streets) {
        street.forEach(point => original.push(createDot(point, 'primary', 2.2, 0.95)));
        for (let index = 0; index + 1 < street.length; index++) {
          const pieces = Math.ceil(getDistance(street[index], street[index + 1]) / p.length);
          for (let piece = 1; piece < pieces; piece++) {
            inserted.push(
              createDot(
                mixPoints(street[index], street[index + 1], piece / pieces),
                'warm',
                1.1,
                0.9,
                piece / pieces
              )
            );
          }
        }
      }
      return [
        createGridLayer(),
        createStrokes(
          streets.map(street => createPath(street, 'primary', 0.9, 0.7)),
          'flow',
          0.18,
          1
        ),
        createPulseSprites(inserted, 0.3),
        createPulseSprites(original, 0.1)
      ];
    }
  },

  'great-circle-arcs': {
    parameters: [createSlider('segments', 'Segments per arc', 3, 24, 1, 10)],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const random = fork('destinations');
      const hub: Point = pointer ?? [36, 64];
      const destinations = Array.from(
        {length: 22},
        (): Point => [random.range(60, 152), random.range(12, 88)]
      );
      const segments = Math.round(p.segments);
      const arcs = destinations.map(destination => {
        const distance = getDistance(hub, destination);
        const control: Point = [
          (hub[0] + destination[0]) / 2,
          (hub[1] + destination[1]) / 2 - distance * 0.34
        ];
        return Array.from({length: segments + 1}, (_, step): Point => {
          const amount = step / segments;
          const inverse = 1 - amount;
          return [
            inverse * inverse * hub[0] +
              2 * inverse * amount * control[0] +
              amount * amount * destination[0],
            inverse * inverse * hub[1] +
              2 * inverse * amount * control[1] +
              amount * amount * destination[1]
          ];
        });
      });
      let longest = 0;
      destinations.forEach((destination, index) => {
        if (getDistance(hub, destination) > getDistance(hub, destinations[longest]))
          longest = index;
      });
      const graticule: PathItem[] = [];
      for (let latitude = 18; latitude < 100; latitude += 26) {
        graticule.push(
          createPath(
            Array.from(
              {length: 17},
              (_, step): Point => [step * 10, latitude + Math.sin((step / 16) * Math.PI) * -6]
            ),
            'dim',
            0.3,
            0.3
          )
        );
      }
      for (let longitude = 20; longitude < 160; longitude += 30) {
        graticule.push(
          createPath(
            Array.from(
              {length: 9},
              (_, step): Point => [
                longitude + Math.sin((step / 8) * Math.PI) * (longitude < 80 ? -5 : 5),
                step * 12.5
              ]
            ),
            'dim',
            0.3,
            0.3
          )
        );
      }
      return [
        createStrokes(graticule),
        createStrokes(
          arcs.map(arc => createPath(arc, 'cool', 0.6, 0.7)),
          'flow',
          0.2,
          1.1
        ),
        createStrokes([createPath(arcs[longest], 'warm', 1.2, 0.95)], 'flow', 0.3, 1.5),
        createPulseSprites(toSprites(arcs[longest], 'warm', 1.1), 0.2),
        createPulseSprites(toSprites(destinations, 'hot', 1.5), 0.25),
        createPulseSprites([createDot(hub, 'warm', 3.8)], 0.3)
      ];
    }
  },

  'line-smooth': {
    parameters: [createSlider('rounds', 'Chaikin rounds', 0, 4, 1, 2)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('shape');
      const zigzag: Point[] = [
        [10, 74],
        [28, 40],
        [46, 70],
        [64, 34],
        [84, 62],
        [104, 28],
        [124, 56],
        [150, 24]
      ].map(([x, y]): Point => [x + random.range(-2, 2), y + random.range(-3, 3)]);
      const square = createRegularPolygon(116, 78, 15, 5, random.range(0, 1)).map(
        ([x, y]): Point => [clamp(x, 6, 154), clamp(y, 6, 94)]
      );
      const rounds = Math.round(p.rounds);
      const smoothed = smoothPath(zigzag, rounds);
      const smoothedRing = smoothRing(square, rounds);
      return [
        createGridLayer(),
        createStrokes([createPath(zigzag, 'dim', 0.6, 0.7), createLoop(square, 'dim', 0.6, 0.7)]),
        createStrokes(
          [createPath(smoothed, 'hot', 1.4, 0.95), createLoop(smoothedRing, 'warm', 1.4, 0.95)],
          'flow',
          0.25,
          1.4
        ),
        createPulseSprites(toSprites([...zigzag, ...square], 'primary', 1.6), 0.2)
      ];
    }
  },

  'line-chunk': {
    parameters: [createSlider('length', 'Chunk length', 8, 60, 2, 26, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const streets = createStreets(fork('streets')).slice(0, 2);
      const strokes: PathItem[] = [];
      const cuts: SpriteItem[] = [];
      let count = 0;
      for (const street of streets) {
        const total = getPathLength(street);
        for (let start = 0; start < total; start += p.length) {
          const piece = sliceRange(street, start + 0.8, Math.min(total, start + p.length) - 0.8);
          if (piece.length >= 2)
            strokes.push(createPath(piece, getRoleCycle(count), 1.5, 0.95, count / 10));
          if (start > 0)
            cuts.push(createDot(getPointAtMeasure(street, start).point, 'ink', 1.2, 0.9));
          count++;
        }
      }
      return [
        createGridLayer(),
        createStrokes(strokes, 'flow', 0.2, 0.8),
        createPulseSprites(cuts, 0.3)
      ];
    }
  },

  'line-split': {
    parameters: [createSlider('gap', 'Pull apart', 0, 4, 0.25, 1.5, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const streets = createStreets(fork('streets'));
      const strokes: PathItem[] = [];
      const nodes: SpriteItem[] = [];
      let count = 0;
      streets.forEach((street, streetIndex) => {
        const positions: number[] = [0];
        let walked = 0;
        for (let index = 0; index + 1 < street.length; index++) {
          const length = getDistance(street[index], street[index + 1]);
          streets.forEach((other, otherIndex) => {
            if (otherIndex === streetIndex) return;
            for (let edge = 0; edge + 1 < other.length; edge++) {
              const hit = getSegmentCrossing(
                street[index],
                street[index + 1],
                other[edge],
                other[edge + 1]
              );
              if (hit !== null) {
                positions.push(walked + hit * length);
                if (otherIndex > streetIndex)
                  nodes.push(
                    createDot(
                      mixPoints(street[index], street[index + 1], hit),
                      'hot',
                      3,
                      1,
                      count / 6
                    )
                  );
              }
            }
          });
          walked += length;
        }
        positions.push(walked);
        positions.sort((first, second) => first - second);
        for (let index = 0; index + 1 < positions.length; index++) {
          const piece = sliceRange(
            street,
            positions[index] + (index > 0 ? p.gap : 0),
            positions[index + 1] - (index + 2 < positions.length ? p.gap : 0)
          );
          if (piece.length >= 2)
            strokes.push(createPath(piece, getRoleCycle(count), 1.4, 0.95, count / 8));
          count++;
        }
      });
      return [
        createGridLayer(),
        createStrokes(strokes, 'flow', 0.2, 0.9),
        createPulseSprites(nodes, 0.45)
      ];
    }
  },

  'line-merge': {
    parameters: [createChoice('mode', 'Show', ['Segments', 'Chains'], 1)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('network');
      const jitter = (point: Point): Point => [
        point[0] + random.range(-1.5, 1.5),
        point[1] + random.range(-1.5, 1.5)
      ];
      const junction: Point = [86, 42];
      const chains: Point[][] = [
        [[12, 74], [38, 56], [62, 58], junction].map(([x, y], index) =>
          index === 3 ? junction : jitter([x, y])
        ),
        [
          junction,
          ...(
            [
              [112, 28],
              [146, 20]
            ] as Point[]
          ).map(jitter)
        ],
        [
          junction,
          ...(
            [
              [98, 66],
              [110, 92]
            ] as Point[]
          ).map(jitter)
        ]
      ];
      const roles: Role[] = ['primary', 'warm', 'cool'];
      const merged = Math.round(p.mode) === 1;
      const strokes: PathItem[] = [];
      const joins: SpriteItem[] = [];
      chains.forEach((chain, chainIndex) => {
        const total = getPathLength(chain);
        const pieces = 6;
        for (let piece = 0; piece < pieces; piece++) {
          const from = (total * piece) / pieces;
          const to = (total * (piece + 1)) / pieces;
          if (!merged)
            strokes.push(
              createPath(
                sliceRange(chain, from + 0.7, to - 0.7),
                getRoleCycle(chainIndex * 6 + piece),
                1.4,
                0.95,
                piece / pieces
              )
            );
          if (piece > 0)
            joins.push(
              createDot(
                getPointAtMeasure(chain, from).point,
                merged ? 'dim' : 'ink',
                merged ? 0.8 : 1.3,
                merged ? 0.5 : 0.9
              )
            );
        }
        if (merged) strokes.push(createPath(chain, roles[chainIndex], 1.8, 0.95, chainIndex / 3));
      });
      return [
        createGridLayer(),
        createStrokes(strokes, merged ? 'flow' : 'pulse', 0.2, 1.2),
        createPulseSprites(joins, 0.2),
        createPulseSprites(
          [
            createDot(junction, 'hot', 3.2),
            ...chains.map(chain => createDot(chain[chain.length - 1], 'hot', 2)),
            createDot(chains[0][0], 'hot', 2)
          ],
          0.35
        )
      ];
    }
  },

  'line-simplification': {
    parameters: [
      createSlider('tolerance', 'Tolerance', 0.2, 5, 0.1, 1.6, 'px'),
      createChoice('method', 'Method', ['Douglas-Peucker', 'Visvalingam'], 0)
    ],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('track');
      const track = Array.from({length: 76}, (_, index): Point => {
        const x = 6 + index * 2;
        return [
          x,
          50 + 20 * Math.sin(x / 21) + 8 * Math.sin(x / 6.5 + 1) + random.gaussian() * 1.3
        ];
      });
      const kept =
        Math.round(p.method) === 0
          ? simplifyDouglasPeucker(track, p.tolerance)
          : simplifyVisvalingam(track, p.tolerance * p.tolerance);
      return [
        createGridLayer(),
        createStrokes([createPath(track, 'dim', 0.5, 0.8)]),
        createPulseSprites(toSprites(track, 'dim', 0.55, undefined, 0.5), 0),
        createStrokes([createPath(kept, 'hot', 1.4, 0.95)], 'flow', 0.25, 1.4),
        createPulseSprites(toSprites(kept, 'warm', 1.7), 0.25)
      ];
    }
  },

  'line-clip-by-polygon': {
    parameters: [createChoice('mode', 'Keep', ['Inside', 'Outside'], 0)],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const streets = createStreets(fork('streets'));
      const center: Point = pointer
        ? [clamp(pointer[0], 30, 130), clamp(pointer[1], 26, 74)]
        : [82, 50];
      const district = createBlob(fork('district'), center[0], center[1], 27, 12, 0.22, 0.85);
      const keepInside = Math.round(p.mode) === 0;
      const kept: PathItem[] = [];
      const discarded: PathItem[] = [];
      const cuts: SpriteItem[] = [];
      for (const street of streets) {
        const {inside, outside} = clipPathByRing(street, district);
        for (const run of keepInside ? inside : outside)
          kept.push(createPath(run, 'warm', 1.6, 0.95));
        for (const run of keepInside ? outside : inside)
          discarded.push(createPath(run, 'dim', 0.7, 0.45));
        for (const run of inside) {
          cuts.push(
            createDot(run[0], 'hot', 1.5, 0.9),
            createDot(run[run.length - 1], 'hot', 1.5, 0.9)
          );
        }
      }
      return [
        createFills([createPolygon(district, 'cool', 0.16)]),
        createStrokes([createLoop(district, 'cool', 1, 0.9)], 'flow', 0.15, 1),
        createStrokes(discarded),
        createStrokes(kept, 'flow', 0.25, 1.1),
        createPulseSprites(cuts, 0.3)
      ];
    }
  },

  'shared-paths': {
    parameters: [createSlider('span', 'Shared vertices', 1, 5, 1, 3)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('streets');
      const base: Point[] = [
        [8, 72],
        [30, 54],
        [52, 60],
        [74, 42],
        [98, 50],
        [122, 32],
        [150, 40]
      ].map(([x, y]): Point => [x + random.range(-1.5, 1.5), y + random.range(-1.5, 1.5)]);
      const first = 1;
      const last = Math.min(base.length - 1, first + Math.round(p.span));
      const shared = base.slice(first, last + 1);
      const second: Point[] = [
        [20, 12],
        [28, 34],
        ...shared,
        [shared[shared.length - 1][0] + 6, shared[shared.length - 1][1] + 30]
      ];
      const thirdRandom = fork('other');
      const other: Point[] = [
        [130, 92],
        [136, 70],
        [150, 70 + thirdRandom.range(-4, 4)]
      ];
      return [
        createGridLayer(),
        createStrokes([
          createPath(base, 'primary', 1.4, 0.8),
          createPath(other, 'primary', 1.4, 0.6)
        ]),
        createStrokes([
          createPath(second, 'violet', 1, 0.9),
          createPath(
            [
              [142, 94],
              [140, 70]
            ],
            'violet',
            1,
            0.6
          )
        ]),
        createStrokes([createPath(shared, 'hot', 2.8, 0.95)], 'flow', 0.28, 1.5),
        createPulseSprites(
          [createDot(shared[0], 'warm', 2.4), createDot(shared[shared.length - 1], 'warm', 2.4)],
          0.4
        )
      ];
    }
  },

  'linear-referencing': {
    parameters: [createSlider('points', 'Points', 3, 14, 1, 8)],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const street = smoothPath(
        [
          [8, 60],
          [34, 62],
          [56, 40],
          [82, 36],
          [102, 56],
          [124, 62],
          [152, 44]
        ],
        2
      );
      const random = fork('points');
      const points = Array.from(
        {length: Math.round(p.points)},
        (): Point => [random.range(10, 150), random.range(10, 90)]
      );
      const probe: Point = pointer ?? [88, 18];
      const items = [...points, probe];
      const links: PathItem[] = [];
      const snaps: SpriteItem[] = [];
      const queries: SpriteItem[] = [];
      for (const point of items) {
        const snap = getNearestOnPath(street, point);
        const isProbe = point === probe;
        const role: Role = isProbe ? 'hot' : snap.side > 0 ? 'warm' : 'cool';
        links.push(
          createPath([point, snap.point], role, isProbe ? 1.1 : 0.5, isProbe ? 0.95 : 0.6)
        );
        snaps.push(createDot(snap.point, 'ink', isProbe ? 1.8 : 1.1, 0.95));
        queries.push(createDot(point, role, isProbe ? 2.8 : 1.6));
      }
      const probeSnap = getNearestOnPath(street, probe);
      return [
        createGridLayer(),
        createStrokes([createPath(street, 'primary', 1.2, 0.85)]),
        createStrokes(
          [createPath(sliceRange(street, 0, probeSnap.measure), 'hot', 2.2, 0.9)],
          'flow',
          0.3,
          1.2
        ),
        createStrokes(links),
        createPulseSprites(snaps, 0.1),
        createPulseSprites(queries, 0.25)
      ];
    }
  },

  'line-locate': {
    parameters: [createSlider('measure', 'Measure', 5, 95, 5, 55, '%')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const routes = createStreets(fork('routes')).map(street => smoothPath(street, 1));
      const strokes: PathItem[] = [];
      const traveled: PathItem[] = [];
      const ticks: SpriteItem[] = [];
      const arrows: PathItem[] = [];
      const markers: SpriteItem[] = [];
      routes.forEach((route, index) => {
        const total = getPathLength(route);
        for (let at = 20; at < total; at += 20)
          ticks.push(createDot(getPointAtMeasure(route, at).point, 'dim', 0.7, 0.6));
        const fraction = clamp(p.measure / 100 + (index - 1) * 0.18, 0.04, 0.96);
        const {point, angle} = getPointAtMeasure(route, total * fraction);
        strokes.push(createPath(route, 'dim', 0.8, 0.6));
        traveled.push(createPath(sliceRange(route, 0, total * fraction), 'cool', 1.5, 0.9));
        const tip: Point = [point[0] + Math.cos(angle) * 9, point[1] + Math.sin(angle) * 9];
        arrows.push(
          createPath([point, tip], 'warm', 1.2, 0.95),
          createPath(
            [tip, [tip[0] - Math.cos(angle - 0.5) * 3.5, tip[1] - Math.sin(angle - 0.5) * 3.5]],
            'warm',
            1.2,
            0.95
          ),
          createPath(
            [tip, [tip[0] - Math.cos(angle + 0.5) * 3.5, tip[1] - Math.sin(angle + 0.5) * 3.5]],
            'warm',
            1.2,
            0.95
          )
        );
        markers.push(createDot(point, 'hot', 3, 1, index / 3));
      });
      return [
        createGridLayer(),
        createStrokes(strokes),
        createStrokes(traveled, 'flow', 0.25, 1.2),
        createPulseSprites(ticks, 0),
        createStrokes(arrows),
        createPulseSprites(markers, 0.35)
      ];
    }
  },

  // ===== Generators =====
  'grid-generator': {
    parameters: [
      createChoice('shape', 'Cells', ['Square', 'Hexagon', 'Triangle', 'Points'], 1),
      createSlider('size', 'Cell size', 7, 24, 1, 13, 'px')
    ],
    pointer: HOVER_POINTER,
    build: ({p, pointer}) => {
      const origin: Point = pointer ?? [10, 10];
      const size = p.size;
      const shape = Math.round(p.shape);
      const cells: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      const points: SpriteItem[] = [];
      const waveOf = (x: number, y: number) => getDistance([x, y], origin) / 110;
      const add = (ring: Ring, x: number, y: number) => {
        if (ring.some(([ringX, ringY]) => ringX > SCENE_WIDTH + 1 || ringY > SCENE_HEIGHT + 1))
          return;
        cells.push(createPolygon(ring, 'cool', 0.3, waveOf(x, y)));
        outlines.push(createLoop(ring, 'primary', 0.5, 0.7));
      };
      if (shape === 0) {
        for (let y = origin[1]; y < SCENE_HEIGHT; y += size) {
          for (let x = origin[0]; x < SCENE_WIDTH; x += size)
            add(createRectangle(x, y, size, size), x, y);
        }
      } else if (shape === 1) {
        const radius = size / Math.sqrt(3);
        for (let row = 0; origin[1] + row * radius * 1.5 < SCENE_HEIGHT; row++) {
          for (let column = 0; origin[0] + column * size < SCENE_WIDTH; column++) {
            const x = origin[0] + column * size + (row % 2 ? size / 2 : 0) + size / 2;
            const y = origin[1] + row * radius * 1.5 + radius;
            add(createRegularPolygon(x, y, radius, 6, Math.PI / 6), x, y);
          }
        }
      } else if (shape === 2) {
        const height = (size * Math.sqrt(3)) / 2;
        for (let row = 0; origin[1] + (row + 1) * height < SCENE_HEIGHT; row++) {
          const shift = row % 2 ? size / 2 : 0;
          const top = origin[1] + row * height;
          for (let column = 0; origin[0] + (column + 1) * size + shift < SCENE_WIDTH; column++) {
            const left = origin[0] + column * size + shift;
            add(
              [
                [left, top + height],
                [left + size, top + height],
                [left + size / 2, top]
              ],
              left,
              top
            );
            add(
              [
                [left + size / 2, top],
                [left + size * 1.5, top],
                [left + size, top + height]
              ],
              left,
              top
            );
          }
        }
      } else {
        for (let y = origin[1]; y < SCENE_HEIGHT; y += size) {
          for (let x = origin[0]; x < SCENE_WIDTH; x += size)
            points.push(createDot([x, y], 'cool', 1.4, 0.9, waveOf(x, y)));
        }
      }
      return [
        createFills(cells),
        createStrokes(outlines),
        createPulseSprites(points, 0.5),
        createPulseSprites([createDot(origin, 'warm', 3.2)], 0.3)
      ];
    }
  },

  'shape-generator': {
    parameters: [
      createSlider('radius', 'Radius', 8, 30, 1, 17, 'px'),
      createSlider('beam', 'Sector angle', 20, 160, 5, 70, '°')
    ],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const random = fork('towers');
      const towers: {center: Point; type: number; azimuth: number}[] = [
        [30, 30],
        [84, 24],
        [132, 38],
        [50, 74],
        [112, 76]
      ].map(([x, y], index) => ({
        center: [x + random.range(-5, 5), y + random.range(-4, 4)] as Point,
        type: index % 3,
        azimuth: random.range(0, Math.PI * 2)
      }));
      const half = ((p.beam / 2) * Math.PI) / 180;
      const rings = towers.map(tower => {
        const [centerX, centerY] = tower.center;
        if (tower.type === 0) return createCircle(centerX, centerY, p.radius, 36);
        if (tower.type === 2)
          return createEllipse(
            centerX,
            centerY,
            p.radius * 1.3,
            p.radius * 0.65,
            tower.azimuth,
            32
          );
        const azimuth = pointer
          ? Math.atan2(pointer[1] - centerY, pointer[0] - centerX)
          : tower.azimuth;
        return [
          tower.center,
          ...Array.from({length: 16}, (_, step): Point => {
            const angle = azimuth - half + (step / 15) * half * 2;
            return [
              centerX + Math.cos(angle) * p.radius * 1.2,
              centerY + Math.sin(angle) * p.radius * 1.2
            ];
          })
        ];
      });
      const roles: Role[] = ['primary', 'warm', 'violet'];
      return [
        createGridLayer(),
        createFills(
          rings.map((ring, index) =>
            createPolygon(ring, roles[towers[index].type], 0.26, index / 5)
          )
        ),
        createStrokes(
          rings.map((ring, index) => createLoop(ring, roles[towers[index].type], 1.1, 0.95)),
          'draw',
          0.15
        ),
        createPulseSprites(
          towers.map((tower, index) => createDot(tower.center, 'ink', 2, 0.95, index / 5)),
          0.3
        )
      ];
    }
  },

  'dorling-cartogram': {
    parameters: [createSlider('iterations', 'Relaxation', 0, 60, 2, 40)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('areas');
      const centers: Point[] = [
        [40, 36],
        [102, 40],
        [78, 72]
      ];
      const origins: Point[] = Array.from(
        {length: 18},
        (_, index): Point => [
          clamp(centers[index % 3][0] + random.gaussian() * 11, 14, 146),
          clamp(centers[index % 3][1] + random.gaussian() * 9, 14, 86)
        ]
      );
      const values = origins.map(() => random.range(0.25, 1));
      const radii = values.map(value => 3 + 7.5 * Math.sqrt(value));
      const positions = origins.map(([x, y]): Point => [x, y]);
      for (let iteration = 0; iteration < Math.round(p.iterations); iteration++) {
        for (let first = 0; first < positions.length; first++) {
          for (let second = first + 1; second < positions.length; second++) {
            const distance = getDistance(positions[first], positions[second]) || 0.01;
            const overlap = radii[first] + radii[second] + 0.6 - distance;
            if (overlap <= 0) continue;
            const dx = ((positions[second][0] - positions[first][0]) / distance) * overlap * 0.5;
            const dy = ((positions[second][1] - positions[first][1]) / distance) * overlap * 0.5;
            positions[first] = [positions[first][0] - dx, positions[first][1] - dy];
            positions[second] = [positions[second][0] + dx, positions[second][1] + dy];
          }
        }
        for (let index = 0; index < positions.length; index++) {
          positions[index] = [
            clamp(
              mixNumbers(positions[index][0], origins[index][0], 0.06),
              radii[index] + 2,
              SCENE_WIDTH - radii[index] - 2
            ),
            clamp(
              mixNumbers(positions[index][1], origins[index][1], 0.06),
              radii[index] + 2,
              SCENE_HEIGHT - radii[index] - 2
            )
          ];
        }
      }
      const roleOf = (value: number): Role =>
        value > 0.75 ? 'hot' : value > 0.5 ? 'warm' : value > 0.35 ? 'primary' : 'cool';
      return [
        createGridLayer(),
        createStrokes(
          origins.map((origin, index) => createPath([origin, positions[index]], 'dim', 0.35, 0.5))
        ),
        createPulseSprites(toSprites(origins, 'dim', 1, undefined, 0.7), 0),
        createFills(
          positions.map((position, index) =>
            createPolygon(
              createCircle(position[0], position[1], radii[index], 28),
              roleOf(values[index]),
              0.3,
              index / 18
            )
          )
        ),
        createStrokes(
          positions.map((position, index) =>
            createLoop(
              createCircle(position[0], position[1], radii[index], 28),
              roleOf(values[index]),
              0.9,
              0.95
            )
          ),
          'pulse',
          0.2
        )
      ];
    }
  },

  'map-coloring': {
    parameters: [createSlider('seed', 'Seed', 0, 12, 1, 0)],
    pointer: NO_POINTER,
    build: ({p}) => {
      const cells = createHexCoverage();
      const order = cells
        .map((cell, index) => ({
          index,
          key: (Math.sin((index + 1) * (p.seed * 12.9898 + 4.1414)) * 43758.5453) % 1
        }))
        .sort((first, second) => second.key - first.key)
        .map(entry => entry.index);
      const colors = new Array<number>(cells.length).fill(-1);
      for (const index of order) {
        const used = new Set(
          cells[index].neighbors.map(neighbor => (neighbor >= 0 ? colors[neighbor] : -1))
        );
        let color = 0;
        while (used.has(color)) color++;
        colors[index] = color;
      }
      const roles: Role[] = ['cool', 'warm', 'violet', 'primary', 'hot'];
      return [
        createFills(
          cells.map((cell, index) =>
            createPolygon(cell.ring, roles[colors[index] % 5], 0.4, (colors[index] % 5) / 5)
          )
        ),
        createStrokes(cells.map(cell => createLoop(cell.ring, 'ink', 0.6, 0.7)))
      ];
    }
  },

  // ===== Table plumbing =====
  'offset-expansion': {
    parameters: [createSlider('parents', 'Parent rows', 2, 7, 1, 5)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const random = fork('counts');
      const parents = Math.round(p.parents);
      const counts = Array.from({length: 7}, () => 2 + random.integer(4)).slice(0, parents);
      const total = counts.reduce((sum, count) => sum + count, 0);
      const cellWidth = Math.min(8, 142 / total);
      const gap = 4;
      const spanWidth = total * cellWidth + (parents - 1) * gap;
      const topStart = (SCENE_WIDTH - spanWidth) / 2;
      const bottomStart = (SCENE_WIDTH - total * cellWidth) / 2;
      const links: PolygonItem[] = [];
      const edges: PathItem[] = [];
      const parentBoxes: PolygonItem[] = [];
      const children: PolygonItem[] = [];
      const offsets: SpriteItem[] = [];
      let consumed = 0;
      counts.forEach((count, index) => {
        const role = getRoleCycle(index);
        const topX = topStart + consumed * cellWidth + index * gap;
        const bottomX = bottomStart + consumed * cellWidth;
        const width = count * cellWidth;
        parentBoxes.push(
          createPolygon(createRectangle(topX, 22, width, 9), role, 0.5, index / parents)
        );
        links.push(
          createPolygon(
            [
              [topX, 31],
              [topX + width, 31],
              [bottomX + width, 66],
              [bottomX, 66]
            ],
            role,
            0.1,
            index / parents
          )
        );
        edges.push(
          createPath(
            [
              [topX, 31],
              [bottomX, 66]
            ],
            role,
            0.5,
            0.7
          ),
          createPath(
            [
              [topX + width, 31],
              [bottomX + width, 66]
            ],
            role,
            0.5,
            0.7
          )
        );
        offsets.push(createDot([topX, 12], 'hot', 1.5, 0.9, index / parents));
        for (let child = 0; child < count; child++) {
          children.push(
            createPolygon(
              createRectangle(bottomX + child * cellWidth + 0.5, 66, cellWidth - 1, 12),
              role,
              0.6,
              (consumed + child) / total
            )
          );
        }
        consumed += count;
      });
      offsets.push(createDot([topStart + spanWidth, 12], 'hot', 1.5, 0.9));
      return [
        createStrokes([
          createPath(
            [
              [topStart, 12],
              [topStart + spanWidth, 12]
            ],
            'dim',
            0.4,
            0.6
          )
        ]),
        createPulseSprites(offsets, 0.3),
        createFills(links, 'static'),
        createStrokes(edges, 'flow', 0.25, 1),
        createFills(parentBoxes),
        createFills(children)
      ];
    }
  },

  'parameter-buffer': {
    parameters: [
      createSlider('distance', 'Distance', 2, 30, 1, 15, 'px'),
      createSlider('tolerance', 'Tolerance', 0, 8, 0.5, 2.5, 'px')
    ],
    pointer: NO_POINTER,
    build: ({p}) => {
      const slots = [p.distance / 30, p.tolerance / 8, 0.62];
      const slotRoles: Role[] = ['warm', 'hot', 'dim'];
      const nodes: Point[] = [
        [62, 22],
        [82, 50],
        [62, 78]
      ];
      const nodeEdges = [
        createPath([nodes[0], nodes[1]], 'primary', 0.7, 0.6),
        createPath([nodes[1], nodes[2]], 'primary', 0.7, 0.6)
      ];
      const feeds = slots
        .slice(0, 2)
        .map((_, index) =>
          createPath(
            [[34, 30 + index * 18], nodes[index === 0 ? 0 : 2]],
            slotRoles[index],
            0.8,
            0.85
          )
        );
      const segments = Math.max(5, Math.round(40 - p.tolerance * 4.5));
      const ring = createCircle(124, 50, 4 + p.distance * 0.95, segments);
      return [
        createFills(
          slots.flatMap((value, index) => [
            createPolygon(
              createRectangle(14 + index * 8, 86 - 66 * value, 6, 66 * value),
              slotRoles[index],
              0.55,
              index / 3
            )
          ])
        ),
        createStrokes(
          slots.map((_, index) =>
            createLoop(createRectangle(14 + index * 8, 20, 6, 66), 'ink', 0.4, 0.5)
          )
        ),
        createStrokes(feeds, 'flow', 0.35, 1.2),
        createStrokes(nodeEdges, 'static'),
        createPulseSprites(
          nodes.map(node => createDot(node, 'primary', 3.6, 0.9)),
          0.1
        ),
        createStrokes([createPath([nodes[1], [102, 50]], 'primary', 0.8, 0.7)], 'flow', 0.3, 1),
        createFills([createPolygon(ring, 'warm', 0.18)]),
        createStrokes([createLoop(ring, 'warm', 1.2, 0.95)], 'static'),
        createPulseSprites([createDot([124, 50], 'hot', 2.2)], 0.3)
      ];
    }
  },

  // ===== Recipes =====
  'building-adjacency-recipe': {
    parameters: [createSlider('touch', 'Touching chance', 0, 1, 0.05, 0.55)],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const columns = 4;
      const rows = 3;
      const cellWidth = 34;
      const cellHeight = 26;
      const startX = (SCENE_WIDTH - columns * cellWidth) / 2;
      const startY = (SCENE_HEIGHT - rows * cellHeight) / 2;
      // Each cell border is shared by two buildings; the border decides whether both reach it.
      const horizontalTouch = (column: number, row: number) =>
        fork(`h-${column}-${row}`).next() < p.touch;
      const verticalTouch = (column: number, row: number) =>
        fork(`v-${column}-${row}`).next() < p.touch;
      const gap = 3.2;
      const buildings = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const left = column > 0 && horizontalTouch(column - 1, row) ? 0 : gap;
          const right = column < columns - 1 && horizontalTouch(column, row) ? 0 : gap;
          const top = row > 0 && verticalTouch(column, row - 1) ? 0 : gap;
          const bottom = row < rows - 1 && verticalTouch(column, row) ? 0 : gap;
          const x = startX + column * cellWidth + left;
          const y = startY + row * cellHeight + top;
          buildings.push(
            createRectangle(x, y, cellWidth - left - right, cellHeight - top - bottom)
          );
        }
      }
      const walls: PathItem[] = [];
      const links: PathItem[] = [];
      const centers = buildings.map(
        (ring): Point => [(ring[0][0] + ring[2][0]) / 2, (ring[0][1] + ring[2][1]) / 2]
      );
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const index = row * columns + column;
          if (column < columns - 1 && horizontalTouch(column, row)) {
            const wallX = startX + (column + 1) * cellWidth;
            const top = Math.max(buildings[index][0][1], buildings[index + 1][0][1]);
            const bottom = Math.min(buildings[index][2][1], buildings[index + 1][2][1]);
            walls.push(
              createPath(
                [
                  [wallX, top],
                  [wallX, bottom]
                ],
                'hot',
                1.8,
                0.95
              )
            );
            links.push(createPath([centers[index], centers[index + 1]], 'warm', 0.6, 0.7));
          }
          if (row < rows - 1 && verticalTouch(column, row)) {
            const wallY = startY + (row + 1) * cellHeight;
            const left = Math.max(buildings[index][0][0], buildings[index + columns][0][0]);
            const right = Math.min(buildings[index][2][0], buildings[index + columns][2][0]);
            walls.push(
              createPath(
                [
                  [left, wallY],
                  [right, wallY]
                ],
                'hot',
                1.8,
                0.95
              )
            );
            links.push(createPath([centers[index], centers[index + columns]], 'warm', 0.6, 0.7));
          }
        }
      }
      return [
        createFills(
          buildings.map((ring, index) => createPolygon(ring, 'cool', 0.2, index / buildings.length))
        ),
        createStrokes(buildings.map(ring => createLoop(ring, 'primary', 0.6, 0.75))),
        createStrokes(links, 'static'),
        createStrokes(walls, 'flow', 0.3, 1.4),
        createPulseSprites(toSprites(centers, 'ink', 1.1), 0.15)
      ];
    }
  },

  'building-tessellation-recipe': {
    parameters: [createSlider('samples', 'Samples per wall', 1, 4, 1, 2)],
    pointer: NO_POINTER,
    build: ({p}) => {
      const buildings: Ring[] = [
        createRectangle(18, 16, 26, 18),
        createRectangle(60, 12, 20, 24),
        createRectangle(100, 18, 30, 16),
        createRectangle(24, 56, 18, 26),
        createRectangle(62, 58, 30, 14),
        createRectangle(108, 54, 22, 26),
        createRectangle(78, 40, 12, 10)
      ];
      const sites: Point[] = [];
      const owners: number[] = [];
      const samples = Math.round(p.samples);
      buildings.forEach((ring, owner) => {
        for (let edge = 0; edge < 4; edge++) {
          for (let sample = 0; sample < samples; sample++) {
            sites.push(mixPoints(ring[edge], ring[(edge + 1) % 4], (sample + 0.5) / samples));
            owners.push(owner);
          }
        }
      });
      const bounds = createRectangle(6, 5, 148, 90);
      const fragments: PolygonItem[] = sites.map((site, index) => {
        let ring = bounds;
        for (let other = 0; other < sites.length && ring.length >= 3; other++) {
          if (other === index) continue;
          ring = clipRingByHalfPlane(ring, mixPoints(site, sites[other], 0.5), [
            site[0] - sites[other][0],
            site[1] - sites[other][1]
          ]);
        }
        return createPolygon(
          ring,
          getRoleCycle(owners[index]),
          0.17,
          owners[index] / buildings.length
        );
      });
      return [
        createFills(fragments),
        createStrokes([createLoop(bounds, 'dim', 0.7, 0.7)]),
        createFills(
          buildings.map((ring, index) =>
            createPolygon(ring, getRoleCycle(index), 0.55, index / buildings.length)
          )
        ),
        createStrokes(
          buildings.map((ring, index) => createLoop(ring, getRoleCycle(index), 1, 0.95)),
          'flow',
          0.2,
          1.1
        ),
        createPulseSprites(toSprites(sites, 'ink', 0.7, undefined, 0.8), 0)
      ];
    }
  },

  'isovists-recipe': {
    parameters: [
      createSlider('rays', 'Rays', 24, 240, 12, 120),
      createSlider('range', 'Maximum distance', 30, 100, 5, 62, 'px')
    ],
    pointer: HOVER_POINTER,
    build: ({p, pointer}) => {
      const obstacles: Ring[] = [
        createRectangle(30, 18, 24, 16),
        createRectangle(104, 14, 24, 22),
        createRectangle(64, 62, 30, 14),
        createRectangle(18, 62, 18, 20),
        createRectangle(120, 62, 20, 22)
      ];
      const viewpoint: Point = pointer ?? [80, 40];
      const count = Math.round(p.rays);
      const hits: Point[] = [];
      for (let ray = 0; ray < count; ray++) {
        const angle = (ray / count) * Math.PI * 2;
        const far: Point = [
          viewpoint[0] + Math.cos(angle) * p.range,
          viewpoint[1] + Math.sin(angle) * p.range
        ];
        let nearest = 1;
        for (const obstacle of obstacles) {
          for (let edge = 0; edge < 4; edge++) {
            const hit = getSegmentCrossing(
              viewpoint,
              far,
              obstacle[edge],
              obstacle[(edge + 1) % 4]
            );
            if (hit !== null && hit < nearest) nearest = hit;
          }
        }
        hits.push(mixPoints(viewpoint, far, nearest));
      }
      const rays: PathItem[] = hits
        .filter((_, index) => index % 4 === 0)
        .map((hit, index) => createPath([viewpoint, hit], 'warm', 0.35, 0.6, index / 30));
      return [
        createGridLayer(),
        createFills([createPolygon(hits, 'warm', 0.3)]),
        createStrokes(rays, 'flow', 0.3, 0.6),
        createStrokes([createLoop(hits, 'hot', 1, 0.9)], 'static'),
        createFills(
          obstacles.map(ring => createPolygon(ring, 'dim', 0.5)),
          'static'
        ),
        createStrokes(obstacles.map(ring => createLoop(ring, 'ink', 0.8, 0.85))),
        createPulseSprites([createDot(viewpoint, 'ink', 2.8)], 0.3)
      ];
    }
  },

  'street-profiles-recipe': {
    parameters: [createSlider('spacing', 'Sample spacing', 8, 24, 1, 12, 'px')],
    pointer: NO_POINTER,
    build: ({fork, p}) => {
      const tilt = -0.14;
      const rotate = ([x, y]: Point): Point => {
        const dx = x - 80;
        const dy = y - 50;
        return [
          80 + dx * Math.cos(tilt) - dy * Math.sin(tilt),
          50 + dx * Math.sin(tilt) + dy * Math.cos(tilt)
        ];
      };
      const sides = [-1, 1].map(side => {
        const random = fork(`side-${side}`);
        const blocks: {from: number; to: number; height: number}[] = [];
        for (let x = 4 + random.range(0, 6); x < 150; ) {
          const width = random.range(14, 26);
          blocks.push({
            from: x,
            to: Math.min(x + width, 156),
            height: random.range(0.1, 1)
          });
          x += width + (random.next() < 0.35 ? random.range(8, 16) : 0.5);
        }
        return {side, blocks};
      });
      const heightRole = (height: number): Role =>
        height > 0.7 ? 'hot' : height > 0.4 ? 'warm' : 'cool';
      const setback = 12;
      const range = 26;
      const fills: PolygonItem[] = [];
      const outlines: PathItem[] = [];
      const rays: PathItem[] = [];
      const hitsSprites: SpriteItem[] = [];
      const censored: SpriteItem[] = [];
      for (const {side, blocks} of sides) {
        for (const block of blocks) {
          const y = 50 + side * setback;
          const ring = createRectangle(
            block.from,
            side < 0 ? y - 14 : y,
            block.to - block.from,
            14
          ).map(rotate);
          fills.push(createPolygon(ring, heightRole(block.height), 0.15 + block.height * 0.4));
          outlines.push(createLoop(ring, heightRole(block.height), 0.5, 0.7));
        }
        for (let x = 10; x < 152; x += p.spacing) {
          const block = blocks.find(candidate => x >= candidate.from && x <= candidate.to);
          const start = rotate([x, 50]);
          if (block) {
            const end = rotate([x, 50 + side * setback]);
            rays.push(createPath([start, end], heightRole(block.height), 0.9, 0.95));
            hitsSprites.push(createDot(end, heightRole(block.height), 1.3));
          } else {
            const end = rotate([x, 50 + side * range]);
            rays.push(createPath([start, end], 'dim', 0.5, 0.45));
            censored.push(createDot(end, 'violet', 1, 0.5));
          }
        }
      }
      const street = [rotate([4, 50]), rotate([156, 50])];
      return [
        createFills(fills),
        createStrokes(outlines),
        createStrokes([createPath(street, 'primary', 2.2, 0.9)], 'flow', 0.2, 1.5),
        createStrokes(rays, 'pulse', 0.35),
        createPulseSprites(hitsSprites, 0.2),
        createPulseSprites(censored, 0)
      ];
    }
  },

  'visibility-graph-recipe': {
    parameters: [createSlider('range', 'Maximum distance', 20, 120, 5, 95, 'px')],
    pointer: HOVER_POINTER,
    build: ({fork, p, pointer}) => {
      const obstacles: Ring[] = [
        createRectangle(34, 26, 26, 20),
        createRectangle(94, 16, 20, 30),
        createRectangle(66, 64, 34, 16)
      ];
      const random = fork('points');
      const points: Point[] = [];
      for (let attempt = 0; attempt < 200 && points.length < 16; attempt++) {
        const candidate: Point = [random.range(8, 152), random.range(8, 92)];
        if (
          !obstacles.some(ring =>
            isInsideRing(
              candidate,
              createRectangle(
                ring[0][0] - 3,
                ring[0][1] - 3,
                ring[2][0] - ring[0][0] + 6,
                ring[2][1] - ring[0][1] + 6
              )
            )
          )
        ) {
          points.push(candidate);
        }
      }
      const probe: Point | null = pointer;
      const all = probe ? [...points, probe] : points;
      const isVisible = (first: Point, second: Point) => {
        for (const obstacle of obstacles) {
          if (isInsideRing(mixPoints(first, second, 0.5), obstacle)) return false;
          for (let edge = 0; edge < 4; edge++) {
            const hit = getSegmentCrossing(first, second, obstacle[edge], obstacle[(edge + 1) % 4]);
            if (hit !== null && hit > 1e-6 && hit < 1 - 1e-6) return false;
          }
        }
        return true;
      };
      const edges: PathItem[] = [];
      const probeEdges: PathItem[] = [];
      const degrees = new Array<number>(all.length).fill(0);
      for (let first = 0; first < all.length; first++) {
        for (let second = first + 1; second < all.length; second++) {
          if (getDistance(all[first], all[second]) > p.range || !isVisible(all[first], all[second]))
            continue;
          degrees[first]++;
          degrees[second]++;
          const item = createPath(
            [all[first], all[second]],
            probe && second === all.length - 1 ? 'warm' : 'primary',
            0.45,
            0.45
          );
          (probe && second === all.length - 1 ? probeEdges : edges).push(item);
        }
      }
      return [
        createStrokes(edges, 'flow', 0.12, 0.7),
        createStrokes(
          probeEdges.map(edge => ({...edge, width: 0.8, alpha: 0.9})),
          'flow',
          0.3,
          1
        ),
        createFills(
          obstacles.map(ring => createPolygon(ring, 'dim', 0.5)),
          'static'
        ),
        createStrokes(obstacles.map(ring => createLoop(ring, 'ink', 0.8, 0.85))),
        createPulseSprites(
          points.map((point, index) =>
            createDot(
              point,
              degrees[index] > 6 ? 'hot' : 'ink',
              1.2 + degrees[index] * 0.22,
              0.95,
              index / 16
            )
          ),
          0.25
        ),
        createPulseSprites(probe ? [createDot(probe, 'warm', 3)] : [], 0.3)
      ];
    }
  }
};
