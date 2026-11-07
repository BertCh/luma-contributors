// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/geometry-processing.ts, keyed by entry id.

import {
  createArc,
  createBlob,
  createCircle,
  createRegularPolygon,
  createWanderPath,
  getDistance,
  clamp,
  mixNumbers
} from '../data.ts';
import {
  clampPoint,
  getSegmentDistance,
  smoothPath,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
import {
  clipRingByHalfPlane,
  closeRing,
  createRectangleRing,
  getConvexHull,
  getNearestOnSegment,
  getRingBounds,
  getSignedArea,
  intersectSegments,
  isInsideRing
} from '../geometry.ts';
import {
  createChoice,
  createDot,
  createFillLayer,
  createPath,
  createSlider,
  createSpriteLayer,
  createStrokeLayer
} from '../layers.ts';
import {projectGlobe, slerpVectors, toLonLat, toVector} from '../globe.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type FillLayer,
  type FieldLayer,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewLayer,
  type Role,
  type SpriteItem,
  type SpriteLayer,
  type StrokeLayer
} from '../types.ts';

// ---------------------------------------------------------------------------------------------
// Layer shorthands with this file's default styling

function createStrokes(
  items: PathItem[],
  animation: StrokeLayer['animation'] = 'static',
  speed = 0.3,
  trail?: number
): StrokeLayer {
  return createStrokeLayer(items, {animation, speed, trail, blend: 'add'});
}

function createFills(
  items: PolygonItem[],
  animation: FillLayer['animation'] = 'static'
): FillLayer {
  return createFillLayer(items, {animation, blend: 'add'});
}

function createSprites(items: SpriteItem[], pulse = 0.3): SpriteLayer {
  return createSpriteLayer(items, {pulse});
}

function createLine(points: Point[], role: Role, width = 0.9, alpha = 0.9, phase = 0): PathItem {
  return createPath(points, role, width, alpha, phase);
}

function createLoop(ring: Point[], role: Role, width = 0.9, alpha = 0.9, phase = 0): PathItem {
  return createPath(closeRing(ring), role, width, alpha, phase);
}

function createArea(ring: Point[], role: Role, alpha = 0.2, phase = 0): PolygonItem {
  return {ring, role, alpha, phase};
}

function createGlowDot(
  point: Point,
  role: Role,
  radius = 1.6,
  alpha = 0.95,
  phase = 0
): SpriteItem {
  return createDot(point, role, radius, alpha, phase);
}

// ---------------------------------------------------------------------------------------------
// Geometry helpers

function getRingPerimeter(ring: Point[]): number {
  let sum = 0;
  for (let index = 0; index < ring.length; index++) {
    sum += getDistance(ring[index], ring[(index + 1) % ring.length]);
  }
  return sum;
}

function getRingCentroid(ring: Point[]): Point {
  let x = 0;
  let y = 0;
  let twiceArea = 0;
  for (let index = 0; index < ring.length; index++) {
    const [ax, ay] = ring[index];
    const [bx, by] = ring[(index + 1) % ring.length];
    const cross = ax * by - bx * ay;
    twiceArea += cross;
    x += (ax + bx) * cross;
    y += (ay + by) * cross;
  }
  return twiceArea === 0 ? ring[0] : [x / (3 * twiceArea), y / (3 * twiceArea)];
}

/** Even-odd containment across several rings, so holes work. */
function isPointInRings(point: Point, rings: Point[][]): boolean {
  let inside = false;
  for (const ring of rings) if (isInsideRing(point, ring)) inside = !inside;
  return inside;
}

function getRingDistance(point: Point, ring: Point[]): number {
  let best = Infinity;
  for (let index = 0; index < ring.length; index++) {
    best = Math.min(best, getSegmentDistance(point, ring[index], ring[(index + 1) % ring.length]));
  }
  return best;
}

function rotatePoint(point: Point, center: Point, angle: number): Point {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  return [center[0] + dx * cosine - dy * sine, center[1] + dx * sine + dy * cosine];
}

/** Like intersectSegments, but also returns the position along each segment (0..1). */
function intersectSegmentsWithParameters(
  firstStart: Point,
  firstEnd: Point,
  secondStart: Point,
  secondEnd: Point
): {firstAmount: number; secondAmount: number; point: Point} | null {
  const firstDx = firstEnd[0] - firstStart[0];
  const firstDy = firstEnd[1] - firstStart[1];
  const secondDx = secondEnd[0] - secondStart[0];
  const secondDy = secondEnd[1] - secondStart[1];
  const denominator = firstDx * secondDy - firstDy * secondDx;
  if (Math.abs(denominator) < 1e-9) return null;
  const offsetX = secondStart[0] - firstStart[0];
  const offsetY = secondStart[1] - firstStart[1];
  const firstAmount = (offsetX * secondDy - offsetY * secondDx) / denominator;
  const secondAmount = (offsetX * firstDy - offsetY * firstDx) / denominator;
  if (firstAmount < 0 || firstAmount > 1 || secondAmount < 0 || secondAmount > 1) return null;
  return {
    firstAmount,
    secondAmount,
    point: [firstStart[0] + firstDx * firstAmount, firstStart[1] + firstDy * firstAmount]
  };
}

/** Keeps the part of a ring where `axisX * x + axisY * y <= limit`, for a unit axis. */
function clipRingBelowLimit(ring: Point[], axisX: number, axisY: number, limit: number): Point[] {
  return clipRingByHalfPlane(ring, [axisX * limit, axisY * limit], [-axisX, -axisY]);
}

type Box = {minimumX: number; minimumY: number; maximumX: number; maximumY: number};

function clipRingToBox(ring: Point[], box: Box): Point[] {
  let current = clipRingBelowLimit(ring, 1, 0, box.maximumX);
  if (current.length) current = clipRingBelowLimit(current, -1, 0, -box.minimumX);
  if (current.length) current = clipRingBelowLimit(current, 0, 1, box.maximumY);
  if (current.length) current = clipRingBelowLimit(current, 0, -1, -box.minimumY);
  return current;
}

/** Liang-Barsky clip of one segment; returns the inside part or null. */
function clipSegmentToBox(start: Point, end: Point, box: Box): [Point, Point] | null {
  let low = 0;
  let high = 1;
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const tests: [number, number][] = [
    [-dx, start[0] - box.minimumX],
    [dx, box.maximumX - start[0]],
    [-dy, start[1] - box.minimumY],
    [dy, box.maximumY - start[1]]
  ];
  for (const [direction, distance] of tests) {
    if (direction === 0) {
      if (distance < 0) return null;
    } else {
      const amount = distance / direction;
      if (direction < 0) low = Math.max(low, amount);
      else high = Math.min(high, amount);
    }
  }
  if (low > high) return null;
  return [
    [start[0] + dx * low, start[1] + dy * low],
    [start[0] + dx * high, start[1] + dy * high]
  ];
}

function getRectangleAtAngle(points: Point[], angle: number) {
  const origin: Point = [0, 0];
  const rotated = points.map(point => rotatePoint(point, origin, -angle));
  const [minimumX, minimumY, maximumX, maximumY] = getRingBounds(rotated);
  const corners = createRectangleRing(minimumX, minimumY, maximumX, maximumY).map(corner =>
    rotatePoint(corner, origin, angle)
  );
  return {corners, area: (maximumX - minimumX) * (maximumY - minimumY)};
}

function getMinimumRectangle(points: Point[]) {
  let best = getRectangleAtAngle(points, 0);
  let bestAngle = 0;
  for (let degrees = 2; degrees < 180; degrees += 2) {
    const candidate = getRectangleAtAngle(points, (degrees * Math.PI) / 180);
    if (candidate.area < best.area) {
      best = candidate;
      bestAngle = (degrees * Math.PI) / 180;
    }
  }
  return {...best, angle: bestAngle};
}

function polylineAt(points: Point[], position: number): Point {
  const index = clamp(Math.floor(position), 0, points.length - 2);
  const amount = position - index;
  return [
    points[index][0] + (points[index + 1][0] - points[index][0]) * amount,
    points[index][1] + (points[index + 1][1] - points[index][1]) * amount
  ];
}

/** Splits every polyline at its crossings with the others; `allow` filters which pairs interact. */
function splitPolylines(
  lines: Point[][],
  allow: (first: number, second: number) => boolean = () => true
) {
  const cuts: number[][] = lines.map(() => []);
  const nodes: Point[] = [];
  for (let first = 0; first < lines.length; first++) {
    for (let second = first; second < lines.length; second++) {
      if (!allow(first, second)) continue;
      const firstLine = lines[first];
      const secondLine = lines[second];
      for (let firstIndex = 0; firstIndex + 1 < firstLine.length; firstIndex++) {
        for (
          let secondIndex = first === second ? firstIndex + 2 : 0;
          secondIndex + 1 < secondLine.length;
          secondIndex++
        ) {
          const hit = intersectSegmentsWithParameters(
            firstLine[firstIndex],
            firstLine[firstIndex + 1],
            secondLine[secondIndex],
            secondLine[secondIndex + 1]
          );
          if (!hit) continue;
          cuts[first].push(firstIndex + hit.firstAmount);
          cuts[second].push(secondIndex + hit.secondAmount);
          nodes.push(hit.point);
        }
      }
    }
  }
  const pieces: Point[][] = [];
  lines.forEach((points, lineIndex) => {
    const stops = [
      0,
      ...cuts[lineIndex].sort((first, second) => first - second),
      points.length - 1
    ];
    for (let stop = 0; stop + 1 < stops.length; stop++) {
      const piece: Point[] = [polylineAt(points, stops[stop])];
      for (let vertex = Math.floor(stops[stop]) + 1; vertex < stops[stop + 1]; vertex++) {
        piece.push(points[vertex]);
      }
      piece.push(polylineAt(points, stops[stop + 1]));
      pieces.push(piece);
    }
  });
  return {pieces, nodes};
}

/** Pulls both ends of a path inward so neighbouring pieces show a visible gap. */
function trimPath(points: Point[], gap: number): Point[] {
  if (points.length < 2) return points;
  const result = points.map(point => [...point] as Point);
  const pull = (from: Point, to: Point): Point => {
    const length = getDistance(from, to);
    if (length < gap * 3) return from;
    return [
      from[0] + ((to[0] - from[0]) / length) * gap,
      from[1] + ((to[1] - from[1]) / length) * gap
    ];
  };
  result[0] = pull(points[0], points[1]);
  result[result.length - 1] = pull(points[points.length - 1], points[points.length - 2]);
  return result;
}

function getPolylineTicks(path: Point[], spacing: number, half: number): Point[][] {
  const ticks: Point[][] = [];
  let next = spacing;
  for (let index = 0; index + 1 < path.length; index++) {
    const [ax, ay] = path[index];
    const [bx, by] = path[index + 1];
    const length = Math.hypot(bx - ax, by - ay) || 1;
    const dx = (bx - ax) / length;
    const dy = (by - ay) / length;
    while (next <= length) {
      const x = ax + dx * next;
      const y = ay + dy * next;
      ticks.push([
        [x - dy * half, y + dx * half],
        [x + dy * half, y - dx * half]
      ]);
      next += spacing;
    }
    next -= length;
  }
  return ticks;
}

function createStar(
  centerX: number,
  centerY: number,
  outer: number,
  inner: number,
  count: number,
  rotation = 0
) {
  return Array.from({length: count * 2}, (_, index): Point => {
    const radius = index % 2 === 0 ? outer : inner;
    const angle = rotation + (index / (count * 2)) * Math.PI * 2;
    return [centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius];
  });
}

function createRectangle(
  centerX: number,
  centerY: number,
  length: number,
  width: number,
  angle: number
) {
  const corners: Point[] = [
    [centerX - length / 2, centerY - width / 2],
    [centerX + length / 2, centerY - width / 2],
    [centerX + length / 2, centerY + width / 2],
    [centerX - length / 2, centerY + width / 2]
  ];
  return corners.map(corner => rotatePoint(corner, [centerX, centerY], angle));
}

function orientRing(ring: Point[], wantPositive: boolean): Point[] {
  return getSignedArea(ring) > 0 === wantPositive ? ring : [...ring].reverse();
}

/** Depth of every cell inside the rings, normalized by the deepest one. */
function createDepthField(rings: Point[][], columns: number, rows: number) {
  const values = new Float32Array(columns * rows);
  let best: Point = [SCENE_WIDTH / 2, SCENE_HEIGHT / 2];
  let bestDepth = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const point: Point = [
        ((column + 0.5) / columns) * SCENE_WIDTH,
        ((row + 0.5) / rows) * SCENE_HEIGHT
      ];
      if (!isPointInRings(point, rings)) continue;
      let depth = Infinity;
      for (const ring of rings) depth = Math.min(depth, getRingDistance(point, ring));
      values[row * columns + column] = depth;
      if (depth > bestDepth) {
        bestDepth = depth;
        best = point;
      }
    }
  }
  // Refine the winner on a finer local grid.
  let step = SCENE_WIDTH / columns / 2;
  for (let pass = 0; pass < 3; pass++) {
    const center = best;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const point: Point = [center[0] + dx * step, center[1] + dy * step];
        if (!isPointInRings(point, rings)) continue;
        let depth = Infinity;
        for (const ring of rings) depth = Math.min(depth, getRingDistance(point, ring));
        if (depth > bestDepth) {
          bestDepth = depth;
          best = point;
        }
      }
    }
    step /= 2;
  }
  for (let index = 0; index < values.length; index++) values[index] /= bestDepth || 1;
  return {values, best, bestDepth};
}

type SkeletonSample = {depth: number; id: number};

/** Depth field plus ridge cells, where the nearest boundary element changes. */
function createSkeletonField(
  columns: number,
  rows: number,
  evaluate: (x: number, y: number) => SkeletonSample,
  outsideValue: number
) {
  const depths = new Float32Array(columns * rows);
  const ids = new Int16Array(columns * rows).fill(-1);
  let maxDepth = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const sample = evaluate(
        ((column + 0.5) / columns) * SCENE_WIDTH,
        ((row + 0.5) / rows) * SCENE_HEIGHT
      );
      depths[row * columns + column] = sample.depth;
      ids[row * columns + column] = sample.id;
      maxDepth = Math.max(maxDepth, sample.depth);
    }
  }
  const values = new Float32Array(columns * rows);
  const ridge: Point[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      values[index] = depths[index] <= 0 ? outsideValue : depths[index] / (maxDepth || 1);
      if (ids[index] < 0) continue;
      const right = column + 1 < columns ? ids[index + 1] : -1;
      const down = row + 1 < rows ? ids[index + columns] : -1;
      if ((right >= 0 && right !== ids[index]) || (down >= 0 && down !== ids[index])) {
        ridge.push([((column + 1) / columns) * SCENE_WIDTH, ((row + 1) / rows) * SCENE_HEIGHT]);
      }
    }
  }
  return {values, ridge, maxDepth};
}

function createCrescent(
  centerX: number,
  centerY: number,
  radius: number,
  thickness: number,
  span: number
) {
  const steps = 18;
  const outer: Point[] = [];
  const inner: Point[] = [];
  for (let index = 0; index <= steps; index++) {
    const amount = index / steps;
    const angle = -span + amount * span * 2;
    const taper = thickness * (0.35 + 0.65 * Math.sin(Math.PI * amount));
    outer.push([centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius]);
    inner.push([
      centerX + Math.cos(angle) * (radius - taper),
      centerY + Math.sin(angle) * (radius - taper)
    ]);
  }
  return [...outer, ...inner.reverse()];
}

// Globe helpers (orthographic) -------------------------------------------------------------

const GLOBE_CENTER: Point = [80, 50];
const GLOBE_RADIUS = 42;

function projectGlobePoint(
  longitude: number,
  latitude: number,
  centerLongitude: number,
  centerLatitude: number
) {
  const {point, depth} = projectGlobe(
    {
      centerX: GLOBE_CENTER[0],
      centerY: GLOBE_CENTER[1],
      radius: GLOBE_RADIUS,
      longitude: centerLongitude,
      latitude: centerLatitude
    },
    longitude,
    latitude
  );
  return {point, visible: depth > 0};
}

/** Projects a lon/lat path, splitting it where it turns behind the globe. */
function projectGlobePath(
  path: [number, number][],
  centerLongitude: number,
  centerLatitude: number
): Point[][] {
  const pieces: Point[][] = [];
  let current: Point[] = [];
  for (const [longitude, latitude] of path) {
    const projected = projectGlobePoint(longitude, latitude, centerLongitude, centerLatitude);
    if (projected.visible) current.push(projected.point);
    else if (current.length) {
      pieces.push(current);
      current = [];
    }
  }
  if (current.length) pieces.push(current);
  return pieces.filter(piece => piece.length > 1);
}

function createGlobeBase(centerLongitude: number, centerLatitude: number): PreviewLayer[] {
  const graticule: PathItem[] = [];
  for (let longitude = -180; longitude < 180; longitude += 30) {
    const meridian: [number, number][] = [];
    for (let latitude = -90; latitude <= 90; latitude += 6) meridian.push([longitude, latitude]);
    for (const piece of projectGlobePath(meridian, centerLongitude, centerLatitude)) {
      graticule.push(createLine(piece, 'dim', 0.25, 0.28));
    }
  }
  for (let latitude = -60; latitude <= 60; latitude += 30) {
    const parallel: [number, number][] = [];
    for (let longitude = -180; longitude <= 180; longitude += 6)
      parallel.push([longitude, latitude]);
    for (const piece of projectGlobePath(parallel, centerLongitude, centerLatitude)) {
      graticule.push(createLine(piece, 'dim', 0.25, 0.28));
    }
  }
  return [
    createFills([
      createArea(createCircle(GLOBE_CENTER[0], GLOBE_CENTER[1], GLOBE_RADIUS, 48), 'cool', 0.05)
    ]),
    createStrokes(graticule),
    createStrokes([
      createLine(
        closeRing(createCircle(GLOBE_CENTER[0], GLOBE_CENTER[1], GLOBE_RADIUS, 64)),
        'cool',
        0.6,
        0.55
      )
    ])
  ];
}

function getGreatCircle(
  from: [number, number],
  to: [number, number],
  steps = 28
): [number, number][] {
  const start = toVector(from[0], from[1]);
  const end = toVector(to[0], to[1]);
  return Array.from({length: steps + 1}, (_, index) =>
    toLonLat(slerpVectors(start, end, index / steps))
  );
}

function getDestination(
  origin: [number, number],
  bearingDegrees: number,
  distanceDegrees: number
): [number, number] {
  const lat1 = (origin[1] * Math.PI) / 180;
  const lon1 = (origin[0] * Math.PI) / 180;
  const bearing = (bearingDegrees * Math.PI) / 180;
  const delta = (distanceDegrees * Math.PI) / 180;
  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(delta) + Math.cos(lat1) * Math.sin(delta) * Math.cos(bearing)
  );
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(bearing) * Math.sin(delta) * Math.cos(lat1),
      Math.cos(delta) - Math.sin(lat1) * Math.sin(lat2)
    );
  return [(lon2 * 180) / Math.PI, (lat2 * 180) / Math.PI];
}

// Geometry tree helpers ---------------------------------------------------------------------

type TreeKind = 'point' | 'line' | 'polygon' | 'collection' | 'empty';
type TreeNode = {kind: TreeKind; children?: TreeNode[]};

const TREE_ROLES: Record<TreeKind, Role> = {
  point: 'cool',
  line: 'warm',
  polygon: 'primary',
  collection: 'violet',
  empty: 'dim'
};

function layoutTree(root: TreeNode, originX: number, stepX: number, top: number, bottom: number) {
  const leafCount = (node: TreeNode): number =>
    node.children && node.children.length
      ? node.children.reduce((sum, child) => sum + leafCount(child), 0)
      : 1;
  const total = leafCount(root);
  const spacing = (bottom - top) / Math.max(total - 1, 1);
  let cursor = 0;
  const nodes: {point: Point; kind: TreeKind; depth: number; leaf: boolean}[] = [];
  const links: Point[][] = [];
  const place = (node: TreeNode, depth: number): Point => {
    if (!node.children || node.children.length === 0) {
      const point: Point = [originX + stepX * depth, top + cursor * spacing];
      cursor++;
      nodes.push({point, kind: node.kind, depth, leaf: true});
      return point;
    }
    const childPoints = node.children.map(child => place(child, depth + 1));
    const meanY = childPoints.reduce((sum, child) => sum + child[1], 0) / childPoints.length;
    const point: Point = [originX + stepX * depth, meanY];
    for (const child of childPoints) links.push([point, [child[0], child[1]]]);
    nodes.push({point, kind: node.kind, depth, leaf: false});
    return point;
  };
  place(root, 0);
  return {nodes, links};
}

// ---------------------------------------------------------------------------------------------

export const GEOMETRY_PROCESSING_PREVIEWS: Record<string, EntryPreview> = {
  // Measures and descriptors ---------------------------------------------------------------

  'geometry-measures': {
    parameters: [createSlider('vertices', 'Vertices', 6, 24, 1, 12)],
    pointer: 'none',
    build: ({fork, p}) => {
      const ring = createBlob(fork('polygon'), 44, 50, 27, Math.round(p.vertices), 0.3, 0.9);
      const centroid = getRingCentroid(ring);
      const path = smoothPath(
        createWanderPath(fork('line'), [92, 78], -0.55, 9, 7, 0.3).map(point =>
          clampPoint(point, 8)
        ),
        2
      );
      const ticks = getPolylineTicks(path, 7, 2.2);
      return [
        createFills([createArea(ring, 'primary', 0.17)], 'pulse'),
        createStrokes([createLoop(ring, 'primary', 0.7, 0.55)]),
        createStrokes(
          [createLoop(createRectangleRing(...getRingBounds(ring)), 'dim', 0.5, 0.8)],
          'pulse',
          0.2
        ),
        createStrokes([createLoop(ring, 'warm', 1.3, 0.95)], 'flow', 0.22, 1.2),
        createStrokes([createLine(path, 'dim', 0.7, 0.6)]),
        createStrokes([createLine(path, 'cool', 1.4, 0.95)], 'flow', 0.3, 1.4),
        createStrokes(ticks.map(tick => createLine(tick, 'warm', 0.7, 0.85))),
        createSprites([createGlowDot(centroid, 'hot', 2.8), ...toSprites(ring, 'warm', 0.8)], 0.35)
      ];
    }
  },

  'shape-descriptors': {
    parameters: [createSlider('width', 'Sliver width', 2, 14, 0.5, 4.5)],
    pointer: 'none',
    build: ({p}) => {
      const circle = createCircle(28, 46, 17, 36);
      const star = createStar(80, 46, 21, 10, 5, -Math.PI / 2);
      const sliver = createRectangle(130, 46, 44, p.width, -0.5);
      const shapes = [
        {ring: circle, role: 'cool' as Role, hull: circle, elongation: 0.03},
        {ring: star, role: 'violet' as Role, hull: getConvexHull(star), elongation: 0.06},
        {ring: sliver, role: 'hot' as Role, hull: sliver, elongation: 1 - p.width / 44}
      ];
      const meters: SpriteItem[] = [];
      shapes.forEach((shape, index) => {
        const center = getRingCentroid(shape.ring);
        const compactness = clamp(
          (4 * Math.PI * Math.abs(getSignedArea(shape.ring))) / getRingPerimeter(shape.ring) ** 2,
          0,
          1
        );
        const convexity = Math.abs(getSignedArea(shape.ring)) / Math.abs(getSignedArea(shape.hull));
        const scores = [compactness, convexity, shape.elongation];
        const roles: Role[] = ['cool', 'primary', 'warm'];
        scores.forEach((score, scoreIndex) => {
          meters.push(
            createGlowDot(
              [center[0] - 9 + scoreIndex * 9, 86],
              roles[scoreIndex],
              1.2 + score * 2,
              0.15 + 0.85 * score,
              index * 0.2
            )
          );
        });
      });
      const hull = getConvexHull(star);
      const axis: Point[] = [
        rotatePoint([100, 46], [130, 46], -0.5),
        rotatePoint([160, 46], [130, 46], -0.5)
      ];
      return [
        createFills(
          shapes.map((shape, index) => createArea(shape.ring, shape.role, 0.2, index * 0.3)),
          'pulse'
        ),
        createStrokes(
          shapes.map(shape => createLoop(shape.ring, shape.role, 1, 0.9)),
          'pulse',
          0.25
        ),
        createStrokes([createLoop(hull, 'dim', 0.6, 0.9)], 'pulse', 0.2),
        createStrokes([createLine(axis, 'warm', 0.9, 0.9)], 'flow', 0.3, 1.4),
        createSprites(meters, 0.15)
      ];
    }
  },

  'label-point': {
    parameters: [createSlider('bend', 'Bend', 0.3, 0.95, 0.05, 0.75)],
    pointer: 'none',
    build: ({p}) => {
      const ring = createCrescent(76, 50, 36, 18, Math.PI * p.bend);
      const centroid = getRingCentroid(ring);
      const depth = createDepthField([ring], 64, 40);
      const values = depth.values.map(value => value ** 1.4);
      const field: FieldLayer = {
        type: 'field',
        columns: 64,
        rows: 40,
        values,
        ramp: 'heat',
        style: 'ramp',
        strength: 0.12,
        speed: 0.3
      };
      return [
        field,
        createStrokes([createLoop(ring, 'primary', 1, 0.85)], 'flow', 0.18, 1.5),
        createSprites(
          [createGlowDot(centroid, 'dim', 2.4, 0.9), createGlowDot(centroid, 'hot', 0.9, 0.7)],
          0.1
        ),
        createStrokes([createLine([centroid, depth.best], 'dim', 0.4, 0.5)], 'static'),
        createSprites([createGlowDot(depth.best, 'warm', 3.6, 1)], 0.5)
      ];
    }
  },

  'maximum-inscribed-circle': {
    parameters: [createSlider('roughness', 'Roughness', 0.1, 0.6, 0.05, 0.35)],
    pointer: 'none',
    build: ({fork, p}) => {
      const ring = createBlob(fork('shape'), 80, 50, 34, 22, p.roughness, 0.85);
      const depth = createDepthField([ring], 64, 40);
      const center = depth.best;
      const radius = getRingDistance(center, ring);
      const contacts: Point[] = [];
      for (let index = 0; index < ring.length; index++) {
        const nearest = getNearestOnSegment(
          center,
          ring[index],
          ring[(index + 1) % ring.length]
        ).point;
        if (getDistance(nearest, center) < radius * 1.04) contacts.push(nearest);
      }
      // Quadtree cells that a best-first search would have refined around the answer.
      const cells: PathItem[] = [];
      for (let level = 0; level < 4; level++) {
        const size = 24 / 2 ** level;
        const x = Math.floor(center[0] / size) * size;
        const y = Math.floor(center[1] / size) * size;
        cells.push(
          createLoop(
            [
              [x, y],
              [x + size, y],
              [x + size, y + size],
              [x, y + size]
            ],
            level < 2 ? 'dim' : 'cool',
            0.4,
            0.4 + level * 0.15,
            level * 0.2
          )
        );
      }
      return [
        createFills([createArea(ring, 'primary', 0.12)]),
        createStrokes(cells, 'pulse', 0.3),
        createStrokes([createLoop(ring, 'primary', 1, 0.9)]),
        createFills(
          [createArea(createCircle(center[0], center[1], radius, 48), 'warm', 0.2)],
          'pulse'
        ),
        createStrokes(
          [createLoop(createCircle(center[0], center[1], radius, 48), 'warm', 1.2, 0.95)],
          'flow',
          0.25,
          1.6
        ),
        createStrokes(
          contacts.map(contact => createLine([center, contact], 'warm', 0.5, 0.7)),
          'pulse',
          0.3
        ),
        createSprites(
          [
            createGlowDot(center, 'warm', 2.4),
            ...contacts.map(contact => createGlowDot(contact, 'hot', 2))
          ],
          0.4
        )
      ];
    }
  },

  'minimum-bounds': {
    parameters: [createSlider('rotation', 'Rotation', 0, 180, 5, 30, '°')],
    pointer: 'none',
    build: ({fork, p}) => {
      const angle = (p.rotation * Math.PI) / 180;
      const ring = createBlob(fork('shape'), 80, 50, 30, 16, 0.22, 0.4).map(point =>
        rotatePoint(point, [80, 50], angle)
      );
      const hull = getConvexHull(ring);
      const best = getMinimumRectangle(hull);
      const candidates: PathItem[] = [];
      for (let degrees = 0; degrees < 180; degrees += 22.5) {
        const candidate = getRectangleAtAngle(hull, (degrees * Math.PI) / 180);
        candidates.push(createLoop(candidate.corners, 'cool', 0.4, 0.3, degrees / 180));
      }
      let longest: [Point, Point] = [hull[0], hull[0]];
      let longestLength = 0;
      for (let first = 0; first < hull.length; first++) {
        for (let second = first + 1; second < hull.length; second++) {
          const length = getDistance(hull[first], hull[second]);
          if (length > longestLength) {
            longestLength = length;
            longest = [hull[first], hull[second]];
          }
        }
      }
      const center: Point = [
        (longest[0][0] + longest[1][0]) / 2,
        (longest[0][1] + longest[1][1]) / 2
      ];
      const circleRadius = Math.max(...hull.map(point => getDistance(point, center)));
      return [
        createStrokes(candidates, 'pulse', 0.35),
        createStrokes([createLoop(createRectangleRing(...getRingBounds(ring)), 'dim', 0.5, 0.9)]),
        createStrokes(
          [createLoop(createCircle(center[0], center[1], circleRadius, 48), 'violet', 0.5, 0.45)],
          'pulse',
          0.2
        ),
        createFills([createArea(ring, 'primary', 0.2)], 'pulse'),
        createStrokes([createLoop(ring, 'primary', 0.8, 0.85)]),
        createStrokes([createLoop(best.corners, 'warm', 1.3, 0.95)], 'flow', 0.25, 1.4),
        createStrokes([createLine(longest, 'hot', 0.8, 0.9)], 'flow', 0.3),
        createSprites(
          best.corners.map(corner => createGlowDot(corner, 'warm', 1.5)),
          0.3
        )
      ];
    }
  },

  'minimum-clearance': {
    parameters: [createSlider('neck', 'Neck width', 3, 22, 0.5, 8)],
    pointer: 'none',
    build: ({p}) => {
      const half = p.neck / 2;
      const radius = 26;
      const lobeA = 40;
      const lobeB = 120;
      const offset = Math.asin(half / radius);
      const reach = Math.sqrt(radius * radius - half * half);
      const ring: Point[] = [];
      const steps = 18;
      for (let index = 0; index <= steps; index++) {
        const angle = -offset - (index / steps) * (Math.PI * 2 - offset * 2);
        ring.push([lobeA + Math.cos(angle) * radius, 50 + Math.sin(angle) * radius]);
      }
      for (let index = 0; index <= steps; index++) {
        const angle = Math.PI - offset - (index / steps) * (Math.PI * 2 - offset * 2);
        ring.push([lobeB + Math.cos(angle) * radius, 50 + Math.sin(angle) * radius]);
      }
      // Smallest distance between vertices that are far apart along the ring.
      let best = Infinity;
      let witness: [Point, Point] = [ring[0], ring[1]];
      for (let first = 0; first < ring.length; first++) {
        for (let second = first + 1; second < ring.length; second++) {
          const gap = Math.min(second - first, ring.length - (second - first));
          if (gap < 8) continue;
          const distance = getDistance(ring[first], ring[second]);
          if (distance < best) {
            best = distance;
            witness = [ring[first], ring[second]];
          }
        }
      }
      const middle: Point = [
        (witness[0][0] + witness[1][0]) / 2,
        (witness[0][1] + witness[1][1]) / 2
      ];
      return [
        createFills([createArea(ring, 'primary', 0.14)]),
        createStrokes([createLoop(ring, 'primary', 1, 0.8)], 'flow', 0.15, 1.5),
        createStrokes(
          [createLoop(createCircle(middle[0], middle[1], best / 2, 24), 'cool', 0.6, 0.8)],
          'pulse',
          0.4
        ),
        createStrokes([createLine(witness, 'hot', 1.3, 1)], 'pulse', 0.5),
        createSprites(
          [
            createGlowDot(witness[0], 'hot', 2.4),
            createGlowDot(witness[1], 'hot', 2.4),
            createGlowDot([reach + lobeA, 50], 'dim', 0.01, 0)
          ],
          0.4
        )
      ];
    }
  },

  // Predicates and validity ------------------------------------------------------------------

  'geometry-predicates': {
    parameters: [createSlider('size', 'Query size', 6, 26, 1, 14)],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const shape = createBlob(fork('shape'), 58, 54, 28, 16, 0.25, 0.9);
      const center: Point = pointer ?? [86, 50];
      const radius = p.size;
      const centerInside = isInsideRing(center, shape);
      const edgeDistance = getRingDistance(center, shape);
      const shapeInside = shape.every(point => getDistance(point, center) <= radius);
      const intersects = centerInside || edgeDistance <= radius || shapeInside;
      const contains = centerInside && edgeDistance >= radius;
      const state: Role = contains
        ? 'primary'
        : shapeInside
          ? 'violet'
          : intersects
            ? 'warm'
            : 'cool';
      const queryRing = createCircle(center[0], center[1], radius, 40);
      const lamps: [boolean, Role][] = [
        [intersects, 'warm'],
        [contains, 'primary'],
        [shapeInside, 'violet'],
        [!intersects, 'cool']
      ];
      return [
        createFills([createArea(shape, 'dim', 0.2), createArea(queryRing, state, 0.22)], 'pulse'),
        createStrokes([createLoop(shape, 'dim', 1, 0.9)]),
        createStrokes([createLoop(queryRing, state, 1.3, 1)], 'flow', 0.3, 1.2),
        createSprites(
          lamps.map(([lit, role], index) =>
            createGlowDot([116 + index * 11, 9], lit ? role : 'dim', lit ? 3 : 1.8, lit ? 1 : 0.35)
          ),
          0.3
        )
      ];
    }
  },

  'geometry-validity': {
    parameters: [createSlider('defect', 'Defect', 0, 1, 0.05, 1)],
    pointer: 'none',
    build: ({fork, p}) => {
      const valid = createBlob(fork('valid'), 28, 46, 19, 14, 0.2);
      const flip = p.defect;
      const bowtie: Point[] = [
        [60, 32],
        [100, 32 + 28 * flip],
        [100, 60 - 28 * flip],
        [60, 60]
      ];
      const crossing = intersectSegments(bowtie[0], bowtie[1], bowtie[2], bowtie[3]);
      const shell: Point[] = [
        [112, 30],
        [144, 30],
        [144, 62],
        [112, 62]
      ];
      const holeShift = 14 * p.defect;
      const hole: Point[] = [
        [119 + holeShift, 38],
        [136 + holeShift, 38],
        [136 + holeShift, 54],
        [119 + holeShift, 54]
      ];
      const holeOutside = holeShift > 8;
      const bits: SpriteItem[] = [];
      const flags: number[][] = [[], crossing ? [0] : [], holeOutside ? [2] : []];
      [28, 80, 132].forEach((x, index) => {
        for (let bit = 0; bit < 4; bit++) {
          const lit = flags[index].includes(bit);
          bits.push(
            createGlowDot(
              [x - 10.5 + bit * 7, 88],
              lit ? 'hot' : 'dim',
              lit ? 2.2 : 1.2,
              lit ? 1 : 0.35
            )
          );
        }
      });
      const markers: SpriteItem[] = [];
      if (crossing) markers.push(createGlowDot(crossing, 'hot', 3.2));
      if (holeOutside) markers.push(createGlowDot(hole[1], 'hot', 3.2));
      return [
        createFills(
          [
            createArea(valid, 'primary', 0.2),
            createArea(bowtie, 'hot', 0.15),
            createArea(shell, 'violet', 0.18)
          ],
          'pulse'
        ),
        createStrokes([createLoop(valid, 'primary', 1, 0.9)]),
        createStrokes([createLoop(bowtie, crossing ? 'hot' : 'primary', 1, 0.9)], 'flow', 0.25),
        createStrokes(
          [
            createLoop(shell, 'violet', 1, 0.9),
            createLoop(hole, holeOutside ? 'hot' : 'cool', 1, 0.9)
          ],
          'flow',
          0.2
        ),
        createSprites([...bits, ...markers], 0.5)
      ];
    }
  },

  'make-valid': {
    parameters: [createSlider('twist', 'Twist', 0, 1, 0.05, 1)],
    pointer: 'none',
    build: ({p}) => {
      const flip = p.twist;
      const bowtie: Point[] = [
        [14, 28],
        [54, 28 + 44 * flip],
        [54, 72 - 44 * flip],
        [14, 72]
      ];
      const crossing = intersectSegments(bowtie[0], bowtie[1], bowtie[2], bowtie[3]);
      const arrow: Point[] = [
        [66, 50],
        [90, 50]
      ];
      const output: Point[][] = crossing
        ? [
            [
              [104, 28],
              [crossing[0] + 90, crossing[1]],
              [104, 72]
            ],
            [
              [crossing[0] + 90, crossing[1]],
              [144, 28 + 44 * flip],
              [144, 72 - 44 * flip]
            ]
          ]
        : [
            [
              [104, 28],
              [144, 28 + 44 * flip],
              [144, 72 - 44 * flip],
              [104, 72]
            ]
          ];
      return [
        createFills([createArea(bowtie, 'hot', 0.16)], 'pulse'),
        createStrokes([createLoop(bowtie, 'hot', 1, 0.9)], 'flow', 0.25),
        createStrokes(
          [
            createLine(arrow, 'cool', 1.4, 0.9),
            createLine([[84, 44], arrow[1], [84, 56]], 'cool', 1, 0.9)
          ],
          'flow',
          0.4,
          1.4
        ),
        createFills(
          output.map((ring, index) =>
            createArea(ring, index === 0 ? 'primary' : 'warm', 0.25, index * 0.5)
          ),
          'pulse'
        ),
        createStrokes(
          output.map((ring, index) =>
            createLoop(ring, index === 0 ? 'primary' : 'warm', 1.2, 0.95)
          ),
          'flow',
          0.2
        ),
        createSprites(
          [
            ...(crossing
              ? [
                  createGlowDot(crossing, 'hot', 3),
                  createGlowDot([crossing[0] + 90, crossing[1]], 'violet', 3)
                ]
              : [])
          ],
          0.5
        )
      ];
    }
  },

  // Linework: polygonize, noding, intersection -------------------------------------------------

  polygonize: {
    parameters: [createSlider('lines', 'Lines each way', 2, 5, 1, 4)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('lines');
      const count = Math.round(p.lines);
      const horizontals = Array.from({length: count}, (_, index) => ({
        y: 20 + (index * 60) / (count - 1),
        slope: random.range(-0.12, 0.12)
      }));
      const verticals = Array.from({length: count}, (_, index) => ({
        x: 30 + (index * 100) / (count - 1),
        slope: random.range(-0.12, 0.12)
      }));
      const corners = horizontals.map(horizontal =>
        verticals.map((vertical): Point => {
          const x =
            (vertical.x + vertical.slope * (horizontal.y - horizontal.slope * 80 - 50)) /
            (1 - vertical.slope * horizontal.slope);
          return [x, horizontal.y + horizontal.slope * (x - 80)];
        })
      );
      const faces: PolygonItem[] = [];
      for (let row = 0; row + 1 < count; row++) {
        for (let column = 0; column + 1 < count; column++) {
          faces.push(
            createArea(
              [
                corners[row][column],
                corners[row][column + 1],
                corners[row + 1][column + 1],
                corners[row + 1][column]
              ],
              (['primary', 'warm', 'violet', 'cool'] as const)[(row + column * 2) % 4],
              0.3,
              (row * 3 + column) / 9
            )
          );
        }
      }
      const linework: PathItem[] = [
        ...horizontals.map(horizontal =>
          createLine(
            [
              [6, horizontal.y + horizontal.slope * -74],
              [154, horizontal.y + horizontal.slope * 74]
            ],
            'cool',
            0.8,
            0.9
          )
        ),
        ...verticals.map(vertical =>
          createLine(
            [
              [vertical.x + vertical.slope * -48, 2],
              [vertical.x + vertical.slope * 48, 98]
            ],
            'warm',
            0.8,
            0.9
          )
        )
      ];
      return [
        createFills(faces, 'pulse'),
        createStrokes(linework, 'flow', 0.22, 1.3),
        createSprites(
          faces.map(face => createGlowDot(getRingCentroid(face.ring), 'hot', 1.2, 0.9, face.phase)),
          0.5
        )
      ];
    }
  },

  'line-noding': {
    parameters: [createSlider('lines', 'Lines', 3, 8, 1, 5)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('lines');
      const count = Math.round(p.lines);
      const lines: Point[][] = Array.from({length: count}, (_, index) => {
        const base = 14 + (index * 72) / Math.max(count - 1, 1);
        return index % 2 === 0
          ? [0, 1, 2, 3, 4].map(
              (step): Point => [8 + step * 36, clamp(base + random.range(-16, 16), 6, 94)]
            )
          : [0, 1, 2, 3, 4].map(
              (step): Point => [clamp(base * 1.7 + random.range(-16, 16), 6, 154), 4 + step * 23]
            );
      });
      const {pieces, nodes} = splitPolylines(lines);
      const roles: Role[] = ['primary', 'warm', 'cool', 'violet'];
      return [
        createStrokes(lines.map(points => createLine(points, 'dim', 0.4, 0.5))),
        createStrokes(
          pieces.map((piece, index) =>
            createLine(
              trimPath(piece, 1.2),
              roles[index % roles.length],
              1.1,
              0.95,
              (index * 0.37) % 1
            )
          ),
          'pulse',
          0.3
        ),
        createSprites(
          nodes.map(node => createGlowDot(node, 'hot', 2.2)),
          0.5
        )
      ];
    }
  },

  'segment-intersection': {
    parameters: [createSlider('lines', 'Lines per set', 2, 6, 1, 4)],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('lines');
      const count = Math.round(p.lines);
      const setA: Point[][] = Array.from({length: count}, (_, index) => {
        const y = 16 + (index * 68) / Math.max(count - 1, 1);
        return [
          [6, y],
          [80, y + random.range(-9, 9)],
          [154, y + random.range(-9, 9)]
        ];
      });
      const setB: Point[][] = Array.from({length: count}, (_, index) => {
        const x = 22 + (index * 116) / Math.max(count - 1, 1);
        return [
          [x, 4],
          [x + random.range(-10, 10), 50],
          [x + random.range(-10, 10), 96]
        ];
      });
      if (pointer) {
        setB[0] = [
          [pointer[0] - 4, 4],
          [pointer[0], 50],
          [pointer[0] + 4, 96]
        ];
      }
      const crossings: Point[] = [];
      for (const first of setA) {
        for (const second of setB) {
          for (let firstIndex = 0; firstIndex + 1 < first.length; firstIndex++) {
            for (let secondIndex = 0; secondIndex + 1 < second.length; secondIndex++) {
              const hit = intersectSegments(
                first[firstIndex],
                first[firstIndex + 1],
                second[secondIndex],
                second[secondIndex + 1]
              );
              if (hit) crossings.push(hit);
            }
          }
        }
      }
      return [
        createStrokes(
          setA.map(points => createLine(points, 'cool', 1, 0.9)),
          'flow',
          0.25,
          1.3
        ),
        createStrokes(
          setB.map(points => createLine(points, 'warm', 1, 0.9)),
          'flow',
          0.3,
          1.3
        ),
        createSprites(
          crossings.map((point, index) => createGlowDot(point, 'hot', 2.4, 1, (index % 5) / 5)),
          0.55
        )
      ];
    }
  },

  // Windows and clipping ---------------------------------------------------------------------

  'bounds-filter': {
    parameters: [
      createSlider('window', 'Window width', 30, 90, 2, 56),
      createChoice('mode', 'Predicate', ['intersects', 'within', 'contains'], 0)
    ],
    pointer: 'sticky',
    build: ({fork, p, pointer}) => {
      const random = fork('features');
      const center: Point = pointer ?? [86, 48];
      const width = p.window;
      const height = width * 0.62;
      const box: Box = {
        minimumX: center[0] - width / 2,
        maximumX: center[0] + width / 2,
        minimumY: center[1] - height / 2,
        maximumY: center[1] + height / 2
      };
      const small = Array.from({length: 40}, (): Point[] => {
        const x = random.range(10, 150);
        const y = random.range(8, 92);
        const angle = random.range(0, Math.PI);
        const half = random.range(2.5, 6);
        return [
          [x - Math.cos(angle) * half, y - Math.sin(angle) * half],
          [x + Math.cos(angle) * half, y + Math.sin(angle) * half]
        ];
      });
      const large =
        p.mode === 2
          ? [
              createBlob(fork('large-a'), 80, 50, 62, 12, 0.1, 0.75),
              createBlob(fork('large-b'), 100, 46, 46, 12, 0.1, 0.8)
            ].map(ring => closeRing(ring))
          : [];
      const features = [...small, ...large];
      const selected = (points: Point[]) => {
        const [minimumX, minimumY, maximumX, maximumY] = getRingBounds(points);
        if (p.mode === 1) {
          return (
            minimumX >= box.minimumX &&
            maximumX <= box.maximumX &&
            minimumY >= box.minimumY &&
            maximumY <= box.maximumY
          );
        }
        if (p.mode === 2) {
          return (
            minimumX <= box.minimumX &&
            maximumX >= box.maximumX &&
            minimumY <= box.minimumY &&
            maximumY >= box.maximumY
          );
        }
        return (
          minimumX <= box.maximumX &&
          maximumX >= box.minimumX &&
          minimumY <= box.maximumY &&
          maximumY >= box.minimumY
        );
      };
      const kept = features.filter(selected);
      const dropped = features.filter(points => !selected(points));
      const frame = createLoop(
        [
          [box.minimumX, box.minimumY],
          [box.maximumX, box.minimumY],
          [box.maximumX, box.maximumY],
          [box.minimumX, box.maximumY]
        ],
        'primary',
        1.2,
        0.95
      );
      return [
        createFills([createArea(frame.points.slice(0, 4), 'primary', 0.07)]),
        createStrokes(dropped.map(points => createLine(points, 'dim', 0.6, 0.55))),
        createStrokes(
          kept.map(points =>
            createLoop(createRectangleRing(...getRingBounds(points)), 'warm', 0.4, 0.45)
          ),
          'pulse',
          0.3
        ),
        createStrokes(
          kept.map(points => createLine(points, 'warm', 1.2, 1)),
          'flow',
          0.3,
          1.5
        ),
        createStrokes([frame], 'flow', 0.2, 1.4)
      ];
    }
  },

  'rectangle-clip': {
    parameters: [createSlider('window', 'Window width', 30, 100, 2, 80)],
    pointer: 'sticky',
    build: ({fork, p, pointer}) => {
      const center: Point = pointer ?? [80, 50];
      const width = p.window;
      const height = width * 0.62;
      const box: Box = {
        minimumX: center[0] - width / 2,
        maximumX: center[0] + width / 2,
        minimumY: center[1] - height / 2,
        maximumY: center[1] + height / 2
      };
      const polygons = [
        createBlob(fork('a'), 46, 42, 26, 14, 0.3),
        createBlob(fork('b'), 112, 62, 28, 14, 0.3),
        createBlob(fork('c'), 120, 24, 14, 10, 0.3)
      ];
      const paths: Point[][] = [
        smoothPath(
          [
            [4, 80],
            [50, 62],
            [92, 82],
            [156, 54]
          ],
          2
        ),
        smoothPath(
          [
            [8, 14],
            [70, 36],
            [100, 8],
            [156, 30]
          ],
          2
        )
      ];
      const clippedRings = polygons
        .map(ring => clipRingToBox(ring, box))
        .filter(ring => ring.length > 2);
      const clippedPieces: Point[][] = [];
      for (const path of paths) {
        let current: Point[] = [];
        for (let index = 0; index + 1 < path.length; index++) {
          const piece = clipSegmentToBox(path[index], path[index + 1], box);
          if (piece) {
            if (current.length === 0) current.push(piece[0]);
            current.push(piece[1]);
          } else if (current.length) {
            clippedPieces.push(current);
            current = [];
          }
        }
        if (current.length) clippedPieces.push(current);
      }
      const frame: Point[] = [
        [box.minimumX, box.minimumY],
        [box.maximumX, box.minimumY],
        [box.maximumX, box.maximumY],
        [box.minimumX, box.maximumY]
      ];
      return [
        createStrokes([
          ...polygons.map(ring => createLoop(ring, 'dim', 0.7, 0.55)),
          ...paths.map(path => createLine(path, 'dim', 0.7, 0.55))
        ]),
        createFills(
          clippedRings.map((ring, index) => createArea(ring, 'primary', 0.3, index * 0.3)),
          'pulse'
        ),
        createStrokes(clippedRings.map(ring => createLoop(ring, 'primary', 1, 0.9))),
        createStrokes(
          clippedPieces.map(piece => createLine(piece, 'warm', 1.4, 1)),
          'flow',
          0.3,
          1.5
        ),
        createStrokes([createLoop(frame, 'cool', 1.1, 0.95)], 'flow', 0.2, 1.4)
      ];
    }
  },

  // Transforms and orientation ------------------------------------------------------------------

  'affine-transform': {
    parameters: [
      createSlider('rotation', 'Rotation', -90, 90, 5, 30, '°'),
      createSlider('scale', 'Scale', 0.5, 1.5, 0.05, 1.1),
      createSlider('shear', 'Shear', -0.8, 0.8, 0.05, 0.3)
    ],
    pointer: 'none',
    build: ({p}) => {
      const glyph: Point[] = [
        [-0.6, -1],
        [0.7, -1],
        [0.7, -0.6],
        [-0.2, -0.6],
        [-0.2, -0.2],
        [0.4, -0.2],
        [0.4, 0.2],
        [-0.2, 0.2],
        [-0.2, 1],
        [-0.6, 1]
      ];
      const angle = (p.rotation * Math.PI) / 180;
      const cos = Math.cos(angle) * p.scale;
      const sin = Math.sin(angle) * p.scale;
      const apply = ([x, y]: Point, origin: Point, unit: number): Point => {
        const sheared = x + p.shear * y;
        return [
          origin[0] + (sheared * cos - y * sin) * unit,
          origin[1] + (sheared * sin + y * cos) * unit
        ];
      };
      const identity = ([x, y]: Point, origin: Point, unit: number): Point => [
        origin[0] + x * unit,
        origin[1] + y * unit
      ];
      const origins: [Point, Point] = [
        [40, 50],
        [118, 50]
      ];
      const unit = 17;
      const lattice = (transform: typeof apply, origin: Point) => {
        const items: PathItem[] = [];
        for (let step = -1.5; step <= 1.5; step += 0.75) {
          items.push(
            createLine(
              [transform([step, -1.5], origin, unit), transform([step, 1.5], origin, unit)],
              'dim',
              0.35,
              0.55
            )
          );
          items.push(
            createLine(
              [transform([-1.5, step], origin, unit), transform([1.5, step], origin, unit)],
              'dim',
              0.35,
              0.55
            )
          );
        }
        return items;
      };
      const source = glyph.map(point => identity(point, origins[0], unit));
      const target = glyph.map(point => apply(point, origins[1], unit));
      const links = source.map((point, index) =>
        createLine(
          createArc(point, target[index], 0.12, 14),
          'cool',
          0.45,
          0.55,
          index / source.length
        )
      );
      return [
        createStrokes(lattice(identity, origins[0])),
        createStrokes(lattice(apply, origins[1])),
        createStrokes(links, 'flow', 0.3, 0.9),
        createFills([createArea(source, 'cool', 0.2), createArea(target, 'warm', 0.28)], 'pulse'),
        createStrokes([createLoop(source, 'cool', 1.1, 0.9)]),
        createStrokes([createLoop(target, 'warm', 1.3, 1)], 'flow', 0.22, 1.4),
        createSprites([...toSprites(source, 'cool', 1), ...toSprites(target, 'warm', 1.2)], 0.3)
      ];
    }
  },

  'geometry-orientation': {
    parameters: [createChoice('convention', 'Convention', ['GeoJSON', 'Shapefile'], 0)],
    pointer: 'none',
    build: ({p}) => {
      const geoJson = p.convention === 0;
      const makeRings = (centerX: number, outerPositive: boolean, holePositive: boolean) => ({
        outer: orientRing(createRegularPolygon(centerX, 50, 28, 8, 0.4), outerPositive),
        hole: orientRing(createRegularPolygon(centerX, 50, 11, 6, 0.2), holePositive)
      });
      const input = makeRings(38, true, true);
      const output = makeRings(122, !geoJson, geoJson);
      const outerChanged = geoJson;
      const holeChanged = !geoJson;
      const asFlow = (ring: Point[], role: Role) => createLoop(ring, role, 1.4, 0.95);
      return [
        createFills([
          createArea(input.outer, 'cool', 0.1),
          createArea(output.outer, 'primary', 0.1)
        ]),
        createStrokes(
          [
            createLine(
              [
                [68, 50],
                [92, 50]
              ],
              'dim',
              0.8,
              0.9
            ),
            createLine(
              [
                [86, 44],
                [92, 50],
                [86, 56]
              ],
              'dim',
              0.8,
              0.9
            )
          ],
          'flow',
          0.3,
          1.3
        ),
        createStrokes([asFlow(input.outer, 'cool'), asFlow(input.hole, 'cool')], 'flow', 0.22, 1.1),
        createStrokes(
          [
            asFlow(output.outer, outerChanged ? 'warm' : 'primary'),
            asFlow(output.hole, holeChanged ? 'warm' : 'primary')
          ],
          'flow',
          0.22,
          1.1
        ),
        createSprites(
          [
            createGlowDot(input.outer[0], 'hot', 2),
            createGlowDot(input.hole[0], 'hot', 2),
            createGlowDot(output.outer[0], 'hot', 2),
            createGlowDot(output.hole[0], 'hot', 2)
          ],
          0.4
        )
      ];
    }
  },

  // Snapping and precision ----------------------------------------------------------------------

  'geometry-cleanup': {
    parameters: [createSlider('grid', 'Grid size', 4, 14, 1, 7)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('path');
      const base = createWanderPath(random, [12, 70], -0.3, 4.6, 40, 0.2).map(point =>
        clampPoint(point, 8)
      );
      const raw: Point[] = [];
      for (const point of base) {
        raw.push(point);
        if (random.next() < 0.45)
          raw.push([point[0] + random.range(-0.5, 0.5), point[1] + random.range(-0.5, 0.5)]);
      }
      const size = p.grid;
      const snapped = raw.map(
        ([x, y]): Point => [Math.round(x / size) * size, Math.round(y / size) * size]
      );
      const kept: Point[] = [];
      const dropped: Point[] = [];
      snapped.forEach((point, index) => {
        const last = kept[kept.length - 1];
        if (last && last[0] === point[0] && last[1] === point[1]) dropped.push(raw[index]);
        else kept.push(point);
      });
      const lattice = new Set<string>();
      for (const [x, y] of kept) {
        for (let dx = -1; dx <= 1; dx++)
          for (let dy = -1; dy <= 1; dy++) lattice.add(`${x + dx * size},${y + dy * size}`);
      }
      const nodes = [...lattice].map(key => key.split(',').map(Number) as Point);
      return [
        createSprites(
          nodes.map(node => createGlowDot(node, 'dim', 0.6, 0.55)),
          0
        ),
        createStrokes([createLine(raw, 'cool', 0.6, 0.45)]),
        createStrokes([createLine(kept, 'warm', 1.3, 1)], 'flow', 0.25, 1.4),
        createSprites(
          [
            ...kept.map(point => createGlowDot(point, 'warm', 1.5)),
            ...dropped.map(point => createGlowDot(point, 'hot', 1.1, 0.8))
          ],
          0.35
        )
      ];
    }
  },

  'vertex-snap': {
    parameters: [createSlider('tolerance', 'Tolerance', 1, 9, 0.5, 5)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('noise');
      const reference: Point[] = [
        [16, 22],
        [74, 16],
        [80, 78],
        [20, 84]
      ];
      const jitter = (point: Point): Point => [
        point[0] + random.range(-3.4, 3.4),
        point[1] + random.range(-3.4, 3.4)
      ];
      const park: Point[] = [
        jitter([74, 16]),
        [142, 24],
        [146, 66],
        [120, 86],
        jitter([80, 78]),
        [104, 50]
      ];
      const snapped = park.map(point => {
        let best: Point | null = null;
        let bestDistance = p.tolerance;
        for (const candidate of reference) {
          const distance = getDistance(point, candidate);
          if (distance <= bestDistance) {
            bestDistance = distance;
            best = candidate;
          }
        }
        return best ?? point;
      });
      const moved = park.filter((point, index) => point !== snapped[index]);
      return [
        createFills(
          [createArea(reference, 'cool', 0.1), createArea(snapped, 'warm', 0.16)],
          'pulse'
        ),
        createStrokes(
          reference.map(point =>
            createLoop(createCircle(point[0], point[1], p.tolerance, 24), 'dim', 0.45, 0.6)
          ),
          'pulse',
          0.3
        ),
        createStrokes([createLoop(reference, 'cool', 1, 0.9)]),
        createStrokes([createLoop(park, 'dim', 0.8, 0.7)]),
        createStrokes([createLoop(snapped, 'warm', 1.3, 1)], 'flow', 0.22, 1.3),
        createStrokes(
          park.map((point, index) => createLine([point, snapped[index]], 'hot', 0.7, 0.9)),
          'flow',
          0.5,
          0.8
        ),
        createSprites(
          [
            ...toSprites(reference, 'cool', 1.7),
            ...moved.map(point => createGlowDot(point, 'hot', 1.5, 0.7))
          ],
          0.3
        )
      ];
    }
  },

  'segment-snap': {
    parameters: [createSlider('tolerance', 'Tolerance', 2, 11, 0.5, 6)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('nodes');
      const original: Point[] = [
        [8, 72],
        [80, 48],
        [152, 66]
      ];
      const nodes: Point[] = [
        [11, 66],
        [149, 61],
        ...[26, 48, 62, 100, 126].map((x): Point => {
          const segment = x < 80 ? 0 : 1;
          const along = segment === 0 ? (x - 8) / 72 : (x - 80) / 72;
          const base = polylineAt(original, segment + along);
          return [x, base[1] + random.range(-9, 9) * (random.next() < 0.5 ? 1 : -1)];
        })
      ];
      // Endpoints snap to their nearest node; inner nodes within tolerance are inserted in order.
      const output: Point[] = [nodes[0]];
      const inserted: Point[] = [];
      for (let segment = 0; segment < 2; segment++) {
        const start = original[segment];
        const end = original[segment + 1];
        const hits = nodes
          .slice(2)
          .filter(
            node =>
              getSegmentDistance(node, start, end) <= p.tolerance &&
              node[0] > start[0] + 2 &&
              node[0] < end[0] - 2
          )
          .sort((first, second) => first[0] - second[0]);
        inserted.push(...hits);
        output.push(...hits);
        if (segment === 0) output.push(original[1]);
      }
      output.push(nodes[1]);
      return [
        createStrokes([createLine(original, 'cool', p.tolerance * 2, 0.07)]),
        createStrokes([createLine(original, 'dim', 0.7, 0.6)]),
        createStrokes([createLine(output, 'warm', 1.4, 1)], 'flow', 0.25, 1.4),
        createSprites(
          [
            ...nodes.map(node => createGlowDot(node, 'cool', 1.6, 0.75)),
            ...inserted.map(node => createGlowDot(node, 'hot', 2.4)),
            createGlowDot(nodes[0], 'hot', 2.4),
            createGlowDot(nodes[1], 'hot', 2.4)
          ],
          0.4
        )
      ];
    }
  },

  'line-precision-reduction': {
    parameters: [
      createSlider('grid', 'Grid size', 4, 16, 1, 9),
      createChoice('policenterY', 'PolicenterY', ['valid-output', 'keep-collapsed'], 1)
    ],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('lines');
      const columns = Math.round(SCENE_WIDTH / p.grid);
      const rows = Math.round(SCENE_HEIGHT / p.grid);
      const stepX = SCENE_WIDTH / columns;
      const stepY = SCENE_HEIGHT / rows;
      const snapPoint = ([x, y]: Point): Point => [
        Math.round(x / stepX) * stepX,
        Math.round(y / stepY) * stepY
      ];
      const longLines = [24, 52, 80].map(y =>
        smoothPath(
          createWanderPath(random, [10, y], 0, 10, 12, 0.22).map(point => clampPoint(point, 6)),
          1
        )
      );
      const shortLines = Array.from({length: 6}, (_, index): Point[] => {
        const x = 22 + index * 22 + random.range(-4, 4);
        const y = random.range(10, 90);
        return [
          [x, y],
          [x + random.range(1.5, 4), y + random.range(-1.5, 1.5)]
        ];
      });
      const result: {points: Point[]; collapsed: boolean}[] = [];
      for (const points of [...longLines, ...shortLines]) {
        const snapped = points
          .map(snapPoint)
          .filter(
            (point, index, array) =>
              index === 0 || point[0] !== array[index - 1][0] || point[1] !== array[index - 1][1]
          );
        result.push({points: snapped, collapsed: snapped.length < 2});
      }
      const lattice: SpriteItem[] = [];
      for (let column = 0; column <= columns; column++) {
        for (let row = 0; row <= rows; row++)
          lattice.push(createGlowDot([column * stepX, row * stepY], 'dim', 0.45, 0.45));
      }
      const collapsed = result.filter(item => item.collapsed);
      return [
        createSprites(lattice, 0),
        createStrokes(
          [...longLines, ...shortLines].map(points => createLine(points, 'cool', 0.6, 0.45))
        ),
        createStrokes(
          result
            .filter(item => !item.collapsed)
            .map(item => createLine(item.points, 'warm', 1.3, 1)),
          'flow',
          0.25,
          1.4
        ),
        createSprites(
          p.policenterY === 1
            ? collapsed.map(item => createGlowDot(item.points[0], 'hot', 2.6))
            : [],
          0.6
        )
      ];
    }
  },

  'polygon-precision-reduction': {
    parameters: [createSlider('grid', 'Grid size', 6, 20, 1, 10)],
    pointer: 'none',
    build: ({fork, p}) => {
      const columns = Math.round(SCENE_WIDTH / p.grid);
      const rows = Math.round(SCENE_HEIGHT / p.grid);
      const stepX = SCENE_WIDTH / columns;
      const stepY = SCENE_HEIGHT / rows;
      const snapRing = (ring: Point[]) => {
        const snapped = ring.map(
          ([x, y]): Point => [Math.round(x / stepX) * stepX, Math.round(y / stepY) * stepY]
        );
        const unique = snapped.filter((point, index, array) => {
          const previous = array[(index + array.length - 1) % array.length];
          return point[0] !== previous[0] || point[1] !== previous[1];
        });
        return unique.length < 3 || Math.abs(getSignedArea(unique)) < 1e-6 ? null : unique;
      };
      const originals = [
        createBlob(fork('a'), 46, 48, 32, 18, 0.3, 0.9),
        createRegularPolygon(114, 38, 20, 6, 0.3),
        createRectangle(112, 78, 40, 1.8, -0.1)
      ];
      const snapped = originals.map(snapRing);
      const values = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const point: Point = [(column + 0.5) * stepX, (row + 0.5) * stepY];
          if (snapped.some(ring => ring && isInsideRing(point, ring)))
            values[row * columns + column] = 0.36;
        }
      }
      const collapsed = originals.filter((_, index) => !snapped[index]);
      const field: FieldLayer = {
        type: 'field',
        columns,
        rows,
        values,
        ramp: 'ice',
        style: 'ramp',
        filter: 'nearest',
        strength: 0.15,
        speed: 0.3
      };
      return [
        field,
        createStrokes(originals.map(ring => createLoop(ring, 'cool', 0.7, 0.55))),
        createStrokes(
          snapped
            .filter((ring): ring is Point[] => !!ring)
            .map(ring => createLoop(ring, 'warm', 1.3, 1)),
          'flow',
          0.22,
          1.2
        ),
        createSprites(
          collapsed.map(ring => createGlowDot(getRingCentroid(ring), 'hot', 2.8)),
          0.6
        )
      ];
    }
  },

  // Cutting loops and offsets ------------------------------------------------------------------

  'line-loop-removal': {
    parameters: [createSlider('curl', 'Curl', 0.4, 2.2, 0.05, 1.7)],
    pointer: 'none',
    build: ({p}) => {
      const samples = 90;
      const path: Point[] = Array.from({length: samples + 1}, (_, index): Point => {
        const t = (index / samples) * Math.PI * 4;
        return [
          14 + t * 9 - Math.sin(t) * 10 * p.curl,
          52 - Math.cos(t) * 12 * p.curl + Math.sin(t * 0.5) * 6
        ];
      });
      // Cut each loop out at its first self-crossing, in drawing order.
      const kept: Point[] = [];
      const loops: Point[][] = [];
      const crossings: Point[] = [];
      let index = 0;
      while (index < path.length - 1) {
        let found = false;
        for (let later = index + 2; later < path.length - 1; later++) {
          const hit = intersectSegments(path[index], path[index + 1], path[later], path[later + 1]);
          if (hit) {
            kept.push(path[index], hit);
            loops.push([hit, ...path.slice(index + 1, later + 1), hit]);
            crossings.push(hit);
            index = later + 1;
            found = true;
            break;
          }
        }
        if (!found) {
          kept.push(path[index]);
          index++;
        }
      }
      kept.push(path[path.length - 1]);
      return [
        createStrokes([createLine(path, 'dim', 0.7, 0.55)]),
        createStrokes(
          loops.map(points => createLine(points, 'hot', 1.2, 0.95)),
          'pulse',
          0.45
        ),
        createStrokes([createLine(kept, 'warm', 1.5, 1)], 'flow', 0.2, 1.8),
        createSprites(
          crossings.map(point => createGlowDot(point, 'hot', 2.4)),
          0.5
        ),
        createSprites(
          [createGlowDot(path[0], 'cool', 2), createGlowDot(path[path.length - 1], 'cool', 2)],
          0.2
        )
      ];
    }
  },

  'offset-curve-components': {
    parameters: [createSlider('distance', 'Distance', 3, 14, 0.5, 8)],
    pointer: 'none',
    build: ({p}) => {
      const source: Point[] = [
        [10, 74],
        [48, 70],
        [62, 36],
        [100, 30],
        [112, 58],
        [152, 52]
      ];
      // Raw one-sided offset: every segment shifted on its own, joined end to start.
      const raw: Point[] = [];
      for (let index = 0; index + 1 < source.length; index++) {
        const [ax, ay] = source[index];
        const [bx, by] = source[index + 1];
        const length = Math.hypot(bx - ax, by - ay);
        const nx = ((by - ay) / length) * p.distance;
        const ny = (-(bx - ax) / length) * p.distance;
        raw.push([ax + nx, ay + ny], [bx + nx, by + ny]);
        // Round join around outer corners, so the raw curve stays at the offset distance.
        if (index + 2 < source.length) {
          const [nextX, nextY] = source[index + 2];
          const nextLength = Math.hypot(nextX - bx, nextY - by);
          const turnsAway =
            ((nextX - bx) / nextLength) * (nx / p.distance) +
              ((nextY - by) / nextLength) * (ny / p.distance) <
            0;
          if (turnsAway) {
            const from = Math.atan2(ny, nx);
            const to = Math.atan2(-(nextX - bx), nextY - by);
            let sweep = to - from;
            while (sweep > Math.PI) sweep -= Math.PI * 2;
            while (sweep < -Math.PI) sweep += Math.PI * 2;
            for (let step = 1; step < 6; step++) {
              const angle = from + (sweep * step) / 6;
              raw.push([bx + Math.cos(angle) * p.distance, by + Math.sin(angle) * p.distance]);
            }
          }
        }
      }
      const {pieces} = splitPolylines([raw]);
      const keptPieces: Point[][] = [];
      const discarded: Point[][] = [];
      for (const piece of pieces) {
        let nearest = Infinity;
        for (let sample = 0; sample < piece.length * 2 - 1; sample++) {
          const probe = polylineAt(piece, sample / 2);
          for (let index = 0; index + 1 < source.length; index++) {
            nearest = Math.min(
              nearest,
              getSegmentDistance(probe, source[index], source[index + 1])
            );
          }
        }
        if (nearest >= p.distance * 0.97) keptPieces.push(piece);
        else discarded.push(piece);
      }
      return [
        createStrokes([createLine(source, 'cool', 6, 0.1)]),
        createStrokes([createLine(source, 'cool', 1.2, 0.9)]),
        createStrokes([createLine(raw, 'dim', 0.6, 0.55)]),
        createStrokes(
          discarded.map(piece => createLine(piece, 'hot', 1, 0.7)),
          'pulse',
          0.4
        ),
        createStrokes(
          keptPieces.map(piece => createLine(piece, 'warm', 1.5, 1)),
          'flow',
          0.25,
          1.5
        )
      ];
    }
  },

  // Normalization and equality ------------------------------------------------------------------

  'normalize-polygons': {
    parameters: [createSlider('vertices', 'Vertices', 5, 12, 1, 8)],
    pointer: 'none',
    build: ({fork, p}) => {
      const unit = createBlob(fork('shape'), 0, 0, 1, Math.round(p.vertices), 0.32);
      const place = (ring: Point[], centerX: number, centerY: number, scale: number) =>
        ring.map(([x, y]): Point => [centerX + x * scale, centerY + y * scale]);
      const count = unit.length;
      const startA = 3 % count;
      const startB = (count - 2) % count;
      const rotateTo = (ring: Point[], start: number) => [
        ...ring.slice(start),
        ...ring.slice(0, start)
      ];
      let canonical = 0;
      unit.forEach((point, index) => {
        if (point[0] + point[1] * 0.01 < unit[canonical][0] + unit[canonical][1] * 0.01)
          canonical = index;
      });
      const inputA = rotateTo(place(unit, 38, 28, 17), startA);
      const inputB = rotateTo(
        [...place(unit, 38, 72, 17)].reverse(),
        (count - 1 - startB + count) % count
      );
      const output = rotateTo(place(unit, 118, 50, 27), canonical);
      return [
        createStrokes(
          [
            createLine(createArc([62, 30], [88, 46], 0.18, 14), 'dim', 0.7, 0.7),
            createLine(createArc([62, 72], [88, 54], -0.18, 14), 'dim', 0.7, 0.7)
          ],
          'flow',
          0.3,
          1.2
        ),
        createFills(
          [
            createArea(inputA, 'cool', 0.12),
            createArea(inputB, 'violet', 0.12),
            createArea(output, 'primary', 0.2)
          ],
          'pulse'
        ),
        createStrokes(
          [createLoop(inputA, 'cool', 1.1, 0.9), createLoop(inputB, 'violet', 1.1, 0.9)],
          'flow',
          0.2,
          1
        ),
        createStrokes([createLoop(output, 'primary', 1.4, 1)], 'flow', 0.2, 1.1),
        createSprites(
          [
            createGlowDot(inputA[0], 'hot', 2.6),
            createGlowDot(inputB[0], 'hot', 2.6),
            createGlowDot(output[0], 'warm', 3.4)
          ],
          0.5
        )
      ];
    }
  },

  'canonical-polygon-equality': {
    parameters: [createSlider('tolerance', 'Tolerance', 0, 2, 0.1, 0.8)],
    pointer: 'none',
    build: ({fork, p}) => {
      const rows = [20, 50, 80];
      const jitterMagnitude = [0, 5, 0.5];
      const outlines: PathItem[] = [];
      const fillItems: PolygonItem[] = [];
      const lamps: SpriteItem[] = [];
      rows.forEach((centerY, index) => {
        const unit = createBlob(fork(`row-${index}`), 0, 0, 1, 7, 0.28);
        const shape = (centerX: number) =>
          unit.map(([x, y]): Point => [centerX + x * 12, centerY + y * 9]);
        const left = shape(24);
        const right = shape(96).map(
          ([x, y], vertex): Point => (vertex === 2 ? [x + jitterMagnitude[index], y] : [x, y])
        );
        const equal = jitterMagnitude[index] <= p.tolerance;
        const overlayA = shape(132);
        const overlayB = right.map(([x, y]): Point => [x + 36, y]);
        fillItems.push(
          createArea(left, 'cool', 0.14, index * 0.3),
          createArea(right, 'warm', 0.14, index * 0.3)
        );
        outlines.push(createLoop(left, 'cool', 1, 0.9), createLoop(right, 'warm', 1, 0.9));
        outlines.push(createLoop(overlayA, 'cool', 1, 0.9), createLoop(overlayB, 'warm', 1, 0.9));
        lamps.push(
          createGlowDot([56, centerY], equal ? 'primary' : 'hot', 2.6),
          createGlowDot([70, centerY], equal ? 'primary' : 'hot', 0.01, 0)
        );
        lamps.push(createGlowDot([156, centerY], equal ? 'primary' : 'hot', 2.4));
      });
      return [
        createFills(fillItems, 'pulse'),
        createStrokes(outlines, 'pulse', 0.25),
        createStrokes(
          rows.map(centerY =>
            createLine(
              [
                [40, centerY],
                [76, centerY]
              ],
              'dim',
              0.5,
              0.5
            )
          ),
          'flow',
          0.3,
          0.8
        ),
        createSprites(lamps, 0.5)
      ];
    }
  },

  'normalize-geometry': {
    parameters: [createSlider('nesting', 'Nesting', 1, 3, 1, 2)],
    pointer: 'none',
    build: ({p}) => {
      const leaf = (kind: TreeKind): TreeNode => ({kind});
      const deeper: TreeNode[] =
        p.nesting > 1
          ? [{kind: 'collection', children: [leaf('point'), leaf('line')]}]
          : [leaf('line')];
      const root: TreeNode = {
        kind: 'collection',
        children: [
          {kind: 'collection', children: [leaf('point'), ...deeper]},
          leaf('polygon'),
          {
            kind: 'collection',
            children: [
              leaf('polygon'),
              ...(p.nesting > 2
                ? [{kind: 'collection', children: [leaf('empty')]} as TreeNode]
                : [leaf('point')]),
              leaf('line')
            ]
          }
        ]
      };
      const tree = layoutTree(root, 14, 20, 14, 86);
      const leaves = tree.nodes
        .filter(node => node.leaf)
        .sort((first, second) => first.point[1] - second.point[1]);
      const glyphs: PreviewLayer[] = [];
      const glyphFills: PolygonItem[] = [];
      const glyphLines: PathItem[] = [];
      const glyphDots: SpriteItem[] = [];
      leaves.forEach(node => {
        const [, y] = node.point;
        if (node.kind === 'point') glyphDots.push(createGlowDot([136, y], 'cool', 2.4));
        else if (node.kind === 'line')
          glyphLines.push(
            createLine(
              [
                [126, y + 2],
                [136, y - 2],
                [146, y + 2]
              ],
              'warm',
              1.2,
              0.95
            )
          );
        else if (node.kind === 'polygon')
          glyphFills.push(createArea(createRegularPolygon(136, y, 4.6, 5, 0.3), 'primary', 0.45));
        else glyphDots.push(createGlowDot([136, y], 'dim', 1.5, 0.6));
      });
      glyphs.push(
        createFills(glyphFills, 'pulse'),
        createStrokes(glyphLines, 'pulse', 0.3),
        createSprites(glyphDots, 0.4)
      );
      return [
        createStrokes(
          tree.links.map(link => createLine(link, 'violet', 0.8, 0.7)),
          'flow',
          0.3,
          1.1
        ),
        createStrokes(
          leaves.map(node => createLine([node.point, [124, node.point[1]]], 'dim', 0.4, 0.5)),
          'flow',
          0.35,
          0.8
        ),
        createSprites(
          tree.nodes.map((node, index) =>
            createGlowDot(
              node.point,
              node.depth === 0 ? 'hot' : TREE_ROLES[node.kind],
              node.leaf ? 2.2 : 2.8,
              0.95,
              index / tree.nodes.length
            )
          ),
          0.3
        ),
        ...glyphs
      ];
    }
  },

  'canonical-geometry-equality': {
    parameters: [createChoice('verdict', 'Trees', ['equal', 'different'], 1)],
    pointer: 'none',
    build: ({p}) => {
      const leaf = (kind: TreeKind): TreeNode => ({kind});
      const different = p.verdict === 1;
      const makeTree = (changed: boolean): TreeNode => ({
        kind: 'collection',
        children: [
          {kind: 'collection', children: [leaf('point'), leaf('polygon')]},
          {kind: 'collection', children: [leaf(changed ? 'line' : 'polygon'), leaf('empty')]},
          leaf('line')
        ]
      });
      const treeA = layoutTree(makeTree(false), 12, 18, 16, 84);
      const treeB = layoutTree(makeTree(different), 148, -18, 16, 84);
      const links: PathItem[] = [];
      const markers: SpriteItem[] = [];
      treeA.nodes.forEach((node, index) => {
        const other = treeB.nodes[index];
        const matches = node.kind === other.kind;
        links.push(
          createLine(
            [node.point, other.point],
            matches ? 'cool' : 'hot',
            matches ? 0.4 : 0.9,
            matches ? 0.35 : 0.95,
            index / treeA.nodes.length
          )
        );
        markers.push(
          createGlowDot(
            node.point,
            node.depth === 0 ? 'hot' : TREE_ROLES[node.kind],
            node.leaf ? 2.2 : 2.8,
            0.95,
            index / 8
          )
        );
        markers.push(
          createGlowDot(
            other.point,
            node.depth === 0 ? 'hot' : TREE_ROLES[other.kind],
            other.leaf ? 2.2 : 2.8,
            0.95,
            index / 8
          )
        );
      });
      const verdict: Role = different ? 'hot' : 'primary';
      return [
        createStrokes(links, 'flow', 0.35, 0.9),
        createStrokes(
          [...treeA.links, ...treeB.links].map(link => createLine(link, 'violet', 0.8, 0.6))
        ),
        createSprites(markers, 0.3),
        createSprites(
          [createGlowDot([80, 50], verdict, 6, 0.4), createGlowDot([80, 50], verdict, 3.4, 1)],
          0.5
        )
      ];
    }
  },

  // Subdivision, hulls, skeletons -------------------------------------------------------------

  'subdivide-polygons': {
    parameters: [createSlider('depth', 'Splits', 1, 5, 1, 3)],
    pointer: 'none',
    build: ({fork, p}) => {
      const coast = createBlob(fork('coast'), 80, 50, 40, 44, 0.55, 0.78);
      let pieces: Point[][] = [coast];
      for (let level = 0; level < Math.round(p.depth); level++) {
        const next: Point[][] = [];
        for (const piece of pieces) {
          const [minimumX, minimumY, maximumX, maximumY] = getRingBounds(piece);
          const vertical = maximumX - minimumX > maximumY - minimumY;
          const middle = vertical ? (minimumX + maximumX) / 2 : (minimumY + maximumY) / 2;
          const first = vertical
            ? clipRingBelowLimit(piece, 1, 0, middle)
            : clipRingBelowLimit(piece, 0, 1, middle);
          const second = vertical
            ? clipRingBelowLimit(piece, -1, 0, -middle)
            : clipRingBelowLimit(piece, 0, -1, -middle);
          for (const half of [first, second]) if (half.length > 2) next.push(half);
        }
        pieces = next;
      }
      const roles: Role[] = ['primary', 'warm', 'cool', 'violet'];
      const shrunk = pieces.map(piece => {
        const center = getRingCentroid(piece);
        return piece.map(
          ([x, y]): Point => [mixNumbers(center[0], x, 0.9), mixNumbers(center[1], y, 0.9)]
        );
      });
      return [
        createStrokes([createLoop(coast, 'dim', 0.7, 0.55)]),
        createFills(
          shrunk.map((piece, index) =>
            createArea(piece, roles[index % 4], 0.3, (index * 0.17) % 1)
          ),
          'pulse'
        ),
        createStrokes(
          shrunk.map((piece, index) =>
            createLoop(piece, roles[index % 4], 0.8, 0.9, (index * 0.17) % 1)
          ),
          'pulse',
          0.25
        )
      ];
    }
  },

  'ratio-concave-hull': {
    parameters: [createSlider('ratio', 'Ratio', 0.05, 1, 0.05, 0.22)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('points');
      const points: Point[] = Array.from({length: 70}, (): Point => {
        const angle = random.range(-1.55, 1.55);
        const radius = 34 + random.range(-9, 9);
        return [58 + Math.cos(angle) * radius * 1.35, 50 + Math.sin(angle) * radius * 1.1];
      });
      const hull = getConvexHull(points);
      const used = new Set(hull);
      let longest = 0;
      for (let index = 0; index < hull.length; index++)
        longest = Math.max(longest, getDistance(hull[index], hull[(index + 1) % hull.length]));
      const limit = mixNumbers(6, longest, p.ratio);
      const boundary = [...hull];
      for (let guard = 0; guard < 120; guard++) {
        let widest = -1;
        let widestLength = limit;
        for (let index = 0; index < boundary.length; index++) {
          const length = getDistance(boundary[index], boundary[(index + 1) % boundary.length]);
          if (length > widestLength) {
            widestLength = length;
            widest = index;
          }
        }
        if (widest < 0) break;
        const from = boundary[widest];
        const to = boundary[(widest + 1) % boundary.length];
        let choice: Point | null = null;
        let choiceScore = Infinity;
        for (const candidate of points) {
          if (used.has(candidate)) continue;
          const toFrom = getDistance(candidate, from);
          const toTo = getDistance(candidate, to);
          if (toFrom >= widestLength || toTo >= widestLength) continue;
          let crosses = false;
          for (let index = 0; index < boundary.length && !crosses; index++) {
            if (index === widest) continue;
            const start = boundary[index];
            const end = boundary[(index + 1) % boundary.length];
            if (start !== from && start !== to && end !== from && end !== to) {
              if (
                intersectSegments(from, candidate, start, end) ||
                intersectSegments(candidate, to, start, end)
              )
                crosses = true;
            }
          }
          if (crosses) continue;
          const score = Math.max(toFrom, toTo);
          if (score < choiceScore) {
            choiceScore = score;
            choice = candidate;
          }
        }
        if (!choice) {
          // This edge cannot be eroded further; stop to keep the result valid.
          break;
        }
        used.add(choice);
        boundary.splice(widest + 1, 0, choice);
      }
      return [
        createStrokes([createLoop(hull, 'dim', 0.8, 0.8)], 'pulse', 0.2),
        createFills([createArea(boundary, 'violet', 0.2)], 'pulse'),
        createStrokes([createLoop(boundary, 'warm', 1.4, 1)], 'flow', 0.22, 1.5),
        createSprites(
          [...toSprites(points, 'primary', 1.1), ...toSprites(boundary, 'warm', 1.3)],
          0.3
        )
      ];
    }
  },

  'vector-straight-skeleton': {
    parameters: [createSlider('offsets', 'Offsets', 2, 12, 1, 6)],
    pointer: 'none',
    build: ({p}) => {
      const outer: Point[] = [
        [12, 10],
        [148, 10],
        [148, 50],
        [96, 50],
        [96, 90],
        [12, 90]
      ];
      const hole: Point[] = [
        [30, 28],
        [66, 28],
        [66, 70],
        [30, 70]
      ];
      const rings = [outer, hole];
      const segments: [Point, Point][] = [];
      for (const ring of rings) {
        for (let index = 0; index < ring.length; index++)
          segments.push([ring[index], ring[(index + 1) % ring.length]]);
      }
      const columns = 80;
      const rowsCount = 50;
      const result = createSkeletonField(
        columns,
        rowsCount,
        (x, y) => {
          if (!isPointInRings([x, y], rings)) return {depth: 0, id: -1};
          let best = Infinity;
          let id = -1;
          segments.forEach(([start, end], index) => {
            const dx = Math.max(Math.min(start[0], end[0]) - x, 0, x - Math.max(start[0], end[0]));
            const dy = Math.max(Math.min(start[1], end[1]) - y, 0, y - Math.max(start[1], end[1]));
            const distance = Math.max(dx, dy);
            if (distance < best) {
              best = distance;
              id = index;
            }
          });
          return {depth: best, id};
        },
        0
      );
      const field: FieldLayer = {
        type: 'field',
        columns,
        rows: rowsCount,
        values: result.values,
        ramp: 'ice',
        style: 'contours',
        levels: Math.round(p.offsets),
        speed: 0.2
      };
      return [
        field,
        createStrokes(
          [createLoop(outer, 'primary', 1.2, 0.95), createLoop(hole, 'primary', 1.2, 0.95)],
          'flow',
          0.2,
          1.4
        ),
        createSprites(
          result.ridge.map((point, index) =>
            createGlowDot(point, 'warm', 0.9, 0.85, (index % 40) / 40)
          ),
          0.35
        )
      ];
    }
  },

  'convex-straight-skeleton': {
    parameters: [createSlider('inset', 'Inset', 5, 95, 5, 45, '%')],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('lot');
      const ring: Point[] = Array.from({length: 6}, (_, index): Point => {
        const angle = (index / 6) * Math.PI * 2 + random.range(-0.2, 0.2);
        return [80 + Math.cos(angle) * 58, 50 + Math.sin(angle) * 36];
      });
      const sign = getSignedArea(ring) > 0 ? 1 : -1;
      const planes = ring.map((start, index) => {
        const end = ring[(index + 1) % ring.length];
        const length = getDistance(start, end);
        return {
          start,
          nx: (-(end[1] - start[1]) / length) * sign,
          ny: ((end[0] - start[0]) / length) * sign
        };
      });
      const columns = 80;
      const rowsCount = 50;
      const result = createSkeletonField(
        columns,
        rowsCount,
        (x, y) => {
          let best = Infinity;
          let id = -1;
          planes.forEach((plane, index) => {
            const depth = (x - plane.start[0]) * plane.nx + (y - plane.start[1]) * plane.ny;
            if (depth < best) {
              best = depth;
              id = index;
            }
          });
          return best > 0 ? {depth: best, id} : {depth: 0, id: -1};
        },
        0
      );
      const distance = (p.inset / 100) * result.maxDepth;
      let inset = ring;
      for (const plane of planes) {
        if (inset.length < 3) break;
        inset = clipRingBelowLimit(
          inset,
          -plane.nx,
          -plane.ny,
          -(plane.nx * plane.start[0] + plane.ny * plane.start[1]) - distance
        );
      }
      const field: FieldLayer = {
        type: 'field',
        columns,
        rows: rowsCount,
        values: result.values,
        ramp: 'ice',
        style: 'expand',
        levels: 3,
        cutoff: p.inset / 100,
        speed: 0.15
      };
      return [
        field,
        createStrokes([createLoop(ring, 'primary', 1.2, 0.95)], 'flow', 0.2, 1.4),
        createSprites(
          result.ridge.map((point, index) =>
            createGlowDot(point, 'warm', 0.9, 0.85, (index % 40) / 40)
          ),
          0.35
        ),
        createStrokes(inset.length > 2 ? [createLoop(inset, 'hot', 1.2, 1)] : [], 'pulse', 0.4)
      ];
    }
  },

  // Geodesy and projection -----------------------------------------------------------------

  'geodesic-pairs': {
    parameters: [createSlider('pairs', 'Pairs', 2, 8, 1, 5)],
    pointer: 'hover',
    build: ({p, pointer}) => {
      const centerLongitude = pointer ? (pointer[0] / SCENE_WIDTH) * 240 - 100 : 20;
      const centerLatitude = 25;
      const cities: [number, number][] = [
        [-74, 40],
        [-0.1, 51.5],
        [139.7, 35.7],
        [151.2, -33.9],
        [-46.6, -23.5],
        [77.2, 28.6],
        [18.4, -33.9],
        [-122.4, 37.8],
        [37.6, 55.7],
        [103.8, 1.35]
      ];
      const arcs: PathItem[] = [];
      const origins: SpriteItem[] = [];
      const targets: SpriteItem[] = [];
      const midpoints: SpriteItem[] = [];
      for (let index = 0; index < Math.round(p.pairs); index++) {
        const from = cities[index];
        const to = cities[(index + 3 + (index % 3)) % cities.length];
        for (const piece of projectGlobePath(
          getGreatCircle(from, to),
          centerLongitude,
          centerLatitude
        )) {
          arcs.push(createLine(piece, 'cool', 1.1, 0.95, index / 8));
        }
        const middle = toLonLat(
          slerpVectors(toVector(from[0], from[1]), toVector(to[0], to[1]), 0.5)
        );
        const projections = [from, to, middle].map(point =>
          projectGlobePoint(point[0], point[1], centerLongitude, centerLatitude)
        );
        if (projections[0].visible) origins.push(createGlowDot(projections[0].point, 'cool', 2));
        if (projections[1].visible) targets.push(createGlowDot(projections[1].point, 'warm', 2));
        if (projections[2].visible)
          midpoints.push(createGlowDot(projections[2].point, 'hot', 1.6, 0.9, index / 8));
      }
      return [
        ...createGlobeBase(centerLongitude, centerLatitude),
        createStrokes(arcs, 'flow', 0.22, 1.2),
        createSprites([...origins, ...targets, ...midpoints], 0.4)
      ];
    }
  },

  'geodesic-destination': {
    parameters: [
      createSlider('distance', 'Distance', 10, 70, 1, 35, '°'),
      createSlider('bearings', 'Bearings', 4, 24, 1, 12)
    ],
    pointer: 'hover',
    build: ({p, pointer}) => {
      const origin: [number, number] = pointer
        ? [-50 + (pointer[0] / SCENE_WIDTH) * 130, 60 - (pointer[1] / SCENE_HEIGHT) * 60]
        : [10, 25];
      const centerLongitude = origin[0];
      const centerLatitude = 30;
      const rays: PathItem[] = [];
      const ends: SpriteItem[] = [];
      const count = Math.round(p.bearings);
      for (let index = 0; index < count; index++) {
        const bearing = (index / count) * 360;
        const end = getDestination(origin, bearing, p.distance);
        for (const piece of projectGlobePath(
          getGreatCircle(origin, end, 16),
          centerLongitude,
          centerLatitude
        )) {
          rays.push(createLine(piece, index === 0 ? 'hot' : 'warm', 0.9, 0.9, index / count));
        }
        const projected = projectGlobePoint(end[0], end[1], centerLongitude, centerLatitude);
        if (projected.visible)
          ends.push(
            createGlowDot(projected.point, index === 0 ? 'hot' : 'warm', 1.8, 1, index / count)
          );
      }
      const ring: [number, number][] = [];
      const sector: [number, number][] = [origin];
      for (let bearing = 0; bearing <= 360; bearing += 6) {
        const destination = getDestination(origin, bearing, p.distance);
        ring.push(destination);
        if (bearing >= 40 && bearing <= 120) sector.push(destination);
      }
      sector.push(origin);
      const sectorPoints = sector
        .map(([longitude, latitude]) =>
          projectGlobePoint(longitude, latitude, centerLongitude, centerLatitude)
        )
        .filter(item => item.visible)
        .map(item => item.point);
      const origin2d = projectGlobePoint(
        origin[0],
        origin[1],
        centerLongitude,
        centerLatitude
      ).point;
      return [
        ...createGlobeBase(centerLongitude, centerLatitude),
        createFills(
          sectorPoints.length > 2 ? [createArea(sectorPoints, 'primary', 0.2)] : [],
          'pulse'
        ),
        createStrokes(
          projectGlobePath(ring, centerLongitude, centerLatitude).map(piece =>
            createLine(piece, 'cool', 1, 0.9)
          ),
          'pulse',
          0.3
        ),
        createStrokes(rays, 'flow', 0.3, 1),
        createSprites([createGlowDot(origin2d, 'hot', 3), ...ends], 0.4)
      ];
    }
  },

  'local-frame-projection': {
    parameters: [createSlider('count', 'Points', 8, 60, 2, 28)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('points');
      const count = Math.round(p.count);
      const unitPoints = Array.from(
        {length: count},
        (): Point => [random.range(0.1, 0.9), random.range(0.1, 0.9)]
      );
      // Degrees: meridians converge toward the top, so cells are trapezoids.
      const toDegrees = ([u, v]: Point): Point => [
        36 + (u - 0.5) * 56 * (0.78 + 0.22 * v) + (v - 0.5) * 0 + 0 * u,
        14 + v * 72
      ];
      const toMeters = ([u, v]: Point): Point => [124 + (u - 0.5) * 56, 14 + v * 72];
      const meridians: PathItem[] = [];
      for (let step = 0; step <= 4; step++) {
        const u = step / 4;
        meridians.push(createLine([toDegrees([u, 0]), toDegrees([u, 1])], 'dim', 0.4, 0.6));
        meridians.push(createLine([toDegrees([0, u]), toDegrees([1, u])], 'dim', 0.4, 0.6));
        meridians.push(createLine([toMeters([u, 0]), toMeters([u, 1])], 'cool', 0.4, 0.5));
        meridians.push(createLine([toMeters([0, u]), toMeters([1, u])], 'cool', 0.4, 0.5));
      }
      const flows = unitPoints.map((point, index) =>
        createLine([toDegrees(point), [80, 50], toMeters(point)], 'warm', 0.3, 0.22, index / count)
      );
      return [
        createStrokes(meridians),
        createStrokes(
          flows.map(flow => ({...flow, points: [flow.points[0], flow.points[2]]})),
          'flow',
          0.35,
          0.8
        ),
        createSprites(
          [
            ...unitPoints.map((point, index) =>
              createGlowDot(toDegrees(point), 'dim', 1.3, 0.9, index / count)
            ),
            ...unitPoints.map((point, index) =>
              createGlowDot(toMeters(point), 'warm', 1.3, 0.95, index / count)
            ),
            createGlowDot(toDegrees([0.5, 0.5]), 'hot', 2.6),
            createGlowDot(toMeters([0.5, 0.5]), 'hot', 2.6)
          ],
          0.35
        )
      ];
    }
  },

  'lon-lat-polygon-preparation': {
    parameters: [createSlider('spacing', 'Densify step', 4, 14, 1, 7, '°')],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const left = 14;
      const width = 132;
      const toX = (longitude: number) => left + ((longitude + 180) / 360) * width;
      const toY = (latitude: number) => 50 - latitude * 0.68;
      const centerLongitude = pointer
        ? clamp(((pointer[0] - left) / width) * 360 - 180, -180, 180)
        : 172;
      const unwrapped = createBlob(fork('shape'), centerLongitude, 6, 32, 14, 0.28, 0.95);
      const wrap = (longitude: number) =>
        longitude > 180 ? longitude - 360 : longitude < -180 ? longitude + 360 : longitude;
      const naive = unwrapped.map(
        ([longitude, latitude]): Point => [toX(wrap(longitude)), toY(latitude)]
      );
      let pieces: Point[][] = [unwrapped];
      const [minimumX, , maximumX] = getRingBounds(unwrapped);
      if (maximumX > 180) {
        pieces = [
          clipRingBelowLimit(unwrapped, 1, 0, 180),
          clipRingBelowLimit(unwrapped, -1, 0, -180).map(([x, y]): Point => [x - 360, y])
        ];
      } else if (minimumX < -180) {
        pieces = [
          clipRingBelowLimit(unwrapped, -1, 0, 180),
          clipRingBelowLimit(unwrapped, 1, 0, -180).map(([x, y]): Point => [x + 360, y])
        ];
      }
      pieces = pieces.filter(piece => piece.length > 2);
      const split = pieces.length > 1;
      const screenPieces = pieces.map(piece => piece.map(([x, y]): Point => [toX(x), toY(y)]));
      const densified: SpriteItem[] = [];
      for (const piece of pieces) {
        for (let index = 0; index < piece.length; index++) {
          const start = piece[index];
          const end = piece[(index + 1) % piece.length];
          const length = getDistance(start, end);
          const inserts = Math.floor(length / p.spacing);
          for (let step = 1; step <= inserts; step++) {
            const amount = step / (inserts + 1);
            densified.push(
              createGlowDot(
                [
                  toX(mixNumbers(start[0], end[0], amount)),
                  toY(mixNumbers(start[1], end[1], amount))
                ],
                'primary',
                0.7,
                0.7,
                amount
              )
            );
          }
        }
      }
      const frame: Point[] = [
        [left, 16],
        [left + width, 16],
        [left + width, 84],
        [left, 84]
      ];
      const meridians: PathItem[] = [];
      for (let longitude = -120; longitude <= 120; longitude += 60)
        meridians.push(
          createLine(
            [
              [toX(longitude), 16],
              [toX(longitude), 84]
            ],
            'dim',
            0.25,
            0.25
          )
        );
      for (let latitude = -30; latitude <= 30; latitude += 30)
        meridians.push(
          createLine(
            [
              [left, toY(latitude)],
              [left + width, toY(latitude)]
            ],
            'dim',
            0.25,
            0.25
          )
        );
      return [
        createStrokes(meridians),
        createFills([createArea(naive, 'hot', 0.07)]),
        createStrokes([createLoop(naive, 'hot', 0.6, 0.35)]),
        createStrokes([createLoop(frame, 'cool', 0.6, 0.6)]),
        createStrokes(
          split
            ? [
                createLine(
                  [
                    [left, 16],
                    [left, 84]
                  ],
                  'hot',
                  1.6,
                  0.95
                ),
                createLine(
                  [
                    [left + width, 16],
                    [left + width, 84]
                  ],
                  'hot',
                  1.6,
                  0.95
                )
              ]
            : [],
          'pulse',
          0.5
        ),
        createFills(
          screenPieces.map((piece, index) =>
            createArea(piece, index === 0 ? 'primary' : 'warm', 0.3, index * 0.5)
          ),
          'pulse'
        ),
        createStrokes(
          screenPieces.map((piece, index) =>
            createLoop(piece, index === 0 ? 'primary' : 'warm', 1.3, 1)
          ),
          'flow',
          0.2,
          1.3
        ),
        createSprites(densified, 0.3)
      ];
    }
  },

  'study-window-preparation': {
    parameters: [createSlider('cells', 'Mask cells', 12, 64, 4, 32)],
    pointer: 'none',
    build: ({fork, p}) => {
      const ring = createBlob(fork('window'), 80, 50, 32, 18, 0.32, 0.85);
      const sign = getSignedArea(ring) > 0 ? 1 : -1;
      const columns = Math.round(p.cells);
      const rows = Math.max(4, Math.round((columns * SCENE_HEIGHT) / SCENE_WIDTH));
      const values = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          if (
            isInsideRing(
              [((column + 0.5) / columns) * SCENE_WIDTH, ((row + 0.5) / rows) * SCENE_HEIGHT],
              ring
            )
          ) {
            values[row * columns + column] = 0.62;
          }
        }
      }
      const field: FieldLayer = {
        type: 'field',
        columns,
        rows,
        values,
        ramp: 'ice',
        style: 'ramp',
        filter: 'nearest',
        strength: 0.15,
        speed: 0.3
      };
      // Interior-on-the-left markers just inside every edge midpoint.
      const markers: SpriteItem[] = ring.map((start, index) => {
        const end = ring[(index + 1) % ring.length];
        const length = getDistance(start, end) || 1;
        return createGlowDot(
          [
            (start[0] + end[0]) / 2 - ((end[1] - start[1]) / length) * 3 * sign,
            (start[1] + end[1]) / 2 + ((end[0] - start[0]) / length) * 3 * sign
          ],
          'warm',
          0.9,
          0.9,
          index / ring.length
        );
      });
      const bounds = createRectangleRing(...getRingBounds(ring));
      return [
        field,
        createStrokes([createLoop(bounds, 'dim', 0.5, 0.7)], 'pulse', 0.2),
        createStrokes([createLoop(ring, 'primary', 1.4, 1)], 'flow', 0.25, 1.4),
        createSprites(
          [...markers, ...bounds.map(corner => createGlowDot(corner, 'cool', 1.2, 0.8))],
          0.4
        )
      ];
    }
  }
};
