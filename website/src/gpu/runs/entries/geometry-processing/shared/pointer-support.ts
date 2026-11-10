// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers for the pointer gestures of the geometry-processing runs.

import type {Point, RunGlyph, RunPointer} from '../../../interaction';
import {findNearestSegment, findPolygonAt} from '../../../pick';
import {
  findNearestVertex,
  getBaseVertex,
  type PolygonSet,
  type VertexDrag,
  type VertexHandle
} from './polygon-set';

/**
 * The point in the world whose distance (as given by `getDistance`) is closest to `target`, nearest
 * to the picture center on ties. Used to place the `rest` pointer of a reach gesture.
 */
export function findPointAtDistance(
  getDistance: (x: number, y: number) => number | null,
  target: number,
  world: {width: number; height: number}
): Point {
  let best: Point = [world.width / 2, world.height / 2];
  let bestScore = Infinity;
  const columns = 64;
  const rows = 36;
  for (let row = 0; row <= rows; row++) {
    for (let column = 0; column <= columns; column++) {
      const x = ((column + 0.5) / (columns + 1)) * world.width;
      const y = ((row + 0.5) / (rows + 1)) * world.height;
      const distance = getDistance(x, y);
      if (distance === null) continue;
      const score =
        Math.abs(distance - target) * 10 + Math.hypot(x - world.width / 2, y - world.height / 2);
      if (score < bestScore) {
        bestScore = score;
        best = [x, y];
      }
    }
  }
  return best;
}

/** The index of the bin (of `count` equal bins over 0 to `span`) that `value` falls in. */
export function getBinIndex(value: number, span: number, count: number): number {
  return Math.min(count - 1, Math.max(0, Math.floor((value / span) * count)));
}

/** The center of bin `index`, as a value between 0 and `span`. */
export function getBinCenter(index: number, span: number, count: number): number {
  return ((index + 0.5) / count) * span;
}

/**
 * What the precision lattice does to a point: the lattice cell it lies in, the node it rounds to,
 * and a link between the two.
 */
export function getLatticeGlyphs(point: Point, gridSize: number): RunGlyph[] {
  const lowX = Math.floor(point[0] / gridSize) * gridSize;
  const lowY = Math.floor(point[1] / gridSize) * gridSize;
  const node: Point = [
    Math.round(point[0] / gridSize) * gridSize,
    Math.round(point[1] / gridSize) * gridSize
  ];
  return [
    {
      shape: 'box',
      minimum: [lowX, lowY],
      maximum: [lowX + gridSize, lowY + gridSize],
      dashed: true
    },
    {shape: 'line', from: point, to: node},
    {shape: 'dot', at: node}
  ];
}

/** The pointer-owned vertex state of a drag run: the set as drawn, the carried vertex and a hero. */
export type VertexDragState = {
  set: PolygonSet;
  drag: VertexDrag | null;
  /** The vertex the poster rests on: it starts where the pointer is at `rest`. */
  hero: VertexHandle;
};

/** A hero vertex of the middle feature, so the poster's handle sits near the picture center. */
export function chooseHeroVertex(
  set: PolygonSet,
  feature = Math.floor(set.count / 2)
): VertexHandle {
  return {feature, vertex: 0};
}

export function getHeroRest(state: VertexDragState): Point {
  return getBaseVertex(state.set, state.hero);
}

/**
 * Carries the shell vertex nearest to the pointer to the pointer (with `spread` neighbors
 * following) and returns the handle glyphs. The caller returns `rerun: true`.
 */
export function carryNearestVertex(
  state: VertexDragState,
  pointer: RunPointer,
  spread = 0,
  hold = 0
): {glyphs: RunGlyph[]; base: Point; target: Point} {
  const handle = findNearestVertex(state.set, pointer.x, pointer.y, state.drag, hold) ?? state.hero;
  const target: Point = [pointer.x, pointer.y];
  const base = getBaseVertex(state.set, handle);
  state.drag = {...handle, target, spread};
  return {
    base,
    target,
    glyphs: [
      {shape: 'line', from: base, to: target, dashed: true},
      {shape: 'ring', at: base, radius: 0.012, dashed: true},
      {shape: 'crosshair', at: target}
    ]
  };
}

/** The points where a closed ring of (x, y) vertices crosses itself. */
export function findRingCrossings(
  positions: ArrayLike<number>,
  first: number,
  count: number
): Point[] {
  const crossings: Point[] = [];
  const read = (index: number): Point => {
    const row = first + (index % count);
    return [positions[row * 2], positions[row * 2 + 1]];
  };
  for (let edge = 0; edge < count; edge++) {
    for (let other = edge + 2; other < count; other++) {
      if (edge === 0 && other === count - 1) continue;
      const [ax, ay] = read(edge);
      const [bx, by] = read(edge + 1);
      const [cx, cy] = read(other);
      const [dx, dy] = read(other + 1);
      const denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
      if (denominator === 0) continue;
      const along = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator;
      const across = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denominator;
      if (along > 0 && along < 1 && across > 0 && across < 1) {
        crossings.push([ax + (bx - ax) * along, ay + (by - ay) * along]);
      }
    }
  }
  return crossings;
}

/** The pointer-owned fields every drag run adds to its inputs. */
export function createDragFields(
  set: PolygonSet,
  feature?: number
): {drag: VertexDrag | null; hero: VertexHandle} {
  return {drag: null, hero: chooseHeroVertex(set, feature)};
}

/**
 * The distance from (x, y) to the nearest shell point (the depth when the point is inside a
 * polygon), that point, and whether (x, y) lies inside a polygon. Outside, only the polygon under
 * the pointer is not searched: every polygon is.
 */
export function getPolygonDepth(
  set: PolygonSet,
  x: number,
  y: number
): {feature: number; depth: number; point: Point; isInside: boolean} | null {
  const inside = findPolygonAt(
    set.fillVertices,
    set.fillIndices,
    set.fillOwners,
    x,
    y,
    set.fillTriangleCount
  );
  let best: {feature: number; distance: number; point: Point} | null = null;
  for (let feature = 0; feature < set.count; feature++) {
    const ring = set.polygonOffsets[set.featureOffsets[feature]];
    const first = set.outlineOffsets[ring];
    const hit = findNearestSegment(
      set.outline.subarray(first * 4),
      x,
      y,
      set.outlineOffsets[ring + 1] - first
    );
    if (hit && (!best || hit.distance < best.distance)) {
      best = {feature, distance: hit.distance, point: hit.point};
    }
  }
  if (!best) return null;
  return {
    feature: best.feature,
    depth: best.distance,
    point: best.point,
    isInside: inside >= 0
  };
}

/** The value of `values` closest to `target`. */
export function getNearestValue(values: readonly number[], target: number): number {
  return values.reduce((best, value) =>
    Math.abs(value - target) < Math.abs(best - target) ? value : best
  );
}

/**
 * Remembers which CPU copy of a buffer belongs to an earlier build. After a `settle` rebuild the
 * peeked copy stays the old build's until the new readback arrives; `readFresh` hides it.
 */
export type StaleGuard = {key: string | null; stale: unknown};

export function createStaleGuard(): StaleGuard {
  return {key: null, stale: null};
}

/** `array` unless it is the copy that was current when `key` (the compile-time values) changed. */
export function readFresh<Copy>(guard: StaleGuard, key: string, array: Copy | null): Copy | null {
  if (guard.key === null) guard.key = key;
  else if (guard.key !== key) {
    guard.key = key;
    guard.stale = array;
  }
  return array !== null && array === guard.stale ? null : array;
}
