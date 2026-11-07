// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Planar geometry for preview builders: rings, hulls, segment intersection and clipping. Rings are
// open (the first vertex is not repeated) unless a function says otherwise.

import {clamp} from './data.ts';
import type {Point} from './types.ts';

/** Bounds as [minimum x, minimum y, maximum x, maximum y]. */
export type Bounds = [number, number, number, number];

/** Repeats the first vertex at the end, for drawing a ring as a closed path. */
export function closeRing(ring: Point[]): Point[] {
  return [...ring, ring[0]];
}

/** Axis-aligned rectangle ring from its corners, clockwise in scene space (y down). */
export function createRectangleRing(
  left: number,
  top: number,
  right: number,
  bottom: number
): Point[] {
  return [
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom]
  ];
}

/** z of the cross product of (first - origin) and (second - origin). */
export function getCross(origin: Point, first: Point, second: Point): number {
  return (
    (first[0] - origin[0]) * (second[1] - origin[1]) -
    (first[1] - origin[1]) * (second[0] - origin[0])
  );
}

/** Shoelace area; positive for counterclockwise rings in y-up coordinates. */
export function getSignedArea(ring: Point[]): number {
  let sum = 0;
  for (let index = 0; index < ring.length; index++) {
    const [ax, ay] = ring[index];
    const [bx, by] = ring[(index + 1) % ring.length];
    sum += ax * by - bx * ay;
  }
  return sum / 2;
}

/** Even-odd point in polygon test. */
export function isInsideRing(point: Point, ring: Point[]): boolean {
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

export function getRingBounds(ring: Point[]): Bounds {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const [x, y] of ring) {
    minimumX = Math.min(minimumX, x);
    minimumY = Math.min(minimumY, y);
    maximumX = Math.max(maximumX, x);
    maximumY = Math.max(maximumY, y);
  }
  return [minimumX, minimumY, maximumX, maximumY];
}

/** Mean of the vertices (not the area centroid). */
export function getVertexMean(points: Point[]): Point {
  let x = 0;
  let y = 0;
  for (const point of points) {
    x += point[0];
    y += point[1];
  }
  return [x / points.length, y / points.length];
}

/**
 * Convex hull by Andrew's monotone chain. Collinear points are dropped; inputs of fewer than three
 * points come back sorted by x, then y.
 */
export function getConvexHull(points: Point[]): Point[] {
  const sorted = [...points].sort((first, second) => first[0] - second[0] || first[1] - second[1]);
  if (sorted.length < 3) return sorted;
  const buildChain = (input: Point[]): Point[] => {
    const chain: Point[] = [];
    for (const point of input) {
      while (
        chain.length >= 2 &&
        getCross(chain[chain.length - 2], chain[chain.length - 1], point) <= 0
      ) {
        chain.pop();
      }
      chain.push(point);
    }
    chain.pop();
    return chain;
  };
  return [...buildChain(sorted), ...buildChain([...sorted].reverse())];
}

/** Intersection point of segments a-b and c-d, or null when they do not cross. */
export function intersectSegments(a: Point, b: Point, c: Point, d: Point): Point | null {
  const denominator = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0]);
  if (Math.abs(denominator) < 1e-9) return null;
  const first = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / denominator;
  const second = ((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / denominator;
  if (first < 0 || first > 1 || second < 0 || second > 1) return null;
  return [a[0] + (b[0] - a[0]) * first, a[1] + (b[1] - a[1]) * first];
}

/** Closest point to `point` on segment start-end, and the parameter along it in 0..1. */
export function getNearestOnSegment(
  point: Point,
  start: Point,
  end: Point
): {point: Point; amount: number; distance: number} {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy || 1;
  const amount = clamp(
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared,
    0,
    1
  );
  const nearest: Point = [start[0] + dx * amount, start[1] + dy * amount];
  return {
    point: nearest,
    amount,
    distance: Math.hypot(point[0] - nearest[0], point[1] - nearest[1])
  };
}

/**
 * Sutherland-Hodgman clip of a ring against the half plane where
 * (point - origin) . normal >= 0.
 */
export function clipRingByHalfPlane(ring: Point[], origin: Point, normal: Point): Point[] {
  const getSide = (point: Point) =>
    (point[0] - origin[0]) * normal[0] + (point[1] - origin[1]) * normal[1];
  const result: Point[] = [];
  for (let index = 0; index < ring.length; index++) {
    const current = ring[index];
    const next = ring[(index + 1) % ring.length];
    const currentSide = getSide(current);
    const nextSide = getSide(next);
    if (currentSide >= 0) result.push(current);
    if (currentSide >= 0 !== nextSide >= 0) {
      const amount = currentSide / (currentSide - nextSide);
      result.push([
        current[0] + (next[0] - current[0]) * amount,
        current[1] + (next[1] - current[1]) * amount
      ]);
    }
  }
  return result;
}
