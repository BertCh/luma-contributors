// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Ear-clipping triangulation of a polygon with holes, for the filled pictures of the
// geometry-processing runs. It always returns `points + 2 * holes - 2` triangles, so a polygon
// keeps the same triangle count while one of its vertices is dragged.

export type Point2 = readonly [number, number];

const EPSILON = 1e-12;

function getSignedArea(ids: readonly number[], points: readonly Point2[]): number {
  let area = 0;
  for (let index = 0; index < ids.length; index++) {
    const [x0, y0] = points[ids[index]];
    const [x1, y1] = points[ids[(index + 1) % ids.length]];
    area += x0 * y1 - x1 * y0;
  }
  return area / 2;
}

/** Orders `ids` counter-clockwise (`direction` 1) or clockwise (-1). */
function orient(ids: number[], points: readonly Point2[], direction: 1 | -1): number[] {
  return getSignedArea(ids, points) * direction < 0 ? [...ids].reverse() : ids;
}

function getCross(a: Point2, b: Point2, c: Point2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function isSamePoint(a: Point2, b: Point2): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

function isCrossing(a: Point2, b: Point2, c: Point2, d: Point2): boolean {
  const first = getCross(a, b, c);
  const second = getCross(a, b, d);
  const third = getCross(c, d, a);
  const fourth = getCross(c, d, b);
  return first * second < 0 && third * fourth < 0;
}

/** Joins a hole (clockwise ids) to the outer ring (counter-clockwise ids) by a visible bridge. */
function bridgeHole(ring: number[], hole: number[], points: readonly Point2[]): number[] {
  let rightmost = 0;
  for (let index = 1; index < hole.length; index++) {
    if (points[hole[index]][0] > points[hole[rightmost]][0]) rightmost = index;
  }
  const holePoint = points[hole[rightmost]];
  const candidates = ring
    .map((id, position) => ({
      position,
      distance: Math.hypot(points[id][0] - holePoint[0], points[id][1] - holePoint[1])
    }))
    .sort((first, second) => first.distance - second.distance);
  const isVisible = (position: number) => {
    const target = points[ring[position]];
    const blocks = (ids: readonly number[]) => {
      for (let index = 0; index < ids.length; index++) {
        const c = points[ids[index]];
        const d = points[ids[(index + 1) % ids.length]];
        if (isSamePoint(c, target) || isSamePoint(d, target)) continue;
        if (isSamePoint(c, holePoint) || isSamePoint(d, holePoint)) continue;
        if (isCrossing(holePoint, target, c, d)) return true;
      }
      return false;
    };
    return !blocks(ring) && !blocks(hole);
  };
  const chosen = (candidates.find(candidate => isVisible(candidate.position)) ?? candidates[0])
    .position;
  const rotated = hole.map((_, offset) => hole[(rightmost + offset) % hole.length]);
  return [
    ...ring.slice(0, chosen + 1),
    ...rotated,
    hole[rightmost],
    ring[chosen],
    ...ring.slice(chosen + 1)
  ];
}

function isInsideTriangle(point: Point2, a: Point2, b: Point2, c: Point2): boolean {
  return (
    getCross(a, b, point) >= -EPSILON &&
    getCross(b, c, point) >= -EPSILON &&
    getCross(c, a, point) >= -EPSILON
  );
}

/**
 * Triangle vertex ids into `[...shell, ...holes.flat()]`. The shell and holes may wind either way.
 */
export function triangulatePolygon(shell: readonly Point2[], holes: readonly Point2[][]): number[] {
  const points: Point2[] = [...shell];
  const holeIds: number[][] = [];
  for (const hole of holes) {
    holeIds.push(hole.map((_, index) => points.length + index));
    points.push(...hole);
  }
  let ring = orient(
    shell.map((_, index) => index),
    points,
    1
  );
  for (const ids of holeIds) ring = bridgeHole(ring, orient(ids, points, -1), points);
  const triangles: number[] = [];
  const list = [...ring];
  while (list.length > 3) {
    const length = list.length;
    let earIndex = -1;
    let bestCross = -Infinity;
    let bestIndex = 0;
    for (let index = 0; index < length && earIndex < 0; index++) {
      const previous = list[(index + length - 1) % length];
      const current = list[index];
      const next = list[(index + 1) % length];
      const a = points[previous];
      const b = points[current];
      const c = points[next];
      const cross = getCross(a, b, c);
      if (cross > bestCross) {
        bestCross = cross;
        bestIndex = index;
      }
      if (cross < 0) continue;
      let isEar = true;
      for (const other of list) {
        if (other === previous || other === current || other === next) continue;
        const point = points[other];
        if (isSamePoint(point, a) || isSamePoint(point, b) || isSamePoint(point, c)) continue;
        if (isInsideTriangle(point, a, b, c)) {
          isEar = false;
          break;
        }
      }
      if (isEar) earIndex = index;
    }
    // A self-intersecting ring has no ear: clip the least reflex corner so the count stays fixed.
    if (earIndex < 0) earIndex = bestIndex;
    triangles.push(
      list[(earIndex + length - 1) % length],
      list[earIndex],
      list[(earIndex + 1) % length]
    );
    list.splice(earIndex, 1);
  }
  triangles.push(list[0], list[1], list[2]);
  return triangles;
}
