// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded polygons with holes for the randomized search of `GPUPolygonTriangulation`, and an exact
// checker of a triangulation that does not use the kernel's formulas.
//
// Every coordinate is a float32 value. The checker decides everything on the exact binary
// expansions (a float64 filter, then BigInt): the triangles must have non-zero area, lie inside
// the polygon, not overlap pairwise, their areas must sum to the polygon area, and their directed
// edges must cancel to the polygon boundary. The last test alone decides whether the triangles
// tile the polygon (see `getBoundaryChainDefect`); the others give a more readable reason.

import {
  createRandom,
  createRandomStarPolygon,
  createTwoOptPolygon
} from './polygon-triangulation-monotone-cases';

export type SearchPoint = readonly [number, number];

export type SearchPolygon = {
  /** Generator family and parameters, for failure messages. */
  name: string;
  /** Shell first, then holes; any winding; float32 coordinates. */
  rings: SearchPoint[][];
};

/** An affine map of lattice points onto float32 coordinates, exact by construction. */
export type SearchFrame = {
  name: string;
  origin: SearchPoint;
  /** Lattice spacing per axis, a power of two. */
  spacing: SearchPoint;
};

/** Small integers: float32 arithmetic on them is exact apart from divisions. */
export const LATTICE_FRAME: SearchFrame = {name: 'lattice', origin: [0, 0], spacing: [1, 1]};

/**
 * Projected-metre magnitude (a UTM easting and northing) with the lattice at float32 resolution:
 * one lattice step is one unit in the last place of `x` (1/32 m) and of `y` (1/2 m).
 */
export const METRE_RESOLUTION_FRAME: SearchFrame = {
  name: 'metre-resolution',
  origin: [500000, 4500000],
  spacing: [1 / 32, 1 / 2]
};

/** Projected-metre magnitude with a 2 m lattice. */
export const METRE_FRAME: SearchFrame = {
  name: 'metre',
  origin: [500000, 4500000],
  spacing: [2, 2]
};

export function mapToFrame(frame: SearchFrame, rings: readonly SearchPoint[][]): SearchPoint[][] {
  return rings.map(ring =>
    ring.map(([x, y]): SearchPoint => {
      const mapped: SearchPoint = [
        frame.origin[0] + x * frame.spacing[0],
        frame.origin[1] + y * frame.spacing[1]
      ];
      if (Math.fround(mapped[0]) !== mapped[0] || Math.fround(mapped[1]) !== mapped[1]) {
        throw new Error(`frame ${frame.name} does not hold lattice point ${x}, ${y} exactly`);
      }
      return mapped;
    })
  );
}

// Exact arithmetic on float32 values.

const floatView = new DataView(new ArrayBuffer(4));

/** `value * 2^149` as an integer; `value` must be a float32. */
function getExactInteger(value: number): bigint {
  floatView.setFloat32(0, value);
  const bits = floatView.getUint32(0);
  const exponentBits = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  const magnitude =
    exponentBits === 0 ? BigInt(fraction) : BigInt(fraction | 0x800000) << BigInt(exponentBits - 1);
  return bits >>> 31 === 1 ? -magnitude : magnitude;
}

type ExactPoints = {
  /** Float64 copies, for the filter. */
  x: number[];
  y: number[];
  /** Exact integers on one common scale. */
  exactX: bigint[];
  exactY: bigint[];
};

function getExactPoints(points: readonly SearchPoint[]): ExactPoints {
  return {
    x: points.map(point => point[0]),
    y: points.map(point => point[1]),
    exactX: points.map(point => getExactInteger(point[0])),
    exactY: points.map(point => getExactInteger(point[1]))
  };
}

function getExactCross(points: ExactPoints, a: number, b: number, c: number): bigint {
  return (
    (points.exactX[b] - points.exactX[a]) * (points.exactY[c] - points.exactY[a]) -
    (points.exactY[b] - points.exactY[a]) * (points.exactX[c] - points.exactX[a])
  );
}

/** Exact sign of `cross(b - a, c - a)` by row. */
function getOrientation(points: ExactPoints, a: number, b: number, c: number): number {
  const left = (points.x[b] - points.x[a]) * (points.y[c] - points.y[a]);
  const right = (points.y[b] - points.y[a]) * (points.x[c] - points.x[a]);
  const determinant = left - right;
  const bound = 3.4e-16 * (Math.abs(left) + Math.abs(right));
  if (determinant > bound) {
    return 1;
  }
  if (-determinant > bound) {
    return -1;
  }
  const exact = getExactCross(points, a, b, c);
  return exact > 0n ? 1 : exact < 0n ? -1 : 0;
}

/** Exact sign of `cross(b - a, c - a)` for float32 points. */
export function getPointOrientation(a: SearchPoint, b: SearchPoint, c: SearchPoint): number {
  return getOrientation(getExactPoints([a, b, c]), 0, 1, 2);
}

function getRingRows(rings: readonly SearchPoint[][]): number[][] {
  const rows: number[][] = [];
  let row = 0;
  for (const ring of rings) {
    rows.push(ring.map(() => row++));
  }
  return rows;
}

function getExactRingArea(points: ExactPoints, rows: readonly number[]): bigint {
  let twiceArea = 0n;
  for (let index = 0; index < rows.length; index++) {
    const from = rows[index];
    const to = rows[(index + 1) % rows.length];
    twiceArea += points.exactX[from] * points.exactY[to] - points.exactX[to] * points.exactY[from];
  }
  return twiceArea < 0n ? -twiceArea : twiceArea;
}

function boxesAreApart(
  points: ExactPoints,
  first: readonly number[],
  second: readonly number[]
): boolean {
  const firstX = first.map(row => points.x[row]);
  const firstY = first.map(row => points.y[row]);
  const secondX = second.map(row => points.x[row]);
  const secondY = second.map(row => points.y[row]);
  return (
    Math.max(...firstX) < Math.min(...secondX) ||
    Math.max(...secondX) < Math.min(...firstX) ||
    Math.max(...firstY) < Math.min(...secondY) ||
    Math.max(...secondY) < Math.min(...firstY)
  );
}

function getGreatestCommonDivisor(first: bigint, second: bigint): bigint {
  let larger = first < 0n ? -first : first;
  let smaller = second < 0n ? -second : second;
  while (smaller !== 0n) {
    [larger, smaller] = [smaller, larger % smaller];
  }
  return larger;
}

/**
 * Why the directed edges of `triangles` (rows of the flattened rings) do not cancel to the
 * boundary of the polygon, or `undefined` when they do. `rings` must be a valid polygon (see
 * {@link getPolygonDefect}; rings may touch at single points). Exact: BigInt on the float32
 * coordinates.
 *
 * This is a complete test of a tiling. Every triangle is taken counter-clockwise, the shell
 * counter-clockwise and the holes clockwise. On each line of the plane the directed edges are
 * added as signed coverage: an edge from `p` to `q` counts +1 at `p` and -1 at `q`, so edges that
 * overlap in opposite directions cancel wherever they overlap, whether or not their ends agree (a
 * triangle side may run past a collinear polygon vertex). The number of counter-clockwise
 * triangles that cover a point is the winding number of their summed boundaries about it. If that
 * sum is the polygon boundary, the count is the polygon's own winding number: 1 inside and 0
 * outside. So the triangles cover the polygon exactly once and nothing else, which is a tiling;
 * and the edges of a tiling do cancel to the boundary. A triangle of three collinear vertices
 * adds nothing on its line; it is reported unless `ignoreZeroAreaTriangles`.
 */
export function getBoundaryChainDefect(
  rings: readonly SearchPoint[][],
  triangles: readonly (readonly number[])[],
  ignoreZeroAreaTriangles: boolean = false
): string | undefined {
  const flat = rings.flat();
  const points = getExactPoints(flat);
  // The common factor of the coordinates (a power of two times the lattice spacing) is dropped,
  // which keeps the integers short.
  let commonFactor = 0n;
  for (const value of [...points.exactX, ...points.exactY]) {
    commonFactor = getGreatestCommonDivisor(commonFactor, value);
  }
  const reduce = (value: bigint) => (commonFactor > 1n ? value / commonFactor : value);
  const x = points.exactX.map(reduce);
  const y = points.exactY.map(reduce);
  // Signed coverage change per line and position on the line, with one row at that position.
  const changes = new Map<string, {sum: number; row: number}>();
  const addChange = (key: string, row: number, weight: number) => {
    const entry = changes.get(key);
    if (entry) {
      entry.sum += weight;
    } else {
      changes.set(key, {sum: weight, row});
    }
  };
  const addEdge = (from: number, to: number, weight: number) => {
    let directionX = x[to] - x[from];
    let directionY = y[to] - y[from];
    if (directionX === 0n && directionY === 0n) {
      return;
    }
    const divisor = getGreatestCommonDivisor(directionX, directionY);
    directionX /= divisor;
    directionY /= divisor;
    // One name per line, whichever way the edge runs along it.
    if (directionX < 0n || (directionX === 0n && directionY < 0n)) {
      directionX = -directionX;
      directionY = -directionY;
    }
    const line = `${directionX},${directionY},${directionX * y[from] - directionY * x[from]}`;
    addChange(`${line}|${directionX * x[from] + directionY * y[from]}`, from, weight);
    addChange(`${line}|${directionX * x[to] + directionY * y[to]}`, to, -weight);
  };
  getRingRows(rings).forEach((rows, ring) => {
    let twiceArea = 0n;
    for (let index = 0; index < rows.length; index++) {
      const from = rows[index];
      const to = rows[(index + 1) % rows.length];
      twiceArea += x[from] * y[to] - x[to] * y[from];
    }
    // The polygon interior is on the left of its boundary: shell counter-clockwise, holes clockwise.
    const isForward = twiceArea > 0n === (ring === 0);
    for (let index = 0; index < rows.length; index++) {
      const from = rows[index];
      const to = rows[(index + 1) % rows.length];
      addEdge(isForward ? from : to, isForward ? to : from, -1);
    }
  });
  for (const triangle of triangles) {
    if (triangle.some(row => !(row >= 0 && row < flat.length))) {
      return `triangle ${triangle} has a row outside the polygon`;
    }
    const cross =
      (x[triangle[1]] - x[triangle[0]]) * (y[triangle[2]] - y[triangle[0]]) -
      (y[triangle[1]] - y[triangle[0]]) * (x[triangle[2]] - x[triangle[0]]);
    if (cross === 0n) {
      if (ignoreZeroAreaTriangles) {
        continue;
      }
      return `triangle ${triangle} has zero area`;
    }
    const [first, second, third] = cross > 0n ? triangle : [triangle[0], triangle[2], triangle[1]];
    addEdge(first, second, 1);
    addEdge(second, third, 1);
    addEdge(third, first, 1);
  }
  for (const {sum, row} of changes.values()) {
    if (sum !== 0) {
      return `the triangle edges do not cancel to the polygon boundary at vertex ${row} (${flat[row]})`;
    }
  }
  return undefined;
}

/**
 * Why `triangles` (rows of the flattened rings) is not a tiling of the polygon, or `undefined`
 * when it is one. `rings` must be a valid polygon (see {@link getPolygonDefect}). With
 * `ignoreZeroAreaTriangles`, triangles of three collinear vertices are skipped instead of
 * reported. The decision is that of {@link getBoundaryChainDefect}, which is complete; the tests
 * before it name the defect more closely where they see it (they miss a polygon edge that runs
 * through a triangle from a vertex on one of its sides to its opposite corner).
 */
export function getTilingDefect(
  rings: readonly SearchPoint[][],
  triangles: readonly (readonly number[])[],
  ignoreZeroAreaTriangles: boolean = false
): string | undefined {
  const flat = rings.flat();
  const points = getExactPoints(flat);
  const ringRows = getRingRows(rings);
  const edges: [number, number][] = [];
  for (const rows of ringRows) {
    for (let index = 0; index < rows.length; index++) {
      edges.push([rows[index], rows[(index + 1) % rows.length]]);
    }
  }
  let polygonArea = 0n;
  ringRows.forEach((rows, ring) => {
    const area = getExactRingArea(points, rows);
    polygonArea += ring === 0 ? area : -area;
  });
  let triangleArea = 0n;
  // Counter-clockwise copies of the triangles.
  const oriented: number[][] = [];
  for (const triangle of triangles) {
    if (triangle.some(row => !(row >= 0 && row < flat.length))) {
      return `triangle ${triangle} has a row outside the polygon`;
    }
    const cross = getExactCross(points, triangle[0], triangle[1], triangle[2]);
    if (cross === 0n) {
      if (ignoreZeroAreaTriangles) {
        continue;
      }
      return `triangle ${triangle} has zero area`;
    }
    triangleArea += cross < 0n ? -cross : cross;
    oriented.push(cross > 0n ? [...triangle] : [triangle[0], triangle[2], triangle[1]]);
  }
  for (const triangle of oriented) {
    // The centroid, times three, against the rings, times three: inside by even-odd parity.
    const sumX = triangle.reduce((sum, row) => sum + points.exactX[row], 0n);
    const sumY = triangle.reduce((sum, row) => sum + points.exactY[row], 0n);
    let inside = false;
    for (const [from, to] of edges) {
      const fromY = 3n * points.exactY[from];
      const toY = 3n * points.exactY[to];
      if (fromY > sumY === toY > sumY) {
        continue;
      }
      const side =
        (3n * points.exactX[to] - 3n * points.exactX[from]) * (sumY - fromY) -
        (toY - fromY) * (sumX - 3n * points.exactX[from]);
      if (side === 0n) {
        return `triangle ${triangle} has its centroid on the boundary`;
      }
      if (toY > fromY === side > 0n) {
        inside = !inside;
      }
    }
    if (!inside) {
      return `triangle ${triangle} has its centroid outside the polygon`;
    }
    for (let corner = 0; corner < 3; corner++) {
      const start = triangle[corner];
      const end = triangle[(corner + 1) % 3];
      for (const edge of edges) {
        if (boxesAreApart(points, [start, end], edge)) {
          continue;
        }
        const first = getOrientation(points, start, end, edge[0]);
        const second = getOrientation(points, start, end, edge[1]);
        if (first * second >= 0) {
          continue;
        }
        const third = getOrientation(points, edge[0], edge[1], start);
        const fourth = getOrientation(points, edge[0], edge[1], end);
        if (third * fourth < 0) {
          return `triangle ${triangle} crosses the polygon edge ${edge}`;
        }
      }
    }
    for (let row = 0; row < flat.length; row++) {
      if (boxesAreApart(points, triangle, [row])) {
        continue;
      }
      if (
        getOrientation(points, triangle[0], triangle[1], row) > 0 &&
        getOrientation(points, triangle[1], triangle[2], row) > 0 &&
        getOrientation(points, triangle[2], triangle[0], row) > 0
      ) {
        return `triangle ${triangle} contains the polygon vertex ${row}`;
      }
    }
  }
  for (let first = 0; first < oriented.length; first++) {
    for (let second = first + 1; second < oriented.length; second++) {
      if (boxesAreApart(points, oriented[first], oriented[second])) {
        continue;
      }
      // Two convex shapes share no interior point exactly when an edge of one has the other
      // entirely on its outer closed side.
      let separated = false;
      for (const [inner, outer] of [
        [oriented[first], oriented[second]],
        [oriented[second], oriented[first]]
      ]) {
        for (let corner = 0; corner < 3 && !separated; corner++) {
          const start = inner[corner];
          const end = inner[(corner + 1) % 3];
          separated = outer.every(row => getOrientation(points, start, end, row) <= 0);
        }
      }
      if (!separated) {
        return `triangles ${oriented[first]} and ${oriented[second]} overlap`;
      }
    }
  }
  if (triangleArea !== polygonArea) {
    return `triangle areas sum to ${triangleArea}, the polygon area is ${polygonArea} (twice, in exact units)`;
  }
  return getBoundaryChainDefect(rings, triangles, ignoreZeroAreaTriangles);
}

/**
 * Sum of the triangle areas, exact for coordinates that are multiples of `spacing` (a power of
 * two) with products below 2^53: integer lattices and the frames of this file.
 */
export function getTriangleAreaSum(
  rings: readonly SearchPoint[][],
  triangles: readonly (readonly number[])[],
  spacing: number = 1
): number {
  const flat = rings.flat();
  let twiceArea = 0n;
  for (const [first, second, third] of triangles) {
    const toLattice = (value: number) => {
      const scaled = value / spacing;
      if (!Number.isInteger(scaled)) {
        throw new Error(`coordinate ${value} is not a multiple of ${spacing}`);
      }
      return BigInt(scaled);
    };
    const [ax, ay] = flat[first].map(toLattice);
    const [bx, by] = flat[second].map(toLattice);
    const [cx, cy] = flat[third].map(toLattice);
    const cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    twiceArea += cross < 0n ? -cross : cross;
  }
  return (Number(twiceArea) * spacing * spacing) / 2;
}

function segmentsMeet(points: ExactPoints, a: number, b: number, c: number, d: number): boolean {
  if (boxesAreApart(points, [a, b], [c, d])) {
    return false;
  }
  const first = getOrientation(points, a, b, c);
  const second = getOrientation(points, a, b, d);
  const third = getOrientation(points, c, d, a);
  const fourth = getOrientation(points, c, d, b);
  // A collinear endpoint lies on the other segment when it is inside that segment's box.
  return (
    (first * second < 0 && third * fourth < 0) ||
    (first === 0 && !boxesAreApart(points, [a, b], [c])) ||
    (second === 0 && !boxesAreApart(points, [a, b], [d])) ||
    (third === 0 && !boxesAreApart(points, [c, d], [a])) ||
    (fourth === 0 && !boxesAreApart(points, [c, d], [b]))
  );
}

function isInsideRing(points: ExactPoints, row: number, rows: readonly number[]): boolean {
  let inside = false;
  for (let index = 0; index < rows.length; index++) {
    const from = rows[index];
    const to = rows[(index + 1) % rows.length];
    if (points.y[from] > points.y[row] === points.y[to] > points.y[row]) {
      continue;
    }
    const side = getOrientation(points, from, to, row);
    if (points.y[to] > points.y[from] === side > 0) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Why `rings` is not a valid polygon, or `undefined` when it is: every ring simple with non-zero
 * area, holes strictly inside the shell and outside each other, and no two rings meeting. With
 * `allowTouchingVertices`, two rings may share vertices (and nothing else); a caller that allows
 * this must keep the interior connected.
 */
export function getPolygonDefect(
  rings: readonly SearchPoint[][],
  allowTouchingVertices: boolean = false
): string | undefined {
  const flat = rings.flat();
  const points = getExactPoints(flat);
  const ringRows = getRingRows(rings);
  for (let ring = 0; ring < rings.length; ring++) {
    const rows = ringRows[ring];
    const count = rows.length;
    if (count < 3 || getExactRingArea(points, rows) === 0n) {
      return `ring ${ring} is degenerate`;
    }
    for (let index = 0; index < count; index++) {
      const a = rows[index];
      const b = rows[(index + 1) % count];
      const c = rows[(index + 2) % count];
      if (points.x[a] === points.x[b] && points.y[a] === points.y[b]) {
        return `ring ${ring} repeats a vertex`;
      }
      // Consecutive collinear edges must continue forward, not fold back.
      if (
        getOrientation(points, a, b, c) === 0 &&
        (points.x[b] - points.x[a]) * (points.x[c] - points.x[b]) +
          (points.y[b] - points.y[a]) * (points.y[c] - points.y[b]) <=
          0
      ) {
        return `ring ${ring} folds back`;
      }
      for (let other = index + 2; other < count; other++) {
        if (index === 0 && other === count - 1) {
          continue;
        }
        if (segmentsMeet(points, a, b, rows[other], rows[(other + 1) % count])) {
          return `ring ${ring} is not simple`;
        }
      }
    }
  }
  for (let first = 0; first < rings.length; first++) {
    for (let second = first + 1; second < rings.length; second++) {
      const firstRows = ringRows[first];
      const secondRows = ringRows[second];
      for (let i = 0; i < firstRows.length; i++) {
        for (let j = 0; j < secondRows.length; j++) {
          const a = firstRows[i];
          const b = firstRows[(i + 1) % firstRows.length];
          const c = secondRows[j];
          const d = secondRows[(j + 1) % secondRows.length];
          if (!segmentsMeet(points, a, b, c, d)) {
            continue;
          }
          // Two edges that share an endpoint meet only there when they are not collinear, or
          // when they leave it in opposite directions.
          const meetOnlyAt = (shared: number, firstEnd: number, secondEnd: number) =>
            getOrientation(points, shared, firstEnd, secondEnd) !== 0 ||
            (points.x[firstEnd] - points.x[shared]) * (points.x[secondEnd] - points.x[shared]) +
              (points.y[firstEnd] - points.y[shared]) * (points.y[secondEnd] - points.y[shared]) <
              0;
          const samePoint = (p: number, q: number) =>
            points.x[p] === points.x[q] && points.y[p] === points.y[q];
          const sharesOnlyAVertex =
            (samePoint(a, c) && meetOnlyAt(a, b, d)) ||
            (samePoint(a, d) && meetOnlyAt(a, b, c)) ||
            (samePoint(b, c) && meetOnlyAt(b, a, d)) ||
            (samePoint(b, d) && meetOnlyAt(b, a, c));
          if (!(allowTouchingVertices && sharesOnlyAVertex)) {
            return `rings ${first} and ${second} meet`;
          }
        }
      }
      // No edges cross, so one vertex decides containment (a touching vertex is skipped).
      const probe = secondRows.find(
        row =>
          !firstRows.some(
            other => points.x[row] === points.x[other] && points.y[row] === points.y[other]
          )
      );
      if (probe === undefined) {
        return `rings ${first} and ${second} share every vertex`;
      }
      const inside = isInsideRing(points, probe, firstRows);
      if (first === 0 ? !inside : inside) {
        return `ring ${second} is on the wrong side of ring ${first}`;
      }
      if (first > 0) {
        const reverseProbe = firstRows.find(
          row =>
            !secondRows.some(
              other => points.x[row] === points.x[other] && points.y[row] === points.y[other]
            )
        );
        if (reverseProbe === undefined || isInsideRing(points, reverseProbe, secondRows)) {
          return `ring ${first} is inside ring ${second}`;
        }
      }
    }
  }
  return undefined;
}

// Generators. All return lattice (integer) rings; map them with `mapToFrame`.

export type RandomHoledPolygonOptions = {
  /** Lattice side. Small values give many collinear triples and equal coordinates. */
  range: number;
  /**
   * Shell shape (default `'two-opt'`): a random tour made simple by 2-opt moves, or a star-shaped
   * ring of random radii, which is cheap enough for more than 80 vertices.
   */
  shell?: 'two-opt' | 'star';
  shellVertexCount: number;
  holeCount: number;
  /** Largest hole radius in lattice steps. */
  holeRadius: number;
  /** Probability that a hole is moved to share one vertex with another ring. */
  touchProbability?: number;
};

function createLatticeHole(random: () => number, range: number, holeRadius: number): SearchPoint[] {
  const vertexCount = 3 + Math.floor(random() * 4);
  const radius = 1 + random() * (holeRadius - 1);
  const centerX = radius + random() * (range - 2 * radius);
  const centerY = radius + random() * (range - 2 * radius);
  const ring: SearchPoint[] = [];
  for (let index = 0; index < vertexCount; index++) {
    const angle = (2 * Math.PI * (index + 0.1 + 0.8 * random())) / vertexCount;
    const length = radius * (0.35 + 0.65 * random());
    const point: SearchPoint = [
      Math.round(centerX + Math.cos(angle) * length),
      Math.round(centerY + Math.sin(angle) * length)
    ];
    const previous = ring[ring.length - 1];
    if (!previous || previous[0] !== point[0] || previous[1] !== point[1]) {
      ring.push(point);
    }
  }
  return ring;
}

/**
 * A random valid polygon with holes on an integer lattice: a 2-opt shell and star-shaped holes
 * kept by rejection. Returns `undefined` when no valid shell was found; the hole count can be
 * lower than asked when the shell has no room.
 */
export function createRandomHoledPolygon(
  seed: number,
  options: RandomHoledPolygonOptions
): SearchPoint[][] | undefined {
  const random = createRandom(seed);
  const radius = Math.floor(options.range / 2);
  const shell =
    options.shell === 'star'
      ? createRandomStarPolygon(options.shellVertexCount, seed, radius).map(
          ([x, y]): SearchPoint => [x + radius, y + radius]
        )
      : createTwoOptPolygon(options.shellVertexCount, seed, options.range);
  if (!shell || getPolygonDefect([shell]) !== undefined) {
    return undefined;
  }
  const rings: SearchPoint[][] = [shell];
  const touchProbability = options.touchProbability ?? 0;
  for (let attempt = 0; attempt < 40 * options.holeCount; attempt++) {
    if (rings.length > options.holeCount) {
      break;
    }
    let hole = createLatticeHole(random, options.range, options.holeRadius);
    if (hole.length < 3) {
      continue;
    }
    let touches = false;
    if (random() < touchProbability) {
      // Translate the hole so that one of its vertices lands on a vertex of another ring.
      const target = rings[Math.floor(random() * rings.length)];
      const targetPoint = target[Math.floor(random() * target.length)];
      const own = hole[Math.floor(random() * hole.length)];
      hole = hole.map(
        ([x, y]): SearchPoint => [x + targetPoint[0] - own[0], y + targetPoint[1] - own[1]]
      );
      touches = true;
    }
    // At most one shared vertex per added hole keeps the interior connected (the rings that
    // touch form a forest).
    const sharedVertexCount = rings.reduce(
      (sum, ring) =>
        sum +
        hole.filter(point => ring.some(other => other[0] === point[0] && other[1] === point[1]))
          .length,
      0
    );
    if (
      sharedVertexCount <= (touches ? 1 : 0) &&
      getPolygonDefect([...rings, hole], touchProbability > 0) === undefined
    ) {
      rings.push(hole);
    }
  }
  return rings;
}

/**
 * Two downward shell edges that the leftward ray of a hole's leftmost vertex crosses less than
 * one lattice step apart, with an upward edge between them (a thin interior sliver, then a thin
 * exterior notch). In exact arithmetic the nearer edge is `tip -> foot`; rounded to the lattice
 * both crossings are the same number. `rotation` rotates the shell's first vertex, which changes
 * the order in which the walk meets the two edges.
 *
 * Shell (counter-clockwise): far corner, `shoulder`, `heel`, `tip`, `foot`, two right corners.
 * The ray is `y = 0`; the hole sits to the right of the origin.
 */
export function createCloseCrossingPolygon(seed: number, rotation: number): SearchPoint[][] {
  const random = createRandom(seed);
  const integer = (low: number, high: number) => low + Math.floor(random() * (high - low + 1));
  const shoulderHeight = integer(2, 6);
  const tipHeight = integer(8, 40);
  const footDepth = tipHeight + integer(1, 60);
  const holeX = integer(2, 40);
  const top = tipHeight + integer(5, 30);
  const shell: SearchPoint[] = [
    [-integer(20, 80), top],
    [-1, shoulderHeight],
    [0, -1],
    [0, tipHeight],
    [1, -footDepth],
    [holeX + integer(8, 30), -footDepth],
    [holeX + integer(8, 30), top]
  ];
  const hole: SearchPoint[] = [
    [holeX, 0],
    [holeX + integer(2, 5), integer(1, 3)],
    [holeX + integer(2, 5), -integer(1, 3)]
  ];
  const shift = ((rotation % shell.length) + shell.length) % shell.length;
  return [[...shell.slice(shift), ...shell.slice(0, shift)], hole];
}

/**
 * Two reflex shell vertices (the tips of two thin notches) whose directions from the hole's
 * leftmost vertex differ by less than a float32 division can tell: `near` has the smaller angle
 * to the ray and is the only bridge that does not cross the other notch. `mirror` flips the
 * shape in `y`, which changes the order in which the walk meets the two tips.
 *
 * The directions are two neighbours of a Farey sequence, `scale / (2 scale + 1)` and
 * `(scale + 1) / (2 scale + 3)`, so they are never collinear with the hole vertex.
 */
export function createCloseTangentPolygon(seed: number, mirror: boolean): SearchPoint[][] {
  const random = createRandom(seed);
  const integer = (low: number, high: number) => low + Math.floor(random() * (high - low + 1));
  const scale = integer(3000, 12000);
  const near: SearchPoint = [-(2 * scale + 1), scale];
  const far: SearchPoint = [-(2 * scale + 3), scale + 1];
  const margin = integer(4000, 9000);
  const top = scale + margin;
  const left = -(2 * scale + margin);
  // The upper left corner must be steeper from the hole than the tips: margin > lean.
  const lean = integer(1, 50);
  const right = integer(20, 200);
  // Counter-clockwise from the top left: down the leaning left edge, along the bottom, up the
  // right side, and back along the top, where the two notches hang down.
  const shell: SearchPoint[] = [
    [left - lean, top],
    [left, -integer(100, 2000)],
    [right, -integer(100, 2000)],
    [right, top],
    [near[0] + integer(2, 9), top],
    near,
    [near[0], top],
    [far[0] + 1, top],
    far,
    [far[0] - integer(2, 9), top]
  ];
  const hole: SearchPoint[] = [
    [0, 0],
    [integer(3, 15), integer(2, 9)],
    [integer(3, 15), -integer(2, 9)]
  ];
  if (!mirror) {
    return [shell, hole];
  }
  return [shell, hole].map(ring => ring.map(([x, y]): SearchPoint => [x, -y]));
}

/**
 * Two holes that share their leftmost vertex, the origin, with a sliver of interior between
 * them: the thin upper hole's first edge and the lower hole's first edge leave the origin in
 * directions `(scale + 1) / (2 scale + 3)` and `scale / (2 scale + 1)`, which a float32 slope
 * cannot tell apart. Earcut orders such holes by that slope. `upperFirst` picks the ring order.
 */
export function createSharedLeftmostPolygon(seed: number, upperFirst: boolean): SearchPoint[][] {
  const random = createRandom(seed);
  const integer = (low: number, high: number) => low + Math.floor(random() * (high - low + 1));
  const scale = integer(8000, 15000);
  const upper: SearchPoint[] = [
    [0, 0],
    [4 * scale + 6, 2 * scale + 2],
    [4 * scale + 4, 2 * scale + 1]
  ];
  const lower: SearchPoint[] = [
    [0, 0],
    [2 * scale + 1, scale],
    [2 * scale + 1, scale - integer(100, 4000)]
  ];
  const left = -integer(10, 3000);
  const right = 4 * scale + integer(100, 3000);
  const shell: SearchPoint[] = [
    [left, -integer(100, 3000)],
    [right, -integer(100, 3000)],
    [right, 2 * scale + integer(100, 3000)],
    [left - integer(0, 50), 2 * scale + integer(100, 3000)]
  ];
  return upperFirst ? [shell, upper, lower] : [shell, lower, upper];
}

/** The next float32 above `value`. */
export function getNextFloat32(value: number): number {
  floatView.setFloat32(0, value);
  const bits = floatView.getUint32(0);
  if (value === 0) {
    floatView.setUint32(0, 1);
  } else {
    floatView.setUint32(0, value > 0 ? bits + 1 : bits - 1);
  }
  return floatView.getFloat32(0);
}

/**
 * A quadrilateral shell with a slanted left edge and a small triangular hole whose leftmost
 * vertex is the first few float32 values to the right of that edge, at float32 coordinates that
 * are not on a lattice. `frame` picks the magnitudes: `'local'` spans zero (the hole vertex has a
 * much finer resolution than the edge ends), `'metre'` is projected-metre magnitude.
 */
export function createNearEdgeHolePolygon(
  seed: number,
  frame: 'local' | 'metre'
): SearchPoint[][] | undefined {
  const random = createRandom(seed);
  const uniform = (low: number, high: number) => Math.fround(low + random() * (high - low));
  const originX = frame === 'metre' ? 500000 : 0;
  const originY = frame === 'metre' ? 4500000 : 0;
  const extent = frame === 'metre' ? 3000 : 2000;
  const topLeft: SearchPoint = [
    Math.fround(originX + uniform(-extent, extent * 0.2)),
    Math.fround(originY + uniform(extent * 0.3, extent))
  ];
  const bottomLeft: SearchPoint = [
    Math.fround(originX + uniform(-extent, extent * 0.2)),
    Math.fround(originY - uniform(extent * 0.3, extent))
  ];
  const right = Math.fround(originX + extent * 2);
  const shell: SearchPoint[] = [topLeft, bottomLeft, [right, bottomLeft[1]], [right, topLeft[1]]];
  const holeY = Math.fround(originY + uniform(-extent * 0.25, extent * 0.25));
  const fraction = (holeY - topLeft[1]) / (bottomLeft[1] - topLeft[1]);
  let holeX = Math.fround(topLeft[0] + fraction * (bottomLeft[0] - topLeft[0]));
  // Step to the first float32 strictly right of the edge, then up to three more.
  for (
    let step = 0;
    step < 8 && getPointOrientation(topLeft, bottomLeft, [holeX, holeY]) <= 0;
    step++
  ) {
    holeX = getNextFloat32(holeX);
  }
  for (let step = Math.floor(random() * 4); step > 0; step--) {
    holeX = getNextFloat32(holeX);
  }
  const width = uniform(extent * 0.01, extent * 0.2);
  const height = uniform(extent * 0.005, extent * 0.05);
  const hole: SearchPoint[] = [
    [holeX, holeY],
    [Math.fround(holeX + width), Math.fround(holeY + height)],
    [Math.fround(holeX + width), Math.fround(holeY - height)]
  ];
  const rings = [shell, hole];
  return getPolygonDefect(rings) === undefined ? rings : undefined;
}
