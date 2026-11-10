// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU lookups for pointer interactions: which cell, point, segment, polygon or chart bar is under
// the pointer. They read the small typed arrays a run already holds in its inputs; nothing here
// touches the GPU. All coordinates are drawing-world units (x right, y up).

export type PickPoint = readonly [number, number];

// ---------------------------------------------------------------------------------------------
// Rasters

export type RasterCell = {column: number; row: number; index: number; isInside: boolean};

/**
 * The raster cell under a world position. `bounds` is the world rectangle the raster covers
 * ([minimumX, minimumY, maximumX, maximumY]); row 0 is at the top, as the raster layer draws it.
 * Positions outside the raster clamp to its edge and report `isInside: false`.
 */
export function getRasterCell(
  columns: number,
  rows: number,
  bounds: readonly [number, number, number, number],
  x: number,
  y: number
): RasterCell {
  const u = (x - bounds[0]) / (bounds[2] - bounds[0]);
  const v = (bounds[3] - y) / (bounds[3] - bounds[1]);
  const column = Math.min(columns - 1, Math.max(0, Math.floor(u * columns)));
  const row = Math.min(rows - 1, Math.max(0, Math.floor(v * rows)));
  return {
    column,
    row,
    index: row * columns + column,
    isInside: u >= 0 && u <= 1 && v >= 0 && v <= 1
  };
}

/** World position of a raster cell's center; the inverse of getRasterCell. */
export function getRasterCellCenter(
  columns: number,
  rows: number,
  bounds: readonly [number, number, number, number],
  column: number,
  row: number
): [number, number] {
  return [
    bounds[0] + ((column + 0.5) / columns) * (bounds[2] - bounds[0]),
    bounds[3] - ((row + 0.5) / rows) * (bounds[3] - bounds[1])
  ];
}

// ---------------------------------------------------------------------------------------------
// Points

export type PointIndex = {
  readonly count: number;
  /** Index of the nearest point, or -1 when none lies within `maximumDistance`. */
  nearest(x: number, y: number, maximumDistance?: number): number;
  /** Indices of the points within `radius`, nearest first. */
  within(x: number, y: number, radius: number): number[];
  /** Distance from (x, y) to point `index`. */
  distanceTo(index: number, x: number, y: number): number;
};

/**
 * A uniform grid over packed (x, y) positions. Build it once in `createInputs`; lookups are fast
 * enough to run on every pointer move for tens of thousands of points.
 */
export function createPointIndex(
  positions: ArrayLike<number>,
  count = positions.length >> 1
): PointIndex {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < count; index++) {
    const x = positions[index * 2];
    const y = positions[index * 2 + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < minimumX) minimumX = x;
    if (x > maximumX) maximumX = x;
    if (y < minimumY) minimumY = y;
    if (y > maximumY) maximumY = y;
  }
  const width = Math.max(maximumX - minimumX, 1e-9);
  const height = Math.max(maximumY - minimumY, 1e-9);
  const side = Math.max(1, Math.min(64, Math.round(Math.sqrt(count / 2))));
  const cellWidth = width / side;
  const cellHeight = height / side;
  const cells: number[][] = Array.from({length: side * side}, () => []);
  const getCell = (value: number, minimum: number, size: number) =>
    Math.min(side - 1, Math.max(0, Math.floor((value - minimum) / size)));
  for (let index = 0; index < count; index++) {
    const x = positions[index * 2];
    const y = positions[index * 2 + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    cells[getCell(y, minimumY, cellHeight) * side + getCell(x, minimumX, cellWidth)].push(index);
  }
  const distanceTo = (index: number, x: number, y: number) =>
    Math.hypot(positions[index * 2] - x, positions[index * 2 + 1] - y);
  const within = (x: number, y: number, radius: number): number[] => {
    const found: {index: number; distance: number}[] = [];
    const column0 = getCell(x - radius, minimumX, cellWidth);
    const column1 = getCell(x + radius, minimumX, cellWidth);
    const row0 = getCell(y - radius, minimumY, cellHeight);
    const row1 = getCell(y + radius, minimumY, cellHeight);
    for (let row = row0; row <= row1; row++) {
      for (let column = column0; column <= column1; column++) {
        for (const index of cells[row * side + column]) {
          const distance = distanceTo(index, x, y);
          if (distance <= radius) found.push({index, distance});
        }
      }
    }
    return found.sort((first, second) => first.distance - second.distance).map(item => item.index);
  };
  return {
    count,
    distanceTo,
    within,
    nearest(x, y, maximumDistance = Infinity) {
      // Grow the search ring until a hit is certain to be the nearest.
      const reach = Math.max(width, height) * 2;
      for (let radius = Math.max(cellWidth, cellHeight); ; radius *= 2) {
        const limit = Math.min(radius, maximumDistance);
        const found = within(x, y, limit);
        if (found.length > 0) return found[0];
        if (radius >= maximumDistance || radius > reach + Math.hypot(x - minimumX, y - minimumY)) {
          return -1;
        }
      }
    }
  };
}

/**
 * Sticky choice among snapped candidates: keeps `previous` unless `candidate` is clearly nearer
 * (by `margin`, default 15%). Stops flicker when the pointer sits between two points.
 */
export function chooseSticky(
  index: PointIndex,
  x: number,
  y: number,
  previous: number,
  margin = 0.15
): number {
  const candidate = index.nearest(x, y);
  if (previous < 0 || previous >= index.count || candidate === previous) return candidate;
  if (candidate < 0) return previous;
  return index.distanceTo(candidate, x, y) < index.distanceTo(previous, x, y) * (1 - margin)
    ? candidate
    : previous;
}

// ---------------------------------------------------------------------------------------------
// Segments and lines

export type SegmentHit = {
  /** Segment row. */
  index: number;
  /** 0 at the segment start, 1 at its end. */
  fraction: number;
  /** Distance from the query to the closest point. */
  distance: number;
  /** Positive when the query is to the left of the segment's direction. */
  signedDistance: number;
  /** The closest point on the segment. */
  point: [number, number];
};

function projectOnSegment(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  x: number,
  y: number,
  index: number
): SegmentHit {
  const deltaX = x1 - x0;
  const deltaY = y1 - y0;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  const fraction =
    lengthSquared > 0
      ? Math.min(1, Math.max(0, ((x - x0) * deltaX + (y - y0) * deltaY) / lengthSquared))
      : 0;
  const pointX = x0 + deltaX * fraction;
  const pointY = y0 + deltaY * fraction;
  const distance = Math.hypot(x - pointX, y - pointY);
  const cross = deltaX * (y - y0) - deltaY * (x - x0);
  return {
    index,
    fraction,
    distance,
    signedDistance: cross >= 0 ? distance : -distance,
    point: [pointX, pointY]
  };
}

/** The closest of `count` packed (x0, y0, x1, y1) segments, or null when there are none. */
export function findNearestSegment(
  segments: ArrayLike<number>,
  x: number,
  y: number,
  count = segments.length >> 2
): SegmentHit | null {
  let best: SegmentHit | null = null;
  for (let index = 0; index < count; index++) {
    const hit = projectOnSegment(
      segments[index * 4],
      segments[index * 4 + 1],
      segments[index * 4 + 2],
      segments[index * 4 + 3],
      x,
      y,
      index
    );
    if (!best || hit.distance < best.distance) best = hit;
  }
  return best;
}

/**
 * The closest point on a polyline of packed (x, y) vertices from `start` to `start + count - 1`.
 * `index` is the first vertex of the closest segment; with `closed`, the last vertex joins the first.
 */
export function findNearestOnPolyline(
  positions: ArrayLike<number>,
  x: number,
  y: number,
  start = 0,
  count = (positions.length >> 1) - start,
  closed = false
): SegmentHit | null {
  let best: SegmentHit | null = null;
  const last = closed ? count : count - 1;
  for (let step = 0; step < last; step++) {
    const first = start + step;
    const second = start + ((step + 1) % count);
    const hit = projectOnSegment(
      positions[first * 2],
      positions[first * 2 + 1],
      positions[second * 2],
      positions[second * 2 + 1],
      x,
      y,
      first
    );
    if (!best || hit.distance < best.distance) best = hit;
  }
  return best;
}

/** The closest edge of a node-and-edge network: `edges` holds packed (from, to) node indices. */
export function findNearestEdge(
  nodes: ArrayLike<number>,
  edges: ArrayLike<number>,
  x: number,
  y: number,
  edgeCount = edges.length >> 1
): SegmentHit | null {
  let best: SegmentHit | null = null;
  for (let index = 0; index < edgeCount; index++) {
    const from = edges[index * 2];
    const to = edges[index * 2 + 1];
    const hit = projectOnSegment(
      nodes[from * 2],
      nodes[from * 2 + 1],
      nodes[to * 2],
      nodes[to * 2 + 1],
      x,
      y,
      index
    );
    if (!best || hit.distance < best.distance) best = hit;
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// Polygons

function isInsideTriangle(
  x: number,
  y: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number
): boolean {
  // A degenerate triangle (padding in an index buffer) or a non-finite input contains nothing.
  const doubleArea = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  if (!(Math.abs(doubleArea) > 0) || !Number.isFinite(x) || !Number.isFinite(y)) return false;
  const first = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
  const second = (cx - bx) * (y - by) - (cy - by) * (x - bx);
  const third = (ax - cx) * (y - cy) - (ay - cy) * (x - cx);
  const hasNegative = first < 0 || second < 0 || third < 0;
  const hasPositive = first > 0 || second > 0 || third > 0;
  return !(hasNegative && hasPositive);
}

/**
 * The polygon under a world position, for polygons drawn as a triangle mesh (the `polygons`
 * layer's inputs): packed vertex positions, three vertex ids per triangle and the owner polygon of
 * each triangle. Returns the owner row, or -1 when the position is in no polygon.
 */
export function findPolygonAt(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  owners: ArrayLike<number>,
  x: number,
  y: number,
  triangleCount = owners.length
): number {
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const first = indices[triangle * 3];
    const second = indices[triangle * 3 + 1];
    const third = indices[triangle * 3 + 2];
    if (
      isInsideTriangle(
        x,
        y,
        positions[first * 2],
        positions[first * 2 + 1],
        positions[second * 2],
        positions[second * 2 + 1],
        positions[third * 2],
        positions[third * 2 + 1]
      )
    ) {
      return owners[triangle];
    }
  }
  return -1;
}

/** Point-in-ring test (even-odd) on packed (x, y) vertices from `start`, `count` of them. */
export function isInsideRing(
  positions: ArrayLike<number>,
  x: number,
  y: number,
  start = 0,
  count = (positions.length >> 1) - start
): boolean {
  let isInside = false;
  for (let index = 0, previous = count - 1; index < count; previous = index++) {
    const xi = positions[(start + index) * 2];
    const yi = positions[(start + index) * 2 + 1];
    const xj = positions[(start + previous) * 2];
    const yj = positions[(start + previous) * 2 + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) isInside = !isInside;
  }
  return isInside;
}

// ---------------------------------------------------------------------------------------------
// Charts and matrices

export type ChartHit = {
  /** 0 to 1 across the plot rectangle, clamped. */
  u: number;
  /** 0 at the bottom of the plot rectangle, 1 at the top, clamped. */
  v: number;
  isInside: boolean;
};

/**
 * Where a world position falls in a chart layer's plot `rectangle` (fractions of the world, the
 * same four numbers the chart layer takes). Use `Math.floor(u * barCount)` for the bar under the
 * pointer, and `xRange[0] + u * (xRange[1] - xRange[0])` for a value on the x axis.
 */
export function getChartPosition(
  rectangle: readonly [number, number, number, number],
  world: {width: number; height: number},
  x: number,
  y: number
): ChartHit {
  const u = (x / world.width - rectangle[0]) / (rectangle[2] - rectangle[0]);
  const v = (y / world.height - rectangle[1]) / (rectangle[3] - rectangle[1]);
  return {
    u: Math.min(1, Math.max(0, u)),
    v: Math.min(1, Math.max(0, v)),
    isInside: u >= 0 && u <= 1 && v >= 0 && v <= 1
  };
}

/** World x of a fraction `u` across a chart's plot rectangle, for a chart rule glyph. */
export function getChartWorldX(
  rectangle: readonly [number, number, number, number],
  world: {width: number; height: number},
  u: number
): number {
  return (rectangle[0] + u * (rectangle[2] - rectangle[0])) * world.width;
}
