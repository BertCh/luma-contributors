// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shoreline and river profiles shared by the geometry-processing runs. Polygons are arranged on a
// jittered grid in the drawing world, packed in the four-level polygon layout the operations read
// (features own polygons own rings own vertices; rings are open, the closing edge is implicit).

import {
  createConvexShoreRing,
  createRiverBend,
  createRiverLine,
  createShoreRing,
  RIVER_REACH_COUNT
} from '../../../../data/hydrography';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import {triangulatePolygon} from './triangulation';

export type World = {width: number; height: number};
export type Random = () => number;

export type PolygonSetOptions = {
  /** Use a six-vertex simplified convex hull of a Great Lake. */
  convexShore?: boolean;
  columns?: number;
  rows?: number;
  /** Vertices of each shell ring. */
  vertexCount?: number;
  /** Controls how strongly the sampled shoreline departs from a circle. */
  irregularity?: number;
  /** Fraction of the cell filled by a polygon. */
  fill?: number;
  /** 0 keeps shapes equal-sided; up to 1 squashes some polygons into slivers. */
  stretch?: number;
  /** Repeats the first vertex at the end of every ring (explicit closure). */
  closeRings?: boolean;
  /** Bends every polygon into a banana; 0 is straight, 1 is strongly curved. */
  bend?: number;
  /** Reverses the winding of every other polygon (and its hole) when true. */
  mixedWinding?: boolean;
  /** Adds a circular hole to every polygon when true. */
  hasHoles?: boolean;
  /** Lifts every polygon by this many world units, to leave a strip of the picture free. */
  originY?: number;
};

export type PolygonSet = {
  count: number;
  positions: Float32Array;
  /** Ring positions as created, before any vertex drag. */
  basePositions: Float32Array;
  /** True when every ring repeats its first vertex at the end. */
  closeRings: boolean;
  /** First outline row of every ring (open rings, closing edge included). */
  outlineOffsets: Uint32Array;
  /** First fill vertex row and first fill triangle of every feature. */
  fillOffsets: Uint32Array;
  triangleOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Shell centers (x, y). */
  centers: Float32Array;
  /** Fill mesh: the ring vertices of each feature, ear-clipped around its holes. */
  fillVertices: Float32Array;
  fillIndices: Uint32Array;
  fillOwners: Uint32Array;
  fillTriangleCount: number;
  /** Every ring edge as (x0, y0, x1, y1), closing edges included. */
  outline: Float32Array;
  outlineCount: number;
};

/** Star-shaped polygons (optionally with a hole each) packed in the four-level layout. */
export function createPolygonSet(
  random: Random,
  world: World,
  options: PolygonSetOptions = {}
): PolygonSet {
  const rows = options.rows ?? 3;
  const columns = options.columns ?? Math.max(2, Math.round(rows * world.width));
  const vertexCount = options.vertexCount ?? 14;
  const irregularity = options.irregularity ?? 0.35;
  const fill = options.fill ?? 0.78;
  const cellWidth = world.width / columns;
  const cellHeight = world.height / rows;
  const positions: number[] = [];
  const ringOffsets = [0];
  const outlineOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const centers: number[] = [];
  const fillOffsets = [0];
  const triangleOffsets = [0];
  const addRing = (ring: number[][]) => {
    for (const [x, y] of options.closeRings ? [...ring, ring[0]] : ring) positions.push(x, y);
    ringOffsets.push(positions.length / 2);
    outlineOffsets.push(outlineOffsets[outlineOffsets.length - 1] + ring.length);
  };
  let featureIndex = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const centerX = (column + 0.5 + (random() - 0.5) * 0.12) * cellWidth;
      const centerY = (row + 0.5 + (random() - 0.5) * 0.12) * cellHeight + (options.originY ?? 0);
      const radius = Math.min(cellWidth, cellHeight) * 0.5 * fill * (0.85 + random() * 0.3);
      const phase = random() * Math.PI * 2;
      const squash = 1 - (options.stretch ?? 0) * random() * 0.85;
      const profile = options.convexShore
        ? createConvexShoreRing(featureIndex, radius)
        : createShoreRing(
            featureIndex,
            0,
            0,
            radius,
            vertexCount,
            Math.min(1, irregularity ** 0.6)
          );
      const shell = profile.map(([localX, profileY]) => {
        const localY =
          profileY * squash * (options.bend ? 0.45 : 1) +
          (((options.bend ?? 0) * (localX * localX)) / radius) * 1.4;
        return [
          centerX + localX * Math.cos(phase) - localY * Math.sin(phase),
          centerY + localX * Math.sin(phase) + localY * Math.cos(phase)
        ];
      });
      const isReversed = Boolean(options.mixedWinding) && featureIndex % 2 === 1;
      addRing(isReversed ? [...shell].reverse() : shell);
      let holeCount = 0;
      if (options.hasHoles) {
        const holeRadius = Math.min(...profile.map(([x, y]) => Math.hypot(x, y))) * squash * 0.35;
        const hole: number[][] = [];
        for (let vertex = 0; vertex < 8; vertex++) {
          const angle = -(vertex / 8) * Math.PI * 2;
          hole.push([
            centerX + Math.cos(angle) * holeRadius,
            centerY + Math.sin(angle) * holeRadius
          ]);
        }
        addRing(isReversed ? [...hole].reverse() : hole);
        holeCount = 1;
      }
      polygonOffsets.push(ringOffsets.length - 1);
      featureOffsets.push(polygonOffsets.length - 1);
      centers.push(centerX, centerY);
      const pointCount = shell.length + holeCount * 8;
      fillOffsets.push(fillOffsets[fillOffsets.length - 1] + pointCount);
      triangleOffsets.push(
        triangleOffsets[triangleOffsets.length - 1] + pointCount + 2 * holeCount - 2
      );
      featureIndex++;
    }
  }
  const triangleCount = triangleOffsets[triangleOffsets.length - 1];
  const outlineCount = outlineOffsets[outlineOffsets.length - 1];
  const set: PolygonSet = {
    count: featureIndex,
    positions: Float32Array.from(positions),
    basePositions: Float32Array.from(positions),
    closeRings: Boolean(options.closeRings),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    outlineOffsets: Uint32Array.from(outlineOffsets),
    fillOffsets: Uint32Array.from(fillOffsets),
    triangleOffsets: Uint32Array.from(triangleOffsets),
    centers: Float32Array.from(centers),
    fillVertices: new Float32Array(fillOffsets[fillOffsets.length - 1] * 2),
    fillIndices: new Uint32Array(triangleCount * 3),
    fillOwners: new Uint32Array(triangleCount),
    fillTriangleCount: triangleCount,
    outline: new Float32Array(outlineCount * 4),
    outlineCount
  };
  for (let feature = 0; feature < set.count; feature++) {
    set.fillOwners.fill(feature, set.triangleOffsets[feature], set.triangleOffsets[feature + 1]);
    rebuildFeature(set, feature);
  }
  return set;
}

/**
 * Rewrites the derived rows of one feature (the closing vertex, outline edges, fill vertices and
 * the ear-clipped fill triangles) from its current ring positions.
 */
export function rebuildFeature(set: PolygonSet, feature: number): void {
  const firstRing = set.polygonOffsets[set.featureOffsets[feature]];
  const lastRing = set.polygonOffsets[set.featureOffsets[feature + 1]];
  const rings: Array<Array<readonly [number, number]>> = [];
  for (let ring = firstRing; ring < lastRing; ring++) {
    const start = set.ringOffsets[ring];
    const count = set.ringOffsets[ring + 1] - start - (set.closeRings ? 1 : 0);
    if (set.closeRings) {
      set.positions[(start + count) * 2] = set.positions[start * 2];
      set.positions[(start + count) * 2 + 1] = set.positions[start * 2 + 1];
    }
    const points: Array<readonly [number, number]> = [];
    for (let vertex = 0; vertex < count; vertex++) {
      points.push([set.positions[(start + vertex) * 2], set.positions[(start + vertex) * 2 + 1]]);
    }
    for (let vertex = 0; vertex < count; vertex++) {
      const next = points[(vertex + 1) % count];
      set.outline.set(
        [points[vertex][0], points[vertex][1], next[0], next[1]],
        (set.outlineOffsets[ring] + vertex) * 4
      );
    }
    rings.push(points);
  }
  const fillBase = set.fillOffsets[feature];
  let row = fillBase;
  for (const ring of rings) {
    for (const [x, y] of ring) set.fillVertices.set([x, y], row++ * 2);
  }
  const triangles = triangulatePolygon(rings[0], rings.slice(1));
  const triangleBase = set.triangleOffsets[feature] * 3;
  for (let index = 0; index < triangles.length; index++) {
    set.fillIndices[triangleBase + index] = fillBase + triangles[index];
  }
}

export type VertexHandle = {feature: number; vertex: number};
export type VertexDrag = VertexHandle & {
  target: readonly [number, number];
  /** Neighbors on each side that follow the vertex with a cosine falloff. */
  spread?: number;
};

/** Where a shell vertex was before any drag. */
export function getBaseVertex(set: PolygonSet, handle: VertexHandle): [number, number] {
  const first = set.ringOffsets[set.polygonOffsets[set.featureOffsets[handle.feature]]];
  return [
    set.basePositions[(first + handle.vertex) * 2],
    set.basePositions[(first + handle.vertex) * 2 + 1]
  ];
}

/**
 * The shell vertex nearest to (x, y) before any drag. Keeps `previous` unless the nearest vertex
 * is clearly closer, so the choice does not flicker between two vertices. Within `hold` of the
 * previous vertex the pointer keeps carrying it, which lets a vertex be dragged across its ring.
 */
export function findNearestVertex(
  set: PolygonSet,
  x: number,
  y: number,
  previous: VertexHandle | null = null,
  hold = 0,
  margin = 0.15
): VertexHandle | null {
  let best: VertexHandle | null = null;
  let bestDistance = Infinity;
  for (let feature = 0; feature < set.count; feature++) {
    const ring = set.polygonOffsets[set.featureOffsets[feature]];
    const first = set.ringOffsets[ring];
    const count = set.ringOffsets[ring + 1] - first - (set.closeRings ? 1 : 0);
    for (let vertex = 0; vertex < count; vertex++) {
      const distance = Math.hypot(
        set.basePositions[(first + vertex) * 2] - x,
        set.basePositions[(first + vertex) * 2 + 1] - y
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        best = {feature, vertex};
      }
    }
  }
  if (!best || !previous) return best;
  const previousBase = getBaseVertex(set, previous);
  const previousDistance = Math.hypot(previousBase[0] - x, previousBase[1] - y);
  // A pointer right on another vertex always picks it, so the poster's rest position is unique.
  if (bestDistance > 0.02 && previousDistance <= hold) return previous;
  return bestDistance < previousDistance * (1 - margin) ? best : previous;
}

/** The shell positions (x, y pairs) of a feature with `drag` applied to the base geometry. */
export function getDraggedShell(set: PolygonSet, drag: VertexDrag): Float32Array {
  const {first, count} = getShellRows(set, drag.feature);
  const shell = set.basePositions.slice(first * 2, (first + count) * 2);
  const spread = drag.spread ?? 0;
  const [baseX, baseY] = getBaseVertex(set, drag);
  for (let offset = -spread; offset <= spread; offset++) {
    const vertex = (((drag.vertex + offset) % count) + count) % count;
    const weight = 0.5 * (1 + Math.cos((Math.PI * offset) / (spread + 1)));
    shell[vertex * 2] += (drag.target[0] - baseX) * weight;
    shell[vertex * 2 + 1] += (drag.target[1] - baseY) * weight;
  }
  return shell;
}

/** The first vertex row and the vertex count of a feature's shell ring. */
export function getShellRows(set: PolygonSet, feature: number): {first: number; count: number} {
  const ring = set.polygonOffsets[set.featureOffsets[feature]];
  const first = set.ringOffsets[ring];
  return {first, count: set.ringOffsets[ring + 1] - first - (set.closeRings ? 1 : 0)};
}

/**
 * Moves one shell vertex (and, with `spread`, its neighbors) toward `target` and rebuilds the
 * feature's outline and fill. A null drag leaves the set as created.
 */
export function applyVertexDrag(set: PolygonSet, drag: VertexDrag | null): void {
  if (!drag) return;
  set.positions.set(getDraggedShell(set, drag), getShellRows(set, drag.feature).first * 2);
  rebuildFeature(set, drag.feature);
}

/** Ring edges as vertex id pairs (uint32x2), closing edges included, for the segments layer. */
export function createRingEdges(ringOffsets: Uint32Array): Uint32Array {
  const edges: number[] = [];
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    const start = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = start; vertex < end; vertex++) {
      edges.push(vertex, vertex + 1 < end ? vertex + 1 : start);
    }
  }
  return Uint32Array.from(edges);
}

/** Consecutive vertex id pairs inside each line (uint32x2), for the segments layer. */
export function createLineEdges(lineOffsets: Uint32Array): Uint32Array {
  const edges: number[] = [];
  for (let line = 0; line + 1 < lineOffsets.length; line++) {
    for (let vertex = lineOffsets[line]; vertex + 1 < lineOffsets[line + 1]; vertex++) {
      edges.push(vertex, vertex + 1);
    }
  }
  return Uint32Array.from(edges);
}

export type PolygonBuffers = {
  positions: RunBuffer<'float32x2'>;
  featureOffsets: RunBuffer<'uint32'>;
  polygonOffsets: RunBuffer<'uint32'>;
  ringOffsets: RunBuffer<'uint32'>;
  fillVertices: RunBuffer<'float32x2'>;
  fillIndices: RunBuffer<'uint32'>;
  fillOwners: RunBuffer<'uint32'>;
  centers: RunBuffer<'float32x2'>;
  outline: RunBuffer<'float32x4'>;
  /** Ring edges over `positions`. */
  edges: RunBuffer<'uint32x2'>;
};

/** Uploads a polygon set: layout buffers for the operation, mesh buffers for drawing. */
export function uploadPolygonSet(builder: RunBuilder, set: PolygonSet): PolygonBuffers {
  return {
    positions: builder.upload('positions', 'float32x2', set.positions),
    featureOffsets: builder.upload('feature-offsets', 'uint32', set.featureOffsets),
    polygonOffsets: builder.upload('polygon-offsets', 'uint32', set.polygonOffsets),
    ringOffsets: builder.upload('ring-offsets', 'uint32', set.ringOffsets),
    fillVertices: builder.upload('fill-vertices', 'float32x2', set.fillVertices),
    fillIndices: builder.upload('fill-indices', 'uint32', set.fillIndices),
    fillOwners: builder.upload('fill-owners', 'uint32', set.fillOwners),
    centers: builder.upload('centers', 'float32x2', set.centers),
    outline: builder.upload('outline', 'float32x4', set.outline),
    edges: builder.upload('edges', 'uint32x2', createRingEdges(set.ringOffsets))
  };
}

export type LineSet = {
  count: number;
  positions: Float32Array;
  lineOffsets: Uint32Array;
  /** Segments (x0, y0, x1, y1) between consecutive vertices. */
  segments: Float32Array;
  segmentCount: number;
};

/** Wandering open polylines across the world, `vertexCount` vertices each. */
export function createLineSet(
  random: Random,
  world: World,
  lineCount = 4,
  vertexCount = 40,
  wobble = 0.35
): LineSet {
  const positions: number[] = [];
  const lineOffsets = [0];
  const segments: number[] = [];
  for (let line = 0; line < lineCount; line++) {
    const baseY = ((line + 0.5) / lineCount) * world.height;
    let previousX = 0;
    let previousY = 0;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const along = vertex / (vertexCount - 1);
      const x = (0.04 + 0.92 * along) * world.width;
      const y =
        baseY +
        Math.sin(along * 9 + line * 1.7) * 0.05 * world.height +
        (random() - 0.5) * 0.045 * wobble * world.height * 2;
      positions.push(x, y);
      if (vertex > 0) segments.push(previousX, previousY, x, y);
      previousX = x;
      previousY = y;
    }
    lineOffsets.push(positions.length / 2);
  }
  return {
    count: lineCount,
    positions: Float32Array.from(positions),
    lineOffsets: Uint32Array.from(lineOffsets),
    segments: Float32Array.from(segments),
    segmentCount: segments.length / 4
  };
}

/** Rewrites every buffer of an uploaded set from a regenerated set of the same size. */
export function writePolygonSet(buffers: PolygonBuffers, set: PolygonSet): void {
  buffers.positions.write(set.positions);
  buffers.fillVertices.write(set.fillVertices);
  buffers.fillIndices.write(set.fillIndices);
  buffers.centers.write(set.centers);
  buffers.outline.write(set.outline);
}

/** The ring row of every vertex row, for adapters that spread ring values onto vertices. */
export function createRingOfVertex(ringOffsets: Uint32Array): Uint32Array {
  const ringOfVertex = new Uint32Array(ringOffsets[ringOffsets.length - 1]);
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    ringOfVertex.fill(ring, ringOffsets[ring], ringOffsets[ring + 1]);
  }
  return ringOfVertex;
}

/** The feature row of every vertex row (features own polygons own rings own vertices). */
export function createFeatureOfVertex(set: PolygonSet): Uint32Array {
  const featureOfVertex = new Uint32Array(set.positions.length / 2);
  for (let feature = 0; feature < set.count; feature++) {
    const firstRing = set.polygonOffsets[set.featureOffsets[feature]];
    const lastRing = set.polygonOffsets[set.featureOffsets[feature + 1]];
    featureOfVertex.fill(feature, set.ringOffsets[firstRing], set.ringOffsets[lastRing]);
  }
  return featureOfVertex;
}

/** Row numbers 0..count-1, for the `paths` layer over a packed vertex buffer. */
export function createIdentityIds(count: number): Uint32Array {
  return Uint32Array.from({length: count}, (_, index) => index);
}

/** Adds seeded jitter (world units) to every coordinate. */
export function addJitter(positions: Float32Array, random: Random, amount: number): Float32Array {
  return Float32Array.from(positions, value => value + (random() - 0.5) * 2 * amount);
}

export type SingleLine = {positions: Float32Array; lineOffsets: Uint32Array; count: number};

/** A sinuous river bend, smoothed with a Catmull-Rom spline and fitted to the canvas. */
export function createSingleLine(_random: Random, world: World, vertexCount = 80): SingleLine {
  const control = createRiverBend(1, 12, world);
  const positions = new Float32Array(vertexCount * 2);
  const segmentCount = control.length - 1;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const scaled = (vertex / (vertexCount - 1)) * segmentCount;
    const segment = Math.min(segmentCount - 1, Math.floor(scaled));
    const t = scaled - segment;
    const p0 = control[Math.max(0, segment - 1)];
    const p1 = control[segment];
    const p2 = control[segment + 1];
    const p3 = control[Math.min(control.length - 1, segment + 2)];
    for (let axis = 0; axis < 2; axis++) {
      positions[vertex * 2 + axis] =
        0.5 *
        (2 * p1[axis] +
          (p2[axis] - p0[axis]) * t +
          (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * t * t +
          (3 * p1[axis] - p0[axis] - 3 * p2[axis] + p3[axis]) * t * t * t);
    }
  }
  return {positions, lineOffsets: Uint32Array.of(0, vertexCount), count: vertexCount};
}

export type CrossingLines = {
  /** Horizontal-ish lines first, then vertical-ish lines, in one packed layout. */
  positions: Float32Array;
  lineOffsets: Uint32Array;
  lineCount: number;
  vertexCount: number;
  /** Every segment (x0, y0, x1, y1): the first group, then the second. */
  segments: Float32Array;
  /** Segments of the first group (x0, y0, x1, y1) and of the second group, views of `segments`. */
  firstSegments: Float32Array;
  secondSegments: Float32Array;
  /** Vertex rows of each group inside `positions`. */
  firstVertexCount: number;
  firstLineCount: number;
  /** The cross-track coordinate each line is laid out around (y for the first group, x for the second). */
  lineBases: Float32Array;
  /**
   * Shifts the chosen lines across their direction (y for the first group, x for the second) and
   * rewrites `positions` and `segments` in place; the vertex and segment counts never change.
   */
  regenerate(shifts: ReadonlyArray<{line: number; offset: number}>): void;
};

/** Two groups of wandering lines that cross each other: west-east and south-north. */
export function createCrossingLines(
  random: Random,
  world: World,
  firstCount = 3,
  secondCount = 4,
  vertexCount = 12,
  wobble = 0.06
): CrossingLines {
  const positions: number[] = [];
  const lineOffsets = [0];
  const lineBases: number[] = [];
  const addLine = (isFirst: boolean, index: number, count: number) => {
    const reach = createRiverLine(Math.floor(random() * RIVER_REACH_COUNT), vertexCount, {
      width: 1,
      height: 1
    });
    const eastings = reach.map(([x]) => x);
    const northings = reach.map(([, y]) => y);
    const west = Math.min(...eastings);
    const south = Math.min(...northings);
    const width = Math.max(...eastings) - west;
    const height = Math.max(...northings) - south;
    const base = (index + 0.5) / count;
    for (const [x, y] of reach) {
      // Normalize the span to guarantee intersections; wobble adjusts only the cross-track scale.
      const along = 0.03 + (0.94 * (x - west)) / width;
      const across = height > 1e-8 ? base + ((y - south) / height - 0.5) * 2 * wobble : base;
      if (isFirst) positions.push(along * world.width, across * world.height);
      else positions.push(across * world.width, along * world.height);
    }
    lineBases.push(base * (isFirst ? world.height : world.width));
    lineOffsets.push(positions.length / 2);
  };
  for (let index = 0; index < firstCount; index++) addLine(true, index, firstCount);
  for (let index = 0; index < secondCount; index++) addLine(false, index, secondCount);
  const basePositions = Float32Array.from(positions);
  const lines = firstCount + secondCount;
  const segmentCount = basePositions.length / 2 - lines;
  const segments = new Float32Array(segmentCount * 4);
  const firstSegmentCount = firstCount * (vertexCount - 1);
  const crossing: CrossingLines = {
    positions: Float32Array.from(basePositions),
    lineOffsets: Uint32Array.from(lineOffsets),
    lineCount: lines,
    vertexCount: basePositions.length / 2,
    segments,
    firstSegments: segments.subarray(0, firstSegmentCount * 4),
    secondSegments: segments.subarray(firstSegmentCount * 4),
    firstVertexCount: firstCount * vertexCount,
    firstLineCount: firstCount,
    lineBases: Float32Array.from(lineBases),
    regenerate(shifts) {
      crossing.positions.set(basePositions);
      for (const {line, offset} of shifts) {
        const axis = line < firstCount ? 1 : 0;
        for (let vertex = lineOffsets[line]; vertex < lineOffsets[line + 1]; vertex++) {
          crossing.positions[vertex * 2 + axis] += offset;
        }
      }
      let row = 0;
      for (let line = 0; line < lines; line++) {
        for (let vertex = lineOffsets[line]; vertex + 1 < lineOffsets[line + 1]; vertex++) {
          segments.set(crossing.positions.subarray(vertex * 2, vertex * 2 + 4), row++ * 4);
        }
      }
    }
  };
  crossing.regenerate([]);
  return crossing;
}

/**
 * The west-east and the south-north line nearest to (x, y), each carried to the pointer:
 * returns the shifts for `regenerate`. Keeps the previous choice unless another is clearly nearer.
 */
export function pickCrossingLines(
  lines: CrossingLines,
  x: number,
  y: number,
  previous: ReadonlyArray<{line: number}> = []
): Array<{line: number; offset: number}> {
  const pick = (first: number, last: number, coordinate: number, held: number | undefined) => {
    let best = first;
    for (let line = first; line < last; line++) {
      if (
        Math.abs(lines.lineBases[line] - coordinate) < Math.abs(lines.lineBases[best] - coordinate)
      )
        best = line;
    }
    if (
      held !== undefined &&
      held >= first &&
      held < last &&
      Math.abs(lines.lineBases[best] - coordinate) >
        Math.abs(lines.lineBases[held] - coordinate) * 0.85
    ) {
      best = held;
    }
    return {line: best, offset: coordinate - lines.lineBases[best]};
  };
  const total = lines.lineCount;
  const split = lines.firstLineCount;
  const heldFirst = previous.find(entry => entry.line < split)?.line;
  const heldSecond = previous.find(entry => entry.line >= split)?.line;
  return [pick(0, split, y, heldFirst), pick(split, total, x, heldSecond)];
}

/**
 * Rewrites every ring with a different start vertex and, optionally, the opposite winding. The
 * ring layout (offsets) is unchanged, so the result pairs with the set's own offsets.
 */
export function restartRings(
  set: PolygonSet,
  getStart: (ring: number) => number,
  shouldReverse: (ring: number) => boolean = () => false
): Float32Array {
  const positions = new Float32Array(set.positions.length);
  for (let ring = 0; ring + 1 < set.ringOffsets.length; ring++) {
    const first = set.ringOffsets[ring];
    const length = set.ringOffsets[ring + 1] - first;
    const start = getStart(ring);
    const direction = shouldReverse(ring) ? -1 : 1;
    for (let vertex = 0; vertex < length; vertex++) {
      const source = first + ((((start + direction * vertex) % length) + length) % length);
      positions[(first + vertex) * 2] = set.positions[source * 2];
      positions[(first + vertex) * 2 + 1] = set.positions[source * 2 + 1];
    }
  }
  return positions;
}

export type DefectKind = 'bow-tie' | 'hole-outside' | 'repeated-vertex';

/**
 * Copies the set's positions and damages the first `count` features, cycling through `kinds`:
 * a bow-tie swaps two shell vertices, a stray hole moves the hole out of its shell, a repeated
 * vertex duplicates a shell vertex. Rings must be those of `createPolygonSet` with holes.
 */
export function injectDefects(
  set: PolygonSet,
  count: number,
  kinds: readonly DefectKind[]
): Float32Array {
  const positions = Float32Array.from(set.positions);
  for (let feature = 0; feature < Math.min(count, set.count); feature++) {
    const shell = set.polygonOffsets[set.featureOffsets[feature]];
    const first = set.ringOffsets[shell];
    const kind = kinds[feature % kinds.length];
    if (kind === 'bow-tie') {
      for (const axis of [0, 1]) {
        const swap = positions[(first + 2) * 2 + axis];
        positions[(first + 2) * 2 + axis] = positions[(first + 3) * 2 + axis];
        positions[(first + 3) * 2 + axis] = swap;
      }
    } else if (kind === 'repeated-vertex') {
      positions[(first + 5) * 2] = positions[(first + 4) * 2];
      positions[(first + 5) * 2 + 1] = positions[(first + 4) * 2 + 1];
    } else if (shell + 1 < set.ringOffsets.length - 1) {
      for (let vertex = set.ringOffsets[shell + 1]; vertex < set.ringOffsets[shell + 2]; vertex++) {
        positions[vertex * 2] += 0.16;
      }
    }
  }
  return positions;
}
