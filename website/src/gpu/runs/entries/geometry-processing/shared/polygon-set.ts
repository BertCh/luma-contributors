// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded polygon and line fixtures shared by the geometry-processing runs: star-shaped blobs on a
// jittered grid in the drawing world, packed in the four-level polygon layout the operations read
// (features own polygons own rings own vertices; rings are open, the closing edge is implicit).

import type {RunBuffer, RunBuilder} from '../../../define-run';

export type World = {width: number; height: number};
export type Random = () => number;

export type PolygonSetOptions = {
  columns?: number;
  rows?: number;
  /** Vertices of each shell ring. */
  vertexCount?: number;
  /** 0 is a circle, 1 is a spiky star. */
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
};

export type PolygonSet = {
  count: number;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Shell centers (x, y). */
  centers: Float32Array;
  /** Fill mesh: shell vertices plus a center vertex per polygon, as triangle fans. */
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
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const centers: number[] = [];
  const fillVertices: number[] = [];
  const fillIndices: number[] = [];
  const fillOwners: number[] = [];
  const outline: number[] = [];
  const addRing = (ring: number[][]) => {
    for (const [x, y] of options.closeRings ? [...ring, ring[0]] : ring) positions.push(x, y);
    ringOffsets.push(positions.length / 2);
    for (let index = 0; index < ring.length; index++) {
      const next = ring[(index + 1) % ring.length];
      outline.push(ring[index][0], ring[index][1], next[0], next[1]);
    }
  };
  let featureIndex = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const centerX = (column + 0.5 + (random() - 0.5) * 0.12) * cellWidth;
      const centerY = (row + 0.5 + (random() - 0.5) * 0.12) * cellHeight;
      const radius = Math.min(cellWidth, cellHeight) * 0.5 * fill * (0.85 + random() * 0.3);
      const phase = random() * Math.PI * 2;
      const squash = 1 - (options.stretch ?? 0) * random() * 0.85;
      const shell: number[][] = [];
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        const angle = (vertex / vertexCount) * Math.PI * 2;
        const spike = vertex % 2 === 0 ? 1 : 1 - irregularity * (0.5 + random() * 0.5);
        const localX = Math.cos(angle) * radius * spike;
        const localY =
          Math.sin(angle) * radius * spike * squash * (options.bend ? 0.45 : 1) +
          (((options.bend ?? 0) * (localX * localX)) / radius) * 1.4;
        shell.push([
          centerX + localX * Math.cos(phase) - localY * Math.sin(phase),
          centerY + localX * Math.sin(phase) + localY * Math.cos(phase)
        ]);
      }
      const isReversed = Boolean(options.mixedWinding) && featureIndex % 2 === 1;
      addRing(isReversed ? [...shell].reverse() : shell);
      if (options.hasHoles) {
        const hole: number[][] = [];
        for (let vertex = 0; vertex < 8; vertex++) {
          const angle = -(vertex / 8) * Math.PI * 2;
          hole.push([
            centerX + Math.cos(angle) * radius * 0.3,
            centerY + Math.sin(angle) * radius * 0.3
          ]);
        }
        addRing(isReversed ? [...hole].reverse() : hole);
      }
      polygonOffsets.push(ringOffsets.length - 1);
      featureOffsets.push(polygonOffsets.length - 1);
      centers.push(centerX, centerY);
      const base = fillVertices.length / 2;
      for (const [x, y] of shell) fillVertices.push(x, y);
      fillVertices.push(centerX, centerY);
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        fillIndices.push(base + vertex, base + ((vertex + 1) % vertexCount), base + vertexCount);
        fillOwners.push(featureIndex);
      }
      featureIndex++;
    }
  }
  return {
    count: featureIndex,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    centers: Float32Array.from(centers),
    fillVertices: Float32Array.from(fillVertices),
    fillIndices: Uint32Array.from(fillIndices),
    fillOwners: Uint32Array.from(fillOwners),
    fillTriangleCount: fillIndices.length / 3,
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4
  };
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

/** One smooth wandering polyline of `vertexCount` vertices across the world. */
export function createSingleLine(random: Random, world: World, vertexCount = 80): SingleLine {
  const positions = new Float32Array(vertexCount * 2);
  const phase = random() * 6;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const along = vertex / (vertexCount - 1);
    positions[vertex * 2] = (0.05 + 0.9 * along) * world.width;
    positions[vertex * 2 + 1] =
      world.height *
      (0.5 + 0.28 * Math.sin(along * 7 + phase) + 0.08 * Math.sin(along * 23 + phase * 2));
  }
  return {positions, lineOffsets: Uint32Array.of(0, vertexCount), count: vertexCount};
}

export type CrossingLines = {
  /** Horizontal-ish lines first, then vertical-ish lines, in one packed layout. */
  positions: Float32Array;
  lineOffsets: Uint32Array;
  lineCount: number;
  vertexCount: number;
  /** Segments of the first group (x0, y0, x1, y1) and of the second group. */
  firstSegments: Float32Array;
  secondSegments: Float32Array;
  /** Vertex rows of each group inside `positions`. */
  firstVertexCount: number;
  firstLineCount: number;
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
  const firstSegments: number[] = [];
  const secondSegments: number[] = [];
  const addLine = (isFirst: boolean, index: number, count: number) => {
    const phase = random() * 6;
    const base = (index + 0.5) / count;
    let previous: number[] | null = null;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const along = vertex / (vertexCount - 1);
      const across = base + Math.sin(along * 5 + phase) * wobble + (random() - 0.5) * wobble * 0.7;
      const point = isFirst
        ? [(0.03 + 0.94 * along) * world.width, across * world.height]
        : [across * world.width, (0.03 + 0.94 * along) * world.height];
      positions.push(point[0], point[1]);
      if (previous) (isFirst ? firstSegments : secondSegments).push(...previous, ...point);
      previous = point;
    }
    lineOffsets.push(positions.length / 2);
  };
  for (let index = 0; index < firstCount; index++) addLine(true, index, firstCount);
  for (let index = 0; index < secondCount; index++) addLine(false, index, secondCount);
  return {
    positions: Float32Array.from(positions),
    lineOffsets: Uint32Array.from(lineOffsets),
    lineCount: firstCount + secondCount,
    vertexCount: positions.length / 2,
    firstSegments: Float32Array.from(firstSegments),
    secondSegments: Float32Array.from(secondSegments),
    firstVertexCount: firstCount * vertexCount,
    firstLineCount: firstCount
  };
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
