// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared pieces of the weights runs: output CSR buffers, a kernel that turns a CSR into drawable
// link segments, and a tract boundaries with controlled registration errors.

import type {
  GPUChunkedSpatialWeights,
  GPUSpatialWeights,
  GPUSpatialWeightsOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';
import {createTracts, type CsrWeights} from '../../../inputs';
import type {Point, RunGlyph, RunPeek} from '../../../interaction';
import {addKernelPass} from './kernel';

/** A CSR the operation writes, with every buffer kept for drawing. */
export type WeightsBuffers = {
  offsets: RunBuffer<'uint32'>;
  neighbors: RunBuffer<'uint32'>;
  weights: RunBuffer<'float32'>;
  distances?: RunBuffer<'float32'>;
  overflow: RunBuffer<'uint32'>;
  rowCount: number;
  capacity: number;
  /** The weights as the producing operation takes them as output. */
  port: GPUSpatialWeightsOutput;
};

/** Allocates output CSR buffers of `rowCount` rows and `capacity` slots. */
export function allocateWeights(
  builder: RunBuilder,
  id: string,
  rowCount: number,
  capacity: number,
  hasDistances = false
): WeightsBuffers {
  const offsets = builder.allocate(`${id}-offsets`, 'uint32', rowCount + 1);
  const neighbors = builder.allocate(`${id}-neighbors`, 'uint32', capacity);
  const weights = builder.allocate(`${id}-weights`, 'float32', capacity);
  const distances = hasDistances
    ? builder.allocate(`${id}-distances`, 'float32', capacity)
    : undefined;
  const overflow = builder.allocate(`${id}-overflow`, 'uint32', 1);
  return {
    offsets,
    neighbors,
    weights,
    distances,
    overflow,
    rowCount,
    capacity,
    port: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      distances: distances?.view,
      status: {overflow: overflow.view}
    }
  };
}

/** The buffers of a CSR as the weights input of the next operation. */
export function getWeightsInput(weights: WeightsBuffers): GPUSpatialWeights {
  return {
    offsets: weights.offsets.view,
    neighbors: weights.neighbors.view,
    weights: weights.weights.view,
    distances: weights.distances?.view
  };
}

const getLinkBody = (filterCondition: string) => /* wgsl */ `
  let total = offsets[offsetsOffset + ROW_COUNT];
  if (index >= total${filterCondition}) {
    segments[segmentsOffset + index * 4u] = -10.0;
    segments[segmentsOffset + index * 4u + 1u] = -10.0;
    segments[segmentsOffset + index * 4u + 2u] = -10.0;
    segments[segmentsOffset + index * 4u + 3u] = -10.0;
    return;
  }
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + (ROW_START + low) * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + (ROW_START + low) * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
`;

/** Adds a kernel that writes one segment per CSR slot, from row position to neighbor position. */
export function addLinkSegments(
  builder: RunBuilder,
  id: string,
  weights: Pick<WeightsBuffers, 'offsets' | 'neighbors' | 'rowCount' | 'capacity'>,
  positions: RunBuffer<'float32x2'>,
  options: {onlyWhere?: RunBuffer<'uint32'>; rowStart?: number} = {}
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-segments`, 'float32x4', weights.capacity);
  addKernelPass(builder, {
    id: `${id}-link-segments`,
    invocationCount: weights.capacity,
    bindings: [
      {name: 'offsets', buffer: weights.offsets, access: 'read'},
      {name: 'neighbors', buffer: weights.neighbors, access: 'read'},
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'},
      ...(options.onlyWhere
        ? [{name: 'onlyWhere', buffer: options.onlyWhere, access: 'read' as const}]
        : [])
    ],
    declarations: `const ROW_COUNT: u32 = ${weights.rowCount}u;\nconst ROW_START: u32 = ${options.rowStart ?? 0}u;`,
    body: getLinkBody(options.onlyWhere ? ' || onlyWhere[onlyWhereOffset + index] == 0u' : '')
  });
  return segments;
}

const CARDINALITY_BODY = /* wgsl */ `
  cardinality[cardinalityOffset + index] = f32(offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index]);
`;

/** Adds a kernel that writes the neighbor count of every row (float32, for coloring polygons). */
export function addCardinality(
  builder: RunBuilder,
  id: string,
  weights: Pick<WeightsBuffers, 'offsets' | 'rowCount'>
): RunBuffer<'float32'> {
  const cardinality = builder.allocate(`${id}-cardinality`, 'float32', weights.rowCount);
  addKernelPass(builder, {
    id: `${id}-cardinality`,
    invocationCount: weights.rowCount,
    bindings: [
      {name: 'offsets', buffer: weights.offsets, access: 'read'},
      {name: 'cardinality', buffer: cardinality, access: 'read_write'}
    ],
    body: CARDINALITY_BODY
  });
  return cardinality;
}

/** The mesh buffers of a tile layout, uploaded for the polygons layer. */
export function uploadTileMesh(builder: RunBuilder, tiles: TileLayout) {
  return {
    vertices: builder.upload('tile-vertices', 'float32x2', tiles.meshVertices),
    indices: builder.upload('tile-indices', 'uint32', tiles.triangleIndices),
    owners: builder.upload('tile-owners', 'uint32', tiles.triangleOwners),
    centers: builder.upload('tile-centers', 'float32x2', tiles.centers),
    outline: builder.upload('tile-outline', 'float32x4', tiles.outline)
  };
}

/** The layer that draws link segments written by {@link addLinkSegments}. */
export function drawLinks(
  segments: RunBuffer<'float32x4'>,
  color: readonly [number, number, number, number] = [0.95, 0.55, 0.2, 0.6]
): RunLayer {
  return {layer: 'lines', segments, color, radius: 1.1};
}

/** Polygons as the four-level layout the polygon operations take, plus a drawing mesh. */
export type TileLayout = {
  count: number;
  /** Ring corners, (x, y) pairs, each ring closed. */
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  /** Tile centers, (x, y) pairs. */
  centers: Float32Array;
  /** Mesh for drawing: vertices (x, y), three per triangle id list, owner tile per triangle. */
  meshVertices: Float32Array;
  triangleIndices: Uint32Array;
  triangleOwners: Uint32Array;
  /** Outline segments (x0, y0, x1, y1). */
  outline: Float32Array;
};

/** Apply a uniform contraction about each Census tract center to model digitization slivers. */
export function createTractTiles(
  random: () => number,
  world: {width: number; height: number},
  inset: number
): TileLayout {
  const tracts = createTracts(random, world, 6);
  const positions: number[] = [];
  const meshVertices = tracts.vertices.slice();
  const outline: number[] = [];
  const ringOffsets = [0];
  for (let tract = 0; tract < tracts.count; tract++) {
    const start = tracts.ringOffsets[tract];
    const end = tracts.ringOffsets[tract + 1];
    const centerX = tracts.centers[tract * 2];
    const centerY = tracts.centers[tract * 2 + 1];
    for (let vertex = start; vertex < end; vertex++) {
      meshVertices[vertex * 2] += (centerX - meshVertices[vertex * 2]) * inset;
      meshVertices[vertex * 2 + 1] += (centerY - meshVertices[vertex * 2 + 1]) * inset;
    }
    for (let vertex = start; vertex < end; vertex++) {
      const next = vertex + 1 === end ? start : vertex + 1;
      const x = meshVertices[vertex * 2];
      const y = meshVertices[vertex * 2 + 1];
      positions.push(x, y);
      outline.push(x, y, meshVertices[next * 2], meshVertices[next * 2 + 1]);
    }
    positions.push(meshVertices[start * 2], meshVertices[start * 2 + 1]);
    ringOffsets.push(positions.length / 2);
  }
  return {
    count: tracts.count,
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from({length: tracts.count + 1}, (_, row) => row),
    featureOffsets: Uint32Array.from({length: tracts.count + 1}, (_, row) => row),
    centers: tracts.centers,
    meshVertices,
    triangleIndices: tracts.triangleIndices,
    triangleOwners: tracts.triangleOwners,
    outline: Float32Array.from(outline)
  };
}

/** A small seeded random generator (mulberry32), to rebuild the same layout with other data. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

const ROW_SUM_BODY = /* wgsl */ `
  var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += slotValues[slotValuesOffset + slot];
  }
  sums[sumsOffset + index] = sum;
`;

/** Adds a kernel that sums a per-slot float value over every row of a CSR. */
export function addRowSums(
  builder: RunBuilder,
  id: string,
  weights: Pick<WeightsBuffers, 'offsets' | 'rowCount'>,
  slotValues: RunBuffer<'float32'>
): RunBuffer<'float32'> {
  const sums = builder.allocate(`${id}-row-sums`, 'float32', weights.rowCount);
  addKernelPass(builder, {
    id: `${id}-row-sums`,
    invocationCount: weights.rowCount,
    bindings: [
      {name: 'offsets', buffer: weights.offsets, access: 'read'},
      {name: 'slotValues', buffer: slotValues, access: 'read'},
      {name: 'sums', buffer: sums, access: 'read_write'}
    ],
    body: ROW_SUM_BODY
  });
  return sums;
}

/** Adds a kernel that copies column `column` of a row-major float table into its own buffer. */
export function addTableColumn(
  builder: RunBuilder,
  id: string,
  table: RunBuffer<'float32'>,
  columnCount: number,
  column: number,
  rowCount: number
): RunBuffer<'float32'> {
  const values = builder.allocate(`${id}-column-${column}`, 'float32', rowCount);
  addKernelPass(builder, {
    id: `${id}-column-${column}`,
    invocationCount: rowCount,
    bindings: [
      {name: 'table', buffer: table, access: 'read'},
      {name: 'values', buffer: values, access: 'read_write'}
    ],
    body: `values[valuesOffset + index] = table[tableOffset + index * ${columnCount}u + ${column}u];`
  });
  return values;
}

/** Uploads a CPU CSR as input weights, keeping the buffers for drawing. */
export function uploadWeightsBuffers(
  builder: RunBuilder,
  id: string,
  csr: CsrWeights
): Omit<WeightsBuffers, 'overflow' | 'port'> & {input: GPUSpatialWeights} {
  const capacity = Math.max(csr.neighbors.length, 1);
  const pad = <Array extends Float32Array | Uint32Array>(values: Array): Array => {
    const padded = new (values.constructor as new (length: number) => Array)(capacity);
    padded.set(values);
    return padded;
  };
  const offsets = builder.upload(`${id}-offsets`, 'uint32', csr.offsets);
  const neighbors = builder.upload(`${id}-neighbors`, 'uint32', pad(csr.neighbors));
  const weights = builder.upload(`${id}-weights`, 'float32', pad(csr.weights));
  const distances = builder.upload(`${id}-distances`, 'float32', pad(csr.distances));
  return {
    offsets,
    neighbors,
    weights,
    distances,
    rowCount: csr.rowCount,
    capacity,
    input: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      distances: distances.view,
      information: {
        rowCount: csr.rowCount,
        transform: 'raw',
        symmetric: 'unknown',
        provenance: {source: 'upload'}
      }
    }
  };
}

/** Adds a kernel that turns uint32 counts into `log(1 + count)` floats, so bars of them stay readable. */
export function addLogCounts(
  builder: RunBuilder,
  id: string,
  counts: RunBuffer<'uint32'> | RunBuffer<'float32'>
): RunBuffer<'float32'> {
  const logged = builder.allocate(`${id}-log-counts`, 'float32', counts.length);
  addKernelPass(builder, {
    id: `${id}-log-counts`,
    invocationCount: counts.length,
    bindings: [
      {name: 'counts', buffer: counts, access: 'read'},
      {name: 'logged', buffer: logged, access: 'read_write'}
    ],
    body: 'logged[loggedOffset + index] = log(1.0 + f32(counts[countsOffset + index]));'
  });
  return logged;
}

/** One chunk of uploaded chunked weights, with its buffers for drawing. */
export type ChunkBuffers = Omit<WeightsBuffers, 'overflow' | 'port'> & {
  rowStart: number;
  input: GPUSpatialWeights;
};

/** Slices a CSR into chunks of consecutive rows and uploads each one as its own CSR. */
export function uploadChunkedWeights(
  builder: RunBuilder,
  id: string,
  csr: CsrWeights,
  chunkRowCounts: readonly number[]
): {chunks: ChunkBuffers[]; input: GPUChunkedSpatialWeights} {
  let rowStart = 0;
  const chunks = chunkRowCounts.map((rowCount, index) => {
    const begin = csr.offsets[rowStart];
    const end = csr.offsets[rowStart + rowCount];
    const slice: CsrWeights = {
      rowCount,
      offsets: Uint32Array.from(
        csr.offsets.subarray(rowStart, rowStart + rowCount + 1),
        value => value - begin
      ),
      neighbors: csr.neighbors.slice(begin, end),
      weights: csr.weights.slice(begin, end),
      distances: csr.distances.slice(begin, end)
    };
    const chunk = {...uploadWeightsBuffers(builder, `${id}-chunk${index}`, slice), rowStart};
    rowStart += rowCount;
    return chunk;
  });
  return {
    chunks,
    input: {
      rowCount: csr.rowCount,
      information: {rowCount: csr.rowCount, transform: 'raw', symmetric: 'unknown'},
      chunks: chunks.map(chunk => ({
        rowStart: chunk.rowStart,
        rowCount: chunk.rowCount,
        weights: chunk.input
      }))
    }
  };
}

/** Splits `rowCount` rows into `chunkCount` nearly equal consecutive chunks. */
export function getChunkRowCounts(rowCount: number, chunkCount: number): number[] {
  const base = Math.floor(rowCount / chunkCount);
  return Array.from({length: chunkCount}, (_, index) =>
    index === chunkCount - 1 ? rowCount - base * (chunkCount - 1) : base
  );
}

/** Sorts points by x, so consecutive row chunks are vertical bands of the map. */
export function sortPointsByX(positions: Float32Array, values: Float32Array) {
  const order = Array.from({length: values.length}, (_, index) => index).sort(
    (first, second) => positions[first * 2] - positions[second * 2]
  );
  return {
    positions: Float32Array.from(
      order.flatMap(index => [positions[index * 2], positions[index * 2 + 1]])
    ),
    values: Float32Array.from(order, index => values[index])
  };
}

/** The neighbor ids of one row of a CSR the operation wrote, or null until its readback arrives. */
export function peekRow(
  peek: RunPeek,
  weights: Pick<WeightsBuffers, 'offsets' | 'neighbors'>,
  row: number
): Uint32Array | null {
  const offsets = peek(weights.offsets);
  const neighbors = peek(weights.neighbors);
  if (!offsets || !neighbors || row < 0 || row + 1 >= offsets.length) return null;
  return (neighbors as Uint32Array).subarray(offsets[row], offsets[row + 1]);
}

/** Link lines from a focus row to its neighbors, with a dot on each neighbor. */
export function getRowLinkGlyphs(
  from: Point,
  targets: readonly Point[],
  options: {dashed?: boolean; dots?: boolean} = {}
): RunGlyph[] {
  return targets.flatMap(to => [
    {shape: 'line' as const, from, to, dashed: options.dashed},
    ...(options.dots === false ? [] : [{shape: 'dot' as const, at: to}])
  ]);
}

/** The positions of the given rows out of packed (x, y) pairs. */
export function getRowPositions(positions: ArrayLike<number>, rows: ArrayLike<number>): Point[] {
  return Array.from(rows, row => [positions[row * 2], positions[row * 2 + 1]] as const);
}

/** The ring of one polygon row as a closed outline glyph (rings in packed (x, y) pairs). */
export function getRingGlyph(
  positions: Float32Array,
  ringOffsets: Uint32Array,
  row: number,
  dashed = false
): RunGlyph {
  return {
    shape: 'polyline',
    points: positions.subarray(ringOffsets[row] * 2, ringOffsets[row + 1] * 2),
    closed: true,
    dashed
  };
}

/** The far ends of the link segments (x0, y0, x1, y1) that start at `from`, or end there. */
export function getSegmentTargets(
  segments: ArrayLike<number> | null,
  from: Point,
  end: 'start' | 'end' = 'start'
): Point[] {
  const targets: Point[] = [];
  if (!segments) return targets;
  const near = end === 'start' ? 0 : 2;
  const far = 2 - near;
  for (let slot = 0; slot + 3 < segments.length; slot += 4) {
    if (segments[slot + near] === from[0] && segments[slot + near + 1] === from[1]) {
      targets.push([segments[slot + far], segments[slot + far + 1]]);
    }
  }
  return targets;
}

/** Vertical rules (one float32x4 segment each) between chunks of x-sorted rows, for a `lines` layer. */
export function uploadChunkBoundaries(
  builder: RunBuilder,
  positions: Float32Array,
  chunkStarts: readonly number[]
): RunBuffer<'float32x4'> {
  const rules = chunkStarts.slice(1).flatMap(start => {
    const x = (positions[start * 2 - 2] + positions[start * 2]) / 2;
    return [x, 0, x, 1];
  });
  return builder.upload('chunk-boundaries', 'float32x4', Float32Array.of(...rules, 0, 0, 0, 0));
}

/** One-based chunk of a row, given the first row of every chunk. */
export function getChunkNumber(chunkStarts: readonly number[], row: number): number {
  return chunkStarts.filter(start => start <= row).length;
}

/** Rescales points so they fill the drawing world (with a margin), keeping their layout. */
export function stretchToWorld(
  positions: Float32Array,
  world: {width: number; height: number},
  margin = 0.06
): Float32Array {
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minimumX = Math.min(minimumX, positions[index]);
    maximumX = Math.max(maximumX, positions[index]);
    minimumY = Math.min(minimumY, positions[index + 1]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  const scaleX = (world.width - 2 * margin) / Math.max(maximumX - minimumX, 1e-9);
  const scaleY = (world.height - 2 * margin) / Math.max(maximumY - minimumY, 1e-9);
  return Float32Array.from(positions, (value, index) =>
    index % 2 === 0 ? margin + (value - minimumX) * scaleX : margin + (value - minimumY) * scaleY
  );
}

/** The same points with their positions stretched to fill the drawing world. */
export function stretchPoints<Points extends {positions: Float32Array}>(
  points: Points,
  world: {width: number; height: number},
  margin?: number
): Points {
  return {...points, positions: stretchToWorld(points.positions, world, margin)};
}
