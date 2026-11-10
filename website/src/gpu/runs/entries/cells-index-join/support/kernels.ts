// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Adapter kernels that turn an operation's output rows into buffers the 2D layers can draw: unit
// vectors to longitude/latitude, integer columns to floats, join pairs and polygon rings to line
// segments. They are extra nodes in the same command graph, so the picture still reads the buffers
// the operation wrote on the GPU.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/** Where parked (invalid) segment endpoints go: far outside any view. */
const FAR_AWAY = 1e6;
const DEGREES_PER_RADIAN = 180 / Math.PI;

type KernelBinding = {name: string; buffer: RunBuffer; access: 'read' | 'read_write'};

function getElementType(buffer: RunBuffer): 'f32' | 'u32' | 'i32' {
  if (buffer.format.startsWith('uint32')) return 'u32';
  if (buffer.format.startsWith('sint32')) return 'i32';
  return 'f32';
}

/** Adds one linear WGSL kernel node to the run's graph; `body` runs once per `index`. */
export function addKernel(
  builder: RunBuilder,
  options: {id: string; invocationCount: number; bindings: KernelBinding[]; body: string}
): void {
  builder.graph.add({
    getCommandNodes: <Parameters>(
      graph: GPUCommandGraph<Parameters>
    ): GPUCommandNode<Parameters>[] => [
      createWGSLKernelNode<Parameters>(graph, {
        id: options.id,
        operation: 'CellsRunAdapter',
        bindings: options.bindings.map(binding => ({
          name: binding.name,
          view: binding.buffer.view,
          type: getElementType(binding.buffer),
          access: binding.access
        })),
        invocationCount: options.invocationCount,
        body: options.body
      })
    ]
  });
}

/** Unit-sphere (x, y, z) rows to longitude/latitude degrees (float32x2). Zero rows stay at 0, 0. */
export function addUnitVectorsToLonLat(
  builder: RunBuilder,
  id: string,
  vectors: RunBuffer<'float32x3'>
): RunBuffer<'float32x2'> {
  const positions = builder.allocate(`${id}-lonlat`, 'float32x2', vectors.length);
  addKernel(builder, {
    id: `${id}-unit-to-lonlat`,
    invocationCount: vectors.length,
    bindings: [
      {name: 'vectors', buffer: vectors, access: 'read'},
      {name: 'positions', buffer: positions, access: 'read_write'}
    ],
    body: `let base = vectorsOffset + index * 3u;
  let vector = vec3<f32>(vectors[base], vectors[base + 1u], vectors[base + 2u]);
  let outBase = positionsOffset + index * 2u;
  positions[outBase] = atan2(vector.y, vector.x) * ${DEGREES_PER_RADIAN};
  positions[outBase + 1u] = asin(clamp(vector.z, -1.0, 1.0)) * ${DEGREES_PER_RADIAN};`
  });
  return positions;
}

/** Integer rows to floats (a uint32 column drawn as a number instead of one of 8 categories). */
export function addUintToFloat(
  builder: RunBuilder,
  id: string,
  values: RunBuffer<'uint32'>,
  rowCount = values.length
): RunBuffer<'float32'> {
  const result = builder.allocate(`${id}-float`, 'float32', rowCount);
  addKernel(builder, {
    id: `${id}-uint-to-float`,
    invocationCount: rowCount,
    bindings: [
      {name: 'values', buffer: values, access: 'read'},
      {name: 'result', buffer: result, access: 'read_write'}
    ],
    body: `let value = values[valuesOffset + index];
  result[resultOffset + index] = select(f32(value), 1.0e30, value == 0xffffffffu);`
  });
  return result;
}

/**
 * Closed outlines of polygons stored `capacity` vertex slots apiece: slot `i` of group `g` joins
 * vertex `i` to vertex `i + 1` (wrapping) while `i < counts[g]`; other slots are parked far away.
 * `counts` rows are `uint32` (one count per group) or `uint32x2` (written, required).
 */
export function addGroupedRingSegments(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  counts: RunBuffer<'uint32' | 'uint32x2'>,
  capacity: number
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-ring-segments`, 'float32x4', positions.length);
  const countStride = counts.format === 'uint32x2' ? 2 : 1;
  addKernel(builder, {
    id: `${id}-ring-segments`,
    invocationCount: positions.length,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'counts', buffer: counts, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let group = index / ${capacity}u;
  let slot = index % ${capacity}u;
  let count = counts[countsOffset + group * ${countStride}u];
  let outBase = segmentsOffset + index * 4u;
  if (slot >= count || count < 2u) {
    segments[outBase] = ${FAR_AWAY};
    segments[outBase + 1u] = ${FAR_AWAY};
    segments[outBase + 2u] = ${FAR_AWAY};
    segments[outBase + 3u] = ${FAR_AWAY};
    return;
  }
  let fromBase = positionsOffset + (group * ${capacity}u + slot) * 2u;
  let toBase = positionsOffset + (group * ${capacity}u + (slot + 1u) % count) * 2u;
  segments[outBase] = positions[fromBase];
  segments[outBase + 1u] = positions[fromBase + 1u];
  segments[outBase + 2u] = positions[toBase];
  segments[outBase + 3u] = positions[toBase + 1u];`
  });
  return segments;
}

/**
 * One segment per join pair, from the left feature position to the right feature position. Rows at
 * or beyond `count` (the GPU-written pair count) are parked far away.
 */
export function addPairSegments(
  builder: RunBuilder,
  id: string,
  options: {
    leftPositions: RunBuffer<'float32x2'>;
    rightPositions: RunBuffer<'float32x2'>;
    leftIds: RunBuffer<'uint32'>;
    rightIds: RunBuffer<'uint32'>;
    count: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32x4'> {
  const capacity = options.leftIds.length;
  const segments = builder.allocate(`${id}-pair-segments`, 'float32x4', capacity);
  addKernel(builder, {
    id: `${id}-pair-segments`,
    invocationCount: capacity,
    bindings: [
      {name: 'leftPositions', buffer: options.leftPositions, access: 'read'},
      {name: 'rightPositions', buffer: options.rightPositions, access: 'read'},
      {name: 'leftIds', buffer: options.leftIds, access: 'read'},
      {name: 'rightIds', buffer: options.rightIds, access: 'read'},
      {name: 'count', buffer: options.count, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let outBase = segmentsOffset + index * 4u;
  if (index >= count[countOffset]) {
    segments[outBase] = ${FAR_AWAY};
    segments[outBase + 1u] = ${FAR_AWAY};
    segments[outBase + 2u] = ${FAR_AWAY};
    segments[outBase + 3u] = ${FAR_AWAY};
    return;
  }
  let leftBase = leftPositionsOffset + leftIds[leftIdsOffset + index] * 2u;
  let rightBase = rightPositionsOffset + rightIds[rightIdsOffset + index] * 2u;
  segments[outBase] = leftPositions[leftBase];
  segments[outBase + 1u] = leftPositions[leftBase + 1u];
  segments[outBase + 2u] = rightPositions[rightBase];
  segments[outBase + 3u] = rightPositions[rightBase + 1u];`
  });
  return segments;
}

/**
 * Closed ring outlines of a packed polygon result (GeoArrow offsets, `features = count[0]` written
 * features): slot `i` joins vertex `i` to its successor in the same ring. Slots beyond the last
 * ring vertex are parked far away.
 */
export function addRingOffsetSegments(
  builder: RunBuilder,
  id: string,
  options: {
    positions: RunBuffer<'float32x2'>;
    featureOffsets: RunBuffer<'uint32'>;
    polygonOffsets: RunBuffer<'uint32'>;
    ringOffsets: RunBuffer<'uint32'>;
    count: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32x4'> {
  const capacity = options.positions.length;
  const segments = builder.allocate(`${id}-offset-segments`, 'float32x4', capacity);
  addKernel(builder, {
    id: `${id}-offset-segments`,
    invocationCount: capacity,
    bindings: [
      {name: 'positions', buffer: options.positions, access: 'read'},
      {name: 'featureOffsets', buffer: options.featureOffsets, access: 'read'},
      {name: 'polygonOffsets', buffer: options.polygonOffsets, access: 'read'},
      {name: 'ringOffsets', buffer: options.ringOffsets, access: 'read'},
      {name: 'count', buffer: options.count, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let features = count[countOffset];
  let polygonCount = featureOffsets[featureOffsetsOffset + features];
  let ringCount = polygonOffsets[polygonOffsetsOffset + polygonCount];
  let total = ringOffsets[ringOffsetsOffset + ringCount];
  let outBase = segmentsOffset + index * 4u;
  if (index >= total) {
    segments[outBase] = ${FAR_AWAY};
    segments[outBase + 1u] = ${FAR_AWAY};
    segments[outBase + 2u] = ${FAR_AWAY};
    segments[outBase + 3u] = ${FAR_AWAY};
    return;
  }
  var start = 0u;
  var end = total;
  for (var ring = 0u; ring < ringCount; ring++) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (index >= ringStart && index < ringEnd) {
      start = ringStart;
      end = ringEnd;
    }
  }
  let next = select(index + 1u, start, index + 1u == end);
  let fromBase = positionsOffset + index * 2u;
  let toBase = positionsOffset + next * 2u;
  segments[outBase] = positions[fromBase];
  segments[outBase + 1u] = positions[fromBase + 1u];
  segments[outBase + 2u] = positions[toBase];
  segments[outBase + 3u] = positions[toBase + 1u];`
  });
  return segments;
}

/**
 * Closed ring outlines of a ring result whose `counts` buffer holds (written vertices, written
 * rings, ...): slot `i` joins vertex `i` to its successor in the same ring; unused slots are parked.
 */
export function addCountedRingSegments(
  builder: RunBuilder,
  id: string,
  options: {
    positions: RunBuffer<'float32x2'>;
    ringOffsets: RunBuffer<'uint32'>;
    counts: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32x4'> {
  const capacity = options.positions.length;
  const segments = builder.allocate(`${id}-counted-segments`, 'float32x4', capacity);
  addKernel(builder, {
    id: `${id}-counted-segments`,
    invocationCount: capacity,
    bindings: [
      {name: 'positions', buffer: options.positions, access: 'read'},
      {name: 'ringOffsets', buffer: options.ringOffsets, access: 'read'},
      {name: 'counts', buffer: options.counts, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let total = counts[countsOffset];
  let ringCount = counts[countsOffset + 1u];
  let outBase = segmentsOffset + index * 4u;
  if (index >= total) {
    segments[outBase] = ${FAR_AWAY};
    segments[outBase + 1u] = ${FAR_AWAY};
    segments[outBase + 2u] = ${FAR_AWAY};
    segments[outBase + 3u] = ${FAR_AWAY};
    return;
  }
  var start = 0u;
  var end = total;
  for (var ring = 0u; ring < ringCount; ring++) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (index >= ringStart && index < ringEnd) {
      start = ringStart;
      end = ringEnd;
    }
  }
  let next = select(index + 1u, start, index + 1u == end);
  let fromBase = positionsOffset + index * 2u;
  let toBase = positionsOffset + next * 2u;
  segments[outBase] = positions[fromBase];
  segments[outBase + 1u] = positions[fromBase + 1u];
  segments[outBase + 2u] = positions[toBase];
  segments[outBase + 3u] = positions[toBase + 1u];`
  });
  return segments;
}

/** One segment per row from `starts` to `ends`; rows with a NaN endpoint (unused slots) are parked. */
export function addPointPairSegments(
  builder: RunBuilder,
  id: string,
  starts: RunBuffer<'float32x2'>,
  ends: RunBuffer<'float32x2'>
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-point-pair-segments`, 'float32x4', starts.length);
  addKernel(builder, {
    id: `${id}-point-pair-segments`,
    invocationCount: starts.length,
    bindings: [
      {name: 'starts', buffer: starts, access: 'read'},
      {name: 'ends', buffer: ends, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let startBase = startsOffset + index * 2u;
  let endBase = endsOffset + index * 2u;
  let startPoint = vec2<f32>(starts[startBase], starts[startBase + 1u]);
  let endPoint = vec2<f32>(ends[endBase], ends[endBase + 1u]);
  let outBase = segmentsOffset + index * 4u;
  let isValid = startPoint.x == startPoint.x && startPoint.y == startPoint.y && endPoint.x == endPoint.x && endPoint.y == endPoint.y;
  segments[outBase] = select(${FAR_AWAY}, startPoint.x, isValid);
  segments[outBase + 1u] = select(${FAR_AWAY}, startPoint.y, isValid);
  segments[outBase + 2u] = select(${FAR_AWAY}, endPoint.x, isValid);
  segments[outBase + 3u] = select(${FAR_AWAY}, endPoint.y, isValid);`
  });
  return segments;
}

/**
 * One segment per CSR entry, from the row's position to the neighbor's position, plus the number
 * of entries (`offsets[rowCount]`) for the `lines` layer.
 */
export function addCsrEdgeSegments(
  builder: RunBuilder,
  id: string,
  options: {
    positions: RunBuffer<'float32x2'>;
    offsets: RunBuffer<'uint32'>;
    neighbors: RunBuffer<'uint32'>;
    rowCount: number;
  }
): {segments: RunBuffer<'float32x4'>; count: RunBuffer<'uint32'>} {
  const segments = builder.allocate(`${id}-csr-segments`, 'float32x4', options.neighbors.length);
  const count = builder.allocate(`${id}-csr-count`, 'uint32', 1);
  addKernel(builder, {
    id: `${id}-csr-segments`,
    invocationCount: options.rowCount,
    bindings: [
      {name: 'positions', buffer: options.positions, access: 'read'},
      {name: 'offsets', buffer: options.offsets, access: 'read'},
      {name: 'neighbors', buffer: options.neighbors, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: `let first = offsets[offsetsOffset + index];
  let last = offsets[offsetsOffset + index + 1u];
  let rowBase = positionsOffset + index * 2u;
  for (var entry = first; entry < last; entry++) {
    let neighborBase = positionsOffset + neighbors[neighborsOffset + entry] * 2u;
    let outBase = segmentsOffset + entry * 4u;
    segments[outBase] = positions[rowBase];
    segments[outBase + 1u] = positions[rowBase + 1u];
    segments[outBase + 2u] = positions[neighborBase];
    segments[outBase + 3u] = positions[neighborBase + 1u];
  }`
  });
  addKernel(builder, {
    id: `${id}-csr-count`,
    invocationCount: 1,
    bindings: [
      {name: 'offsets', buffer: options.offsets, access: 'read'},
      {name: 'count', buffer: count, access: 'read_write'}
    ],
    body: `count[countOffset] = offsets[offsetsOffset + ${options.rowCount}u];`
  });
  return {segments, count};
}

/** Sum of each CSR row's weights (float32 per row). */
export function addCsrRowSums(
  builder: RunBuilder,
  id: string,
  options: {offsets: RunBuffer<'uint32'>; weights: RunBuffer<'float32'>; rowCount: number}
): RunBuffer<'float32'> {
  const sums = builder.allocate(`${id}-row-sums`, 'float32', options.rowCount);
  addKernel(builder, {
    id: `${id}-row-sums`,
    invocationCount: options.rowCount,
    bindings: [
      {name: 'offsets', buffer: options.offsets, access: 'read'},
      {name: 'weights', buffer: options.weights, access: 'read'},
      {name: 'sums', buffer: sums, access: 'read_write'}
    ],
    body: `var total = 0.0;
  for (var entry = offsets[offsetsOffset + index]; entry < offsets[offsetsOffset + index + 1u]; entry++) {
    total += weights[weightsOffset + entry];
  }
  sums[sumsOffset + index] = total;`
  });
  return sums;
}

/**
 * Reads a 3-bit field of the high word of 64-bit identifier rows (`uint32x2`, `(low, high)`) as a
 * float: for H3 directed edges and vertexes bits 56-58 hold the direction or vertex number.
 */
export function addHighWordField(
  builder: RunBuilder,
  id: string,
  identifiers: RunBuffer<'uint32x2'>,
  shift: number
): RunBuffer<'float32'> {
  const result = builder.allocate(`${id}-field`, 'float32', identifiers.length);
  addKernel(builder, {
    id: `${id}-high-word-field`,
    invocationCount: identifiers.length,
    bindings: [
      {name: 'identifiers', buffer: identifiers, access: 'read'},
      {name: 'result', buffer: result, access: 'read_write'}
    ],
    body: `let base = identifiersOffset + index * 2u;
  let isEmpty = identifiers[base] == 0u && identifiers[base + 1u] == 0u;
  result[resultOffset + index] = select(f32((identifiers[base + 1u] >> ${shift}u) & 7u), -1.0, isEmpty);`
  });
  return result;
}

/**
 * The level (0 to 30) of 64-bit S2 cell ids (`uint32x2`, `(low, high)`) as a float: the lowest set
 * bit of an id sits at position `2 * (30 - level)`. Empty rows (key 0) read -1.
 */
export function addS2Levels(
  builder: RunBuilder,
  id: string,
  cells: RunBuffer<'uint32x2'>
): RunBuffer<'float32'> {
  const levels = builder.allocate(`${id}-levels`, 'float32', cells.length);
  addKernel(builder, {
    id: `${id}-s2-levels`,
    invocationCount: cells.length,
    bindings: [
      {name: 'cells', buffer: cells, access: 'read'},
      {name: 'levels', buffer: levels, access: 'read_write'}
    ],
    body: `let base = cellsOffset + index * 2u;
  let low = cells[base];
  let high = cells[base + 1u];
  let zeros = select(countTrailingZeros(low), 32u + countTrailingZeros(high), low == 0u);
  levels[levelsOffset + index] = select(f32(30u - zeros / 2u), -1.0, low == 0u && high == 0u);`
  });
  return levels;
}

/**
 * The zone each left row was paired with (the first pair that names it), or 0xffffffff when the row
 * has no pair: lets matched points be colored by zone.
 */
export function addPairedRightIds(
  builder: RunBuilder,
  id: string,
  options: {
    rowCount: number;
    leftIds: RunBuffer<'uint32'>;
    rightIds: RunBuffer<'uint32'>;
    count: RunBuffer<'uint32'>;
  }
): RunBuffer<'uint32'> {
  const zoneIds = builder.allocate(`${id}-paired-ids`, 'uint32', options.rowCount);
  addKernel(builder, {
    id: `${id}-paired-ids`,
    invocationCount: options.rowCount,
    bindings: [
      {name: 'leftIds', buffer: options.leftIds, access: 'read'},
      {name: 'rightIds', buffer: options.rightIds, access: 'read'},
      {name: 'count', buffer: options.count, access: 'read'},
      {name: 'zoneIds', buffer: zoneIds, access: 'read_write'}
    ],
    body: `var zone = 0xffffffffu;
  let total = min(count[countOffset], ${options.leftIds.length}u);
  for (var pair = 0u; pair < total; pair++) {
    if (leftIds[leftIdsOffset + pair] == index) {
      zone = rightIds[rightIdsOffset + pair];
      break;
    }
  }
  zoneIds[zoneIdsOffset + index] = zone;`
  });
  return zoneIds;
}

/**
 * A `columns` x `rows` raster (row 0 at the top) over a longitude/latitude window: 1 where the cell
 * center is inside the closed rings given as (x0, y0, x1, y1) segments (even-odd, so holes stay
 * empty), 1e30 (drawn as invalid) elsewhere. Parked segments far away never cross a ray.
 */
export function addSegmentFill(
  builder: RunBuilder,
  id: string,
  segments: RunBuffer<'float32x4'>,
  window: readonly [number, number, number, number],
  columns: number,
  rows: number
): RunBuffer<'float32'> {
  const [west, south, east, north] = window;
  const fill = builder.allocate(`${id}-fill`, 'float32', columns * rows);
  addKernel(builder, {
    id: `${id}-segment-fill`,
    invocationCount: columns * rows,
    bindings: [
      {name: 'segments', buffer: segments, access: 'read'},
      {name: 'fill', buffer: fill, access: 'read_write'}
    ],
    body: `let column = index % ${columns}u;
  let row = index / ${columns}u;
  let x = ${west} + (f32(column) + 0.5) / ${columns}.0 * ${east - west};
  let y = ${north} - (f32(row) + 0.5) / ${rows}.0 * ${north - south};
  var isInside = false;
  for (var segment = 0u; segment < ${segments.length}u; segment++) {
    let base = segmentsOffset + segment * 4u;
    let x0 = segments[base];
    let y0 = segments[base + 1u];
    let x1 = segments[base + 2u];
    let y1 = segments[base + 3u];
    if ((y0 > y) != (y1 > y) && x < (x1 - x0) * (y - y0) / (y1 - y0) + x0) {
      isInside = !isInside;
    }
  }
  fill[fillOffset + index] = select(1.0e30, 1.0, isInside);`
  });
  return fill;
}
