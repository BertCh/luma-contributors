// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Line fixtures and drawing helpers shared by the geometry-construction runs. Operations write
// packed lines (positions plus line offsets, with a GPU-written count); `addPackedSegments` turns
// that packed output into a segment list on the GPU so the `lines` layer can draw exactly what the
// operation wrote.

import {createRiverBend, createRiverLine, RIVER_REACH_COUNT} from '../../../../data/hydrography';
import type {GPULineGeometryOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {Point, RunGlyph} from '../../../interaction';
import {findNearestOnPolyline} from '../../../pick';
import {addKernelPass} from './kernel';

export type World = {width: number; height: number};
export type Random = () => number;

export type LineSet = {
  lineCount: number;
  positions: Float32Array;
  lineOffsets: Uint32Array;
  /** Consecutive vertex pairs as (x0, y0, x1, y1). */
  segments: Float32Array;
};

/** Largest distance of a line from its chord, relative to the chord: 0 is perfectly straight. */
function getBend(line: number[][]): number {
  const [x0, y0] = line[0];
  const [x1, y1] = line[line.length - 1];
  const chord = Math.hypot(x1 - x0, y1 - y0) || 1;
  return Math.max(
    ...line.map(([x, y]) => Math.abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / chord / chord)
  );
}

/** River reach indices from `first` on, skipping the nearly straight ones. */
export function getBendingReaches(first: number, count: number, world: World): number[] {
  const reaches: number[] = [];
  for (let index = first; reaches.length < count && index < first + RIVER_REACH_COUNT; index++) {
    if (getBend(createRiverLine(index, 12, world)) > 0.06) reaches.push(index);
  }
  for (let index = first; reaches.length < count; index++) reaches.push(index);
  return reaches;
}

/** Resampled river centerlines, displayed side by side for editing and comparison. */
export function createLineSet(
  random: Random,
  world: World,
  lineCount = 4,
  vertexCount = 8
): LineSet {
  const firstReach = Math.floor(random() * RIVER_REACH_COUNT);
  const reaches = getBendingReaches(firstReach, lineCount, world);
  const lines = reaches.map((reach, index) =>
    createRiverLine(reach, vertexCount, world, index, lineCount)
  );
  const positions = Float32Array.from(lines.flat(2));
  const lineOffsets = Uint32Array.from({length: lineCount + 1}, (_, index) => index * vertexCount);
  return {
    lineCount,
    positions,
    lineOffsets,
    segments: createPackedSegments(positions, lineOffsets, false)
  };
}

/** Two or more bending river reaches stacked in panels, close enough for buffers to fuse. */
export function createBendSet(world: World, lineCount = 2, vertexCount = 16): LineSet {
  const lines = Array.from({length: lineCount}, (_, index) =>
    createRiverBend(index, vertexCount, world, index, lineCount)
  );
  const positions = Float32Array.from(lines.flat(2));
  const lineOffsets = Uint32Array.from({length: lineCount + 1}, (_, index) => index * vertexCount);
  return {
    lineCount,
    positions,
    lineOffsets,
    segments: createPackedSegments(positions, lineOffsets, false)
  };
}

/** Segment list of packed lines or rings on the CPU, for drawing the inputs. */
export function createPackedSegments(
  positions: Float32Array,
  offsets: Uint32Array,
  closed: boolean
): Float32Array {
  const segments: number[] = [];
  for (let line = 0; line + 1 < offsets.length; line++) {
    const first = offsets[line];
    const end = offsets[line + 1];
    for (let row = first; row < end; row++) {
      const next = row + 1 < end ? row + 1 : closed && end - first > 2 ? first : -1;
      if (next < 0) continue;
      segments.push(
        positions[row * 2],
        positions[row * 2 + 1],
        positions[next * 2],
        positions[next * 2 + 1]
      );
    }
  }
  return Float32Array.from(segments);
}

/** A `GPULineGeometryOutput` backed by allocated buffers. */
export type LineOutputBuffers = {
  positions: RunBuffer<'float32x2'>;
  lineOffsets: RunBuffer<'uint32'>;
  count: RunBuffer<'uint32'>;
  overflow: RunBuffer<'uint32'>;
  lineCount: RunBuffer<'uint32'>;
  sourceLineIndices: RunBuffer<'uint32'>;
  output: GPULineGeometryOutput;
};

/** Allocates the output of a line operation: `rowCapacity` vertices over `lineCapacity` lines. */
export function allocateLineOutput(
  builder: RunBuilder,
  prefix: string,
  rowCapacity: number,
  lineCapacity: number
): LineOutputBuffers {
  const positions = builder.allocate(`${prefix}-positions`, 'float32x2', rowCapacity);
  const lineOffsets = builder.allocate(`${prefix}-line-offsets`, 'uint32', lineCapacity + 1);
  const count = builder.allocate(`${prefix}-count`, 'uint32', 1);
  const overflow = builder.allocate(`${prefix}-overflow`, 'uint32', 1);
  const lineCount = builder.allocate(`${prefix}-line-count`, 'uint32', 1);
  const sourceLineIndices = builder.allocate(`${prefix}-source-lines`, 'uint32', lineCapacity);
  return {
    positions,
    lineOffsets,
    count,
    overflow,
    lineCount,
    sourceLineIndices,
    output: {
      positions: positions.view,
      lineOffsets: lineOffsets.view,
      status: {count: count.view, overflow: overflow.view, lineCount: lineCount.view},
      sourceLineIndices: sourceLineIndices.view
    }
  };
}

/** Maps packed coordinates into the drawing world: `position * scale + offset`. */
export type DrawTransform = {
  scale: readonly [number, number];
  offset: readonly [number, number];
};

const IDENTITY_TRANSFORM: DrawTransform = {scale: [1, 1], offset: [0, 0]};

export type PackedDrawingOptions = {
  /** Each line is a ring: draw the closing edge too. */
  closed?: boolean;
  transform?: DrawTransform;
  /** GPU-written number of valid lines; without it every offset row counts. */
  lineCount?: RunBuffer<'uint32'>;
  /** Per-row flags: when given, only rows with a nonzero flag produce a visible `segments` entry. */
  flags?: RunBuffer<'uint32'>;
  /** Per-line flags: only lines with a nonzero flag produce a visible `segments` entry. */
  lineFlags?: RunBuffer<'uint32'>;
  /** Per-line values (for example a group label per ring) that `lineIds` carries instead of the line index. */
  lineValues?: RunBuffer<'uint32'>;
};

/** Drawable form of packed lines: a segment and a point per packed row (off-screen when unused). */
export type PackedDrawing = {
  segments: RunBuffer<'float32x4'>;
  points: RunBuffer<'float32x2'>;
  /** Edge (row, next row) per packed row, over `points`; (row, row) at line ends. */
  edges: RunBuffer<'uint32x2'>;
  /** Line index per packed row, for coloring by line. */
  lineIds: RunBuffer<'uint32'>;
};

/**
 * Adds a compute pass that writes one drawable segment per packed vertex: from the vertex to the
 * next vertex of the same line (or ring, when `closed`). Rows past the packed total, and line
 * ends, get an off-screen segment, so the `lines` layer draws every row.
 */
export function addPackedSegments(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  lineOffsets: RunBuffer<'uint32'>,
  options: PackedDrawingOptions = {}
): PackedDrawing {
  const {
    closed = false,
    transform = IDENTITY_TRANSFORM,
    lineCount: lineCountBuffer,
    flags,
    lineFlags,
    lineValues
  } = options;
  const segments = builder.allocate(`${id}-segments`, 'float32x4', positions.length);
  const points = builder.allocate(`${id}-points`, 'float32x2', positions.length);
  const edges = builder.allocate(`${id}-edges`, 'uint32x2', positions.length);
  const lineIds = builder.allocate(`${id}-line-ids`, 'uint32', positions.length);
  const lineCapacity = lineOffsets.length - 1;
  const condition =
    [
      flags ? 'flags[flagsOffset + index] != 0u' : '',
      lineFlags ? 'lineFlags[lineFlagsOffset + low - 1u] != 0u' : ''
    ]
      .filter(Boolean)
      .join(' && ') || 'true';
  addKernelPass(builder, {
    id: `${id}-segment-pass`,
    invocationCount: positions.length,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'lineOffsets', buffer: lineOffsets, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'},
      {name: 'points', buffer: points, access: 'read_write'},
      {name: 'edges', buffer: edges, access: 'read_write'},
      {name: 'lineIds', buffer: lineIds, access: 'read_write'},
      ...(lineCountBuffer
        ? [{name: 'lineCountBuffer', buffer: lineCountBuffer, access: 'read' as const}]
        : []),
      ...(flags ? [{name: 'flags', buffer: flags, access: 'read' as const}] : []),
      ...(lineFlags ? [{name: 'lineFlags', buffer: lineFlags, access: 'read' as const}] : []),
      ...(lineValues ? [{name: 'lineValues', buffer: lineValues, access: 'read' as const}] : [])
    ],
    body: /* wgsl */ `
  let lineCount = ${lineCountBuffer ? `min(lineCountBuffer[lineCountBufferOffset], ${lineCapacity}u)` : `${lineCapacity}u`};
  var low = 0u;
  var high = lineCount + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
  }
  segments[segmentsOffset + index] = vec4f(-9.0);
  points[pointsOffset + index] = vec2f(-9.0);
  edges[edgesOffset + index] = vec2u(index, index);
  lineIds[lineIdsOffset + index] = 0u;
  if (low == 0u || low > lineCount) { return; }
  let first = lineOffsets[lineOffsetsOffset + low - 1u];
  let end = lineOffsets[lineOffsetsOffset + low];
  let scale = vec2f(${transform.scale[0]}, ${transform.scale[1]});
  let shift = vec2f(${transform.offset[0]}, ${transform.offset[1]});
  let start = positions[positionsOffset + index] * scale + shift;
  points[pointsOffset + index] = start;
  lineIds[lineIdsOffset + index] = ${lineValues ? 'lineValues[lineValuesOffset + low - 1u]' : 'low - 1u'};
  if (index + 1u < end) {
    if (${condition}) {
      segments[segmentsOffset + index] = vec4f(start, positions[positionsOffset + index + 1u] * scale + shift);
    }
    edges[edgesOffset + index] = vec2u(index, index + 1u);
  } else if (${closed ? 'true' : 'false'} && end - first > 2u) {
    if (${condition}) {
      segments[segmentsOffset + index] = vec4f(start, positions[positionsOffset + first] * scale + shift);
    }
    edges[edgesOffset + index] = vec2u(index, first);
  }`
  });
  return {segments, points, edges, lineIds};
}

/** Accent colors for outputs drawn on top of muted inputs. */
export const OUTPUT_COLOR = [0.38, 0.68, 1, 1] as const;
export const HIGHLIGHT_COLOR = [1, 0.58, 0.25, 1] as const;

/** Line index of every packed row, computed from line offsets on the CPU. */
export function createRowLines(lineOffsets: Uint32Array): Uint32Array {
  const rowLines = new Uint32Array(lineOffsets[lineOffsets.length - 1]);
  for (let line = 0; line + 1 < lineOffsets.length; line++) {
    rowLines.fill(line, lineOffsets[line], lineOffsets[line + 1]);
  }
  return rowLines;
}

/**
 * Adds a compute pass that draws a selection of rows as lines: one segment between consecutive
 * selected row ids when both rows belong to the same line, off-screen otherwise (and past `count`).
 */
export function addSelectedSegments(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  rowLines: RunBuffer<'uint32'>,
  ids: RunBuffer<'uint32'>,
  count: RunBuffer<'uint32'>
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(`${id}-segments`, 'float32x4', ids.length);
  addKernelPass(builder, {
    id: `${id}-selected-pass`,
    invocationCount: ids.length,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'rowLines', buffer: rowLines, access: 'read'},
      {name: 'ids', buffer: ids, access: 'read'},
      {name: 'count', buffer: count, access: 'read'},
      {name: 'segments', buffer: segments, access: 'read_write'}
    ],
    body: /* wgsl */ `
  segments[segmentsOffset + index] = vec4f(-9.0);
  if (index + 1u >= count[countOffset]) { return; }
  let first = ids[idsOffset + index];
  let second = ids[idsOffset + index + 1u];
  if (rowLines[rowLinesOffset + first] != rowLines[rowLinesOffset + second]) { return; }
  segments[segmentsOffset + index] = vec4f(positions[positionsOffset + first], positions[positionsOffset + second]);`
  });
  return segments;
}

/** Buffers behind a `GPULineSplitPieces` output (line split and clip by polygon). */
export type PieceOutputBuffers = {
  positions: RunBuffer<'float32x2'>;
  offsets: RunBuffer<'uint32'>;
  count: RunBuffer<'uint32'>;
  pieces: {
    geometry: {
      kind: 'lines';
      positions: GraphDataView<'float32x2'>;
      lineOffsets: GraphDataView<'uint32'>;
    };
    sourceIds: GraphDataView<'uint32'>;
    status: {
      count: GraphDataView<'uint32'>;
      overflow: GraphDataView<'uint32'>;
      requiredCount: GraphDataView<'uint32'>;
      vertexCount: GraphDataView<'uint32'>;
      requiredVertexCount: GraphDataView<'uint32'>;
    };
  };
};

/** Allocates the `pieces` output of an operation that cuts lines into pieces. */
export function allocatePieceOutput(
  builder: RunBuilder,
  prefix: string,
  pieceCapacity: number,
  vertexCapacity: number
): PieceOutputBuffers {
  const positions = builder.allocate(`${prefix}-positions`, 'float32x2', vertexCapacity);
  const offsets = builder.allocate(`${prefix}-offsets`, 'uint32', pieceCapacity + 1);
  const sourceIds = builder.allocate(`${prefix}-sources`, 'uint32', pieceCapacity);
  const scalar = (name: string) => builder.allocate(`${prefix}-${name}`, 'uint32', 1);
  const count = scalar('count');
  return {
    positions,
    offsets,
    count,
    pieces: {
      geometry: {kind: 'lines', positions: positions.view, lineOffsets: offsets.view},
      sourceIds: sourceIds.view,
      status: {
        count: count.view,
        overflow: scalar('overflow').view,
        requiredCount: scalar('required-count').view,
        vertexCount: scalar('vertex-count').view,
        requiredVertexCount: scalar('required-vertex-count').view
      }
    }
  };
}

// ---------------------------------------------------------------------------------------------
// CPU lookups under the pointer. Lines are packed positions plus line offsets, like the inputs of
// the operations; they work in drawing-world units.

export type PackedLines = {
  positions: ArrayLike<number>;
  lineOffsets: ArrayLike<number>;
};

export type LineHit = {
  line: number;
  /** Positive when the query is to the left of the line direction. */
  signedDistance: number;
  distance: number;
  /** The closest point on the line. */
  point: Point;
  /** Arc length from the start of the line to `point`. */
  measure: number;
  /** Length of the whole line. */
  length: number;
  /** Vertex row (packed index) where the closest segment starts. */
  row: number;
  /** 0 to 1 along the closest segment. */
  fraction: number;
};

export function getLineLength(
  positions: ArrayLike<number>,
  first: number,
  end: number,
  closed = false
): number {
  let length = 0;
  const last = closed ? end - first : end - first - 1;
  for (let step = 0; step < last; step++) {
    const from = first + step;
    const to = first + ((step + 1) % (end - first));
    length += Math.hypot(
      positions[to * 2] - positions[from * 2],
      positions[to * 2 + 1] - positions[from * 2 + 1]
    );
  }
  return length;
}

/** The closest line (or ring, when `closed`) to (x, y), with the measure of the closest point. */
export function findNearestLine(
  lines: PackedLines,
  x: number,
  y: number,
  closed = false
): LineHit | null {
  let best: LineHit | null = null;
  for (let line = 0; line + 1 < lines.lineOffsets.length; line++) {
    const first = lines.lineOffsets[line];
    const end = lines.lineOffsets[line + 1];
    const hit = findNearestOnPolyline(lines.positions, x, y, first, end - first, closed);
    if (!hit || (best && hit.distance >= best.distance)) continue;
    best = {
      line,
      signedDistance: hit.signedDistance,
      distance: hit.distance,
      point: hit.point,
      measure: 0,
      length: 0,
      row: hit.index,
      fraction: hit.fraction
    };
  }
  if (!best) return null;
  const first = lines.lineOffsets[best.line];
  const end = lines.lineOffsets[best.line + 1];
  const toRow = getLineLength(lines.positions, first, best.row + 1, false);
  const next = best.row + 1 < end ? best.row + 1 : first;
  const segmentLength = Math.hypot(
    lines.positions[next * 2] - lines.positions[best.row * 2],
    lines.positions[next * 2 + 1] - lines.positions[best.row * 2 + 1]
  );
  best.measure = toRow + segmentLength * best.fraction;
  best.length = getLineLength(lines.positions, first, end, closed);
  return best;
}

/** The vertex nearest to (x, y) among `count` packed vertices. */
export function findNearestVertex(
  positions: ArrayLike<number>,
  x: number,
  y: number,
  count = positions.length >> 1
): {row: number; distance: number; point: Point} | null {
  let best: {row: number; distance: number; point: Point} | null = null;
  for (let row = 0; row < count; row++) {
    const distance = Math.hypot(positions[row * 2] - x, positions[row * 2 + 1] - y);
    if (!best || distance < best.distance) {
      best = {row, distance, point: [positions[row * 2], positions[row * 2 + 1]]};
    }
  }
  return best;
}

/** The point at `fraction` along segment `row` of a packed line, moved `offset` to its left. */
export function getPointBesideLine(
  positions: ArrayLike<number>,
  row: number,
  fraction: number,
  offset: number
): Point {
  const x0 = positions[row * 2];
  const y0 = positions[row * 2 + 1];
  const deltaX = positions[row * 2 + 2] - x0;
  const deltaY = positions[row * 2 + 3] - y0;
  const length = Math.hypot(deltaX, deltaY) || 1;
  return [
    x0 + deltaX * fraction - (deltaY / length) * offset,
    y0 + deltaY * fraction + (deltaX / length) * offset
  ];
}

/** A polyline displaced by `offset` to its left along averaged vertex normals, as x, y pairs. */
export function createOffsetPolyline(
  positions: ArrayLike<number>,
  first: number,
  end: number,
  offset: number
): number[] {
  const points: number[] = [];
  for (let row = first; row < end; row++) {
    const before = Math.max(row - 1, first);
    const after = Math.min(row + 1, end - 1);
    const deltaX = positions[after * 2] - positions[before * 2];
    const deltaY = positions[after * 2 + 1] - positions[before * 2 + 1];
    const length = Math.hypot(deltaX, deltaY) || 1;
    points.push(
      positions[row * 2] - (deltaY / length) * offset,
      positions[row * 2 + 1] + (deltaX / length) * offset
    );
  }
  return points;
}

/** Dashed corridor of half-width `halfWidth` along line `line`: the band a tolerance covers. */
export function createCorridorGlyphs(
  lines: PackedLines,
  line: number,
  halfWidth: number
): RunGlyph[] {
  const first = lines.lineOffsets[line];
  const end = lines.lineOffsets[line + 1];
  return [halfWidth, -halfWidth].map(offset => ({
    shape: 'polyline' as const,
    points: createOffsetPolyline(lines.positions, first, end, offset),
    dashed: true
  }));
}

/** The point `measure` along line `line` (arc length), with the segment it falls on. */
export function getPointAtMeasure(
  lines: PackedLines,
  line: number,
  measure: number
): {point: Point; row: number; fraction: number} {
  const first = lines.lineOffsets[line];
  const end = lines.lineOffsets[line + 1];
  let remaining = measure;
  for (let row = first; row + 1 < end; row++) {
    const deltaX = lines.positions[row * 2 + 2] - lines.positions[row * 2];
    const deltaY = lines.positions[row * 2 + 3] - lines.positions[row * 2 + 1];
    const length = Math.hypot(deltaX, deltaY);
    if (remaining <= length || row + 2 === end) {
      const fraction = length > 0 ? Math.min(1, remaining / length) : 0;
      return {
        point: [
          lines.positions[row * 2] + deltaX * fraction,
          lines.positions[row * 2 + 1] + deltaY * fraction
        ],
        row,
        fraction
      };
    }
    remaining -= length;
  }
  return {
    point: [lines.positions[first * 2], lines.positions[first * 2 + 1]],
    row: first,
    fraction: 0
  };
}

/** A short tick across the line at `point`, perpendicular to segment `row`. */
export function createTickGlyph(
  positions: ArrayLike<number>,
  row: number,
  point: Point,
  halfLength: number
): RunGlyph {
  const deltaX = positions[row * 2 + 2] - positions[row * 2];
  const deltaY = positions[row * 2 + 3] - positions[row * 2 + 1];
  const length = Math.hypot(deltaX, deltaY) || 1;
  const normalX = (-deltaY / length) * halfLength;
  const normalY = (deltaX / length) * halfLength;
  return {
    shape: 'line',
    from: [point[0] - normalX, point[1] - normalY],
    to: [point[0] + normalX, point[1] + normalY]
  };
}

/**
 * A pointer position (drawing world) where `measure` returns `target`, found beside the segments
 * of `lines` (each side, a few fractions). The poster is captured here, so it must reproduce the
 * default parameter value exactly. `isUsable` keeps the position inside the picture.
 */
export function findRestBesideLines(
  lines: PackedLines,
  target: number,
  measure: (x: number, y: number) => number,
  world: World,
  tolerance = 5e-4
): Point {
  let fallback: Point | null = null;
  for (let line = 0; line + 1 < lines.lineOffsets.length; line++) {
    for (let row = lines.lineOffsets[line]; row + 1 < lines.lineOffsets[line + 1]; row++) {
      for (const fraction of [0.5, 0.3, 0.7]) {
        for (const side of [1, -1]) {
          const candidate = getPointBesideLine(
            lines.positions,
            row,
            fraction,
            side * Math.abs(target)
          );
          fallback ??= candidate;
          const isInside =
            candidate[0] > 0.03 &&
            candidate[0] < world.width - 0.03 &&
            candidate[1] > 0.03 &&
            candidate[1] < world.height - 0.03;
          if (isInside && Math.abs(measure(candidate[0], candidate[1]) - target) < tolerance) {
            return candidate;
          }
        }
      }
    }
  }
  return fallback ?? [world.width / 2, world.height / 2];
}

/** A pointer position `distance` from `origin` where `measure` agrees (angles searched). */
export function findRestAround(
  origin: Point,
  distance: number,
  measure: (x: number, y: number) => number,
  world: World,
  tolerance = 5e-4
): Point {
  for (let step = 0; step < 48; step++) {
    const angle = (step / 48) * Math.PI * 2;
    const candidate: Point = [
      origin[0] + Math.cos(angle) * distance,
      origin[1] + Math.sin(angle) * distance
    ];
    const isInside =
      candidate[0] > 0.03 &&
      candidate[0] < world.width - 0.03 &&
      candidate[1] > 0.03 &&
      candidate[1] < world.height - 0.03;
    if (isInside && Math.abs(measure(candidate[0], candidate[1]) - distance) < tolerance) {
      return candidate;
    }
  }
  return [origin[0] + distance, origin[1]];
}
