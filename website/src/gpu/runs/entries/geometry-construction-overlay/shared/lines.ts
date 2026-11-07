// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Line fixtures and drawing helpers shared by the geometry-construction runs. Operations write
// packed lines (positions plus line offsets, with a GPU-written count); `addPackedSegments` turns
// that packed output into a segment list on the GPU so the `lines` layer can draw exactly what the
// operation wrote.

import type {GPULineGeometryOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';
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

/** Wandering polylines across the world: `lineCount` lines of `vertexCount` vertices. */
export function createLineSet(
  random: Random,
  world: World,
  lineCount = 4,
  vertexCount = 8,
  wobble = 1
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
      const x = (0.05 + 0.9 * along) * world.width;
      const y =
        baseY +
        Math.sin(along * 7 + line * 1.9) * 0.05 * world.height +
        (random() - 0.5) * 0.09 * wobble * world.height;
      positions.push(x, y);
      if (vertex > 0) segments.push(previousX, previousY, x, y);
      previousX = x;
      previousY = y;
    }
    lineOffsets.push(positions.length / 2);
  }
  return {
    lineCount,
    positions: Float32Array.from(positions),
    lineOffsets: Uint32Array.from(lineOffsets),
    segments: Float32Array.from(segments)
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

/** Sharp zigzag polylines, for operations whose corner treatment matters (joins, mitre limits). */
export function createZigzagLines(
  random: Random,
  world: World,
  lineCount = 2,
  vertexCount = 6,
  amplitude = 0.12
): LineSet {
  const positions: number[] = [];
  const lineOffsets = [0];
  const segments: number[] = [];
  for (let line = 0; line < lineCount; line++) {
    const baseY = ((line + 0.5) / lineCount) * world.height;
    let previousX = 0;
    let previousY = 0;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const x = (0.08 + (0.84 * vertex) / (vertexCount - 1)) * world.width;
      const y = baseY + (vertex % 2 === 0 ? -1 : 1) * amplitude * (0.6 + 0.8 * random());
      positions.push(x, y);
      if (vertex > 0) segments.push(previousX, previousY, x, y);
      previousX = x;
      previousY = y;
    }
    lineOffsets.push(positions.length / 2);
  }
  return {
    lineCount,
    positions: Float32Array.from(positions),
    lineOffsets: Uint32Array.from(lineOffsets),
    segments: Float32Array.from(segments)
  };
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
