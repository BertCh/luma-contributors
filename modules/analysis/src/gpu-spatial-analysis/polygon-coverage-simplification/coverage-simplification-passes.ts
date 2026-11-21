// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {KEY_PAIR_INVALID} from '../spatial-weights/key-pair-grouping';
import {GPUSegmentIntersection} from '../segment-intersection/index';

import {NODE_OVERLAPS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';

const OPERATION = 'GPUCoverageSimplification';
const INVALID = `${KEY_PAIR_INVALID}u`;
const SPAN_LINK_STRIDE = 3;
const SWEPT_BIT = 2;

/** Per-vertex keep flag from the shared per-point keep decisions. */
export function createKeepMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  pointIds: GraphDataView<'uint32'>,
  keepPoint: GraphDataView<'uint32'>,
  keepMask: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'keep-mask',
    bindings: [
      {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
      {name: 'keepPoint', view: keepPoint, type: 'u32', access: 'read'},
      {name: 'keepMask', view: keepMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: pointIds.length,
    body: `let pointId = pointIds[pointIdsOffset + index];
  keepMask[keepMaskOffset + index] =
    select(0u, keepPoint[keepPointOffset + pointId], pointId != ${INVALID});`
  });
}

/** Writes the kept vertices, in order, to `outputPositions`, bounded by its capacity. */
export function createEmitPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    keepMask: GraphDataView<'uint32'>;
    keepRank: GraphDataView<'uint32'>;
    outputPositions: GraphDataView<'float32x2'>;
    vertexCount: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'emit-positions',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'keepMask', view: props.keepMask, type: 'u32', access: 'read'},
      {name: 'keepRank', view: props.keepRank, type: 'u32', access: 'read'},
      {name: 'outputPositions', view: props.outputPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.vertexCount,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;`,
    body: `let rank = keepRank[keepRankOffset + index];
  if (keepMask[keepMaskOffset + index] != 0u && rank < CAPACITY) {
    outputPositions[outputPositionsOffset + rank * 2u] = positions[positionsOffset + index * 2u];
    outputPositions[outputPositionsOffset + rank * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];
  }`
  });
}

/** Writes the ring offsets of the kept vertices, plus the overflow flag and optional total. */
export function createEmitOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    ringOffsets: GraphDataView<'uint32'>;
    keepMask: GraphDataView<'uint32'>;
    keepRank: GraphDataView<'uint32'>;
    outputRingOffsets: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
    capacity: number;
  }
): GPUCommandNode<Parameters> {
  const {requiredCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'emit-offsets',
    bindings: [
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'keepMask', view: props.keepMask, type: 'u32', access: 'read'},
      {name: 'keepRank', view: props.keepRank, type: 'u32', access: 'read'},
      {name: 'outputRingOffsets', view: props.outputRingOffsets, type: 'u32', access: 'read_write'},
      {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
      ...(requiredCount
        ? [
            {
              name: 'requiredCount',
              view: requiredCount,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    invocationCount: props.ringCount + 1,
    declarations: `const VERTEX_COUNT: u32 = ${props.vertexCount}u;
const RING_COUNT: u32 = ${props.ringCount}u;
const CAPACITY: u32 = ${props.capacity}u;`,
    body: `let total = keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
  var begin = total;
  if (index < RING_COUNT) {
    let ringBegin = ringOffsets[ringOffsetsOffset + index];
    if (ringBegin < VERTEX_COUNT) {
      begin = keepRank[keepRankOffset + ringBegin];
    }
  }
  outputRingOffsets[outputRingOffsetsOffset + index] = min(begin, CAPACITY);
  if (index == RING_COUNT) {
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''}
  }`
  });
}

/**
 * Restores vertices of rings that kept fewer than three distinct points: the farthest original
 * vertex from the kept point, then the farthest from the line through the two. Reads the frozen
 * keep decisions and writes a separate buffer that is merged afterwards, so the result does not
 * depend on thread order.
 */
export function getRingCollapseNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    vertexPolygon: GraphDataView<'uint32'>;
    pointIds: GraphDataView<'uint32'>;
    keepPoint: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, vertexCount, ringCount} = props;
  const restorePoint = createTransientView(graph, `${id}-restore-point`, 'uint32', vertexCount);
  return [
    createFillNode<Parameters>(graph, {
      id: `${id}-restore-point-clear`,
      operation: OPERATION,
      view: restorePoint,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-ring-minimum`,
      operation: OPERATION,
      variant: 'ring-minimum',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
        {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
        {name: 'keepPoint', view: props.keepPoint, type: 'u32', access: 'read'},
        {name: 'restorePoint', view: restorePoint, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: ringCount,
      declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isKept(vertex: u32) -> bool {
  let pointId = pointIds[pointIdsOffset + vertex];
  return pointId != ${INVALID} && keepPoint[keepPointOffset + pointId] != 0u;
}`,
      body: `let begin = ringOffsets[ringOffsetsOffset + index];
  var end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end > begin + 1u && pointIds[pointIdsOffset + end - 1u] == pointIds[pointIdsOffset + begin]) {
    end = end - 1u;
  }
  var keptCount = 0u;
  var firstKept = ${INVALID};
  var secondKept = ${INVALID};
  for (var vertex = begin; vertex < end; vertex++) {
    if (isKept(vertex)) {
      if (keptCount == 0u) { firstKept = vertex; } else if (keptCount == 1u) { secondKept = vertex; }
      keptCount++;
    }
  }
  if (keptCount >= 1u && keptCount < 3u && end > begin + 2u) {
    let anchor = vertexAt(firstKept);
    var other = vertexAt(firstKept);
    var otherVertex = secondKept;
    var haveOther = keptCount >= 2u;
    if (haveOther) {
      other = vertexAt(secondKept);
    } else {
      var best = 0.0;
      for (var vertex = begin; vertex < end; vertex++) {
        if (!isKept(vertex) && pointIds[pointIdsOffset + vertex] != ${INVALID}) {
          let distanceToAnchor = distance(vertexAt(vertex), anchor);
          if (distanceToAnchor > best) { best = distanceToAnchor; otherVertex = vertex; }
        }
      }
      if (otherVertex != ${INVALID}) {
        haveOther = true;
        other = vertexAt(otherVertex);
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + otherVertex]], 1u);
      }
    }
    if (haveOther) {
      let direction = other - anchor;
      let directionLength = max(length(direction), 1e-30);
      var best = 0.0;
      var thirdVertex = ${INVALID};
      for (var vertex = begin; vertex < end; vertex++) {
        let pointId = pointIds[pointIdsOffset + vertex];
        if (!isKept(vertex) && pointId != ${INVALID} && vertex != otherVertex &&
            pointId != pointIds[pointIdsOffset + otherVertex]) {
          let offset = vertexAt(vertex) - anchor;
          let distanceToLine = abs(direction.x * offset.y - direction.y * offset.x) / directionLength;
          if (distanceToLine > best) { best = distanceToLine; thirdVertex = vertex; }
        }
      }
      if (thirdVertex != ${INVALID}) {
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + thirdVertex]], 1u);
      }
    }
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-merge-restored`,
      operation: OPERATION,
      variant: 'merge-restored',
      bindings: [
        {name: 'restorePoint', view: restorePoint, type: 'u32', access: 'read'},
        {name: 'keepPoint', view: props.keepPoint, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      body: 'keepPoint[keepPointOffset + index] = max(keepPoint[keepPointOffset + index], restorePoint[restorePointOffset + index]);'
    })
  ];
}

/**
 * Span ends and split candidates of the compacted simplification (see the call site). `spanKey`
 * receives, per kept vertex, the walk-order key of its farthest dropped vertex (`INVALID` when none
 * is strictly away from the span): the vertex index, plus `vertexCount` for vertices the walk
 * reaches only after wrapping past the ring end.
 */
function getSpanNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    keepMask: GraphDataView<'uint32'>;
    keepRank: GraphDataView<'uint32'>;
    spanEnd: GraphDataView<'uint32'>;
    spanKey: GraphDataView<'uint32'>;
    /**
     * {@link SPAN_LINK_STRIDE} words per vertex for the sweep test. Word 0: the kept vertex whose
     * span holds the input edge that starts at this vertex, when that span dropped a vertex, else
     * `INVALID`. Word 1: this vertex's span end (`INVALID` for a dropped vertex). Word 2: the f32
     * bits of the largest distance of a dropped vertex of this vertex's span from its chord.
     */
    spanLinks: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, vertexCount, ringCount, keepMask, keepRank, ringOffsets} = props;
  const keptVertex = createTransientView(graph, `${id}-kept-vertex`, 'uint32', vertexCount);
  const distanceBits = createTransientView(graph, `${id}-distance-bits`, 'uint32', vertexCount);
  const spanBest = createTransientView(graph, `${id}-span-best`, 'uint32', vertexCount);
  const shared = `const VERTEX_COUNT: u32 = ${vertexCount}u;
const RING_COUNT: u32 = ${ringCount}u;
fn getRing(vertex: u32) -> u32 {
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= vertex) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  return ring;
}
fn getKeptTotal() -> u32 {
  return keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
}
// Number of kept vertices before 'position'.
fn getKeptBefore(position: u32) -> u32 {
  return select(keepRank[keepRankOffset + position], getKeptTotal(), position >= VERTEX_COUNT);
}`;
  const readOnly = (name: string, view: GraphDataView) =>
    ({name, view, type: 'u32', access: 'read'}) as const;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-kept-vertex`,
      operation: OPERATION,
      variant: 'kept-vertex',
      bindings: [
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        {name: 'keptVertex', view: keptVertex, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      body: `if (keepMask[keepMaskOffset + index] != 0u) {
    keptVertex[keptVertexOffset + keepRank[keepRankOffset + index]] = index;
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-ends`,
      operation: OPERATION,
      variant: 'span-ends',
      bindings: [
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        {name: 'spanEnd', view: props.spanEnd, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: shared,
      // The next kept vertex in rank order, unless that one starts the next ring: then the first
      // kept vertex of this ring (a single kept vertex is its own span end).
      body: `var end = ${INVALID};
  if (keepMask[keepMaskOffset + index] != 0u) {
    let ring = getRing(index);
    let ringBegin = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let rank = keepRank[keepRankOffset + index];
    end = keptVertex[keptVertexOffset + getKeptBefore(ringBegin)];
    if (rank + 1u < getKeptTotal()) {
      let candidate = keptVertex[keptVertexOffset + rank + 1u];
      if (candidate < ringEnd) {
        end = candidate;
      }
    }
  }
  spanEnd[spanEndOffset + index] = end;`
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-span-best-clear`,
      operation: OPERATION,
      view: spanBest,
      type: 'u32',
      value: '0u'
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-span-key-clear`,
      operation: OPERATION,
      view: props.spanKey,
      type: 'u32',
      value: INVALID
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-distance`,
      operation: OPERATION,
      variant: 'span-distance',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        readOnly('spanEnd', props.spanEnd),
        {name: 'distanceBits', view: distanceBits, type: 'u32', access: 'read_write'},
        {name: 'spanBest', view: spanBest, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `${shared}
${SPAN_OWNER_WGSL}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`,
      // Distance bits of positive floats order like the floats, so atomicMax finds the farthest.
      body: `var bits = 0u;
  if (keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${INVALID}) {
      let a = vertexAt(owner);
      let p = vertexAt(index);
      let direction = vertexAt(spanEnd[spanEndOffset + owner]) - a;
      let segmentLength = length(direction);
      let offset = p - a;
      let separation = select(
        distance(p, a),
        abs(direction.x * offset.y - direction.y * offset.x) / max(segmentLength, 1e-30),
        segmentLength > 0.0);
      if (separation > 0.0) {
        bits = bitcast<u32>(separation);
        atomicMax(&spanBest[spanBestOffset + owner], bits);
      }
    }
  }
  distanceBits[distanceBitsOffset + index] = bits;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-split`,
      operation: OPERATION,
      variant: 'span-split',
      bindings: [
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        readOnly('distanceBits', distanceBits),
        readOnly('spanBest', spanBest),
        {name: 'spanKey', view: props.spanKey, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `${shared}
${SPAN_OWNER_WGSL}`,
      // Ties keep the vertex the walk from the span start meets first: later indices than the
      // span start come first, wrapped (smaller) indices after them.
      body: `let bits = distanceBits[distanceBitsOffset + index];
  if (bits != 0u && keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${INVALID} && bits == spanBest[spanBestOffset + owner]) {
      atomicMin(&spanKey[spanKeyOffset + owner], select(index + VERTEX_COUNT, index, index > owner));
    }
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-links`,
      operation: OPERATION,
      variant: 'span-links',
      bindings: [
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        readOnly('spanEnd', props.spanEnd),
        readOnly('spanBest', spanBest),
        {name: 'spanLinks', view: props.spanLinks, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `${shared}
${SPAN_OWNER_WGSL}`,
      // The input edge that starts at a vertex lies in the span of that vertex when it is kept, and
      // of the kept vertex before it otherwise. The span dropped a vertex exactly when the vertex
      // after its owner in the ring is not kept.
      body: `var owner = index;
  if (keepMask[keepMaskOffset + index] == 0u) { owner = getSpanOwner(index); }
  var edgeOwner = ${INVALID};
  if (owner != ${INVALID}) {
    let ring = getRing(owner);
    let ringBegin = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (owner >= ringBegin && owner < ringEnd) {
      let after = select(owner + 1u, ringBegin, owner + 1u >= ringEnd);
      if (keepMask[keepMaskOffset + after] == 0u) { edgeOwner = owner; }
    }
  }
  let base = spanLinksOffset + index * ${SPAN_LINK_STRIDE}u;
  spanLinks[base] = edgeOwner;
  spanLinks[base + 1u] = spanEnd[spanEndOffset + index];
  spanLinks[base + 2u] = spanBest[spanBestOffset + index];`
    })
  ];
}

/** WGSL: kept vertex whose span contains the dropped vertex, in the same ring, or INVALID. */
const SPAN_OWNER_WGSL = `
fn getSpanOwner(vertex: u32) -> u32 {
  let ring = getRing(vertex);
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let firstRank = getKeptBefore(ringBegin);
  let endRank = getKeptBefore(ringEnd);
  if (vertex < ringBegin || vertex >= ringEnd || endRank <= firstRank) {
    return ${INVALID};
  }
  // The kept vertex before it in the ring, else the ring's last one (the span wraps over the end).
  let before = keepRank[keepRankOffset + vertex];
  return keptVertex[keptVertexOffset + select(endRank - 1u, before - 1u, before > firstRank)];
}`;

/**
 * One topology round: compact the current simplified rings, find intersecting segment pairs,
 * classify them against the coverage points, count the offending ones and (when `repair`) restore
 * the farthest original vertex of every offending segment's span.
 */
export function getTopologyRoundNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    featureOffsets: GraphDataView<'uint32'>;
    pointIds: GraphDataView<'uint32'>;
    vertexNext: GraphDataView<'uint32'>;
    ownedVertex: GraphDataView<'uint32'>;
    keepPoint: GraphDataView<'uint32'>;
    /** BVH over the input segments (leaf IDs are start-vertex rows), for the sweep test. */
    sweepBVH: {
      nodeMinima: GraphDataView;
      nodeMaxima: GraphDataView;
      leafIds: GraphDataView;
      internalNodeCount: number;
    };
    vertexCount: number;
    ringCount: number;
    pairCapacity: number;
    repair: boolean;
    statsView: GraphDataView<'uint32'>;
    /** Stats word that receives the repairable crossing count of this round. */
    crossingSlot: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, vertexCount, ringCount, pairCapacity, statsView} = props;
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const mask = view('mask', vertexCount);
  const rank = view('rank', vertexCount);
  const compactPositions = createTransientView(
    graph,
    `${id}-compact-positions`,
    'float32x2',
    vertexCount
  );
  const compactRingOffsets = view('compact-ring-offsets', ringCount + 1);
  const compactOverflow = view('compact-overflow', 1);
  const spanEnd = view('span-end', vertexCount);
  const spanSplit = view('span-split', vertexCount);
  const spanLinks = view('span-links', vertexCount * SPAN_LINK_STRIDE);
  const rowInfo = view('row-info', vertexCount * 4);
  const badFlag = view('bad', vertexCount);
  const pairLeft = view('pair-left', pairCapacity);
  const pairRight = view('pair-right', pairCapacity);
  const pairCount = view('pair-count', 1);
  const pairOverflow = view('pair-overflow', 1);
  const pairKinds = view('pair-kinds', pairCapacity);
  const {crossingSlot} = props;

  const nodes: GPUCommandNode<Parameters>[] = [
    createKeepMaskNode<Parameters>(graph, `${id}-keep-mask`, props.pointIds, props.keepPoint, mask),
    ...new GPUScan({
      id: `${id}-rank-scan`,
      input: mask,
      output: rank,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createEmitPositionsNode<Parameters>(graph, {
      id: `${id}-emit-positions`,
      positions: props.positions,
      keepMask: mask,
      keepRank: rank,
      outputPositions: compactPositions,
      vertexCount
    }),
    createEmitOffsetsNode<Parameters>(graph, {
      id: `${id}-emit-offsets`,
      ringOffsets: props.ringOffsets,
      keepMask: mask,
      keepRank: rank,
      outputRingOffsets: compactRingOffsets,
      overflow: compactOverflow,
      vertexCount,
      ringCount,
      capacity: vertexCount
    }),
    // For every kept vertex: the next kept vertex of its ring and the farthest dropped vertex
    // between them (the vertex a Douglas-Peucker step would restore). Instead of one thread walking
    // each span twice (a long arc that simplifies to two points is a serial walk of its length),
    // the next kept vertex comes from the rank order and every dropped vertex scores itself
    // against the span that owns it: two atomic passes pick the maximum distance and, among equal
    // distances, the first vertex in walk order (the old walk's strict `>` rule).
    ...getSpanNodes(graph, {
      id,
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      keepMask: mask,
      keepRank: rank,
      spanEnd,
      spanKey: spanSplit,
      spanLinks,
      vertexCount,
      ringCount
    }),
    // Segment rows of the compacted rings: coverage points of both ends, vertex, ownership.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-row-info`,
      operation: OPERATION,
      variant: 'row-info',
      bindings: [
        {name: 'keepMask', view: mask, type: 'u32', access: 'read'},
        {name: 'keepRank', view: rank, type: 'u32', access: 'read'},
        {name: 'spanEnd', view: spanEnd, type: 'u32', access: 'read'},
        {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
        {name: 'ownedVertex', view: props.ownedVertex, type: 'u32', access: 'read'},
        {name: 'rowInfo', view: rowInfo, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      body: `if (keepMask[keepMaskOffset + index] != 0u) {
    let base = rowInfoOffset + keepRank[keepRankOffset + index] * 4u;
    rowInfo[base] = pointIds[pointIdsOffset + index];
    rowInfo[base + 1u] = pointIds[pointIdsOffset + spanEnd[spanEndOffset + index]];
    rowInfo[base + 2u] = index;
    rowInfo[base + 3u] = ownedVertex[ownedVertexOffset + index];
  }`
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-bad-clear`,
      operation: OPERATION,
      view: badFlag,
      type: 'u32',
      value: '0u'
    }),
    ...new GPUSegmentIntersection({
      id: `${id}-intersect`,
      left: {
        kind: 'polygons',
        positions: compactPositions,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: compactRingOffsets
      },
      output: {
        pairs: {
          leftRows: pairLeft,
          rightRows: pairRight,
          status: {count: pairCount, overflow: pairOverflow}
        },
        kinds: pairKinds
      }
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-classify`,
      operation: OPERATION,
      variant: `classify-${crossingSlot}`,
      bindings: [
        {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
        {name: 'pairRight', view: pairRight, type: 'u32', access: 'read'},
        {name: 'pairKinds', view: pairKinds, type: 'u32', access: 'read'},
        {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
        {name: 'pairOverflow', view: pairOverflow, type: 'u32', access: 'read'},
        {name: 'rowInfo', view: rowInfo, type: 'u32', access: 'read'},
        {name: 'badFlag', view: badFlag, type: 'atomic<u32>', access: 'read_write'},
        {name: 'stats', view: statsView, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: pairCapacity,
      declarations: `const SLOT: u32 = ${crossingSlot}u;`,
      // Pair kinds: 1 proper, 2 touch, 3 collinear touch, 4 overlap, 5 uncertain. Segments that
      // share a coverage point meet legitimately at it; identical segments are one shared edge.
      body: `if (index == 0u && pairOverflow[pairOverflowOffset] != 0u) {
    atomicMax(&stats[statsOffset + 3u], 1u);
  }
  if (index < pairCount[pairCountOffset]) {
    let kind = pairKinds[pairKindsOffset + index];
    let leftBase = rowInfoOffset + pairLeft[pairLeftOffset + index] * 4u;
    let rightBase = rowInfoOffset + pairRight[pairRightOffset + index] * 4u;
    let a0 = rowInfo[leftBase];
    let a1 = rowInfo[leftBase + 1u];
    let b0 = rowInfo[rightBase];
    let b1 = rowInfo[rightBase + 1u];
    let shares = a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
    let identical = (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0);
    let offending = select(!shares, !identical, kind == 4u);
    if (offending) {
      atomicMax(&badFlag[badFlagOffset + rowInfo[leftBase + 2u]], 1u);
      atomicMax(&badFlag[badFlagOffset + rowInfo[rightBase + 2u]], 1u);
      if (rowInfo[leftBase + 3u] != 0u && rowInfo[rightBase + 3u] != 0u) {
        atomicAdd(&stats[statsOffset + SLOT], 1u);
      }
    }
  }`
    }),
    // Sweep test. A shortcut that crosses nothing can still pass over a whole ring (an island and
    // the hole it fills are tied to the arc network by no junction), which moves that ring into
    // the neighbouring polygon. Every kept vertex asks whether it lies strictly inside the area
    // between some span's original chain and its shortcut. Such a vertex is inside a closed
    // region whose boundary is the chain plus one straight segment, so the horizontal line through
    // it meets a chain edge on at least one side: the BVH over the input segments finds every
    // candidate span. A candidate is first rejected when the vertex is farther from the chord than
    // the span's farthest dropped vertex (the area lies within that distance), and only then is the
    // chain walked with the exact point-in-ring test. A swept span is repaired like a crossing.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sweep`,
      operation: OPERATION,
      variant: 'sweep',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'nodeMinima', view: props.sweepBVH.nodeMinima, type: 'f32', access: 'read'},
        {name: 'nodeMaxima', view: props.sweepBVH.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leafIds', view: props.sweepBVH.leafIds, type: 'u32', access: 'read'},
        {name: 'vertexNext', view: props.vertexNext, type: 'u32', access: 'read'},
        {name: 'spanLinks', view: spanLinks, type: 'u32', access: 'read'},
        {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
        {name: 'badFlag', view: badFlag, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
const INTERNAL_NODE_COUNT: u32 = ${props.sweepBVH.internalNodeCount}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${NODE_OVERLAPS_WGSL}
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
// Whether 'point' can lie in the area between the span's chain and its chord: that area is within
// the span's reach (its farthest dropped vertex) of the chord's line. f32, with slack.
fn isWithinSpanReach(point: vec2f, owner: u32, spanEndVertex: u32) -> bool {
  let reach = bitcast<f32>(spanLinks[spanLinksOffset + owner * ${SPAN_LINK_STRIDE}u + 2u]);
  if (!(reach > 0.0)) { return false; }
  let start = vertexAt(owner);
  let direction = vertexAt(spanEndVertex) - start;
  let offset = point - start;
  let chordLength = length(direction);
  let separation = select(
    length(offset),
    abs(direction.x * offset.y - direction.y * offset.x) / max(chordLength, 1e-30),
    chordLength > 0.0);
  let slack = 1e-5 * (abs(offset.x) + abs(offset.y) + abs(direction.x) + abs(direction.y));
  return separation <= reach * 1.001 + slack;
}`,
      body: `let spanEndOfVertex = spanLinks[spanLinksOffset + index * ${SPAN_LINK_STRIDE}u + 1u];
  let pointId = pointIds[pointIdsOffset + index];
  // Only kept vertices are tested: a dropped vertex is not part of the result.
  if (spanEndOfVertex == ${INVALID} || pointId == ${INVALID}) { return; }
  let point = vertexAt(index);
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  let queryMinimum = vec2f(-FLOAT32_MAXIMUM, point.y);
  let queryMaximum = vec2f(FLOAT32_MAXIMUM, point.y);
  var lastOwner = ${INVALID};
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let edge = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      var owner = ${INVALID};
      if (edge < VERTEX_COUNT) { owner = spanLinks[spanLinksOffset + edge * ${SPAN_LINK_STRIDE}u]; }
      if (owner != ${INVALID} && owner != lastOwner) {
        lastOwner = owner;
        let spanEndVertex = spanLinks[spanLinksOffset + owner * ${SPAN_LINK_STRIDE}u + 1u];
        // The ends of the span are on the area's boundary, not inside it.
        if (spanEndVertex != ${INVALID} && pointIds[pointIdsOffset + owner] != pointId &&
            pointIds[pointIdsOffset + spanEndVertex] != pointId &&
            isWithinSpanReach(point, owner, spanEndVertex)) {
          // Winding of the closed ring owner -> chain -> span end -> owner about the point.
          var winding = 0;
          var flags = 0;
          var chainVertex = owner;
          for (var step = 0u; step <= VERTEX_COUNT; step++) {
            var chainNext = vertexNext[vertexNextOffset + chainVertex];
            let isClosing = step > 0u && chainVertex == spanEndVertex;
            if (isClosing) { chainNext = owner; }
            // The single call site of the exact helpers in this kernel.
            let crossing = exactProbeCrossing(probe, vertexAt(chainVertex), vertexAt(chainNext));
            winding += crossing.x;
            flags = flags | crossing.y;
            if (isClosing) { break; }
            chainVertex = chainNext;
          }
          if ((winding & 1) != 0 && flags == 0) {
            atomicOr(&badFlag[badFlagOffset + owner], ${SWEPT_BIT}u);
          }
        }
      }
    }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }`
    }),
    // A shared arc is swept in both rings that hold it: count the owned copy, as for crossings.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sweep-count`,
      operation: OPERATION,
      variant: `sweep-count-${crossingSlot}`,
      bindings: [
        {name: 'badFlag', view: badFlag, type: 'u32', access: 'read'},
        {name: 'ownedVertex', view: props.ownedVertex, type: 'u32', access: 'read'},
        {name: 'stats', view: statsView, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `const SLOT: u32 = ${crossingSlot}u;`,
      body: `if ((badFlag[badFlagOffset + index] & ${SWEPT_BIT}u) != 0u && ownedVertex[ownedVertexOffset + index] != 0u) {
    atomicAdd(&stats[statsOffset + SLOT], 1u);
  }`
    })
  ];
  if (props.repair) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-repair`,
        operation: OPERATION,
        variant: 'repair',
        bindings: [
          {name: 'badFlag', view: badFlag, type: 'u32', access: 'read'},
          {name: 'spanSplit', view: spanSplit, type: 'u32', access: 'read'},
          {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: props.keepPoint, type: 'atomic<u32>', access: 'read_write'},
          {name: 'stats', view: statsView, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        // spanSplit holds the first-in-walk-order key: vertex, or vertex + VERTEX_COUNT when the
        // walk wrapped past the ring end.
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let key = spanSplit[spanSplitOffset + index];
  let split = select(key, key - VERTEX_COUNT, key >= VERTEX_COUNT && key != ${INVALID});
  if (badFlag[badFlagOffset + index] != 0u && key != ${INVALID}) {
    let pointId = pointIds[pointIdsOffset + split];
    if (pointId != ${INVALID}) {
      if (atomicMax(&keepPoint[keepPointOffset + pointId], 1u) == 0u) {
        atomicAdd(&stats[statsOffset + 2u], 1u);
      }
    }
  }`
      })
    );
  }
  return nodes;
}
