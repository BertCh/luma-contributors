// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSaturatedScanTotalNodes} from '../../analysis-core/kernels/scan-total';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createListRankingNodes,
  createListRankingRoundNode,
  getListRankingRoundCount
} from '../line-merge/list-ranking';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSegmentIntersectionMaximumVertexCount,
  GPUSegmentIntersection
} from '../segment-intersection/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPULineGeometryPort,
  GPUPackedGeometryPort,
  GPUVertexCountStatusPort
} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUSharedPaths';

/** Number of `u32` words per span-table row (one row per intersecting segment pair). */
const SPAN_STRIDE = 12;

/** Number of `u32` words per run-table row: vertex start, left line, right line, forward, head span. */
const RUN_STRIDE = 5;

/**
 * Largest `intersectionCapacity` of {@link GPUSharedPaths} on `device`: the span table holds 12
 * `u32` words, 48 bytes, per intersection in one storage binding, so 2796202 on a default device
 * (128 MiB bindings).
 */
export function getGPUSharedPathsMaximumIntersectionCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 48);
}

/** Value written to `runs.leftSourceIds` and `runs.rightSourceIds` slots that hold no run. */
export const GPU_SHARED_PATHS_NONE = 0xffffffff;

/**
 * Caller-owned, capacity-bounded output runs of {@link GPUSharedPaths}, in GeoArrow linestring
 * layout. Run `q` is the vertex run `positions[offsets[q] .. offsets[q + 1])`, ordered along the
 * left line.
 */
export type GPUSharedPathsRuns = {
  /**
   * Row of the `left` linestring of each run (two-input provenance, contributor contract rule 8).
   * Capacity is the length; unused slots hold `GPU_SHARED_PATHS_NONE`. Renamed from `leftLineIds`
   * (breaking, 2026-11-10).
   */
  leftSourceIds: GraphDataView<'uint32'>;
  /**
   * Row of the `right` linestring of each run; same length as `leftSourceIds`. Renamed from
   * `rightLineIds` (breaking, 2026-11-10).
   */
  rightSourceIds: GraphDataView<'uint32'>;
  /**
   * `1` when the right line runs the same way as the left line along the run (Shapely's forward
   * collection), `0` when it runs the opposite way (the backward collection).
   */
  forward: GraphDataView<'uint32'>;
  /**
   * Run-to-vertex offsets with `leftSourceIds.length + 1` entries, first 0. Entries after `count`
   * repeat the written vertex count, so the array is always monotone.
   */
  offsets: GraphDataView<'uint32'>;
  /** Vertices of all runs. Capacity is the length. */
  positions: GraphDataView<'float32x2'>;
};

/** Caller-owned outputs of {@link GPUSharedPaths}. */
export type GPUSharedPathsOutput = {
  /** Output runs. */
  runs: GPUSharedPathsRuns;
  /** Status scalars, one uint32 row each. */
  status: GPUVertexCountStatusPort & {
    /** One-row scalar receiving the number of complete runs written (runs, not vertices). */
    count: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving `1` when any capacity was exceeded. */
    overflow?: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the unclamped number of runs. */
    requiredCount?: GraphDataView<'uint32'>;
    /** Optional one-row count of segment pairs the intersection predicates could not certify. */
    uncertainCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row scalar receiving `1` when the `intersectionCapacity` pair list (an
     * intermediate capacity) overflowed, so runs may be missing. That case also sets `overflow`.
     */
    candidateOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUSharedPaths}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, capacities,
 * `leafCapacity`, `spatialSort`.
 */
export type GPUSharedPathsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'shared-paths'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Left linestrings. */
  left: GPUPackedGeometryPort<GPULineGeometryPort>;
  /** Right linestrings, indexed by a BVH. */
  right: GPUPackedGeometryPort<GPULineGeometryPort>;
  /**
   * Capacity of the internal `GPUSegmentIntersection` pair list (one row per intersecting segment
   * pair, not only the shared ones). When it overflows, runs are produced from the sorted prefix
   * and `output.status.overflow` is set (and `output.status.candidateOverflow` when given). At most
   * {@link getGPUSharedPathsMaximumIntersectionCapacity} (2796202 on a default device);
   * `left.positions` and `right.positions` have at most
   * {@link getGPUSegmentIntersectionMaximumVertexCount} rows each.
   */
  intersectionCapacity: number;
  /** Output runs and status. */
  output: GPUSharedPathsOutput;
  /** Passed to `GPUSegmentIntersection`. */
  spatialSort?: boolean;
  /** Passed to `GPUSegmentIntersection`. */
  leafCapacity?: number;
};

/**
 * Finds the paths that two sets of linestrings share (Shapely `shared_paths`, GeoPandas has no
 * direct name), with the direction of the right line relative to the left line.
 *
 * Pipeline: `GPUSegmentIntersection` (two-sided) lists every intersecting segment pair; the
 * collinear overlaps are the shared spans. Every span is oriented along its left segment and
 * tagged forward or backward by the sign of the product of the two segment directions along the
 * dominant axis (exact). Spans of one (left line, right line, direction) combination that meet
 * end to start, on the same left segment or on consecutive ones, are chained into one run, so a
 * road shared by two networks over many segments is one output row. A run from line pair `(i, j)`
 * is a single polyline along line `i`; points where the run crosses a left vertex are kept and
 * points inside a left segment are not.
 *
 * Output order is deterministic: runs are ordered by the first left segment they cover and then
 * by the pair of right segments. A closed left line is cut at its first vertex, so a fully shared
 * ring is one run starting there. Two coincident right segments report a run each. Overlap is
 * decided with exact predicates; a point shared by two lines is not a path and yields no run.
 * Runs beyond the capacities are dropped as a suffix and `output.status.overflow` is set. Nothing is read
 * back.
 *
 * Chains are sized and written by list ranking (Wyllie pointer jumping, the helper of
 * `GPULineMerge`) in `ceil(log2(intersectionCapacity))` fixed rounds per pass instead of one
 * invocation walking each run: a road shared over 10,000 spans costs the same number of
 * dispatches as one shared over 10 spans, and every output vertex is written by its own
 * invocation. The work is `O(intersectionCapacity * log(intersectionCapacity))`.
 */
export class GPUSharedPaths implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSharedPathsProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of left segment slots (vertices). */
  readonly leftVertexCount: number;
  /** Run capacity. */
  readonly runCapacity: number;
  /** Output vertex capacity. */
  readonly vertexCapacity: number;

  constructor(props: GPUSharedPathsProps) {
    this.id = props.id ?? 'shared-paths';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
    const {runs, status} = props.output;
    for (const [name, lines] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      if (lines.kind !== 'lines') {
        throw new Error(`${id} ${name} must be linestring geometry`);
      }
      validateSinglePackedView(lines.positions, ['float32x2'], `${id} ${name}.positions`);
      validateSinglePackedView(lines.lineOffsets, ['uint32'], `${id} ${name}.lineOffsets`);
      if (lines.lineOffsets.length < 2) {
        throw new Error(
          `${id} ${name}.lineOffsets requires at least one line and a terminal entry`
        );
      }
      if (lines.positions.length < 1) {
        throw new Error(`${id} ${name}.positions must hold at least one vertex`);
      }
    }
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    this.leftVertexCount = props.left.positions.length;
    for (const [name, view] of [
      ['output.runs.leftSourceIds', runs.leftSourceIds],
      ['output.runs.rightSourceIds', runs.rightSourceIds],
      ['output.runs.forward', runs.forward],
      ['output.runs.offsets', runs.offsets]
    ] as const) {
      validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    }
    validateSinglePackedView(runs.positions, ['float32x2'], `${id} output.runs.positions`);
    this.runCapacity = runs.leftSourceIds.length;
    this.vertexCapacity = runs.positions.length;
    if (
      this.runCapacity < 1 ||
      runs.rightSourceIds.length !== this.runCapacity ||
      runs.forward.length !== this.runCapacity
    ) {
      throw new Error(`${id} output.runs ids need equal nonzero length`);
    }
    if (runs.offsets.length !== this.runCapacity + 1) {
      throw new Error(`${id} output.runs.offsets needs capacity + 1 rows`);
    }
    if (this.vertexCapacity < 1) {
      throw new Error(`${id} output.runs.positions must be non-empty`);
    }
    for (const [name, view] of [
      ['count', status.count],
      ['vertexCount', status.vertexCount],
      ['overflow', status.overflow],
      ['requiredCount', status.requiredCount],
      ['requiredVertexCount', status.requiredVertexCount],
      ['uncertainCount', status.uncertainCount],
      ['candidateOverflow', status.candidateOverflow]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.status.${name} needs one row`);
        }
      }
    }
  }

  /** Returns intersection, span, link, run and output nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, runCapacity, vertexCapacity} = this;
    const {left, right} = props;
    const {runs, status} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      left.positions,
      left.lineOffsets,
      right.positions,
      right.lineOffsets,
      runs.leftSourceIds,
      runs.rightSourceIds,
      runs.forward,
      runs.offsets,
      runs.positions,
      status.count,
      status.vertexCount,
      status.overflow,
      status.requiredCount,
      status.requiredVertexCount,
      status.uncertainCount,
      status.candidateOverflow
    ]);
    // Both sides feed the segment tables of the internal segment intersection.
    for (const [name, lines] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      validateMaximumCount(
        id,
        `${name}.positions rows`,
        lines.positions.length,
        getGPUSegmentIntersectionMaximumVertexCount(graph.device)
      );
    }
    validateCapacity(
      id,
      'intersectionCapacity',
      props.intersectionCapacity,
      1,
      getGPUSharedPathsMaximumIntersectionCapacity(graph.device)
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const pairCapacity = props.intersectionCapacity;
    const T = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    // 1. Segment pairs between the two sets of lines.
    const pairLeft = T('pair-left', 'uint32', pairCapacity);
    const pairRight = T('pair-right', 'uint32', pairCapacity);
    const pairCount = T('pair-count', 'uint32', 1);
    const pairOverflow = T('pair-overflow', 'uint32', 1);
    const kinds = T('kinds', 'uint32', pairCapacity);
    const points = T('points', 'float32x2', pairCapacity);
    const endPoints = T('end-points', 'float32x2', pairCapacity);
    const leftRings = T('left-rings', 'uint32', pairCapacity);
    const rightRings = T('right-rings', 'uint32', pairCapacity);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-intersection`,
        left,
        right,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        output: {
          pairs: {
            leftIds: pairLeft,
            rightIds: pairRight,
            status: {count: pairCount, overflow: pairOverflow}
          },
          status: {uncertainCount: status.uncertainCount},
          kinds,
          points,
          endPoints,
          leftRings,
          rightRings
        }
      }).getCommandNodes(graph)
    );

    const geometryWGSL = (prefix: string) => `
fn ${prefix}Vertex(vertex: u32) -> vec2f {
  return vec2f(${prefix}Positions[${prefix}PositionsOffset + vertex * 2u], ${prefix}Positions[${prefix}PositionsOffset + vertex * 2u + 1u]);
}`;
    const keyWGSL = `
// Coordinate of p along direction d on the dominant axis; ascending along d.
fn keyOf(p: vec2f, d: vec2f) -> f32 {
  if (abs(d.x) >= abs(d.y)) { return p.x * select(-1.0, 1.0, d.x > 0.0); }
  return p.y * select(-1.0, 1.0, d.y > 0.0);
}
// Component of d on the dominant axis of reference.
fn axisComponent(d: vec2f, reference: vec2f) -> f32 {
  if (abs(reference.x) >= abs(reference.y)) { return d.x; }
  return d.y;
}`;

    // 2. Span table: validity and direction, then oriented end points and lines.
    // Words: 0 flags (1 overlap, 2 forward), 1 next span, 2 has previous, 3..6 start and end bits,
    // 7 left line, 8 right line, 9 left segment.
    const spanTable = T('span-table', 'uint32', pairCapacity * SPAN_STRIDE);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directions`,
        operation: OPERATION,
        variant: 'directions',
        bindings: [
          {name: 'leftPositions', view: left.positions, type: 'f32', access: 'read'},
          {name: 'rightPositions', view: right.positions, type: 'f32', access: 'read'},
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairRight', view: pairRight, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'kinds', view: kinds, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `${geometryWGSL('left')}${geometryWGSL('right')}${keyWGSL}`,
        body: `let row = spanTableOffset + index * ${SPAN_STRIDE}u;
  var flags = 0u;
  if (index < pairCount[pairCountOffset] && kinds[kindsOffset + index] == 4u) {
    let leftSegment = pairLeft[pairLeftOffset + index];
    let rightSegment = pairRight[pairRightOffset + index];
    let leftDirection = leftVertex(leftSegment + 1u) - leftVertex(leftSegment);
    let rightDirection = rightVertex(rightSegment + 1u) - rightVertex(rightSegment);
    flags = 1u;
    if ((axisComponent(rightDirection, leftDirection) > 0.0) == (axisComponent(leftDirection, leftDirection) > 0.0)) { flags = flags | 2u; }
    spanTable[row + 9u] = leftSegment;
  }
  spanTable[row] = flags;
  spanTable[row + 1u] = 0xffffffffu;
  spanTable[row + 2u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-spans`,
        operation: OPERATION,
        variant: 'spans',
        bindings: [
          {name: 'leftPositions', view: left.positions, type: 'f32', access: 'read'},
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'endPoints', view: endPoints, type: 'f32', access: 'read'},
          {name: 'leftRings', view: leftRings, type: 'u32', access: 'read'},
          {name: 'rightRings', view: rightRings, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `${geometryWGSL('left')}${keyWGSL}`,
        body: `let row = spanTableOffset + index * ${SPAN_STRIDE}u;
  if ((spanTable[row] & 1u) == 0u) { return; }
  let segment = pairLeft[pairLeftOffset + index];
  let direction = leftVertex(segment + 1u) - leftVertex(segment);
  var first = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var second = vec2f(endPoints[endPointsOffset + index * 2u], endPoints[endPointsOffset + index * 2u + 1u]);
  if (keyOf(second, direction) < keyOf(first, direction)) {
    let swap = first;
    first = second;
    second = swap;
  }
  spanTable[row + 3u] = bitcast<u32>(first.x);
  spanTable[row + 4u] = bitcast<u32>(first.y);
  spanTable[row + 5u] = bitcast<u32>(second.x);
  spanTable[row + 6u] = bitcast<u32>(second.y);
  spanTable[row + 7u] = leftRings[leftRingsOffset + index];
  spanTable[row + 8u] = rightRings[rightRingsOffset + index];`
      })
    );

    // 3. Links: the previous and next span of the same left line, right line and direction.
    const spanAccessWGSL = `
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const SPAN_NONE: u32 = 0xffffffffu;
fn spanWord(span: u32, word: u32) -> u32 { return spanTable[spanTableOffset + span * ${SPAN_STRIDE}u + word]; }
fn isSpan(span: u32) -> bool { return span < PAIR_CAPACITY && (spanWord(span, 0u) & 1u) != 0u; }
fn spanSegment(span: u32) -> u32 { return spanWord(span, 9u); }
// Whether span second starts where span first ends, on the same line pair and direction.
fn follows(first: u32, second: u32) -> bool {
  return isSpan(first) && isSpan(second) && first != second &&
    spanWord(first, 7u) == spanWord(second, 7u) && spanWord(first, 8u) == spanWord(second, 8u) &&
    (spanWord(first, 0u) & 2u) == (spanWord(second, 0u) & 2u) &&
    spanWord(first, 5u) == spanWord(second, 3u) && spanWord(first, 6u) == spanWord(second, 4u);
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-links`,
        operation: OPERATION,
        variant: 'links',
        bindings: [
          {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `${spanAccessWGSL}
// In-kernel lower bound on data-dependent keys; GPUGallopingSearch is a batch node and would
// add a pass and a buffer.
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = pairCount[pairCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairLeft[pairLeftOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `if (!isSpan(index)) { return; }
  let segment = spanSegment(index);
  var first = 0u;
  if (segment > 0u) { first = lowerBound(segment - 1u); }
  let last = lowerBound(segment + 2u);
  var hasPrevious = 0u;
  var next = SPAN_NONE;
  for (var other = first; other < last; other++) {
    if (follows(other, index) && spanSegment(other) <= segment) { hasPrevious = 1u; }
    if (next == SPAN_NONE && follows(index, other) && spanSegment(other) >= segment) { next = other; }
  }
  spanTable[spanTableOffset + index * ${SPAN_STRIDE}u + 1u] = next;
  spanTable[spanTableOffset + index * ${SPAN_STRIDE}u + 2u] = hasPrevious;`
      })
    );

    // 4. Runs: every span without a previous span starts one. Spans form chains through their
    // `next` pointers (several chains may share a tail). Chains are sized by Wyllie pointer
    // jumping over `next` (`ceil(log2(pairCapacity))` rounds, each span learning how many chain
    // vertices lie from itself to the end), so no thread walks a chain.
    const runFlags = T('run-flags', 'uint32', pairCapacity);
    const runRanks = T('run-ranks', 'uint32', pairCapacity);
    const vertexCounts = T('vertex-counts', 'uint32', pairCapacity);
    const vertexStarts = T('vertex-starts', 'uint32', pairCapacity);
    // Repeated overlaps can start several runs that share a tail, so their summed vertex counts
    // can exceed uint32 even though the run count is bounded by pairCapacity.
    const vertexTotal = T('vertex-total', 'uint32', 1);
    const chainWGSL = `${spanAccessWGSL}
fn startsRun(span: u32) -> bool { return isSpan(span) && spanWord(span, 2u) == 0u; }
// Whether the end of span is a vertex of the run: not when the chain goes on in the same segment.
fn emitsEnd(span: u32) -> bool {
  let next = spanWord(span, 1u);
  return next == SPAN_NONE || spanSegment(next) != spanSegment(span);
}`;
    const pointerPair = (name: string) =>
      [T(`${name}-a`, 'uint32', pairCapacity), T(`${name}-b`, 'uint32', pairCapacity)] as const;
    const sizePointers = pointerPair('size-pointer');
    const sizeTotals = pointerPair('size-total');
    const emitPointers = pointerPair('emit-pointer');
    const rounds = getListRankingRoundCount(pairCapacity);
    const sizing = createListRankingNodes<Parameters>(graph, {
      id: `${id}-size-round`,
      operation: OPERATION,
      variant: 'size-round',
      length: pairCapacity,
      rounds,
      pointers: sizePointers,
      values: [{name: 'total', views: sizeTotals, rule: 'add'}]
    });
    const sizeTotalFinal = sizeTotals[sizing.finalIndex];
    // Pointer to the first span at or after a span whose end is a run vertex: the "emitting" spans
    // of a chain, which are the ones that give the run its vertices.
    const emitting = createListRankingNodes<Parameters>(graph, {
      id: `${id}-emit-round`,
      operation: OPERATION,
      variant: 'emit-round',
      length: pairCapacity,
      rounds,
      pointers: emitPointers,
      values: []
    });
    const emitFinal = emitPointers[emitting.finalIndex];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-size-init`,
        operation: OPERATION,
        variant: 'size-init',
        bindings: [
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'sizePointer', view: sizePointers[0], type: 'u32', access: 'read_write'},
          {name: 'sizeTotal', view: sizeTotals[0], type: 'u32', access: 'read_write'},
          {name: 'emitPointer', view: emitPointers[0], type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: chainWGSL,
        body: `let valid = isSpan(index);
  sizePointer[sizePointerOffset + index] = select(SPAN_NONE, spanWord(index, 1u), valid);
  sizeTotal[sizeTotalOffset + index] = select(0u, select(0u, 1u, emitsEnd(index)), valid);
  emitPointer[emitPointerOffset + index] = select(SPAN_NONE, select(spanWord(index, 1u), index, emitsEnd(index)), valid);`
      }),
      ...sizing.nodes,
      ...emitting.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-sizes`,
        operation: OPERATION,
        variant: 'run-sizes',
        bindings: [
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'sizeTotal', view: sizeTotalFinal, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read_write'},
          {name: 'vertexCounts', view: vertexCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: chainWGSL,
        body: `let starts = startsRun(index);
  runFlags[runFlagsOffset + index] = select(0u, 1u, starts);
  // The start point plus one vertex per emitting span of the chain.
  vertexCounts[vertexCountsOffset + index] = select(0u, 1u + sizeTotal[sizeTotalOffset + index], starts);`
      }),
      ...new GPUScan({
        id: `${id}-run-scan`,
        input: runFlags,
        output: runRanks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: vertexCounts,
        output: vertexStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: `${id}-vertex`,
        operation: OPERATION,
        counts: vertexCounts,
        offsets: vertexStarts,
        rowCount: pairCapacity,
        total: vertexTotal
      })
    );

    // 5. Run table, capacity clamp and outputs.
    const runTable = T('run-table', 'uint32', pairCapacity * RUN_STRIDE);
    const state = T('state', 'uint32', 8);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-table`,
        operation: OPERATION,
        variant: 'run-table',
        bindings: [
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexStarts', view: vertexStarts, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        body: `if (runFlags[runFlagsOffset + index] == 0u) { return; }
  let row = runTableOffset + runRanks[runRanksOffset + index] * ${RUN_STRIDE}u;
  let source = spanTableOffset + index * ${SPAN_STRIDE}u;
  runTable[row] = vertexStarts[vertexStartsOffset + index];
  runTable[row + 1u] = spanTable[source + 7u];
  runTable[row + 2u] = spanTable[source + 8u];
  runTable[row + 3u] = (spanTable[source] >> 1u) & 1u;
  runTable[row + 4u] = index;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'runFlags', view: runFlags, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'vertexTotal', view: vertexTotal, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read'},
          {name: 'pairOverflow', view: pairOverflow, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LAST: u32 = ${pairCapacity - 1}u;
const RUN_CAPACITY: u32 = ${runCapacity}u;
const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
        body: `let totalRuns = runRanks[runRanksOffset + LAST] + runFlags[runFlagsOffset + LAST];
  let totalVertices = vertexTotal[vertexTotalOffset];
  var low = 0u;
  var high = min(totalRuns, RUN_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    var end = totalVertices;
    if (middle < totalRuns) { end = runTable[runTableOffset + middle * ${RUN_STRIDE}u]; }
    if (end <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = totalVertices;
  if (low < totalRuns) { written = runTable[runTableOffset + low * ${RUN_STRIDE}u]; }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalRuns;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalRuns || totalVertices > VERTEX_CAPACITY || pairOverflow[pairOverflowOffset] != 0u);
  state[stateOffset + 5u] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-runs`,
        operation: OPERATION,
        variant: 'write-runs',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read'},
          {name: 'leftSourceIds', view: runs.leftSourceIds, type: 'u32', access: 'read_write'},
          {name: 'rightSourceIds', view: runs.rightSourceIds, type: 'u32', access: 'read_write'},
          {name: 'forward', view: runs.forward, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: runs.offsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: runCapacity + 1,
        declarations: `const RUN_CAPACITY: u32 = ${runCapacity}u;`,
        body: `if (index < state[stateOffset]) {
    let row = runTableOffset + index * ${RUN_STRIDE}u;
    leftSourceIds[leftSourceIdsOffset + index] = runTable[row + 1u];
    rightSourceIds[rightSourceIdsOffset + index] = runTable[row + 2u];
    forward[forwardOffset + index] = runTable[row + 3u];
    offsets[offsetsOffset + index] = runTable[row];
  } else {
    if (index < RUN_CAPACITY) {
      leftSourceIds[leftSourceIdsOffset + index] = 0xffffffffu;
      rightSourceIds[rightSourceIdsOffset + index] = 0xffffffffu;
      forward[forwardOffset + index] = 0u;
    }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`
      })
    );

    // 6. Vertices. One thread per output vertex row: the row finds its run by binary search over
    // the run table. Row 1 of a run is the first emitting span of the head's chain; row u + 1 is
    // reached from row u by `emitSuccessor`, and rows double their reach each round (row u + 2^j
    // is the 2^j-th successor of row u), so a chain of n vertices costs `O(log n)` rounds and
    // no thread walks it.
    const rowHeads = T('row-heads', 'uint32', vertexCapacity);
    const rowSteps = T('row-steps', 'uint32', vertexCapacity);
    const rowNodes = T('row-nodes', 'uint32', vertexCapacity);
    const roundDistance = T('round-distance', 'uint32', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-heads`,
        operation: OPERATION,
        variant: 'row-heads',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'runTable', view: runTable, type: 'u32', access: 'read'},
          {name: 'vertexTotal', view: vertexTotal, type: 'u32', access: 'read'},
          {name: 'emitFinal', view: emitFinal, type: 'u32', access: 'read'},
          {name: 'rowHeads', view: rowHeads, type: 'u32', access: 'read_write'},
          {name: 'rowSteps', view: rowSteps, type: 'u32', access: 'read_write'},
          {name: 'rowNodes', view: rowNodes, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCapacity,
        declarations: `const SPAN_NONE: u32 = 0xffffffffu;`,
        // Bound the search by the runs actually written (state[0]): the run starts of unwritten
        // rows are not monotone when the uint32 vertex scan wraps.
        body: `var head = SPAN_NONE;
  var step = 0u;
  let totalRuns = state[stateOffset];
  if (index < vertexTotal[vertexTotalOffset] && totalRuns > 0u) {
    // Last run whose first vertex row is at most index.
    var low = 0u;
    var high = totalRuns;
    while (high - low > 1u) {
      let middle = (low + high) / 2u;
      if (runTable[runTableOffset + middle * ${RUN_STRIDE}u] <= index) { low = middle; } else { high = middle; }
    }
    head = runTable[runTableOffset + low * ${RUN_STRIDE}u + 4u];
    step = index - runTable[runTableOffset + low * ${RUN_STRIDE}u];
  }
  rowHeads[rowHeadsOffset + index] = head;
  rowSteps[rowStepsOffset + index] = step;
  rowNodes[rowNodesOffset + index] = select(SPAN_NONE, emitFinal[emitFinalOffset + head], head != SPAN_NONE && step == 1u);`
      }),
      // Successor of an emitting span: the first emitting span after it. It reuses the sizing
      // pointer pair, which is free once the run sizes are known.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit-successor`,
        operation: OPERATION,
        variant: 'emit-successor',
        bindings: [
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'emitFinal', view: emitFinal, type: 'u32', access: 'read'},
          {name: 'successor', view: sizePointers[0], type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: chainWGSL,
        body: `let next = spanWord(index, 1u);
  var result = SPAN_NONE;
  if (isSpan(index) && next != SPAN_NONE) { result = emitFinal[emitFinalOffset + next]; }
  successor[successorOffset + index] = result;`
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-round-distance`,
        operation: OPERATION,
        view: roundDistance,
        type: 'u32',
        value: '1u'
      })
    );
    for (let round = 0; round < rounds; round++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-row-round-${round}`,
          operation: OPERATION,
          variant: 'row-round',
          bindings: [
            {name: 'sizeTotal', view: sizeTotalFinal, type: 'u32', access: 'read'},
            {name: 'rowHeads', view: rowHeads, type: 'u32', access: 'read'},
            {name: 'rowSteps', view: rowSteps, type: 'u32', access: 'read'},
            {name: 'roundDistance', view: roundDistance, type: 'u32', access: 'read'},
            {name: 'successor', view: sizePointers[round % 2], type: 'u32', access: 'read'},
            {name: 'rowNodes', view: rowNodes, type: 'u32', access: 'read_write'}
          ],
          invocationCount: vertexCapacity,
          declarations: 'const SPAN_NONE: u32 = 0xffffffffu;',
          body: `let head = rowHeads[rowHeadsOffset + index];
  if (head == SPAN_NONE) { return; }
  let step = rowSteps[rowStepsOffset + index];
  let distance = roundDistance[roundDistanceOffset];
  // Row u = step - 1 of a chain with sizeTotal rows is filled in the round where distance <= u < 2 * distance.
  if (step <= distance || step - 1u >= 2u * distance || step > sizeTotal[sizeTotalOffset + head]) { return; }
  rowNodes[rowNodesOffset + index] = successor[successorOffset + rowNodes[rowNodesOffset + index - distance]];`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-row-distance-${round}`,
          operation: OPERATION,
          variant: 'row-distance',
          bindings: [
            {name: 'roundDistance', view: roundDistance, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: 'roundDistance[roundDistanceOffset] = roundDistance[roundDistanceOffset] * 2u;'
        }),
        createListRankingRoundNode<Parameters>(
          graph,
          {
            id: `${id}-row-pointer`,
            operation: OPERATION,
            variant: 'row-pointer',
            length: pairCapacity,
            pointers: sizePointers,
            values: []
          },
          round
        )
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-vertices`,
        operation: OPERATION,
        variant: 'write-vertices',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'spanTable', view: spanTable, type: 'u32', access: 'read'},
          {name: 'runRanks', view: runRanks, type: 'u32', access: 'read'},
          {name: 'rowHeads', view: rowHeads, type: 'u32', access: 'read'},
          {name: 'rowSteps', view: rowSteps, type: 'u32', access: 'read'},
          {name: 'rowNodes', view: rowNodes, type: 'u32', access: 'read'},
          {name: 'outPositions', view: runs.positions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCapacity,
        declarations: chainWGSL,
        body: `let head = rowHeads[rowHeadsOffset + index];
  if (head == SPAN_NONE || runRanks[runRanksOffset + head] >= state[stateOffset]) { return; }
  let step = rowSteps[rowStepsOffset + index];
  // Row 0 is the start of the head span; later rows are the ends of the emitting spans in order.
  let span = select(rowNodes[rowNodesOffset + index], head, step == 0u);
  let word = select(5u, 3u, step == 0u);
  outPositions[outPositionsOffset + index * 2u] = bitcast<f32>(spanWord(span, word));
  outPositions[outPositionsOffset + index * 2u + 1u] = bitcast<f32>(spanWord(span, word + 1u));`
      })
    );

    // 6. Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'}
    ];
    const scalars: [string, GraphDataView<'uint32'> | undefined, number][] = [
      ['count', status.count, 0],
      ['vertexCount', status.vertexCount, 1],
      ['requiredCount', status.requiredCount, 2],
      ['requiredVertexCount', status.requiredVertexCount, 3],
      ['overflow', status.overflow, 4],
      ['candidateOverflow', status.candidateOverflow, 5]
    ];
    for (const [name, view] of scalars) {
      if (view) {
        scalarBindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scalars`,
        operation: OPERATION,
        variant: 'scalars',
        bindings: scalarBindings,
        invocationCount: 1,
        body: scalars
          .filter(([, view]) => view)
          .map(([name, , word]) => `${name}[${name}Offset] = state[stateOffset + ${word}u];`)
          .join('\n  ')
      })
    );
    return nodes;
  }
}
