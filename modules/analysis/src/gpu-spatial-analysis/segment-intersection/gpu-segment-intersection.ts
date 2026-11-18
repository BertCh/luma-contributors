// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotSinglePartGeometryPort} from '../../analysis-core/contracts/ports';

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createScanCarryNode} from '../../analysis-core/kernels/scan-total';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotJoinGeometry,
  snapshotJoinPairs,
  snapshotJoinStatus
} from '../spatial-join/spatial-join-construction-snapshot';
import {
  validateSinglePackedView,
  validateMaximumCount,
  validateRequirement,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSegmentGeometry, GPUSegmentIntersectionOutput} from './segment-intersection-types';
import {
  getSegmentTableAccessorsWGSL,
  getGPUSegmentIntersectionMaximumVertexCount,
  SEGMENT_PREDICATES_WGSL,
  SEGMENT_TABLE_STRIDE
} from './segment-intersection-wgsl';
import {getSegmentBVHNodes} from './segment-bvh';
import {createSegmentTableNode, getSegmentSlotCount} from './segment-table';
import {NODE_OVERLAPS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getNextPowerOfTwo} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {validateSegmentIntersectionScratch} from '../segment-intersection/segment-storage-validation';

const OPERATION = 'GPUSegmentIntersection';

/** Vertex count from which `spatialSort` defaults to on. */
const SPATIAL_SORT_MINIMUM_SLOTS = 1024;

/**
 * Properties for {@link GPUSegmentIntersection}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, `sameFeatureOnly`,
 * `leafCapacity`, whether `right` is given, and which optional views exist.
 */
export type GPUSegmentIntersectionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'segment-intersection'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Left segments. Output rows are left segment IDs (start-vertex indices of `left`). Each side has
   * at most {@link getGPUSegmentIntersectionMaximumVertexCount} `positions` rows (4194304 on a default
   * device).
   */
  left: GPUSegmentGeometry;
  /**
   * Right segments, indexed by a BVH. Omit for self mode, where `left` is intersected with itself.
   * The two sides may use different geometry kinds.
   */
  right?: GPUSegmentGeometry;
  /**
   * Self mode only: keep only pairs whose segments belong to the same feature. This keeps the
   * output small when the question is per-feature validity rather than overlap between features.
   * Defaults to `false`.
   */
  sameFeatureOnly?: boolean;
  /**
   * Compile-time. Reorders the right segments along a Morton (Z-order) curve before the BVH build
   * and sorts each left segment's hits afterwards. The output is identical either way; only cost
   * changes. Without it, a source order that jumps around the plane makes upper BVH nodes cover
   * everything and every probe visits most of the tree. Defaults to `true` from 1024 vertices on,
   * and `false` below, where the extra passes cost more than they save. On overflow the last
   * partly written left segment may hold an arbitrary subset of its hits when this is enabled.
   */
  spatialSort?: boolean;
  /** Power-of-two BVH leaf slots over the right segments. Defaults to the next power of two of its vertex count. */
  leafCapacity?: number;
  /**
   * Caller-owned outputs. `output.pairs` holds the `(left, right)` segment pairs, sorted by left
   * then right segment row, with the capacity `pairs.leftRows.length`. `pairs.status.count` is
   * `min(total, capacity)`; `pairs.status.overflow` is 1 only when total exceeds pair capacity.
   * `pairs.status.candidateOverflow` reports BVH leaf loss and is required when leafCapacity is
   * below the right vertex-slot count (left in self mode). Both flags are rewritten on reuse.
   * The output holds the sorted prefix of the retained pairs. The optional per-pair columns and
   * `status.uncertainCount` are described by {@link GPUSegmentIntersectionOutput}.
   */
  output: GPUSegmentIntersectionOutput;
};

/**
 * Finds every intersecting pair of line segments between two geometries, or within one.
 *
 * Pipeline: a segment table per side (one row per vertex, so segment IDs are vertex indices),
 * a `GPUBVH` over the right segment bounds, and a counted and scanned probe from every left segment
 * that classifies each candidate exactly (the count pass and the write pass each classify a
 * candidate once; the write pass records the kind, and the kind, uncertain-count and point
 * columns are derived from that record, with no third classification). Candidates come out ordered by left
 * segment and then right segment (BVH leaves keep source order), so the output is sorted and
 * deterministic with no sort pass, and a capacity overflow keeps a consistent sorted prefix.
 * Nothing is read back.
 *
 * Classification uses exact orientation signs (an f32 filter with an exact integer fallback), so
 * a pair is `proper`, `touch`, `collinearTouch` or `overlap` exactly, for finite coordinates whose
 * products stay within 200 binary orders of magnitude of each other. A pair the predicates cannot certify is listed as `uncertain` and counted
 * in `uncertainCount`; it is never dropped. The reported intersection point of a `proper`
 * crossing is rounded to f32; touch and overlap points are input vertices, reported exactly.
 *
 * **Two-sided mode** (`right` given) lists every `(left, right)` segment pair: turf `lineIntersect`,
 * and the segment-level step of line/polygon overlay and noding.
 *
 * **Self mode** (`right` omitted) lists each unordered pair once with `left < right`, within one
 * geometry (turf `kinks`, self-intersection tests, `gpu-network` noding). Segments that follow
 * each other in the same ring or linestring are adjacent and are skipped, because meeting at the
 * shared vertex is not an intersection; an adjacent pair is still listed when it `overlap`s (a
 * spike that doubles back). The successor of a segment is the next non-degenerate segment, so a
 * repeated vertex does not make its neighbors count as non-adjacent. A closed linestring or a
 * polygon ring is cyclic. A segment that touches a non-adjacent segment of its own ring is listed.
 *
 * Skipped segments never intersect anything: zero-length segments, segments with non-finite
 * coordinates, polygon rings with fewer than three vertices and linestrings with one vertex.
 * Segment slots are vertices, so the cost scales with the vertex count. The self-mode successor
 * search looks at most 1024 vertices ahead.
 *
 * Output beyond `pairs.count` is unspecified.
 */
export class GPUSegmentIntersection implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSegmentIntersectionProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Whether the contributor intersects `left` with itself. */
  readonly isSelfMode: boolean;
  /** Number of left segment slots (vertices). */
  readonly leftSlotCount: number;
  /** Number of right segment slots (vertices). */
  readonly rightSlotCount: number;
  /** Pair capacity. */
  readonly pairCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether segments are Morton sorted before the BVH build. */
  readonly spatialSort: boolean;

  constructor(props: GPUSegmentIntersectionProps) {
    const contextProps = snapshotConstructionFields(props, ['id', 'spatialContext']);
    this.id = contextProps.id ?? 'segment-intersection';
    const spatialContext =
      contextProps.spatialContext &&
      snapshotConstructionSpatialContext(contextProps.spatialContext);
    // Reject unsupported coordinates before evaluating geometry or output getters.
    this.spatialContext = resolveGPUSpatialContext(this.id, spatialContext);
    const snapshot = snapshotConstructionFields(props, [
      'left',
      'right',
      'sameFeatureOnly',
      'spatialSort',
      'leafCapacity',
      'output'
    ]);
    const left = snapshotSinglePartGeometryPort(this.id, 'left', snapshot.left);
    const right =
      snapshot.right && snapshotSinglePartGeometryPort(this.id, 'right', snapshot.right);
    const output = snapshotConstructionFields(snapshot.output, [
      'pairs',
      'status',
      'kinds',
      'points',
      'endPoints',
      'leftFeatures',
      'rightFeatures',
      'leftRings',
      'rightRings'
    ]);
    props = Object.freeze({
      ...contextProps,
      ...snapshot,
      spatialContext,
      left: snapshotJoinGeometry(left),
      ...(right ? {right: snapshotJoinGeometry(right)} : {}),
      output: Object.freeze({
        ...output,
        pairs: snapshotJoinPairs(output.pairs),
        ...(output.status ? {status: snapshotJoinStatus(output.status)} : {})
      })
    });
    this.props = props;
    const {id} = this;
    this.isSelfMode = !props.right;
    for (const [name, geometry] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      if (!geometry) {
        continue;
      }
      validateSinglePackedView(geometry.positions, ['float32x2'], `${id} ${name}.positions`);
      const offsetViews =
        geometry.kind === 'lines'
          ? [['lineOffsets', geometry.lineOffsets] as const]
          : [
              ['featureOffsets', geometry.featureOffsets] as const,
              ['polygonOffsets', geometry.polygonOffsets] as const,
              ['ringOffsets', geometry.ringOffsets] as const
            ];
      for (const [offsetName, view] of offsetViews) {
        validatePackedUint32View(view, `${id} ${name}.${offsetName}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name}.${offsetName} requires a terminal entry`);
        }
      }
      if (geometry.positions.length < 1) {
        throw new Error(`${id} ${name} must contain at least one vertex`);
      }
    }
    if (props.sameFeatureOnly && !this.isSelfMode) {
      throw new Error(`${id} sameFeatureOnly requires self mode (omit right)`);
    }
    this.leftSlotCount = getSegmentSlotCount(props.left);
    this.rightSlotCount = getSegmentSlotCount(props.right ?? props.left);
    this.spatialSort = props.spatialSort ?? this.rightSlotCount >= SPATIAL_SORT_MINIMUM_SLOTS;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(this.rightSlotCount);
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    const {pairs} = props.output;
    validateRequirement(
      id,
      Boolean(pairs.leftRows && pairs.rightRows) && !('leftIds' in pairs) && !('rightIds' in pairs),
      'pairs.leftRows and pairs.rightRows are required; leftIds and rightIds are unsupported application-ID names'
    );
    validateRequirement(
      id,
      this.leafCapacity >= this.rightSlotCount || pairs.status.candidateOverflow !== undefined,
      'output.pairs.status.candidateOverflow is required when leafCapacity is below the right segment slot count'
    );
    validatePackedUint32View(pairs.leftRows, `${id} pairs.leftRows`);
    validatePackedUint32View(pairs.rightRows, `${id} pairs.rightRows`);
    this.pairCapacity = pairs.leftRows.length;
    if (this.pairCapacity < 1 || pairs.rightRows.length !== this.pairCapacity) {
      throw new Error(`${id} pairs.leftRows and pairs.rightRows must have equal nonzero length`);
    }
    for (const [name, view] of [
      ['count', pairs.status.count],
      ['overflow', pairs.status.overflow],
      ['candidateOverflow', pairs.status.candidateOverflow],
      ['requiredCount', pairs.status.requiredCount],
      ['status.uncertainCount', props.output.status?.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    for (const [name, view] of [
      ['kinds', props.output.kinds],
      ['leftFeatures', props.output.leftFeatures],
      ['rightFeatures', props.output.rightFeatures],
      ['leftRings', props.output.leftRings],
      ['rightRings', props.output.rightRings]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== this.pairCapacity) {
          throw new Error(`${id} ${name} length must equal the pair capacity`);
        }
      }
    }
    for (const [name, view] of [
      ['points', props.output.points],
      ['endPoints', props.output.endPoints]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32x2'], `${id} ${name}`);
        if (view.length !== this.pairCapacity) {
          throw new Error(`${id} ${name} length must equal the pair capacity`);
        }
      }
    }
  }

  private getInputViews(): GraphDataView[] {
    const views: GraphDataView[] = [];
    for (const geometry of [this.props.left, this.props.right]) {
      if (!geometry) {
        continue;
      }
      views.push(geometry.positions);
      if (geometry.kind === 'lines') {
        views.push(geometry.lineOffsets);
      } else {
        views.push(geometry.featureOffsets, geometry.polygonOffsets, geometry.ringOffsets);
      }
    }
    return views;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {pairs, status, ...columns} = this.props.output;
    const uncertainCount = status?.uncertainCount;
    return [
      pairs.leftRows,
      pairs.rightRows,
      pairs.status.count,
      pairs.status.overflow,
      pairs.status.candidateOverflow,
      pairs.status.requiredCount,
      uncertainCount,
      columns.kinds,
      columns.points,
      columns.endPoints,
      columns.leftFeatures,
      columns.rightFeatures,
      columns.leftRings,
      columns.rightRings
    ];
  }

  /** Returns table, BVH, probe, scan, classification, column and scalar nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, isSelfMode, leftSlotCount, rightSlotCount, pairCapacity} = this;
    const {pairs} = props.output;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...this.getInputViews(), ...this.getOutputViews()].map((view, index) => [
          String(index),
          view
        ])
      )
    );
    for (const [name, geometry] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      if (geometry) {
        validateMaximumCount(
          id,
          `${name}.positions rows`,
          geometry.positions.length,
          getGPUSegmentIntersectionMaximumVertexCount(graph.device)
        );
      }
    }
    validateSegmentIntersectionScratch(
      graph,
      id,
      leftSlotCount,
      rightSlotCount,
      pairCapacity,
      this.leafCapacity,
      this.spatialSort
    );
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Segment tables, with bounds for the indexed (right) side.
    const rightGeometry = props.right ?? props.left;
    const rightTable = createTransientView(
      graph,
      `${id}-right-table`,
      'uint32',
      rightSlotCount * SEGMENT_TABLE_STRIDE
    );
    const minima = createTransientView(graph, `${id}-bounds-minima`, 'float32x2', rightSlotCount);
    const maxima = createTransientView(graph, `${id}-bounds-maxima`, 'float32x2', rightSlotCount);
    nodes.push(
      createSegmentTableNode<Parameters>(graph, {
        id: `${id}-right-table`,
        operation: OPERATION,
        geometry: rightGeometry,
        table: rightTable,
        minima,
        maxima
      })
    );
    let leftTable = rightTable;
    if (!isSelfMode) {
      leftTable = createTransientView(
        graph,
        `${id}-left-table`,
        'uint32',
        leftSlotCount * SEGMENT_TABLE_STRIDE
      );
      nodes.push(
        createSegmentTableNode<Parameters>(graph, {
          id: `${id}-left-table`,
          operation: OPERATION,
          geometry: props.left,
          table: leftTable
        })
      );
    }

    // BVH over right segment bounds, Morton sorted for spatial coherence when requested.
    const {bvh, nodes: bvhNodes} = getSegmentBVHNodes(graph, {
      id,
      operation: OPERATION,
      minima,
      maxima,
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort
    });
    nodes.push(...bvhNodes);

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', leftSlotCount);
    const offsets = createTransientView(graph, `${id}-offsets`, 'uint32', leftSlotCount);

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
        invocationCount: 4,
        body: 'state[stateOffset + index] = 0u;'
      })
    );

    const pairDeclarations = `${SEGMENT_PREDICATES_WGSL}
${getSegmentTableAccessorsWGSL('left')}
${getSegmentTableAccessorsWGSL('right')}
const SELF: bool = ${isSelfMode};
const SAME_FEATURE: bool = ${Boolean(props.sameFeatureOnly)};
const RIGHT_COUNT: u32 = ${rightSlotCount}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
fn pairHit(left: u32, right: u32, withGeometry: bool) -> SegmentHit {
  var none = SegmentHit(KIND_NONE, vec2f(0.0), vec2f(0.0), 0u);
  // The ID test needs no table read, so it goes first: half of all self-mode candidates fail it.
  if (SELF && right <= left) { return none; }
  if (!leftValid(left) || !rightValid(right)) { return none; }
  if (SAME_FEATURE && leftFeature(left) != rightFeature(right)) { return none; }
  var hit = classifySegments(leftStart(left), leftEnd(left), rightStart(right), rightEnd(right), withGeometry);
  if (SELF && (hit.kind == KIND_TOUCH || hit.kind == KIND_COLLINEAR_TOUCH) &&
      (leftSuccessor(left) == right || rightSuccessor(right) == left)) {
    hit.kind = KIND_NONE;
  }
  return hit;
}
${NODE_OVERLAPS_WGSL}`;
    const probeBindings: WGSLKernelBinding[] = [
      {name: 'leftTable', view: leftTable, type: 'u32', access: 'read'},
      {name: 'rightTable', view: rightTable, type: 'u32', access: 'read'},
      {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'}
    ];
    // Self mode with a Morton-sorted tree: thread `i` probes with the segment at leaf slot `i`, so
    // the threads of a workgroup start from neighboring leaves and walk nearly the same nodes (one
    // coherent traversal, shared node reads) whatever order the input lists its segments in.
    // Results go to the probe segment's own row, so outputs equal those of row-order probing.
    // Requires every segment to own a leaf (the default leaf capacity does).
    const probeInLeafOrder = isSelfMode && this.spatialSort && this.leafCapacity >= leftSlotCount;
    const probeBody = (leafBody: string, prologue: string, epilogue: string) => `${
      probeInLeafOrder
        ? `let leftRow = leafIds[leafIdsOffset + index];
  if (leftRow >= RIGHT_COUNT) {
    return;
  }`
        : 'let leftRow = index;'
    }
  ${prologue}
  if (leftValid(leftRow)) {
    let start = leftStart(leftRow);
    let end = leftEnd(leftRow);
    let queryMinimum = min(start, end);
    let queryMaximum = max(start, end);
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          let hit = pairHit(leftRow, rightRow, false);
          if (hit.kind != KIND_NONE) {
            ${leafBody}
          }
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${epilogue}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          ...probeBindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: leftSlotCount,
        declarations: pairDeclarations,
        body: probeBody(
          'found = found + 1u;',
          'var found = 0u;',
          'counts[countsOffset + leftRow] = found;'
        )
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'total',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `state[stateOffset] = offsets[offsetsOffset + ${leftSlotCount - 1}u] + counts[countsOffset + ${leftSlotCount - 1}u];`
      })
    );
    // A pair total of 2^32 or more wraps in the uint32 scan; saturate it so overflow is reported (S-8).
    nodes.push(
      createScanCarryNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        counts,
        offsets,
        rowCount: leftSlotCount,
        total: state
      })
    );
    // Kinds, uncertain counts and points all derive from one classification per pair: the write
    // pass records each pair's kind word, and later passes read it instead of classifying again.
    const {kinds, points, endPoints} = props.output;
    const uncertainCount = props.output.status?.uncertainCount;
    const needsHitCodes = Boolean(kinds || points || endPoints || uncertainCount);
    const hitCodes = needsHitCodes
      ? createTransientView(graph, `${id}-hit-codes`, 'uint32', pairCapacity)
      : undefined;
    const writeBindings: WGSLKernelBinding[] = [
      ...probeBindings,
      {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
      {name: 'pairRight', view: pairs.rightRows, type: 'u32', access: 'read_write'}
    ];
    if (hitCodes) {
      // The 8 storage bindings of a default device are full, so pairLeft is expanded afterwards.
      writeBindings.push({name: 'hitCodes', view: hitCodes, type: 'u32', access: 'read_write'});
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write`,
        operation: OPERATION,
        variant: 'write',
        bindings: writeBindings,
        invocationCount: leftSlotCount,
        declarations: pairDeclarations,
        body: probeBody(
          `let slot = offsets[offsetsOffset + leftRow] + found;
            if (slot < PAIR_CAPACITY) {
              pairRight[pairRightOffset + slot] = rightRow;
              ${hitCodes ? 'hitCodes[hitCodesOffset + slot] = hit.kind | (hit.selector << 8u);' : ''}
            }
            found = found + 1u;`,
          'var found = 0u;',
          ''
        )
      })
    );

    // Left segment of every pair slot: the last left row whose first slot is at or before it.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand-left`,
        operation: OPERATION,
        variant: 'expand-left',
        bindings: [
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'pairLeft', view: pairs.leftRows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const LEFT_COUNT: u32 = ${leftSlotCount}u;`,
        body: `if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  var low = 0u;
  var high = LEFT_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
  }
  pairLeft[pairLeftOffset + index] = max(low, 1u) - 1u;`
      })
    );

    if (this.spatialSort) {
      // Morton-ordered leaves emit each left segment's hits out of order: sort every range.
      const sortBindings: WGSLKernelBinding[] = [
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'pairRight', view: pairs.rightRows, type: 'u32', access: 'read_write'}
      ];
      if (hitCodes) {
        sortBindings.push({name: 'hitCodes', view: hitCodes, type: 'u32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sort-hits`,
          operation: OPERATION,
          variant: 'sort-hits',
          bindings: sortBindings,
          invocationCount: leftSlotCount,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const GAPS = array<u32, 8>(701u, 301u, 132u, 57u, 23u, 10u, 4u, 1u);`,
          body: `let first = offsets[offsetsOffset + index];
  let last = min(first + counts[countsOffset + index], PAIR_CAPACITY);
  if (last <= first + 1u) { return; }
  // Shell sort with Ciura gaps; ranges are short unless one segment crosses very many others.
  for (var gapIndex = 0u; gapIndex < 8u; gapIndex++) {
    let gap = GAPS[gapIndex];
    for (var slot = first + gap; slot < last; slot++) {
      let value = pairRight[pairRightOffset + slot];
      ${hitCodes ? 'let valueCode = hitCodes[hitCodesOffset + slot];' : ''}
      var hole = slot;
      loop {
        if (hole < first + gap) { break; }
        let earlier = pairRight[pairRightOffset + hole - gap];
        if (earlier <= value) { break; }
        pairRight[pairRightOffset + hole] = earlier;
        ${hitCodes ? 'hitCodes[hitCodesOffset + hole] = hitCodes[hitCodesOffset + hole - gap];' : ''}
        hole = hole - gap;
      }
      pairRight[pairRightOffset + hole] = value;
      ${hitCodes ? 'hitCodes[hitCodesOffset + hole] = valueCode;' : ''}
    }
  }`
        })
      );
    }

    // Per-pair columns, derived from the recorded kind words.
    if (hitCodes && (kinds || uncertainCount)) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'hitCodes', view: hitCodes, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
      ];
      if (kinds) {
        bindings.push({name: 'kinds', view: kinds, type: 'u32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-kinds`,
          operation: OPERATION,
          variant: 'kinds',
          bindings,
          invocationCount: pairCapacity,
          declarations: `${SEGMENT_PREDICATES_WGSL}
const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(atomicLoad(&state[stateOffset]), PAIR_CAPACITY)) { return; }
  let kind = hitCodes[hitCodesOffset + index] & 0xffu;
  if (kind == KIND_UNCERTAIN) { atomicAdd(&state[stateOffset + 1u], 1u); }
  ${kinds ? 'kinds[kindsOffset + index] = kind;' : ''}`
        })
      );
    }
    if (hitCodes && (points || endPoints)) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'pairLeft', view: pairs.leftRows, type: 'u32', access: 'read'},
        {name: 'pairRight', view: pairs.rightRows, type: 'u32', access: 'read'},
        {name: 'leftTable', view: leftTable, type: 'u32', access: 'read'},
        {name: 'rightTable', view: rightTable, type: 'u32', access: 'read'},
        {name: 'hitCodes', view: hitCodes, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read'}
      ];
      if (points) {
        bindings.push({name: 'points', view: points, type: 'f32', access: 'read_write'});
      }
      if (endPoints) {
        bindings.push({name: 'endPoints', view: endPoints, type: 'f32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-points`,
          operation: OPERATION,
          variant: 'points',
          bindings,
          invocationCount: pairCapacity,
          declarations: `${SEGMENT_PREDICATES_WGSL}
${getSegmentTableAccessorsWGSL('left')}
${getSegmentTableAccessorsWGSL('right')}
const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  let code = hitCodes[hitCodesOffset + index];
  let left = pairLeft[pairLeftOffset + index];
  let right = pairRight[pairRightOffset + index];
  let hit = getSegmentHitGeometry(code & 0xffu, code >> 8u, leftStart(left), leftEnd(left), rightStart(right), rightEnd(right));
  ${points ? 'points[pointsOffset + index * 2u] = hit.point.x; points[pointsOffset + index * 2u + 1u] = hit.point.y;' : ''}
  ${endPoints ? 'endPoints[endPointsOffset + index * 2u] = hit.endPoint.x; endPoints[endPointsOffset + index * 2u + 1u] = hit.endPoint.y;' : ''}`
        })
      );
    }
    for (const [side, ids, table, featureColumn, ringColumn] of [
      ['left', pairs.leftRows, leftTable, props.output.leftFeatures, props.output.leftRings],
      ['right', pairs.rightRows, rightTable, props.output.rightFeatures, props.output.rightRings]
    ] as const) {
      if (!featureColumn && !ringColumn) {
        continue;
      }
      const bindings: WGSLKernelBinding[] = [
        {name: 'ids', view: ids, type: 'u32', access: 'read'},
        {name: `${side}Table`, view: table, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read'}
      ];
      if (featureColumn) {
        bindings.push({
          name: 'featureColumn',
          view: featureColumn,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (ringColumn) {
        bindings.push({name: 'ringColumn', view: ringColumn, type: 'u32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${side}-columns`,
          operation: OPERATION,
          variant: `${side}-columns`,
          bindings,
          invocationCount: pairCapacity,
          declarations: `const SEGMENT_NONE: u32 = 0xffffffffu;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
${getSegmentTableAccessorsWGSL(side)}`,
          body: `if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  let row = ids[idsOffset + index];
  ${featureColumn ? `featureColumn[featureColumnOffset + index] = ${side}Feature(row);` : ''}
  ${ringColumn ? `ringColumn[ringColumnOffset + index] = ${side}Ring(row);` : ''}`
        })
      );
    }

    // Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'}
    ];
    if (pairs.status.candidateOverflow) {
      scalarBindings.push({name: 'bvhOverflow', view: bvh.overflow, type: 'u32', access: 'read'});
    }
    const scalars: [string, GraphDataView<'uint32'> | undefined][] = [
      ['count', pairs.status.count],
      ['overflow', pairs.status.overflow],
      ['candidateOverflow', pairs.status.candidateOverflow],
      ['requiredCount', pairs.status.requiredCount],
      ['uncertainCount', uncertainCount]
    ];
    for (const [name, view] of scalars) {
      if (view) {
        scalarBindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    const has = (name: string) => scalars.some(([key, view]) => key === name && view);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: scalarBindings,
        invocationCount: 1,
        declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `let total = state[stateOffset];
  let overflowed = total > PAIR_CAPACITY;
  ${has('count') ? 'count[countOffset] = min(total, PAIR_CAPACITY);' : ''}
  ${has('overflow') ? 'overflow[overflowOffset] = select(0u, 1u, overflowed);' : ''}
  ${has('candidateOverflow') ? 'candidateOverflow[candidateOverflowOffset] = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);' : ''}
  ${has('requiredCount') ? 'requiredCount[requiredCountOffset] = total;' : ''}
  ${has('uncertainCount') ? 'uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];' : ''}`
      })
    );
    return nodes;
  }
}
