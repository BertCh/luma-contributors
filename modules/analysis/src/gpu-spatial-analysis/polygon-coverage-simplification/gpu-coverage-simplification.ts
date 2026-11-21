// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createKeepMaskNode,
  createEmitPositionsNode,
  createEmitOffsetsNode,
  getRingCollapseNodes,
  getTopologyRoundNodes
} from './coverage-simplification-passes';
import {resolveCoverageSnapGridSize} from '../polygon-coverage-topology/coverage-snap-settings';
import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getNextPowerOfTwo,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  getKeyGroupNodes,
  getKeyPairSortNodes,
  KEY_PAIR_INVALID
} from '../spatial-weights/key-pair-grouping';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../segment-intersection/index';
import {
  GPULineSimplification,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  type GPULineSimplificationProps
} from '../line-simplification/index';
import {createSegmentTableNode} from '../segment-intersection/segment-table';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import {SEGMENT_TABLE_STRIDE} from '../segment-intersection/segment-intersection-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';

import {
  createCoverageSnapRangeNodes,
  getCoverageSnapKeyWGSL
} from '../polygon-coverage-topology/coverage-topology-kernels';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

/** Spatial contexts supported by this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUCoverageSimplification';
const INVALID = `${KEY_PAIR_INVALID}u`;
/** Bit of a vertex's `prev` entry that marks the first vertex of its ring. */
const RING_BEGIN_BIT = '0x80000000u';
/** Default number of detect-and-repair iterations of the topology pass. */
const DEFAULT_TOPOLOGY_ITERATION_COUNT = 4;
/** Number of rows `output.topologyStatistics` must hold. */
export const GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATISTICS_LENGTH = 4;
const TOPOLOGY_STATS_LENGTH = GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATISTICS_LENGTH;
/**
 * Work-stats word that absorbs the crossing counts of the intermediate repair rounds. Nothing reads
 * it; a shared word keeps the round index out of the classify kernel (one pipeline for all of them).
 */
const INTERMEDIATE_CROSSING_SLOT = TOPOLOGY_STATS_LENGTH;
/** Vertex count from which the sweep BVH is Morton sorted, as `GPUSegmentIntersection`. */
const SWEEP_SPATIAL_SORT_MINIMUM_SLOTS = 1024;

/** Caller-owned outputs of {@link GPUCoverageSimplification}. */
export type GPUCoverageSimplificationOutput = {
  /**
   * Simplified ring vertices, ring after ring in input order, vertex order preserved. Its length is
   * the capacity; `positions.length` of the input always suffices.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * `ringCount + 1` offsets into `positions`; ring `r` keeps rows `[ringOffsets[r], ringOffsets[r + 1])`.
   * The ring count is unchanged, so the input `featureOffsets` and `polygonOffsets` still apply and
   * `positions` with these `ringOffsets` is a polygon port with those offsets. Offsets are clamped
   * to the capacity so rings stay consistent after an overflow.
   */
  ringOffsets: GraphDataView<'uint32'>;
  /** Optional per-input-vertex `1`/`0` output mask, `positions.length` rows of the input. */
  outputMask?: GraphDataView<'uint32'>;
  /**
   * Optional {@link GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATISTICS_LENGTH} rows describing the
   * topology pass (all zero when `topologyIterationCount` is 0):
   * - `[0]` crossings before any repair: pairs of simplified arc segments that intersect (cross,
   *   touch in a T-junction or overlap) without sharing a coverage point, plus simplified spans
   *   whose shortcut passes over a kept vertex of another part of the coverage (a swept ring).
   * - `[1]` crossings remaining after the last repair iteration.
   * - `[2]` vertices restored by the repair iterations (the ring-minimum step is not counted).
   * - `[3]` 1 when the candidate pair capacity overflowed in some iteration, so the counts are
   *   lower bounds.
   * Crossings fixed is `[0] - [1]`.
   */
  topologyStatistics?: GraphDataView<'uint32'>;
  /** Status scalars. */
  status: {
    /** Optional one-row scalar: 1 when every arc was decided within `maximumIterations`, else 0. */
    converged?: GraphDataView<'uint32'>;
    /** One-row flag: 1 when more vertices were kept than `positions` can hold, else 0. */
    overflow: GraphDataView<'uint32'>;
    /** Optional one-row unclamped kept-vertex total. */
    requiredCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row count of vertices whose snap cell could not be certified: finite vertices
     * `2^24 * snapGridSize` or farther from the first vertex on either axis (see `snapGridSize`).
     * Their point identity comes from an inexact key, so such a vertex may be merged with a vertex
     * of a neighbouring cell and the shared arcs are not reliable. Always 0 without a snap
     * tolerance. It is not an `overflow`: no capacity changes it.
     */
    uncertainCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUCoverageSimplification}.
 *
 * Per-frame (no recompile): the contents of `parameters` (the tolerance) and of the input
 * buffers, as long as lengths stay the same. Compile-time: `snapGridSize`, `maximumIterations`, all
 * view lengths and which optional outputs are present.
 */
export type GPUCoverageSimplificationProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'coverage-simplification'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Layout notes. Rows are features: the coverage identity of a ring is its feature, and two rings
   * of one feature never partner each other. `positions` are flattened ring vertices, planar and
   * finite (use tile- or view-local units because distances are f32). Rings close implicitly and
   * every ring (shells, holes) is simplified. `ringOffsets`, `polygonOffsets` and `featureOffsets`
   * each carry a terminal entry.
   */
  /**
   * Vertex snap tolerance. Two vertices that snap to the same point are one point of the coverage.
   * Defaults to `0` (exact f32 equality).
   *
   * A vertex snaps to cell `floor((coordinate - origin) / snapGridSize + 0.5)` per axis (round
   * half up, as `GPUContiguityWeights.snapTolerance`), with the first ring-owned vertex
   * (`ringOffsets[0]`) as `origin`, so the limit is on the extent of the data and not on its distance from zero: every
   * vertex must be within `2^24 * snapGridSize` of the first vertex on both axes (float32 cannot
   * tell cells apart beyond that). Vertices outside that range are counted in the optional
   * `output.status.uncertainCount`; when it is not zero, use a larger tolerance or split the data.
   */
  /** Grid spacing in position units; 0 disables quantization. Ties round toward positive infinity. */
  snapGridSize?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH}
   * elements written with `getGPULineSimplificationParameterValues`: the Douglas-Peucker tolerance.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Whether the outer boundary of the coverage is simplified. Defaults to `true`. With `false`
   * (Shapely `coverage_simplify(simplify_boundary=False)`), every vertex on an edge that no other
   * polygon shares is kept, so only arcs shared by two polygons are simplified; hole boundaries
   * of the coverage count as outer boundary. Compile-time.
   */
  simplifyBoundary?: boolean;
  /** Compile-time cap on Douglas-Peucker iterations, as `GPULineSimplification.maximumIterations`. */
  maximumIterations?: number;
  /**
   * Compile-time number of detect-and-repair iterations that make the simplified coverage
   * topology-preserving. Each round finds simplified segments that cross, touch or overlap
   * without sharing a coverage point, and shortcuts that pass over a kept vertex (an island swept
   * into the neighbouring polygon), and restores the original vertex farthest from each of them.
   * `0` skips detection and repair (the ring minimum still applies). Defaults to `4`.
   */
  topologyIterationCount?: number;
  /**
   * Compile-time capacity of the candidate pair list of each detection iteration. Shared arcs appear
   * in two rings and every node touches several segments, so the list holds those legitimate
   * pairs too. Defaults to `max(256, 4 * vertices)`. On overflow `output.topologyStatistics[3]` is set.
   */
  topologyPairCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUCoverageSimplificationOutput;
};

/**
 * Gap-free polygon-coverage simplification (the Shapely `simplify_coverage` and PostGIS
 * `ST_CoverageSimplify` idea): every arc shared by two polygons is simplified once, so both
 * neighbours keep identical vertices along it.
 *
 * **Definition.**
 * - Vertices are snapped as in `GPUContiguityWeights` and get dense point IDs; every ring edge gets
 *   the key `(min, max)` of its endpoint IDs. The *partner* of an edge is the lowest polygon ID
 *   (other than its own) that has an edge with the same key, or none for outer boundary.
 * - An *arc* is a maximal run of consecutive ring edges with the same partner. Arcs also break at
 *   the first vertex of every ring, so a shared arc that crosses a ring start keeps one extra
 *   vertex.
 * - An arc is *owned* by the lower polygon ID of its pair (an outer-boundary arc by its own
 *   polygon). Owned arcs are copied into one polyline each (a ring without any break is one
 *   closed polyline) and simplified by `GPULineSimplification`. A vertex is kept when its point is
 *   kept in any owned arc, or is a junction (an arc endpoint); the non-owning neighbour reads that
 *   decision through the shared point ID, so it cannot disagree.
 * - Output rings are the input rings filtered by the keep mask. Ring and vertex order and
 *   direction are unchanged.
 *
 * With `simplifyBoundary: false` the keep decision additionally marks every vertex that touches an
 * edge without a partner, so boundary arcs are copied unchanged.
 *
 * **Guarantees.** Neighbours stay gap-free and overlap-free along shared arcs: the simplified shared
 * boundary of two polygons is the same polyline in both rings. The kept set along each arc equals
 * Douglas-Peucker on that arc when `status.converged` is 1, before the topology steps below add vertices.
 *
 * **Topology preservation** (Shapely `coverage_simplify`, PostGIS `ST_CoverageSimplify`). Arc
 * endpoints (junctions where the partner changes, and ring starts) are always kept, so nodes never
 * move. A ring never keeps fewer than three vertices: if Douglas-Peucker leaves fewer, the
 * farthest original vertices are restored. Then `topologyIterationCount` iterations run: the simplified rings
 * are compacted, `GPUSegmentIntersection` lists intersecting segment pairs with exact predicates,
 * pairs that share a coverage point (a node) or are the same shared edge are ignored, and for every
 * remaining pair the original vertex farthest from each segment's span is restored (one
 * Douglas-Peucker split step; decisions are shared through point IDs, so neighbours still agree).
 * A shortcut can also change the topology without crossing anything: an island, and the hole it
 * fills, share no junction with the arc beside them, so a shortcut that passes over the island
 * moves it into the neighbouring polygon. Each round therefore also tests every kept vertex against
 * the area between each span's original chain and its shortcut (exact point-in-ring signs) and
 * repairs a span that holds one in the same way. GEOS avoids the same case by refusing to remove
 * a corner that contains another vertex. `output.topologyStatistics` reports crossings and swept
 * spans before, those remaining after the last iteration, and restored vertices. Crossings that need more splits than the rounds provide stay in the output
 * and are counted as remaining; an input that is itself self-intersecting cannot be repaired.
 * If a topology pair list overflows, every original coverage vertex is restored before output
 * compaction, even when topologyStatistics is omitted. The final overflow still describes vertex
 * capacity only.
 * Non-finite vertices are unsupported (never kept).
 *
 * **Cost.** Arcs are copied by one thread each walking the arc, so a single very long arc costs a
 * serial walk of its length. The sweep test builds one BVH over the input segments; in every
 * round each kept vertex walks it along the horizontal line through the vertex and, for every
 * input edge on that line whose span dropped a vertex and whose chord is within the span's reach
 * of the vertex, walks that span's original chain once (worst case kept vertices x edges on the
 * line x span length; a vertex far from a span's chord costs four multiplications). Intermediate buffers are `O(vertices)`; the simplification rows are
 * bounded by `2 * vertices`. The output is bounded by the `output.positions` capacity with a GPU
 * overflow flag; `positions.length` of the input always suffices.
 */
export class GPUCoverageSimplification implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCoverageSimplificationProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUCoverageSimplificationProps) {
    validateRequirement(
      props.id ?? 'coverage-simplification',
      !('snapTolerance' in props),
      'snapTolerance was removed; use snapGridSize'
    );
    const captured = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'positions',
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets',
      'snapGridSize',
      'parameters',
      'simplifyBoundary',
      'maximumIterations',
      'topologyIterationCount',
      'topologyPairCapacity',
      'output'
    ]);
    const capturedOutput = snapshotConstructionFields(captured.output, [
      'positions',
      'ringOffsets',
      'outputMask',
      'topologyStatistics',
      'status'
    ]);
    props = Object.freeze({
      ...captured,
      output: Object.freeze({
        ...capturedOutput,
        status: snapshotConstructionFields(capturedOutput.status, [
          'converged',
          'overflow',
          'requiredCount',
          'uncertainCount'
        ])
      })
    });
    const id = props.id ?? 'coverage-simplification';
    this.id = id;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    props = Object.freeze({
      ...props,
      snapGridSize: resolveCoverageSnapGridSize(id, props)
    });
    this.props = props;
    if (props.maximumIterations !== undefined) {
      validateCapacity(
        id,
        'maximumIterations',
        props.maximumIterations,
        1,
        GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS
      );
    }
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} positions must hold at least one vertex`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
    validatePackedUint32View(props.featureOffsets, `${id} featureOffsets`);
    for (const [name, view] of [
      ['ringOffsets', props.ringOffsets],
      ['polygonOffsets', props.polygonOffsets],
      ['featureOffsets', props.featureOffsets]
    ] as const) {
      if (view.length < 2) {
        throw new Error(`${id} ${name} must hold at least two entries`);
      }
    }
    if (props.topologyIterationCount !== undefined) {
      validateCapacity(id, 'topologyIterationCount', props.topologyIterationCount, 0, 16);
    }
    if (props.topologyPairCapacity !== undefined) {
      validateCapacity(id, 'topologyPairCapacity', props.topologyPairCapacity);
    }
    if (props.simplifyBoundary !== undefined && typeof props.simplifyBoundary !== 'boolean') {
      throw new Error(`${id} simplifyBoundary must be a boolean`);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} float32 values`
      );
    }
    const {output} = props;
    validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length < 1) {
      throw new Error(`${id} output.positions must hold at least one row`);
    }
    validatePackedUint32View(output.ringOffsets, `${id} output.ringOffsets`);
    if (output.ringOffsets.length !== props.ringOffsets.length) {
      throw new Error(`${id} output.ringOffsets length must equal ringOffsets length`);
    }
    const {overflow, requiredCount} = output.status;
    validatePackedUint32View(overflow, `${id} output.status.overflow`);
    if (overflow.length < 1) {
      throw new Error(`${id} output.status.overflow must hold one uint32`);
    }
    if (output.outputMask) {
      validatePackedUint32View(output.outputMask, `${id} output.outputMask`);
      if (output.outputMask.length !== props.positions.length) {
        throw new Error(`${id} output.outputMask length must equal positions length`);
      }
    }
    if (output.topologyStatistics) {
      validatePackedUint32View(output.topologyStatistics, `${id} output.topologyStatistics`);
      if (output.topologyStatistics.length < TOPOLOGY_STATS_LENGTH) {
        throw new Error(
          `${id} output.topologyStatistics must hold ${TOPOLOGY_STATS_LENGTH} uint32 rows`
        );
      }
    }
    if (requiredCount) {
      validatePackedUint32View(requiredCount, `${id} output.status.requiredCount`);
      if (requiredCount.length < 1) {
        throw new Error(`${id} output.status.requiredCount must hold one uint32`);
      }
    }
    if (output.status.converged) {
      validatePackedUint32View(output.status.converged, `${id} output.status.converged`);
      if (output.status.converged.length < 1) {
        throw new Error(`${id} output.status.converged must hold one uint32`);
      }
    }
    if (output.status.uncertainCount) {
      validatePackedUint32View(output.status.uncertainCount, `${id} output.status.uncertainCount`);
      if (output.status.uncertainCount.length < 1) {
        throw new Error(`${id} output.status.uncertainCount must hold one uint32`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.positions,
        output.ringOffsets,
        output.outputMask,
        overflow,
        requiredCount,
        output.topologyStatistics,
        output.status.converged,
        output.status.uncertainCount
      ],
      [
        props.positions,
        props.ringOffsets,
        props.polygonOffsets,
        props.featureOffsets,
        props.parameters
      ]
    );
  }

  /** Returns the coverage-simplification nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, ringOffsets, polygonOffsets, featureOffsets, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      props.parameters,
      output.status.converged,
      output.positions,
      output.ringOffsets,
      output.outputMask,
      output.status.overflow,
      output.status.requiredCount,
      output.status.uncertainCount,
      output.topologyStatistics
    ]);
    // Every topology round builds a segment table over the compacted vertices.
    validateMaximumCount(
      id,
      'positions rows',
      props.positions.length,
      getGPUSegmentIntersectionMaximumVertexCount(graph.device)
    );
    const vertexCount = positions.length;
    const ringCount = ringOffsets.length - 1;
    const featureCount = featureOffsets.length - 1;
    const outputCapacity = output.positions.length;
    const snapTolerance = props.snapGridSize ?? 0;
    const lineRowCapacity = 2 * vertexCount;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    // First ring of every feature (terminal entry included), so a kernel finds the feature of a
    // ring with one search instead of reading both offset levels.
    const featureRingStarts = view('feature-ring-starts', featureCount + 1);
    const vertexHigh = view('vertex-high', vertexCount);
    const vertexLow = view('vertex-low', vertexCount);
    const vertexPolygon = view('vertex-polygon', vertexCount);
    const vertexNext = view('vertex-next', vertexCount);
    const vertexPrev = view('vertex-prev', vertexCount);

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-feature-ring-starts`,
        operation: OPERATION,
        variant: 'feature-ring-starts',
        bindings: [
          {name: 'featureOffsets', view: featureOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureRingStarts', view: featureRingStarts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: featureCount + 1,
        body: 'featureRingStarts[featureRingStartsOffset + index] = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertex-topology`,
        operation: OPERATION,
        variant: 'vertex-topology',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'featureRingStarts', view: featureRingStarts, type: 'u32', access: 'read'},
          {name: 'vertexHigh', view: vertexHigh, type: 'u32', access: 'read_write'},
          {name: 'vertexLow', view: vertexLow, type: 'u32', access: 'read_write'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read_write'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read_write'},
          {name: 'vertexPrev', view: vertexPrev, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const RING_COUNT: u32 = ${ringCount}u;
const FEATURE_COUNT: u32 = ${featureCount}u;

${IS_FINITE_FLOAT_WGSL}
${getCoverageSnapKeyWGSL(snapTolerance, ringCount)}`,
        body: `var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  // The coverage identity of a ring is its feature: first ring of feature f is featureRingStarts[f].
  var polygon = 0u;
  high = FEATURE_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (featureRingStarts[featureRingStartsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let inRing = index >= ringBegin && index < ringEnd &&
    ring >= featureRingStarts[featureRingStartsOffset] && ring < featureRingStarts[featureRingStartsOffset + FEATURE_COUNT] &&
    ring < featureRingStarts[featureRingStartsOffset + polygon + 1u];
  let valid = inRing && isFiniteFloat(x) && isFiniteFloat(y);
  let snapOrigin = getSnapOrigin();
  vertexHigh[vertexHighOffset + index] = select(${INVALID}, quantise(x, snapOrigin.x), valid);
  vertexLow[vertexLowOffset + index] = select(${INVALID}, quantise(y, snapOrigin.y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${INVALID}, polygon, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);
  let previous = select(index - 1u, ringEnd - 1u, index == ringBegin);
  vertexPrev[vertexPrevOffset + index] = select(previous, previous | ${RING_BEGIN_BIT}, index == ringBegin);`
      })
    ];

    // Dense point IDs, as the rook branch of GPUContiguityWeights.
    const vertexSort = getKeyPairSortNodes(
      graph,
      `${id}-vertex`,
      OPERATION,
      vertexCount,
      vertexHigh,
      vertexLow
    );
    const vertexGroups = getKeyGroupNodes(
      graph,
      `${id}-vertex`,
      OPERATION,
      vertexCount,
      vertexHigh,
      vertexLow,
      vertexSort.sortedItems
    );
    const pointIds = view('point-ids', vertexCount);
    const edgeHigh = view('edge-high', vertexCount);
    const edgeLow = view('edge-low', vertexCount);
    nodes.push(
      ...vertexSort.nodes,
      ...vertexGroups.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-ids`,
        operation: OPERATION,
        variant: 'point-ids',
        bindings: [
          {name: 'sortedItems', view: vertexSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: vertexGroups.groupIndex, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${INVALID},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${INVALID});`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-keys`,
        operation: OPERATION,
        variant: 'edge-keys',
        bindings: [
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'edgeHigh', view: edgeHigh, type: 'u32', access: 'read_write'},
          {name: 'edgeLow', view: edgeLow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${INVALID} && second != ${INVALID} && first != second &&
    vertexPolygon[vertexPolygonOffset + index] != ${INVALID};
  edgeHigh[edgeHighOffset + index] = select(${INVALID}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${INVALID}, max(first, second), valid);`
      })
    );
    const edgeSort = getKeyPairSortNodes(
      graph,
      `${id}-edge`,
      OPERATION,
      vertexCount,
      edgeHigh,
      edgeLow
    );
    const edgeGroups = getKeyGroupNodes(
      graph,
      `${id}-edge`,
      OPERATION,
      vertexCount,
      edgeHigh,
      edgeLow,
      edgeSort.sortedItems
    );
    nodes.push(...edgeSort.nodes, ...edgeGroups.nodes);

    // Partner polygon of every edge: the lowest other polygon holding the same edge.
    const edgePartner = view('edge-partner', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-partner`,
        operation: OPERATION,
        variant: 'edge-partner',
        bindings: [
          {name: 'sortedItems', view: edgeSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'edgeHigh', view: edgeHigh, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: edgeGroups.groupIndex, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: edgeGroups.groupStarts, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let item = sortedItems[sortedItemsOffset + index];
  let polygon = vertexPolygon[vertexPolygonOffset + item];
  var partner = ${INVALID};
  if (polygon != ${INVALID} && edgeHigh[edgeHighOffset + item] != ${INVALID}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = vertexPolygon[vertexPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${INVALID} && otherPolygon != polygon) {
        partner = min(partner, otherPolygon);
      }
    }
  }
  edgePartner[edgePartnerOffset + item] = partner;`
      })
    );

    // Arc starts, junction flags and ownership. Bit 0: arc start. Bit 1: junction (partner change).
    const arcFlags = view('arc-flags', vertexCount);
    const startFlag = view('start-flag', vertexCount);
    const startRank = view('start-rank', vertexCount);
    // Arc starts in vertex order, plus one sentinel entry: the end of the last ring.
    const startList = view('start-list', vertexCount + 1);
    const ownedLength = view('owned-length', vertexCount);
    const ownedFlag = view('owned-flag', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-arc-flags`,
        operation: OPERATION,
        variant: 'arc-flags',
        bindings: [
          {name: 'vertexPrev', view: vertexPrev, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read_write'},
          {name: 'startFlag', view: startFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let prevEntry = vertexPrev[vertexPrevOffset + index];
  let previous = prevEntry & 0x7fffffffu;
  let ringBegin = (prevEntry & ${RING_BEGIN_BIT}) != 0u;
  let valid = vertexPolygon[vertexPolygonOffset + index] != ${INVALID};
  let junction = valid && edgePartner[edgePartnerOffset + previous] != edgePartner[edgePartnerOffset + index];
  arcFlags[arcFlagsOffset + index] = select(0u, 1u, ringBegin || junction) | select(0u, 2u, junction);
  startFlag[startFlagOffset + index] = select(0u, 1u, ringBegin || junction);`
      }),
      ...new GPUScan({
        id: `${id}-start-rank-scan`,
        input: startFlag,
        output: startRank,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-start-list`,
        operation: OPERATION,
        variant: 'start-list',
        bindings: [
          {name: 'startFlag', view: startFlag, type: 'u32', access: 'read'},
          {name: 'startRank', view: startRank, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'startList', view: startList, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
const RING_COUNT: u32 = ${ringCount}u;`,
        body: `let flag = startFlag[startFlagOffset + index];
  let rank = startRank[startRankOffset + index];
  if (flag != 0u) {
    startList[startListOffset + rank] = index;
  }
  if (index == VERTEX_COUNT - 1u) {
    startList[startListOffset + rank + flag] = min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-arc-lengths`,
        operation: OPERATION,
        variant: 'arc-lengths',
        bindings: [
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'startRank', view: startRank, type: 'u32', access: 'read'},
          {name: 'startList', view: startList, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
          {name: 'ownedLength', view: ownedLength, type: 'u32', access: 'read_write'},
          {name: 'ownedFlag', view: ownedFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `var length = 0u;
  var owned = 0u;
  let polygon = vertexPolygon[vertexPolygonOffset + index];
  if ((arcFlags[arcFlagsOffset + index] & 1u) != 0u && polygon != ${INVALID}) {
    let partner = edgePartner[edgePartnerOffset + index];
    if (partner == ${INVALID} || polygon < partner) {
      owned = 1u;
      // The arc runs to the next arc start in vertex order, or to the end of the ring, whose
      // first vertex is itself an arc start. The closing vertex is repeated.
      let arcEnd = startList[startListOffset + startRank[startRankOffset + index] + 1u];
      length = arcEnd - index + 1u;
    }
  }
    ownedLength[ownedLengthOffset + index] = length;
  ownedFlag[ownedFlagOffset + index] = owned;`
      })
    );
    const arcIndex = view('arc-index', vertexCount);
    const lineStart = view('line-start', vertexCount);
    const trackOffsets = view('track-offsets', vertexCount + 1);
    const linePositions = createTransientView(
      graph,
      `${id}-line-positions`,
      'float32x2',
      lineRowCapacity
    );
    const linePoint = view('line-point', lineRowCapacity);
    nodes.push(
      ...new GPUScan({
        id: `${id}-arc-index-scan`,
        input: ownedFlag,
        output: arcIndex,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-line-start-scan`,
        input: ownedLength,
        output: lineStart,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-track-offsets-fill`,
        operation: OPERATION,
        variant: 'track-offsets-fill',
        bindings: [
          {name: 'lineStart', view: lineStart, type: 'u32', access: 'read'},
          {name: 'ownedLength', view: ownedLength, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount + 1,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `trackOffsets[trackOffsetsOffset + index] =
    lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-track-offsets`,
        operation: OPERATION,
        variant: 'track-offsets',
        bindings: [
          {name: 'ownedFlag', view: ownedFlag, type: 'u32', access: 'read'},
          {name: 'arcIndex', view: arcIndex, type: 'u32', access: 'read'},
          {name: 'lineStart', view: lineStart, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `if (ownedFlag[ownedFlagOffset + index] != 0u) {
    trackOffsets[trackOffsetsOffset + arcIndex[arcIndexOffset + index]] = lineStart[lineStartOffset + index];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-copy-arcs`,
        operation: OPERATION,
        variant: 'copy-arcs',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read'},
          {name: 'ownedLength', view: ownedLength, type: 'u32', access: 'read'},
          {name: 'lineStart', view: lineStart, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'linePositions', view: linePositions, type: 'f32', access: 'read_write'},
          {name: 'linePoint', view: linePoint, type: 'u32', access: 'read_write'}
        ],
        invocationCount: lineRowCapacity,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        // One invocation per output row, so a long arc is copied in parallel. The arc owning a
        // row is the last vertex whose line start is at or before it; row k of an arc is its
        // start for k = 0 and the successor of vertex start + k - 1 afterwards.
        body: `let total = lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];
  if (index < total) {
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (lineStart[lineStartOffset + middle] <= index) {
        low = middle;
      } else {
        high = middle;
      }
    }
    let step = index - lineStart[lineStartOffset + low];
    let vertex = select(vertexNext[vertexNextOffset + low + step - 1u], low, step == 0u);
    let pointId = pointIds[pointIdsOffset + vertex];
    let x = positions[positionsOffset + vertex * 2u];
    let y = positions[positionsOffset + vertex * 2u + 1u];
    let finite = pointId != ${INVALID};
    linePositions[linePositionsOffset + index * 2u] = select(0.0, x, finite);
    linePositions[linePositionsOffset + index * 2u + 1u] = select(0.0, y, finite);
    linePoint[linePointOffset + index] = pointId;
  }`
      })
    );

    // Simplify every owned arc once.
    const lineImportance = createTransientView(
      graph,
      `${id}-line-importance`,
      'float32',
      lineRowCapacity
    );
    const lineKeep = view('line-keep', lineRowCapacity);
    const selectionIds = view('selection-ids', lineRowCapacity);
    const selectionCount = view('selection-count', 1);
    const selectionOverflow = view('selection-overflow', 1);
    const lineProps: GPULineSimplificationProps = {
      id: `${id}-arcs`,
      spatialContext: this.spatialContext,
      positions: linePositions,
      lineOffsets: trackOffsets,
      importance: lineImportance,
      maximumIterations: props.maximumIterations,
      parameters: props.parameters,
      output: {
        ids: selectionIds,
        outputMask: lineKeep,
        status: {
          count: selectionCount,
          overflow: selectionOverflow,
          ...(output.status.converged ? {converged: output.status.converged} : {})
        }
      }
    };
    nodes.push(...new GPULineSimplification(lineProps).getCommandNodes(graph));

    // Share the decisions through point IDs: junctions and kept arc vertices.
    const keepPoint = view('keep-point', vertexCount);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-keep-point-clear`,
        operation: OPERATION,
        view: keepPoint,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mark-junctions`,
        operation: OPERATION,
        variant: 'mark-junctions',
        bindings: [
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'vertexPrev', view: vertexPrev, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: keepPoint, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const KEEP_BOUNDARY: bool = ${props.simplifyBoundary === false};`,
        // A vertex is on the outer boundary when its outgoing or incoming edge has no partner.
        body: `let pointId = pointIds[pointIdsOffset + index];
  let previous = vertexPrev[vertexPrevOffset + index] & 0x7fffffffu;
  let isBoundary = KEEP_BOUNDARY && (edgePartner[edgePartnerOffset + index] == ${INVALID} ||
    edgePartner[edgePartnerOffset + previous] == ${INVALID});
  if (((arcFlags[arcFlagsOffset + index] & 2u) != 0u || isBoundary) && pointId != ${INVALID}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mark-arcs`,
        operation: OPERATION,
        variant: 'mark-arcs',
        bindings: [
          {name: 'lineKeep', view: lineKeep, type: 'u32', access: 'read'},
          {name: 'linePoint', view: linePoint, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: keepPoint, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: lineRowCapacity,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let pointId = linePoint[linePointOffset + index];
  if (index < trackOffsets[trackOffsetsOffset + VERTEX_COUNT] && lineKeep[lineKeepOffset + index] != 0u &&
      pointId != ${INVALID}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`
      })
    );

    // Topology repair: keep at least three vertices per ring, then detect and repair crossings.
    const topologyRounds = props.topologyIterationCount ?? DEFAULT_TOPOLOGY_ITERATION_COUNT;
    {
      nodes.push(
        ...getRingCollapseNodes(graph, {
          id,
          positions,
          ringOffsets,
          vertexPolygon,
          pointIds,
          keepPoint,
          vertexCount,
          ringCount
        })
      );
    }
    if (topologyRounds === 0 && output.topologyStatistics) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-topology-stats-zero`,
          operation: OPERATION,
          view: output.topologyStatistics,
          type: 'u32',
          value: '0u'
        })
      );
    }
    if (topologyRounds > 0) {
      const workStats = view('topology-work', TOPOLOGY_STATS_LENGTH + 1);
      const ownedVertex = view('owned-vertex', vertexCount);
      const pairCapacity = props.topologyPairCapacity ?? Math.max(256, 4 * vertexCount);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-topology-stats-clear`,
          operation: OPERATION,
          view: workStats,
          type: 'u32',
          value: '0u'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-owned-vertex`,
          operation: OPERATION,
          variant: 'owned-vertex',
          bindings: [
            {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
            {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
            {name: 'ownedVertex', view: ownedVertex, type: 'u32', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          body: `let polygon = vertexPolygon[vertexPolygonOffset + index];
  let partner = edgePartner[edgePartnerOffset + index];
  ownedVertex[ownedVertexOffset + index] =
    select(0u, select(0u, 1u, partner == ${INVALID} || polygon < partner), polygon != ${INVALID});`
        })
      );
      // The sweep test of every round walks one BVH over the input segments (a segment ID is its
      // start vertex), built once: the input does not change between rounds.
      const sweepTable = view('sweep-segment-table', vertexCount * SEGMENT_TABLE_STRIDE);
      const sweepMinima = createTransientView(
        graph,
        `${id}-sweep-minima`,
        'float32x2',
        vertexCount
      );
      const sweepMaxima = createTransientView(
        graph,
        `${id}-sweep-maxima`,
        'float32x2',
        vertexCount
      );
      nodes.push(
        createSegmentTableNode<Parameters>(graph, {
          id: `${id}-sweep-segment-table`,
          operation: OPERATION,
          geometry: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},
          table: sweepTable,
          minima: sweepMinima,
          maxima: sweepMaxima
        })
      );
      const {bvh: sweepBVH, nodes: sweepBVHNodes} = getSegmentBVHNodes(graph, {
        id: `${id}-sweep`,
        operation: OPERATION,
        minima: sweepMinima,
        maxima: sweepMaxima,
        leafCapacity: getNextPowerOfTwo(vertexCount),
        spatialSort: vertexCount >= SWEEP_SPATIAL_SORT_MINIMUM_SLOTS
      });
      nodes.push(...sweepBVHNodes);
      // Final detection also certifies pair capacity after the last repair. Geometry must not
      // depend on whether the caller requests the optional statistics copy.
      for (let round = 0; round <= topologyRounds; round++) {
        const isInitial = round === 0;
        const isFinal = round === topologyRounds;
        nodes.push(
          ...getTopologyRoundNodes(graph, {
            id: `${id}-round-${round}`,
            positions,
            ringOffsets,
            polygonOffsets,
            featureOffsets,
            pointIds,
            vertexNext,
            ownedVertex,
            keepPoint,
            sweepBVH: {
              nodeMinima: sweepBVH.nodeMinima,
              nodeMaxima: sweepBVH.nodeMaxima,
              leafIds: sweepBVH.leafIds,
              internalNodeCount: sweepBVH.internalNodeCount
            },
            vertexCount,
            ringCount,
            pairCapacity,
            repair: !isFinal,
            statsView: workStats,
            crossingSlot: isInitial ? 0 : isFinal ? 1 : INTERMEDIATE_CROSSING_SLOT
          })
        );
      }
      // A truncated pair list cannot certify any shortcut. Restore every coverage point before
      // compaction, including when the caller does not request the optional statistics.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-topology-fallback`,
          operation: OPERATION,
          variant: 'topology-fallback',
          bindings: [
            {name: 'workStats', view: workStats, type: 'u32', access: 'read'},
            {name: 'keepPoint', view: keepPoint, type: 'u32', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          body: `if (workStats[workStatsOffset + 3u] != 0u) {
    keepPoint[keepPointOffset + index] = 1u;
  }`
        })
      );
      if (output.topologyStatistics) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-topology-stats-copy`,
            operation: OPERATION,
            variant: 'topology-stats-copy',
            bindings: [
              {name: 'workStats', view: workStats, type: 'u32', access: 'read'},
              {
                name: 'topologyStatistics',
                view: output.topologyStatistics,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: TOPOLOGY_STATS_LENGTH,
            body: 'topologyStatistics[topologyStatisticsOffset + index] = workStats[workStatsOffset + index];'
          })
        );
      }
    }

    // Per-vertex mask, rank, and ring-wise compaction.
    const keepMask = output.outputMask ?? view('keep-mask', vertexCount);
    const keepRank = view('keep-rank', vertexCount);
    nodes.push(
      createKeepMaskNode<Parameters>(graph, `${id}-keep-mask`, pointIds, keepPoint, keepMask),
      ...new GPUScan({
        id: `${id}-keep-rank-scan`,
        input: keepMask,
        output: keepRank,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createEmitPositionsNode<Parameters>(graph, {
        id: `${id}-emit-positions`,
        positions,
        keepMask,
        keepRank,
        outputPositions: output.positions,
        vertexCount
      }),
      createEmitOffsetsNode<Parameters>(graph, {
        id: `${id}-emit-offsets`,
        ringOffsets,
        keepMask,
        keepRank,
        outputRingOffsets: output.ringOffsets,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        vertexCount,
        ringCount,
        capacity: outputCapacity
      })
    );
    // A snap key that left its exact range may have merged distinct vertices. That is a vertex
    // identity that could not be certified, not a dropped row: it has its own count.
    if (output.status.uncertainCount) {
      nodes.push(
        ...createCoverageSnapRangeNodes<Parameters>(graph, {
          id: `${id}-snap-range`,
          operation: OPERATION,
          positions,
          ringOffsets,
          snapTolerance,
          uncertainCount: output.status.uncertainCount
        })
      );
    }
    return nodes;
  }
}
