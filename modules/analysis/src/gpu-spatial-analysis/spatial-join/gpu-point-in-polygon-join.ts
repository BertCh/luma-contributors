// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createGPUMultipartRowOwnersNode,
  validateGPUMultipartGeometryPort
} from '../../analysis-core/contracts/ports';

import type {
  GPUCandidateStatusPort,
  GPULonLatPolygonStatusPort
} from '../../analysis-core/contracts/index';
import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/kernels/lon-lat-ring-wgsl';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {
  createTransientView,
  getViewBindingRange,
  type GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {GPU_POINT_IN_POLYGON_CLASSIFICATION} from '@luma.gl/graph/geospatial';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {
  getPolygonEdgeSlabBindings,
  getPolygonEdgeSlabLoopWGSL,
  getPolygonEdgeSlabNodes,
  getPolygonEdgeSlabQueryWGSL,
  POLYGON_EDGE_SLAB_PARAMS_STRIDE,
  type PolygonEdgeSlabs
} from './polygon-edge-slabs';
import {
  createWGSLActiveCountDispatch,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUCompactOutput,
  GPUUint32Rows
} from '../../analysis-core/kernels/gpu-contributor-types';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUMultipartPointGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  isGraphVectorView,
  getStorageBindingElementLimit,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateCompactOutput,
  validateRequirement,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createSpatialJoinBoundsNode,
  createSpatialJoinChunkNode,
  getPolygonEdgeLayoutLength,
  createSpatialJoinClearNode,
  createSpatialJoinFinalizeNodes,
  getSortedFeatureBVHNodes,
  type SpatialSortCurve,
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getDefaultEdgeSlabs,
  getDefaultQuerySort,
  getPolygonEdgeLayoutNodes,
  getSpatialJoinProbeNodes,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';
import {
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry,
  validateStatusScalars
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';

import {
  validateSpatialJoinBindings,
  validateSpatialJoinReadWriteBindings
} from './spatial-join-binding-preflight';

import {Buffer} from '@luma.gl/core';
import {
  SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT,
  planSpatialJoinCandidatePages
} from './spatial-join-candidate-pages';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {getMultipartPointCountNodes} from './multipart-point-reduction';

const OPERATION = 'GPUPointInPolygonJoin';

/** Row bases, row counts and generated WGSL u32 count literals must remain representable. */
const GPU_POINT_IN_POLYGON_JOIN_MAXIMUM_ROWS = 0xffffffff;
/** Preserve the signed-compatible global cutoff below the saturated uint32 requirement sentinel. */
const GPU_POINT_IN_POLYGON_JOIN_MAXIMUM_CANDIDATES = 0x7ffffffe;
/** Preserve the shared slab builder's four indexed edge copies per vertex default budget. */
const GPU_POINT_IN_POLYGON_JOIN_SLAB_ENTRIES_PER_VERTEX = 4;
/** Preserve a 1024-entry index budget for small polygons; account for it before allocation. */
const GPU_POINT_IN_POLYGON_JOIN_MINIMUM_SLAB_ENTRIES = 1024;

/** Allowance for the BVH build, feature and global query radix sorts, edge layout and candidates. */
const GPU_POINT_IN_POLYGON_JOIN_FIXED_NODE_ALLOWANCE = 1024;
/**
 * Conservative bound for ordered point-chunk passes, retaining the former radix-sort headroom.
 * Chunks now emit only runtime key generation, probes, assignment, flag copy and collect passes;
 * the two global query sorts are covered by the fixed allowance.
 */
const GPU_POINT_IN_POLYGON_JOIN_CHUNK_NODE_ALLOWANCE = 80;
/** Unsorted chunks only emit two probes, assignments, collect passes and an optional flag copy. */
const GPU_POINT_IN_POLYGON_JOIN_UNSORTED_CHUNK_NODE_ALLOWANCE = 8;

/**
 * The spatial context to give a `GPUPointInPolygonJoin` for an operation resolved with `context`:
 * a longitude/latitude context becomes containment-only (`metric: 'none'`, polygon edges straight
 * in degrees). The declared edge model and datum are preserved so unsupported edges still fail
 * validation; any other context is passed through unchanged.
 */
export function getPointInPolygonContainmentContext(context: GPUSpatialContext): GPUSpatialContext {
  return context.coordinateSpace === 'longitude-latitude'
    ? {
        coordinateSpace: 'longitude-latitude',
        metric: 'none',
        ...(context.edgeModel === undefined ? {} : {edgeModel: context.edgeModel}),
        ...(context.datum === undefined ? {} : {datum: context.datum})
      }
    : context;
}

/** Planar coordinates, or longitude/latitude with no metric (topological containment). */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  metrics: Object.freeze(['none', 'native'] as const)
});

/**
 * How {@link GPUPointInPolygonJoin} treats a point exactly on a polygon boundary: `'half-open'`
 * (each boundary point goes to exactly one feature of a valid coverage), `'closed'` (boundary counts
 * as contained, the smallest row wins) or `'open'` (boundary belongs to no feature).
 */
export type GPUPointInPolygonBoundaryRule = 'half-open' | 'closed' | 'open';

/** Bit of a classification word set when a boundary point is owned under the half-open rule. */
const HALF_OPEN_OWNER_BIT = 0x100;

/** Caller-owned outputs of {@link GPUPointInPolygonJoin}. */
export type GPUPointInPolygonJoinOutput<
  TruncationRows extends GPUUint32Rows = GraphDataView<'uint32'>
> = {
  /**
   * Per-point containing feature ID (`polygons.sourceIds`) or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`.
   * Chunked like `points.positions` for single points. Multipart points use one packed row per
   * feature: the lowest polygon row matching any part wins, before mapping polygon source IDs.
   */
  pointFeatureIds?: GPUUint32Rows;
  /** Polygon rows before source-ID remapping, or 0xffffffff; same topology as pointFeatureIds. */
  pointFeatureRows?: GPUUint32Rows;
  /** Per-polygon count of points assigned to the lowest matching polygon row. */
  polygonAssignmentCounts?: GraphDataView<'uint32'>;
  /** Per-polygon count of distinct input features matching that polygon. */
  polygonMembershipCounts?: GraphDataView<'uint32'>;
  /** Optional number of distinct matching polygons per input feature, including zero for empty features. */
  pointMatchCounts?: GraphDataView<'uint32'>;
  /** Optional packed feature-major 0/1 membership matrix: input feature row * polygon count + polygon row. Includes every matching polygon under boundaryRule. */
  featurePolygonMembership?: GraphDataView<'uint32'>;
  /**
   * Optional compact stable IDs of joined points in global source-row order across chunks.
   * Its overflow flag reports final match-capacity loss only; optional candidateOverflow reports
   * BVH or candidate loss even when the retained matches fit ids.length. requiredCount
   * counts the retained assignments before final compaction; it cannot recover dropped candidates.
   */
  matches?: GPUCompactOutput;
  /**
   * Optional per-point-row flag (one `uint32` per point, in row order across chunks), 1 when that
   * point lost candidates because the candidate total exceeded `candidateCapacity`. Candidates are
   * laid out in point-row order and the slots past the capacity are dropped, so the flagged points
   * are the point whose candidates cross the capacity (it keeps its first ones) and every later
   * point that had a candidate (it keeps none); earlier points are complete. A flagged point may
   * show `GPU_SPATIAL_JOIN_NO_FEATURE` although a feature contains it. A BVH overflow (more
   * features than `leafCapacity`) drops features that any point may fall in, so it flags every
   * point. See `status.candidateOverflow`. A chunked view follows `points.positions`; a single
   * packed view covering every point is also accepted for compatibility. Multipart inputs instead
   * use one packed row per feature, OR-ing its parts' flags; empty features have no lost parts.
   */
  candidateTruncated?: TruncationRows;
  /**
   * Completeness scalars. Optional `antimeridianUncertainCount` counts unprepared crossing rings;
   * optional `poleUncertainCount` counts unprepared pole-enclosing rings. Prepare these polygons
   * with `GPULonLatPolygonPreparation`. Both fields are left untouched in planar contexts.
   */
  status: GPULonLatPolygonStatusPort &
    GPUCandidateStatusPort & {
      /**
       * One-row flag: 1 when BVH leaf, candidate, or `matches` capacity overflowed. Its meaning is
       * unchanged; `candidateOverflow` isolates the intermediate capacities (BVH leaf and
       * candidate), leaving out `matches`.
       */
      overflow: GraphDataView<'uint32'>;
      /** Optional one-row candidate count for sizing `candidateCapacity`; saturates at `0xffffffff`. */
      candidateCount?: GraphDataView<'uint32'>;
      /**
       * Optional one-row count of refined (point, feature) candidates classified `uncertain`.
       * Uncertain pairs are never assigned: a point whose only candidates were uncertain gets no
       * feature, so a nonzero value means the assignments and polygon counts may be incomplete.
       */
      uncertainCount?: GraphDataView<'uint32'>;
    };
};

/**
 * Properties for {@link GPUPointInPolygonJoin}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths and chunking,
 * `candidateCapacity`, `leafCapacity`, `boundaryRule`, `spatialSort`, and which optional views exist.
 */
export type GPUPointInPolygonJoinProps<
  TruncationRows extends GPUUint32Rows = GraphDataView<'uint32'>,
  Parameters = unknown,
  PointGeometry extends GPUPointGeometryPort | GPUMultipartPointGeometryPort =
    | GPUPointGeometryPort
    | GPUMultipartPointGeometryPort
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-in-polygon-join'`. */
  id?: string;
  /**
   * Planar points, possibly chunked. Optional `sourceIds` (same chunk topology) are the IDs
   * written to `output.matches`. Multipart outputs and source IDs use one packed row per feature;
   * any matching part qualifies, empty features have no assignment. Per-polygon counts may sum
   * to more than the input feature count. Each part uses the same boundary rule as a single point.
   */
  points: PointGeometry;
  /**
   * Polygon or multipolygon features in the four-level layout (ring 0 of a polygon is its shell,
   * later rings are holes; rings close implicitly). Optional `sourceIds` are written to
   * `output.pointFeatureIds` instead of feature rows.
   */
  polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  /**
   * Optional prepared polygon index over the same polygon views. Bounds, offsets and exact
   * refinement share the generation-gated geometry snapshot; source IDs remain live caller data.
   * Add the handle before the join. Its leafCapacity and spatialSort replace this join's;
   * assignments are recomputed every encoding.
   */
  prepared?: GPUSpatialJoinPrepared<Parameters>;
  /** Maximum (point part, polygon feature) bounding-box candidates per encoding; multipart feature results inherit any part's loss. */
  candidateCapacity: number;
  /** Optional scratch page capacity; defaults to the device binding/buffer envelope. Global cutoff and results are unchanged. */
  candidateTileCapacity?: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Compile-time. How a point exactly on a polygon boundary is treated. The decision uses the exact
   * orientation predicate, so "exactly on" is exact for the f32 vertices.
   *
   * - `'half-open'` (default): the boundary is split by the crossing-number convention of a
   *   half-open region, so a point on a boundary shared by several features of a valid coverage
   *   belongs to exactly one of them, whatever the feature rows are. The owner is the feature that
   *   contains the point after an infinitesimal move toward +x and +y (+y dominates), which means
   *   an edge or vertex belongs to the feature whose interior is to its right/above: left and
   *   bottom boundaries are included, right and top boundaries are excluded. A horizontal edge
   *   belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the
   *   feature whose interior touches the upper right of the vertex. Points on the outer right or
   *   top boundary of a coverage belong to no feature (they are not lost by a bug: the rule that
   *   makes interior boundaries unique makes those open).
   * - `'closed'`: every boundary point counts as contained by every feature whose boundary it
   *   touches, then the smallest feature row wins (the assignment keeps one feature per point).
   *   Nothing is lost on the outer boundary, but the owner of a shared-boundary point depends on
   *   the feature row order.
   * - `'open'`: boundary points belong to no feature (only strict interior counts).
   *
   * A point strictly inside is unaffected by the rule. Overlapping features (an invalid coverage)
   * still resolve to the smallest containing row. `matches` lists exactly the points that get a
   * feature under the same rule.
   */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
  /**
   * Coordinate space and metric of the points and polygons. Omitted means native planar
   * coordinates. Containment is a topological test evaluated in the plane of the given
   * coordinates: polygon edges are straight lines in the coordinates (as in GeoJSON), not
   * geodesics. `{coordinateSpace: 'longitude-latitude', metric: 'none'}` is therefore accepted and
   * behaves exactly like planar. A longitude/latitude context with any metric (for example
   * `'great-circle'`) throws at construction because there are no geodesic edges; project to a
   * planar frame first. Prepare polygons that cross ±180 or enclose a pole with
   * `GPULonLatPolygonPreparation`; optional status counters report unprepared rings.
   * Longitude/latitude requires `edgeModel: 'linear'` (the default); great-circle edges require preparation.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `polygons` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  polygonsSpatialContext?: GPUSpatialContext;
  /**
   * Compile-time. When true, features are reordered along a Hilbert curve (see `spatialSortCurve`) of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (assignment still keeps the smallest containing feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 14,400 small square
   * features and 250,000 points: shuffled rows drop from 218 ms to 6.7 ms per encoding (about 9,500
   * to 61 BVH nodes visited per point); row-major rows are 6.0 ms unsorted and 5.6 ms sorted.
   *
   * Defaults to on from 256 features and off below, where the BVH is too shallow for the order to
   * matter. On coherent data the sort costs about 1 ms; pass `false` to skip it. Candidate order is
   * then unspecified (the point-in-polygon result is not affected).
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /**
   * Compile-time. When true, each chunk of points is processed in Hilbert order of the points
   * (quantized over the root bounds of the polygon BVH, non-finite points last) instead of input
   * order, so neighboring invocations walk neighboring BVH subtrees. Per-point results, candidate
   * slots and counts stay addressed by the original row, so `pointFeatureIds`, `matches`,
   * polygon counts, the candidate pairs and what a `candidateCapacity` overflow drops are
   * identical either way; only traversal cost changes. Costs one key pass per chunk and two global
   * radix sorts per encoding. A global Hilbert sort followed by stable chunk regrouping preserves each chunk's
   * order without per-chunk sorting pipelines or padding. Singleton chunks retain identity order.
   * Each chunk's key pass dispatches only its own rows.
   *
   * Defaults to on from 4096 points and 256 features (provisional; see
   * `spatial-join-query-order.spec.ts` for the A/B harness), and off below, where the sort cannot
   * pay for itself.
   */
  querySort?: boolean;
  /**
   * Compile-time. When true, the exact refine of a candidate `(point, feature)` pair tests only
   * the edges of the feature's y-slab at `point.y` (a prepared edge-slab index over the polygon
   * parts) instead of every edge of the feature. The classification is a parity of edges that
   * straddle `point.y` plus an on-boundary test on edges whose closed y-range contains it, so every
   * edge that can change the result is in the slab: `pointFeatureIds`, `matches`, polygon counts,
   * `candidateCount` and `uncertainCount` are identical for all three boundary rules, including the
   * half-open owner and malformed offsets (UNCERTAIN). Parity is kept per polygon part and
   * combined with OR, exactly as the unslabbed walk does, because each part is its own slab owner.
   *
   * One difference, at the numeric edge: the exact orientation test answers UNCERTAIN when its
   * terms span more than 200 binary orders of magnitude. The unslabbed walk reports that for any
   * edge of the feature; the slab refine only sees edges of the slab and so classifies a pair that
   * the walk would have reported UNCERTAIN because of an irrelevant far edge. Features with a
   * non-finite or `|coordinate| >= 1e30` vertex are never slabbed, so they keep the walk's
   * behavior exactly.
   *
   * Features under 8 vertices per part are walked. Costs a few passes over the polygon vertices
   * (done once in a {@link GPUSpatialJoinPrepared} created with `edgeSlabs: true`, which this join
   * then reuses) and changes the candidate buffer layout (four words per candidate). Defaults to
   * on when `polygons.positions.length * candidateCapacity` is at least 4,000,000 and features
   * average 8 vertices (provisional; see `spatial-join-edge-slabs.spec.ts`), or when `prepared`
   * carries the index; pass `false` to refine by walking.
   */
  edgeSlabs?: boolean;
  /** Caller-owned outputs. */
  output: GPUPointInPolygonJoinOutput<TruncationRows>;
};

/**
 * Joins planar points to the polygon features that contain them.
 *
 * A `GPUBVH` over feature bounds produces bounding-box candidates, an exact even/odd classifier
 * (exact orientation signs, see `boundaryRule`) refines them, and each point keeps the containing
 * feature with the smallest row.
 *
 * **Boundary points.** By default (`boundaryRule: 'half-open'`) a point on an edge or vertex shared
 * by several features of a valid coverage is assigned to exactly one of them, independent of the
 * feature row order, so counts over a coverage partition the points (only points on the outer right
 * and top boundary are left out). `'closed'` and `'open'` select the previous behaviors; see
 * {@link GPUPointInPolygonJoinProps.boundaryRule}. Zonal statistics rely on the half-open default. Per-feature counts and an optional compact list of joined points are produced on
 * the GPU. Every capacity reports overflow on the GPU.
 *
 * Candidates whose containment cannot be proven (`uncertain`) are not assigned and are not
 * silently ignored: they are counted in `uncertainCount`. Read it to know whether assignments may
 * be incomplete.
 *
 * Point positions, source IDs and point feature IDs may be chunked; source-aligned vectors have
 * matching ordered chunk lengths. Polygon views stay packed. Feature counts accumulate across
 * all chunks, and matches retain global source-row order, including the retained prefix on
 * overflow. Candidate capacity is shared across the encoding, so chunking does not change which
 * points lose candidates. Candidates stream through fixed-capacity scratch pages, including
 * pages that split one point's candidate segment. The GPU finds each page's query interval in
 * every source chunk, then dispatches only those rows. Pages preserve the existing BVH traversal order,
 * while assignments and compact matches retain their documented global source-row order.
 * Row-proportional assignment, probe and compaction scratch still needs
 * one packed storage binding, checked against the device before any nodes are created.
 */
export class GPUPointInPolygonJoin<
  TruncationRows extends GPUUint32Rows = GraphDataView<'uint32'>,
  Parameters = unknown,
  PointGeometry extends GPUPointGeometryPort | GPUMultipartPointGeometryPort =
    | GPUPointGeometryPort
    | GPUMultipartPointGeometryPort
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointInPolygonJoinProps<TruncationRows, Parameters>;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Construction-sized candidate scratch page. */
  readonly candidatePages: ReturnType<typeof planSpatialJoinCandidatePages>;
  /** Resolved boundary rule. */
  readonly boundaryRule: GPUPointInPolygonBoundaryRule;
  /** Resolved spatial context: native planar coordinates. */
  readonly spatialContext: GPUSpatialContext;
  /** Whether features are spatially sorted before the BVH build. */
  readonly spatialSort: boolean;
  /** Whether point chunks are processed in Hilbert order. */
  readonly querySort: boolean;
  /** Whether the exact refine tests only the edges of the y-slab at the point. */
  readonly edgeSlabs: boolean;

  /** Conservative node upper bound including published primitives. */
  readonly estimatedCommandNodeCount: number;

  constructor(inputProps: GPUPointInPolygonJoinProps<TruncationRows, Parameters, PointGeometry>) {
    this.id = inputProps.id ?? 'point-in-polygon-join';
    const {id} = this;
    const {output: requestedOutput, ...remainingProps} = inputProps;
    validateRequirement(
      id,
      !('featureCounts' in requestedOutput),
      'output.featureCounts was removed; use output.polygonAssignmentCounts or output.polygonMembershipCounts'
    );
    const capturedPoints = snapshotConstructionFields(inputProps.points, [
      'kind',
      'positions',
      'featureOffsets',
      'sourceIds'
    ]);
    const capturedOutput = snapshotConstructionFields(requestedOutput, [
      'pointFeatureIds',
      'pointFeatureRows',
      'polygonAssignmentCounts',
      'polygonMembershipCounts',
      'featurePolygonMembership',
      'pointMatchCounts',
      'matches',
      'candidateTruncated',
      'status'
    ]);
    const capturedMatches = capturedOutput.matches
      ? snapshotConstructionFields(capturedOutput.matches, ['ids', 'status'])
      : undefined;
    const props: GPUPointInPolygonJoinProps<TruncationRows, Parameters> = Object.freeze({
      ...remainingProps,
      points: Object.freeze({
        kind: capturedPoints.kind,
        positions: capturedPoints.positions,
        featureOffsets: capturedPoints.featureOffsets,
        sourceIds: capturedPoints.sourceIds
      }),
      polygons: snapshotConstructionFields(inputProps.polygons, [
        'kind',
        'positions',
        'featureOffsets',
        'polygonOffsets',
        'ringOffsets',
        'sourceIds'
      ]),
      output: Object.freeze({
        ...capturedOutput,
        status: snapshotConstructionFields(capturedOutput.status, [
          'overflow',
          'candidateOverflow',
          'candidateCount',
          'uncertainCount',
          'antimeridianUncertainCount',
          'poleUncertainCount'
        ]),
        matches: capturedMatches
          ? Object.freeze({
              ...capturedMatches,
              status: snapshotConstructionFields(capturedMatches.status, [
                'count',
                'overflow',
                'requiredCount',
                'candidateOverflow'
              ])
            })
          : undefined
      })
    });
    this.props = props;
    const {points, polygons, output} = props;
    const {status} = output;
    validateRequirement(
      id,
      Boolean(output.pointFeatureIds || output.pointFeatureRows),
      'output needs pointFeatureIds or pointFeatureRows'
    );
    if (points.kind !== 'points' || polygons.kind !== 'polygons') {
      throw new Error(`${id} needs point and polygon geometry`);
    }
    const multipart = points.featureOffsets !== undefined;
    const queryCount = multipart ? points.featureOffsets!.length - 1 : points.positions.length;
    if (multipart) validateGPUMultipartGeometryPort(id, points as GPUMultipartPointGeometryPort);
    for (const chunk of getGraphViewChunks(points.positions)) {
      validateSinglePackedView(chunk, ['float32x2'], `${id} points.positions`);
    }
    // Validate count metadata before it can become a WGSL literal or runtime uint32 word.
    for (const [name, view] of Object.entries({...getInputs(props), ...getOutputs(props)})) {
      if (!view) continue;
      validateCapacity(
        id,
        `${name}.length`,
        view.length,
        0,
        GPU_POINT_IN_POLYGON_JOIN_MAXIMUM_ROWS
      );
      for (const chunk of getGraphViewChunks(view)) {
        validateCapacity(
          id,
          `${name} chunk.length`,
          chunk.length,
          0,
          GPU_POINT_IN_POLYGON_JOIN_MAXIMUM_ROWS
        );
      }
    }
    validateSpatialJoinGeometry(id, 'polygons', polygons, {allowEmpty: true});
    this.featureCount = polygons.featureOffsets.length - 1;
    if (output.featurePolygonMembership) {
      validateSinglePackedView(
        output.featurePolygonMembership,
        ['uint32'],
        `${id} output.featurePolygonMembership`
      );
      validateRequirement(
        id,
        output.featurePolygonMembership.length === queryCount * this.featureCount,
        'output.featurePolygonMembership needs input feature count * polygon count rows'
      );
    }
    if (output.pointMatchCounts) {
      validateSinglePackedView(
        output.pointMatchCounts,
        ['uint32'],
        `${id} output.pointMatchCounts`
      );
      validateRequirement(
        id,
        output.pointMatchCounts.length === queryCount,
        `output.pointMatchCounts needs one row per input feature (${queryCount})`
      );
    }
    for (const [name, view] of [
      ['output.pointFeatureIds', output.pointFeatureIds],
      ['output.pointFeatureRows', output.pointFeatureRows],
      ['points.sourceIds', points.sourceIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validateSinglePackedView(chunk, ['uint32'], `${id} ${name}`);
      }
      if (multipart) {
        if (view)
          validateRequirement(
            id,
            !isGraphVectorView(view) && view.length === queryCount,
            `${name} needs one packed row per multipart feature (${queryCount})`
          );
      } else validateMatchingChunks(id, name, points.positions, view);
    }
    for (const [name, view] of [
      ['polygons.sourceIds', polygons.sourceIds],
      ['output.polygonAssignmentCounts', output.polygonAssignmentCounts],
      ['output.polygonMembershipCounts', output.polygonMembershipCounts]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} ${name} needs one row per feature`);
        }
      }
    }
    validateStatusScalars(id, status);
    validateSpatialJoinReadWriteBindings(
      id,
      [output.featurePolygonMembership],
      [output.polygonMembershipCounts, output.pointMatchCounts]
    );
    if (output.candidateTruncated) {
      for (const chunk of getGraphViewChunks(output.candidateTruncated)) {
        validateSinglePackedView(chunk, ['uint32'], `${id} output.candidateTruncated`);
      }
      if (multipart)
        validateRequirement(
          id,
          !isGraphVectorView(output.candidateTruncated),
          'output.candidateTruncated needs packed multipart feature rows'
        );
      if (!multipart && isGraphVectorView(output.candidateTruncated)) {
        validateMatchingChunks(
          id,
          'output.candidateTruncated',
          points.positions,
          output.candidateTruncated
        );
      }
      if (output.candidateTruncated.length !== queryCount) {
        throw new Error(`${id} output.candidateTruncated needs one row per point`);
      }
    }
    if (output.matches) {
      validateCompactOutput(id, output.matches);
    }
    this.candidateCapacity = props.candidateCapacity;
    // Candidate pairs are stored as two words each in a uint32-addressed buffer.
    validateCapacity(
      id,
      'candidateCapacity',
      this.candidateCapacity,
      1,
      GPU_POINT_IN_POLYGON_JOIN_MAXIMUM_CANDIDATES
    );
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    if (
      props.spatialContext?.coordinateSpace === 'longitude-latitude' &&
      props.spatialContext.metric !== 'none'
    ) {
      throw new Error(`${id} needs metric 'none' on longitude-latitude`);
    }
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
    resolveGPUSecondarySpatialContext(
      id,
      'polygons',
      props.polygonsSpatialContext,
      this.spatialContext,
      SUPPORT.edgeModels
    );
    if (
      props.boundaryRule !== undefined &&
      !['half-open', 'closed', 'open'].includes(props.boundaryRule)
    ) {
      throw new Error(`${id} boundaryRule must be 'half-open', 'closed' or 'open'`);
    }
    this.boundaryRule = props.boundaryRule ?? 'half-open';
    this.spatialSort =
      props.prepared?.spatialSort ?? props.spatialSort ?? getDefaultSpatialSort(this.featureCount);
    this.querySort =
      props.querySort ?? getDefaultQuerySort(props.points.positions.length, this.featureCount);
    this.edgeSlabs =
      props.edgeSlabs ??
      props.prepared?.edgeSlabs ??
      getDefaultEdgeSlabs(polygons.positions.length, this.candidateCapacity, this.featureCount);
    if (props.prepared) {
      this.leafCapacity = props.prepared.leafCapacity;
      if (!isSameSpatialJoinGeometry(props.prepared.geometry, polygons)) {
        throw new Error(`${id} prepared must index the same polygon views`);
      }
    }
    validateCapacity(
      id,
      'point offset scratch words',
      points.positions.length +
        SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT * getGraphViewChunks(points.positions).length,
      0,
      GPU_POINT_IN_POLYGON_JOIN_MAXIMUM_ROWS
    );
    this.candidatePages = planSpatialJoinCandidatePages(
      (polygons.positions.buffer.graph as GPUCommandGraph).device,
      id,
      this.candidateCapacity,
      props.candidateTileCapacity,
      this.edgeSlabs ? 16 : 8,
      0,
      getGraphViewChunks(points.positions).length
    );
    const hasQuerySort =
      this.querySort && getGraphViewChunks(points.positions).some(chunk => chunk.length >= 2);
    const pointChunkCount = getGraphViewChunks(points.positions).filter(
      chunk => chunk.length > 0
    ).length;
    this.estimatedCommandNodeCount =
      this.candidatePages.estimatedCommandNodeCount +
      GPU_POINT_IN_POLYGON_JOIN_FIXED_NODE_ALLOWANCE +
      pointChunkCount *
        (hasQuerySort
          ? GPU_POINT_IN_POLYGON_JOIN_CHUNK_NODE_ALLOWANCE
          : GPU_POINT_IN_POLYGON_JOIN_UNSORTED_CHUNK_NODE_ALLOWANCE);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateDisjointOutputs(
      id,
      Object.values(getInputs(props, this.edgeSlabs)),
      Object.values(getOutputs(props)),
      [
        {
          pointFeatureIds: output.pointFeatureIds,
          pointFeatureRows: output.pointFeatureRows,
          polygonAssignmentCounts: output.polygonAssignmentCounts
        },
        {
          overflow: output.status.overflow,
          candidateCount: output.status.candidateCount,
          uncertainCount: output.status.uncertainCount,
          matchCount: output.matches?.status.count,
          matchOverflow: output.matches?.status.overflow,
          matchRequired: output.matches?.status.requiredCount
        },
        ...(this.spatialContext.coordinateSpace === 'longitude-latitude'
          ? [
              {
                antimeridian: output.status.antimeridianUncertainCount,
                pole: output.status.poleUncertainCount
              }
            ]
          : [])
      ]
    );
    if (points.positions.length > 0)
      validateSpatialJoinReadWriteBindings(
        id,
        [polygons.sourceIds],
        [output.pointFeatureIds, output.pointFeatureRows, output.polygonAssignmentCounts]
      );
    if (props.prepared)
      validateSpatialJoinReadWriteBindings(
        id,
        [
          (props.prepared.isDeclaredIn(polygons.positions.buffer.graph)
            ? props.prepared.storage
            : props.prepared.props.storage
          )?.overflow
        ],
        [
          output.status.overflow,
          output.status.candidateCount,
          output.status.uncertainCount,
          output.matches?.status.count,
          output.matches?.status.overflow,
          output.matches?.status.requiredCount
        ]
      );
  }

  /** Returns bounds, BVH, candidate, refinement, assignment, and finalize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, leafCapacity, candidateCapacity} = this;
    const {output} = props;
    const points = props.points.positions;
    const {matches, status} = output;
    const multipart = props.points.featureOffsets !== undefined;
    const queryCount = multipart ? props.points.featureOffsets!.length - 1 : points.length;
    if (props.prepared && !props.prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} requires prepared to be added to the graph first`);
    }
    const captured = props.prepared?.getGeometry(graph);
    const polygons =
      captured?.kind === 'polygons'
        ? {...captured, sourceIds: props.polygons.sourceIds}
        : props.polygons;
    const views = {...getInputs(props, this.edgeSlabs), ...getOutputs(props)};
    validateSpatialJoinBindings(graph, id, views, {});
    const checkScratch = (name: string, byteLength: number, hint: string) => {
      validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, hint);
      validateCapacity(
        id,
        `${name} byteLength (device.limits.maxBufferSize)`,
        Math.max(4, byteLength),
        0,
        graph.device.limits.maxBufferSize
      );
    };
    if (output.featurePolygonMembership) {
      checkScratch(
        'feature-polygon membership',
        queryCount * featureCount * 4,
        'reduce the input feature or polygon count'
      );
      checkScratch(
        'multipart feature assignment',
        queryCount * 4,
        'reduce the input feature count'
      );
    }
    if (multipart)
      checkScratch('multipart feature assignment', queryCount * 4, 'reduce input feature count');
    if (output.pointMatchCounts || output.polygonMembershipCounts)
      checkScratch(
        'multipart distinct candidate pairs',
        this.candidateCapacity * 4,
        'reduce candidateCapacity'
      );
    checkScratch(
      'point assignment, probe and match scratch',
      (points.length + SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT * getGraphViewChunks(points).length) *
        Uint32Array.BYTES_PER_ELEMENT,
      'reduce points.positions.length; join scratch requires one packed binding'
    );
    checkScratch('state', 20, 'use a device supporting at least 20-byte storage bindings');
    checkScratch(
      'candidate pairs',
      this.candidatePages.capacity * (this.edgeSlabs ? 4 : 2) * Uint32Array.BYTES_PER_ELEMENT,
      'reduce candidateCapacity'
    );
    if (!props.prepared) {
      checkScratch(
        'BVH node bounds and children',
        (2 * leafCapacity - 1) * 8,
        'reduce leafCapacity'
      );
      checkScratch('BVH leaf IDs', leafCapacity * 4, 'reduce leafCapacity');
      checkScratch(
        'feature bounds and sorted bounds',
        featureCount * 8,
        'reduce polygons feature count'
      );
      checkScratch(
        'feature radix histogram',
        16 * Math.ceil(featureCount / 256) * 4,
        'reduce polygons feature count'
      );
      checkScratch(
        'feature sort tile bounds',
        Math.ceil(featureCount / 256) * 16,
        'reduce polygons feature count'
      );
    } else {
      for (const view of Object.values(props.prepared.storage)) {
        checkScratch(
          'prepared BVH storage',
          getViewBindingRange(view).size,
          'reduce prepared leafCapacity'
        );
      }
    }
    if (this.querySort) {
      checkScratch('point query sort', points.length * 4, 'reduce points.positions.length');
      checkScratch(
        'point query radix histogram',
        16 * Math.ceil(points.length / 256) * 4,
        'reduce points.positions.length'
      );
    }
    let slabEntryCapacity: number | undefined;
    if (this.edgeSlabs) {
      const preparedSlabs = props.prepared?.edgeSlabStorage;
      if (preparedSlabs) {
        for (const view of [
          preparedSlabs.slabs.params,
          preparedSlabs.slabs.slabOffsets,
          preparedSlabs.slabs.slabEntries,
          preparedSlabs.layout
        ]) {
          checkScratch(
            'prepared edge slab storage',
            getViewBindingRange(view).size,
            'reduce the prepared polygon index size'
          );
        }
      } else {
        const vertexCount = polygons.positions.length;
        const ownerCount = polygons.polygonOffsets.length - 1;
        checkScratch(
          'edge slab parameters',
          Math.max(ownerCount, 1) * POLYGON_EDGE_SLAB_PARAMS_STRIDE * 4,
          'reduce polygons part count or disable edgeSlabs'
        );
        checkScratch(
          'edge slab owner scan',
          (ownerCount + 1) * 4,
          'reduce polygons part count or disable edgeSlabs'
        );
        checkScratch(
          'edge slab ring owners',
          Math.max(polygons.ringOffsets.length - 1, 1) * 4,
          'reduce polygons ring count or disable edgeSlabs'
        );
        checkScratch(
          'edge slab vertex owners',
          vertexCount * 4,
          'reduce polygons vertex count or disable edgeSlabs'
        );
        checkScratch(
          'edge slab offsets',
          (Math.floor(vertexCount / 4) + 2) * 4,
          'reduce polygons vertex count or disable edgeSlabs'
        );
        checkScratch(
          'edge slab base edges',
          vertexCount * 8,
          'reduce polygons vertex count or disable edgeSlabs'
        );
        // Supply the planned budget to the builder so its complete entry allocation, including
        // the unslabbed edge prefix, is the size validated here against both device limits.
        slabEntryCapacity = Math.min(
          getStorageBindingElementLimit(graph.device, 8, vertexCount),
          Math.max(
            GPU_POINT_IN_POLYGON_JOIN_SLAB_ENTRIES_PER_VERTEX * vertexCount,
            GPU_POINT_IN_POLYGON_JOIN_MINIMUM_SLAB_ENTRIES
          )
        );
        checkScratch(
          'edge slab entries',
          (vertexCount + slabEntryCapacity) * 8,
          'reduce polygons vertex count or disable edgeSlabs'
        );
        checkScratch(
          'edge slab layout',
          getPolygonEdgeLayoutLength(vertexCount, featureCount) * 4,
          'reduce polygons feature count or disable edgeSlabs'
        );
      }
    }
    validateGraphChunkGroupKernelSupport(graph, id, 8);
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      hasLonLatPolygonStatus(status)
    ) {
      for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
        const view = status[name];
        if (view) {
          nodes.push(
            createFillNode(graph, {
              id: `${id}-clear-${name}`,
              operation: OPERATION,
              view,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          );
        }
      }
      nodes.push(
        createLonLatPolygonCheckNode(graph, {
          id: `${id}-lon-lat-polygon-check-0`,
          operation: OPERATION,
          positions: polygons.positions,
          ringOffsets: polygons.ringOffsets,
          status
        })
      );
    }

    let bvh: Pick<
      GPUBVH,
      'nodeMinima' | 'nodeMaxima' | 'leafIds' | 'overflow' | 'internalNodeCount'
    >;
    if (props.prepared) {
      const {storage} = props.prepared;
      bvh = {...storage, internalNodeCount: props.prepared.bvhInternalNodeCount};
    } else {
      const minima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', featureCount);
      const maxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', featureCount);
      if (featureCount > 0) {
        nodes.push(
          createSpatialJoinBoundsNode<Parameters>(graph, {
            id: `${id}-bounds`,
            operation: OPERATION,
            featureCount,
            source: polygons,
            minima,
            maxima
          })
        );
      }
      const built = getSortedFeatureBVHNodes(
        graph,
        id,
        OPERATION,
        minima,
        maxima,
        leafCapacity,
        this.spatialSort,
        undefined,
        this.props.spatialSortCurve
      );
      nodes.push(...built.nodes);
      bvh = built.bvh;
    }

    const state = createTransientView(graph, `${id}-state`, 'uint32', 5);
    const assignment = createTransientView(graph, `${id}-assignment`, 'uint32', pointCount);
    const publishedAssignment = multipart
      ? createTransientView(graph, `${id}-feature-assignment`, 'uint32', queryCount)
      : assignment;
    if (multipart)
      nodes.push(
        createFillNode(graph, {
          id: `${id}-clear-feature-assignment`,
          operation: OPERATION,
          view: publishedAssignment,
          type: 'u32',
          value: '0xffffffffu'
        })
      );
    const needsDistinct = Boolean(output.pointMatchCounts || output.polygonMembershipCounts);
    const matchOwners = needsDistinct
      ? createTransientView(graph, `${id}-match-owners`, 'uint32', this.candidateCapacity)
      : undefined;
    const matchPolygons = needsDistinct
      ? createTransientView(graph, `${id}-match-polygons`, 'uint32', this.candidateCapacity)
      : undefined;
    for (const view of [matchOwners, matchPolygons])
      if (view)
        nodes.push(
          createFillNode(graph, {
            id: `${id}-clear-${view === matchOwners ? 'owners' : 'polygons'}`,
            operation: OPERATION,
            view,
            type: 'u32',
            value: '0xffffffffu'
          })
        );
    let featurePairs: GraphDataView<'uint32'> | undefined;
    let partOwners: GraphDataView<'uint32'> | undefined;
    if (multipart || output.featurePolygonMembership || needsDistinct) {
      featurePairs = output.featurePolygonMembership;
      if (multipart) {
        partOwners = createTransientView(graph, `${id}-part-owners`, 'uint32', pointCount);
        nodes.push(
          createGPUMultipartRowOwnersNode(
            graph,
            `${id}-part-owners`,
            props.points as GPUMultipartPointGeometryPort,
            partOwners
          )
        );
      }
      if (featurePairs)
        nodes.push(
          createFillNode(graph, {
            id: `${id}-clear-feature-pairs`,
            operation: OPERATION,
            view: featurePairs,
            type: 'u32',
            value: '0u'
          })
        );
    }
    // With slabs a candidate is four words in one binding: row, feature, point x and y bits.
    const useSlabs = this.edgeSlabs;
    const pairStride = useSlabs ? 4 : 2;
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      this.candidatePages.capacity * (pairStride / 2)
    );
    const candidatePoints = useSlabs
      ? undefined
      : createTransientView(
          graph,
          `${id}-candidate-points`,
          'float32x2',
          this.candidatePages.capacity
        );
    let slabs: PolygonEdgeSlabs | undefined;
    let edgeLayout: GraphDataView<'uint32'> | undefined;
    if (useSlabs) {
      const preparedSlabs = props.prepared?.edgeSlabStorage;
      if (preparedSlabs) {
        ({slabs, layout: edgeLayout} = preparedSlabs);
      } else {
        const built = getPolygonEdgeSlabNodes<Parameters>(graph, {
          id: `${id}-edges`,
          operation: OPERATION,
          owner: 'polygon',
          entryCapacity: slabEntryCapacity || undefined,
          positions: polygons.positions,
          polygonOffsets: polygons.polygonOffsets,
          ringOffsets: polygons.ringOffsets
        });
        const layoutNodes = getPolygonEdgeLayoutNodes<Parameters>(graph, {
          id: `${id}-edges`,
          operation: OPERATION,
          polygons
        });
        nodes.push(...built.nodes, ...layoutNodes.nodes);
        slabs = built.slabs;
        edgeLayout = layoutNodes.layout;
      }
    }
    const pairClassifications = createTransientView(
      graph,
      `${id}-pair-classifications`,
      'uint32',
      this.candidatePages.capacity
    );
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        state,
        assignment,
        pointCount
      })
    );
    for (const [name, view] of [
      ['assignment-counts', output.polygonAssignmentCounts],
      ['membership-counts', output.polygonMembershipCounts]
    ] as const)
      if (view)
        nodes.push(
          createFillNode(graph, {
            id: `${id}-clear-${name}`,
            operation: OPERATION,
            view,
            type: 'u32',
            value: '0u'
          })
        );
    const callerTruncation: GPUUint32Rows | undefined = multipart
      ? createTransientView(graph, `${id}-part-truncated`, 'uint32', pointCount)
      : output.candidateTruncated;
    const candidateTruncated = isGraphVectorView(callerTruncation)
      ? createTransientView(graph, `${id}-candidate-truncated`, 'uint32', pointCount)
      : callerTruncation;
    const pageCounts = createTransientView(graph, `${id}-page-counts`, 'uint32', pointCount);
    const pageOffsets = createTransientView(
      graph,
      `${id}-page-offsets`,
      'uint32',
      pointCount + SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT * getGraphViewChunks(points).length,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    const pageOrders = new Map<number, GraphDataView<'uint32'>>();
    for (let pageIndex = 0; pageIndex < this.candidatePages.pageCount; pageIndex++) {
      const id = this.candidatePages.pageCount > 1 ? `${this.id}-tile-${pageIndex}` : this.id;
      const candidateCapacity = this.candidatePages.capacity;
      const firstSlot = pageIndex * candidateCapacity;
      const pageLength = Math.min(candidateCapacity, this.candidateCapacity - firstSlot);
      nodes.push(
        ...getSpatialJoinProbeNodes<Parameters>(graph, {
          id: `${id}-probe`,
          operation: OPERATION,
          points,
          bvh,
          featureCount,
          candidateCapacity: this.candidateCapacity,
          page: {
            firstSlot,
            length: pageLength,
            initialize: pageIndex === 0,
            counts: pageCounts,
            offsets: pageOffsets,
            orders: pageOrders,
            metadataRow: pointCount
          },
          state,
          candidatePairs,
          candidatePoints,
          packedCandidates: useSlabs,
          pointRowScale: 1,
          pointRowOffset: 0,
          querySort: this.querySort,
          candidateTruncated,
          candidateOverflow: output.status.candidateOverflow,
          bvhOverflow: bvh.overflow
        })
      );
      if (
        pageIndex === 0 &&
        !multipart &&
        isGraphVectorView(output.candidateTruncated) &&
        candidateTruncated
      ) {
        let firstRow = 0;
        for (const [chunkIndex, chunk] of output.candidateTruncated.data.entries()) {
          if (chunk.length > 0) {
            nodes.push(
              createSpatialJoinChunkNode<Parameters>(graph, {
                id: `${id}-candidate-truncated-${chunkIndex}`,
                operation: OPERATION,
                bindings: [
                  {name: 'flags', view: candidateTruncated, type: 'u32', access: 'read'},
                  {name: 'outputFlags', view: chunk, type: 'u32', access: 'read_write'}
                ],
                invocationCount: chunk.length,
                firstRow,
                body: `outputFlags[outputFlagsOffset + index] = flags[flagsOffset + CHUNK_FIRST_ROW + index];`
              })
            );
          }
          firstRow += chunk.length;
        }
      }
      nodes.push(
        createSpatialJoinChunkNode(graph, {
          id: `${id}-page-count`,
          operation: OPERATION,
          invocationCount: 1,
          firstRow: 0,
          candidatePageFirstSlot: firstSlot,
          candidatePageLength: pageLength,
          bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
          body: `let retained = min(state[stateOffset], ${this.candidateCapacity}u);
  let first = min(CANDIDATE_PAGE_FIRST_SLOT, retained);
  state[stateOffset + 4u] = min(retained - first, CANDIDATE_PAGE_LENGTH);`
        })
      );
      const candidateDispatch = createWGSLActiveCountDispatch(graph, {
        id: `${id}-candidate-dispatch`,
        operation: OPERATION,
        count: state,
        countRow: 4,
        maximumItemCount: candidateCapacity
      });
      nodes.push(candidateDispatch.updateNode);
      if (slabs && edgeLayout) {
        nodes.push(
          this.createSlabClassifyNode(graph, {
            polygons,
            condition: candidateDispatch.condition,
            id,
            capacity: candidateCapacity,
            state,
            candidatePairs,
            pairClassifications,
            slabs,
            edgeLayout
          })
        );
      } else {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-classify`,
            operation: OPERATION,
            variant: 'classify-exact',
            bindings: [
              {name: 'state', view: state, type: 'u32', access: 'read'},
              {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
              {name: 'candidatePoints', view: candidatePoints!, type: 'f32', access: 'read'},
              {name: 'featureOffsets', view: polygons.featureOffsets, type: 'u32', access: 'read'},
              {name: 'polygonPositions', view: polygons.positions, type: 'f32', access: 'read'},
              {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
              {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
              {
                name: 'pairClassifications',
                view: pairClassifications,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: candidateCapacity,
            condition: candidateDispatch.condition,
            declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
  const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;
  const RING_COUNT: u32 = ${polygons.ringOffsets.length - 1}u;
  const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
  const OUTSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.outside}u;
  const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
  const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
  const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
  const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
  ${EXACT_ORIENTATION_WGSL}
  fn readVertex(vertexIndex: u32) -> vec2f {
    return vec2f(
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
    );
  }
  // Exact even/odd classification, or UNCERTAIN when some orientation has no sign. A BOUNDARY result
  // also carries HALF_OPEN_OWNER_BIT when the point belongs to the feature under the half-open rule.
  //
  // Half-open rule: classify the point moved by (epsilon^2, epsilon) with epsilon infinitesimal. The
  // crossing test against an edge only changes when the point is on the edge's supporting line, where
  // the sign of the moved point is the sign of dx (the epsilon term), or of -dy for a vertical edge
  // (the epsilon^2 term). The y comparisons are unchanged because a vertex y is either equal to the
  // point's y or at least one ulp away.
  fn classifyExactly(point: vec2f, geometryStart: u32, geometryEnd: u32) -> u32 {
    var geometryInside = false;
    var geometryOwned = false;
    var onBoundary = false;
    for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
      let polygonStart = polygonOffsets[polygonOffsetsOffset + polygonIndex];
      let polygonEnd = polygonOffsets[polygonOffsetsOffset + polygonIndex + 1u];
      // Malformed offsets cannot be classified; an empty part contains nothing (as in shapely) and
      // must not hide the other parts of the feature.
      if (polygonStart > polygonEnd || polygonEnd > RING_COUNT) { return UNCERTAIN; }
      var polygonInside = false;
      var polygonOwned = false;
      for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
        let ringStart = ringOffsets[ringOffsetsOffset + ringIndex];
        let ringEnd = ringOffsets[ringOffsetsOffset + ringIndex + 1u];
        if (ringEnd < ringStart || ringEnd > VERTEX_COUNT) { return UNCERTAIN; }
        // A ring with fewer than 3 vertices bounds no area: skip it, as shapely treats empty parts.
        if (ringEnd < ringStart + 3u) { continue; }
        var previous = readVertex(ringEnd - 1u);
        for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
          let current = readVertex(vertexIndex);
          let sign = orientSign(previous, current, point);
          if (sign == 2) { return UNCERTAIN; }
          var side = sign;
          if (sign == 0) {
            if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
                point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
              onBoundary = true;
            }
            side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
            if (side == 0) {
              side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
            }
          }
          if ((previous.y > point.y) != (current.y > point.y)) {
            if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
            if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
            if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
              polygonOwned = !polygonOwned;
            }
          }
          previous = current;
        }
      }
      geometryInside = geometryInside || polygonInside;
      geometryOwned = geometryOwned || polygonOwned;
    }
    if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
    return select(OUTSIDE, INSIDE, geometryInside);
  }`,
            body: `let activeCount = min(state[stateOffset + 4u], CANDIDATE_CAPACITY);
    if (index >= activeCount) { return; }
    let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let geometryStart = featureOffsets[featureOffsetsOffset + featureRow];
    let geometryEnd = featureOffsets[featureOffsetsOffset + featureRow + 1u];
    var classification = UNCERTAIN;
    if (geometryStart <= geometryEnd && geometryEnd <= POLYGON_COUNT) {
      let point = vec2f(candidatePoints[candidatePointsOffset + index * 2u], candidatePoints[candidatePointsOffset + index * 2u + 1u]);
      classification = classifyExactly(point, geometryStart, geometryEnd);
    }
    pairClassifications[pairClassificationsOffset + index] = classification;`
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-resolve`,
          operation: OPERATION,
          variant: 'resolve',
          bindings: [
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'pairClassifications', view: pairClassifications, type: 'u32', access: 'read'},
            {
              name: 'assignment',
              view: publishedAssignment,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            ...(partOwners
              ? [
                  {
                    name: 'partOwners',
                    view: partOwners,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            ...(featurePairs
              ? [
                  {
                    name: 'featurePairs',
                    view: featurePairs,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(matchOwners && matchPolygons
              ? [
                  {
                    name: 'matchOwners',
                    view: matchOwners,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  },
                  {
                    name: 'matchPolygons',
                    view: matchPolygons,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: candidateCapacity,
          condition: candidateDispatch.condition,
          declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
const BOUNDARY_CLOSED: bool = ${this.boundaryRule === 'closed'};
const BOUNDARY_HALF_OPEN: bool = ${this.boundaryRule === 'half-open'};`,
          body: `let activeCount = min(atomicLoad(&state[stateOffset + 4u]), CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * ${pairStride}u];
  let featureRow = candidatePairs[candidatePairsOffset + index * ${pairStride}u + 1u];
  let word = pairClassifications[pairClassificationsOffset + index];
  let classification = word & ~HALF_OPEN_OWNER_BIT;
  let owned = BOUNDARY_CLOSED || (BOUNDARY_HALF_OPEN && (word & HALF_OPEN_OWNER_BIT) != 0u);
  if (classification == INSIDE || (owned && classification == BOUNDARY)) {
    let owner = ${partOwners ? 'partOwners[partOwnersOffset + pointRow]' : 'pointRow'};
    if (owner < ${queryCount}u) {
      atomicMin(&assignment[assignmentOffset + owner],featureRow);
      ${featurePairs ? 'atomicStore(&featurePairs[featurePairsOffset + owner * ' + featureCount + 'u + featureRow],1u);' : ''}
      ${matchOwners ? 'matchOwners[matchOwnersOffset + ' + firstSlot + 'u + index] = owner; matchPolygons[matchPolygonsOffset + ' + firstSlot + 'u + index] = featureRow;' : ''}
    }
  } else if (classification == UNCERTAIN) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`
        })
      );
    }
    if (matchOwners && matchPolygons)
      nodes.push(
        ...getMultipartPointCountNodes(graph, {
          id: `${id}-distinct`,
          owners: matchOwners,
          polygons: matchPolygons,
          queryCount,
          polygonCount: featureCount,
          polygonMembershipCounts: output.polygonMembershipCounts,
          pointMatchCounts: output.pointMatchCounts
        })
      );
    if (multipart && output.candidateTruncated && candidateTruncated)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-reduce-truncation`,
          operation: OPERATION,
          bindings: [
            {name: 'offsets', view: props.points.featureOffsets!, type: 'u32', access: 'read'},
            {name: 'parts', view: candidateTruncated, type: 'u32', access: 'read'},
            {
              name: 'features',
              view: output.candidateTruncated as GraphDataView<'uint32'>,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: queryCount,
          body: `var truncated = 0u;
  for (var part = offsets[offsetsOffset + index]; part < min(offsets[offsetsOffset + index + 1u], ${pointCount}u); part++) { truncated |= parts[partsOffset + part]; }
  features[featuresOffset + index] = truncated;`
        })
      );
    nodes.push(
      ...getSpatialJoinAssignNodes<Parameters>(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        assignment: publishedAssignment,
        pointFeatureIds: output.pointFeatureIds,
        pointFeatureRows: output.pointFeatureRows,
        featureIds: polygons.sourceIds,
        polygonAssignmentCounts: output.polygonAssignmentCounts
      })
    );
    if (matches) {
      nodes.push(
        ...getSpatialJoinCollectNodes<Parameters>(graph, {
          id: `${id}-collect-matches`,
          operation: OPERATION,
          points: multipart ? (output.pointFeatureIds ?? output.pointFeatureRows!) : points,
          assignment: publishedAssignment,
          state,
          sourceIds: props.points.sourceIds,
          matches
        })
      );
    }
    nodes.push(
      ...createSpatialJoinFinalizeNodes<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        state,
        bvhOverflow: bvh.overflow,
        candidateCapacity,
        overflow: status.overflow,
        candidateCount: status.candidateCount,
        uncertainCount: status.uncertainCount,
        matches
      })
    );
    return nodes;
  }

  /**
   * Slab-based exact refine: one thread per candidate classifies the point against the edges of
   * the y-slab of each polygon part of the feature. Exactly eight bindings: state, the packed
   * candidates, classifications, polygon positions, the three slab views and the edge layout.
   */
  private createSlabClassifyNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: {
      polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
      condition: GPUCommandNode<Parameters>['condition'];
      id: string;
      capacity: number;
      state: GraphDataView<'uint32'>;
      candidatePairs: GraphDataView;
      pairClassifications: GraphDataView<'uint32'>;
      slabs: PolygonEdgeSlabs;
      edgeLayout: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters> {
    const {id, capacity: candidateCapacity} = props;
    const {polygons} = props;
    const edgeLoop = getPolygonEdgeSlabLoopWGSL({
      owner: 'polygonIndex',
      lowY: 'point.y',
      highY: 'point.y',
      getVertex: 'readVertex',
      pointNames: ['previous', 'current'],
      // One call site for the exact helpers. The bit is set when the edge starts in a ring of at
      // least three vertices, the rings the unslabbed walk uses.
      body: `let isUsableRing = ((edgeLayout[edgeLayoutOffset + (edgeStart >> 5u)] >> (edgeStart & 31u)) & 1u) != 0u;
        if (!isUsableRing) { continue; }
        let sign = orientSign(previous, current, point);
        if (sign == 2) { return UNCERTAIN; }
        var side = sign;
        if (sign == 0) {
          if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
              point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
            onBoundary = true;
          }
          side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
          if (side == 0) {
            side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
          }
        }
        if ((previous.y > point.y) != (current.y > point.y)) {
          if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
          if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
          if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
            polygonOwned = !polygonOwned;
          }
        }`
    });
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-classify`,
      operation: OPERATION,
      variant: 'classify-exact-slabs',
      bindings: [
        {name: 'state', view: props.state, type: 'u32', access: 'read'},
        {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
        {name: 'polygonPositions', view: polygons.positions, type: 'f32', access: 'read'},
        ...getPolygonEdgeSlabBindings(props.slabs),
        {name: 'edgeLayout', view: props.edgeLayout, type: 'u32', access: 'read'},
        {
          name: 'pairClassifications',
          view: props.pairClassifications,
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: candidateCapacity,
      condition: props.condition,
      declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const OUTSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.outside}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
const NO_PART: u32 = 0xffffffffu;
const FEATURE_BASE: u32 = ${Math.max(Math.ceil(polygons.positions.length / 32), 1)}u;
${EXACT_ORIENTATION_WGSL}
${getPolygonEdgeSlabQueryWGSL(props.slabs)}
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
  );
}
// Same classification as the unslabbed classifyExactly, over the edges of the slab at point.y of
// each part: see the edgeSlabs prop. Parity is per part (each part is a slab owner) and combined
// with OR across parts.
fn classifySlabs(point: vec2f, partStart: u32, partEnd: u32) -> u32 {
  var geometryInside = false;
  var geometryOwned = false;
  var onBoundary = false;
  for (var polygonIndex = partStart; polygonIndex < partEnd; polygonIndex++) {
    var polygonInside = false;
    var polygonOwned = false;
    ${edgeLoop}
    geometryInside = geometryInside || polygonInside;
    geometryOwned = geometryOwned || polygonOwned;
  }
  if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
  return select(OUTSIDE, INSIDE, geometryInside);
}`,
      body: `let activeCount = min(state[stateOffset + 4u], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let candidate = candidatePairsOffset + index * 4u;
  let featureRow = candidatePairs[candidate + 1u];
  let point = vec2f(bitcast<f32>(candidatePairs[candidate + 2u]), bitcast<f32>(candidatePairs[candidate + 3u]));
  let feature = edgeLayoutOffset + FEATURE_BASE + featureRow * 2u;
  var classification = UNCERTAIN;
  let partStart = edgeLayout[feature];
  // A malformed feature (partStart == NO_PART) is UNCERTAIN, as in the unslabbed walk.
  if (partStart != NO_PART) {
    classification = classifySlabs(point, partStart, edgeLayout[feature + 1u]);
  }
  pairClassifications[pairClassificationsOffset + index] = classification;`
    });
  }
}

function getInputs(
  props: GPUPointInPolygonJoinProps<GPUUint32Rows, never>,
  edgeSlabs = true
): Record<string, GraphDataView | GraphVectorView | undefined> {
  const {points, polygons} = props;
  return {
    'points.positions': points.positions,
    'points.featureOffsets': points.featureOffsets,
    'points.sourceIds':
      props.output.matches && points.positions.length > 0 ? points.sourceIds : undefined,
    'polygons.positions': polygons.positions,
    'polygons.featureOffsets': polygons.featureOffsets,
    'polygons.polygonOffsets': polygons.polygonOffsets,
    'polygons.ringOffsets': polygons.ringOffsets,
    'polygons.sourceIds': points.positions.length > 0 ? polygons.sourceIds : undefined,
    ...Object.fromEntries(
      Object.entries(
        props.prepared?.isDeclaredIn(polygons.positions.buffer.graph)
          ? props.prepared.storage
          : (props.prepared?.props.storage ?? {})
      ).map(([name, view]) => [`prepared.${name}`, view])
    ),
    ...(edgeSlabs && props.prepared?.edgeSlabStorage
      ? {
          'prepared.slabParams': props.prepared.edgeSlabStorage.slabs.params,
          'prepared.slabOffsets': props.prepared.edgeSlabStorage.slabs.slabOffsets,
          'prepared.slabEntries': props.prepared.edgeSlabStorage.slabs.slabEntries,
          'prepared.edgeLayout': props.prepared.edgeSlabStorage.layout
        }
      : {})
  };
}

function getOutputs(
  props: GPUPointInPolygonJoinProps<GPUUint32Rows, never>
): Record<string, GraphDataView | GraphVectorView | undefined> {
  const {output} = props;
  const {matches, status} = output;
  return {
    'output.pointFeatureIds': output.pointFeatureIds,
    'output.pointFeatureRows': output.pointFeatureRows,
    'output.polygonAssignmentCounts': output.polygonAssignmentCounts,
    'output.polygonMembershipCounts': output.polygonMembershipCounts,
    'output.pointMatchCounts': output.pointMatchCounts,
    'output.featurePolygonMembership': output.featurePolygonMembership,
    'output.candidateTruncated': output.candidateTruncated,
    'output.status.overflow': status.overflow,
    'output.status.candidateOverflow': status.candidateOverflow,
    'output.status.candidateCount': status.candidateCount,
    'output.status.uncertainCount': status.uncertainCount,
    ...(props.spatialContext?.coordinateSpace === 'longitude-latitude'
      ? {
          'output.status.antimeridianUncertainCount': status.antimeridianUncertainCount,
          'output.status.poleUncertainCount': status.poleUncertainCount
        }
      : {}),
    'output.matches.ids': matches?.ids,
    'output.matches.status.count': matches?.status.count,
    'output.matches.status.overflow': matches?.status.overflow,
    'output.matches.status.requiredCount': matches?.status.requiredCount,
    'output.matches.status.candidateOverflow': matches?.status.candidateOverflow
  };
}
