// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCandidateStatusPort,
  GPULonLatPolygonStatusPort
} from '../../analysis-core/contracts/index';
import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/kernels/lon-lat-ring-wgsl';
import {
  validateSpatialJoinBindings,
  validateSpatialJoinReadWriteBindings
} from './spatial-join-binding-preflight';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandGraphGPUIndirectCondition,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '@luma.gl/graph/geospatial';
import {
  getWGSLFloatLiteral,
  createWGSLActiveCountDispatch,
  createWGSLKernelNode,
  type WGSLActiveCountDispatch,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  captureGraphCommandNodes,
  validateCompactOutput,
  validateGraphViewsBelongToGraph,
  validatePowerOfTwo,
  validateCapacity,
  validateFiniteNumber,
  validateNonZeroFiniteNumber,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport,
  getGPUSphereRadius,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry,
  isMultipartJoinGeometry,
  validateStatusScalars
} from './spatial-join-geometry';
import {
  getSpatialJoinCandidateNodes,
  type SpatialJoinCandidateTree
} from './spatial-join-candidates';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
import {
  getFeatureBVHNodes,
  getNextPowerOfTwo,
  SPATIAL_JOIN_WGSL_HELPERS,
  validateDisjointOutputs
} from './spatial-join-passes';
import {SPATIAL_GEODESIC_MAXIMUM_ANGLE} from './spatial-geodesic-wgsl';
import {
  getSpatialPredicateWGSL,
  type SpatialDistanceMetric,
  type SpatialDistanceParameter,
  type SpatialLegacyPredicateName,
  type SpatialPredicateName,
  type SpatialPredicateSide
} from './spatial-predicate-wgsl';
import {
  doesRelatePatternAdmitDisjoint,
  getPredicateRelatePatterns,
  getRelateMatchesParametersWGSL,
  getRelateMatchesWGSL,
  getRelatePatterns,
  GPU_SPATIAL_RELATE_PATTERN_WORDS,
  getSpatialKindDimension,
  GPU_SPATIAL_RELATE_UNCERTAIN_BIT,
  packGPUSpatialRelate,
  type GPUSpatialRelatePattern
} from './spatial-relate-types';
import {
  getSpatialDwithinWorkgroupWGSL,
  SPATIAL_DWITHIN_WORKGROUP_SIZE
} from './spatial-dwithin-wgsl';
import {
  getSpatialRelateWGSL,
  getSpatialRelateWorkgroupWGSL,
  SPATIAL_RELATE_WORKGROUP_SIZE
} from './spatial-relate-wgsl';
import {type GPUSpatialJoinGeometry, type GPUSpatialJoinOnAttribute} from './spatial-join-types';
import type {GPUCompactPairPort} from '../../analysis-core/contracts/index';
import {
  resolveSpatialPredicateCandidateCapacity,
  getSpatialPredicateCandidatePagePlan,
  getSpatialPredicateScratchByteLengths
} from './spatial-predicate-capacity-plan';
import {
  addSpatialPredicateUnmatchedNodes,
  type SpatialPredicateUnmatchedContext
} from './spatial-predicate-unmatched';
import {
  getGPUSpatialJoinCapacityPlan,
  type GPUSpatialJoinCapacityPlan
} from './spatial-join-capacity-plan';
import {validateGPUCompactPairPort} from '../../analysis-core/contracts/index';

import {Buffer} from '@luma.gl/core';
import {
  SPATIAL_JOIN_MAXIMUM_INDEX,
  SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT,
  planSpatialJoinCandidatePages
} from './spatial-join-candidate-pages';
import {
  validateGraphChunkGroupKernelSupport,
  createGraphChunkGroupKernelNode
} from '../../analysis-core/kernels/graph-chunk-groups';
import {SCAN_TOTAL_WGSL} from '../../analysis-core/kernels/scan-total';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotJoinGeometry,
  snapshotJoinPairs,
  snapshotJoinStatus,
  snapshotJoinWeights
} from './spatial-join-construction-snapshot';

const OPERATION = 'GPUSpatialPredicateJoin';
// Diagnostic contributor IDs may be reused by replacements before either producer is added.
let predicateOutputAllocation = 0;

import {
  createSpatialJoinLiveInputNodes,
  createSpatialJoinLiveTruncationNode,
  createSpatialJoinLiveRingOffsets,
  validateSpatialJoinLeftInput,
  type GPUSpatialJoinLeftStatus
} from './spatial-join-live-input';
export type {GPUSpatialJoinLeftStatus} from './spatial-join-live-input';
import {
  getGPUSpatialPredicateJoinExecutionPlan,
  validateSpatialPredicateRelateOutput
} from './spatial-predicate-route-plan';
export {
  SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT,
  getGPUSpatialPredicateJoinExecutionPlan,
  GPU_SPATIAL_PREDICATE_JOIN_PLANNER,
  GPU_SPATIAL_PREDICATE_JOIN_ROUTES
} from './spatial-predicate-route-plan';
export type {GPUSpatialPredicateJoinRouteShape} from './spatial-predicate-route-plan';

/** Numeric distances are emitted as f32 WGSL literals, which cannot represent larger finite values. */
const GPU_SPATIAL_PREDICATE_JOIN_MAXIMUM_DISTANCE = 3.4028234663852886e38;

/** Bits of a matrix word copied to the `relate` output: the 18 matrix bits and the uncertain bit. */
const RELATE_OUTPUT_MASK = (0x3ffff | GPU_SPATIAL_RELATE_UNCERTAIN_BIT) >>> 0;
/** Storage buffers per shader stage on the default device limits. */
const MAXIMUM_STORAGE_BUFFERS = 8;

/**
 * Spatial contexts of {@link GPUSpatialPredicateJoin}: planar coordinates, or longitude/latitude
 * degrees. On longitude/latitude, `dwithin` takes `metric: 'great-circle'` (meters); every other
 * predicate is topological and takes `metric: 'none'`.
 */
const SPATIAL_PREDICATE_JOIN_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: ['planar', 'longitude-latitude'],
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  metrics: ['none', 'native', 'great-circle']
} as const);

/** Distance predicates additionally require a frame suitable for distance measurements. */
const SPATIAL_DISTANCE_PREDICATE_JOIN_CONTEXT_SUPPORT = Object.freeze({
  ...SPATIAL_PREDICATE_JOIN_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/**
 * Predicates supported by {@link GPUSpatialPredicateJoin}: `intersects`, `contains`, `within`,
 * `dwithin`, `covers`, `coveredBy`, `touches`, `crosses`, `overlaps`, `equals`,
 * `containsProperly`, and `relate` (a user DE-9IM `pattern`).
 */
export type GPUSpatialPredicate = SpatialPredicateName;

/** `how` of {@link GPUSpatialPredicateJoinProps}: matched pairs, or left rows without a match. */
export type GPUSpatialJoinHow = 'inner' | 'anti';

const LEGACY_PREDICATES: readonly SpatialLegacyPredicateName[] = [
  'intersects',
  'contains',
  'within',
  'dwithin'
];
const RELATE_PREDICATES: readonly SpatialPredicateName[] = [
  'covers',
  'coveredBy',
  'touches',
  'crosses',
  'overlaps',
  'equals',
  'containsProperly',
  'relate'
];

/** Whether a `distance` or `pattern` prop is a graph view (per-frame) rather than a plain value. */
function isGraphView<View extends GraphDataView>(value: unknown): value is View {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Caller-owned outputs of {@link GPUSpatialPredicateJoin}. */
export type GPUSpatialPredicateJoinOutput = {
  /**
   * Optional per-left-row flag, one `uint32` per left feature, 1 when that row lost candidates
   * because the candidate total exceeded `candidateCapacity`. Candidates are laid out in left-row
   * order and the slots past the capacity are dropped, so the flagged rows are the row whose
   * candidates cross the capacity (it keeps its first ones) and every later row that had any
   * candidate (it keeps none); earlier rows are complete. A BVH overflow (more right features than
   * `leafCapacity`) drops right features that any row may match, so it flags every left row. See
   * `status.candidateOverflow`.
   */
  candidateTruncated?: GraphDataView<'uint32'>;
  /** Compact left rows without a match, ascending. Required for `how: 'anti'`; optional with inner pairs/weights. Computed before final pair truncation; unknown rows are excluded. Any uncertified classification conservatively withholds every unmatched row; request uncertainCount to distinguish this case. */
  unmatched?: GPUCompactOutput;
  /** Matched pairs sorted by `(left, right)`. With `how: 'inner'`, at least one of `pairs` and `weights` is required. */
  pairs?: GPUCompactPairPort;
  /**
   * Optional packed DE-9IM matrix of every matched pair, aligned with `pairs` (slot `k` belongs to
   * `(pairs.leftRows[k], pairs.rightRows[k])`). Requires `pairs` and the same length as
   * `pairs.leftRows`. Layout: see {@link packGPUSpatialRelate}; decode with
   * {@link formatGPUSpatialRelate}. Bit 31 ({@link GPU_SPATIAL_RELATE_UNCERTAIN_BIT}) of a value is
   * set when an orientation of that pair had no sign (see `uncertainCount`); {@link formatGPUSpatialRelate}
   * ignores it, mask with `value & 0x3ffff` to compare matrices. Not available for `'dwithin'`.
   * Setting it makes every predicate evaluate through the relate engine.
   */
  relate?: GraphDataView<'uint32'>;
  /**
   * Matches as cross spatial weights: one CSR row per left feature, neighbors are right rows
   * ascending, every weight is `1`. `weights.neighbors.length` is the pair capacity and must equal
   * `pairs.leftRows.length` when both are given. Optional `weights.distances` stores the minimum
   * closed-geometry distance aligned with each neighbor (zero for intersecting pairs): coordinate
   * units in a planar context, meters for great-circle `dwithin`. Unused slots are NaN.
   * Longitude/latitude with `metric: 'none'` cannot produce distances.
   * Pass the contributor's outputWeights to consumers: it always carries overflow and
   * candidateOverflow, allocating graph-owned scalar views when the caller omitted them.
   * The optional status map accepts only count, requiredCount, overflow, candidateOverflow and
   * uncertainCount. Other captured bounded-result and join-status diagnostics are rejected at
   * construction; unrelated metadata keys are omitted from the captured status.
   */
  weights?: Omit<GPUSpatialWeights, 'status'> & {
    status?: Pick<
      NonNullable<GPUSpatialWeights['status']>,
      'count' | 'requiredCount' | 'overflow' | 'candidateOverflow' | 'uncertainCount'
    >;
  };
  /**
   * Completeness scalars. Optional `antimeridianUncertainCount` counts unprepared crossing rings;
   * optional `poleUncertainCount` counts unprepared pole-enclosing rings. Prepare these polygons
   * with `GPULonLatPolygonPreparation`. Both fields are left untouched in planar contexts.
   */
  status?: GPULonLatPolygonStatusPort &
    GPUCandidateStatusPort & {
      /** One-row flag: 1 when the BVH, candidate, or pair capacity overflowed. Required for `'inner'` without `pairs`. */
      overflow?: GraphDataView<'uint32'>;
      /**
       * Optional one-row count of left rows that lost candidates to `candidateCapacity` or to a
       * BVH overflow (the rows flagged in `output.candidateTruncated`; every left row after a BVH
       * overflow), restricted to leftFeatureCount when supplied. Nonzero implies candidate loss;
       * aggregate loss may still be set for an empty live prefix. With `how: 'anti'` these rows
       * are unknown, so they are kept out of `output.unmatched`.
       */
      candidateTruncatedCount?: GraphDataView<'uint32'>;
      /** Optional one-row unclamped candidate count, for sizing `candidateCapacity`. */
      candidateCount?: GraphDataView<'uint32'>;
      /**
       * Optional one-row count of candidates whose classification could not be certified. A candidate
       * is counted here when
       * - the robust point/polygon classifier returned `uncertain` (point/polygon pairs), or
       * - on any other kind pair, in either engine, an orientation sign does not exist: a non-finite
       *   coordinate, or product exponents spanning more than 200 bits. Every other orientation is
       *   decided exactly (f32 filter, then exact integer arithmetic).
       *
       * An uncertain pair is never dropped or decided at random. The undecidable orientation is treated
       * as collinear (sign 0) and the predicate is evaluated with that value; for point/polygon pairs
       * the f32 relate/predicate kernel decides. So an uncertain pair can be wrong in either
       * direction, and the count says how many results to distrust. Zero means every decision is
       * certified. The pairs themselves are identifiable through the `relate` output: bit 31
       * ({@link GPU_SPATIAL_RELATE_UNCERTAIN_BIT}) of a matched pair's value is set when that pair was
       * uncertain (a matched pair only; an uncertain pair that did not match is counted but has no
       * row). Eight-binding scalar kernels retain the diagnostic in candidate flags and count it
       * in a subsequent pass, including cooperative `dwithin` (`engine: 'relate'`).
       */
      uncertainCount?: GraphDataView<'uint32'>;
    };
};

/**
 * Properties for {@link GPUSpatialPredicateJoin}.
 *
 * Per-frame: the contents of every input buffer, plus `distance` and `pattern` when they are given
 * as views. Compile-time: view lengths, `predicate` (it selects the kernel, and for the named
 * predicates the DE-9IM masks and the geometry-kind dimension logic folded into them), `distance`
 * and `pattern` when they are plain values, `how`, `candidateCapacity`, `leafCapacity`,
 * `excludeSameRow`, `prepared`, and which optional views exist.
 */
export type GPUSpatialPredicateJoinProps<Parameters = unknown> = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-predicate-join'`. */
  id?: string;
  /** Left features. Output rows (and CSR rows of `weights`) are left features. */
  left: GPUSpatialJoinGeometry;
  /** Resident live left feature count, bounded by the left geometry capacity. Spare slots are ignored. */
  leftFeatureCount?: GraphDataView<'uint32'>;
  /** Upstream loss propagates as candidate loss; unknown rows are never reported unmatched. */
  leftStatus?: GPUSpatialJoinLeftStatus;
  /** Right features, indexed by a BVH. Prefer the side with more features or larger extent. */
  right: GPUSpatialJoinGeometry;
  /**
   * Predicate evaluated as `predicate(left, right)`, following OGC Simple Features and the
   * DE-9IM matrix (Shapely and GEOS semantics). `within(a, b)` is exactly `contains(b, a)`. See
   * {@link GPUSpatialPredicateJoin} for the boundary rules and the list of predicates.
   */
  predicate: GPUSpatialPredicate;
  /**
   * Compile-time. Which kernel evaluates `intersects`, `contains`, `within` and `dwithin` (every
   * other predicate always uses the relate engine). `'fast'` is the short-circuiting kernel, one
   * invocation per candidate, which is quickest for small features. `'relate'` is the DE-9IM
   * engine for `intersects`, `contains` and `within`, and a workgroup kernel for `dwithin`
   * (which has no matrix): one workgroup per candidate with bounding-box pruning of the edge work.
   * Both are 3 to 15 times faster from about 32 vertices per feature (measured on polygons with
   * 200 to 1000 vertices). `'auto'` (default) picks `'relate'`
   * when neither side is points and the product of the average vertex counts per feature of the
   * two sides is at least {@link SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT}. Both engines
   * use the same exact orientation signs (see {@link GPUSpatialPredicateJoin}), so the result of a
   * predicate does not depend on the engine or on the feature size, including shared-edge and
   * collinear configurations; only the cost differs.
   * Requesting the `relate` output always uses the relate engine. A great-circle `dwithin` always
   * uses the fast kernel (`'relate'` throws for it, `'auto'` picks `'fast'`).
   */
  engine?: 'auto' | 'fast' | 'relate';
  /**
   * Coordinate space and metric of both sides. Omitted means native planar coordinates.
   *
   * - Planar (default): all predicates work in the plane of the coordinates; `dwithin` distance is
   *   in coordinate units.
   * - `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`, `dwithin` only: positions
   *   are longitude/latitude degrees and `distance` is a great-circle distance in meters on the mean
   *   Earth sphere (`sphereRadius` of the context overrides the radius; `units: 'kilometers'` is
   *   rejected). The candidate stage widens each left box by the angular radius in latitude and by
   *   the matching longitude reach (all longitudes when the radius reaches a pole, plus a second
   *   probe when the box crosses the antimeridian), and the exact test measures the central angle
   *   between vertices and segments (straight in longitude/latitude, with longitude differences
   *   wrapped), so a pair across the antimeridian or near a pole is measured the short way round. Pairs whose geometries intersect
   *   in the degree plane count as distance 0. Distances up to about 8000 km are supported (a larger
   *   compile-time value throws, a larger per-frame value selects nothing). Edges are straight
   *   lines in longitude/latitude for distance, intersection and containment alike (the same edge
   *   model as the network snapping contributors); the point-to-segment foot is found in the local
   *   equirectangular frame of the point, accurate for edges of road length, so densify long edges
   *   first.
   * - `{coordinateSpace: 'longitude-latitude', metric: 'none'}`, every predicate except `dwithin`:
   *   topology in the degree plane, as GEOS and PostGIS `geometry` do for coordinates in degrees.
   *   There is no metric, so no distances are involved. `dwithin` with this metric throws.
   *
   * Prepare polygons that cross ±180 or enclose a pole with `GPULonLatPolygonPreparation`;
   * optional status counters report unprepared rings on both polygon sides. Longitude/latitude
   * requires `edgeModel: 'linear'` (the default); great-circle edges require preparation.
   *
   * Any other combination throws at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `right` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  rightSpatialContext?: GPUSpatialContext;
  /**
   * Distance for `'dwithin'` (inclusive). Required for it, ignored otherwise. Planar distance in
   * coordinate units, or meters with a great-circle `spatialContext`.
   *
   * A number is compile-time; short great-circle distances warn to use `createGPULocalFrame`
   * because of float32 longitude/latitude precision. A one-row float32 view makes the distance per-frame: it is read on
   * every encoding by the candidate probe (box expansion), the exact kernel and the workgroup
   * kernel, so moving a slider is a buffer write with no recompile and the results are identical to
   * the same number given at compile time. A negative, NaN or infinite value selects nothing.
   * `candidateCapacity` must cover the largest distance written, since the candidate count grows
   * with it.
   */
  distance?: number | GraphDataView<'float32'>;
  /**
   * DE-9IM pattern (or any-of list of patterns) for `predicate: 'relate'`, in the OGC text form:
   * nine characters from `T`, `F`, `*`, `0`, `1`, `2`. Required for `'relate'` and rejected for any
   * other predicate. Only bounding-box candidates are enumerated, so a pattern that admits
   * disjoint geometries (all of II, IB, BI and BB allow `F`) is rejected; use `how: 'anti'` for those.
   *
   * A string or list is compile-time. A `uint32` view makes the pattern per-frame: fill it with
   * {@link packGPUSpatialRelatePattern} (two words per pattern slot, so its length fixes the most
   * patterns an any-of list may hold and must be a positive multiple of
   * {@link GPU_SPATIAL_RELATE_PATTERN_WORDS}). The mask kernel reads it on every encoding, so a
   * change is a buffer write with no recompile; unused slots are zero and match nothing. The
   * packing helper does the pattern validation (including the disjoint check) on the CPU.
   *
   * `predicate` itself stays compile-time: it picks the kernel family (the short-circuiting
   * `intersects`, `contains`, `within` and `dwithin` kernels, the workgroup `dwithin`, or the
   * relate engine) and, for the named predicates, patterns that depend on the geometry kinds, so a
   * per-frame predicate would force the slowest path on every join. For a per-frame choice among
   * named predicates use `predicate: 'relate'` with their patterns.
   */
  pattern?: GPUSpatialRelatePattern | GraphDataView<'uint32'>;
  /**
   * `'inner'` (default) emits matched pairs. `'anti'` emits the left features that match no right
   * feature (the complement of `intersects`, and of any other predicate, for example `disjoint` as
   * an anti-join of `intersects`) into `output.unmatched`; `pairs`, `weights` and `relate` are then
   * not available. If the candidate capacity overflows, the left rows whose candidates were dropped
   * (`output.candidateTruncated`) cannot be classified: they are kept out of `output.unmatched`
   * and counted in `status.candidateTruncatedCount`, so the output never reports a row that may
   * have a match (it can omit true unmatched rows among them). A BVH overflow (more right features
   * than `leafCapacity`) marks every left row, so the anti join then emits no rows; before
   * 2026-11-10 it listed rows whose only matches were among the dropped right features. If the
   * unmatched capacity overflows, check `output.unmatched.overflow`.
   */
  how?: GPUSpatialJoinHow;
  /**
   * Skip pairs with equal left and right rows. Set it for a self-join (`left` and `right` are the
   * same features) so each feature is not matched with itself, as `GPUSpatialWeights` requires.
   * Defaults to `false`.
   */
  excludeSameRow?: boolean;
  /**
   * Attribute-equality condition (GeoPandas `sjoin(on_attribute=...)`): a candidate pair survives
   * only when `onAttribute.left[leftRow] === onAttribute.right[rightRow]`, then goes through the
   * predicate as usual. Both views are `uint32` with one key per feature of their side; their
   * contents are per-frame, their presence is compile-time. Candidates failing the key test are
   * dropped before the exact kernels, so `candidateCount` still reports the pre-filter bounding-box
   * count and `candidateCapacity` must cover it. With `how: 'anti'` a left row is unmatched when no
   * key-equal right row satisfies the predicate.
   */
  onAttribute?: GPUSpatialJoinOnAttribute;
  /**
   * Maximum `(left, right)` bounding-box candidates evaluated per encoding. Defaults to the
   * {@link getGPUSpatialJoinCapacityPlan} estimate, `min(rightCount, 8)` per left feature. In a
   * grow loop, pass the plan's `candidates.capacity` computed from the observed `candidateCount`.
   */
  candidateCapacity?: number;
  /** Candidate scratch page; defaults to the device envelope. Does not change the global cutoff. */
  candidateTileCapacity?: number;
  /** Power-of-two BVH leaf slots over `right`. Defaults to the next power of two of its feature count. */
  leafCapacity?: number;
  /**
   * A prepared (static) right-hand side over the same views as `right`: its retained geometry
   * changes only when invalidated. Public BVH builds replay on that snapshot each encoding.
   * Exact refinement reads the same retained coordinates and offsets as the bounds. Source IDs
   * remain live caller data. The handle
   * must be added to the graph before the join. Spatially sorted candidates are restored to source
   * row order before refinement. Its `leafCapacity` wins over this join's.
   */
  prepared?: GPUSpatialJoinPrepared<Parameters>;
  /** Caller-owned outputs. */
  output: GPUSpatialPredicateJoinOutput;
};

/** Read-back counts of one {@link GPUSpatialPredicateJoin} run, for {@link GPUSpatialPredicateJoin.getCapacityPlan}. */
export type GPUSpatialPredicateJoinObservation = {
  /** `output.status.candidateCount`: the unclamped bounding-box candidate count. */
  candidateCount?: number;
  /** `output.pairs.status.requiredCount`: the unclamped match count (a lower bound while candidates overflow). */
  requiredCount?: number;
};

/** Flat internal view of the props that the node builders read. */
type SpatialPredicateJoinViews = Omit<GPUSpatialPredicateJoinProps<never>, 'output'> &
  Omit<GPUSpatialPredicateJoinOutput, 'status'> &
  NonNullable<GPUSpatialPredicateJoinOutput['status']>;

/**
 * Joins two sets of features by a spatial predicate: the GPU equivalent of a GeoPandas
 * `sjoin` or a PostGIS `JOIN ... ON ST_Intersects(l, r)`. Coordinates are planar by default; a
 * longitude/latitude `spatialContext` adds great-circle `dwithin` in meters and planar-in-degrees
 * topology for the other predicates (see `spatialContext`).
 *
 * Candidate scratch streams through device-sized pages; a single query may span pages. The GPU
 * finds each page's intersecting query interval and dispatches only that interval. Page bases and
 * query ranges are runtime values. Matches append at a saturating running offset. Global candidate
 * and output cutoffs, statuses, and left-row/right-row output order match the packed route.
 *
 * Pipeline: per-feature bounds, a `GPUBVH` over the right bounds (or a reused `prepared` one), a
 * counted and scanned bounding box probe from every left feature (candidates come out ordered by
 * left row and then right row), an exact predicate test per candidate, and a scan-based stable
 * compaction into the caller's capacity-bounded output. Nothing is read back; `output.status.overflow` reports
 * when a capacity was exceeded, in which case the output holds a prefix-consistent subset
 * (candidates past `candidateCapacity` and matches past the pair capacity are dropped).
 *
 * Geometry kinds on either side: points, linestrings and polygons or multipolygons with holes (see
 * {@link GPUSpatialJoinGeometry}); all nine combinations are supported.
 *
 * **Semantics (OGC / DE-9IM).** A point's interior is the point and its boundary is empty. A
 * linestring's boundary is its two endpoints, unless it is closed, and the rest is interior. A
 * polygon's boundary is its rings and its interior is the area between them (holes are exterior).
 * - `intersects(a, b)`: the closed geometries share at least one point. A point on a polygon
 *   boundary intersects the polygon; polygons that only touch along an edge or at a corner intersect.
 * - `contains(a, b)`: every point of `b` lies in the closure of `a` and the interiors intersect.
 *   A point on the boundary of a polygon is not contained by it; a linestring lying entirely on a
 *   polygon boundary is not contained; a polygon contains itself; a linestring does not contain its
 *   own endpoints (unless closed); a polygon contains a polygon that touches its boundary from
 *   inside; a point contains only an equal point; a lower-dimension geometry never contains a
 *   higher-dimension one.
 * - `within(a, b)`: `contains(b, a)`.
 * - `dwithin(a, b)`: the minimum distance between the closed geometries is at most `distance`:
 *   planar in coordinate units, or great-circle in meters with a longitude/latitude
 *   `spatialContext` (see {@link GPUSpatialPredicateJoinProps.spatialContext}). Zero distance counts,
 *   so it includes everything that intersects.
 * - `covers` / `coveredBy`: like `contains` / `within`, but boundary contact is allowed (a polygon
 *   covers a point on its boundary and a line lying on its boundary).
 * - `touches`: the geometries share boundary points but no interior points (never true for two
 *   points).
 * - `crosses`: the interiors intersect and the intersection has a lower dimension than the larger
 *   geometry (line/line crossing at points, line through a polygon); never true for point/point,
 *   point/point or polygon/polygon.
 * - `overlaps`: same dimension, interiors intersect, and neither contains the other.
 * - `equals`: the same point set, however the vertices are ordered or repeated.
 * - `containsProperly`: `b` lies in the interior of `a` (no boundary contact).
 * - `relate` with `pattern`: any DE-9IM pattern.
 *
 * **One engine.** All of the above are masks over one per-pair DE-9IM classification (the
 * "relate" matrix), as in GEOS RelateNG and SedonaDB. `intersects`, `contains`, `within` and
 * `dwithin` keep their short-circuiting kernels (an `intersects` test stops at the first shared
 * point, where the matrix needs every part located) and are routed through the matrix only when
 * `relate` output is requested; the test suite checks both paths agree. `covers`, `coveredBy`,
 * `touches`, `crosses`, `overlaps`, `equals`, `containsProperly` and `relate` use the matrix.
 * Multipart `coveredBy` retains its fast coverage route unless the matrix route is requested.
 *
 * Multipart point/line operands support every predicate and the relate output. A feature has one
 * matrix: line endpoints use the mod-2 boundary and exterior cells test coverage by the union of
 * all parts, using exact supplied-coordinate interval merging for line coverage. Polygon
 * featureOffsets already group all polygon rings and use the same whole-feature classification.
 *
 * Single-point/polygon pairs use the robust double-single classifier `GPUPairwisePointInPolygon` for the
 * point location in every predicate, except resident live-left polygon prefixes: those may retain
 * unused allocated tails and use the feature-bounded exact kernels instead of compact global
 * offset validation. Every other combination uses exact orientation signs in both
 * the short-circuiting kernels and the relate engine (`orientSign`, an f32 filter with an exact
 * 256-bit integer fallback, one inlined call per kernel), so `engine: 'fast'`, `'relate'` and the
 * `'auto'` choice agree on shared-edge and collinear configurations. The exactness is with respect
 * to the f32 vertices; distances and the piece midpoints used by `contains` on lines and polygons
 * are f32. Pairs with no sign (non-finite input, exponent span over 200 bits) are counted in
 * `uncertainCount` and treated as collinear; see `uncertainCount`. Coordinates must be finite; empty or invalid features never match (their matrix is all `F`).
 * Polygon-in-polygon containment assumes valid polygons; a polygon that coincides with a hole is
 * handled by probing both sides of shared edges.
 *
 * **Relate output.** `output.relate` writes the packed 9-cell matrix of every matched pair next to
 * `pairs`. Only bounding-box candidates are enumerated, so pairs that do not intersect never appear.
 *
 * **Anti joins.** `how: 'anti'` emits the left rows with no match into `output.unmatched`.
 * Supply `output.unmatched.status.candidateOverflow` unless `candidateCapacity` covers the product
 * of allocated left and right feature counts, the BVH has at least one leaf slot per right feature,
 * and there are no live left count or completeness views.
 *
 * **Static right-hand side.** Pass a {@link GPUSpatialJoinPrepared} handle as `prepared` to reuse the
 * right-hand BVH storage across encodings. Bounds and exact refinement read the same retained
 * positions and full offset hierarchy until an invalidated generation executes. Source-ID mapping
 * remains live, and left/query inputs are evaluated for every encoding.
 *
 * Weights output: a join is a cross spatial-weights matrix (rows are left features, neighbors are
 * right rows, ascending), so `output.weights` writes the CSR directly from the sorted pairs. For a
 * self-join pass `excludeSameRow` to satisfy the no-self-neighbor invariant.
 *
 * **Capacity grow loop.** Pass `output.status.candidateCount` and `output.pairs.status.requiredCount`,
 * read them after a frame and call {@link GPUSpatialPredicateJoin.getCapacityPlan}: when
 * `output.status.overflow` is 1, rebuild with `candidateCapacity: plan.candidates.capacity` and
 * `plan.pairs.capacity` pair slots. Grow the candidate stage first: while it overflows, the
 * required pair count only covers the candidates that fit.
 */
export class GPUSpatialPredicateJoin<FrameParameters = unknown> implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialPredicateJoinProps<FrameParameters>;
  /** Compact pairs with candidate loss attached, including private or prepared BVH loss. */
  readonly outputPairs?: GPUCompactPairPort & {
    status: GPUCompactPairPort['status'] & {candidateOverflow: GraphDataView<'uint32'>};
  };
  /** CSR result with both completeness signals attached. Pass this port to statistics. */
  readonly outputWeights?: Omit<GPUSpatialWeights, 'status'> & {
    status: NonNullable<NonNullable<GPUSpatialPredicateJoinOutput['weights']>['status']> & {
      overflow: GraphDataView<'uint32'>;
      candidateOverflow: GraphDataView<'uint32'>;
    };
  };
  /** Flat view of `props` read by the node builders. */
  private readonly views: SpatialPredicateJoinViews;
  /** Scalar publication order, including graph-owned weights diagnostics. */
  private readonly scalarOutputs: readonly [string, GraphDataView<'uint32'>][];
  /** Number of left features. */
  readonly leftCount: number;
  /** Number of right features. */
  readonly rightCount: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Bounded candidate scratch shared by ordered pages. */
  readonly candidatePages: ReturnType<typeof planSpatialJoinCandidatePages>;
  private readonly usesRobustPointPolygon: boolean;
  /** Conservative page and primitive command budget. */
  readonly estimatedCommandNodeCount: number;
  /** Matched pair capacity. Zero for `how: 'anti'`. */
  readonly pairCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Join mode. */
  readonly how: GPUSpatialJoinHow;
  /** Whether candidates are classified by the DE-9IM relate engine. */
  readonly usesRelateEngine: boolean;
  /** Whether `dwithin` is evaluated by the workgroup kernel (one workgroup per candidate). */
  readonly usesWorkgroupDistance: boolean;
  /** Resolved spatial context (native planar when none was given). */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in meters of a great-circle `dwithin`, or undefined for planar distances. */
  readonly sphereRadius: number | undefined;

  /** DE-9IM patterns that a candidate's matrix must match; empty when the engine is not used or the pattern is per-frame. */
  private readonly relatePatterns: readonly string[];
  /** Per-frame pattern words, or undefined when the pattern is compile-time. */
  private readonly patternView: GraphDataView<'uint32'> | undefined;
  /** Per-frame `dwithin` distance row, or undefined when the distance is compile-time. */
  private readonly distanceView: GraphDataView<'float32'> | undefined;
  private readonly graphOwnedStatusCount: number;

  constructor(props: GPUSpatialPredicateJoinProps<FrameParameters>) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'left',
      'leftFeatureCount',
      'leftStatus',
      'right',
      'predicate',
      'engine',
      'spatialContext',
      'rightSpatialContext',
      'distance',
      'pattern',
      'how',
      'excludeSameRow',
      'onAttribute',
      'candidateCapacity',
      'candidateTileCapacity',
      'leafCapacity',
      'prepared',
      'output'
    ]);
    const output = snapshotConstructionFields(snapshot.output, [
      'candidateTruncated',
      'unmatched',
      'pairs',
      'relate',
      'weights',
      'status'
    ]);
    const unmatched =
      output.unmatched && snapshotConstructionFields(output.unmatched, ['ids', 'status']);
    const weights = output.weights && snapshotJoinWeights(output.weights);
    props = Object.freeze({
      ...snapshot,
      left: snapshotJoinGeometry(snapshot.left),
      ...(snapshot.leftStatus
        ? {
            leftStatus: snapshotConstructionFields(snapshot.leftStatus, [
              'overflow',
              'candidateOverflow',
              'complete'
            ])
          }
        : {}),
      right: snapshotJoinGeometry(snapshot.right),
      ...(snapshot.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
        : {}),
      ...(snapshot.rightSpatialContext
        ? {rightSpatialContext: snapshotConstructionSpatialContext(snapshot.rightSpatialContext)}
        : {}),
      ...(snapshot.onAttribute
        ? {onAttribute: snapshotConstructionFields(snapshot.onAttribute, ['left', 'right'])}
        : {}),
      ...(Array.isArray(snapshot.pattern) ? {pattern: Object.freeze([...snapshot.pattern])} : {}),
      output: Object.freeze({
        ...output,
        ...(output.pairs ? {pairs: snapshotJoinPairs(output.pairs)} : {}),
        ...(unmatched
          ? {unmatched: Object.freeze({...unmatched, status: snapshotJoinStatus(unmatched.status)})}
          : {}),
        ...(weights ? {weights} : {}),
        ...(output.status ? {status: snapshotJoinStatus(output.status)} : {})
      })
    });
    this.id = props.id ?? 'spatial-predicate-join';
    this.props = props;
    const {output: resolvedOutput, ...inputs} = props;
    const views: SpatialPredicateJoinViews = {
      ...inputs,
      ...resolvedOutput,
      ...resolvedOutput.status
    };
    this.views = views;
    const {id} = this;
    const weightsStatus = views.weights?.status as GPUSpatialWeights['status'];
    for (const [name, destination] of Object.entries(weightsStatus ?? {})) {
      validateRequirement(
        id,
        destination === undefined ||
          ['count', 'requiredCount', 'overflow', 'candidateOverflow', 'uncertainCount'].includes(
            name
          ),
        `output.weights.status.${name} is unsupported because this join publishes only count, requiredCount, overflow, candidateOverflow and uncertainCount; omit ${name} and use those result diagnostics`
      );
    }
    const multipart = isMultipartJoinGeometry(views.left) || isMultipartJoinGeometry(views.right);
    validateSpatialJoinGeometry(id, 'left', views.left, {allowMultipart: true});
    validateSpatialJoinGeometry(id, 'right', views.right, {allowMultipart: true});
    this.leftCount = getSpatialJoinFeatureCount(views.left);
    this.rightCount = getSpatialJoinFeatureCount(views.right);
    validateSpatialJoinLeftInput(id, views.leftFeatureCount, views.leftStatus);
    const {predicate} = views;
    const isLegacy =
      (LEGACY_PREDICATES as readonly string[]).includes(predicate) ||
      (multipart && predicate === 'coveredBy');
    this.distanceView =
      predicate === 'dwithin' && isGraphView(views.distance) ? views.distance : undefined;
    this.patternView = isGraphView(views.pattern) ? views.pattern : undefined;
    if (predicate === 'dwithin') {
      if (this.distanceView) {
        validateSinglePackedView(this.distanceView, ['float32'], `${id} distance`);
        if (this.distanceView.length < 1) {
          throw new Error(`${id} distance must contain one float32 row`);
        }
      } else {
        validateRequirement(
          id,
          typeof views.distance === 'number' && views.distance >= 0,
          'distance must be a non-negative number'
        );
        validateFiniteNumber(
          id,
          'distance',
          views.distance as number,
          0,
          GPU_SPATIAL_PREDICATE_JOIN_MAXIMUM_DISTANCE
        );
      }
    } else if (!isLegacy && !(RELATE_PREDICATES as readonly string[]).includes(predicate)) {
      throw new Error(
        `${id} predicate must be ${[...LEGACY_PREDICATES, ...RELATE_PREDICATES].join(', ')}; received ${String(predicate)}`
      );
    }
    const context = resolveGPUSpatialContext(
      id,
      views.spatialContext,
      predicate === 'dwithin' || views.weights?.distances
        ? SPATIAL_DISTANCE_PREDICATE_JOIN_CONTEXT_SUPPORT
        : SPATIAL_PREDICATE_JOIN_CONTEXT_SUPPORT
    );
    this.spatialContext = context;
    resolveGPUSecondarySpatialContext(
      id,
      'right',
      views.rightSpatialContext,
      this.spatialContext,
      SPATIAL_PREDICATE_JOIN_CONTEXT_SUPPORT.edgeModels
    );
    if (context.coordinateSpace === 'longitude-latitude') {
      if (predicate === 'dwithin' && context.metric !== 'great-circle') {
        throw new Error(`${id} longitude-latitude dwithin needs metric 'great-circle'`);
      }
      if (predicate !== 'dwithin' && context.metric !== 'none') {
        throw new Error(`${id} ${predicate} needs metric 'none' on longitude-latitude`);
      }
    }
    if (predicate === 'dwithin' && typeof views.distance === 'number') {
      warnGPUShortLongitudeLatitudeDistance(
        'GPUSpatialPredicateJoin',
        this.spatialContext,
        views.distance
      );
    }
    if (context.metric === 'great-circle') {
      if (context.units === 'kilometers') {
        throw new Error(`${id} great-circle dwithin distances are in meters`);
      }
      this.sphereRadius = getGPUSphereRadius(context);
      validateNonZeroFiniteNumber(id, 'spatialContext.sphereRadius', this.sphereRadius);
      if (
        typeof views.distance === 'number' &&
        views.distance / this.sphereRadius > SPATIAL_GEODESIC_MAXIMUM_ANGLE
      ) {
        throw new Error(
          `${id} distance exceeds ${Math.floor(SPATIAL_GEODESIC_MAXIMUM_ANGLE * this.sphereRadius)} m`
        );
      }
    } else {
      this.sphereRadius = undefined;
    }
    this.how = views.how ?? 'inner';
    if (this.how !== 'inner' && this.how !== 'anti') {
      throw new Error(`${id} how must be inner or anti; received ${String(this.how)}`);
    }
    if (predicate === 'relate') {
      if (views.pattern === undefined) {
        throw new Error(`${id} predicate 'relate' requires a pattern`);
      }
    } else if (views.pattern !== undefined) {
      throw new Error(`${id} pattern requires predicate 'relate'`);
    }
    validateSpatialPredicateRelateOutput(id, predicate, views.relate !== undefined);
    const engine = views.engine ?? 'auto';
    if (engine !== 'auto' && engine !== 'fast' && engine !== 'relate') {
      throw new Error(`${id} engine must be auto, fast or relate; received ${String(engine)}`);
    }
    if (engine === 'fast' && !isLegacy) {
      throw new Error(`${id} engine 'fast' needs intersects, contains, within or dwithin`);
    }
    if (this.sphereRadius !== undefined && engine === 'relate') {
      throw new Error(`${id} engine 'relate' is not available for great-circle dwithin`);
    }
    const route = getGPUSpatialPredicateJoinExecutionPlan(
      {
        predicate,
        engine,
        leftKind: views.left.kind,
        rightKind: views.right.kind,
        leftCount: this.leftCount,
        rightCount: this.rightCount,
        leftVertexCount: views.left.positions.length,
        rightVertexCount: views.right.positions.length,
        multipart,
        outputRelate: views.relate !== undefined,
        greatCircle: this.sphereRadius !== undefined
      },
      id
    );
    this.usesRelateEngine = route.usesRelateEngine;
    this.usesWorkgroupDistance = route.usesWorkgroupDistance;
    if (this.patternView) {
      const words = this.patternView;
      validateSinglePackedView(words, ['uint32'], `${id} pattern`);
      if (words.length < 2 || words.length % GPU_SPATIAL_RELATE_PATTERN_WORDS !== 0) {
        throw new Error(
          `${id} pattern view must hold a positive multiple of ` +
            `${GPU_SPATIAL_RELATE_PATTERN_WORDS} words`
        );
      }
      this.relatePatterns = [];
    } else if (this.usesRelateEngine) {
      const patterns =
        predicate === 'relate'
          ? [...getRelatePatterns(views.pattern as GPUSpatialRelatePattern)]
          : [
              ...getPredicateRelatePatterns(
                predicate as Parameters<typeof getPredicateRelatePatterns>[0],
                getSpatialKindDimension(views.left.kind),
                getSpatialKindDimension(views.right.kind)
              )
            ];
      if (predicate === 'relate' && patterns.length === 0) {
        throw new Error(`${id} pattern must not be empty`);
      }
      if (doesRelatePatternAdmitDisjoint(patterns)) {
        throw new Error(`${id} pattern admits disjoint pairs; use how: 'anti'`);
      }
      this.relatePatterns = patterns;
    } else {
      this.relatePatterns = [];
    }
    this.candidateCapacity = resolveSpatialPredicateCandidateCapacity(
      id,
      this.leftCount,
      this.rightCount,
      views.candidateCapacity
    );
    // Cooperative kernels launch one workgroup per candidate slot, and their invocation index is
    // a u32, so the lane product must stay below 2^32 (the dispatch itself is by live count).
    const isPointPolygonPair =
      !multipart &&
      predicate !== 'dwithin' &&
      // The public pairwise classifier requires compact global terminal coverage. A live
      // polygon prefix can retain unused allocated vertex/ring tails, so classify that route
      // through the existing feature-bounded exact predicate/relate kernels instead.
      !(views.left.kind === 'polygons' && views.leftFeatureCount) &&
      ((views.left.kind === 'points' && views.right.kind === 'polygons') ||
        (views.left.kind === 'polygons' && views.right.kind === 'points'));
    this.usesRobustPointPolygon = isPointPolygonPair;
    const cooperativeLaneCount = this.usesWorkgroupDistance
      ? SPATIAL_DWITHIN_WORKGROUP_SIZE
      : this.usesRelateEngine && !isPointPolygonPair
        ? SPATIAL_RELATE_WORKGROUP_SIZE
        : 1;
    this.leafCapacity =
      views.prepared?.leafCapacity ??
      views.leafCapacity ??
      getNextPowerOfTwo(Math.max(this.rightCount, 1));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    if (views.prepared) {
      if (!isSameSpatialJoinGeometry(views.prepared.geometry, views.right)) {
        throw new Error(`${id} prepared must index the same geometry views as right`);
      }
    }
    if (this.how === 'anti') {
      if (!views.unmatched) {
        throw new Error(`${id} how 'anti' needs output.unmatched`);
      }
      if (views.pairs || views.weights || views.relate) {
        throw new Error(`${id} how 'anti' writes only output.unmatched`);
      }
      validateCompactOutput(id, views.unmatched);
      const maximumCandidateCount = this.leftCount * this.rightCount;
      validateRequirement(
        id,
        Boolean(views.unmatched.status.candidateOverflow) ||
          (this.candidateCapacity >= maximumCandidateCount &&
            this.leafCapacity >= this.rightCount &&
            !views.leftFeatureCount &&
            !views.leftStatus?.overflow &&
            !views.leftStatus?.candidateOverflow &&
            !views.leftStatus?.complete),
        `how 'anti' can lose candidates. Add output.unmatched.status.candidateOverflow unless candidateCapacity (${this.candidateCapacity}) covers all ${maximumCandidateCount} potential feature pairs, leafCapacity (${this.leafCapacity}) covers all ${this.rightCount} right features and there are no live left count or completeness views.`
      );
      this.pairCapacity = 0;
    } else {
      if (views.unmatched) validateCompactOutput(id, views.unmatched);
      if (!views.pairs && !views.weights) {
        throw new Error(`${id} needs output.pairs or output.weights`);
      }
      if (!views.pairs && !views.overflow) {
        throw new Error(`${id} needs output.status.overflow without pairs`);
      }
      if (views.pairs) {
        const {pairs} = views;
        validateGPUCompactPairPort(`${id} output.pairs`, pairs);
        for (const view of [
          pairs.leftRows,
          pairs.rightRows,
          pairs.status.count,
          pairs.status.overflow,
          pairs.status.candidateOverflow,
          pairs.status.requiredCount
        ]) {
          if (view) {
            validateSinglePackedView(view, ['uint32'], `${id} output.pairs`);
          }
        }
        if (pairs.leftRows.length < 1) {
          throw new Error(`${id} output.pairs needs at least one slot`);
        }
      }
      if (views.weights) {
        const rows = validateGPUSpatialWeights(id, views.weights);
        if (rows !== this.leftCount) {
          throw new Error(`${id} output.weights needs one row per left feature`);
        }
        validateRequirement(
          id,
          !views.weights.distances || context.metric !== 'none',
          "output.weights.distances requires spatialContext.metric 'native' or 'great-circle'"
        );
        if (views.pairs && views.pairs.leftRows.length !== views.weights.neighbors.length) {
          throw new Error(`${id} output.weights capacity must equal pairs`);
        }
      }
      this.pairCapacity = views.pairs
        ? views.pairs.leftRows.length
        : (views.weights as GPUSpatialWeights).neighbors.length;
      if (views.relate) {
        if (!views.pairs) {
          throw new Error(`${id} output.relate needs output.pairs`);
        }
        validateSinglePackedView(views.relate, ['uint32'], `${id} output.relate`);
        if (views.relate.length !== this.pairCapacity) {
          throw new Error(`${id} output.relate needs one row per pair`);
        }
      }
    }
    validateStatusScalars(id, props.output.status ?? {});
    if (views.candidateTruncated) {
      validateSinglePackedView(
        views.candidateTruncated,
        ['uint32'],
        `${id} output.candidateTruncated`
      );
      if (views.candidateTruncated.length !== this.leftCount) {
        throw new Error(`${id} output.candidateTruncated needs one row per left feature`);
      }
    }
    if (views.onAttribute) {
      const {left, right} = views.onAttribute;
      validateSinglePackedView(left, ['uint32'], `${id} onAttribute.left`);
      validateSinglePackedView(right, ['uint32'], `${id} onAttribute.right`);
      if (left.length !== this.leftCount || right.length !== this.rightCount) {
        throw new Error(`${id} onAttribute needs one key per feature`);
      }
    }
    validateCapacity(
      id,
      'left offset scratch words',
      this.leftCount + SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT,
      1,
      SPATIAL_JOIN_MAXIMUM_INDEX
    );
    this.candidatePages = getSpatialPredicateCandidatePagePlan(
      (views.left.positions.buffer.graph as GPUCommandGraph).device,
      id,
      this.candidateCapacity,
      props.candidateTileCapacity,
      isPointPolygonPair,
      Boolean(this.distanceView),
      cooperativeLaneCount
    );
    this.estimatedCommandNodeCount = this.candidatePages.estimatedCommandNodeCount;
    const scalarOutputs = this.getScalarOutputs().filter(
      ([name, view]) =>
        view !== undefined ||
        Boolean(
          (views.weights && (name === 'weightsOverflow' || name === 'weightsCandidateOverflow')) ||
            (views.pairs && name === 'pairsCandidateOverflow')
        )
    );
    const scalarGroups = [];
    for (let first = 0; first < scalarOutputs.length; first += 6)
      scalarGroups.push(Object.fromEntries(scalarOutputs.slice(first, first + 6)));
    validateDisjointOutputs(id, this.getInputViews(), this.getOutputViews(), [
      {leftRows: views.pairs?.leftRows, rightRows: views.pairs?.rightRows, relate: views.relate},
      {
        rightRows: views.weights ? views.pairs?.rightRows : undefined,
        neighbors: views.weights?.neighbors,
        weights: views.weights?.weights
      },
      ...scalarGroups,
      ...(this.spatialContext.coordinateSpace === 'longitude-latitude'
        ? [{antimeridian: views.antimeridianUncertainCount, pole: views.poleUncertainCount}]
        : [])
    ]);
    if (views.weights)
      validateSpatialJoinReadWriteBindings(
        id,
        [views.pairs?.rightRows],
        [views.weights.neighbors, views.weights.weights]
      );
    if (views.prepared)
      validateSpatialJoinReadWriteBindings(
        id,
        [
          (views.prepared.isDeclaredIn(views.left.positions.buffer.graph)
            ? views.prepared.storage
            : views.prepared.props.storage
          )?.overflow
        ],
        scalarOutputs.map(([, view]) => view)
      );
    this.graphOwnedStatusCount =
      Number(Boolean(views.pairs && !views.pairs.status.candidateOverflow)) +
      Number(Boolean(views.weights && !views.weights.status?.overflow)) +
      Number(Boolean(views.weights && !views.weights.status?.candidateOverflow));
    if (views.pairs) {
      const graph = views.left.positions.buffer.graph as GPUCommandGraph;
      this.outputPairs = Object.freeze({
        ...views.pairs,
        status: Object.freeze({
          ...views.pairs.status,
          candidateOverflow:
            views.pairs.status.candidateOverflow ??
            createTransientView(
              graph,
              `${id}-pairs-candidate-overflow-${predicateOutputAllocation++}`,
              'uint32',
              1
            )
        })
      });
      views.pairs = this.outputPairs;
    }
    if (views.weights) {
      const graph = views.left.positions.buffer.graph as GPUCommandGraph;
      const weights = views.weights;
      const status = {
        ...(weights.status ? snapshotJoinStatus(weights.status) : {}),
        overflow:
          weights.status?.overflow ??
          createTransientView(
            graph,
            `${id}-weights-overflow-${predicateOutputAllocation++}`,
            'uint32',
            1
          ),
        candidateOverflow:
          weights.status?.candidateOverflow ??
          createTransientView(
            graph,
            `${id}-weights-candidate-overflow-${predicateOutputAllocation++}`,
            'uint32',
            1
          )
      };
      this.outputWeights = Object.freeze({...weights, status: Object.freeze(status)});
      views.weights = this.outputWeights;
    }
    const resolvedScalarOutputs = new Map(this.getScalarOutputs());
    this.scalarOutputs = scalarOutputs.map(([name]) => [name, resolvedScalarOutputs.get(name)!]);
  }

  /** Six scalar outputs share each final publication dispatch. */
  private getScalarOutputs(): [string, GraphDataView<'uint32'> | undefined][] {
    const {pairs, weights, unmatched} = this.views;
    return [
      ['overflow', this.views.overflow],
      ['weightsCount', weights?.status?.count],
      ['weightsTotal', weights?.status?.requiredCount],
      ['weightsOverflow', weights?.status?.overflow],
      ['weightsCandidateOverflow', weights?.status?.candidateOverflow],
      ['weightsUncertainCount', weights?.status?.uncertainCount],
      ['pairsOverflow', pairs?.status.overflow],
      ['pairsCandidateOverflow', pairs?.status.candidateOverflow],
      ['pairsCount', pairs?.status.count],
      ['pairsTotal', pairs?.status.requiredCount],
      ['unmatchedOverflow', unmatched?.status.overflow],
      ['unmatchedCandidateOverflow', unmatched?.status.candidateOverflow],
      ['unmatchedCount', unmatched?.status.count],
      ['unmatchedTotal', unmatched?.status.requiredCount],
      ['candidateCount', this.views.candidateCount],
      ['uncertainCount', this.views.uncertainCount],
      ['candidateTruncatedCount', this.views.candidateTruncatedCount]
    ];
  }

  private getInputViews(): GraphDataView[] {
    return [
      ...getSpatialJoinGeometryViews(this.views.left),
      ...(this.views.leftFeatureCount ? [this.views.leftFeatureCount] : []),
      ...Object.values(this.views.leftStatus ?? {}),
      ...getSpatialJoinGeometryViews(this.views.right),
      ...Object.values(
        this.views.prepared?.isDeclaredIn(this.views.left.positions.buffer.graph)
          ? this.views.prepared.storage
          : (this.views.prepared?.props.storage ?? {})
      ),
      ...(this.distanceView ? [this.distanceView] : []),
      ...(this.patternView ? [this.patternView] : []),
      ...(this.views.onAttribute ? [this.views.onAttribute.left, this.views.onAttribute.right] : [])
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {
      pairs,
      weights,
      overflow,
      candidateCount,
      uncertainCount,
      unmatched,
      relate,
      candidateOverflow,
      candidateTruncatedCount,
      candidateTruncated
    } = this.views;
    return [
      pairs?.leftRows,
      pairs?.rightRows,
      pairs?.status.count,
      pairs?.status.overflow,
      pairs?.status.candidateOverflow,
      pairs?.status.requiredCount,
      relate,
      weights?.offsets,
      weights?.neighbors,
      weights?.weights,
      weights?.distances,
      ...Object.values(weights?.status ?? {}),
      unmatched?.ids,
      unmatched?.status.count,
      unmatched?.status.overflow,
      unmatched?.status.candidateOverflow,
      unmatched?.status.requiredCount,
      overflow,
      candidateCount,
      uncertainCount,
      candidateOverflow,
      candidateTruncatedCount,
      candidateTruncated,
      ...(this.spatialContext.coordinateSpace === 'longitude-latitude'
        ? [this.views.antimeridianUncertainCount, this.views.poleUncertainCount]
        : [])
    ];
  }

  /** Returns bounds, BVH, candidate, exact-test, compaction, and optional CSR nodes in order. */
  /**
   * Plans the next candidate and pair capacities of this join from read-back counts, with the
   * graph-owned bytes this join allocates (see {@link getGPUSpatialJoinCapacityPlan}).
   */
  getCapacityPlan(
    observation: GPUSpatialPredicateJoinObservation = {}
  ): GPUSpatialJoinCapacityPlan {
    const {views} = this;
    const plan = getGPUSpatialJoinCapacityPlan({
      leftCount: this.leftCount,
      rightCount: this.rightCount,
      leftKind: views.left.kind,
      rightKind: views.right.kind,
      relate: this.usesRelateEngine,
      robustPointPolygon: this.usesRobustPointPolygon,
      leafCapacity: this.leafCapacity,
      candidateTileCapacity: this.candidatePages.capacity,
      graphOwnedStatusCount: this.graphOwnedStatusCount,
      observedCandidateCount: observation.candidateCount,
      observedRequiredCount: observation.requiredCount
    });
    if (views.weights?.distances) {
      plan.pairs.estimatedPeakTransientBytes +=
        plan.pairs.capacity * 2 * Uint32Array.BYTES_PER_ELEMENT;
      plan.estimatedPeakTransientBytes += plan.pairs.estimatedPeakTransientBytes;
    }
    return plan;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, views, leftCount, rightCount, candidateCapacity, pairCapacity, how} = this;
    const {left, predicate, pairs, weights, prepared, unmatched, relate} = views;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    if (prepared && !prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} requires prepared to be added to the graph first`);
    }
    const right = prepared
      ? {...prepared.getGeometry(graph), sourceIds: views.right.sourceIds}
      : views.right;
    // All caller bindings (including alignment prefixes) and every derived allocation are
    // checked before bounds, BVH or candidate nodes can mutate the graph. A failure is retryable.
    const boundViews = [
      ...this.getInputViews(),
      ...this.getOutputViews(),
      ...Object.values(prepared?.storage ?? {})
    ];
    const usesRobustPointPolygon = this.usesRobustPointPolygon;
    const tileCapacity = this.candidatePages.capacity;
    const scratchBytes = getSpatialPredicateScratchByteLengths({
      leftCount,
      rightCount,
      tileCapacity,
      pairCapacity,
      leafCapacity: this.leafCapacity,
      prepared: Boolean(prepared),
      hasDistanceView: Boolean(this.distanceView),
      hasWeightsDistances: Boolean(weights?.distances),
      hasWeights: Boolean(weights),
      hasUnmatched: Boolean(unmatched),
      hasCandidateTruncatedCount: Boolean(views.candidateTruncatedCount),
      robustPointPolygon: usesRobustPointPolygon,
      liveGeographicRingOffsetCount:
        left.kind === 'polygons' &&
        views.leftFeatureCount &&
        this.spatialContext.coordinateSpace === 'longitude-latitude' &&
        hasLonLatPolygonStatus(views)
          ? left.ringOffsets.length
          : 0
    });
    validateSpatialJoinBindings(
      graph,
      id,
      Object.fromEntries(boundViews.map((view, index) => [`binding[${index}]`, view])),
      scratchBytes
    );
    validateGraphChunkGroupKernelSupport(graph, id, 8);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      hasLonLatPolygonStatus(views)
    ) {
      for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
        const view = views[name];
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
      for (const [polygonIndex, polygons] of [left, right]
        .filter(geometry => geometry.kind === 'polygons')
        .entries()) {
        let ringOffsets = polygons.ringOffsets;
        if (polygons === left && views.leftFeatureCount) {
          const liveRings = createSpatialJoinLiveRingOffsets(
            graph,
            id,
            polygons,
            views.leftFeatureCount
          );
          nodes.push(liveRings.node);
          ringOffsets = liveRings.ringOffsets;
        }
        nodes.push(
          createLonLatPolygonCheckNode(graph, {
            id: `${id}-lon-lat-polygon-check-${polygonIndex}`,
            operation: OPERATION,
            positions: polygons.positions,
            ringOffsets,
            status: {
              antimeridianUncertainCount: views.antimeridianUncertainCount,
              poleUncertainCount: views.poleUncertainCount
            }
          })
        );
      }
    }

    const isAnti = how === 'anti';

    // Per-side feature-to-ring tables (polygons) and bounds. A prepared right side brings its own.
    const sides = [
      {name: 'left' as const, geometry: left, count: leftCount, isPrepared: false},
      {name: 'right' as const, geometry: right, count: rightCount, isPrepared: Boolean(prepared)}
    ].map(side => {
      let featureRings: GraphDataView<'uint32x2'> | undefined;
      let minima: GraphDataView<'float32x2'> | undefined;
      let maxima: GraphDataView<'float32x2'> | undefined;
      if (side.isPrepared) {
        featureRings = prepared?.storage.featureRings;
        return {...side, minima, maxima, featureRings};
      }
      minima = createTransientView(graph, `${id}-${side.name}-minima`, 'float32x2', side.count);
      maxima = createTransientView(graph, `${id}-${side.name}-maxima`, 'float32x2', side.count);
      if (side.geometry.kind === 'polygons') {
        featureRings = createTransientView(
          graph,
          `${id}-${side.name}-feature-rings`,
          'uint32x2',
          side.count
        );
        nodes.push(
          createFeatureRingsNode<Parameters>(graph, `${id}-${side.name}-rings`, {
            featureCount: side.count,
            geometry: side.geometry,
            featureRings
          })
        );
      }
      nodes.push(
        createBoundsNode<Parameters>(graph, `${id}-${side.name}-bounds`, {
          rejectNonFiniteFeature: true,
          featureCount: side.count,
          geometry: side.geometry,
          featureRings,
          minima,
          maxima
        })
      );
      return {...side, minima, maxima, featureRings};
    });
    const [leftSide, rightSide] = sides;

    let tree: SpatialJoinCandidateTree;
    let bvhOverflow: GraphDataView<'uint32'>;
    if (prepared) {
      tree = {
        nodeMinima: prepared.storage.nodeMinima,
        nodeMaxima: prepared.storage.nodeMaxima,
        leafIds: prepared.storage.leafIds,
        internalNodeCount: prepared.bvhInternalNodeCount,
        spatialSort: prepared.spatialSort
      };
      bvhOverflow = prepared.storage.overflow;
    } else {
      const {bvh, nodes: bvhNodes} = getFeatureBVHNodes(
        graph,
        id,
        rightSide.minima as GraphDataView<'float32x2'>,
        rightSide.maxima as GraphDataView<'float32x2'>,
        this.leafCapacity
      );
      nodes.push(...bvhNodes);
      tree = {
        nodeMinima: bvh.nodeMinima as GraphDataView<'float32x2'>,
        nodeMaxima: bvh.nodeMaxima as GraphDataView<'float32x2'>,
        leafIds: bvh.leafIds,
        internalNodeCount: bvh.internalNodeCount
      };
      bvhOverflow = bvh.overflow;
    }

    const liveInput = createSpatialJoinLiveInputNodes(graph, id, {
      capacity: leftCount,
      count: views.leftFeatureCount,
      status: views.leftStatus,
      minima: leftSide.minima!,
      maxima: leftSide.maxima!,
      bvhOverflow
    });
    nodes.push(...liveInput.nodes);
    bvhOverflow = liveInput.loss;

    // state: [candidate total, uncertain count, match total, unmatched total, truncated rows, live page rows]
    const state = createTransientView(graph, `${id}-state`, 'uint32', 6);
    // Left rows that lost candidates; an anti join needs them to keep unknown rows out of its output.
    const candidateTruncated =
      views.candidateTruncated ??
      (unmatched || views.candidateTruncatedCount
        ? createTransientView(graph, `${id}-candidate-truncated`, 'uint32', leftCount)
        : undefined);
    const pageCounts = createTransientView(graph, `${id}-page-counts`, 'uint32', leftCount);
    const pageOffsets = createTransientView(
      graph,
      `${id}-page-offsets`,
      'uint32',
      leftCount + SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    // A per-frame dwithin distance rides in one extra trailing row of the candidate table (its
    // float32 bits in the first word, written by the clear kernel), because a polygon-polygon exact
    // kernel already binds eight storage buffers, the default device limit.
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      tileCapacity + (this.distanceView ? 1 : 0)
    );
    // Packed matched pairs keep polygon/polygon distance evaluation within eight storage bindings.
    const distancePairs = weights?.distances
      ? createTransientView(graph, `${id}-distance-pairs`, 'uint32x2', pairCapacity)
      : undefined;
    if (distancePairs) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-distance-pairs`,
          operation: OPERATION,
          view: distancePairs,
          type: 'u32',
          value: '0xffffffffu',
          componentCount: pairCapacity * 2
        })
      );
    }
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', tileCapacity + 1);
    const flagOffsets = isAnti
      ? undefined
      : createTransientView(graph, `${id}-flag-offsets`, 'uint32', tileCapacity);
    const leftRows = isAnti
      ? undefined
      : (pairs?.leftRows ?? createTransientView(graph, `${id}-left-ids`, 'uint32', pairCapacity));
    const rightRows = isAnti
      ? undefined
      : (pairs?.rightRows ?? createTransientView(graph, `${id}-right-ids`, 'uint32', pairCapacity));
    const leftMatchCounts =
      weights || unmatched
        ? createTransientView(graph, `${id}-left-match-counts`, 'uint32', leftCount + 1)
        : undefined;
    const matrices = this.usesRelateEngine
      ? createTransientView(graph, `${id}-matrices`, 'uint32', tileCapacity)
      : undefined;

    const pairRowCount = 2 * tileCapacity + 1;
    const polygonSide = left.kind === 'polygons' ? left : right.kind === 'polygons' ? right : null;
    const robust = usesRobustPointPolygon
      ? {
          pairPoints: createTransientView(graph, `${id}-pair-points`, 'float32x2', pairRowCount),
          pairGeometryOffsets: createTransientView(
            graph,
            `${id}-pair-geometry-offsets`,
            'uint32',
            pairRowCount + 1
          ),
          pairClassifications: createTransientView(
            graph,
            `${id}-pair-classifications`,
            'uint32',
            pairRowCount
          )
        }
      : undefined;

    nodes.push(
      createFillNode(graph, {
        id: `${id}-clear-state`,
        operation: OPERATION,
        view: state,
        type: 'u32',
        value: '0u'
      })
    );
    if (leftMatchCounts)
      nodes.push(
        createFillNode(graph, {
          id: `${id}-clear-left-counts`,
          operation: OPERATION,
          view: leftMatchCounts,
          type: 'u32',
          value: '0u'
        })
      );
    const sameRowTest = views.excludeSameRow ? ' && left != right' : '';
    const metric: SpatialDistanceMetric =
      this.sphereRadius === undefined
        ? {kind: 'planar'}
        : {kind: 'great-circle', radius: this.sphereRadius};
    const sideSpecs: SpatialPredicateSide[] = sides.map(side => ({
      prefix: side.name,
      kind: side.geometry.kind,
      vertexCount: side.geometry.positions.length,
      partCount: isMultipartJoinGeometry(side.geometry)
        ? side.geometry.kind === 'points'
          ? side.geometry.positions.length
          : side.geometry.lineOffsets.length - 1
        : undefined
    }));
    const sideBindings: WGSLKernelBinding[] = [];
    const distance: number | SpatialDistanceParameter = this.distanceView
      ? {
          expression: `bitcast<f32>(candidatePairs[candidatePairsOffset + ${tileCapacity * 2}u])`
        }
      : ((views.distance as number | undefined) ?? 0);
    for (const side of sides) {
      const prefix = side.name;
      if (isMultipartJoinGeometry(side.geometry))
        sideBindings.push({
          name: `${prefix}FeatureOffsets`,
          view: side.geometry.featureOffsets,
          type: 'u32',
          access: 'read'
        });
      sideBindings.push({
        name: `${prefix}Positions`,
        view: side.geometry.positions,
        type: 'f32',
        access: 'read'
      });
      if (side.geometry.kind === 'lines') {
        sideBindings.push({
          name: `${prefix}RingOffsets`,
          view: side.geometry.lineOffsets,
          type: 'u32',
          access: 'read'
        });
      } else if (side.geometry.kind === 'polygons') {
        sideBindings.push(
          {
            name: `${prefix}FeatureRings`,
            view: side.featureRings as GraphDataView,
            type: 'u32',
            access: 'read'
          },
          {
            name: `${prefix}RingOffsets`,
            view: side.geometry.ringOffsets,
            type: 'u32',
            access: 'read'
          }
        );
      }
    }
    for (let pageIndex = 0; pageIndex < this.candidatePages.pageCount; pageIndex++) {
      const id = this.candidatePages.pageCount > 1 ? `${this.id}-tile-${pageIndex}` : this.id;
      const candidateCapacity = tileCapacity;
      const firstSlot = pageIndex * candidateCapacity;
      const pageLength = Math.min(candidateCapacity, this.candidateCapacity - firstSlot);
      // Clear state, candidate sentinels, match counts, and the robust pair layout.
      {
        const bindings: WGSLKernelBinding[] = [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
        ];
        if (this.distanceView) {
          bindings.push({
            name: 'distanceRow',
            view: this.distanceView,
            type: 'f32',
            access: 'read'
          });
        }
        if (robust) {
          bindings.push(
            {name: 'pairPoints', view: robust.pairPoints, type: 'f32', access: 'read_write'},
            {
              name: 'pairGeometryOffsets',
              view: robust.pairGeometryOffsets,
              type: 'u32',
              access: 'read_write'
            }
          );
        }
        const lengths = [5, candidateCapacity, robust?.pairGeometryOffsets.length ?? 0];
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-clear`,
            operation: OPERATION,
            variant: 'clear',
            bindings,
            invocationCount: Math.max(...lengths),
            declarations: SPATIAL_JOIN_WGSL_HELPERS,
            body: `if (index == 0u) { flags[flagsOffset + ${candidateCapacity}u] = state[stateOffset + 2u]; }
  if (index < ${candidateCapacity}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
    flags[flagsOffset + index] = 0u;
  }
  ${this.distanceView ? `if (index == 0u) { candidatePairs[candidatePairsOffset + ${candidateCapacity * 2}u] = bitcast<u32>(distanceRow[distanceRowOffset]); }` : ''}
  
  ${
    robust
      ? `if (index < ${robust.pairPoints.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${robust.pairGeometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${(polygonSide as {polygonOffsets: GraphDataView}).polygonOffsets.length - 1}u, index == ${robust.pairGeometryOffsets.length - 1}u);
  }`
      : ''
  }`
          })
        );
      }

      // Candidate generation: count per left feature, scan, then write in (left, right) order.
      nodes.push(
        ...getSpatialJoinCandidateNodes<Parameters>(graph, {
          id,
          leftCount,
          rightCount,
          leftMinima: leftSide.minima as GraphDataView<'float32x2'>,
          leftMaxima: leftSide.maxima as GraphDataView<'float32x2'>,
          tree,
          candidateCapacity: this.candidateCapacity,
          page: {
            firstSlot,
            length: pageLength,
            initialize: pageIndex === 0,
            counts: pageCounts,
            offsets: pageOffsets,
            metadataRow: leftCount
          },
          margin: predicate === 'dwithin' ? (this.distanceView ?? (views.distance as number)) : 0,
          sphereRadius: this.sphereRadius,
          state,
          candidatePairs,
          candidateTruncated,
          candidateOverflow: views.candidateOverflow,
          bvhOverflow
        })
      );
      if (pageIndex === 0 && views.leftFeatureCount && candidateTruncated)
        nodes.push(
          createSpatialJoinLiveTruncationNode(
            graph,
            this.id,
            views.leftFeatureCount,
            candidateTruncated
          )
        );
      if (pageIndex === 0 && views.candidateTruncatedCount && candidateTruncated) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-count-truncated`,
            operation: OPERATION,
            variant: 'count-truncated',
            bindings: [
              {name: 'candidateTruncated', view: candidateTruncated, type: 'u32', access: 'read'},
              {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
            ],
            invocationCount: leftCount,
            body: `if (candidateTruncated[candidateTruncatedOffset + index] != 0u) {
    atomicAdd(&state[stateOffset + 4u], 1u);
  }`
          })
        );
      }
      nodes.push(
        createGraphChunkGroupKernelNode(graph, {
          id: `${id}-page-count`,
          operation: OPERATION,
          invocationCount: 1,
          parameterValues: [firstSlot, pageLength],
          bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
          body: `let retained = min(state[stateOffset], ${this.candidateCapacity}u);
  let first = min(getChunkParameter(36u), retained);
  state[stateOffset + 5u] = min(retained - first, getChunkParameter(37u));`
        })
      );
      const candidateDispatch = createWGSLActiveCountDispatch(graph, {
        id: `${id}-candidate-dispatch`,
        operation: OPERATION,
        count: state,
        countRow: 5,
        maximumItemCount: candidateCapacity
      });
      nodes.push(candidateDispatch.updateNode);
      const cooperativeDispatches = new Map<number, WGSLActiveCountDispatch<Parameters>>();
      const getCooperativeDispatch = (
        workgroupSize: number
      ): WGSLActiveCountDispatch<Parameters> => {
        let dispatch = cooperativeDispatches.get(workgroupSize);
        if (!dispatch) {
          dispatch = createWGSLActiveCountDispatch(graph, {
            id: `${id}-candidate-dispatch-${workgroupSize}`,
            operation: OPERATION,
            count: state,
            countRow: 5,
            maximumItemCount: candidateCapacity,
            workgroupSize,
            invocationsPerItem: workgroupSize
          });
          cooperativeDispatches.set(workgroupSize, dispatch);
          nodes.push(dispatch.updateNode);
        }
        return dispatch;
      };

      // Attribute equality: candidates with unequal keys become empty slots, which every later
      // kernel already skips.
      if (views.onAttribute) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-on-attribute`,
            operation: OPERATION,
            variant: 'on-attribute',
            bindings: [
              {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'},
              {name: 'leftKeys', view: views.onAttribute.left, type: 'u32', access: 'read'},
              {name: 'rightKeys', view: views.onAttribute.right, type: 'u32', access: 'read'}
            ],
            invocationCount: candidateCapacity,
            condition: candidateDispatch.condition,
            declarations: SPATIAL_JOIN_WGSL_HELPERS,
            body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (leftKeys[leftKeysOffset + left] != rightKeys[rightKeysOffset + right]) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }`
          })
        );
      }

      // Exact predicate per candidate slot.
      if (robust) {
        const pointsOnLeft = left.kind === 'points';
        const pointGeometry = pointsOnLeft ? left : right;
        const polygonGeometry = (pointsOnLeft ? right : left) as Extract<
          GPUSpatialJoinGeometry,
          {kind: 'polygons'}
        >;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-expand`,
            operation: OPERATION,
            variant: 'expand',
            bindings: [
              {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
              {name: 'pointPositions', view: pointGeometry.positions, type: 'f32', access: 'read'},
              {
                name: 'featureOffsets',
                view: polygonGeometry.featureOffsets,
                type: 'u32',
                access: 'read'
              },
              {name: 'pairPoints', view: robust.pairPoints, type: 'f32', access: 'read_write'},
              {
                name: 'pairGeometryOffsets',
                view: robust.pairGeometryOffsets,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: candidateCapacity,
            condition: candidateDispatch.condition,
            declarations: SPATIAL_JOIN_WGSL_HELPERS,
            body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let pointRow = ${pointsOnLeft ? 'left' : 'right'};
  let polygonRow = ${pointsOnLeft ? 'right' : 'left'};
  let row = index * 2u + 1u;
  pairPoints[pairPointsOffset + row * 2u] = pointPositions[pointPositionsOffset + pointRow * 2u];
  pairPoints[pairPointsOffset + row * 2u + 1u] = pointPositions[pointPositionsOffset + pointRow * 2u + 1u];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row] = featureOffsets[featureOffsetsOffset + polygonRow];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row + 1u] = featureOffsets[featureOffsetsOffset + polygonRow + 1u];`
          })
        );
        const robustDispatch = createWGSLActiveCountDispatch(graph, {
          id: `${id}-robust-dispatch`,
          operation: OPERATION,
          count: state,
          countRow: 5,
          maximumItemCount: candidateCapacity,
          invocationsPerItem: 2,
          additionalInvocationCount: 1
        });
        nodes.push(robustDispatch.updateNode);
        nodes.push(
          ...captureGraphCommandNodes(graph, () =>
            new GPUPairwisePointInPolygon({
              id: `${id}-classify`,
              points: robust.pairPoints,
              polygonPositions: polygonGeometry.positions,
              geometryOffsets: robust.pairGeometryOffsets,
              polygonOffsets: polygonGeometry.polygonOffsets,
              ringOffsets: polygonGeometry.ringOffsets,
              output: robust.pairClassifications
            }).addToGraph(graph)
          ).map(node => {
            // GPUPairwisePointInPolygon declares compute passes only; the dispatch condition below
            // applies to compute passes.
            validateRequirement(
              id,
              node.type === 'compute',
              'robust point-in-polygon classification must declare compute passes only'
            );
            return {...node, condition: robustDispatch.condition};
          })
        );
      }

      if (matrices) {
        this.addRelateNodes(graph, nodes, {
          id,
          candidateCapacity,
          sides,
          sideSpecs,
          sideBindings,
          candidatePairs,
          matrices,
          flags,
          state,
          robust,
          sameRowTest,
          candidateCondition: candidateDispatch.condition,
          cooperativeCondition: robust
            ? undefined
            : getCooperativeDispatch(SPATIAL_RELATE_WORKGROUP_SIZE).condition
        });
      } else if (robust) {
        // A point never contains a polygon and a polygon is never within a point; otherwise
        // `contains`/`within` need the point strictly inside and `intersects` accepts the boundary.
        const pointsOnLeft = left.kind === 'points';
        const impossible =
          (predicate === 'contains' && pointsOnLeft) || (predicate === 'within' && !pointsOnLeft);
        const accepted =
          predicate === 'intersects'
            ? 'classification == INSIDE || classification == BOUNDARY'
            : 'classification == INSIDE';
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-resolve`,
            operation: OPERATION,
            variant: 'resolve',
            bindings: [
              {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
              {
                name: 'pairClassifications',
                view: robust.pairClassifications,
                type: 'u32',
                access: 'read'
              },
              {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
              {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
            ],
            invocationCount: candidateCapacity,
            condition: candidateDispatch.condition,
            declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;`,
            body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    if (classification == UNCERTAIN) {
      atomicAdd(&state[stateOffset + 1u], 1u);
    } else {
      matched = ${impossible ? 'false' : `(${accepted})`}${sameRowTest};
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
          })
        );
        // Candidates the robust classifier could not certify are decided by the f32 test.
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-fallback`,
            operation: OPERATION,
            variant: `fallback-${left.kind}-${right.kind}-${predicate}`,
            bindings: [
              ...sideBindings,
              {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
              {
                name: 'pairClassifications',
                view: robust.pairClassifications,
                type: 'u32',
                access: 'read'
              },
              {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
            ],
            invocationCount: candidateCapacity,
            condition: candidateDispatch.condition,
            declarations: `${getSpatialPredicateWGSL(sideSpecs[0], sideSpecs[1], predicate as SpatialLegacyPredicateName, 0)}
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;`,
            body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  flags[flagsOffset + index] = select(0u, 1u, pairMatches(left, right)${sameRowTest});`
          })
        );
      } else {
        // At eight bindings retain uncertainty in bit 1 of the candidate flag, then reduce and
        // normalize those flags before any match scan/anti-join consumer reads them.
        const tracksUncertainty =
          !this.usesWorkgroupDistance && sideBindings.length + 3 <= MAXIMUM_STORAGE_BUFFERS;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-exact`,
            operation: OPERATION,
            variant: `exact-${left.kind}-${right.kind}-${predicate}${this.sphereRadius === undefined ? '' : '-great-circle'}`,
            bindings: [
              ...sideBindings,
              {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
              {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
              ...(tracksUncertainty
                ? [{name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'} as const]
                : [])
            ],
            ...(this.usesWorkgroupDistance
              ? {
                  // One workgroup per candidate slot: its lanes split the edges of the left feature.
                  invocationCount: candidateCapacity * SPATIAL_DWITHIN_WORKGROUP_SIZE,
                  workgroupSize: SPATIAL_DWITHIN_WORKGROUP_SIZE,
                  guardIndex: false,
                  condition: getCooperativeDispatch(SPATIAL_DWITHIN_WORKGROUP_SIZE).condition,
                  declarations: getSpatialDwithinWorkgroupWGSL(
                    sideSpecs[0],
                    sideSpecs[1],
                    distance
                  ),
                  body: `let slot = index / ${SPATIAL_DWITHIN_WORKGROUP_SIZE}u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matched = dwithinWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    flags[flagsOffset + slot] = select(0u, 1u, matched${sameRowTest}) |
      select(0u, 2u, atomicLoad(&dwithinUncertain) != 0u);
  }`
                }
              : {
                  invocationCount: candidateCapacity,
                  condition: candidateDispatch.condition,
                  declarations: getSpatialPredicateWGSL(
                    sideSpecs[0],
                    sideSpecs[1],
                    predicate as SpatialLegacyPredicateName,
                    distance,
                    metric
                  ),
                  // An orientation with no sign (non-finite input) counts as collinear and is
                  // reported through the uncertain count.
                  body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  uncertainOrientation = false;
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    matched = pairMatches(left, right)${sameRowTest};
    ${tracksUncertainty ? 'if (uncertainOrientation) { atomicAdd(&state[stateOffset + 1u], 1u); }' : ''}
  }
  flags[flagsOffset + index] = select(0u, 1u, matched)${tracksUncertainty ? '' : ' | select(0u, 2u, uncertainOrientation)'};`
                })
          })
        );
        if (!tracksUncertainty) {
          nodes.push(
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-count-uncertainty`,
              operation: OPERATION,
              variant: 'candidate-uncertainty',
              bindings: [
                {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
                {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
              ],
              invocationCount: candidateCapacity,
              condition: candidateDispatch.condition,
              body: `let flag = flags[flagsOffset + index];
  if ((flag & 2u) != 0u) { atomicAdd(&state[stateOffset + 1u], 1u); }
  flags[flagsOffset + index] = flag & 1u;`
            })
          );
        }
      }

      if (unmatched && leftMatchCounts) {
        this.addAntiNodes(graph, nodes, {
          id,
          candidateCapacity,
          countOnly: true,
          candidatePairs,
          flags,
          leftMatchCounts,
          unmatched,
          state,
          candidateTruncated,
          candidateCondition: candidateDispatch.condition
        });
      }
      if (leftRows && rightRows && flagOffsets) {
        // Stable compaction of flagged candidates into the (left, right) output.
        nodes.push(
          ...new GPUScan({
            id: `${id}-scan-flags`,
            input: graph.createDataView(flags.buffer, {
              format: 'uint32',
              length: candidateCapacity
            }),
            output: flagOffsets,
            mode: 'exclusive'
          }).getCommandNodes(graph)
        );
        const scatterBindings: WGSLKernelBinding[] = [
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'flagOffsets', view: flagOffsets, type: 'u32', access: 'read'},
          {name: 'leftRows', view: leftRows, type: 'u32', access: 'read_write'},
          {name: 'rightRows', view: rightRows, type: 'u32', access: 'read_write'}
        ];
        if (distancePairs) {
          scatterBindings.push({
            name: 'distancePairs',
            view: distancePairs,
            type: 'u32',
            access: 'read_write'
          });
        }
        if (relate && matrices) {
          scatterBindings.push(
            {name: 'matrices', view: matrices, type: 'u32', access: 'read'},
            {name: 'relateOut', view: relate, type: 'u32', access: 'read_write'}
          );
        }
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-scatter`,
            operation: OPERATION,
            variant: relate ? 'scatter-relate' : 'scatter',
            bindings: scatterBindings,
            invocationCount: candidateCapacity,
            condition: candidateDispatch.condition,
            declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
            body: `if (flags[flagsOffset + index] != 0u) {
    let base = flags[flagsOffset + ${candidateCapacity}u];
    let localSlot = flagOffsets[flagOffsetsOffset + index];
    let slot = base + min(localSlot, 0xffffffffu - base);
    if (slot < PAIR_CAPACITY) {
      leftRows[leftRowsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u];
      rightRows[rightRowsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
      ${
        distancePairs
          ? `distancePairs[distancePairsOffset + slot * 2u] = candidatePairs[candidatePairsOffset + index * 2u];
      distancePairs[distancePairsOffset + slot * 2u + 1u] = candidatePairs[candidatePairsOffset + index * 2u + 1u];`
          : ''
      }
      ${relate && matrices ? `relateOut[relateOutOffset + slot] = matrices[matricesOffset + index] & ${RELATE_OUTPUT_MASK}u;` : ''}
    }
  }`
          })
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-match-total`,
            operation: OPERATION,
            variant: 'match-total',
            bindings: [
              {name: 'state', view: state, type: 'u32', access: 'read_write'},
              {name: 'flags', view: flags, type: 'u32', access: 'read'},
              {name: 'flagOffsets', view: flagOffsets, type: 'u32', access: 'read'}
            ],
            invocationCount: 1,
            declarations: `${SCAN_TOTAL_WGSL}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
            body: `state[stateOffset + 2u] = getSaturatedScanSum(state[stateOffset + 2u],
    flagOffsets[flagOffsetsOffset + CANDIDATE_CAPACITY - 1u] +
    flags[flagsOffset + CANDIDATE_CAPACITY - 1u]);`
          })
        );
      }
    }
    if (unmatched && leftMatchCounts)
      this.addAntiNodes(graph, nodes, {
        id,
        candidateCapacity: tileCapacity,
        finishOnly: true,
        candidatePairs,
        flags,
        leftMatchCounts,
        unmatched,
        state,
        candidateTruncated,
        candidateCondition: undefined
      });
    // Scalars. Two inputs plus at most 6 scalar outputs per kernel (core limit of 8 storage
    // buffers); with more outputs wired, further kernels repeat the same totals for the rest.
    {
      const wired = this.scalarOutputs;
      for (let start = 0; start === 0 || start < wired.length; start += 6) {
        const group = wired.slice(start, start + 6);
        const bindings: WGSLKernelBinding[] = [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'bvhOverflow', view: bvhOverflow, type: 'u32', access: 'read'},
          ...group.map(
            ([name, view]): WGSLKernelBinding => ({
              name,
              view,
              type: 'u32',
              access: 'read_write'
            })
          )
        ];
        // Generate only wired assignments. A table keeps the audit from expanding sixteen
        // independent optional branches into 2^16 whole-shader variants.
        const scalarExpressions: Record<string, string> = {
          overflow: 'overflowValue',
          weightsCount: 'min(matchTotal, PAIR_CAPACITY)',
          weightsTotal: 'matchTotal',
          weightsOverflow: 'select(0u, 1u, pairIncomplete)',
          weightsCandidateOverflow: 'select(0u, 1u, candidateIncomplete)',
          weightsUncertainCount: 'state[stateOffset + 1u]',
          pairsOverflow: 'select(0u, 1u, pairIncomplete)',
          pairsCandidateOverflow: 'select(0u, 1u, candidateIncomplete)',
          pairsCount: 'min(matchTotal, PAIR_CAPACITY)',
          pairsTotal: 'matchTotal',
          unmatchedOverflow: 'select(0u, 1u, unmatchedIncomplete)',
          unmatchedCandidateOverflow: 'select(0u, 1u, candidateIncomplete)',
          unmatchedCount: 'min(unmatchedTotalValue, UNMATCHED_CAPACITY)',
          unmatchedTotal: 'unmatchedTotalValue',
          candidateCount: 'candidateTotal',
          uncertainCount: 'state[stateOffset + 1u]',
          candidateTruncatedCount: 'state[stateOffset + 4u]'
        };
        const assignments = group
          .map(([name]) => `${name}[${name}Offset] = ${scalarExpressions[name]};`)
          .join('\n  ');
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: start === 0 ? `${id}-finalize` : `${id}-finalize-${start / 6}`,
            operation: OPERATION,
            variant: 'finalize',
            bindings,
            invocationCount: 1,
            declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const UNMATCHED_CAPACITY: u32 = ${unmatched?.ids.length ?? 0}u;`,
            body: `let candidateTotal = state[stateOffset];
  let matchTotal = ${isAnti ? '0u' : 'state[stateOffset + 2u]'};
  let unmatchedTotalValue = ${unmatched ? 'state[stateOffset + 3u]' : '0u'};
  let candidateIncomplete = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY;
  let pairIncomplete = matchTotal > PAIR_CAPACITY;
  let unmatchedIncomplete = unmatchedTotalValue > UNMATCHED_CAPACITY;
  let overflowValue = select(0u, 1u, candidateIncomplete || pairIncomplete || unmatchedIncomplete);
  ${assignments}`
          })
        );
      }
    }

    // Optional CSR: rows are left features and pairs are already sorted by left row.
    if (weights && leftMatchCounts && leftRows && rightRows) {
      if (unmatched)
        nodes.push(
          createFillNode(graph, {
            id: `${id}-clear-csr-counts`,
            operation: OPERATION,
            view: leftMatchCounts,
            type: 'u32',
            value: '0u'
          })
        );
      const pairDispatch = createWGSLActiveCountDispatch(graph, {
        id: `${id}-pair-dispatch`,
        operation: OPERATION,
        count: state,
        countRow: 2,
        maximumItemCount: pairCapacity
      });
      nodes.push(pairDispatch.updateNode);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count-matches`,
          operation: OPERATION,
          variant: 'count-matches',
          bindings: [
            {name: 'leftRows', view: leftRows, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {
              name: 'leftMatchCounts',
              view: leftMatchCounts,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: pairCapacity,
          condition: pairDispatch.condition,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  atomicAdd(&leftMatchCounts[leftMatchCountsOffset + leftRows[leftRowsOffset + index]], 1u);`
        })
      );
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan-matches`,
          input: leftMatchCounts,
          output: weights.offsets,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fill-weights`,
          operation: OPERATION,
          variant: 'fill-weights',
          bindings: [
            {name: 'rightRows', view: rightRows, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'weightValues', view: weights.weights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: pairCapacity,
          condition: pairDispatch.condition,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  neighbors[neighborsOffset + index] = rightRows[rightRowsOffset + index];
  weightValues[weightValuesOffset + index] = 1.0;`
        })
      );
    }
    if (weights?.distances && distancePairs) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-distances`,
          operation: OPERATION,
          variant: 'distances',
          bindings: [
            ...sideBindings,
            {name: 'distancePairs', view: distancePairs, type: 'u32', access: 'read'},
            {name: 'distances', view: weights.distances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: pairCapacity,
          // Reuse the same closed-geometry distance as dwithin, including holes and containment.
          declarations: getSpatialPredicateWGSL(sideSpecs[0], sideSpecs[1], 'dwithin', 0, metric),
          body: `let left = distancePairs[distancePairsOffset + index * 2u];
  let right = distancePairs[distancePairsOffset + index * 2u + 1u];
  if (left == NO_FEATURE || right == NO_FEATURE) {
    distances[distancesOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    return;
  }
  let distanceValue = pairDistance(left, right);
  distances[distancesOffset + index] = ${this.sphereRadius === undefined ? 'distanceValue' : `distanceValue * ${getWGSLFloatLiteral(this.sphereRadius)}`};`
        })
      );
    }
    return nodes;
  }

  /**
   * Adds the relate-engine classification (matrix per candidate slot) and the mask that turns the
   * matrices into match flags and the uncertain count.
   */
  private addRelateNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    nodes: GPUCommandNode<Parameters>[],
    context: {
      id: string;
      candidateCapacity: number;
      sides: {geometry: GPUSpatialJoinGeometry}[];
      sideSpecs: SpatialPredicateSide[];
      sideBindings: WGSLKernelBinding[];
      candidatePairs: GraphDataView<'uint32x2'>;
      matrices: GraphDataView<'uint32'>;
      flags: GraphDataView<'uint32'>;
      state: GraphDataView<'uint32'>;
      robust:
        | {
            pairClassifications: GraphDataView<'uint32'>;
          }
        | undefined;
      sameRowTest: string;
      candidateCondition: GPUCommandGraphGPUIndirectCondition;
      cooperativeCondition?: GPUCommandGraphGPUIndirectCondition;
    }
  ): void {
    const {views} = this;
    const {id, candidateCapacity} = context;
    const {left, right} = views;
    const {sideSpecs, sideBindings, candidatePairs, matrices, flags, state, robust} = context;
    const uncertainBit = `${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u`;
    const classification = GPU_POINT_IN_POLYGON_CLASSIFICATION;
    if (robust) {
      // Point/polygon pairs: the matrix follows from the robust point location alone.
      const pointsOnLeft = left.kind === 'points';
      const [inside, boundary, outside] = pointsOnLeft
        ? ['0FFFFF212', 'F0FFFF212', 'FF0FFF212']
        : ['0F2FF1FF2', 'FF20F1FF2', 'FF2FF10F2'];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-resolve-relate`,
          operation: OPERATION,
          variant: 'resolve-relate',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'matrices', view: matrices, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          condition: context.candidateCondition,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const INSIDE: u32 = ${classification.inside}u;
const BOUNDARY: u32 = ${classification.boundary}u;
const UNCERTAIN: u32 = ${classification.uncertain}u;`,
          body: `var matrix = 0u;
  if (candidatePairs[candidatePairsOffset + index * 2u] != NO_FEATURE) {
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    matrix = select(select(select(${packGPUSpatialRelate(outside)}u, ${packGPUSpatialRelate(boundary)}u, classification == BOUNDARY), ${packGPUSpatialRelate(inside)}u, classification == INSIDE), ${uncertainBit}, classification == UNCERTAIN);
  }
  matrices[matricesOffset + index] = matrix;`
        })
      );
      // Candidates the robust classifier could not certify are classified by the f32 relate engine.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fallback-relate`,
          operation: OPERATION,
          variant: `fallback-relate-${left.kind}-${right.kind}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'matrices', view: matrices, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          condition: context.candidateCondition,
          declarations: `${getSpatialRelateWGSL(sideSpecs[0], sideSpecs[1])}
const UNCERTAIN: u32 = ${classification.uncertain}u;`,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  matrices[matricesOffset + index] = pairRelate(left, right) | ${uncertainBit};`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-relate`,
          operation: OPERATION,
          variant: `relate-${left.kind}-${right.kind}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'matrices', view: matrices, type: 'u32', access: 'read_write'}
          ],
          // One workgroup per candidate slot: its lanes split the edge loops of the pair.
          invocationCount: candidateCapacity * SPATIAL_RELATE_WORKGROUP_SIZE,
          workgroupSize: SPATIAL_RELATE_WORKGROUP_SIZE,
          guardIndex: false,
          condition: context.cooperativeCondition,
          declarations: `${getSpatialRelateWGSL(sideSpecs[0], sideSpecs[1], true)}
${getSpatialRelateWorkgroupWGSL(sideSpecs)}`,
          body: `let slot = index / ${SPATIAL_RELATE_WORKGROUP_SIZE}u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matrix = relateWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    matrices[matricesOffset + slot] = matrix;
  }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mask`,
        operation: OPERATION,
        variant: 'mask',
        bindings: [
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'matrices', view: matrices, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
          ...(this.patternView
            ? [{name: 'patternWords', view: this.patternView, type: 'u32', access: 'read'} as const]
            : [])
        ],
        invocationCount: candidateCapacity,
        condition: context.candidateCondition,
        declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
${
  this.patternView
    ? getRelateMatchesParametersWGSL(this.patternView.length / GPU_SPATIAL_RELATE_PATTERN_WORDS)
    : getRelateMatchesWGSL(this.relatePatterns)
}`,
        body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let raw = matrices[matricesOffset + index];
    if ((raw & ${uncertainBit}) != 0u) { atomicAdd(&state[stateOffset + 1u], 1u); }
    let matrix = raw & 0x3ffffu;
    // An all-F matrix marks an empty or invalid feature.
    matched = matrix != 0u && relateMatches(matrix)${context.sameRowTest};
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
      })
    );
  }

  /** Adds the anti-join nodes: per-left match counts, flags, scan and compaction of unmatched rows. */
  private addAntiNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    nodes: GPUCommandNode<Parameters>[],
    context: Omit<SpatialPredicateUnmatchedContext, 'leftCount' | 'leftFeatureCount'>
  ): void {
    addSpatialPredicateUnmatchedNodes(graph, nodes, {
      ...context,
      leftCount: this.leftCount,
      leftFeatureCount: this.views.leftFeatureCount
    });
  }
}
