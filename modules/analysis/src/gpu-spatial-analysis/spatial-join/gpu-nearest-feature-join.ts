// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

import type {
  GPUCandidateStatusPort,
  GPULonLatPolygonStatusPort
} from '../../analysis-core/contracts/index';
import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/kernels/lon-lat-ring-wgsl';
import {createStorageTopKNode} from '../../analysis-core/kernels/storage-backed-top-k';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLActiveCountDispatch,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUCompactOutput,
  GPUUint32Rows
} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validateRequirement,
  validateFiniteNumber,
  validatePowerOfTwo,
  validateCapacity,
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createSpatialJoinClearNode,
  createSpatialJoinFinalizeNodes,
  getChunkNodeId,
  getDefaultSpatialSort,
  type SpatialSortCurve,
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getDefaultQuerySort,
  getLeafOverflowStatusNodes,
  getSpatialJoinProbeNodes,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';
import {NEAREST_NODE_WORDS, NEAREST_COMMON_WGSL, NEAREST_DISTANCE_RANK_WGSL} from './nearest-wgsl';
import {validateSpatialJoinReadWriteBindings} from './spatial-join-binding-preflight';
import type {GPUNearestFeatureSource, GPUSpatialJoinOnAttribute} from './spatial-join-types';
import {
  validateStatusScalars,
  validateSpatialJoinGeometry,
  isMultipartJoinGeometry,
  isSameSpatialJoinGeometry
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
import {
  getNearestEstimatedCommandNodeCount,
  getNearestBVHNodes,
  getNearestGeometryCount,
  getNearestGeometryViews,
  getNearestNeighborNodes,
  validateNearestNeighborBindings,
  type NearestNeighborNodeProps
} from './nearest-nodes';
import {
  NEAREST_MAXIMUM_NEIGHBOR_CAPACITY,
  type GPUNearestFeatureGeometry,
  type GPUNearestQueryGeometry,
  type GPUNearestTieMode
} from './nearest-types';

import {
  GPU_LINEAR_EDGE_MODELS,
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {Device} from '@luma.gl/core';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotJoinStatus} from './spatial-join-construction-snapshot';

/** Planar or great-circle contexts suitable for distance measurements in their length unit. */
const NEAREST_FEATURE_JOIN_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  measure: 'distance'
} as const);

const OPERATION = 'GPUNearestFeatureJoin';
/** Spherical distance shaders emit the sphere radius as an f32 literal. */
const MAXIMUM_SHADER_FLOAT = 3.4028234663852886e38;
/** A positive radius must remain nonzero in the shader's f32 representation. */
const MINIMUM_SHADER_POSITIVE_FLOAT = 1.401298464324817e-45;

import {
  getNearestFeatureJoinViews,
  getNearestFeatureJoinInputViews as getInputs,
  getNearestFeatureJoinOutputViews as getOutputs,
  type NearestFeatureJoinViews,
  type NearestDistanceRows
} from './nearest-feature-views';

/** Caller-owned outputs of {@link GPUNearestFeatureJoin}. */
export type GPUNearestFeatureJoinOutput<
  IdRows extends GPUUint32Rows = GraphDataView<'uint32'>,
  DistanceRows extends NearestDistanceRows = GraphDataView<'float32'>
> = {
  /**
   * Nearest-feature mode: per-query nearest feature ID or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`.
   * Chunked like single-point queries; packed with one row per multipart, line or polygon query.
   */
  nearestFeatureIds?: IdRows;
  /** Nearest feature row before source-ID remapping, or 0xffffffff. Same topology as nearestFeatureIds. */
  nearestFeatureRows?: IdRows;
  /**
   * Neighbors mode: feature ID (or row, without `features.sourceIds`) per `(query, slot)` at index
   * `query * neighborCapacity + slot`, ordered by distance then feature row, padded with
   * `GPU_SPATIAL_JOIN_NO_FEATURE`. Length `queryCount * neighborCapacity`.
   */
  neighborIds?: GraphDataView<'uint32'>;
  /**
   * Neighbors mode: feature row before applying `features.sourceIds`, with the same layout and
   * missing sentinel as `neighborIds`. Wire this to `GPUPairGather.neighbors.rows` to gather
   * feature attributes. Requires `neighborIds`; both outputs must be disjoint.
   */
  neighborRows?: GraphDataView<'uint32'>;
  /** Neighbors mode, required with `neighborIds`: number of filled slots per query. Length `queryCount`. */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Neighbors mode: distance per slot (in the spatial context length unit), `-1` in unused slots. Same layout as `neighborIds`. */
  neighborDistances?: GraphDataView<'float32'>;
  /**
   * Neighbors mode: the point of the feature nearest to the query per slot (the query point
   * projected onto the feature, the crossing point when geometries cross, or the contained vertex
   * for polygon containment); NaN in unused slots. Same layout as `neighborIds`.
   */
  neighborFootPoints?: GraphDataView<'float32x2'>;
  /**
   * Neighbors mode: absolute index in the feature's position buffer of the first vertex of the
   * edge attaining the distance (`GPU_NEAREST_NO_SEGMENT` for point features, containment and
   * unused slots; always 0 for `segments` features). Same layout as `neighborIds`.
   */
  neighborSegmentIndices?: GraphDataView<'uint32'>;
  /**
   * Neighbors mode: the point of the query geometry nearest to the feature per slot (the query
   * point itself for point queries); NaN in unused slots. Same layout as `neighborIds`.
   * `(neighborQueryPoints, neighborFootPoints)` is `shapely.shortest_line` and
   * `shapely.ops.nearest_points` for every kind pair (line and polygon queries included). For
   * crossing geometries both are the crossing point and for polygon containment both are the
   * contained vertex. Where several point pairs attain the distance (parallel edges) the first in
   * vertex order is returned, which may differ from GEOS.
   */
  neighborQueryPoints?: GraphDataView<'float32x2'>;
  /** Nearest-feature mode: optional per-query distance to the nearest feature (in the spatial context length unit), or -1. Same row topology as nearestFeatureIds. */
  nearestDistances?: DistanceRows;
  /** Nearest-feature mode: optional per-feature count of query features whose nearest feature it is. */
  featureCounts?: GraphDataView<'uint32'>;
  /** Optional compact stable query IDs (or query rows without sourceIds) with a feature within the radius. */
  matches?: GPUCompactOutput;
  /**
   * Optional per-query-row flag (one `uint32` per query, in row order across point chunks), 1 when
   * that query's search may be incomplete. A flagged query may report a farther feature than the
   * true nearest, or none (in neighbors mode, wrong or missing neighbors).
   *
   * - Planar single-point queries against single points/segments in nearest-feature mode: 1 when that point lost candidates because the candidate total
   *   exceeded `candidateCapacity`. Candidates are laid out in point-row order and the slots past
   *   the capacity are dropped, so the flagged points are the point whose candidates cross the
   *   capacity (it keeps its first ones) and every later point that had a candidate (it keeps
   *   none); earlier points are complete.
   * - Every mode (planar and great-circle nearest-feature, neighbors): a BVH overflow (more
   *   features than `leafCapacity`) drops features that may be the nearest of any query, so it
   *   flags every query. Neighbors mode, multipart/line/polygon nearest-feature mode and the great-circle search have no candidate table, so
   *   this is their only cause.
   */
  candidateTruncated?: GraphDataView<'uint32'>;
  /**
   * Completeness scalars. Optional `antimeridianUncertainCount` counts unprepared crossing rings;
   * optional `poleUncertainCount` counts unprepared pole-enclosing rings. Prepare these polygons
   * with `GPULonLatPolygonPreparation`. Both fields are left untouched in planar contexts.
   */
  status: GPULonLatPolygonStatusPort &
    GPUCandidateStatusPort & {
      /**
       * One-row flag: 1 when BVH leaf, candidate, `matches` or (neighbors mode) tie capacity
       * overflowed. Its meaning is unchanged; `candidateOverflow` (every mode) isolates the
       * intermediate capacities: the BVH leaf capacity and, in planar nearest-feature mode, the
       * candidate capacity.
       */
      overflow: GraphDataView<'uint32'>;
      /** Optional one-row unclamped candidate count; zero on geometry-aware and great-circle traversal routes, including neighbors mode. */
      candidateCount?: GraphDataView<'uint32'>;
    };
};

/**
 * Largest packed node table of {@link GPUNearestFeatureJoin} on `device`, in uint32 words, in
 * neighbors mode, geometry-aware nearest-feature mode and on a great-circle context:
 * `5 * (2 * leafCapacity - 1) + 2 * queryCount`, plus
 * `2 * featureCount` with `exclusive` or `onAttribute`. The table is read at random by the BVH
 * traversal and must fit one storage binding, so 2^25 words on a default device (128 MiB
 * bindings). Without queries or filters that allows `leafCapacity` up to 2^21 (2097152 features).
 * The per-query traversal scratch is already chunked and does not add a limit.
 */
export function getGPUNearestFeatureJoinMaximumNodeTableWordCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest `leafCapacity` of {@link GPUNearestFeatureJoin} on `device`: the BVH holds
 * `2 * leafCapacity - 1` float32x2 node bounds per corner table, 8 bytes per node, in one storage
 * binding, so 2^23 leaves (features) on a default device.
 */
export function getGPUNearestFeatureJoinMaximumLeafCapacity(device: Device): number {
  // Largest power of two L with (2 * L - 1) nodes of 8 bytes in one binding.
  return 2 ** Math.floor(Math.log2(Math.floor((getStorageBindingElementLimit(device, 8) + 1) / 2)));
}

/**
 * Properties for {@link GPUNearestFeatureJoin}.
 *
 * Two output modes share one class:
 *
 * - **Nearest-feature mode**: `queries`, `radius` and `output.nearestFeatureIds`. One nearest
 *   feature per query within a radius. Multipart points/lines and polygons use minimum part-pair
 *   distance in both modes; k-nearest and counts always count logical features, not parts.
 *   Planar single-point queries against single points/segments also require `candidateCapacity`.
 * - **Neighbors mode**: set `output.neighborIds` and `output.neighborCounts`. Each query (a chunked
 *   point set, or any line or polygon set) gets its `k` nearest features from a
 *   branch-and-bound BVH traversal, in dense slots of `neighborCapacity` per query, ordered by
 *   `(distance, feature row)`. Features may be points, segments, linestrings or polygons, so this
 *   mode covers point, line/line, line/polygon and polygon/polygon minimum distance. It is
 *   `sjoin_nearest` and PostGIS `ORDER BY a <-> b LIMIT k`.
 *
 * Per-frame: `radius` and the contents of every input buffer. Topology: view
 * lengths and chunking, `neighborCount`, `neighborCapacity`, `ties`, `candidateCapacity`, `leafCapacity`,
 * `spatialSort`, and which optional views exist.
 */
export type GPUNearestFeatureJoinProps<
  IdRows extends GPUUint32Rows = GraphDataView<'uint32'>,
  DistanceRows extends NearestDistanceRows = GraphDataView<'float32'>,
  QueryGeometry extends GPUNearestQueryGeometry = GPUNearestQueryGeometry,
  Parameters = unknown
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'nearest-feature-join'`. */
  id?: string;
  /**
   * Coordinate space and metric of the points and features.
   *
   * - Omitted, or planar: native planar coordinates; distances (`radius`, `nearestDistances`,
   *   `neighborDistances`) are planar in coordinate units.
   * - `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`: positions are longitude and
   *   latitude in degrees and every distance is a great-circle distance in the context length unit on the mean Earth
   *   sphere (meters by default, or kilometers with `units: 'kilometers'`; `sphereRadius`
   *   overrides the radius in that unit). Both modes use the branch-and-bound BVH traversal with a
   *   conservative great-circle lower bound on node boxes (latitude gap, longitude gap through the
   *   antimeridian when shorter, and the cosine of the latitude farthest from the equator), so
   *   `candidateCapacity` is not used (it is accepted but ignored, `candidateCount` reads 0) and a
   *   pair across the antimeridian or near a pole is measured the short way round.
   *   Edges are straight in longitude/latitude (longitude differences wrapped at +-180), the same
   *   edge model as the planar-in-degrees containment and crossing tests, so densify long edges;
   *   foot points are on the edge and longitudes are wrapped to +-180. Prepare polygons that
   *   cross ±180 or enclose a pole with `GPULonLatPolygonPreparation`; optional status counters
   *   report unprepared rings on both polygon sides. Longitude/latitude requires `edgeModel: 'linear'`
   *   (the default); great-circle edges require preparation. Positions are float32, so distances
   *   carry the coordinate quantization (about 1 m at longitude 100).
   * - Any other longitude/latitude metric throws at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `features` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  featuresSpatialContext?: GPUSpatialContext;
  /**
   * Query geometry, one query per feature row in both output modes. Single points may be chunked;
   * multipart points/lines and polygons are packed. Optional `sourceIds` (one per query feature,
   * chunked like single points) are the IDs written to `output.matches`.
   */
  queries: QueryGeometry;
  /**
   * Features: points, segments, linestrings or polygons in both output modes. Multipart points
   * and lines group parts through featureOffsets. Distance is the minimum over part pairs;
   * polygon containment gives distance 0. Ties between features choose the lowest feature row.
   * Optional `sourceIds` are stable feature IDs written instead of feature rows.
   */
  features: GPUNearestFeatureGeometry;
  /**
   * Per-frame inclusive search radius (in the spatial context length unit), one float32
   * row; features farther than this are never returned. NaN, negative, or infinite matches
   * nothing. Required in nearest-feature mode; omitted in neighbors mode means unbounded.
   */
  radius?: GraphDataView<'float32'>;
  /**
   * Skip features whose ID equals the query's ID (`STRtree.query_nearest(exclusive=True)`, the
   * self-neighbor rule of a nearest self-join). A feature's ID is `features.sourceIds[row]`, or its row
   * without them; a query's ID is `queryIds[row]`, or its row without `queryIds`. Skipped
   * features are never counted, so `k` neighbors are the nearest *other* features. Compile-time.
   * Works in both modes.
   */
  exclusive?: boolean;
  /**
   * Optional `uint32` ID per query row (length is the query count), compared with feature IDs by
   * `exclusive`. Requires `exclusive`. Unlike point `sourceIds` it is never chunked.
   */
  queryIds?: GraphDataView<'uint32'>;
  /**
   * Attribute-equality condition (GeoPandas `sjoin_nearest` after `on_attribute`): only features
   * whose `right` key equals the query's `left` key compete, so the nearest *key-equal* feature is
   * returned. `left` has one key per query row and `right` one per feature row. Compile-time
   * presence, per-frame contents. Combines with `exclusive`. Works in both modes.
   */
  onAttribute?: GPUSpatialJoinOnAttribute;
  /** Planar single-point queries against single points/segments in nearest-feature mode: maximum bounding-box candidates per encoding. Other geometry or great-circle routes ignore this prop and write candidateCount = 0. */
  candidateCapacity?: number;
  /** Neighbors mode: neighbors wanted per query, a positive integer. Default 1. */
  neighborCount?: number;
  /**
   * Neighbors mode: output slots per query, at least `neighborCount` (storage-backed above 64). Defaults to `neighborCount`. It only
   * matters with `ties: 'all'`: ties beyond the capacity are dropped and `overflow` is set.
   */
  neighborCapacity?: number;
  /** Neighbors mode: tie rule at the k-th distance. Default keeps the lowest rows. */
  ties?: GPUNearestTieMode;
  /**
   * Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. In
   * neighbors mode and on a great-circle context the node table of `leafCapacity` and the query
   * count is at most {@link getGPUNearestFeatureJoinMaximumNodeTableWordCount} words (2^25 on a default device).
   */
  leafCapacity?: number;
  /**
   * Compile-time. When true, features are reordered along a Hilbert curve (see `spatialSortCurve`) of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (the nearest feature wins and ties keep the smallest feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 50,000 shuffled short
   * segments, 100,000 points and radius 1: 309 ms drops to 4.1 ms per encoding (about 33,000 to 76
   * BVH nodes visited per point). Candidate counts are identical.
   *
   * Defaults to on from 256 features and off below, where the BVH is too shallow for the order to
   * matter. On coherent data the sort costs about 1 ms; pass `false` to skip it.
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /**
   * Compile-time. When true, queries are processed in Hilbert order of points (each chunk) or
   * line/polygon bounds centers (quantized over the feature BVH root bounds, non-finite points
   * and empty bounds last) instead
   * of input order, so neighboring invocations walk neighboring BVH subtrees. Every query still
   * reads and writes its own row, so all outputs are identical either way; only traversal cost
   * changes. In neighbors and great-circle modes, point keys dispatch at each chunk's actual
   * length; two shared stable sorts group Hilbert order back into chunk slices. Costs geometry bounds and centers
   * for line/polygon queries. Polygon query topology arrays are packed to stay within eight bindings.
   *
   * In nearest-feature mode with `radius` (planar), the second candidate pass keeps input order
   * to stay within eight storage bindings.
   *
   * Defaults off for line and polygon queries: bounds, centers and radix sorting cost more than
   * traversal saves in the measured 8192-line and polygon workloads. Explicit `true` still enables
   * it. Point queries default to on from 4096 queries and 256 features (see
   * `spatial-join-query-order.spec.ts` for the A/B harness), and off below, where the sort cannot
   * pay for itself.
   */
  querySort?: boolean;
  /** Caller-owned outputs; `neighborIds` selects neighbors mode. */
  output: GPUNearestFeatureJoinOutput<IdRows, DistanceRows>;
  /**
   * Prepared feature index from {@link GPUSpatialJoinPrepared}, taking leafCapacity and spatialSort
   * from the handle. Add the handle before this join with the same feature views. Traversal and
   * distance refinement read the retained coordinates and offsets until an invalidated generation
   * executes; source IDs remain live caller data. Results are recomputed for every encoding.
   */
  prepared?: GPUSpatialJoinPrepared<Parameters>;
};

/**
 * Joins queries to their nearest features by planar or great-circle distance.
 *
 * Planar nearest-feature mode with single-point queries and single points/segments: a `GPUBVH`
 * over feature bounds produces candidates inside the radius box, a squared point-to-segment
 * distance kernel measures them, and deterministic `atomicMin` passes keep the nearest feature
 * (ties go to the smallest feature row). Multipart, line/polygon and great-circle inputs instead
 * use the feature-level neighbors traversal with `k = 1` and publish the same output columns.
 *
 * Neighbors mode (`neighborIds` set): every query walks the BVH with a stack, nearer child first,
 * pruning nodes farther than its current k-th distance, and keeps the best rows ordered by
 * `(distance, feature row)`. The result is exact and deterministic: ties at the k-th distance keep
 * the lowest feature rows, or all of them with `ties: 'all'`. Distances are measured between
 * the true geometries (point, linestring or polygon on either side), so no radius guess or
 * candidate capacity is needed; the only bounded output is the per-query slot count, and tie
 * truncation raises `overflow`.
 */
export class GPUNearestFeatureJoin<
  IdRows extends GPUUint32Rows = GraphDataView<'uint32'>,
  DistanceRows extends NearestDistanceRows = GraphDataView<'float32'>,
  QueryGeometry extends GPUNearestQueryGeometry = GPUNearestQueryGeometry,
  Parameters = unknown
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNearestFeatureJoinProps<IdRows, DistanceRows, QueryGeometry, Parameters>;
  /** Flat view of `props` read by the node builders. */
  private readonly views: NearestFeatureJoinViews;
  /** `'neighbors'` when `neighborIds` is set, otherwise the original `'nearest-feature'` mode. */
  readonly mode: 'nearest-feature' | 'neighbors';
  /** Number of features. */
  readonly featureCount: number;
  /** Number of queries (points or query geometries). */
  readonly queryCount: number;
  /** Conservative upper bound including device-sized nearest scratch chunks. */
  readonly estimatedCommandNodeCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Candidate pair capacity (nearest-feature mode; 0 in neighbors mode). */
  readonly candidateCapacity: number;
  /** Whether features are spatially sorted before the BVH build. */
  readonly spatialSort: boolean;
  /** Whether queries are processed in Hilbert order. */
  readonly querySort: boolean;
  /** Resolved neighbors per query (1 in nearest-feature mode). */
  readonly neighborCount: number;
  /** Resolved output slots per query (1 in nearest-feature mode). */
  readonly neighborCapacity: number;
  /** Resolved tie rule. */
  readonly ties: GPUNearestTieMode;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in the context length unit of a great-circle search, or undefined for planar distances. */
  readonly sphereRadius: number | undefined;
  /** Geometry-aware nearest-feature mode uses the same feature-level traversal as neighbors. */
  private readonly traversalNearestFeature: boolean;

  constructor(props: GPUNearestFeatureJoinProps<IdRows, DistanceRows, QueryGeometry, Parameters>) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'featuresSpatialContext',
      'queries',
      'features',
      'radius',
      'exclusive',
      'queryIds',
      'onAttribute',
      'candidateCapacity',
      'neighborCount',
      'neighborCapacity',
      'ties',
      'leafCapacity',
      'spatialSort',
      'spatialSortCurve',
      'querySort',
      'output',
      'prepared'
    ]);
    const id = snapshot.id ?? 'nearest-feature-join';
    validateRequirement(
      {id, contributor: 'GPUNearestFeatureJoin', messagePrefix: `${id}:`},
      !('k' in props),
      'k was renamed; use neighborCount'
    );
    validateRequirement(
      id,
      (snapshot.ties as unknown) !== 'lowest-id',
      "ties 'lowest-id' was renamed; use 'lowest-row'"
    );
    validateRequirement(
      id,
      snapshot.ties === undefined || ['lowest-row', 'all'].includes(snapshot.ties),
      `ties must be 'lowest-row' or 'all'; received ${String(snapshot.ties)}`
    );
    const snapshotGeometry = <Geometry extends {kind: string}>(geometry: Geometry): Geometry =>
      snapshotConstructionFields(geometry, [
        'kind',
        'positions',
        'sourceIds',
        'lineOffsets',
        'featureOffsets',
        'polygonOffsets',
        'ringOffsets',
        'starts',
        'ends'
      ] as readonly (keyof Geometry)[]);
    const output = snapshotConstructionFields(snapshot.output, [
      'nearestFeatureIds',
      'nearestFeatureRows',
      'neighborIds',
      'neighborRows',
      'neighborCounts',
      'neighborDistances',
      'neighborFootPoints',
      'neighborSegmentIndices',
      'neighborQueryPoints',
      'nearestDistances',
      'featureCounts',
      'matches',
      'candidateTruncated',
      'status'
    ]);
    const matches = output.matches && snapshotConstructionFields(output.matches, ['ids', 'status']);
    const spatialContext =
      snapshot.spatialContext && snapshotConstructionSpatialContext(snapshot.spatialContext);
    const featuresSpatialContext =
      snapshot.featuresSpatialContext === snapshot.spatialContext
        ? spatialContext
        : snapshot.featuresSpatialContext &&
          snapshotConstructionSpatialContext(snapshot.featuresSpatialContext);
    props = Object.freeze({
      ...snapshot,
      queries: snapshotGeometry(snapshot.queries),
      features: snapshotGeometry(snapshot.features),
      ...(spatialContext ? {spatialContext} : {}),
      ...(featuresSpatialContext ? {featuresSpatialContext} : {}),
      ...(snapshot.onAttribute
        ? {onAttribute: snapshotConstructionFields(snapshot.onAttribute, ['left', 'right'])}
        : {}),
      output: Object.freeze({
        ...output,
        status: snapshotJoinStatus(output.status),
        ...(matches
          ? {
              matches: Object.freeze({
                ...matches,
                status: snapshotJoinStatus(matches.status)
              })
            }
          : {})
      })
    });
    this.id = id;
    const multipart =
      isMultipartJoinGeometry(props.queries) || isMultipartJoinGeometry(props.features);
    if (multipart && props.output.neighborIds) {
      if (props.features.kind !== 'segments')
        validateSpatialJoinGeometry(this.id, 'features', props.features, {
          allowMultipart: true,
          allowEmpty: true
        });
      if (isMultipartJoinGeometry(props.queries))
        validateSpatialJoinGeometry(this.id, 'queries', props.queries, {
          allowMultipart: true,
          allowEmpty: true
        });
    }
    this.props = props;
    const views = getNearestFeatureJoinViews(props);
    this.views = views;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      views.spatialContext,
      NEAREST_FEATURE_JOIN_SPATIAL_CONTEXT_SUPPORT
    );
    resolveGPUSecondarySpatialContext(
      id,
      'features',
      views.featuresSpatialContext,
      this.spatialContext,
      NEAREST_FEATURE_JOIN_SPATIAL_CONTEXT_SUPPORT.edgeModels
    );
    this.sphereRadius = getGPUGreatCircleRadius(this.spatialContext);
    if (this.sphereRadius !== undefined) {
      validateFiniteNumber(
        id,
        'spatialContext.sphereRadius',
        this.sphereRadius,
        MINIMUM_SHADER_POSITIVE_FLOAT,
        MAXIMUM_SHADER_FLOAT
      );
    }
    this.mode = views.neighborIds ? 'neighbors' : 'nearest-feature';
    this.traversalNearestFeature =
      this.sphereRadius !== undefined ||
      multipart ||
      views.queries !== undefined ||
      (views.features.kind !== 'points' && views.features.kind !== 'segments');
    this.featureCount = getNearestGeometryCount(views.features);
    this.neighborCount = views.neighborCount ?? 1;
    this.ties = views.ties ?? 'lowest-row';
    const {prepared} = views;
    if (prepared) {
      if (
        views.features.kind === 'segments' ||
        !isSameSpatialJoinGeometry(prepared.geometry, views.features)
      ) {
        throw new Error(`${id} prepared must index the same point, line or polygon features`);
      }
      if (prepared.featureCount !== this.featureCount) {
        throw new Error(`${id} prepared feature count must equal the feature count`);
      }
      if (
        (views.leafCapacity !== undefined && views.leafCapacity !== prepared.leafCapacity) ||
        (views.spatialSort !== undefined && views.spatialSort !== prepared.spatialSort)
      ) {
        throw new Error(`${id} leafCapacity and spatialSort come from the prepared handle`);
      }
    }
    this.spatialSort = prepared
      ? prepared.spatialSort
      : (views.spatialSort ?? getDefaultSpatialSort(this.featureCount));
    this.leafCapacity =
      prepared?.leafCapacity ??
      views.leafCapacity ??
      getNextPowerOfTwo(Math.max(this.featureCount, 1));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    if (views.radius) {
      validateSinglePackedView(views.radius, ['float32'], `${id} radius`);
      if (views.radius.length !== 1) {
        throw new Error(`${id} radius needs one float32 row`);
      }
    }
    for (const view of getNearestGeometryViews(views.features)) {
      validateSinglePackedView(
        view,
        view.format === 'uint32' ? ['uint32'] : ['float32x2'],
        `${id} features`
      );
    }
    if (views.features.kind === 'segments') {
      if (views.features.starts.length !== views.features.ends.length) {
        throw new Error(`${id} features.starts and ends lengths differ`);
      }
    }
    for (const [name, view] of [
      ['features.sourceIds', views.featureIds],
      ['output.featureCounts', views.featureCounts]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} ${name} needs one row per feature`);
        }
      }
    }
    validateStatusScalars(id, props.output.status);
    if (this.mode === 'neighbors') {
      this.neighborCapacity = views.neighborCapacity ?? this.neighborCount;
      this.candidateCapacity = 0;
      this.queryCount = this.validateNeighbors();
    } else {
      this.neighborCapacity = 1;
      this.candidateCapacity = this.traversalNearestFeature ? 0 : (views.candidateCapacity ?? 0);
      this.queryCount = this.validateNearestFeature();
    }
    if (views.candidateTruncated) {
      validateSinglePackedView(
        views.candidateTruncated,
        ['uint32'],
        `${id} output.candidateTruncated`
      );
      validateRequirement(
        id,
        views.candidateTruncated.length === this.queryCount,
        'output.candidateTruncated needs one row per query'
      );
    }
    this.querySort =
      views.querySort ??
      (views.queries === undefined && getDefaultQuerySort(this.queryCount, this.featureCount));
    const queryChunks = views.points ? getGraphViewChunks(views.points) : [];
    const owner = views.overflow.buffer.graph as GPUCommandGraph;
    this.estimatedCommandNodeCount = getNearestEstimatedCommandNodeCount(
      this.queryCount,
      this.neighborCapacity,
      Boolean(
        views.neighborFootPoints || views.neighborQueryPoints || views.neighborSegmentIndices
      ),
      Math.max(queryChunks.length, 1),
      owner.device.limits.maxStorageBufferBindingSize,
      undefined,
      this.querySort
    );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.validateFilters();
    if (this.mode === 'neighbors') {
      validateSpatialJoinReadWriteBindings(
        id,
        [views.featureIds],
        [
          views.neighborIds,
          views.neighborRows,
          views.neighborCounts,
          views.neighborDistances,
          views.overflow
        ]
      );
    }
    validateDisjointWritableViews(
      id,
      Object.fromEntries(getOutputs(views).map((view, index) => [`output ${index}`, view]))
    );
    validateDisjointOutputs(id, getInputs(views), getOutputs(views));
  }

  /** Validates `exclusive`, `queryIds` and `onAttribute` against the query and feature counts. */
  private validateFilters(): void {
    const {id, views, queryCount, featureCount} = this;
    if (views.queryIds && !views.exclusive) {
      throw new Error(`${id} queryIds requires exclusive`);
    }
    if (views.queryIds) {
      validateSinglePackedView(views.queryIds, ['uint32'], `${id} queryIds`);
      if (views.queryIds.length !== queryCount) {
        throw new Error(`${id} queryIds needs one row per query`);
      }
    }
    if (views.onAttribute) {
      const {left, right} = views.onAttribute;
      validateSinglePackedView(left, ['uint32'], `${id} onAttribute.left`);
      validateSinglePackedView(right, ['uint32'], `${id} onAttribute.right`);
      if (left.length !== queryCount || right.length !== featureCount) {
        throw new Error(`${id} onAttribute needs one key per query and feature`);
      }
    }
  }

  /** Validates the neighbors-mode views and returns the query count. */
  private validateNeighbors(): number {
    const {id, views, neighborCount, neighborCapacity} = this;
    validateCapacity(id, 'neighborCount', neighborCount, 1, NEAREST_MAXIMUM_NEIGHBOR_CAPACITY);
    validateCapacity(
      id,
      'neighborCapacity',
      neighborCapacity,
      neighborCount,
      NEAREST_MAXIMUM_NEIGHBOR_CAPACITY
    );
    if (neighborCapacity > neighborCount && this.ties !== 'all') {
      throw new Error(`${id} neighborCapacity above neighborCount needs ties: 'all'`);
    }
    if (
      views.nearestFeatureIds ||
      views.nearestFeatureRows ||
      views.nearestDistances ||
      views.featureCounts ||
      views.matches
    ) {
      throw new Error(`${id} nearest-feature outputs need nearestFeatureIds`);
    }
    let queryCount: number;
    if (views.points) {
      for (const chunk of getGraphViewChunks(views.points)) {
        validateSinglePackedView(chunk, ['float32x2'], `${id} queries.positions`);
      }
      queryCount = views.points.length;
    } else {
      for (const view of getNearestGeometryViews(views.queries!)) {
        validateSinglePackedView(
          view,
          view.format === 'uint32' ? ['uint32'] : ['float32x2'],
          `${id} queries`
        );
      }
      queryCount = getNearestGeometryCount(views.queries!);
    }
    const slotCount = queryCount * neighborCapacity;
    for (const [name, view, length] of [
      ['neighborIds', views.neighborIds, slotCount],
      ['neighborRows', views.neighborRows, slotCount],
      ['neighborCounts', views.neighborCounts, queryCount],
      ['neighborDistances', views.neighborDistances, slotCount],
      ['neighborFootPoints', views.neighborFootPoints, slotCount],
      ['neighborSegmentIndices', views.neighborSegmentIndices, slotCount],
      ['neighborQueryPoints', views.neighborQueryPoints, slotCount]
    ] as const) {
      if (name === 'neighborCounts' && !view) {
        throw new Error(`${id} neighborIds needs neighborCounts`);
      }
      if (view) {
        validateSinglePackedView(
          view,
          name === 'neighborDistances'
            ? ['float32']
            : name === 'neighborFootPoints' || name === 'neighborQueryPoints'
              ? ['float32x2']
              : ['uint32'],
          `${id} output.${name}`
        );
        if (view.length !== length) {
          throw new Error(`${id} output.${name} needs ${length} rows`);
        }
      }
    }
    return queryCount;
  }

  /** Validates the nearest-feature-mode views and returns the point count. */
  private validateNearestFeature(): number {
    const {id, views} = this;
    const {features} = views;
    for (const port of [
      'neighborRows',
      'neighborCounts',
      'neighborDistances',
      'neighborFootPoints',
      'neighborSegmentIndices',
      'neighborQueryPoints'
    ] as const) {
      validateRequirement(id, !views[port], `output.${port} requires output.neighborIds`);
    }
    if (views.neighborCount !== undefined || views.neighborCapacity || views.ties) {
      throw new Error(`${id} neighborCount and ties need neighborIds`);
    }
    if (!views.nearestFeatureIds && !views.nearestFeatureRows) {
      throw new Error(`${id} needs nearestFeatureIds or nearestFeatureRows`);
    }
    if (!views.radius) {
      throw new Error(`${id} needs radius`);
    }
    const queryCount = views.points?.length ?? getNearestGeometryCount(views.queries!);
    if (views.queries) {
      validateSpatialJoinGeometry(id, 'queries', views.queries, {
        allowMultipart: true,
        allowEmpty: true
      });
    }
    if (features.kind !== 'segments' && this.traversalNearestFeature) {
      validateSpatialJoinGeometry(id, 'features', features, {
        allowMultipart: true,
        allowEmpty: true
      });
    }
    for (const chunk of views.points ? getGraphViewChunks(views.points) : []) {
      validateSinglePackedView(chunk, ['float32x2'], `${id} queries.positions`);
    }
    for (const [name, view] of [
      ['output.nearestFeatureIds', views.nearestFeatureIds],
      ['output.nearestFeatureRows', views.nearestFeatureRows],
      ['queries.sourceIds', views.sourceIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validateSinglePackedView(chunk, ['uint32'], `${id} ${name}`);
      }
      if (views.points) validateMatchingChunks(id, name, views.points, view);
      else if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        validateRequirement(
          id,
          view.length === queryCount,
          `${name} needs one row per query feature`
        );
      }
    }
    for (const chunk of views.nearestDistances ? getGraphViewChunks(views.nearestDistances) : []) {
      validateSinglePackedView(chunk, ['float32'], `${id} output.nearestDistances`);
    }
    if (views.points) {
      validateMatchingChunks(id, 'output.nearestDistances', views.points, views.nearestDistances);
    } else if (views.nearestDistances) {
      validateSinglePackedView(
        views.nearestDistances,
        ['float32'],
        `${id} output.nearestDistances`
      );
      validateRequirement(
        id,
        views.nearestDistances.length === queryCount,
        'output.nearestDistances needs one row per query feature'
      );
    }
    if (views.matches) {
      validateCompactOutput(id, views.matches);
    }
    if (!this.traversalNearestFeature) {
      validateCapacity(id, 'candidateCapacity', this.candidateCapacity);
    }
    return queryCount;
  }

  /** Resolved neighbors-mode inputs, shared by preflight and declaration. */
  private getNeighborNodeProps(): NearestNeighborNodeProps {
    const {views} = this;
    return {
      id: this.id,
      operation: OPERATION,
      queries: views.points
        ? {kind: 'points', points: views.points}
        : {kind: 'geometry', geometry: views.queries!},
      queryCount: this.queryCount,
      features: views.features,
      featureCount: this.featureCount,
      neighborCount: this.neighborCount,
      capacity: this.neighborCapacity,
      ties: this.ties,
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort,
      spatialSortCurve: views.spatialSortCurve,
      prepared: views.prepared,
      maxDistance: views.radius,
      featureIds: views.featureIds,
      neighborIds: views.neighborIds!,
      neighborRows: views.neighborRows,
      neighborCounts: views.neighborCounts!,
      neighborDistances: views.neighborDistances,
      neighborFootPoints: views.neighborFootPoints,
      neighborSegmentIndices: views.neighborSegmentIndices,
      neighborQueryPoints: views.neighborQueryPoints,
      exclusive: views.exclusive,
      queryIds: views.queryIds,
      onAttribute: views.onAttribute,
      overflow: views.overflow,
      sphereRadius: this.sphereRadius,
      querySort: this.querySort
    };
  }

  /**
   * Returns the command nodes: bounds, BVH, then either the candidate/distance/reduction passes of
   * nearest-feature mode or the traversal and column passes of neighbors mode.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {views} = this;
    validateGraphViewsBelongToGraph(this.id, graph, [...getInputs(views), ...getOutputs(views)]);
    validatePowerOfTwo(
      this.id,
      'leafCapacity',
      this.leafCapacity,
      1,
      getGPUNearestFeatureJoinMaximumLeafCapacity(graph.device)
    );
    if (this.mode === 'neighbors' || this.traversalNearestFeature) {
      // The traversal reads one packed node table: 5 words per BVH node (2 * leafCapacity - 1
      // nodes), 2 per query and, with a candidate filter, 2 per feature.
      validateMaximumCount(
        this.id,
        'node table words (leafCapacity, queries)',
        (2 * this.leafCapacity - 1) * NEAREST_NODE_WORDS +
          2 * this.queryCount +
          (views.exclusive || views.onAttribute ? 2 * this.featureCount : 0),
        getGPUNearestFeatureJoinMaximumNodeTableWordCount(graph.device)
      );
    }
    validateGraphChunkBindingRanges(
      graph,
      this.id,
      Object.fromEntries(
        [...getInputs(views), ...getOutputs(views)].map((view, index) => [
          `join view ${index}`,
          view
        ])
      )
    );
    for (const [name, bytes] of [
      ['join state and indirect dispatch', 16],
      ['BVH bounds and children', (2 * this.leafCapacity - 1) * 8],
      ['feature bounds and sorting scratch', this.featureCount * 8],
      ['query assignment and scan scratch', Math.max(this.queryCount, 1) * 4],
      ['candidate pair and geometry scratch', this.candidateCapacity * 8]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        this.id,
        name,
        bytes,
        'reduce leafCapacity, query count or candidateCapacity'
      );
    }
    if (views.prepared) {
      validateRequirement(
        this.id,
        views.prepared.isDeclaredIn(graph),
        'prepared handle must be added to the target graph before the join'
      );
      validateGraphChunkBindingRanges(graph, this.id, views.prepared.storage);
    }
    if (this.mode === 'neighbors') {
      validateNearestNeighborBindings(graph, this.getNeighborNodeProps());
    } else if (this.traversalNearestFeature) {
      // Preflight the traversal before declaring the nearest-feature adapter's own scratch.
      validateNearestNeighborBindings(graph, {
        id: this.id,
        operation: OPERATION,
        queries: views.points
          ? {kind: 'points', points: views.points}
          : {kind: 'geometry', geometry: views.queries!},
        queryCount: this.queryCount,
        features: views.features,
        featureCount: this.featureCount,
        neighborCount: 1,
        capacity: 1,
        leafCapacity: this.leafCapacity,
        spatialSort: this.spatialSort,
        querySort: this.querySort,
        prepared: views.prepared,
        exclusive: views.exclusive,
        onAttribute: views.onAttribute,
        neighborIds: views.overflow,
        neighborCounts: views.overflow,
        overflow: views.overflow
      });
    }
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
              id: `${this.id}-clear-${name}`,
              operation: OPERATION,
              view,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          );
        }
      }
      for (const [polygonIndex, polygons] of [
        views.queries,
        views.prepared?.getGeometry(graph) ?? views.features
      ]
        .filter(geometry => geometry?.kind === 'polygons')
        .entries()) {
        nodes.push(
          createLonLatPolygonCheckNode(graph, {
            id: `${this.id}-lon-lat-polygon-check-${polygonIndex}`,
            operation: OPERATION,
            positions: polygons.positions,
            ringOffsets: polygons.ringOffsets,
            status: {
              antimeridianUncertainCount: views.antimeridianUncertainCount,
              poleUncertainCount: views.poleUncertainCount
            }
          })
        );
      }
    }
    if (this.mode === 'neighbors') {
      const neighbors = getNearestNeighborNodes<Parameters>(graph, this.getNeighborNodeProps());
      nodes.push(
        ...neighbors.nodes,
        ...getLeafOverflowStatusNodes<Parameters>(graph, {
          id: this.id,
          operation: OPERATION,
          rowCount: this.queryCount,
          bvhOverflow: neighbors.bvh.overflow,
          candidateTruncated: views.candidateTruncated,
          candidateOverflow: views.candidateOverflow
        })
      );
      if (views.candidateCount) {
        nodes.push(
          createFillNode(graph, {
            id: `${this.id}-clear-candidate-count`,
            operation: OPERATION,
            view: views.candidateCount,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
    } else {
      nodes.push(
        ...(this.traversalNearestFeature
          ? this.getTraversalNearestFeatureNodes(graph)
          : this.getNearestFeatureNodes(graph))
      );
    }
    return nodes;
  }

  /**
   * Geometry-aware nearest-feature mode: the neighbors traversal with `k = 1` (ties to the
   * smallest feature row) followed by the nearest-feature outputs (IDs, distances, counts,
   * matches, overflow).
   */
  private getTraversalNearestFeatureNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, views, featureCount, queryCount} = this;
    const {matches} = views;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const featureRows = createTransientView(graph, `${id}-feature-rows`, 'uint32', queryCount);
    const rowCounts = createTransientView(graph, `${id}-row-counts`, 'uint32', queryCount);
    const rowDistances = createTransientView(graph, `${id}-row-distances`, 'float32', queryCount);
    const traversalOverflow = createTransientView(graph, `${id}-traversal-overflow`, 'uint32', 1);
    // Clears the state and the per-feature counts before the traversal fills the rows.
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear-state`,
        operation: OPERATION,
        state,
        assignment: featureRows,
        pointCount: queryCount,
        featureCounts: views.featureCounts
      })
    );
    const {nodes: traversalNodes, bvh: traversalBVH} = getNearestNeighborNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      queries: views.points
        ? {kind: 'points', points: views.points}
        : {kind: 'geometry', geometry: views.queries!},
      queryCount,
      features: views.features,
      featureCount,
      neighborCount: 1,
      capacity: 1,
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort,
      spatialSortCurve: views.spatialSortCurve,
      prepared: views.prepared,
      maxDistance: views.radius!,
      featureIds: views.featureIds,
      neighborIds: featureRows,
      neighborCounts: rowCounts,
      neighborDistances: rowDistances,
      exclusive: views.exclusive,
      queryIds: views.queryIds,
      onAttribute: views.onAttribute,
      overflow: traversalOverflow,
      sphereRadius: this.sphereRadius,
      writeFeatureRows: true,
      querySort: this.querySort
    });
    nodes.push(...traversalNodes);
    nodes.push(
      ...getLeafOverflowStatusNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        rowCount: queryCount,
        bvhOverflow: traversalBVH.overflow,
        candidateTruncated: views.candidateTruncated,
        candidateOverflow: views.candidateOverflow
      })
    );

    const distanceChunks = views.nearestDistances ? getGraphViewChunks(views.nearestDistances) : [];
    let chunkFirstRow = 0;
    const primaryOutput = views.nearestFeatureIds ?? views.nearestFeatureRows!;
    const rowChunks =
      views.nearestFeatureIds && views.nearestFeatureRows
        ? getGraphViewChunks(views.nearestFeatureRows)
        : [];
    for (const [chunkIndex, chunk] of getGraphViewChunks(primaryOutput).entries()) {
      if (chunk.length > 0) {
        const bindings: WGSLKernelBinding[] = [
          {
            name: 'featureRows',
            view: graph.createDataView(featureRows.buffer, {
              format: 'uint32',
              byteOffset: chunkFirstRow * 4,
              length: chunk.length
            }),
            type: 'u32',
            access: 'read'
          },
          {name: 'pointFeatureIds', view: chunk, type: 'u32', access: 'read_write'}
        ];
        const rowChunk = rowChunks[chunkIndex];
        if (rowChunk)
          bindings.push({
            name: 'pointFeatureRows',
            view: rowChunk,
            type: 'u32',
            access: 'read_write'
          });
        if (views.featureIds && views.nearestFeatureIds) {
          bindings.push({name: 'featureIds', view: views.featureIds, type: 'u32', access: 'read'});
        }
        if (views.featureCounts) {
          bindings.push({
            name: 'featureCounts',
            view: views.featureCounts,
            type: 'atomic<u32>',
            access: 'read_write'
          });
        }
        const distanceChunk = distanceChunks[chunkIndex];
        if (distanceChunk) {
          bindings.push(
            {
              name: 'rowDistances',
              view: graph.createDataView(rowDistances.buffer, {
                format: 'float32',
                byteOffset: chunkFirstRow * 4,
                length: chunk.length
              }),
              type: 'f32',
              access: 'read'
            },
            {name: 'distances', view: distanceChunk, type: 'f32', access: 'read_write'}
          );
        }
        nodes.push(
          createStorageTopKNode<Parameters>(graph, {
            id: getChunkNodeId(`${id}-assign`, primaryOutput, chunkIndex),
            operation: OPERATION,
            variant: 'assign-great-circle',
            bindings,
            invocationCount: chunk.length,
            declarations: `const NO_FEATURE: u32 = 0xffffffffu;`,
            body: `let row = index;
  let featureRow = featureRows[featureRowsOffset + row];
  let matched = featureRow != NO_FEATURE;
  var featureId = featureRow;
  ${views.featureIds && views.nearestFeatureIds ? 'if (matched) { featureId = featureIds[featureIdsOffset + featureRow]; }' : ''}
  pointFeatureIds[pointFeatureIdsOffset + index] = featureId;
  ${rowChunk ? 'pointFeatureRows[pointFeatureRowsOffset + index] = featureRow;' : ''}
  ${views.featureCounts ? 'if (matched) { atomicAdd(&featureCounts[featureCountsOffset + featureRow], 1u); }' : ''}
  ${distanceChunk ? 'distances[distancesOffset + index] = select(-1.0, rowDistances[rowDistancesOffset + row], matched);' : ''}`
          })
        );
      }
      chunkFirstRow += chunk.length;
    }
    if (matches) {
      nodes.push(
        ...getSpatialJoinCollectNodes<Parameters>(graph, {
          id: `${id}-collect-matches`,
          operation: OPERATION,
          // Collection uses only row topology, never coordinates. Geometry queries are packed
          // and featureRows has exactly one row per logical feature, unlike their vertex table.
          points: views.points ?? featureRows,
          assignment: featureRows,
          state,
          sourceIds: views.sourceIds,
          matches
        })
      );
    }
    nodes.push(
      ...createSpatialJoinFinalizeNodes<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        state,
        // The traversal clear pass already folds the BVH overflow into its overflow flag.
        bvhOverflow: traversalOverflow,
        candidateCapacity: 0,
        overflow: views.overflow,
        candidateCount: views.candidateCount,
        matches
      })
    );
    return nodes;
  }

  /** Nearest-feature mode: bounds, BVH, candidate, distance, reduction, assignment, finalize. */
  private getNearestFeatureNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, views, featureCount, leafCapacity, candidateCapacity} = this;
    const points = views.points!;
    const features = (
      views.prepared
        ? {...views.prepared.getGeometry(graph), sourceIds: views.features.sourceIds}
        : views.features
    ) as GPUNearestFeatureSource;
    const radius = views.radius!;
    const {matches} = views;
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const {bvh, nodes: bvhNodes} = getNearestBVHNodes(graph, {
      id,
      operation: OPERATION,
      features,
      featureCount,
      leafCapacity,
      spatialSort: this.spatialSort,
      spatialSortCurve: views.spatialSortCurve,
      prepared: views.prepared
    });
    // The probe pass only reads node bounds, leaf IDs and the internal node count.
    nodes.push(...bvhNodes);

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const assignment = createTransientView(graph, `${id}-assignment`, 'uint32', pointCount);
    const bestDistanceBits = createTransientView(
      graph,
      `${id}-best-distance-bits`,
      'uint32',
      pointCount
    );
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity
    );
    const candidatePoints = createTransientView(
      graph,
      `${id}-candidate-points`,
      'float32x2',
      candidateCapacity
    );
    const candidateStarts = createTransientView(
      graph,
      `${id}-candidate-starts`,
      'float32x2',
      candidateCapacity
    );
    const candidateEnds = createTransientView(
      graph,
      `${id}-candidate-ends`,
      'float32x2',
      candidateCapacity
    );
    const candidateSquaredDistances = createTransientView(
      graph,
      `${id}-candidate-squared-distances`,
      'float32',
      candidateCapacity
    );
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        state,
        assignment,
        pointCount,
        featureCounts: views.featureCounts,
        bestDistanceBits
      })
    );
    nodes.push(
      ...getSpatialJoinProbeNodes<Parameters>(graph, {
        id: `${id}-probe`,
        operation: OPERATION,
        points,
        bvh,
        featureCount,
        candidateCapacity,
        state,
        candidatePairs,
        candidatePoints,
        pointRowScale: 1,
        pointRowOffset: 0,
        radius,
        querySort: this.querySort,
        candidateTruncated: views.candidateTruncated,
        candidateOverflow: views.candidateOverflow,
        bvhOverflow: bvh.overflow
      })
    );
    const candidateDispatch = createWGSLActiveCountDispatch(graph, {
      id: `${id}-candidate-dispatch`,
      operation: OPERATION,
      count: state,
      maximumItemCount: candidateCapacity
    });
    nodes.push(candidateDispatch.updateNode);

    const expandBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'},
      {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
      {
        name: 'featureStarts',
        view: features.kind === 'points' ? features.positions : features.starts,
        type: 'f32',
        access: 'read'
      },
      {name: 'candidateStarts', view: candidateStarts, type: 'f32', access: 'read_write'},
      {name: 'candidateEnds', view: candidateEnds, type: 'f32', access: 'read_write'}
    ];
    if (features.kind === 'segments') {
      expandBindings.push({name: 'featureEnds', view: features.ends, type: 'f32', access: 'read'});
    }
    const endSource = features.kind === 'points' ? 'featureStarts' : 'featureEnds';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand`,
        operation: OPERATION,
        variant: `expand-${features.kind}`,
        bindings: expandBindings,
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  candidateStarts[candidateStartsOffset + index * 2u] = featureStarts[featureStartsOffset + featureRow * 2u];
  candidateStarts[candidateStartsOffset + index * 2u + 1u] = featureStarts[featureStartsOffset + featureRow * 2u + 1u];
  candidateEnds[candidateEndsOffset + index * 2u] = ${endSource}[${endSource}Offset + featureRow * 2u];
  candidateEnds[candidateEndsOffset + index * 2u + 1u] = ${endSource}[${endSource}Offset + featureRow * 2u + 1u];`
      })
    );
    // Keep the same robust squared-distance rank as neighbors mode through both reductions.
    const distanceBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'},
      {name: 'radius', view: radius, type: 'f32', access: 'read'},
      {name: 'candidatePoints', view: candidatePoints, type: 'f32', access: 'read'},
      {name: 'candidateStarts', view: candidateStarts, type: 'f32', access: 'read'},
      {name: 'candidateEnds', view: candidateEnds, type: 'f32', access: 'read'},
      {
        name: 'candidateSquaredDistances',
        view: candidateSquaredDistances,
        type: 'f32',
        access: 'read_write'
      }
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-distance`,
        operation: OPERATION,
        variant: `squared-distance-${features.kind}`,
        bindings: distanceBindings,
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `${NEAREST_COMMON_WGSL}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let point = vec2f(candidatePoints[candidatePointsOffset + index * 2u], candidatePoints[candidatePointsOffset + index * 2u + 1u]);
  let start = vec2f(candidateStarts[candidateStartsOffset + index * 2u], candidateStarts[candidateStartsOffset + index * 2u + 1u]);
  let end = vec2f(candidateEnds[candidateEndsOffset + index * 2u], candidateEnds[candidateEndsOffset + index * 2u + 1u]);
  if (!(isFiniteValue(point.x) && isFiniteValue(point.y) && isFiniteValue(start.x) && isFiniteValue(start.y) && isFiniteValue(end.x) && isFiniteValue(end.y))) {
    candidateSquaredDistances[candidateSquaredDistancesOffset + index] = NEAREST_INVALID_DISTANCE;
    return;
  }
  var rank = pointSegmentDistanceSq(point, start, end);
  let searchRadius = radius[radiusOffset];
  if (searchRadius >= 0.0 && isFiniteValue(searchRadius)) {
    let radiusRank = encodeNearestDistance(searchRadius);
    if (nearestRadiusNeedsCertification(rank, radiusRank)) {
      if (!pointWithinSegmentDistance(point, start, end, searchRadius)) { rank = NEAREST_INVALID_DISTANCE; }
      else { rank = minimumNearestDistance(rank, radiusRank); }
    }
  }
  candidateSquaredDistances[candidateSquaredDistancesOffset + index] = rank;`
      })
    );
    // Candidates failing `exclusive` or `onAttribute` never compete for the nearest feature.
    const filterBindings: WGSLKernelBinding[] = [];
    const filterTests: string[] = [];
    if (views.exclusive) {
      if (views.queryIds) {
        filterBindings.push({name: 'queryIds', view: views.queryIds, type: 'u32', access: 'read'});
      }
      if (views.featureIds) {
        filterBindings.push({
          name: 'featureIds',
          view: views.featureIds,
          type: 'u32',
          access: 'read'
        });
      }
      const queryId = views.queryIds ? 'queryIds[queryIdsOffset + pointRow]' : 'pointRow';
      const featureId = views.featureIds
        ? 'featureIds[featureIdsOffset + candidateFeature]'
        : 'candidateFeature';
      filterTests.push(`if (${queryId} == ${featureId}) { rejectCandidate(index); return; }`);
    }
    if (views.onAttribute) {
      filterBindings.push(
        {name: 'queryKeys', view: views.onAttribute.left, type: 'u32', access: 'read'},
        {name: 'featureKeys', view: views.onAttribute.right, type: 'u32', access: 'read'}
      );
      filterTests.push(
        'if (queryKeys[queryKeysOffset + pointRow] != featureKeys[featureKeysOffset + candidateFeature]) { rejectCandidate(index); return; }'
      );
    }
    if (filterTests.length) {
      // A separate pass gives filtered candidates the invalid rank, which reductions reject.
      // Folding the filter views into the reductions bound up to 10 storage buffers, above the
      // default limit of 8.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-filter`,
          operation: OPERATION,
          variant: 'filter',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'candidateSquaredDistances',
              view: candidateSquaredDistances,
              type: 'f32',
              access: 'read_write'
            },
            ...filterBindings
          ],
          invocationCount: candidateCapacity,
          condition: candidateDispatch.condition,
          declarations: `${NEAREST_DISTANCE_RANK_WGSL}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
fn rejectCandidate(index: u32) { candidateSquaredDistances[candidateSquaredDistancesOffset + index] = NEAREST_INVALID_DISTANCE; }`,
          body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let candidateFeature = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  ${filterTests.join('\n  ')}`
        })
      );
    }
    const reducePrologue = `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let distance = candidateSquaredDistances[candidateSquaredDistancesOffset + index];
  let searchRadius = radius[radiusOffset];
  let distanceBits = bitcast<u32>(distance);
  if (!(searchRadius >= 0.0 && searchRadius <= 3.402823466e+38) ||
      distanceBits >= NEAREST_OVERFLOW_DISTANCE ||
      !nearestDistanceAtMost(distance, encodeNearestDistance(searchRadius))) { return; }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce-distance`,
        operation: OPERATION,
        variant: 'reduce-distance',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {
            name: 'candidateSquaredDistances',
            view: candidateSquaredDistances,
            type: 'f32',
            access: 'read'
          },
          {name: 'radius', view: radius, type: 'f32', access: 'read'},
          {
            name: 'bestDistanceBits',
            view: bestDistanceBits,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `${NEAREST_DISTANCE_RANK_WGSL}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `${reducePrologue}
  atomicMin(&bestDistanceBits[bestDistanceBitsOffset + pointRow], distanceBits);`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce-feature`,
        operation: OPERATION,
        variant: 'reduce-feature',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {
            name: 'candidateSquaredDistances',
            view: candidateSquaredDistances,
            type: 'f32',
            access: 'read'
          },
          {name: 'radius', view: radius, type: 'f32', access: 'read'},
          {name: 'bestDistanceBits', view: bestDistanceBits, type: 'u32', access: 'read'},
          {name: 'assignment', view: assignment, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `${NEAREST_DISTANCE_RANK_WGSL}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `${reducePrologue}
  if (distanceBits == bestDistanceBits[bestDistanceBitsOffset + pointRow]) {
    let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    atomicMin(&assignment[assignmentOffset + pointRow], featureRow);
  }`
      })
    );
    nodes.push(
      ...getSpatialJoinAssignNodes<Parameters>(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        assignment,
        pointFeatureIds: views.nearestFeatureIds,
        pointFeatureRows: views.nearestFeatureRows,
        featureIds: views.featureIds,
        featureCounts: views.featureCounts,
        bestDistanceBits,
        distanceRankWGSL: NEAREST_DISTANCE_RANK_WGSL,
        distances: views.nearestDistances
      })
    );
    if (matches) {
      nodes.push(
        ...getSpatialJoinCollectNodes<Parameters>(graph, {
          id: `${id}-collect-matches`,
          operation: OPERATION,
          points,
          assignment,
          state,
          sourceIds: views.sourceIds,
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
        overflow: views.overflow,
        candidateCount: views.candidateCount,
        matches
      })
    );
    return nodes;
  }
}
