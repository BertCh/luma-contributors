// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {getGPUGreatCircleRadius} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createTiledWGSLKernelNodes} from '../../analysis-core/kernels/wgsl-kernel-tiles';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUGridIndexMaximumCellCount,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getBVHNearestNeighborSearchWGSL,
  getBVHNeighborSearchDeclarationsWGSL,
  getBVHRadiusNeighborLoopWGSL,
  getBVHThreadToRowWGSL,
  getGeodesicNearestNeighborSearchWGSL,
  getGeodesicRadiusNeighborLoopWGSL,
  getInsertNeighborWGSL,
  getNearestNeighborSearchWGSL,
  getNeighborSearchLatticeWGSL,
  getRadiusNeighborLoopWGSL,
  NEIGHBOR_SEARCH_BVH_NODE_WORDS,
  NEIGHBOR_WEIGHT_WGSL,
  type NeighborSearchTimeAxis
} from './neighbor-search-kernels';
import {getNearestBVHNodes} from '../spatial-join/nearest-nodes';
import {getNextPowerOfTwo} from '../spatial-join/spatial-join-passes';
import {
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
} from './neighbor-search-parameters';
import {
  createPointNeighborGridIndexNodes,
  POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL
} from '../shared/point-neighbor-grid-index';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  type GPUSpatialWeightsTransformState,
  validateGPUSpatialWeightsOutput
} from '../spatial-weights/spatial-weights';
import {GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON} from '../spatial-weights/gpu-spatial-weights-transform';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/index';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';

const OPERATION = 'GPUNeighborSearch';

/** Planar or great-circle contexts suitable for neighbor distance measurements. */
export const GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metricsByCoordinateSpace: Object.freeze({
    planar: Object.freeze(['none', 'native'] as const),
    'longitude-latitude': Object.freeze(['great-circle'] as const)
  }),
  // Measuring operators reject planar 'none' in resolveGPUSpatialContext.
  measure: 'distance'
});

/** Weight transform states a caller can declare for {@link GPUNeighborSearchProps.weightTransform}. */
export type GPUNeighborSearchWeightTransform = Extract<
  GPUSpatialWeightsTransformState,
  'binary' | 'raw' | 'kernel' | 'row'
>;

/**
 * Search structure of {@link GPUNeighborSearch}: `'grid'` buckets targets into a cell lattice,
 * `'bvh'` traverses a Hilbert-packed bounding-volume hierarchy over the valid targets, and `'auto'`
 * uses the grid. The BVH remains an explicit opt-in until a measured workload justifies selection.
 */
export type GPUNeighborSearchStructure = 'auto' | 'grid' | 'bvh';

/**
 * Largest supported compile-time `k`: the per-query private `(distance, id)` list is two
 * `array<_, k>` of function-scope memory, 2 KiB at this value, well inside the 8 KiB a WGSL
 * implementation must support for one function. `rows * k * 4` bytes must also fit one storage
 * binding (checked when the nodes are created).
 */
export const GPU_NEIGHBOR_SEARCH_MAXIMUM_K = 256;

/**
 * Largest {@link GPUNeighborSearchProps.timeBucketCount}. A bound on the extra cell count; the
 * product with the grid is also checked against the index cell limit.
 */
export const GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS = 65536;

/** Space-time search reads relative float32 times, such as the output of `GPUTimeWordsRelative`. */
const NEIGHBOR_SEARCH_TEMPORAL_CONTEXT_SUPPORT = Object.freeze({
  encodings: Object.freeze(['float32-relative'] as const)
});

/**
 * Caller-owned outputs of {@link GPUNeighborSearch}: the CSR with its status scalars, plus an
 * optional unclamped neighbor count per query row.
 */
export type GPUNeighborSearchOutput = GPUSpatialWeightsOutput & {
  /** Optional unclamped neighbor count per query row. */
  neighborCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUNeighborSearch}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `queryPositions`, `mask`,
 * `queryMask` and `parameters` (bounds, radius, weight function, row standardization).
 * Compile-time: `mode`, `k`, `gridSize`, `searchStructure`, row counts, capacity and which optional
 * views exist.
 */
export type GPUNeighborSearchProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'neighbor-search'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Compile-time. Omitted means native planar
   * coordinates with Euclidean distance (unchanged behavior). The great-circle path is
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units?: 'meters' | 'kilometers',
   * sphereRadius?}`: positions are `[longitude, latitude]` degrees, `bounds` are
   * `[minLongitude, minLatitude, maxLongitude, maxLatitude]` degrees (longitude span at most 360),
   * `radius`, `distanceFloor` and the output `distances` are great-circle distances in `units`
   * (default meters), and `sphereRadius` is in meters (default `GPU_EARTH_MEAN_RADIUS_METERS`).
   * Any other space or metric throws; project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /** `'knn'`: the `k` nearest targets per query. `'radius'`: every target within `radius`. */
  mode: 'knn' | 'radius';
  /** Packed planar target points. At least one and fewer than 2^31 rows. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional packed planar query points (a cross join). When omitted the search is a self join of
   * `positions` that never lists a row as its own neighbor.
   */
  queryPositions?: GraphDataView<'float32x2'>;
  /**
   * Optional relative time of each target, which turns the search into a space-time cylinder search:
   * a target is a neighbor only when it also satisfies `|t_target - t_query| <= timeRadius`
   * (per-frame, closed like the spatial test). kNN then means the `k` nearest targets inside the
   * time window. A target or self-join row whose time is non-finite or outside the per-frame
   * `timeBounds` is invalid like a row outside `bounds`: it is empty, never a neighbor, and counted
   * by `invalidCount`. A cross-join query needs only a finite time: outside
   * `timeBounds` it still finds its exact neighbors.
   * Requires `temporalContext` and parameters of
   * {@link GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH} words. Great-circle search takes times;
   * `searchStructure: 'bvh'` does not. Compile-time presence; per-frame contents.
   */
  times?: GraphDataView<'float32'>;
  /** Relative time of each query row. Required with `times` in a cross join, rejected otherwise. */
  queryTimes?: GraphDataView<'float32'>;
  /**
   * Describes `times` and `queryTimes`: required with `times`, rejected without it. Only
   * `'float32-relative'` times are read; any unit, epoch (a bigint origin), time zone and axis.
   * `unit` is the unit of `timeRadius` and `timeBounds`. Nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /**
   * Time buckets of the index. Compile-time, default 1. Results never depend on it, only speed: at 1
   * every target of a spatial cell is tested against the time window, and above 1 the targets are
   * also indexed by time, so a window much shorter than `timeBounds` skips most of them. Multiplies
   * the cell count (`gridSize` cells times this is at most
   * {@link getGPUGridIndexMaximumCellCount}). Requires `times`; at most
   * {@link GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS}.
   */
  timeBucketCount?: number;
  /** Optional target selection: nonzero includes the target. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional query selection: nonzero includes the query row; excluded rows are empty. A self join
   * without `queryMask` uses `mask` for the query rows too.
   */
  queryMask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: packed float32 view of at least `GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH`
   * elements written with `getGPUNeighborSearchParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the cell lattice. Compile-time. Results never depend on it, only
   * speed. kNN mode spreads exactly this many cells over the bounds. `columns * rows` is at most
   * {@link getGPUGridIndexMaximumCellCount} (2^25 - 1 on a default device).
   */
  gridSize: readonly [number, number];
  /**
   * Search structure. Compile-time; results (IDs, offsets, distances, weights) never depend on it,
   * only speed. Default `'auto'`.
   *
   * - `'grid'`: the cell lattice only. Fastest on evenly spread points; on clustered points one
   *   cell holds most targets and each query degrades toward a scan of them.
   * - `'bvh'`: a Hilbert-packed point BVH only (near-first depth-first traversal with box pruning). Planar positions only; longitude-latitude throws.
   * - `'auto'`: uses exactly the grid graph, with no BVH allocation, build or occupancy pass.
   *   The 200k-point clustered and uniform kNN/radius benchmarks all favor the grid; occupancy
   *   alone did not predict traversal cost. Great-circle search also always uses the grid.
   *
   * The `gridSize` shapes the grid path.
   */
  searchStructure?: GPUNeighborSearchStructure;
  /**
   * Rows per dispatch for the search, count, emit and sort stages. Tiles keep global row indices,
   * so results never depend on this value; it only bounds the work of one dispatch (GPU watchdog).
   */
  rowTileSize?: number;
  /**
   * Neighbors per query in kNN mode, an integer in `[1, GPU_NEIGHBOR_SEARCH_MAXIMUM_K]` (256).
   * Compile-time. Ignored in radius mode.
   */
  k?: number;
  /**
   * Caller-owned output CSR: `offsets` has `queryRows + 1` entries; the slot capacity is
   * `neighbors.length`. `distances` is optional. Row `i` lists neighbor target IDs ascending.
   * `status.overflow` is 1 when the neighbors did not fit the capacity, else 0.
   * `status.requiredCount` is the optional unclamped total neighbor count. `status.invalidCount` is
   * the optional count of rows that took part by mask but were dropped because a coordinate is not
   * finite or lies outside the per-frame `bounds` (or, for great-circle search, has a latitude
   * beyond +-90). Such rows are silently empty and never a neighbor, so a nonzero count means
   * `bounds` is too small or the data has missing coordinates. It counts target rows, plus the
   * query rows of a cross join (`queryPositions`). A cross-join query is invalid only when
   * its coordinates (or time) are not finite, or its great-circle latitude is beyond +-90;
   * `bounds` and `timeBounds` do not exclude it. Queries outside them get the same neighbors
   * as a brute-force scan, including across the longitude seam.
   */
  output: GPUNeighborSearchOutput;
  /**
   * Declares the weight function the per-frame parameters will select, so `outputWeights.information` can
   * say what the weights are. Weight kind and row standardization are per-frame parameters that
   * the node cannot read at construction, so this is metadata the caller keeps true (derive it from
   * the same parameters with `getGPUNeighborSearchWeightTransform`). Omitted means `'unknown'`.
   */
  weightTransform?: GPUNeighborSearchWeightTransform;
  /**
   * Compile-time row standardization. When given, it is the single source of truth: the weights
   * kernel uses it as a constant and ignores the per-frame `rowStandardize` parameter, and
   * `outputWeights.information.transform` is `'row'` for `true`, else `weightTransform` (or `'unknown'`).
   * Consumers re-standardize rows over the neighbors left after masking only when the transform is
   * `'row'`, so set this (rather than the per-frame field) for any row-standardized band. Passing
   * `true` with a `weightTransform` other than `'row'`, or `false` with `'row'`, throws. When
   * omitted, the per-frame `rowStandardize` parameter applies and `weightTransform` is the
   * caller's declaration.
   */
  rowStandardize?: boolean;
};

/**
 * Exact k-nearest-neighbor and distance-band queries between points, written as a
 * {@link GPUSpatialWeights} CSR: the reusable spatial-weights structure of the statistics contributors.
 * Points are planar (default) or longitude/latitude degrees with great-circle distances; see
 * {@link GPUNeighborSearchProps.spatialContext}.
 *
 * Definition, which the GPU result matches exactly (IDs and offsets) or within f32 rounding
 * (distances, weights):
 * - A target is valid when its mask is nonzero, its coordinates are finite and it lies inside the
 *   per-frame bounds. A self join skips the row itself and its query rows are those targets. A
 *   cross-join query (`queryPositions`, with `queryMask`) is valid when its coordinates are finite;
 *   a query outside the bounds is searched exactly. Cell lookups clamp to the nearest edge cell;
 *   great-circle cell ranges use the equivalent longitude nearest the lattice midpoint and
 *   require latitude within +-90. Spherical distance tests reduce both stored longitudes modulo
 *   360 with integer significand arithmetic, independently of the lattice-relative shift, so even
 *   very large finite longitudes and tiny offsets retain their stored angular position.
 *   Cost: a planar query far outside the bounds (beyond a side or a corner by many lattice
 *   widths) cannot stop its kNN ring walk early, because the clearance of the sides across from it
 *   stays small, so it visits every lattice cell in one invocation (cell tests only; target loads
 *   are still pruned). Measured with 200,000 targets in a 512 x 512 lattice, k = 8, 2,000 queries:
 *   about 7.5 ms with the queries inside, about 25 ms with them 1e6 beyond a side or a corner.
 *   Keep queries within a few lattice widths of the bounds when that matters.
 * - When every squared distance of a query overflows float32 (offsets beyond about 1.8e19 give
 *   infinity), all targets tie at infinity: kNN returns the lowest target IDs, the same as the
 *   `(distance, id)` order of the scan, on both structures. Distances written for such rows are
 *   infinite.
 * - Planar distances are Euclidean, `d^2 = dx^2 + dy^2` in f32 with `dx = target - query`. Planar
 *   f32 coordinates keep about 7 significant digits, so UTM northings (~4e6 m, ulp 0.5 m) or Web
 *   Mercator coordinates (~1e7, ulp 1 m) quantize `dx` and make ties and band edges noisy. Project
 *   to a local frame (`createGPULocalFrame`) or subtract an origin before upload; the node cannot
 *   recover precision lost in its f32 inputs.
 * - Great-circle distances (`spatialContext` longitude-latitude) are haversine/central-angle
 *   distances on a sphere (`GEODESIC_WGSL`, f32, the inputs' own f32 quantization is the limit:
 *   about 1 m at longitude 180) in `units`, and `radius` is in `units`. The lattice stays in
 *   degrees. Radius mode visits every cell that can hold a target in the cap: the latitude span is
 *   `radius / R`, the longitude span is the exact cap extent `asin(sin(a) / cos(latitude))` (the
 *   whole circle when the cap reaches a pole) and bands wrap across +-180. kNN visits expanding
 *   rings (wrapping columns when `bounds` span exactly 360 degrees) and stops only when a
 *   conservative spherical lower bound of the distance to every unvisited cell exceeds the K-th
 *   distance, so results do not depend on `gridSize`; a looser bound only costs speed (queries near
 *   the poles or narrow longitude spans scan more cells). Points with latitude beyond +-90 are invalid.
 *   See "Precision of longitude/latitude positions" for the 10·Q ≈ 17 m distance warning threshold.
 * - Point and query rows that are not valid never take part. Report them with `invalidCount`.
 * - kNN: the `k` valid targets with the smallest `(d^2, id)` (ties go to the lowest ID), optionally
 *   only those with `d <= radius`. Fewer valid targets give a shorter row.
 * - Radius: every valid target with `d <= radius`.
 * - Each row lists its neighbors by ascending ID, with `distances` and `weights` aligned.
 * - Weights: binary `1`; inverse distance `max(d, distanceFloor)^-power` (default floor `1e-6`, as
 *   in `GPUNearestFeatureWeights`, so coincident points weigh `1e6^power`); or a kernel `K(d / h)`
 *   with bandwidth `h = radius` (radius mode) or the row's k-th neighbor distance (kNN, PySAL's
 *   adaptive bandwidth, `h_i` = the row's k-th neighbor distance times 1.0000001 =
 *   `GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON`, as libpysal `Kernel(fixed=False)` after
 *   removing the self slot; a row with fewer than `k` neighbors uses its farthest listed neighbor).
 *   Non-finite weights are written as 0. Row standardization divides by the row sum (summed in
 *   slot order). Rows never list themselves: the kernel's self weight `K(0)` is the consumer's
 *   `selfWeight` (`getGPUSpatialWeightsKernelValue(kernel, 0)`). A slot whose weight is 0 (for
 *   example a triangular kernel at `d = radius`, or coincident points with inverse distance and an
 *   explicit `distanceFloor: 0`) is NOT a neighbor under the stream-wide definition: consumers skip zero-weight slots.
 * - Coincident points under the default floor. Two distinct rows at the same position weigh
 *   `1e6^power` to each other, far more than any neighbor at an ordinary distance (a neighbor
 *   100 units away weighs 0.01 at power 1). With `rowStandardize` the coincident neighbor
 *   therefore takes nearly all of the row (1e6 against 0.01 is a share of 0.99999999), and a
 *   spatial lag of such a row is in effect the value of the duplicate. Data with repeated
 *   positions (geocoded addresses, events on one site) should pass a `distanceFloor` on the scale
 *   of the data, such as the typical nearest-neighbor spacing, or `distanceFloor: 0` to leave
 *   duplicates out of each other's rows.
 * - `outputWeights` is `output` with `information` (`rowCount`, `transform` from `weightTransform` or
 *   `'unknown'`; `'row'` when `rowStandardize: true`, `symmetric` true only for an unmasked-query
 *   planar radius self join with a declared unstandardized transform, else `'unknown'`, `provenance` criterion `'knn'` or `'distance-band'`
 *   with `k`) and `status` (`overflow`, `requiredCount`, `invalidCount`) attached.
 *   `'adaptive'` kernel transforms accept it because kNN output has criterion `'knn'`. For PySAL's
 *   `min_threshold_distance` (the smallest band keeping every unit connected), run kNN with `k = 1`
 *   and take the maximum of the returned `distances`.
 * - Capacity: rows are laid out in query order; slots past `neighbors.length` are dropped. Offsets
 *   are clamped to the capacity, so a reader always sees consistent (possibly truncated) rows,
 *   and `status.overflow` is set. `status.requiredCount` and `neighborCounts` stay unclamped.
 *
 * Space-time search: with `times` the relation gains `|t_target - t_query| <= timeRadius`; see
 * {@link GPUNeighborSearchProps.times}. A kNN query whose window holds fewer than `k` targets
 * finishes every ring and so scans its whole grid; set a `radius` when windows may be sparse.
 *
 * Algorithm: `searchStructure` selects between the grid below and a point BVH (see
 * {@link GPUNeighborSearchProps.searchStructure}); both produce the identical definition above. The
 * BVH path builds a Hilbert-packed BVH over the valid targets (invalid targets get empty boxes),
 * walks it near-first depth-first per query with the same `(d^2, id)` top-k list and radius limit, and then
 * shares the grid path's scan, offsets, emit and sort-by-ID stages. Grid path: targets are bucketed by `GPUGridIndex` into a per-frame lattice (cells at least
 * `radius` wide in radius mode; `gridSize` cells over the bounds in kNN mode), whose domain is
 * read from a buffer so changing bounds or radius never recompiles. kNN keeps a private sorted
 * top-k list per query and searches expanding cell rings until the k-th distance is strictly
 * inside the visited box. Radius mode counts, scans, emits and then insertion-sorts each row by
 * ID. Order within a grid cell is unspecified, but every row is sorted by ID and no float atomics
 * are used, so results are bitwise reproducible, except which neighbors the one row cut by an
 * exceeded capacity keeps.
 */
export class GPUNeighborSearch implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNeighborSearchProps;
  /** Resolved coordinate space and metric (native planar when `spatialContext` was omitted). */
  readonly spatialContext: GPUSpatialContext;
  /** The output CSR with `info` and `status` attached, for consumers. */
  readonly outputWeights: GPUSpatialWeights;
  /**
   * Requested search structure: `props.searchStructure` (default `'auto'`), or `'grid'` for
   * great-circle search. `'auto'` resolves to the grid before allocating either structure.
   */
  readonly searchStructure: GPUNeighborSearchStructure;
  /** Resolved context of `times`, else undefined. */
  readonly temporalContext?: GPUTemporalContext;

  constructor(props: GPUNeighborSearchProps) {
    const id = props.id ?? 'neighbor-search';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT
    );
    if (props.mode !== 'knn' && props.mode !== 'radius') {
      throw new Error(`${id} mode must be 'knn' or 'radius'`);
    }
    const searchStructure = props.searchStructure ?? 'auto';
    if (!['auto', 'grid', 'bvh'].includes(searchStructure)) {
      throw new Error(`${id} searchStructure must be 'auto', 'grid' or 'bvh'`);
    }
    if (searchStructure === 'bvh' && this.spatialContext.coordinateSpace === 'longitude-latitude') {
      throw new Error(`${id} searchStructure 'bvh' supports planar positions only`);
    }
    this.searchStructure =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'grid' : searchStructure;
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    const targetRows = props.positions.length;
    validateCapacity(id, 'positions length', targetRows, 1, 2 ** 31 - 1);
    if (props.queryPositions) {
      validateSinglePackedView(props.queryPositions, ['float32x2'], `${id} queryPositions`);
      validateCapacity(id, 'queryPositions length', props.queryPositions.length, 1, 2 ** 31 - 1);
    }
    const queryRows = this.getQueryRows();
    if (props.times) {
      validateRequirement(id, props.temporalContext !== undefined, 'times need a temporalContext');
      this.temporalContext = resolveGPUTemporalContext(
        id,
        props.temporalContext,
        NEIGHBOR_SEARCH_TEMPORAL_CONTEXT_SUPPORT
      );
      validateRequirement(
        id,
        this.searchStructure !== 'bvh',
        "times need searchStructure 'grid' or 'auto'"
      );
      validateSinglePackedView(props.times, ['float32'], `${id} times`);
      validateRequirement(
        id,
        props.times.length === targetRows,
        'times length must equal positions length'
      );
      if (props.queryPositions) {
        validateRequirement(
          id,
          props.queryTimes !== undefined,
          'queryTimes are required with times in a cross join'
        );
        validateSinglePackedView(props.queryTimes!, ['float32'], `${id} queryTimes`);
        validateRequirement(
          id,
          props.queryTimes!.length === queryRows,
          'queryTimes length must equal queryPositions length'
        );
      } else {
        validateRequirement(id, props.queryTimes === undefined, 'queryTimes need queryPositions');
      }
      validateCapacity(
        id,
        'timeBucketCount',
        props.timeBucketCount ?? 1,
        1,
        GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS
      );
    } else {
      validateRequirement(
        id,
        props.temporalContext === undefined &&
          props.queryTimes === undefined &&
          props.timeBucketCount === undefined,
        'temporalContext, queryTimes and timeBucketCount need times'
      );
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = props.times
      ? GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
      : GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH;
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    const [columns, rows] = props.gridSize;
    validateCapacity(id, 'gridSize columns', columns, 1, Number.MAX_SAFE_INTEGER);
    validateCapacity(id, 'gridSize rows', rows, 1, Number.MAX_SAFE_INTEGER);
    if (props.mode === 'knn') {
      validateCapacity(id, 'k', props.k ?? Number.NaN, 1, GPU_NEIGHBOR_SEARCH_MAXIMUM_K);
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== targetRows) {
        throw new Error(`${id} mask length must equal positions length`);
      }
    }
    if (props.queryMask) {
      validateSinglePackedView(props.queryMask, ['uint32'], `${id} queryMask`);
      if (props.queryMask.length !== queryRows) {
        throw new Error(`${id} queryMask length must equal the query row count`);
      }
    }
    const {output} = props;
    const weightRows = validateGPUSpatialWeightsOutput(id, output);
    if (weightRows !== queryRows) {
      throw new Error(`${id} output.offsets length must equal the query row count + 1`);
    }
    const {requiredCount, invalidCount} = output.status;
    for (const [name, view] of [
      ['output.status.overflow', output.status.overflow],
      ['output.status.requiredCount', requiredCount],
      ['output.status.invalidCount', invalidCount]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must hold one uint32`);
        }
      }
    }
    if (output.neighborCounts) {
      validateSinglePackedView(output.neighborCounts, ['uint32'], `${id} output.neighborCounts`);
      if (output.neighborCounts.length !== queryRows) {
        throw new Error(`${id} output.neighborCounts length must equal the query row count`);
      }
    }
    if (
      props.weightTransform !== undefined &&
      !['binary', 'raw', 'kernel', 'row'].includes(props.weightTransform)
    ) {
      throw new Error(`${id} weightTransform must be 'binary', 'raw', 'kernel' or 'row'`);
    }
    if (
      props.rowStandardize !== undefined &&
      props.weightTransform !== undefined &&
      props.rowStandardize !== (props.weightTransform === 'row')
    ) {
      throw new Error(
        `${id} rowStandardize ${props.rowStandardize} conflicts with weightTransform '${props.weightTransform}'`
      );
    }
    const transform = props.rowStandardize ? 'row' : (props.weightTransform ?? 'unknown');
    const planarSelfJoin =
      this.spatialContext.coordinateSpace === 'planar' &&
      props.mode === 'radius' &&
      !props.queryPositions &&
      !props.queryMask;
    this.outputWeights = describeGPUSpatialWeights(output, {
      spatialContext: this.spatialContext,
      information: {
        rowCount: queryRows,
        transform,
        symmetric:
          planarSelfJoin && transform !== 'unknown' && transform !== 'row' ? true : 'unknown',
        hasSelfWeights: false,
        provenance: {
          source: OPERATION,
          criterion: props.mode === 'knn' ? 'knn' : 'distance-band',
          k: props.mode === 'knn' ? props.k : undefined
        }
      }
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        invalidCount,
        output.offsets,
        output.neighbors,
        output.weights,
        output.distances,
        output.status.overflow,
        requiredCount,
        output.neighborCounts
      ],
      [
        props.positions,
        props.queryPositions,
        props.mask,
        props.queryMask,
        props.parameters,
        props.times,
        props.queryTimes
      ]
    );
  }

  /** Number of query rows: `queryPositions.length`, or `positions.length` for a self join. */
  getQueryRows(): number {
    return (this.props.queryPositions ?? this.props.positions).length;
  }

  /** Returns the neighbor-search nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, queryPositions, parameters, gridSize, output: weights, mode} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      queryPositions,
      props.mask,
      props.queryMask,
      parameters,
      props.times,
      props.queryTimes,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      weights.status.overflow,
      weights.status.requiredCount,
      weights.status.invalidCount,
      weights.neighborCounts
    ]);
    const {times} = props;
    const timeBucketCount = times ? (props.timeBucketCount ?? 1) : 1;
    const timeAxis: NeighborSearchTimeAxis | undefined = times
      ? {bucketCount: timeBucketCount}
      : undefined;
    const bucketed = timeBucketCount > 1;
    const cellCount = gridSize[0] * gridSize[1] * timeBucketCount;
    validateMaximumCount(
      id,
      'gridSize cells',
      cellCount,
      getGPUGridIndexMaximumCellCount(graph.device)
    );
    const targetRows = positions.length;
    const queryRows = this.getQueryRows();
    const capacity = weights.neighbors.length;
    const k = props.k ?? 1;
    if (mode === 'knn') {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'k-nearest lists',
        queryRows * k * 4,
        'lower k or the query row count'
      );
    }
    const {spatialContext} = this;
    const geodesic = spatialContext.coordinateSpace === 'longitude-latitude';
    // Sphere radius in the context's length unit.
    const sphereRadius = getGPUGreatCircleRadius(spatialContext);
    const latticeWGSL = `${getNeighborSearchLatticeWGSL(mode, gridSize, sphereRadius, timeAxis)}
${POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL}
${
  timeAxis
    ? `fn getGridUnitTime(lattice: Lattice, time: f32) -> f32 {
  // The grid drops a coordinate above 1, and float32 rounding can put the latest time there.
  return min(1.0, (time - lattice.minimumTime) / (lattice.timeBucketWidth * f32(TIME_BUCKETS)));
}`
    : ''
}`;
    const radiusLoop = geodesic ? getGeodesicRadiusNeighborLoopWGSL : getRadiusNeighborLoopWGSL;
    const crossJoin = Boolean(queryPositions);
    const queryBinding: WGSLKernelBinding[] = queryPositions
      ? [{name: 'queryPositions', view: queryPositions, type: 'f32', access: 'read'}]
      : [];
    // A target or query record carries its time as a fourth word.
    const recordWords = times ? 4 : 3;
    const queryTimes = props.queryTimes ?? times;
    const timeBindings = (name: 'times' | 'queryTimes'): WGSLKernelBinding[] => {
      const view = name === 'times' ? times : queryTimes;
      return view ? [{name, view, type: 'f32', access: 'read'}] : [];
    };
    const readTimeWGSL = (name: 'times' | 'queryTimes', row: string) =>
      `${name}[${name}Offset + ${row}]`;
    const queryTimeName: 'times' | 'queryTimes' = crossJoin ? 'queryTimes' : 'times';
    const queryTimeBinding = timeBindings(queryTimeName);
    // A cross-join query needs finite coordinates and time (and a spherical latitude); self-join rows
    // are its target rows, which stay bound by the bounds.
    const queryPointInBoundsWGSL = crossJoin ? 'isCrossJoinQueryInBounds' : 'isPointInBounds';
    const queryPointValidWGSL = crossJoin ? 'isCrossJoinQueryValid' : 'isPointValid';
    const queryTimeInBoundsWGSL = (row: string) =>
      crossJoin
        ? `isCrossJoinQueryTimeInBounds(${readTimeWGSL(queryTimeName, row)})`
        : `isTimeInBounds(lattice, ${readTimeWGSL(queryTimeName, row)})`;
    const queryTimeValidWGSL = (row: string) => (times ? ` && ${queryTimeInBoundsWGSL(row)}` : '');
    const queryName = crossJoin ? 'queryPositions' : 'positions';
    const readQueryWGSL = (row: string) => `let x = ${queryName}[${queryName}Offset + ${row} * 2u];
  let y = ${queryName}[${queryName}Offset + ${row} * 2u + 1u];`;
    const selfCondition = crossJoin ? 'true' : 'neighbor != queryIndex';
    // A self join applies the target mask to the query rows too unless a query mask is given.
    const queryMask = props.queryMask ?? (crossJoin ? undefined : props.mask);
    // Threads of the search kernels walk the valid queries in grid-cell order, so neighboring
    // threads search neighboring cells (coalesced reads, coherent control flow) however the
    // input is ordered. The thread's query row (`row`) is where its results are written, so
    // outputs are identical to launching in row order. A self join whose query set equals the
    // target set reuses the target index; otherwise the queries get an index of their own.
    const sharesTargetIndex = !crossJoin && !props.queryMask;

    // Resolve before allocating/building: the measured clustered and uniform workloads favor
    // the grid. An unused BVH must not add a sort/build to every auto encoding.
    const useBvh = this.searchStructure === 'bvh';
    const useGrid = !useBvh;
    // Grid-only scratch is one element when the grid is not used.
    const gridSized = (length: number) => (useGrid ? length : 1);

    // Time is a third lattice dimension only when the time axis is bucketed.
    const gridDimension = bucketed ? 3 : 2;
    const gridFormat = bucketed ? ('float32x3' as const) : ('float32x2' as const);
    // Typed as two components because the BVH path, which never buckets, reads it that way.
    const gridPositions = createTransientView(
      graph,
      `${id}-grid-positions`,
      gridFormat,
      targetRows
    ) as GraphDataView<'float32x2'>;
    const fullGridSize = bucketed
      ? ([gridSize[0], gridSize[1], timeBucketCount] as const)
      : ([gridSize[0], gridSize[1]] as const);
    const targetIndex = useGrid
      ? createPointNeighborGridIndexNodes<Parameters>(graph, {
          id,
          positions: gridPositions,
          gridSize: fullGridSize,
          rowCount: targetRows
        })
      : undefined;
    const cellOffsets =
      targetIndex?.cellOffsets ?? createTransientView(graph, `${id}-cell-offsets`, 'uint32', 1);
    const sortedRows =
      targetIndex?.sortedRows ?? createTransientView(graph, `${id}-sorted-rows`, 'uint32', 1);
    const gridCount =
      targetIndex?.gridCount ?? createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
    // Target `[x, y, row]` records (coordinates as float bits; with times a fourth time word) in
    // cell order: the search loops read them sequentially instead of gathering, and one binding
    // carries every field.
    const sortedTargets = createTransientView(
      graph,
      `${id}-sorted-targets`,
      'uint32',
      gridSized(targetRows * recordWords)
    );
    const counts =
      weights.neighborCounts ?? createTransientView(graph, `${id}-counts`, 'uint32', queryRows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', queryRows);
    const distances =
      weights.distances ?? createTransientView(graph, `${id}-distances`, 'float32', capacity);
    // Replaced by the query index below when the queries are not the targets.
    let queryOrder: GraphDataView<'uint32'> = sortedRows;
    // Valid query records in cell order; a shared index reuses the target records.
    const sortedQueries = sharesTargetIndex
      ? sortedTargets
      : createTransientView(
          graph,
          `${id}-sorted-queries`,
          'uint32',
          gridSized(queryRows * recordWords)
        );
    let queryCount: GraphDataView<'uint32'> = gridCount;

    const nodes: GPUCommandNode<Parameters>[] = [
      // The grid index has no mask and its domain is wider than the valid extent, so excluded
      // targets become NaN points, which it ignores. The BVH build treats a NaN point as an empty
      // box, so this one array carries target validity (mask, finite, bounds) into both structures.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-grid-positions`,
        operation: OPERATION,
        variant: 'grid-positions',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          ...timeBindings('times'),
          {name: 'gridPositions', view: gridPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: targetRows,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'} && isPointValid(lattice, x, y)${times ? ` && isTimeInBounds(lattice, ${readTimeWGSL('times', 'index')})` : ''};
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${useGrid ? 'let unit = getGridUnitPosition(lattice, x, y);' : 'let unit = vec2f(x, y);'}
  gridPositions[gridPositionsOffset + index * ${gridDimension}u] = select(invalid, unit.x, included);
  gridPositions[gridPositionsOffset + index * ${gridDimension}u + 1u] = select(invalid, unit.y, included);${
    bucketed
      ? `
  gridPositions[gridPositionsOffset + index * 3u + 2u] = select(invalid, getGridUnitTime(lattice, ${readTimeWGSL('times', 'index')}), included);`
      : ''
  }`
      }),
      // Buckets targets into the lattice: `cellOffsets` and `sortedRows` (target rows by cell, in
      // unspecified order within a cell; every consumer below is order independent or sorts).
      ...(targetIndex?.nodes ?? []),
      // Query validity, stored in `counts` until the search pass overwrites it with the count.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-query-valid`,
        operation: OPERATION,
        variant: 'query-valid',
        bindings: [
          ...(crossJoin
            ? queryBinding
            : [
                {name: 'positions', view: positions, type: 'f32' as const, access: 'read' as const}
              ]),
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(queryMask
            ? [
                {
                  name: 'queryMask',
                  view: queryMask,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...queryTimeBinding,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: queryRows,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  ${readQueryWGSL('index')}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'};
  counts[countsOffset + index] = select(0u, 1u, included && ${queryPointValidWGSL}(lattice, x, y)${queryTimeValidWGSL('index')});`
      })
    ];

    if (useGrid && !sharesTargetIndex) {
      const queryGridPositions = createTransientView(
        graph,
        `${id}-query-grid-positions`,
        gridFormat,
        queryRows
      ) as GraphDataView<'float32x2'>;
      // Only the cell order of the queries is used; the cell offsets are scratch.
      const queryIndex = createPointNeighborGridIndexNodes<Parameters>(graph, {
        id: `${id}-query`,
        positions: queryGridPositions,
        gridSize: fullGridSize,
        rowCount: queryRows
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-query-grid-positions`,
          operation: OPERATION,
          variant: 'query-grid-positions',
          bindings: [
            ...(crossJoin
              ? queryBinding
              : [
                  {
                    name: 'positions',
                    view: positions,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
                ]),
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            ...(queryMask
              ? [
                  {
                    name: 'queryMask',
                    view: queryMask,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            ...queryTimeBinding,
            {
              name: 'queryGridPositions',
              view: queryGridPositions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: queryRows,
          declarations: latticeWGSL,
          body: `let lattice = readLattice();
  ${readQueryWGSL('index')}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'} && ${queryPointValidWGSL}(lattice, x, y)${queryTimeValidWGSL('index')};
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  // Only the cell order of the queries is used, so a query outside the lattice sorts with the
  // nearest edge cell; the grid index would otherwise skip it.
  let unit = clamp(getGridUnitPosition(lattice, x, y), vec2f(0.0), vec2f(1.0));
  queryGridPositions[queryGridPositionsOffset + index * ${gridDimension}u] = select(invalid, unit.x, included);
  queryGridPositions[queryGridPositionsOffset + index * ${gridDimension}u + 1u] = select(invalid, unit.y, included);${
    bucketed
      ? `
  queryGridPositions[queryGridPositionsOffset + index * 3u + 2u] = select(invalid, max(0.0, getGridUnitTime(lattice, ${readTimeWGSL(queryTimeName, 'index')})), included);`
      : ''
  }`
        }),
        ...queryIndex.nodes
      );
      queryOrder = queryIndex.sortedRows;
      queryCount = queryIndex.gridCount;
    }
    // Kernels bind at most 8 storage buffers, so the target and query triples are single buffers
    // and a shared index binds each of them once under the target names.
    const queryBufferName = sharesTargetIndex ? 'sortedTargets' : 'sortedQueries';
    const queryCountName = sharesTargetIndex ? 'gridCount' : 'queryCount';
    const searchBindings: WGSLKernelBinding[] = [
      {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read'},
      ...(sharesTargetIndex
        ? []
        : [
            {
              name: 'sortedQueries',
              view: sortedQueries,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]),
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: queryCountName, view: queryCount, type: 'u32', access: 'read'}
    ];
    // Maps the thread to the valid query at that cell-order slot and reads its point; surplus
    // threads (invalid queries have no work and a zero count from the query-valid pass) return.
    const threadToRowWGSL = `if (index >= ${queryCountName}[${queryCountName}Offset]) {
    return;
  }
  let queryIndex = ${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS + 2u];
  let x = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS]);
  let y = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS + 1u]);${
    times
      ? `
  let queryTime = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS + 3u]);`
      : ''
  }`;

    if (useGrid) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sorted-targets`,
          operation: OPERATION,
          variant: 'sorted-targets',
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...timeBindings('times'),
            {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
            {name: 'gridCount', view: gridCount, type: 'u32', access: 'read'},
            {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read_write'}
          ],
          invocationCount: targetRows,
          body: `if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u + 2u] = row;${
    times
      ? `
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u + 3u] = bitcast<u32>(${readTimeWGSL('times', 'row')});`
      : ''
  }`
        })
      );
    }
    if (useGrid && !sharesTargetIndex) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sorted-queries`,
          operation: OPERATION,
          variant: 'sorted-queries',
          bindings: [
            {
              name: 'queryPositions',
              view: queryPositions ?? positions,
              type: 'f32',
              access: 'read'
            },
            ...timeBindings('queryTimes'),
            {name: 'queryOrder', view: queryOrder, type: 'u32', access: 'read'},
            {name: 'queryCount', view: queryCount, type: 'u32', access: 'read'},
            {name: 'sortedQueries', view: sortedQueries, type: 'u32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          body: `if (index >= queryCount[queryCountOffset]) {
    return;
  }
  let row = queryOrder[queryOrderOffset + index];
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u]);
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u + 1u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u + 1u]);
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u + 2u] = row;${
    times
      ? `
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u + 3u] = bitcast<u32>(${readTimeWGSL('queryTimes', 'row')});`
      : ''
  }`
        })
      );
    }

    // BVH over the valid targets (NaN points of `gridPositions` become empty boxes), packed with
    // the leaf rows into one node table so a search kernel binds the whole tree once.
    let bvhNodeData: GraphDataView<'uint32'> | undefined;
    let bvhDeclarations = '';
    if (useBvh) {
      const {bvh, nodes: bvhNodes} = getNearestBVHNodes(graph, {
        id: `${id}-bvh`,
        operation: OPERATION,
        features: {kind: 'points', positions: gridPositions},
        featureCount: targetRows,
        leafCapacity: Math.max(2, getNextPowerOfTwo(targetRows)),
        spatialSort: true
      });
      nodes.push(...bvhNodes);
      bvhNodeData = createTransientView(
        graph,
        `${id}-bvh-node-data`,
        'uint32',
        bvh.nodeCount * NEIGHBOR_SEARCH_BVH_NODE_WORDS
      );
      bvhDeclarations = getBVHNeighborSearchDeclarationsWGSL({
        targetRows,
        internalNodeCount: bvh.internalNodeCount,
        levelCount: bvh.levelCount
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-bvh-node-data`,
          operation: OPERATION,
          variant: 'bvh-node-data',
          bindings: [
            {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
            {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
            {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
            {name: 'nodeData', view: bvhNodeData, type: 'u32', access: 'read_write'}
          ],
          invocationCount: bvh.nodeCount,
          declarations: `const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const NO_ROW: u32 = 0xffffffffu;`,
          body: `let base = nodeDataOffset + index * ${NEIGHBOR_SEARCH_BVH_NODE_WORDS}u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_ROW, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`
        })
      );
    }

    // BVH kernels bind the tree, the query points and the parameters (and the validity in `counts`).
    const bvhSearchBindings: WGSLKernelBinding[] = bvhNodeData
      ? [
          {name: 'nodeData', view: bvhNodeData, type: 'u32', access: 'read'},
          {name: queryName, view: queryPositions ?? positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'}
        ]
      : [];
    const bvhThreadToRowWGSL = getBVHThreadToRowWGSL(crossJoin);

    let knnIds: GraphDataView<'uint32'> | undefined;
    let knnDistances: GraphDataView<'float32'> | undefined;
    if (mode === 'knn') {
      knnIds = createTransientView(graph, `${id}-knn-ids`, 'uint32', queryRows * k);
      knnDistances = createTransientView(graph, `${id}-knn-distances`, 'float32', queryRows * k);
      const knnTailWGSL = `// Re-sort the selected neighbors by ID (insertion sort over at most K entries).
  for (var slot = 1u; slot < found; slot++) {
    let neighborId = bestIds[slot];
    let distanceSquared = bestDistances[slot];
    var destination = slot;
    while (destination > 0u && bestIds[destination - 1u] > neighborId) {
      bestIds[destination] = bestIds[destination - 1u];
      bestDistances[destination] = bestDistances[destination - 1u];
      destination--;
    }
    bestIds[destination] = neighborId;
    bestDistances[destination] = distanceSquared;
  }
  for (var slot = 0u; slot < found; slot++) {
    knnIds[knnIdsOffset + queryIndex * K + slot] = bestIds[slot];
    knnDistances[knnDistancesOffset + queryIndex * K + slot] = ${
      geodesic ? 'bestDistances[slot]' : 'sqrt(bestDistances[slot])'
    };
  }
  counts[countsOffset + queryIndex] = found;`;
      const knnColumns: WGSLKernelBinding[] = [
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
        {name: 'knnIds', view: knnIds, type: 'u32', access: 'read_write'},
        {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read_write'}
      ];
      if (useGrid) {
        nodes.push(
          ...createTiledWGSLKernelNodes<Parameters>(
            graph,
            {
              id: `${id}-knn`,
              operation: OPERATION,
              variant: 'knn',
              bindings: [...searchBindings, ...knnColumns],
              invocationCount: queryRows,
              declarations: `${latticeWGSL}
${getInsertNeighborWGSL(k)}`,
              body: `${threadToRowWGSL}
  var found = 0u;
  var bestDistances: array<f32, ${k}>;
  var bestIds: array<u32, ${k}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${
      geodesic
        ? getGeodesicNearestNeighborSearchWGSL(selfCondition, timeAxis)
        : getNearestNeighborSearchWGSL(selfCondition, timeAxis)
    }
  }
  ${knnTailWGSL}`
            },
            props.rowTileSize
          )
        );
      }
      if (useBvh) {
        nodes.push(
          ...createTiledWGSLKernelNodes<Parameters>(
            graph,
            {
              id: `${id}-bvh-knn`,
              operation: OPERATION,
              variant: 'bvh-knn',
              bindings: [...bvhSearchBindings, ...knnColumns],
              invocationCount: queryRows,
              declarations: `${latticeWGSL}
${getInsertNeighborWGSL(k)}
${bvhDeclarations}`,
              body: `${bvhThreadToRowWGSL}
  var found = 0u;
  var bestDistances: array<f32, ${k}>;
  var bestIds: array<u32, ${k}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    ${getBVHNearestNeighborSearchWGSL(selfCondition)}
  }
  ${knnTailWGSL}`
            },
            props.rowTileSize
          )
        );
      }
    } else {
      if (useGrid) {
        nodes.push(
          ...createTiledWGSLKernelNodes<Parameters>(
            graph,
            {
              id: `${id}-count`,
              operation: OPERATION,
              variant: 'radius-count',
              bindings: [
                ...searchBindings,
                {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
              ],
              invocationCount: queryRows,
              declarations: latticeWGSL,
              body: `${threadToRowWGSL}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${radiusLoop(
      `if (${selfCondition}) {
          count++;
        }`,
      timeAxis
    )}
  }
  counts[countsOffset + queryIndex] = count;`
            },
            props.rowTileSize
          )
        );
      }
      if (useBvh) {
        nodes.push(
          ...createTiledWGSLKernelNodes<Parameters>(
            graph,
            {
              id: `${id}-bvh-count`,
              operation: OPERATION,
              variant: 'bvh-radius-count',
              bindings: [
                ...bvhSearchBindings,
                {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
              ],
              invocationCount: queryRows,
              declarations: `${latticeWGSL}
${bvhDeclarations}`,
              body: `${bvhThreadToRowWGSL}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    ${getBVHRadiusNeighborLoopWGSL(`if (${selfCondition}) {
        count++;
      }`)}
  }
  counts[countsOffset + queryIndex] = count;`
            },
            props.rowTileSize
          )
        );
      }
    }

    const scanTotal = createTransientView(graph, `${id}-scan-total`, 'uint32', 1);
    nodes.push(
      ...createSaturatedCountScanNodes<Parameters>(graph, {
        id: `${id}-scan`,
        operation: OPERATION,
        counts,
        offsets: starts,
        rowCount: queryRows,
        total: scanTotal
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'scanTotal', view: scanTotal, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: weights.status.overflow, type: 'u32', access: 'read_write'},
          ...(weights.status.requiredCount
            ? [
                {
                  name: 'totalNeighbors',
                  view: weights.status.requiredCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: queryRows + 1,
        declarations: `const QUERY_ROWS: u32 = ${queryRows}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `if (index < QUERY_ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    offsets[offsetsOffset + QUERY_ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${weights.status.requiredCount ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}
  }`
      })
    );

    if (knnIds && knnDistances) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-emit`,
            operation: OPERATION,
            variant: 'knn-emit',
            bindings: [
              {name: 'knnIds', view: knnIds, type: 'u32', access: 'read'},
              {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read'},
              {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
              {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
              {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
            ],
            invocationCount: queryRows,
            declarations: `const K: u32 = ${k}u;`,
            body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin; slot < end; slot++) {
    let source = index * K + slot - begin;
    neighbors[neighborsOffset + slot] = knnIds[knnIdsOffset + source];
    distances[distancesOffset + slot] = knnDistances[knnDistancesOffset + source];
  }`
          },
          props.rowTileSize
        )
      );
    } else {
      const emitColumns: WGSLKernelBinding[] = [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
        {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
      ];
      if (useBvh) {
        nodes.push(
          ...createTiledWGSLKernelNodes<Parameters>(
            graph,
            {
              id: `${id}-bvh-emit`,
              operation: OPERATION,
              variant: 'bvh-radius-emit',
              bindings: [...bvhSearchBindings, ...emitColumns],
              invocationCount: queryRows,
              declarations: `${latticeWGSL}
${bvhDeclarations}`,
              body: `${bvhThreadToRowWGSL}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    var next = begin;
    ${getBVHRadiusNeighborLoopWGSL(`if (${selfCondition} && next < end) {
        neighbors[neighborsOffset + next] = neighbor;
        distances[distancesOffset + next] = sqrt(distanceSquared);
        next++;
      }`)}
  }`
            },
            props.rowTileSize
          )
        );
      }
      if (useGrid) {
        nodes.push(
          ...createTiledWGSLKernelNodes<Parameters>(
            graph,
            {
              id: `${id}-emit`,
              operation: OPERATION,
              variant: 'radius-emit',
              bindings: [...searchBindings, ...emitColumns],
              invocationCount: queryRows,
              declarations: latticeWGSL,
              body: `${threadToRowWGSL}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    var next = begin;
    ${radiusLoop(
      `if (${selfCondition} && next < end) {
          neighbors[neighborsOffset + next] = neighbor;
          distances[distancesOffset + next] = ${geodesic ? 'distance' : 'sqrt(distanceSquared)'};
          next++;
        }`,
      timeAxis
    )}
  }`
            },
            props.rowTileSize
          )
        );
      }
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-sort-rows`,
            operation: OPERATION,
            variant: 'radius-sort-rows',
            bindings: [
              {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
              {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
              {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
            ],
            invocationCount: queryRows,
            body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let neighborId = neighbors[neighborsOffset + slot];
    let distance = distances[distancesOffset + slot];
    var destination = slot;
    while (destination > begin && neighbors[neighborsOffset + destination - 1u] > neighborId) {
      neighbors[neighborsOffset + destination] = neighbors[neighborsOffset + destination - 1u];
      distances[distancesOffset + destination] = distances[distancesOffset + destination - 1u];
      destination--;
    }
    neighbors[neighborsOffset + destination] = neighborId;
    distances[distancesOffset + destination] = distance;
  }`
          },
          props.rowTileSize
        )
      );
    }

    if (weights.status.invalidCount) {
      // Included rows (mask nonzero) with a non-finite coordinate or outside the bounds, summed
      // with a scan so no atomics are needed. Self joins count targets; cross joins add queries.
      const flagRows = targetRows + (crossJoin ? queryRows : 0);
      const invalidFlags = createTransientView(graph, `${id}-invalid-flags`, 'uint32', flagRows);
      const invalidStarts = createTransientView(graph, `${id}-invalid-starts`, 'uint32', flagRows);
      const readPoint = (
        name: string,
        row: string
      ) => `let x = ${name}[${name}Offset + ${row} * 2u];
    let y = ${name}[${name}Offset + ${row} * 2u + 1u];`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-invalid-flags`,
          operation: OPERATION,
          variant: 'invalid-flags',
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...queryBinding,
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            ...timeBindings('times'),
            ...(crossJoin ? timeBindings('queryTimes') : []),
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : []),
            ...(crossJoin && props.queryMask
              ? [
                  {
                    name: 'queryMask',
                    view: props.queryMask,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: flagRows,
          declarations: `${latticeWGSL}
const TARGET_ROWS: u32 = ${targetRows}u;`,
          body: `let lattice = readLattice();
  var included = true;
  var inside = false;
  if (index < TARGET_ROWS) {
    ${readPoint('positions', 'index')}
    included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
    inside = isPointInBounds(lattice, x, y)${times ? ` && isTimeInBounds(lattice, ${readTimeWGSL('times', 'index')})` : ''};
  } else {
    ${
      crossJoin
        ? `let queryRow = index - TARGET_ROWS;
    ${readPoint('queryPositions', 'queryRow')}
    included = ${crossJoin && props.queryMask ? 'queryMask[queryMaskOffset + queryRow] != 0u' : 'true'};
    inside = ${queryPointInBoundsWGSL}(lattice, x, y)${times ? ` && ${queryTimeInBoundsWGSL('queryRow')}` : ''};`
        : ''
    }
  }
  invalidFlags[invalidFlagsOffset + index] = select(0u, 1u, included && !inside);`
        }),
        ...new GPUScan({
          id: `${id}-invalid-scan`,
          input: invalidFlags,
          output: invalidStarts,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-invalid-count`,
          operation: OPERATION,
          variant: 'invalid-count',
          bindings: [
            {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read'},
            {name: 'invalidStarts', view: invalidStarts, type: 'u32', access: 'read'},
            {
              name: 'invalidCount',
              view: weights.status.invalidCount,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: `invalidCount[invalidCountOffset] =
    invalidStarts[invalidStartsOffset + ${flagRows - 1}u] + invalidFlags[invalidFlagsOffset + ${flagRows - 1}u];`
        })
      );
    }

    const bandwidthWGSL =
      mode === 'radius'
        ? 'let bandwidth = readParameter(4u);'
        : // Adaptive bandwidth: the k-th neighbor distance (the row maximum) times libpysal's epsilon.
          `var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  bandwidth = bandwidth * ${getWGSLFloatLiteral(GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON)};`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weights`,
        operation: OPERATION,
        variant: 'weights',
        bindings: [
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'}
        ],
        invocationCount: queryRows,
        declarations: `${latticeWGSL}
${NEIGHBOR_WEIGHT_WGSL}`,
        body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${bandwidthWGSL}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let weight = getNeighborWeight(distances[distancesOffset + slot], bandwidth);
    weights[weightsOffset + slot] = weight;
    sum += weight;
  }
  if (${
    props.rowStandardize === undefined ? 'readParameter(9u) != 0.0' : String(props.rowStandardize)
  } && sum > 0.0 && isFiniteFloat(sum)) {
    for (var slot = begin; slot < end; slot++) {
      weights[weightsOffset + slot] = weights[weightsOffset + slot] / sum;
    }
  }`
      })
    );
    return nodes;
  }
}
