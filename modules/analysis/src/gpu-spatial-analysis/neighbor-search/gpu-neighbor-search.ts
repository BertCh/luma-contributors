// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUNeighborSearchExecutionPlan,
  GPU_NEIGHBOR_SEARCH_ROUTES,
  resolveGPUNeighborSearchStructure
} from './neighbor-search-planning';
import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';
import {validateSpatialStatisticsWritableDispatches} from '../spatial-weights/spatial-statistics-writable-dispatches';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGPUGridIndexMaximumCellCount,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
} from './neighbor-search-parameters';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  type GPUSpatialWeightsTransformState,
  validateGPUSpatialWeightsOutput
} from '../spatial-weights/spatial-weights';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/index';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';

import {getGPUNeighborSearchNodes} from './neighbor-search-nodes';
import {
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS
} from './neighbor-search-limits';
export {
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS
} from './neighbor-search-limits';

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
 * Compile-time: `mode`, `neighborCount`, `gridSize`, `searchStructure`, row counts, capacity and which optional
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
   * `searchRadius`, `maximumDistance`, `distanceFloor` and output `distances` are in `units`
   * (default meters), and `sphereRadius` is in meters (default `GPU_EARTH_MEAN_RADIUS_METERS`).
   * Any other space or metric throws; project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /** `'knn'`: the `k` nearest targets per query. `'radius'`: every target within the search radius. */
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
  neighborCount?: number;
  /**
   * Caller-owned output CSR: `offsets` has `queryRows + 1` entries; the slot capacity is
   * `neighbors.length`. `distances` is optional. Row `i` lists neighbor target IDs ascending.
   * `status.overflow` is 1 when the neighbors did not fit the capacity, else 0.
   * A requested status.candidateOverflow is written as 0 on every submission: this search
   * has no capacity-bounded intermediate candidate stage.
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
 *   only those with `d <= maximumDistance`. Fewer valid targets give a shorter row.
 * - Radius: every valid target with `d <= searchRadius`.
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
 *   `min_threshold_distance` (the smallest band keeping every unit connected), run kNN with `neighborCount: 1`
 *   and take the maximum of the returned `distances`.
 * - Capacity: rows are laid out in query order; slots past `neighbors.length` are dropped. Offsets
 *   are clamped to the capacity, so a reader always sees consistent (possibly truncated) rows,
 *   and `status.overflow` is set. `status.requiredCount` and `neighborCounts` stay unclamped.
 *
 * Space-time search: with `times` the relation gains `|t_target - t_query| <= timeRadius`; see
 * {@link GPUNeighborSearchProps.times}. A kNN query whose window holds fewer than `k` targets
 * finishes every ring and so scans its whole grid; set `maximumDistance` when windows may be sparse.
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
  /** Queryable catalog route constraints; view identity/alias checks still happen at declaration. */
  static readonly limits = GPU_NEIGHBOR_SEARCH_ROUTES;
  /** The same numeric route and sizing plan used before allocating graph scratch. */
  static readonly getExecutionPlan = getGPUNeighborSearchExecutionPlan;
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
    const captured = snapshotWeightsFields(props, [
      'id',
      'mode',
      'searchStructure',
      'spatialContext',
      'temporalContext',
      'positions',
      'queryPositions',
      'mask',
      'queryMask',
      'times',
      'queryTimes',
      'gridSize',
      'neighborCount',
      'parameters',
      'rowTileSize',
      'timeBucketCount',
      'rowStandardize',
      'weightTransform',
      'output'
    ]);
    const id = captured.id ?? 'neighbor-search';
    validateRequirement(
      {id, contributor: 'GPUNeighborSearch', messagePrefix: `${id}:`},
      !('k' in props),
      'k was renamed; use neighborCount'
    );
    const capturedOutput = snapshotWeightsFields(captured.output, [
      'offsets',
      'neighbors',
      'weights',
      'distances',
      'status',
      'neighborCounts'
    ]);
    const temporal = captured.temporalContext
      ? snapshotWeightsFields(captured.temporalContext, [
          'unit',
          'epoch',
          'encoding',
          'timeZone',
          'axis'
        ])
      : undefined;
    const axis = temporal?.axis
      ? snapshotWeightsFields(temporal.axis, ['kind', 'step'])
      : undefined;
    props = Object.freeze({
      ...captured,
      output: Object.freeze({
        ...capturedOutput,
        status: snapshotWeightsFields(capturedOutput.status, [
          'overflow',
          'candidateOverflow',
          'requiredCount',
          'invalidCount',
          'count',
          'uncertainCount',
          'iterationCount',
          'converged',
          'iterationLimitReached',
          'approximation',
          'complete',
          'unresolvedCount'
        ])
      }),
      ...(captured.gridSize
        ? {gridSize: Object.freeze([...captured.gridSize]) as GPUNeighborSearchProps['gridSize']}
        : {}),
      ...(temporal && axis
        ? {
            temporalContext: Object.freeze({
              ...temporal,
              axis: Object.freeze({
                ...axis,
                ...('step' in axis && axis.step && typeof axis.step === 'object'
                  ? {step: snapshotWeightsFields(axis.step, ['calendarUnit', 'interval'])}
                  : {})
              })
            })
          }
        : {}),
      ...(captured.spatialContext
        ? {
            spatialContext: snapshotWeightsPort({spatialContext: captured.spatialContext})
              .spatialContext
          }
        : {})
    });
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
    resolveGPUNeighborSearchStructure(
      id,
      searchStructure,
      this.spatialContext.coordinateSpace,
      Boolean(props.times)
    );
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
      validateCapacity(
        id,
        'neighborCount',
        props.neighborCount ?? Number.NaN,
        1,
        GPU_NEIGHBOR_SEARCH_MAXIMUM_K
      );
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
    for (const field of [
      'count',
      'uncertainCount',
      'iterationCount',
      'converged',
      'iterationLimitReached',
      'approximation',
      'complete',
      'unresolvedCount'
    ] as const) {
      validateRequirement(
        id,
        output.status[field] === undefined,
        `output.status.${field} is unsupported by exact, non-iterative neighbor search; use output.offsets[queryRows] for the written count and status.requiredCount, overflow, candidateOverflow or invalidCount for diagnostics`
      );
    }
    const weightRows = validateGPUSpatialWeightsOutput(id, output);
    validateSpatialStatisticsWritableDispatches(
      id,
      {
        offsets: output.offsets,
        neighbors: output.neighbors,
        weights: output.weights,
        distances: output.distances,
        overflow: output.status.overflow,
        candidateOverflow: output.status.candidateOverflow,
        requiredCount: output.status.requiredCount,
        invalidCount: output.status.invalidCount,
        neighborCounts: output.neighborCounts
      },
      []
    );
    if (weightRows !== queryRows) {
      throw new Error(`${id} output.offsets length must equal the query row count + 1`);
    }
    const {requiredCount, invalidCount} = output.status;
    for (const [name, view] of [
      ['output.status.overflow', output.status.overflow],
      ['output.status.candidateOverflow', output.status.candidateOverflow],
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
          k: props.mode === 'knn' ? props.neighborCount : undefined
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
        output.status.candidateOverflow,
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
    return getGPUNeighborSearchNodes(this, graph);
  }
}
