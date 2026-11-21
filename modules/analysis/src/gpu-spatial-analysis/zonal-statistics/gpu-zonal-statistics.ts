// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateZonalStatisticsBindings,
  validateZonalStatisticsReadWriteDispatches
} from './zonal-statistics-validation';
import {
  createTransientView,
  GPUGroupAggregation,
  GraphVectorView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createTransientUint32Rows
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUFloat32Positions} from '@luma.gl/graph/geospatial';
import type {GPUUint32Rows} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getGraphViewChunks,
  isGraphVectorView,
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateGraphViewTopology,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUStatusPort,
  GPU_LINEAR_EDGE_MODELS,
  type GPUCandidateStatusPort,
  type GPULonLatPolygonStatusPort,
  type GPUPackedGeometryPort,
  type GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {GPUPointInPolygonJoin, type GPUPointInPolygonBoundaryRule} from '../spatial-join/index';
import {getPointInPolygonContainmentContext} from '../spatial-join/gpu-point-in-polygon-join';
import {validateDisjointOutputs} from '../spatial-join/spatial-join-passes';
import {
  createTransientFloat32Rows,
  createZonalStatisticsAreaNode,
  createZonalStatisticsDensityNode,
  createZonalStatisticsDeviationNode,
  createZonalStatisticsMeanNode,
  createZonalStatisticsRefinedMeanNode,
  createZonalStatisticsVarianceNode,
  createZonalStatisticsWeightedDensityNode,
  createZonalStatisticsPrepareNodes,
  getZonalStatisticsExtentNodes,
  getZonalStatisticsPlan,
  type ZonalStatisticsPlan
} from './zonal-statistics-passes';
import {getSortedSegmentSumNodes} from '../../analysis-core/kernels/sorted-segment-sums';

import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/kernels/lon-lat-ring-wgsl';

const OPERATION = 'GPUZonalStatistics';
/** Planar points and polygons, or longitude/latitude degrees with caller-supplied areas. */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metricsByCoordinateSpace: Object.freeze({
    planar: Object.freeze(['none', 'native'] as const),
    'longitude-latitude': Object.freeze(['none'] as const)
  })
});

/** Planar polygon areas require an area-compatible frame. */
const AREA_SUPPORT = Object.freeze({...SUPPORT, measure: 'area'} as const);

/**
 * Feature membership from polygon features (the four-level `GPUPolygonGeometryPort` layout, one
 * output row per feature): points are joined to polygons with `GPUPointInPolygonJoin` every
 * encoding. A point belongs to the containing feature with the
 * smallest row; boundary points follow `boundaryRule` (half-open by default, so a point on a shared
 * edge of a coverage is counted exactly once regardless of row order).
 *
 * Ring 0 of each polygon is the shell and rings close implicitly. `sourceIds` is not read: every
 * output is indexed by feature row. The polygon buffers' contents are per-frame; everything else
 * here is compile-time topology.
 */
export type GPUZonalStatisticsPolygons = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Maximum `(point, feature)` bounding-box candidates per encoding. Compile-time. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. Compile-time. */
  leafCapacity?: number;
  /**
   * How a point exactly on a polygon boundary is assigned; see
   * `GPUPointInPolygonJoinProps.boundaryRule`. Defaults to `'half-open'`: a point on an edge or
   * vertex shared by features of a valid coverage is counted exactly once, by the feature whose
   * interior lies to its right/above, independent of the feature row order. Points on the outer
   * right or top boundary of the coverage are counted by no feature. `'closed'` counts every
   * boundary point (the smallest containing row wins, so shared-boundary ownership depends on row
   * order); `'open'` counts only strict interiors. Compile-time.
   */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
};

/**
 * Feature membership computed by the caller (for example by a join with different semantics, a
 * lookup table or a previous frame's result). No geometry is read.
 */
export type GPUZonalStatisticsFeatureRows = {
  /** Feature source discriminator. */
  kind: 'feature-rows';
  /**
   * Per-point feature row in `[0, featureCount)`. `GPU_SPATIAL_JOIN_NO_FEATURE` (or any value
   * `>= featureCount`) marks an unassigned point. Per-frame contents; the chunk topology defines
   * the point count, and `values` and `weights` must have the same total length (their chunk
   * boundaries may differ in the `'atomic'` sum order).
   */
  pointFeatureRows: GPUUint32Rows;
  /** Number of features, which is the row count of every per-feature output. Compile-time. */
  featureCount: number;
  /**
   * Optional upstream membership diagnostics, copied on every encoding. Each supplied scalar
   * requires its matching `output.status` view so incomplete membership cannot appear clean.
   * `uncertainCount` preserves the upstream count, rather than reducing it to a boolean.
   */
  status?: Pick<
    NonNullable<GPUZonalStatisticsOutput['status']>,
    | 'overflow'
    | 'candidateOverflow'
    | 'uncertainCount'
    | 'antimeridianUncertainCount'
    | 'poleUncertainCount'
  >;
};

/** Feature membership source for {@link GPUZonalStatistics}. */
export type GPUZonalStatisticsFeatures = GPUZonalStatisticsPolygons | GPUZonalStatisticsFeatureRows;

/**
 * How per-feature sums and weight sums are accumulated. Compile-time.
 *
 * - `'atomic'`: compare-exchange float atomics through `GPUGroupAggregation`. Fast, but the
 *   addition order depends on GPU scheduling, so sums can differ in the last bits between runs and
 *   devices.
 * - `'sorted'`: points are stably sorted by feature row, then reduced per feature in a fixed tree
 *   order, which is bitwise reproducible across runs on the same device. The tree order differs
 *   from a sequential CPU sum, so results match a CPU oracle only within a float32 tolerance. The
 *   sort costs O(N) work per radix pass with `ceil(log2(featureCount + 1) / 4)` passes, plus a
 *   scan and gather, and the segmented reduction uses one 256-thread workgroup per feature.
 */
export type GPUZonalStatisticsSumOrder = 'atomic' | 'sorted';

/** Statistic whose `[min, max]` extent over features is published to `output.extent`. */
export type GPUZonalStatisticsExtentStatistic =
  | 'count'
  | 'sum'
  | 'mean'
  | 'minimum'
  | 'maximum'
  | 'density';

/**
 * Caller-owned outputs of {@link GPUZonalStatistics}. Which views are present is compile-time.
 *
 * Every per-feature output has exactly `featureCount` rows and is rewritten on every encoding.
 */
type GPUZonalStatisticsOutputFields = {
  /** Points assigned to the feature, regardless of value validity. */
  counts?: GraphDataView<'uint32'>;
  /**
   * Assigned points with a finite value (and a finite weight when weights are given).
   * Includes rows whose weighted product overflows float32 and is omitted from `sums`.
   */
  valueCounts?: GraphDataView<'uint32'>;
  /**
   * `sum(v)`, or `sum(w * v)` with weights, over value-valid rows, omitting products that
   * overflow float32. Their rows still contribute to `valueCounts`, `weightSums` and extrema.
   * 0 for empty features. Applies in both sum orders.
   */
  sums?: GraphDataView<'float32'>;
  /**
   * `sum(w)` over value-valid rows, including rows whose product is omitted from `sums`.
   * Requires `weights`. 0 for empty features.
   */
  weightSums?: GraphDataView<'float32'>;
  /**
   * `sum(v) / valueCount`, or `sum(w * v) / sum(w)` with weights. Quiet NaN (`0x7fc00000`) when the
   * feature has no valid rows or the weight sum is 0.
   * An overflowing product is omitted only from the numerator: values `[2^127, 2]` and weights
   * `[4, 1]` give `sums = 2`, `weightSums = 5`, `valueCounts = 2` and a mean of `0.4` in both
   * sum orders. Variance refinement starts from this mean; extreme magnitudes can also overflow
   * its float32 deviation products, so variance and standard deviation need not be finite.
   */
  means?: GraphDataView<'float32'>;
  /** Minimum of the valid values, never weighted. NaN for features without valid rows. */
  minima?: GraphDataView<'float32'>;
  /** Maximum of the valid values, never weighted. NaN for features without valid rows. */
  maxima?: GraphDataView<'float32'>;
  /**
   * Unbiased sample variance of the valid values. Unweighted: `sum (v - mean)^2 / (n - 1)`. With
   * `weights`, which act as reliability weights: `sum w (v - m)^2 / (V1 - V2 / V1)` with
   * `m` the weighted mean, `V1 = sum w` and `V2 = sum w^2`, which reduces to the unweighted form
   * for unit weights (`numpy.cov(v, aweights=w, ddof=1)`). Computed in float32 around a refined
   * mean: one pass over the points refines the float32 mean with the sum of `w (v - mean)`, a
   * second sums `w d^2` and `w d` around it, and the corrected two-pass term
   * `sum w d^2 - (sum w d)^2 / V1` is used, so values at projected-coordinate magnitudes (1e7)
   * over many rows keep their small variance. Quiet NaN for features with fewer than two valid rows
   * or a non-positive denominator. Requires packed (not chunked) point views.
   * Deviation products that overflow float32 are omitted. This can leave a finite, misleading
   * result: values `[2^127, 2]` with weights `[4, 1]` yield variance about `0.8192` although the
   * true variance is enormous. A finite result does not certify that every term was retained.
   * Signed weights remain accepted; they can give a negative variance and NaN standard deviation.
   * The mean-refinement and deviation reductions use fixed trees in both public sum orders.
   */
  variances?: GraphDataView<'float32'>;
  /**
   * `sqrt(variances)`; NaN for NaN or negative variance. Same requirements and overflow-term
   * limitations as `variances`: the `[2^127, 2]`, weights `[4, 1]` example gives about `0.9051`,
   * a finite but misleading result.
   */
  standardDeviations?: GraphDataView<'float32'>;
  /**
   * `counts / area` when the area is finite and positive, otherwise NaN. The area is `areas` when
   * given, otherwise the GPU polygon area (planar only: on a longitude/latitude context, pass
   * spherical `areas`, for example from `GPUGeometryMeasures`).
   */
  densities?: GraphDataView<'float32'>;
  /**
   * `weightSums / area` (weighted density) when the area is finite and positive, otherwise NaN.
   * Requires `weights`. Same area rules as `densities`.
   */
  weightedDensities?: GraphDataView<'float32'>;
  /**
   * GPU-computed polygon areas. Only valid for `kind: 'polygons'` without `areas`.
   * Per feature and polygon: `|area(shell)| - sum |area(hole)|`, summed over polygons, with each
   * ring's shoelace sum taken relative to the ring's first finite vertex. Non-finite vertices are
   * skipped (their neighbors are joined), and ranges that do not fit the offset arrays contribute
   * 0. One invocation handles a whole feature, so the cost per thread is proportional to the
   * vertex count of the largest feature. Overlapping polygons of one feature are summed, not
   * unioned.
   */
  featureAreas?: GraphDataView<'float32'>;
  /**
   * `kind: 'polygons'` only: receives the join's per-point feature row (or
   * `GPU_SPATIAL_JOIN_NO_FEATURE`), with the chunk topology of `positions`.
   */
  pointFeatureRows?: GPUUint32Rows;
  /**
   * `kind: 'polygons'` only: optional per-point flag, one packed `uint32` per point in row order
   * across chunks, 1 when that point's membership may be incomplete because the join dropped
   * candidates (`features.candidateCapacity`) or BVH leaves (`features.leafCapacity`). A flagged
   * point may be unassigned although a feature contains it, so the statistics of the features it
   * belongs to may be low. It is `GPUPointInPolygonJoinOutput.candidateTruncated` passed through.
   */
  candidateTruncated?: GraphDataView<'uint32'>;
  /**
   * Two rows `[minimum, maximum]` of `extentStatistic` over features whose count is positive and
   * whose statistic is finite. `[0, 0]` when no feature qualifies. Counts and extrema are exact;
   * counts above 2^24 lose precision as float32. Sum and mean extents follow the overflowing
   * product exclusions documented on `sums` and `means`.
   */
  extent?: GraphDataView<'float32'>;
  /**
   * Statistic that `extent` summarizes. Required with `extent`. The statistic is computed into a
   * graph transient when its own output view is absent. Compile-time.
   */
  extentStatistic?: GPUZonalStatisticsExtentStatistic;
  /**
   * Optional membership status. `antimeridianUncertainCount` counts unprepared crossing rings;
   * `poleUncertainCount` counts uncapped pole rings. Feature-row inputs copy supplied geographic
   * counts on every encoding and write zero when absent. Planar polygon inputs leave both untouched.
   */
  status?: GPULonLatPolygonStatusPort &
    GPUCandidateStatusPort & {
      /**
       * One row: 1 when the join overflowed a BVH leaf or candidate capacity, otherwise 0. Always 0
       * when `kind: 'feature-rows'` has no upstream overflow. Feature-row inputs copy the
       * corresponding `features.status` scalar. `candidateOverflow` reports intermediate
       * capacity loss (or copies the upstream candidate flag for feature rows); see
       * `candidateTruncated` for the points affected.
       */
      overflow?: GraphDataView<'uint32'>;
      /**
       * One row: the number of (point, feature) candidate pairs whose containment could not be
       * proven (non-finite input, malformed offsets, or a predicate outside the double-single
       * arithmetic envelope). Uncertain pairs are never assigned: a point whose only candidates were
       * uncertain stays unassigned, so a nonzero value means `counts` and the other statistics may be
       * low. Feature-row inputs copy `features.status.uncertainCount`, or write 0 when absent.
       */
      uncertainCount?: GraphDataView<'uint32'>;
    };
};

/**
 * Properties for {@link GPUZonalStatistics}.
 *
 * Per-frame: the contents of every input buffer (points, values, weights, areas, geometry, rows).
 * Compile-time: view lengths and chunking, which outputs exist, `features` capacities and flags,
 * `sumOrder`, and `extentStatistic`.
 */
export type GPUZonalStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'zonal-statistics'`. */
  id?: string;
  /** Feature membership source. */
  features: GPUZonalStatisticsFeatures;
  /**
   * Coordinate context of the points and polygons. Omitted means native planar. Planar and
   * `longitude-latitude` with the `none` metric are accepted; other spaces throw.
   * Membership is a point-in-polygon test in the coordinates as given, so in longitude/latitude
   * it is topological with straight edges
   * in degrees (not geodesics). Prepare polygons that cross ±180 or enclose a pole with
   * `GPULonLatPolygonPreparation`; the optional status fields report unprepared rings. Areas are
   * where the space matters: planar polygon area in degrees squared is wrong, so with a
   * longitude/latitude context `densities`, `weightedDensities` and `featureAreas` from the GPU
   * area are rejected at construction; pass spherical `areas` (square meters from
   * `GPUGeometryMeasures`) instead. GPU-computed planar areas reject azimuthal-equidistant
   * frames; use a Lambert equal-area frame or a projected CRS. Caller-supplied areas bypass
   * this area-frame restriction. Counts, sums, means, extrema and variances are unaffected.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `features` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  featuresSpatialContext?: GPUSpatialContext;
  /** Point positions. Required for `kind: 'polygons'`, ignored for `kind: 'feature-rows'`. */
  positions?: GPUFloat32Positions;
  /**
   * Per-point values with the same total length as the points. Required by every value statistic
   * (`valueCounts`, `sums`, `weightSums`, `means`, `minima`, `maxima` and extents of them).
   * Non-finite values never contribute.
   */
  values?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * Per-point weights with the topology of `values`. Requires `values`. With weights, `sums` is
   * `sum(w * v)` and `means` is `sum(w * v) / sum(w)`; a row with a non-finite weight is invalid
   * for every value statistic. Products that overflow float32 are dropped from sums.
   * Such rows remain in value counts, weight sums and extrema; see `output.means` for the
   * resulting numerator/denominator asymmetry and its effect on variance refinement.
   */
  weights?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * Optional per-feature areas (`featureCount` rows). They replace the GPU-computed polygon area
   * and are required for densities with `kind: 'feature-rows'` or a longitude/latitude context.
   * Per-frame contents.
   */
  areas?: GraphDataView<'float32'>;
  /**
   * Accumulation order of sums and weight sums. Defaults to `'sorted'` when every point-rate input
   * is packed (sorted sums require packed views), else `'atomic'`. Sorted sums are bitwise
   * reproducible and bounded in cost. `'atomic'` skips the sort but serializes when many points
   * share a feature, because float compare-exchange atomics retry on one address. Compile-time.
   */
  sumOrder?: GPUZonalStatisticsSumOrder;
  /** Caller-owned outputs. At least one must be present. */
  output: GPUZonalStatisticsOutput;
};

const PER_FEATURE_FLOAT_OUTPUTS = [
  'sums',
  'weightSums',
  'means',
  'minima',
  'maxima',
  'variances',
  'standardDeviations',
  'densities',
  'weightedDensities',
  'featureAreas'
] as const;
const PER_FEATURE_UINT32_OUTPUTS = ['counts', 'valueCounts'] as const;
const OUTPUT_NAMES = [
  ...PER_FEATURE_UINT32_OUTPUTS,
  ...PER_FEATURE_FLOAT_OUTPUTS,
  'pointFeatureRows',
  'extent'
] as const;

/** Every caller-owned output view, status scalars included. */
function getOutputViews(output: GPUZonalStatisticsOutput) {
  return [
    ...OUTPUT_NAMES.map(name => output[name]),
    output.candidateTruncated,
    output.status?.overflow,
    output.status?.candidateOverflow,
    output.status?.uncertainCount,
    output.status?.antimeridianUncertainCount,
    output.status?.poleUncertainCount
  ];
}

type RequireZonalOutputLeaf<Fields> = {
  [Field in keyof Fields]-?: Required<Pick<Fields, Field>>;
}[keyof Fields];

/** At least one destination view is required; extentStatistic and an empty status map are not destinations. */
export type GPUZonalStatisticsOutput = GPUZonalStatisticsOutputFields &
  (
    | RequireZonalOutputLeaf<Omit<GPUZonalStatisticsOutputFields, 'extentStatistic' | 'status'>>
    | {
        status: NonNullable<GPUZonalStatisticsOutputFields['status']> &
          RequireZonalOutputLeaf<NonNullable<GPUZonalStatisticsOutputFields['status']>>;
      }
  );

/**
 * Choropleth aggregation: joins points to features and reduces optional per-point values and
 * weights into per-feature statistics, all on the GPU.
 *
 * Membership comes from `GPUPointInPolygonJoin` (`kind: 'polygons'`) or from caller-provided
 * feature rows. The statistics computed are chosen at compile time by which output views exist;
 * statistics that other outputs depend on (for example sums and value counts for means) are
 * computed into graph transients.
 *
 * ## Semantics
 *
 * - A row is valid for value statistics when it is assigned to a feature row `< featureCount`,
 *   its value is finite, and, when weights are given, its weight is finite. Rows with non-finite
 *   values are still counted in `counts`.
 * - Empty features (no valid rows) have `sums` 0, `weightSums` 0, `valueCounts` 0, and NaN
 *   `means`, `minima` and `maxima` (quiet NaN `0x7fc00000`, as `GPUGroupAggregation`). A weighted
 *   mean with a weight sum of 0 is NaN.
 * - Density is `counts[f] / area[f]` when the area is finite and positive, otherwise NaN. Weighted
 *   density is `weightSums[f] / area[f]`.
 * - Variance and standard deviation are the unbiased estimates described on
 *   `GPUZonalStatisticsOutput.variances`, with NaN below two valid rows.
 * - The extent covers features with `counts > 0` and a finite statistic, and is `[0, 0]` when none
 *   qualifies.
 *
 * ## Determinism
 *
 * Counts, value counts, minima, maxima and extents of those are exact and independent of
 * scheduling. With `sumOrder: 'atomic'`, sums and weight sums use compare-exchange float
 * addition whose order depends on GPU scheduling: results can differ in the last bits between
 * runs and devices (and from a sequential CPU sum). With `sumOrder: 'sorted'` they are bitwise
 * reproducible on one device, at the cost of a stable sort of the points by feature row. The
 * sorted order is the default: the same compare-exchange pattern measured 10-600x slower than
 * sorted sums on contended keys in raster zonal statistics and flow aggregation.
 *
 * ## Non-goals
 *
 * No median or quantiles (they need a per-feature sort of the values), no weighted minima or
 * maxima, no GPU spherical polygon areas (pass `areas`), and no union of overlapping polygons
 * within one feature. Membership is by point location, not by fractional coverage. The
 * contributor never compiles, encodes, submits, or reads back.
 */
export class GPUZonalStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUZonalStatisticsProps;
  /** Number of features, the row count of every per-feature output. */
  readonly featureCount: number;
  /** Resolved coordinate context: native planar unless a longitude/latitude one was declared. */
  readonly spatialContext: GPUSpatialContext;
  /** Accumulation order of sums and weight sums. */
  readonly sumOrder: GPUZonalStatisticsSumOrder;
  /** Number of points. */
  readonly pointCount: number;
  /** Which statistics are computed, including transient ones. */
  readonly plan: ZonalStatisticsPlan;

  constructor(props: GPUZonalStatisticsProps) {
    if (props.features.kind === 'feature-rows') {
      props = snapshotConstructionFields(props, [
        'id',
        'features',
        'output',
        'positions',
        'values',
        'weights',
        'areas',
        'spatialContext',
        'featuresSpatialContext',
        'sumOrder'
      ]);
      // Capture declared fields rather than spreading caller maps: views may be inherited
      // or exposed through non-enumerable getters.
      const outputFields = snapshotConstructionFields(props.output, [
        ...OUTPUT_NAMES,
        'candidateTruncated',
        'extentStatistic',
        'status'
      ]);
      // Snapshotting the parent map retains the already checked discriminator.
      const featureRows = props.features as GPUZonalStatisticsFeatureRows;
      const features = snapshotConstructionFields(featureRows, [
        'kind',
        'pointFeatureRows',
        'featureCount',
        'status'
      ]);
      const status =
        features.status &&
        snapshotConstructionFields(features.status, [
          'overflow',
          'candidateOverflow',
          'uncertainCount',
          'antimeridianUncertainCount',
          'poleUncertainCount'
        ]);
      const output = {
        ...outputFields,
        status:
          outputFields.status &&
          snapshotConstructionFields(outputFields.status, [
            'overflow',
            'candidateOverflow',
            'uncertainCount',
            'antimeridianUncertainCount',
            'poleUncertainCount'
          ])
      };
      props = {
        ...props,
        features: Object.freeze({...features, status}),
        // Every declared leaf was captured above; spreading the status snapshot widens its union.
        output: Object.freeze(output) as GPUZonalStatisticsOutput
      };
    }
    this.id = props.id ?? 'zonal-statistics';
    this.props = props;
    const {features, output, values, weights, areas} = props;
    this.sumOrder = props.sumOrder ?? getDefaultZonalSumOrder(props);
    const {id} = this;
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }

    if (features.kind === 'polygons') {
      if (!props.positions) {
        throw new Error(`${id} positions are required for polygon features`);
      }
      for (const chunk of getGraphViewChunks(props.positions)) {
        validateSinglePackedView(chunk, ['float32x2'], `${id} positions`);
      }
      validateSinglePackedView(features.positions, ['float32x2'], `${id} features.positions`);
      for (const [name, view] of [
        ['featureOffsets', features.featureOffsets],
        ['polygonOffsets', features.polygonOffsets],
        ['ringOffsets', features.ringOffsets]
      ] as const) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} requires a terminal entry`);
        }
      }
      this.featureCount = features.featureOffsets.length - 1;
      this.pointCount = props.positions.length;
    } else {
      for (const chunk of getGraphViewChunks(features.pointFeatureRows)) {
        validatePackedUint32View(chunk, `${id} pointFeatureRows`);
      }
      validateCapacity(id, 'featureCount', features.featureCount, 0);
      if (output.pointFeatureRows) {
        throw new Error(`${id} output.pointFeatureRows requires polygon features`);
      }
      if (output.featureAreas) {
        throw new Error(`${id} output.featureAreas requires polygon features`);
      }
      this.featureCount = features.featureCount;
      this.pointCount = features.pointFeatureRows.length;
      if (features.status) {
        validateGPUStatusPort(`${id} features.status`, features.status);
        for (const name of [
          'overflow',
          'candidateOverflow',
          'uncertainCount',
          'antimeridianUncertainCount',
          'poleUncertainCount'
        ] as const) {
          const view = features.status[name];
          if (view) {
            validatePackedUint32View(view, `${id} features.status.${name}`);
            validateRequirement(id, view.length === 1, `features.status.${name} requires one row`);
            validateRequirement(
              id,
              Boolean(output.status?.[name]),
              `features.status.${name} requires output.status.${name}`
            );
          }
        }
      }
    }
    if (this.featureCount < 1) {
      throw new Error(`${id} requires at least one feature`);
    }

    if (!getOutputViews(output).some(Boolean)) {
      throw new Error(`${id} requires at least one output`);
    }
    if (output.extent && !output.extentStatistic) {
      throw new Error(`${id} output.extent requires output.extentStatistic`);
    }
    if (
      output.extentStatistic &&
      !['count', 'sum', 'mean', 'minimum', 'maximum', 'density'].includes(output.extentStatistic)
    ) {
      throw new Error(`${id} output.extentStatistic is not a supported statistic`);
    }
    if (output.featureAreas && areas) {
      throw new Error(`${id} output.featureAreas cannot be requested together with areas`);
    }

    if (weights && !values) {
      throw new Error(`${id} weights require values`);
    }
    if (output.weightSums && !weights) {
      throw new Error(`${id} output.weightSums requires weights`);
    }
    for (const [name, view] of [
      ['values', values],
      ['weights', weights]
    ] as const) {
      if (!view) {
        continue;
      }
      for (const chunk of getGraphViewChunks(view)) {
        validateSinglePackedView(chunk, ['float32'], `${id} ${name}`);
      }
      if (view.length !== this.pointCount) {
        throw new Error(`${id} ${name} length must equal the point count`);
      }
    }
    if (values && weights) {
      validateGraphViewTopology(id, 'weights', values, weights);
    }
    if (areas) {
      validateSinglePackedView(areas, ['float32'], `${id} areas`);
      if (areas.length !== this.featureCount) {
        throw new Error(`${id} areas length must equal the feature count`);
      }
    }

    for (const name of PER_FEATURE_UINT32_OUTPUTS) {
      const view = output[name];
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} output.${name} length must equal the feature count`);
        }
      }
    }
    for (const name of PER_FEATURE_FLOAT_OUTPUTS) {
      const view = output[name];
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} output.${name} length must equal the feature count`);
        }
      }
    }
    if (output.extent) {
      validateSinglePackedView(output.extent, ['float32'], `${id} output.extent`);
      if (output.extent.length !== 2) {
        throw new Error(`${id} output.extent must contain two float32 rows`);
      }
    }
    if (output.status) {
      validateGPUStatusPort(`${id} output.status`, output.status);
    }
    if (output.candidateTruncated) {
      validateRequirement(
        id,
        features.kind === 'polygons',
        "output.candidateTruncated needs features.kind 'polygons'"
      );
      validatePackedUint32View(output.candidateTruncated, `${id} output.candidateTruncated`);
      validateRequirement(
        id,
        output.candidateTruncated.length === this.pointCount,
        `output.candidateTruncated needs one row per point (${this.pointCount})`
      );
    }
    if (output.pointFeatureRows) {
      for (const chunk of getGraphViewChunks(output.pointFeatureRows)) {
        validatePackedUint32View(chunk, `${id} output.pointFeatureRows`);
      }
      validateGraphViewTopology(
        id,
        'output.pointFeatureRows',
        props.positions!,
        output.pointFeatureRows
      );
    }

    this.plan = getZonalStatisticsPlan({
      outputs: {
        counts: Boolean(output.counts),
        valueCounts: Boolean(output.valueCounts),
        sums: Boolean(output.sums),
        weightSums: Boolean(output.weightSums),
        means: Boolean(output.means),
        minima: Boolean(output.minima),
        maxima: Boolean(output.maxima),
        densities: Boolean(output.densities),
        featureAreas: Boolean(output.featureAreas),
        extent: Boolean(output.extent),
        weightedDensities: Boolean(output.weightedDensities),
        variances: Boolean(output.variances),
        standardDeviations: Boolean(output.standardDeviations)
      },
      extentStatistic: output.extentStatistic,
      hasWeights: Boolean(weights),
      hasAreas: Boolean(areas),
      sorted: this.sumOrder === 'sorted'
    });
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      features.kind === 'polygons' && this.plan.needGPUAreas ? AREA_SUPPORT : SUPPORT
    );
    resolveGPUSecondarySpatialContext(
      id,
      'features',
      props.featuresSpatialContext,
      this.spatialContext,
      SUPPORT.edgeModels
    );
    if (this.plan.needValueStatistics && !values) {
      throw new Error(`${id} value statistics require values`);
    }
    if (
      features.kind === 'feature-rows' &&
      (this.plan.needDensities || this.plan.needWeightedDensities) &&
      !areas
    ) {
      throw new Error(`${id} densities require areas with feature-rows features`);
    }
    if (output.weightedDensities && !weights) {
      throw new Error(`${id} output.weightedDensities requires weights`);
    }
    if (this.spatialContext.coordinateSpace === 'longitude-latitude' && this.plan.needGPUAreas) {
      // Planar shoelace area of degrees is wrong; the caller supplies spherical areas.
      throw new Error(`${id} longitude-latitude densities need areas in square meters`);
    }
    if (this.plan.needVariances) {
      for (const [name, view] of [
        ['positions', features.kind === 'polygons' ? props.positions : undefined],
        ['values', values],
        ['weights', weights],
        [
          'pointFeatureRows',
          features.kind === 'polygons' ? output.pointFeatureRows : features.pointFeatureRows
        ]
      ] as const) {
        if (isGraphVectorView(view)) {
          throw new Error(`${id} variances require packed ${name}`);
        }
      }
    }

    if (this.sumOrder === 'sorted') {
      const rows =
        features.kind === 'polygons' ? output.pointFeatureRows : features.pointFeatureRows;
      for (const [name, view] of [
        ['positions', features.kind === 'polygons' ? props.positions : undefined],
        ['values', values],
        ['weights', weights],
        ['pointFeatureRows', rows]
      ] as const) {
        if (isGraphVectorView(view)) {
          throw new Error(`${id} sorted sum order requires packed ${name}`);
        }
      }
    }

    validateDisjointOutputs(
      id,
      [
        props.positions,
        values,
        weights,
        areas,
        features.kind === 'polygons' ? features.positions : undefined,
        features.kind === 'polygons' ? features.featureOffsets : undefined,
        features.kind === 'polygons' ? features.polygonOffsets : undefined,
        features.kind === 'polygons' ? features.ringOffsets : undefined,
        features.kind === 'feature-rows' ? features.pointFeatureRows : undefined,
        ...(features.kind === 'feature-rows' ? Object.values(features.status ?? {}) : [])
      ],
      getOutputViews(output)
    );
    validateZonalStatisticsReadWriteDispatches(id, props, this.plan, this.sumOrder === 'atomic');
  }

  /**
   * Returns optional polygon checks, membership, preparation, aggregation and extent nodes.
   *
   * @throws If any view belongs to another graph.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, plan, sumOrder} = this;
    const {features, output, values, weights, areas} = props;
    const sortedSums = sumOrder === 'sorted' && (plan.needSums || plan.needWeightSums);
    validateZonalStatisticsBindings(
      graph,
      id,
      props,
      this.pointCount,
      featureCount,
      plan,
      sortedSums,
      getOutputViews(output)
    );
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      values,
      weights,
      areas,
      features.kind === 'polygons' ? features.positions : undefined,
      features.kind === 'polygons' ? features.featureOffsets : undefined,
      features.kind === 'polygons' ? features.polygonOffsets : undefined,
      features.kind === 'polygons' ? features.ringOffsets : undefined,
      features.kind === 'feature-rows' ? features.pointFeatureRows : undefined,
      ...(features.kind === 'feature-rows' ? Object.values(features.status ?? {}) : []),
      ...getOutputViews(output)
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      features.kind === 'polygons' &&
      hasLonLatPolygonStatus(output.status)
    ) {
      const status = output.status!;
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
          positions: features.positions,
          ringOffsets: features.ringOffsets,
          status
        })
      );
    }

    const uint32Rows = (name: string, view: GraphDataView<'uint32'> | undefined, needed: boolean) =>
      view ??
      (needed ? createTransientView(graph, `${id}-${name}`, 'uint32', featureCount) : undefined);
    const float32Rows = (
      name: string,
      view: GraphDataView<'float32'> | undefined,
      needed: boolean
    ) =>
      view ??
      (needed ? createTransientView(graph, `${id}-${name}`, 'float32', featureCount) : undefined);

    const counts = uint32Rows('feature-counts', output.counts, plan.needCounts);
    const valueCounts = uint32Rows(
      'feature-value-counts',
      output.valueCounts,
      plan.needValueCounts
    );
    const sums = float32Rows('feature-sums', output.sums, plan.needSums);
    const weightSums = float32Rows('feature-weight-sums', output.weightSums, plan.needWeightSums);
    const means = float32Rows('feature-means', output.means, plan.needMeans);
    const minima = float32Rows('feature-minima', output.minima, plan.needMinima);
    const maxima = float32Rows('feature-maxima', output.maxima, plan.needMaxima);
    const densities = float32Rows('feature-densities', output.densities, plan.needDensities);
    const weightedDensities = float32Rows(
      'feature-weighted-densities',
      output.weightedDensities,
      plan.needWeightedDensities
    );

    // 1. Membership.
    let pointFeatureRows: GPUUint32Rows;
    if (features.kind === 'polygons') {
      pointFeatureRows =
        output.pointFeatureRows ??
        createTransientUint32Rows(graph, `${id}-point-feature-rows`, props.positions!);
      const overflow =
        output.status?.overflow ?? createTransientView(graph, `${id}-overflow`, 'uint32', 1);
      nodes.push(
        ...new GPUPointInPolygonJoin({
          id: `${id}-join`,
          spatialContext: getPointInPolygonContainmentContext(this.spatialContext),
          polygonsSpatialContext: props.featuresSpatialContext,
          points: {kind: 'points', positions: props.positions!},
          polygons: {
            kind: 'polygons',
            positions: features.positions,
            featureOffsets: features.featureOffsets,
            polygonOffsets: features.polygonOffsets,
            ringOffsets: features.ringOffsets,
            sourceIds: features.sourceIds
          },
          candidateCapacity: features.candidateCapacity,
          leafCapacity: features.leafCapacity,
          boundaryRule: features.boundaryRule,
          output: {
            pointFeatureRows,
            polygonAssignmentCounts: counts,
            candidateTruncated: output.candidateTruncated,
            status: {
              overflow,
              candidateOverflow: output.status?.candidateOverflow,
              uncertainCount: output.status?.uncertainCount
            }
          }
        }).getCommandNodes(graph)
      );
    } else {
      pointFeatureRows = features.pointFeatureRows;
      if (counts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-count-aggregation`,
            keys: getZonalAggregationView(pointFeatureRows),
            output: counts
          }).getCommandNodes(graph)
        );
      }
      for (const [name, suffix] of [
        ['overflow', 'overflow'],
        ['candidateOverflow', 'candidate-overflow'],
        ['uncertainCount', 'uncertain-count'],
        ['antimeridianUncertainCount', 'antimeridianUncertainCount'],
        ['poleUncertainCount', 'poleUncertainCount']
      ] as const) {
        const target = output.status?.[name];
        if (!target) continue;
        const source = features.status?.[name];
        nodes.push(
          source
            ? createWGSLKernelNode<Parameters>(graph, {
                id: `${id}-${suffix}`,
                operation: OPERATION,
                invocationCount: 1,
                bindings: [
                  {name: 'inputStatus', view: source, type: 'u32', access: 'read'},
                  {name: 'outputStatus', view: target, type: 'u32', access: 'read_write'}
                ],
                body: 'outputStatus[outputStatusOffset] = inputStatus[inputStatusOffset];'
              })
            : createFillNode<Parameters>(graph, {
                id: `${id}-${suffix}`,
                operation: OPERATION,
                view: target,
                type: 'u32',
                value: '0u',
                componentCount: 1
              })
        );
      }
    }

    // 2. Per-point validity and contributions.
    let valueMask: GPUUint32Rows | undefined;
    let sumContributions: ReturnType<typeof createTransientFloat32Rows> | undefined;
    let weightContributions: ReturnType<typeof createTransientFloat32Rows> | undefined;
    if (plan.needValueStatistics && values) {
      valueMask = createTransientUint32Rows(graph, `${id}-value-mask`, values);
      if (plan.needSums && (weights || sumOrder === 'sorted')) {
        sumContributions = createTransientFloat32Rows(graph, `${id}-sum-contributions`, values);
      }
      if (plan.needWeightSums && sumOrder === 'sorted') {
        weightContributions = createTransientFloat32Rows(
          graph,
          `${id}-weight-contributions`,
          values
        );
      }
      nodes.push(
        ...createZonalStatisticsPrepareNodes<Parameters>(graph, {
          id,
          values,
          weights,
          valueMask,
          sumContributions,
          weightContributions
        })
      );
    }

    // 3. Polygon areas.
    let featureAreas = areas;
    if (features.kind === 'polygons' && plan.needGPUAreas) {
      featureAreas = float32Rows('feature-areas', output.featureAreas, true)!;
      nodes.push(
        createZonalStatisticsAreaNode<Parameters>(graph, {
          id: `${id}-area`,
          featureCount,
          positions: features.positions,
          featureOffsets: features.featureOffsets,
          polygonOffsets: features.polygonOffsets,
          ringOffsets: features.ringOffsets,
          areas: featureAreas
        })
      );
    }

    // 4. Aggregations over valid rows.
    if (valueMask && values) {
      if (valueCounts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-value-count-aggregation`,
            keys: getZonalAggregationView(pointFeatureRows),
            mask: getZonalAggregationView(valueMask),
            output: valueCounts
          }).getCommandNodes(graph)
        );
      }
      const statistics: [
        string,
        'sum' | 'min' | 'max',
        typeof values | undefined,
        GraphDataView<'float32'> | undefined
      ][] = [
        [
          'sum',
          'sum',
          weights ? sumContributions : values,
          sumOrder === 'atomic' ? sums : undefined
        ],
        ['weight-sum', 'sum', weights, sumOrder === 'atomic' ? weightSums : undefined],
        ['minimum', 'min', values, minima],
        ['maximum', 'max', values, maxima]
      ];
      for (const [name, operation, source, destination] of statistics) {
        if (source && destination) {
          nodes.push(
            ...new GPUGroupAggregation({
              id: `${id}-${name}-aggregation`,
              keys: getZonalAggregationView(pointFeatureRows),
              values: getZonalAggregationView(source),
              mask: getZonalAggregationView(valueMask),
              output: destination,
              operation
            }).getCommandNodes(graph)
          );
        }
      }
    }
    if (sortedSums) {
      nodes.push(
        ...getSortedSegmentSumNodes<Parameters>(graph, {
          id,
          operation: OPERATION,
          segmentCount: featureCount,
          segmentKeys: pointFeatureRows as GraphDataView<'uint32'>,
          sumContributions: sumContributions as GraphDataView<'float32'> | undefined,
          weightContributions: weightContributions as GraphDataView<'float32'> | undefined,
          sums: plan.needSums ? sums : undefined,
          weightSums: plan.needWeightSums ? weightSums : undefined
        })
      );
    }

    // 5. Finalize derived statistics.
    if (plan.needMeans) {
      nodes.push(
        createZonalStatisticsMeanNode<Parameters>(graph, {
          id: `${id}-means`,
          featureCount,
          sums: sums!,
          valueCounts: weights ? undefined : valueCounts,
          weightSums: weights ? weightSums : undefined,
          means: means!
        })
      );
    }
    if (plan.needDensities) {
      nodes.push(
        createZonalStatisticsDensityNode<Parameters>(graph, {
          id: `${id}-densities`,
          featureCount,
          counts: counts!,
          areas: featureAreas!,
          densities: densities!
        })
      );
    }

    // 5b. Variances. A float32 mean of values near 1e7 over many rows can be several units off,
    // and `sum (v - mean)^2` then carries `n * error^2`. Pass one sums `w * (v - mean)` to refine
    // the mean; pass two sums `w * (v - refined)^2` and `w * (v - refined)`, and the variance node
    // subtracts the corrected two-pass term `(sum w d)^2 / V1` (Chan, Golub and LeVeque).
    if (plan.needVariances && valueMask && values) {
      const packedValues = values as GraphDataView<'float32'>;
      const packedWeights = weights as GraphDataView<'float32'> | undefined;
      const packedMask = valueMask as GraphDataView<'uint32'>;
      const packedRows = pointFeatureRows as GraphDataView<'uint32'>;
      const pointRows = (name: string) =>
        createTransientView(graph, `${id}-${name}`, 'float32', this.pointCount);
      const featureRows = (name: string) =>
        createTransientView(graph, `${id}-${name}`, 'float32', featureCount);
      const firstLinearDeviations = pointRows('first-linear-deviations');
      const firstLinearDeviationSums = featureRows('first-linear-deviation-sums');
      const refinedMeans = featureRows('refined-mean-rows');
      const squaredDeviations = pointRows('deviations');
      const linearDeviations = pointRows('linear-deviations');
      const weightSquares = plan.needWeightSquareSums ? pointRows('weight-squares') : undefined;
      const squaredDeviationSums = featureRows('squared-deviation-sums');
      const linearDeviationSums = featureRows('linear-deviation-sums');
      const weightSquareSums = weightSquares ? featureRows('weight-square-sums') : undefined;
      const minimumWeights = packedWeights ? featureRows('minimum-weights') : undefined;
      if (packedWeights && minimumWeights) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-minimum-weight-aggregation`,
            keys: packedRows,
            values: packedWeights,
            mask: packedMask,
            output: minimumWeights,
            operation: 'min'
          }).getCommandNodes(graph)
        );
      }
      const sumRows = (
        name: string,
        rows: readonly (readonly [string, GraphDataView<'float32'>, GraphDataView<'float32'>])[]
      ): GPUCommandNode<Parameters>[] => {
        // The public sums may be atomic. Variance corrections need stable reductions even
        // then: a second long sequence of identical atomic additions can reintroduce error.
        return getSortedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-${name}`,
          operation: OPERATION,
          segmentCount: featureCount,
          segmentKeys: packedRows,
          reductions: rows.map(([reductionName, contributions, output]) => ({
            name: reductionName,
            contributions,
            output
          }))
        });
      };
      nodes.push(
        createZonalStatisticsDeviationNode<Parameters>(graph, {
          id: `${id}-first-deviation`,
          featureCount,
          values: packedValues,
          weights: packedWeights,
          valueMask: packedMask,
          pointFeatureRows: packedRows,
          means: means!,
          linearDeviations: firstLinearDeviations
        }),
        ...sumRows('mean-refinement', [
          ['first-linear-deviation', firstLinearDeviations, firstLinearDeviationSums]
        ]),
        createZonalStatisticsRefinedMeanNode<Parameters>(graph, {
          id: `${id}-refined-means`,
          featureCount,
          means: means!,
          linearDeviationSums: firstLinearDeviationSums,
          valueCounts: weights ? undefined : valueCounts,
          weightSums: weights ? weightSums : undefined,
          refinedMeans
        }),
        createZonalStatisticsDeviationNode<Parameters>(graph, {
          id: `${id}-deviation`,
          featureCount,
          values: packedValues,
          weights: packedWeights,
          valueMask: packedMask,
          pointFeatureRows: packedRows,
          means: refinedMeans,
          squaredDeviations,
          linearDeviations,
          weightSquares
        }),
        ...sumRows('variance', [
          ['deviation', squaredDeviations, squaredDeviationSums],
          ['linear-deviation', linearDeviations, linearDeviationSums],
          ...(weightSquares && weightSquareSums
            ? [['weight-square', weightSquares, weightSquareSums] as const]
            : [])
        ]),
        createZonalStatisticsVarianceNode<Parameters>(graph, {
          id: `${id}-variances`,
          featureCount,
          valueCounts: valueCounts!,
          squaredDeviationSums,
          linearDeviationSums,
          weightSums: weightSquares ? weightSums : undefined,
          weightSquareSums,
          minimumWeights,
          variances: output.variances,
          standardDeviations: output.standardDeviations
        })
      );
    }
    if (plan.needWeightedDensities) {
      nodes.push(
        createZonalStatisticsWeightedDensityNode<Parameters>(graph, {
          id: `${id}-weighted-densities`,
          featureCount,
          weightSums: weightSums!,
          areas: featureAreas!,
          weightedDensities: weightedDensities!
        })
      );
    }

    // 6. Extent.
    if (output.extent) {
      const statisticViews = {
        count: undefined,
        sum: sums,
        mean: means,
        minimum: minima,
        maximum: maxima,
        density: densities
      };
      nodes.push(
        ...getZonalStatisticsExtentNodes<Parameters>(graph, {
          id,
          featureCount,
          statistic: output.extentStatistic!,
          counts: counts!,
          statisticView: statisticViews[output.extentStatistic!],
          extent: output.extent
        })
      );
    }
    return nodes;
  }
}

/** Sorted sums need packed point-rate views; chunked inputs fall back to atomic sums. */
function getDefaultZonalSumOrder(props: GPUZonalStatisticsProps): GPUZonalStatisticsSumOrder {
  const {features, output, values, weights} = props;
  const rows = features.kind === 'polygons' ? output.pointFeatureRows : features.pointFeatureRows;
  const views = [features.kind === 'polygons' ? props.positions : undefined, values, weights, rows];
  return views.some(view => isGraphVectorView(view)) ? 'atomic' : 'sorted';
}
/** Adapt foreign-format vectors to the public upstream constructor at its aggregation boundary. */
function getZonalAggregationView<Format extends 'float32' | 'uint32'>(
  view: GraphDataView<Format> | GraphVectorView<Format>
): GraphDataView<Format> | GraphVectorView<Format> {
  return isGraphVectorView(view) ? new GraphVectorView<Format>({...view, data: view.data}) : view;
}
