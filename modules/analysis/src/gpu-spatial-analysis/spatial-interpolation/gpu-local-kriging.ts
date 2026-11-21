// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateInterpolationPorts} from './interpolation-view-validation';
import {
  createInterpolationNeighborhoodLossNodes,
  getInterpolationNeighborhoodLossViews,
  decodeInterpolationPredictionStatus,
  type InterpolationNeighborhoodWeights
} from './interpolation-neighborhood-loss';

import {snapshotInterpolationConstructionProps} from './interpolation-construction-snapshot';

import {validateInterpolationOutputsDisjointFromInputs} from './interpolation-view-validation';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateFiniteNumber,
  validateNonZeroFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_NEIGHBOR_SEARCH_MAXIMUM_K} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {
  planGraphChunkGroups,
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport,
  sliceGraphChunkGroupView
} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createPointNeighborGridIndexNodes,
  POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL
} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {
  getNeighborSearchLatticeWGSL,
  getNearestNeighborSearchWGSL,
  getInsertNeighborWGSL
} from '../neighbor-search/neighbor-search-kernels';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';
import {
  getGPULocalKrigingParameterLength,
  getGPULocalKrigingParameterValues,
  getPerFramePredictionCovarianceWGSL,
  getPredictionCovarianceWGSL,
  validatePredictionCovariance,
  type GPUPredictionCovariance,
  type GPULocalKrigingSettings
} from './prediction-covariance';
import {
  createLocalKrigingParameterNodes,
  type LocalKrigingParameterState
} from './local-kriging-parameters';

export {getPredictionCovarianceWGSL, validatePredictionCovariance};
export type {GPUPredictionCovariance};

/** Bounds the private augmented dense system (up to eight drift columns) and cubic solve work. */
export const GPU_LOCAL_KRIGING_MAXIMUM_NEIGHBOR_COUNT = 32;
/** Bounds the private quadrature weights and the quadratic within-support covariance evaluation. */
export const GPU_LOCAL_KRIGING_MAXIMUM_SUPPORT_COUNT = 64;

/**
 * Largest sample count that {@link GPULocalKrigingProps.sampleIndex} `'auto'` scans exhaustively.
 * Above it the candidates come from a `GPUNeighborSearch` grid. The value follows the brute-force
 * cut-over of the other indexed contributors (`URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT`).
 */
export const GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT = 4096;

/**
 * Largest `samples * queries` distance tests of an explicit `sampleIndex: 'brute-force'`. One query
 * thread scans every sample, so the dispatch grows with the product; the value is the
 * exhaustive-scan ceiling of the point-cloud contributors (`GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK`).
 */
export const GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK = 2 ** 32;

/**
 * Smallest number of extra nearest candidates the grid index returns beyond `neighborCount`; the
 * index returns `neighborCount + max(neighborCount, 8)`. The index orders targets by float32
 * `dx^2 + dy^2` and the kriging kernel, like the scan, by `distance()` with ties to the lower row.
 * The extra candidates keep a distance tie at the `neighborCount`-th neighbor inside the list for
 * lattice-like inputs (rings of 4, 8 or 12 equidistant samples), so the per-query rescan below
 * stays rare.
 */
const LOCAL_KRIGING_INDEX_MINIMUM_CANDIDATE_MARGIN = 8;

/**
 * Relative gap between the farthest grid candidate and the retained boundary below which the
 * kriging kernel rescans every sample for that query. 1e-6 is about eight float32 steps, more than
 * the rounding difference between the index's `dx^2 + dy^2` and WGSL `distance()` (whose `sqrt`
 * may be 2.5 steps off).
 */
const LOCAL_KRIGING_INDEX_TIE_TOLERANCE = 1e-6;
/** Covers the published shared grid build, bounds reduction, covariance upload and packing. */
const LOCAL_KRIGING_SHARED_COMMAND_NODE_ALLOWANCE = 2048;
/** Local dense systems retain the packed route's 32-thread workgroup size. */
const LOCAL_KRIGING_QUERY_WORKGROUP_SIZE = 32;

/** Candidates per query that the grid index lists. */
function getSampleIndexCandidateCount(neighborCount: number): number {
  return Math.min(
    neighborCount + Math.max(neighborCount, LOCAL_KRIGING_INDEX_MINIMUM_CANDIDATE_MARGIN),
    GPU_NEIGHBOR_SEARCH_MAXIMUM_K
  );
}

/**
 * Queries per grid-index chunk: each chunk's candidate rows share one binding with a copy of the
 * sample positions, which must fit `bindingByteLimit`. Returns 0 when not even one query fits.
 */
function getSampleIndexChunkQueryCount(
  bindingByteLimit: number,
  sampleCount: number,
  candidateCount: number
): number {
  return Math.max(0, Math.floor((bindingByteLimit - sampleCount * 8) / (candidateCount * 4)));
}

/** Local kriging measures planar Euclidean distances only. */
const LOCAL_KRIGING_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance' as const
});

/**
 * How {@link GPULocalKriging} finds the neighbor candidates of each query when no `neighborhoods`
 * are supplied: `'brute-force'` scans every sample, `'grid'` takes the nearest candidates from a
 * `GPUNeighborSearch` grid built over the usable samples, `'auto'` scans up to
 * {@link GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT} samples and uses the grid above it.
 */
export type GPULocalKrigingSampleIndex = 'auto' | 'brute-force' | 'grid';

/**
 * Explicit local, planar prediction contract. Counts specialize the graph. Covariance parameters
 * always use a runtime buffer, supplied by the caller or initialized from the convenience props.
 */
export type GPULocalKrigingProps<
  SupportRows extends
    | GraphDataView<'float32x2' | 'float32x4'>
    | GraphVectorView<'float32x2' | 'float32x4'> = GraphDataView<'float32x2' | 'float32x4'>,
  ValueRows extends
    | GraphDataView<'float32x4'>
    | GraphVectorView<'float32x4'> = GraphDataView<'float32x4'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  WeightRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  DriftRows extends
    | GraphDataView<'float32' | 'float32x4'>
    | GraphVectorView<'float32' | 'float32x4'> = GraphDataView<'float32' | 'float32x4'>
> = {
  id?: string;
  /**
   * Coordinate space of `positions` and `supports`. Compile-time. Omitted means native planar
   * coordinates with Euclidean distance (unchanged behavior). Only planar coordinates with the
   * native metric are accepted; longitude/latitude throws, so project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Rows [value, observationErrorVariance, variableIndex (0..variableCount-1), validity (> 0 includes; zero, negative and NaN skip)]. */
  observations: GraphDataView<'float32x4'>;
  /** Query-major quadrature: float32x2 has equal weights; float32x4 stores [x,y,weight,reserved]. */
  supports: SupportRows;
  supportCount?: number;
  mode: 'simple' | 'ordinary' | 'universal' | 'external-drift' | 'co-kriging';
  /**
   * Initial covariance of the univariate modes and of the two-by-two `coregionalization`
   * shortcut. Required unless `parameters` is supplied, in which case it is ignored. Update with
   * `setCovariance` between submitted frames; the compiled shader does not change.
   */
  covariance?: GPUPredictionCovariance;
  /**
   * Per-frame covariance buffer written by `getGPULocalKrigingParameterValues`: model, nugget, sill
   * and range change between frames without a recompile. Counts one more storage binding. Replaces
   * `covariance`, `coregionalization` and the values of `coregionalizationModels`; a buffer that
   * was never written reports status 1 for every query.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Covariance structures in `parameters`, 1..4, for co-kriging. Compile time. Defaults to the
   * length of `coregionalizationModels`, else 1.
   */
  covarianceStructureCount?: number;
  /** Known means for simple or known-mean co-kriging. Defaults to zero. */
  means?: readonly number[];
  /** Number of response variables for co-kriging, 2..4. */
  variableCount?: number;
  /** Separate unknown constant for every variable, or supplied known means (default). */
  coKrigingMean?: 'known' | 'ordinary';
  /** Sum of up to four PSD covariance structures, with row-major square loading factors. */
  coregionalizationModels?: readonly {
    covariance: GPUPredictionCovariance;
    loadings: readonly number[];
  }[];
  /** Two-by-two row-major loading matrix. B = L L^T guarantees a PSD coregionalization model. */
  coregionalization?: readonly [number, number, number, number];
  targetVariable?: number;
  /**
   * Optional candidate rows supplied by a resident neighbor index, replacing the scan of every
   * sample (query-major with `neighborhoodCapacity` slots, or CSR with `neighborhoodOffsets`).
   * 0xffffffff pads. The nearest `neighborCount` candidates to the support center are retained.
   * Each extra binding counts against the device storage-buffer limit of the kernel.
   */
  neighborhoods?: GraphDataView<'uint32'>;
  /** Resident CSR including optional live producer overflow/candidateOverflow evidence. */
  neighborhoodWeights?: InterpolationNeighborhoodWeights;
  /** Producer loss for legacy fixed-width or CSR candidate ports. */
  neighborhoodStatus?: InterpolationNeighborhoodWeights['status'];
  /** Optional CSR offsets (query count + 1) replacing fixed candidate row widths. */
  neighborhoodOffsets?: GraphDataView<'uint32'>;
  /** Fixed candidate slots per query. Each row must contain distinct sample indices. */
  neighborhoodCapacity?: number;
  neighborCount?: number;
  /**
   * How candidates are found when `neighborhoods` is omitted. Compile-time, default `'auto'`; see
   * {@link GPULocalKrigingSampleIndex}. Every choice retains the same neighbors in the same order
   * (the nearest usable samples to the support center by `distance()`, ties to the lower sample
   * row), so results do not depend on it, only speed. `'grid'` with `neighborhoods` throws;
   * `'brute-force'` is rejected above {@link GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK}
   * sample-query tests. `'auto'` scans only when there are at most
   * {@link GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT} samples and at most that many tests,
   * and uses the grid otherwise, so it never runs an uncapped scan; it throws only when neither
   * fits (packed samples exceed the device binding limit). The grid
   * kernel binds as many storage buffers as the scan (sample positions travel in the candidate
   * binding) and splits the queries into chunks whose candidate rows fit one binding.
   */
  sampleIndex?: GPULocalKrigingSampleIndex;
  searchRadius?: number;
  /**
   * What a query with no usable sample in its neighbourhood reports. Compile-time.
   * `'insufficient'` (default): status 1 and NaN. `'prior'`, mode `'simple'` only: status 0 with
   * the known mean, the prior (point or block) variance, pivot ratio 1 and retained count 0, which
   * is the simple kriging predictor of an empty neighbourhood. `GPUVecchiaLogLikelihood` uses it
   * so that a site whose conditioning rows are all excluded contributes its marginal term.
   */
  emptyNeighborhood?: 'insufficient' | 'prior';
  /** External drift includes all supplied columns, including an intercept if wanted; 1..8; flat float32 stores row-major driftCount columns. */
  driftCount?: number;
  sampleDrift?: GraphDataView<'float32' | 'float32x4'>;
  /** Query-major support-averaged external drift, one row per query. */
  queryDrift?: DriftRows;
  output: {
    /** [prediction, latent point/block variance, pivot ratio, retained neighbor count]. */
    values: ValueRows;
    /** 0 success; 1 invalid/missing/insufficient data; 2 singular or unreliable variance; 3 incomplete candidate input (overflow or candidateOverflow). */
    status: StatusRows;
    /** Optional query-major weights in neighbor order (distance, original row); padded with 0. */
    weights?: WeightRows;
  };
};

type PackedLocalKrigingProps = Omit<GPULocalKrigingProps, 'supports' | 'queryDrift' | 'output'> & {
  supports: GraphDataView<'float32x2' | 'float32x4'>;
  queryDrift?: GraphDataView<'float32' | 'float32x4'>;
  output: {
    values: GraphDataView<'float32x4'>;
    status: GraphDataView<'uint32'>;
    weights?: GraphDataView<'float32'>;
  };
};

/**
 * Local covariance kriging with equilibrated partial-pivot bounded solves. Neighborhood selection
 * reads the candidates of a caller's resident neighbor index when `neighborhoods` is supplied;
 * otherwise it scans every sample or reads the nearest candidates from a `GPUNeighborSearch` grid
 * over the usable samples (`sampleIndex`), and retains at most 32 rows. The default never runs an
 * uncapped scan: `'auto'` scans only up to
 * {@link GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT} samples and
 * {@link GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK} sample-query tests (the cap an explicit
 * `'brute-force'` has) and uses the grid beyond either. Both paths keep the same neighbors: the
 * grid kernel ranks its candidates like the scan and rescans the samples for a query whose
 * candidate list ends inside a distance tie. No readback or CPU solve occurs.
 *
 * Compile-time: row counts, `neighborCount`, `supportCount`, `mode`, `driftCount`,
 * `variableCount`, `sampleIndex`, `spatialContext`, `searchRadius`, `means`, and which optional
 * views exist, and the covariance structure count. Supports, query drift and outputs may be
 * chunked, with complete support/drift/weight groups in each query chunk. Shared sample views
 * stay packed. A single sample grid build is reused by every internal/external query chunk;
 * query lengths, row bases and binding offsets are uniforms, with exact per-node dispatch sizes. Per-frame: the contents of every view and
 * the covariance model (write `parameters` or call `setCovariance`). Convenience parameter buffers
 * are owned by the compiled graph and freed when it is destroyed.
 * Universal drift defaults to affine [1, x, y], centered/scaled locally, or uses up to eight
 * caller-supplied columns. Co-kriging sums up to four PSD structures over two to four variables
 * with supplied known means or one unknown constant for each variable. With unknown constants, a
 * secondary variable that has no sample in a neighbourhood drops out of that neighbourhood's
 * system (with every secondary variable absent the result is ordinary kriging of the target), and
 * the system needs only as many retained samples as constraints kept, so one target sample alone
 * predicts. A neighbourhood without a sample of the target variable reports status 1 when it
 * retains fewer samples than constraints kept, and status 2 otherwise. A query with no usable
 * sample at all is status 1, or the prior with `emptyNeighborhood: 'prior'`. Duplicates are separate
 * observations: a zero-error singular system is reported, never silently jittered. Positive
 * measurement errors smooth exact hits. All block quantities use the supplied normalized nonnegative
 * support quadrature, including the within-block covariance and process nugget.
 */
export class GPULocalKriging<
  SupportRows extends
    | GraphDataView<'float32x2' | 'float32x4'>
    | GraphVectorView<'float32x2' | 'float32x4'> = GraphDataView<'float32x2' | 'float32x4'>,
  ValueRows extends
    | GraphDataView<'float32x4'>
    | GraphVectorView<'float32x4'> = GraphDataView<'float32x4'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  WeightRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  DriftRows extends
    | GraphDataView<'float32' | 'float32x4'>
    | GraphVectorView<'float32' | 'float32x4'> = GraphDataView<'float32' | 'float32x4'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  readonly props: GPULocalKrigingProps<SupportRows, ValueRows, StatusRows, WeightRows, DriftRows>;
  readonly neighborCount: number;
  readonly supportCount: number;
  readonly driftCount: number;
  readonly queryCount: number;
  readonly variableCount: number;
  readonly suppliedDrift: boolean;
  readonly covarianceStructureCount: number;
  /** Resolved coordinate space (native planar when `spatialContext` was omitted). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved candidate source: supplied `neighborhoods`, an exhaustive scan, or the grid index. */
  readonly sampleIndex: 'supplied' | 'brute-force' | 'grid';
  private readonly covarianceParameters?: LocalKrigingParameterState;
  /** Optional derived-query chunk ceiling for specialized producers. @internal */
  protected get sampleIndexChunkQueryCount(): number | undefined {
    return undefined;
  }

  constructor(
    props: GPULocalKrigingProps<SupportRows, ValueRows, StatusRows, WeightRows, DriftRows>
  ) {
    this.props = props = snapshotInterpolationConstructionProps(
      props,
      [
        'coKrigingMean',
        'coregionalization',
        'coregionalizationModels',
        'covariance',
        'covarianceStructureCount',
        'driftCount',
        'emptyNeighborhood',
        'id',
        'means',
        'mode',
        'neighborCount',
        'neighborhoodCapacity',
        'neighborhoodOffsets',
        'neighborhoods',
        'neighborhoodWeights',
        'neighborhoodStatus',
        'observations',
        'output',
        'parameters',
        'positions',
        'queryDrift',
        'sampleDrift',
        'sampleIndex',
        'searchRadius',
        'spatialContext',
        'supportCount',
        'supports',
        'targetVariable',
        'variableCount'
      ],
      ['values', 'status', 'weights']
    );
    this.id = props.id ?? 'local-kriging';
    validateInterpolationPorts({id: this.id, contributor: 'GPULocalKriging'}, props, {
      positions: {required: true, viewType: 'data', formats: ['float32x2']},
      observations: {required: true, viewType: 'data', formats: ['float32x4']},
      supports: {required: true, viewType: 'either', formats: ['float32x2', 'float32x4']},
      sampleDrift: {required: false, viewType: 'data', formats: ['float32', 'float32x4']},
      queryDrift: {required: false, viewType: 'either', formats: ['float32', 'float32x4']},
      neighborhoods: {required: false, viewType: 'data', formats: ['uint32']},
      neighborhoodOffsets: {required: false, viewType: 'data', formats: ['uint32']},
      parameters: {required: false, viewType: 'data', formats: ['float32']},
      'output.values': {required: true, viewType: 'either', formats: ['float32x4']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']},
      'output.weights': {required: false, viewType: 'either', formats: ['float32']}
    });
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      LOCAL_KRIGING_SPATIAL_CONTEXT_SUPPORT
    );
    this.neighborCount = props.neighborCount ?? 16;
    this.supportCount = props.supportCount ?? 1;
    this.variableCount = props.mode === 'co-kriging' ? (props.variableCount ?? 2) : 1;
    this.suppliedDrift =
      props.mode === 'external-drift' ||
      (props.mode === 'universal' && Boolean(props.sampleDrift || props.queryDrift));
    this.driftCount = this.suppliedDrift
      ? (props.driftCount ?? 1)
      : props.mode === 'co-kriging' && props.coKrigingMean === 'ordinary'
        ? this.variableCount
        : props.mode === 'universal'
          ? 3
          : props.mode === 'ordinary'
            ? 1
            : props.mode === 'external-drift'
              ? (props.driftCount ?? 1)
              : 0;
    this.queryCount = props.supports.length / this.supportCount;
    validateCapacity(
      this.id,
      'neighborCount',
      this.neighborCount,
      1,
      GPU_LOCAL_KRIGING_MAXIMUM_NEIGHBOR_COUNT
    );
    validateCapacity(
      this.id,
      'supportCount',
      this.supportCount,
      1,
      GPU_LOCAL_KRIGING_MAXIMUM_SUPPORT_COUNT
    );
    validateCapacity(this.id, 'queryCount', this.queryCount, 0);
    if (props.searchRadius !== undefined && props.searchRadius !== Infinity)
      validateNonZeroFiniteNumber(this.id, 'searchRadius', props.searchRadius);
    for (const [index, mean] of (props.means ?? []).entries())
      validateFiniteNumber(this.id, `means[${index}]`, mean);
    if (props.neighborhoods && !props.neighborhoodOffsets)
      validateCapacity(this.id, 'neighborhoodCapacity', props.neighborhoodCapacity ?? 0);
    this.covarianceStructureCount =
      props.covarianceStructureCount ?? props.coregionalizationModels?.length ?? 1;
    if (!props.parameters && !(props.mode === 'co-kriging' && props.coregionalizationModels)) {
      validateRequirement(
        this.id,
        Boolean(props.covariance),
        `covariance or parameters must be supplied (or coregionalizationModels in co-kriging mode)`
      );
      validatePredictionCovariance(props.covariance!, this.id);
    }
    validateRequirement(
      this.id,
      !(
        props.parameters &&
        (props.parameters.format !== 'float32' ||
          !Number.isInteger(this.covarianceStructureCount) ||
          this.covarianceStructureCount < 1 ||
          this.covarianceStructureCount > 4 ||
          (props.mode !== 'co-kriging' && this.covarianceStructureCount !== 1) ||
          props.parameters.length <
            getGPULocalKrigingParameterLength(this.covarianceStructureCount, this.variableCount))
      ),
      `parameters must be float32 with getGPULocalKrigingParameterLength(structureCount, variableCount) elements; covarianceStructureCount must be an integer in [1, 4] and equal 1 outside co-kriging`
    );
    validateRequirement(
      this.id,
      !(
        this.driftCount < 0 ||
        this.driftCount > 8 ||
        !Number.isInteger(this.driftCount) ||
        props.positions.length !== props.observations.length ||
        props.output.values.length < this.queryCount ||
        props.output.status.length < this.queryCount ||
        (props.output.weights && props.output.weights.length < this.queryCount * this.neighborCount)
      ),
      `driftCount must be an integer in [0, 8]; observations.length must equal positions.length; output.values/status must cover queryCount and output.weights must cover queryCount * neighborCount`
    );
    validateRequirement(
      this.id,
      !(
        (props.neighborhoodOffsets &&
          (!props.neighborhoods || props.neighborhoodOffsets.length !== this.queryCount + 1)) ||
        (props.neighborhoods &&
          !props.neighborhoodOffsets &&
          props.neighborhoods.length !== this.queryCount * (props.neighborhoodCapacity ?? 0))
      ),
      `neighborhoodOffsets requires neighborhoods and queryCount + 1 rows; fixed neighborhoods.length must equal queryCount * neighborhoodCapacity`
    );
    validateRequirement(
      this.id,
      !(
        !['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'].includes(props.mode) ||
        !(props.searchRadius === undefined || props.searchRadius > 0)
      ),
      `mode must be simple, ordinary, universal, external-drift or co-kriging; searchRadius must be positive when supplied`
    );
    validateRequirement(
      this.id,
      !(
        this.suppliedDrift &&
        (!props.sampleDrift ||
          !props.queryDrift ||
          props.sampleDrift.length !==
            props.positions.length *
              (props.sampleDrift.format === 'float32' ? this.driftCount : 1) ||
          props.queryDrift.length !==
            this.queryCount * (props.queryDrift.format === 'float32' ? this.driftCount : 1) ||
          (this.driftCount > 4 &&
            (props.sampleDrift.format !== 'float32' || props.queryDrift.format !== 'float32')) ||
          this.driftCount < 1)
      ),
      `sampleDrift and queryDrift must provide driftCount columns per sample/query; driftCount must be in [1, 8], with scalar float32 required above 4 columns`
    );
    validateRequirement(
      this.id,
      !(
        props.mode === 'co-kriging' &&
        !props.parameters &&
        !props.coregionalization &&
        !props.coregionalizationModels
      ),
      `co-kriging requires parameters, coregionalization or coregionalizationModels with PSD loading factors`
    );
    validateRequirement(
      this.id,
      !![...(props.means ?? [0, 0]), ...(props.coregionalization ?? [1, 0, 0, 1])].every(
        Number.isFinite
      ),
      `means and coregionalization entries must be finite`
    );
    const structures = props.coregionalizationModels;
    validateRequirement(
      this.id,
      !(
        this.variableCount < (props.mode === 'co-kriging' ? 2 : 1) ||
        this.variableCount > 4 ||
        !Number.isInteger(this.variableCount) ||
        !Number.isInteger(props.targetVariable ?? 0) ||
        (props.targetVariable ?? 0) < 0 ||
        (props.targetVariable ?? 0) >= this.variableCount ||
        (props.means && props.means.length < this.variableCount) ||
        (props.coKrigingMean && !['known', 'ordinary'].includes(props.coKrigingMean)) ||
        (props.mode === 'co-kriging' &&
          this.variableCount !== 2 &&
          !structures &&
          !props.parameters) ||
        (structures && (structures.length < 1 || structures.length > 4))
      ),
      `variableCount must be an integer in [2, 4] for co-kriging or 1 otherwise; targetVariable must be an integer in [0, variableCount); means must cover variableCount; coKrigingMean must be known or ordinary; 1..4 coregionalizationModels are required above 2 variables without parameters`
    );
    for (const structure of structures ?? []) {
      validatePredictionCovariance(structure.covariance, this.id);
      validateRequirement(
        this.id,
        !(
          structure.loadings.length !== this.variableCount ** 2 ||
          !structure.loadings.every(Number.isFinite)
        ),
        `coregionalizationModels.loadings must contain variableCount squared finite entries`
      );
    }
    const emptyNeighborhood = props.emptyNeighborhood ?? 'insufficient';
    validateRequirement(
      this.id,
      ['insufficient', 'prior'].includes(emptyNeighborhood),
      "emptyNeighborhood must be 'insufficient' or 'prior'"
    );
    validateRequirement(
      this.id,
      emptyNeighborhood === 'insufficient' || props.mode === 'simple',
      "emptyNeighborhood 'prior' needs mode 'simple' (only a known mean has a prior prediction)"
    );
    const sampleIndex = props.sampleIndex ?? 'auto';
    validateRequirement(
      this.id,
      ['auto', 'brute-force', 'grid'].includes(sampleIndex),
      "sampleIndex must be 'auto', 'brute-force' or 'grid'"
    );
    validateRequirement(
      this.id,
      !(props.neighborhoods && sampleIndex === 'grid'),
      "sampleIndex 'grid' needs neighborhoods to be omitted"
    );
    const sampleCount = props.positions.length;
    if (props.neighborhoods) {
      this.sampleIndex = 'supplied';
    } else if (sampleIndex === 'brute-force') {
      validateMaximumCount(
        this.id,
        "sampleIndex 'brute-force' work (samples times queries)",
        sampleCount * this.queryCount,
        GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK
      );
      this.sampleIndex = 'brute-force';
    } else {
      // 'auto' scans only within both the sample cut-over and the scan's work cap. Otherwise it
      // uses the grid, which binds as many storage buffers as the scan and chunks its candidate
      // rows, so it runs wherever the scan did. The grid needs at least one sample and one query
      // (an empty side has no work). Only the packed sample records constrain the shared index.
      const bruteForceWork = sampleCount * this.queryCount;
      const scanFits =
        sampleCount <= GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT &&
        bruteForceWork <= GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK;
      const bindingByteLimit = (props.positions.buffer.graph as GPUCommandGraph).device.limits
        .maxStorageBufferBindingSize;
      const gridFits =
        sampleCount > 0 &&
        this.queryCount > 0 &&
        sampleCount * 12 <= bindingByteLimit &&
        getSampleIndexChunkQueryCount(
          bindingByteLimit,
          sampleCount,
          getSampleIndexCandidateCount(this.neighborCount)
        ) > 0;
      this.sampleIndex = sampleIndex === 'grid' || (!scanFits && gridFits) ? 'grid' : 'brute-force';
      validateRequirement(
        this.id,
        this.sampleIndex !== 'grid' || (sampleCount > 0 && this.queryCount > 0),
        "sampleIndex 'grid' needs at least one sample and one query"
      );
      if (this.sampleIndex === 'brute-force' && sampleIndex === 'auto') {
        validateMaximumCount(
          this.id,
          "sampleIndex 'auto' scan work (samples times queries) where the grid does not fit",
          bruteForceWork,
          GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK
        );
      }
      if (this.sampleIndexChunkQueryCount !== undefined) {
        validateCapacity(this.id, 'sampleIndexChunkQueryCount', this.sampleIndexChunkQueryCount, 1);
      }
    }
    this.chunkPlan = planGraphChunkGroups(
      {id: this.id, contributor: 'GPULocalKriging'},
      {view: props.supports, rowsPerQuery: this.supportCount},
      {
        supports: {view: props.supports, rowsPerQuery: this.supportCount},
        results: {view: props.output.values, allowPackedCapacity: true},
        statuses: {view: props.output.status, allowPackedCapacity: true},
        weights: props.output.weights
          ? {
              view: props.output.weights,
              rowsPerQuery: this.neighborCount,
              allowPackedCapacity: true
            }
          : undefined,
        queryDrift:
          this.suppliedDrift && props.queryDrift
            ? {
                view: props.queryDrift,
                rowsPerQuery: props.queryDrift.format === 'float32' ? this.driftCount : 1
              }
            : undefined
      }
    );

    const lossViews = getInterpolationNeighborhoodLossViews(this.id, props);
    const sharedInputs = {
      ...Object.fromEntries(lossViews.map((view, index) => [`neighborhoodLoss${index}`, view])),
      positions: props.positions,
      observations: props.observations,
      sampleDrift: props.sampleDrift,
      neighborhoods: props.neighborhoods,
      neighborhoodOffsets: props.neighborhoodOffsets,
      parameters: props.parameters
    };
    validateDisjointWritableViews(this.id, props.output);
    validateInterpolationOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      ...Object.values(sharedInputs),
      props.supports,
      props.queryDrift
    ]);
    const owner = props.positions.buffer.graph as GPUCommandGraph;
    const gridChunkLength = Math.max(
      1,
      Math.min(
        this.sampleIndexChunkQueryCount ?? Infinity,
        getSampleIndexChunkQueryCount(
          owner.device.limits.maxStorageBufferBindingSize,
          sampleCount,
          getSampleIndexCandidateCount(this.neighborCount)
        )
      )
    );
    const dispatchCount =
      this.sampleIndex === 'grid'
        ? this.chunkPlan.chunks.reduce(
            (total, chunk) => total + Math.ceil(chunk.length / gridChunkLength) * 2,
            0
          )
        : this.chunkPlan.nonemptyChunkCount;
    this.estimatedCommandNodeCount =
      LOCAL_KRIGING_SHARED_COMMAND_NODE_ALLOWANCE +
      dispatchCount +
      lossViews.length * this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    if (!props.parameters) {
      this.covarianceParameters = {
        values: getGPULocalKrigingParameterValues({
          covariance: props.covariance,
          ...(props.mode === 'co-kriging'
            ? {
                coregionalization: props.coregionalization,
                coregionalizationModels: props.coregionalizationModels,
                variableCount: this.variableCount
              }
            : {})
        } as GPULocalKrigingSettings),
        buffers: new Set()
      };
      validateRequirement(
        this.id,
        this.covarianceParameters.values.length ===
          getGPULocalKrigingParameterLength(this.covarianceStructureCount, this.variableCount),
        'covarianceStructureCount must match the initial covariance structures'
      );
    }
  }

  /**
   * Replace convenience covariance values without recompiling. Structure and variable counts
   * stay fixed. Submit preceding encodings before calling, as with `GPUParameterBuffer.write`.
   * With caller-owned `parameters`, write that buffer instead.
   */
  setCovariance(settings: GPULocalKrigingSettings): void {
    const state = this.covarianceParameters;
    validateRequirement(
      this.id,
      Boolean(state),
      'setCovariance requires convenience covariance props'
    );
    const values = getGPULocalKrigingParameterValues(settings);
    validateRequirement(
      this.id,
      values.length === state!.values.length &&
        (settings.variableCount ??
          (settings.coregionalization || settings.coregionalizationModels ? 2 : 1)) ===
          this.variableCount,
      'setCovariance must preserve covarianceStructureCount and variableCount'
    );
    state!.values = values;
    for (const buffer of state!.buffers) buffer.write(values);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, driftCount} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      observations: props.observations,
      supports: props.supports,
      sampleDrift: props.sampleDrift,
      queryDrift: props.queryDrift,
      neighborhoods: props.neighborhoods,
      neighborhoodOffsets: props.neighborhoodOffsets,
      parameters: props.parameters,
      ...props.output
    });
    const bindingCount =
      6 +
      (this.suppliedDrift ? 2 : 0) +
      (props.neighborhoods && this.sampleIndex !== 'grid' ? 1 : 0) +
      (props.neighborhoodOffsets ? 1 : 0) +
      (props.output.weights ? 1 : 0);
    const shouldPackObservations =
      Boolean(this.covarianceParameters) &&
      bindingCount > graph.device.limits.maxStorageBuffersPerShaderStage;
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      bindingCount - Number(shouldPackObservations),
      LOCAL_KRIGING_QUERY_WORKGROUP_SIZE
    );
    const observationLength = props.observations.length * 4;
    const parameterLength = props.parameters?.length ?? this.covarianceParameters!.values.length;
    const packedLength = observationLength + parameterLength;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'covariance parameters',
      parameterLength * 4,
      'reduce covarianceStructureCount'
    );
    if (shouldPackObservations)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'observations plus covariance',
        packedLength * 4,
        'reduce the sample count'
      );
    const sampleCount = props.positions.length;
    const candidateCount = getSampleIndexCandidateCount(this.neighborCount);
    const deviceGridChunkLength = getSampleIndexChunkQueryCount(
      graph.device.limits.maxStorageBufferBindingSize,
      sampleCount,
      candidateCount
    );
    const maximumInputChunkLength = this.chunkPlan.chunks.reduce(
      (maximum, chunk) => Math.max(maximum, chunk.length),
      0
    );
    const maximumGridChunkLength = Math.min(
      maximumInputChunkLength,
      this.sampleIndexChunkQueryCount ?? Infinity,
      deviceGridChunkLength
    );
    if (this.sampleIndex === 'grid') {
      validateRequirement(
        id,
        maximumGridChunkLength > 0,
        "sampleIndex 'grid' candidate binding must hold at least one query and the packed sample positions"
      );
      const columns = Math.max(1, Math.ceil(Math.sqrt(sampleCount / 2)));
      for (const [name, bytes] of Object.entries({
        gridPositions: sampleCount * 8,
        sortedTargets: sampleCount * 12,
        cellOffsets: (columns * columns + 1) * 4,
        candidateRows: maximumGridChunkLength * candidateCount * 4 + sampleCount * 8,
        gridParameters: 12 * 4
      }))
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          bytes,
          'reduce the packed sample count or query chunk size'
        );
    }
    const sampleDriftStride = props.sampleDrift?.format === 'float32' ? driftCount : 4;
    const usableSampleWGSL = `fn isUsableSample(sampleIndex: u32) -> bool {
  let observation = read_observations(sampleIndex);
  let position = read_positions(sampleIndex);
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { return false; }
  ${props.mode === 'co-kriging' ? `if (!isFiniteValue(observation.z) || observation.z < 0.0 || observation.z >= ${this.variableCount}.0 || floor(observation.z) != observation.z) { return false; }` : ''}
  ${this.suppliedDrift ? `for (var column = 0u; column < ${driftCount}u; column++) { if (!isFiniteValue(sampleDrift[sampleDriftOffset + sampleIndex * ${sampleDriftStride}u + column])) { return false; } }` : ''}
  return true;
}`;
    const uploaded = this.covarianceParameters
      ? createLocalKrigingParameterNodes(graph, id, this.covarianceParameters)
      : undefined;
    const parameters = props.parameters ?? uploaded!.parameters;
    const nodes: GPUCommandNode<Parameters>[] = [...(uploaded?.nodes ?? [])];
    let packedObservations: GraphDataView<'float32'> | undefined;
    if (shouldPackObservations) {
      packedObservations = createTransientView(
        graph,
        `${id}-observations-covariance`,
        'float32',
        packedLength
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-pack-observations-covariance`,
          operation: 'GPULocalKriging',
          variant: 'pack-observations-covariance',
          invocationCount: packedLength,
          bindings: [
            {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'packed', view: packedObservations, type: 'f32', access: 'read_write'}
          ],
          body: `if (index < ${observationLength}u) { packed[packedOffset + index] = observations[observationsOffset + index]; }
else { packed[packedOffset + index] = parameters[parametersOffset + index - ${observationLength}u]; }`
        })
      );
    }
    const grid =
      this.sampleIndex === 'grid'
        ? this.getSampleIndexNodes(graph, usableSampleWGSL, maximumGridChunkLength)
        : undefined;
    if (grid) nodes.push(...grid.nodes);
    let dispatchIndex = 0;
    for (const chunk of this.chunkPlan.chunks) {
      const step = grid ? maximumGridChunkLength : Math.max(chunk.length, 1);
      for (let queryStart = 0; queryStart < chunk.length; queryStart += step) {
        const queryCount = Math.min(step, chunk.length - queryStart);
        const borrow = (name: string, width: number) =>
          chunk.views[name]
            ? sliceGraphChunkGroupView(
                graph,
                chunk.views[name],
                queryStart * width,
                queryCount * width
              )
            : undefined;
        const localProps: PackedLocalKrigingProps = {
          ...props,
          supports: borrow('supports', this.supportCount) as PackedLocalKrigingProps['supports'],
          queryDrift: borrow(
            'queryDrift',
            props.queryDrift?.format === 'float32' ? driftCount : 1
          ) as PackedLocalKrigingProps['queryDrift'],
          output: {
            values: borrow('results', 1) as GraphDataView<'float32x4'>,
            status: borrow('statuses', 1) as GraphDataView<'uint32'>,
            weights: borrow('weights', this.neighborCount) as GraphDataView<'float32'> | undefined
          }
        };
        const nodeId = dispatchIndex === 0 ? id : `${id}-chunk-${dispatchIndex}`;
        if (grid)
          nodes.push(this.getGridCandidateNode(graph, localProps, grid, `${nodeId}-candidates`));
        nodes.push(
          this.getKrigingKernelNode(
            graph,
            usableSampleWGSL,
            parameters,
            packedObservations,
            grid
              ? {
                  nodeId,
                  candidateCapacity: candidateCount,
                  candidates: grid.candidates
                }
              : undefined,
            localProps,
            chunk.rowBase + queryStart
          )
        );
        dispatchIndex++;
      }
    }
    const losses = getInterpolationNeighborhoodLossViews(id, props);
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.length > 0)
        nodes.push(
          ...createInterpolationNeighborhoodLossNodes(
            graph,
            `${id}-chunk-${chunk.chunkIndex}`,
            losses,
            {
              values: chunk.views['results'] as GraphDataView<'float32x4'>,
              status: chunk.views['statuses'] as GraphDataView<'uint32'>,
              weights: chunk.views['weights'] as GraphDataView<'float32'> | undefined
            },
            chunk.length
          )
        );
    }
    return nodes;
  }

  /**
   * The kriging kernel over all queries, or over one grid chunk: `grid.candidates` holds the
   * chunk's fixed-width candidate rows followed by the sample positions as float32 bits.
   */
  private getKrigingKernelNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    usableSampleWGSL: string,
    parameters: GraphDataView<'float32'>,
    packedObservations?: GraphDataView<'float32'>,
    grid?: {
      nodeId: string;
      candidateCapacity: number;
      candidates: GraphDataView<'uint32'>;
    },
    localProps?: PackedLocalKrigingProps,
    rowBase: number = 0
  ): GPUCommandNode<Parameters> {
    const {id, neighborCount, supportCount, driftCount} = this;
    const props = localProps ?? (this.props as PackedLocalKrigingProps);
    const queryCount = props.supports.length / supportCount;
    const sampleDriftStride = props.sampleDrift?.format === 'float32' ? driftCount : 4;
    const queryDriftStride = props.queryDrift?.format === 'float32' ? driftCount : 4;
    const candidateNeighbors = grid ? grid.candidates : props.neighborhoods;
    const candidateOffsets = props.neighborhoodOffsets;
    const candidateCapacity = grid ? grid.candidateCapacity : props.neighborhoodCapacity;
    const bindings: WGSLKernelBinding[] = [
      ...(grid
        ? []
        : [
            {
              name: 'positions',
              view: props.positions,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]),
      {
        name: 'observations',
        view: packedObservations ?? props.observations,
        type: 'f32',
        access: 'read'
      },
      {name: 'supports', view: props.supports, type: 'f32', access: 'read'},
      {name: 'results', view: props.output.values, type: 'f32', access: 'read_write'},
      {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'}
    ];
    if (this.suppliedDrift) {
      bindings.push(
        {name: 'sampleDrift', view: props.sampleDrift!, type: 'f32', access: 'read'},
        {name: 'queryDrift', view: props.queryDrift!, type: 'f32', access: 'read'}
      );
    }
    if (candidateOffsets) {
      bindings.push({
        name: 'neighborhoodOffsets',
        view: candidateOffsets,
        type: 'u32',
        access: 'read'
      });
    }
    if (candidateNeighbors) {
      bindings.push({
        name: 'neighborhoods',
        view: candidateNeighbors,
        type: 'u32',
        access: 'read'
      });
    }
    if (!packedObservations) {
      bindings.push({name: 'params', view: parameters, type: 'f32', access: 'read'});
    }
    if (props.output.weights) {
      bindings.push({
        name: 'weights',
        view: props.output.weights,
        type: 'f32',
        access: 'read_write'
      });
    }
    validateGraphViewsBelongToGraph(
      id,
      graph,
      bindings.map(binding => binding.view)
    );
    // `createWGSLKernelNode` rejects more bindings than the device's maxStorageBuffersPerShaderStage
    // or a binding above maxStorageBufferBindingSize, naming this kernel, before any pipeline exists.
    const targetVariable = props.mode === 'co-kriging' ? (props.targetVariable ?? 0) : 0;
    const means = props.means ?? Array.from({length: this.variableCount}, () => 0);
    // Ordinary co-kriging has one unbiasedness constraint per variable. A secondary variable with
    // no retained sampleIndex in a neighbourhood has the vacuous constraint `0 = 0`: kept, it is a zero
    // row that fails the solve, so it is left out and the system reduces towards ordinary kriging
    // of the target (as gstat does). The constraint of the target variable is always kept, so a
    // neighbourhood without a target sample still fails (`sum of no weights = 1` cannot hold).
    const dropsEmptyVariableConstraints =
      props.mode === 'co-kriging' && !this.suppliedDrift && driftCount > 0;
    const driftColumnSelection = dropsEmptyVariableConstraints
      ? `var driftColumns: array<u32, ${driftCount}>;
var driftColumnCount = 0u;
for (var variable = 0u; variable < DRIFT_COUNT; variable++) {
  var present = variable == TARGET_VARIABLE;
  for (var row = 0u; row < retained; row++) {
    if (u32(read_observations(neighbors[row]).z) == variable) { present = true; break; }
  }
  if (present) { driftColumns[driftColumnCount] = variable; driftColumnCount++; }
}`
      : 'let driftColumnCount = DRIFT_COUNT;';
    const drift = this.suppliedDrift
      ? `sampleDrift[sampleDriftOffset + neighbors[row] * ${sampleDriftStride}u + column]`
      : props.mode === 'co-kriging'
        ? `select(0.0, 1.0, u32(observation.z) == ${dropsEmptyVariableConstraints ? 'driftColumns[column]' : 'column'})`
        : props.mode === 'universal'
          ? 'select((read_positions(neighbors[row]) - center)[min(column - 1u, 1u)] / coordinateScale, 1.0, column == 0u)'
          : '1.0';
    const queryDrift = this.suppliedDrift
      ? `queryDrift[queryDriftOffset + index * ${queryDriftStride}u + column]`
      : props.mode === 'co-kriging'
        ? `select(0.0, 1.0, TARGET_VARIABLE == ${dropsEmptyVariableConstraints ? 'driftColumns[column]' : 'column'})`
        : 'select(0.0, 1.0, column == 0u)';
    const parameterOffset = packedObservations
      ? `(observationsOffset + ${props.observations.length * 4}u)`
      : 'paramsOffset';
    const parameterBinding = packedObservations ? 'observations' : 'params';
    const covarianceWGSL = getPerFramePredictionCovarianceWGSL(
      this.covarianceStructureCount,
      this.variableCount
    )
      .replace(/\bparamsOffset\b/g, parameterOffset)
      .replace(/\bparams\b/g, parameterBinding);
    const searchRadius = getWGSLFloatLiteral(
      Number.isFinite(props.searchRadius) ? props.searchRadius! : 3.402823e38
    );
    // Retain by (distance(), original sample row), independent of grid visit order.
    const retainSample = `if (sampleIndex < SAMPLE_COUNT && isUsableSample(sampleIndex)) {
  let position = read_positions(sampleIndex);
  let separation = distance(position, center);
  ${grid ? 'farthestCandidate = max(farthestCandidate, separation);' : ''}
  if (separation <= SEARCH_RADIUS) {
    var insertion = retained;
    for (var neighbor = 0u; neighbor < retained; neighbor++) { if (separation < distances[neighbor] || (separation == distances[neighbor] && sampleIndex < neighbors[neighbor])) { insertion = neighbor; break; } }
    if (insertion < NEIGHBOR_COUNT) {
      var shift = min(retained, NEIGHBOR_COUNT - 1u);
      loop { if (shift <= insertion) { break; } neighbors[shift] = neighbors[shift - 1u]; distances[shift] = distances[shift - 1u]; shift--; }
      neighbors[insertion] = sampleIndex; distances[insertion] = separation; retained = min(retained + 1u, NEIGHBOR_COUNT);
    }
  }
}`;
    // Grid candidates are the CANDIDATE_COUNT nearest usable samples by the index's float32
    // `dx^2 + dy^2`; retention resolves distance() ties explicitly by original row. A sampleIndex outside a full candidate list has an index d^2 at
    // least that of the farthest candidate, so its distance() is at least the farthest candidate's
    // within rounding. When the farthest candidate is not clearly beyond the retained boundary (the
    // NEIGHBOR_COUNT-th distance, or the search radius when fewer are retained), a tie or near-tie
    // may continue past the list: that query rescans every sample, giving exactly the scan's
    // neighbors. Lattice ties rarely reach past the extra candidates, so rescans are rare.
    const selection = grid
      ? `var farthestCandidate = -1.0;
var listed = 0u;
for (var candidate = 0u; candidate < CANDIDATE_COUNT; candidate++) {
  let sampleIndex = neighborhoods[neighborhoodsOffset + candidateQuery * CANDIDATE_COUNT + candidate];
  if (sampleIndex < SAMPLE_COUNT) { listed++; }
  ${retainSample}
}
let boundary = select(SEARCH_RADIUS, distances[NEIGHBOR_COUNT - 1u], retained == NEIGHBOR_COUNT);
if (listed == CANDIDATE_COUNT && farthestCandidate <= boundary * ${getWGSLFloatLiteral(1 + LOCAL_KRIGING_INDEX_TIE_TOLERANCE)}) {
  retained = 0u;
  for (var sampleIndex = 0u; sampleIndex < SAMPLE_COUNT; sampleIndex++) {
    ${retainSample}
  }
}`
      : `${candidateOffsets ? `let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + queryIndex];\nlet candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + queryIndex + 1u];\nif (candidateBegin > candidateEnd || candidateEnd > arrayLength(&neighborhoods) - neighborhoodsOffset) { return; }` : ''}
for (var candidate = ${candidateOffsets ? 'candidateBegin' : '0u'}; candidate < ${candidateOffsets ? 'candidateEnd' : candidateNeighbors ? `${candidateCapacity}u` : 'SAMPLE_COUNT'}; candidate++) {
  let sampleIndex = ${candidateNeighbors ? `neighborhoods[neighborhoodsOffset + ${candidateOffsets ? '' : `queryIndex * ${candidateCapacity}u + `}candidate]` : 'candidate'};
  ${retainSample}
}`;
    const supportAccessors = getPackedPredictionAccessors(
      {
        ...(grid ? {} : {positions: 2 as const}),
        observations: 4,
        supports: props.supports.format === 'float32x4' ? 4 : 2
      },
      ['results']
    );
    const gridDeclarations = grid
      ? `const CANDIDATE_COUNT: u32 = ${grid.candidateCapacity}u;
fn read_positions(row: u32) -> vec2<f32> { return vec2<f32>(bitcast<f32>(neighborhoods[arrayLength(&neighborhoods) - SAMPLE_COUNT * 2u + row * 2u]), bitcast<f32>(neighborhoods[arrayLength(&neighborhoods) - SAMPLE_COUNT * 2u + row * 2u + 1u])); }`
      : '';
    return createGraphChunkGroupKernelNode(graph, {
      id: grid ? grid.nodeId : rowBase === 0 ? id : `${id}-chunk-${rowBase}`,
      pipelineId: id,
      rowBase,
      totalLength: this.queryCount,
      operation: 'GPULocalKriging',
      variant: props.mode,
      bindings,
      invocationCount: queryCount,
      workgroupSize: LOCAL_KRIGING_QUERY_WORKGROUP_SIZE,
      declarations: `${supportAccessors}\n${gridDeclarations}\nconst SEARCH_RADIUS: f32 = ${searchRadius};\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${usableSampleWGSL}
${getEquilibratedSolveWGSL(neighborCount + driftCount)}\n${covarianceWGSL}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const DRIFT_COUNT: u32 = ${driftCount}u;
const SUPPORT_COUNT: u32 = ${supportCount}u;
const SAMPLE_COUNT: u32 = ${props.positions.length}u;
const TARGET_VARIABLE: u32 = ${targetVariable}u;
const MEANS: array<f32, ${means.length}> = array<f32, ${means.length}>(${means.map(getWGSLFloatLiteral).join(', ')});
`,
      body: `let queryIndex = getChunkParameter(0u) + index;
${grid ? 'let candidateQuery = index;' : ''}
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
${props.output.weights ? 'for (var column = 0u; column < NEIGHBOR_COUNT; column++) { weights[weightsOffset + index * NEIGHBOR_COUNT + column] = 0.0; }' : ''}
if (!loadCovarianceParameters()) { return; }
let covarianceScale = ${parameterBinding}[${parameterOffset}];
var center = vec2<f32>(0.0);
var supportWeightSum = 0.0;
var supportWeights: array<f32, ${supportCount}>;
for (var support = 0u; support < SUPPORT_COUNT; support++) {
  let supportValue = read_supports(index * SUPPORT_COUNT + support);
  let point = supportValue.xy;
  let weight = ${props.supports.format === 'float32x4' ? 'supportValue.z' : '1.0'};
  if (!isFiniteValue(weight) || weight < 0.0) { return; }
  supportWeights[support] = weight; supportWeightSum += weight;
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return; }
  center += point * weight;
}
if (!(supportWeightSum > 0.0) || !isFiniteValue(supportWeightSum)) { return; }
center /= supportWeightSum;
for (var support = 0u; support < SUPPORT_COUNT; support++) { supportWeights[support] /= supportWeightSum; }
var neighbors: array<u32, ${neighborCount}>;
var distances: array<f32, ${neighborCount}>;
var retained = 0u;
${selection}
${driftColumnSelection}
${
  props.emptyNeighborhood === 'prior'
    ? `if (retained == 0u) {
  // Simple kriging of an empty neighbourhood: the known mean and the prior variance of the support.
  var priorVariance = 0.0;
  for (var first = 0u; first < SUPPORT_COUNT; first++) {
    for (var second = 0u; second < SUPPORT_COUNT; second++) {
      priorVariance += crossCovariance(read_supports(index * SUPPORT_COUNT + first).xy, read_supports(index * SUPPORT_COUNT + second).xy, TARGET_VARIABLE, TARGET_VARIABLE) * supportWeights[first] * supportWeights[second];
    }
  }
  if (isFiniteValue(priorVariance) && priorVariance >= 0.0) {
    write_results(index, vec4<f32>(MEANS[TARGET_VARIABLE], priorVariance, 1.0, 0.0));
    statuses[statusesOffset + index] = 0u;
  }
  return;
}`
    : ''
}
if (retained == 0u || retained < driftColumnCount) { return; }
let coordinateScale = max(distances[retained - 1u], 1e-20);
var crossValues: array<f32, ${neighborCount}>;
var queryDesign: array<f32, ${Math.max(driftCount, 1)}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  let observation = read_observations(neighbors[row]);
  let variable = ${props.mode === 'co-kriging' ? 'u32(observation.z)' : '0u'};
  for (var column = 0u; column < retained; column++) {
    let otherObservation = read_observations(neighbors[column]);
    matrix[row * SYSTEM_CAPACITY + column] = crossCovariance(position, read_positions(neighbors[column]), variable, ${props.mode === 'co-kriging' ? 'u32(otherObservation.z)' : '0u'}) / covarianceScale;
  }
  matrix[row * SYSTEM_CAPACITY + row] += observation.y / covarianceScale;
  var crossValue = 0.0;
  for (var support = 0u; support < SUPPORT_COUNT; support++) { crossValue += crossCovariance(position, read_supports(index * SUPPORT_COUNT + support).xy, variable, TARGET_VARIABLE) * supportWeights[support]; }
  crossValues[row] = crossValue; right[row] = crossValue / covarianceScale;
  for (var column = 0u; column < driftColumnCount; column++) {
    let value = ${drift};
    matrix[row * SYSTEM_CAPACITY + retained + column] = value;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = value;
  }
}
for (var column = 0u; column < driftColumnCount; column++) {
  let value = ${queryDrift};
  if (!isFiniteValue(value)) { return; }
  queryDesign[column] = value; right[retained + column] = value;
  for (var other = 0u; other < driftColumnCount; other++) { matrix[(retained + column) * SYSTEM_CAPACITY + retained + other] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + driftColumnCount)) { results[resultsOffset + index * 4u + 2u] = pivotRatio; return; }
var prediction = ${driftCount ? '0.0' : 'MEANS[TARGET_VARIABLE]'};
var variance = 0.0;
for (var first = 0u; first < SUPPORT_COUNT; first++) {
  for (var second = 0u; second < SUPPORT_COUNT; second++) {
    variance += crossCovariance(read_supports(index * SUPPORT_COUNT + first).xy, read_supports(index * SUPPORT_COUNT + second).xy, TARGET_VARIABLE, TARGET_VARIABLE) * supportWeights[first] * supportWeights[second];
  }
}
let varianceScale = max(variance, 1e-20);
for (var row = 0u; row < retained; row++) {
  let observation = read_observations(neighbors[row]);
  prediction += solution[row] * (observation.x ${driftCount ? '' : `- MEANS[${props.mode === 'co-kriging' ? 'u32(observation.z)' : '0u'}]`});
  variance -= solution[row] * crossValues[row];
}
for (var column = 0u; column < driftColumnCount; column++) { variance -= solution[retained + column] * queryDesign[column] * covarianceScale; }
if (!isFiniteValue(prediction) || !isFiniteValue(variance) || variance < -1e-4 * varianceScale) { return; }
write_results(index, vec4<f32>(prediction, max(variance, 0.0), pivotRatio, f32(retained)));
statuses[statusesOffset + index] = 0u;
${props.output.weights ? 'for (var row = 0u; row < retained; row++) { weights[weightsOffset + index * NEIGHBOR_COUNT + row] = solution[row]; }' : ''}
`
    });
  }

  /** Builds usable sample positions and the published grid once for all query dispatches. */
  private getSampleIndexNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    usableSampleWGSL: string,
    maximumChunkQueryCount: number
  ) {
    const {props, id} = this;
    const sampleCount = props.positions.length;
    const candidateCount = getSampleIndexCandidateCount(this.neighborCount);
    const parameters = createNeighborSearchDerivedParameterNodes(graph, {
      id: `${id}-index-parameters`,
      operation: 'GPULocalKriging',
      positions: props.positions,
      radius: Infinity
    });
    const gridPositions = createTransientView(
      graph,
      `${id}-index-unit-positions`,
      'float32x2',
      sampleCount
    );
    const latticeWGSL = getNeighborSearchLatticeWGSL('knn', parameters.gridSize);
    const prepare = createWGSLKernelNode(graph, {
      id: `${id}-index-prepare`,
      operation: 'GPULocalKriging',
      variant: 'sample-index-prepare',
      invocationCount: sampleCount,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
        ...(this.suppliedDrift
          ? [
              {
                name: 'sampleDrift',
                view: props.sampleDrift!,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'parameters', view: parameters.parameters, type: 'f32', access: 'read'},
        {name: 'gridPositions', view: gridPositions, type: 'f32', access: 'read_write'}
      ],
      // Encode the reader's cell centre so index assignment and the walk share one cell rule.
      declarations: `${getPackedPredictionAccessors({positions: 2, observations: 4})}\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${latticeWGSL}\n${usableSampleWGSL}\n${POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL}`,
      body: `let lattice = readLattice(); var unitPosition = vec2<f32>(getNaN());
if (lattice.valid && isUsableSample(index)) { unitPosition = getGridUnitPosition(lattice, read_positions(index).x, read_positions(index).y); }
gridPositions[gridPositionsOffset + index * 2u] = unitPosition.x;
gridPositions[gridPositionsOffset + index * 2u + 1u] = unitPosition.y;`
    });
    const index = createPointNeighborGridIndexNodes(graph, {
      id: `${id}-index`,
      positions: gridPositions,
      gridSize: parameters.gridSize,
      rowCount: sampleCount
    });
    const sortedTargets = createTransientView(
      graph,
      `${id}-index-sorted-targets`,
      'uint32',
      sampleCount * 3
    );
    const candidates = createTransientView(
      graph,
      `${id}-index-rows`,
      'uint32',
      maximumChunkQueryCount * candidateCount + sampleCount * 2
    );
    const pack = createWGSLKernelNode(graph, {
      id: `${id}-index-pack-targets`,
      operation: 'GPULocalKriging',
      variant: 'sample-index-pack-targets',
      invocationCount: sampleCount,
      bindings: [
        {name: 'positions', view: props.positions, type: 'u32', access: 'read'},
        {name: 'rows', view: index.sortedRows, type: 'u32', access: 'read'},
        {name: 'count', view: index.gridCount, type: 'u32', access: 'read'},
        {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read_write'},
        {name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}
      ],
      body: `if (index < count[countOffset]) {
  let row = rows[rowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * 3u] = positions[positionsOffset + row * 2u];
  sortedTargets[sortedTargetsOffset + index * 3u + 1u] = positions[positionsOffset + row * 2u + 1u];
  sortedTargets[sortedTargetsOffset + index * 3u + 2u] = row;
}
let positionsBase = arrayLength(&candidates) - ${sampleCount * 2}u;
candidates[positionsBase + index * 2u] = positions[positionsOffset + index * 2u];
candidates[positionsBase + index * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];`
    });
    return {
      nodes: [...parameters.nodes, prepare, ...index.nodes, pack],
      candidates,
      sortedTargets,
      cellOffsets: index.cellOffsets,
      parameters: parameters.parameters,
      gridSize: parameters.gridSize
    };
  }

  /** Lists exact grid candidates into reusable query-sized scratch; shared sample tail is untouched. */
  private getGridCandidateNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: PackedLocalKrigingProps,
    grid: Omit<ReturnType<GPULocalKriging['getSampleIndexNodes']>, 'nodes'>,
    id: string
  ): GPUCommandNode<Parameters> {
    const candidateCount = getSampleIndexCandidateCount(this.neighborCount);
    const latticeWGSL = getNeighborSearchLatticeWGSL('knn', grid.gridSize);
    return createGraphChunkGroupKernelNode(graph, {
      id,
      pipelineId: `${this.id}-index-candidates`,
      operation: 'GPULocalKriging',
      variant: 'sample-index-candidates',
      invocationCount: props.supports.length / this.supportCount,
      workgroupSize: LOCAL_KRIGING_QUERY_WORKGROUP_SIZE,
      bindings: [
        {name: 'supports', view: props.supports, type: 'f32', access: 'read'},
        {name: 'parameters', view: grid.parameters, type: 'f32', access: 'read'},
        {name: 'cellOffsets', view: grid.cellOffsets, type: 'u32', access: 'read'},
        {name: 'sortedTargets', view: grid.sortedTargets, type: 'u32', access: 'read'},
        {name: 'candidates', view: grid.candidates, type: 'u32', access: 'read_write'}
      ],
      declarations: `${getPackedPredictionAccessors({supports: props.supports.format === 'float32x4' ? 4 : 2})}\n${latticeWGSL}\n${IS_FINITE_VALUE_WGSL}\n${getInsertNeighborWGSL(candidateCount)}`,
      body: `for (var slot = 0u; slot < K; slot++) { candidates[candidatesOffset + index * K + slot] = 0xffffffffu; }
var center = vec2<f32>(0.0); var weightSum = 0.0;
for (var support = 0u; support < ${this.supportCount}u; support++) {
  let value = read_supports(index * ${this.supportCount}u + support);
  let weight = ${props.supports.format === 'float32x4' ? 'value.z' : '1.0'};
  if (!isFiniteValue(value.x) || !isFiniteValue(value.y) || !isFiniteValue(weight) || weight < 0.0) { return; }
  center += value.xy * weight; weightSum += weight;
}
if (!(weightSum > 0.0) || !isFiniteValue(weightSum)) { return; }
center /= weightSum;
let x = center.x; let y = center.y; let lattice = readLattice();
if (!lattice.valid || cellOffsets[cellOffsetsOffset + CELL_COUNT] == 0u) { return; }
var found = 0u; var bestDistances: array<f32, ${candidateCount}>; var bestIds: array<u32, ${candidateCount}>;
${getNearestNeighborSearchWGSL('true')}
for (var slot = 0u; slot < found; slot++) { candidates[candidatesOffset + index * K + slot] = bestIds[slot]; }`
    });
  }
}

/** Decode one query status: 0 complete, 1 invalid/insufficient input, 2 singular solve, 3 incomplete candidates. */
export function decodeGPULocalKrigingStatus(
  input: Uint32Array | Uint8Array | ArrayBuffer,
  row: number = 0
) {
  return decodeInterpolationPredictionStatus('decodeGPULocalKrigingStatus', input, row);
}
