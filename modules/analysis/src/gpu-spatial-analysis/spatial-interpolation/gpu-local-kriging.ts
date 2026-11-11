// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
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
  GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPUNeighborSearch
} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {
  getPackedPredictionAccessors,
  validatePredictionViews
} from '../../analysis-core/kernels/gpu-prediction-utils';
import {getEquilibratedSolveWGSL} from './equilibrated-solve-wgsl';
import {
  getCoregionalizationCoefficient,
  getGPULocalKrigingParameterLength,
  getPerFramePredictionCovarianceWGSL,
  getPredictionCovarianceWGSL,
  validatePredictionCovariance,
  type GPUPredictionCovariance
} from './prediction-covariance';

export {getPredictionCovarianceWGSL, validatePredictionCovariance};
export type {GPUPredictionCovariance};

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
 * specialize it unless the `parameters` buffer is supplied, in which case they change per frame
 * without a recompile.
 */
export type GPULocalKrigingProps = {
  id?: string;
  /**
   * Coordinate space of `positions` and `supports`. Compile-time. Omitted means native planar
   * coordinates with Euclidean distance (unchanged behavior). Only planar coordinates with the
   * native metric are accepted; longitude/latitude throws, so project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Rows [value, observationErrorVariance, variableIndex (0..variableCount-1), validity (0 skips)]. */
  observations: GraphDataView<'float32x4'>;
  /** Query-major quadrature: float32x2 has equal weights; float32x4 stores [x,y,weight,reserved]. */
  supports: GraphDataView<'float32x2' | 'float32x4'>;
  supportCount?: number;
  mode: 'simple' | 'ordinary' | 'universal' | 'external-drift' | 'co-kriging';
  /**
   * Compile-time covariance of the univariate modes and of the two-by-two `coregionalization`
   * shortcut. Required unless `parameters` is supplied, in which case it is ignored.
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
   * fits (more than 2^24 samples plus queries, beyond the grid's 128 MiB target binding). The grid
   * kernel binds as many storage buffers as the scan (sample positions travel in the candidate
   * binding) and splits the queries into chunks whose candidate rows fit one binding.
   */
  sampleIndex?: GPULocalKrigingSampleIndex;
  /**
   * Largest number of queries per grid-index chunk. Compile-time; omitted means as many as fit the
   * device's `maxStorageBufferBindingSize`. Exists so tests can exercise chunking on small inputs.
   * @internal
   */
  sampleIndexChunkQueryCount?: number;
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
  queryDrift?: GraphDataView<'float32' | 'float32x4'>;
  output: {
    /** [prediction, latent point/block variance, pivot ratio, retained neighbor count]. */
    values: GraphDataView<'float32x4'>;
    /** 0 success; 1 invalid/missing/insufficient data; 2 singular or unreliable variance. */
    status: GraphDataView<'uint32'>;
    /** Optional query-major weights in neighbor order (distance, original row); padded with 0. */
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
 * candidate list ends inside a distance tie. No readback, implicit repacking, or CPU solve occurs.
 *
 * Compile-time: row counts, `neighborCount`, `supportCount`, `mode`, `driftCount`,
 * `variableCount`, `sampleIndex`, `spatialContext`, `searchRadius`, `means`, and which optional
 * views exist; without `parameters` also the covariance model. Per-frame: the contents of every
 * view, and the covariance model when `parameters` is supplied.
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
export class GPULocalKriging implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocalKrigingProps;
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

  constructor(props: GPULocalKrigingProps) {
    this.props = props;
    this.id = props.id ?? 'local-kriging';
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
    validateCapacity(this.id, 'neighborCount', this.neighborCount, 1);
    validateCapacity(this.id, 'supportCount', this.supportCount, 1);
    validateCapacity(this.id, 'queryCount', this.queryCount, 0);
    this.covarianceStructureCount =
      props.covarianceStructureCount ?? props.coregionalizationModels?.length ?? 1;
    if (props.covariance) {
      validatePredictionCovariance(props.covariance);
    } else if (!props.parameters) {
      throw new Error(`${this.id} needs covariance or parameters`);
    }
    if (
      props.parameters &&
      (props.parameters.format !== 'float32' ||
        !Number.isInteger(this.covarianceStructureCount) ||
        this.covarianceStructureCount < 1 ||
        this.covarianceStructureCount > 4 ||
        (props.mode !== 'co-kriging' && this.covarianceStructureCount !== 1) ||
        props.parameters.length <
          getGPULocalKrigingParameterLength(this.covarianceStructureCount, this.variableCount))
    ) {
      throw new Error(`${this.id} invalid parameters`);
    }
    if (
      this.neighborCount > 32 ||
      this.supportCount > 64 ||
      this.driftCount < 0 ||
      this.driftCount > 8 ||
      !Number.isInteger(this.driftCount) ||
      props.positions.length !== props.observations.length ||
      props.output.values.length < this.queryCount ||
      props.output.status.length < this.queryCount ||
      (props.output.weights && props.output.weights.length < this.queryCount * this.neighborCount)
    ) {
      throw new Error(`${this.id} invalid capacities`);
    }
    if (
      (props.neighborhoodOffsets &&
        (!props.neighborhoods || props.neighborhoodOffsets.length !== this.queryCount + 1)) ||
      (props.neighborhoods &&
        !props.neighborhoodOffsets &&
        props.neighborhoods.length !== this.queryCount * (props.neighborhoodCapacity ?? 0))
    ) {
      throw new Error(`${this.id} invalid neighborhoods`);
    }
    if (
      !['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'].includes(props.mode) ||
      !(props.searchRadius === undefined || props.searchRadius > 0)
    ) {
      throw new Error(`${this.id} invalid mode or radius`);
    }
    if (
      this.suppliedDrift &&
      (!props.sampleDrift ||
        !props.queryDrift ||
        props.sampleDrift.length !==
          props.positions.length * (props.sampleDrift.format === 'float32' ? this.driftCount : 1) ||
        props.queryDrift.length !==
          this.queryCount * (props.queryDrift.format === 'float32' ? this.driftCount : 1) ||
        (this.driftCount > 4 &&
          (props.sampleDrift.format !== 'float32' || props.queryDrift.format !== 'float32')) ||
        this.driftCount < 1)
    ) {
      throw new Error(`${this.id} missing external drift`);
    }
    if (
      props.mode === 'co-kriging' &&
      !props.parameters &&
      !props.coregionalization &&
      !props.coregionalizationModels
    ) {
      throw new Error(`${this.id} co-kriging requires PSD loading factors`);
    }
    if (
      ![...(props.means ?? [0, 0]), ...(props.coregionalization ?? [1, 0, 0, 1])].every(
        Number.isFinite
      )
    ) {
      throw new Error(`${this.id} nonfinite model`);
    }
    const structures = props.coregionalizationModels;
    if (
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
    ) {
      throw new Error(`${this.id} invalid multivariate model`);
    }
    for (const structure of structures ?? []) {
      validatePredictionCovariance(structure.covariance);
      if (
        structure.loadings.length !== this.variableCount ** 2 ||
        !structure.loadings.every(Number.isFinite)
      ) {
        throw new Error(`${this.id} invalid PSD loading dimensions`);
      }
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
      // (an empty side has no work) and its search targets (samples then centers) in one binding.
      const bruteForceWork = sampleCount * this.queryCount;
      const scanFits =
        sampleCount <= GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT &&
        bruteForceWork <= GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK;
      const gridFits =
        sampleCount > 0 &&
        this.queryCount > 0 &&
        (sampleCount + this.queryCount) * 8 <= GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE &&
        getSampleIndexChunkQueryCount(
          GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE,
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
      if (props.sampleIndexChunkQueryCount !== undefined) {
        validateCapacity(
          this.id,
          'sampleIndexChunkQueryCount',
          props.sampleIndexChunkQueryCount,
          1
        );
      }
    }
    validatePredictionViews(
      this.id,
      [
        props.positions,
        props.observations,
        props.supports,
        ...[props.sampleDrift, props.queryDrift].filter(
          (view): view is GraphDataView<'float32' | 'float32x4'> => Boolean(view)
        ),
        ...(props.neighborhoods ? [props.neighborhoods] : []),
        ...(props.neighborhoodOffsets ? [props.neighborhoodOffsets] : []),
        ...(props.parameters ? [props.parameters] : [])
      ],
      [
        props.output.values,
        props.output.status,
        ...(props.output.weights ? [props.output.weights] : [])
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, driftCount} = this;
    const sampleDriftStride = props.sampleDrift?.format === 'float32' ? driftCount : 4;
    // The one definition of a usable sample, shared by the scan and the grid index mask.
    const usableSampleWGSL = `fn isUsableSample(sample: u32) -> bool {
  let observation = read_observations(sample);
  let position = read_positions(sample);
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { return false; }
  ${props.mode === 'co-kriging' ? `if (!isFiniteValue(observation.z) || observation.z < 0.0 || observation.z >= ${this.variableCount}.0 || floor(observation.z) != observation.z) { return false; }` : ''}
  ${this.suppliedDrift ? `for (var column = 0u; column < ${driftCount}u; column++) { if (!isFiniteValue(sampleDrift[sampleDriftOffset + sample * ${sampleDriftStride}u + column])) { return false; } }` : ''}
  return true;
}`;
    // The grid runs one kriging kernel per query chunk; each reads its candidate rows, followed by
    // a copy of the sample positions, from one binding, so it binds as many buffers as the scan.
    const gridIndex =
      this.sampleIndex === 'grid' ? this.getSampleIndexNodes(graph, usableSampleWGSL) : undefined;
    if (gridIndex) {
      const nodes: GPUCommandNode<Parameters>[] = [...gridIndex.nodes];
      gridIndex.chunks.forEach((chunk, chunkIndex) => {
        nodes.push(
          this.getKrigingKernelNode(graph, usableSampleWGSL, {
            nodeId: chunkIndex === 0 ? id : `${id}-chunk-${chunkIndex}`,
            queryStart: chunk.queryStart,
            queryCount: chunk.queryCount,
            candidateCapacity: gridIndex.capacity,
            candidates: chunk.candidates
          })
        );
      });
      return nodes;
    }
    return [this.getKrigingKernelNode(graph, usableSampleWGSL)];
  }

  /**
   * The kriging kernel over all queries, or over one grid chunk: `grid.candidates` holds the
   * chunk's fixed-width candidate rows followed by the sample positions as float32 bits.
   */
  private getKrigingKernelNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    usableSampleWGSL: string,
    grid?: {
      nodeId: string;
      queryStart: number;
      queryCount: number;
      candidateCapacity: number;
      candidates: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters> {
    const {props, id, neighborCount, supportCount, driftCount} = this;
    const queryCount = grid ? grid.queryCount : this.queryCount;
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
      {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
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
    if (props.parameters) {
      bindings.push({name: 'params', view: props.parameters, type: 'f32', access: 'read'});
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
    const factors = props.coregionalization ?? [1, 0, 0, 1];
    const means = props.means ?? Array.from({length: this.variableCount}, () => 0);
    // Ordinary co-kriging has one unbiasedness constraint per variable. A secondary variable with
    // no retained sample in a neighbourhood has the vacuous constraint `0 = 0`: kept, it is a zero
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
    const structures =
      props.coregionalizationModels ??
      (props.covariance ? [{covariance: props.covariance, loadings: factors}] : []);
    // The system is divided by the largest structure variance (times its cross loading), so the
    // covariance block is O(1) next to the unit-scale drift rows for any sill from 1e-4 to 1e6.
    // Weights are unchanged, the Lagrange multipliers scale by the same constant.
    const covarianceScale = structures.reduce((largest, structure) => {
      let loadingScale = 1;
      if (props.mode === 'co-kriging') {
        loadingScale = 0;
        for (let pair = 0; pair < this.variableCount ** 2; pair++) {
          loadingScale = Math.max(
            loadingScale,
            Math.abs(getCoregionalizationCoefficient(structure.loadings, this.variableCount, pair))
          );
        }
      }
      const variance = structure.covariance.sill + (structure.covariance.nugget ?? 0);
      return Math.max(largest, variance * loadingScale);
    }, 0);
    const crossModel =
      props.mode === 'co-kriging'
        ? structures
            .map((structure, structureIndex) => {
              const coefficients = Array.from({length: this.variableCount ** 2}, (_, pair) =>
                getWGSLFloatLiteral(
                  getCoregionalizationCoefficient(structure.loadings, this.variableCount, pair)
                )
              );
              return `${getPredictionCovarianceWGSL(structure.covariance).replace('fn covariance(', `fn covariance${structureIndex}(`)}\nconst CROSS${structureIndex}: array<f32, ${coefficients.length}> = array<f32, ${coefficients.length}>(${coefficients.join(', ')});`;
            })
            .join('\n')
        : '';
    const searchRadius = getWGSLFloatLiteral(
      Number.isFinite(props.searchRadius) ? props.searchRadius! : 3.402823e38
    );
    // Keeps the nearest NEIGHBOR_COUNT usable samples by (distance(), visit order). The scan and
    // the grid candidates both visit samples in ascending row, so ties go to the lower row.
    const retainSample = `if (sample < SAMPLE_COUNT && isUsableSample(sample)) {
  let position = read_positions(sample);
  let separation = distance(position, center);
  ${grid ? 'farthestCandidate = max(farthestCandidate, separation);' : ''}
  if (separation <= SEARCH_RADIUS) {
    var insertion = retained;
    for (var neighbor = 0u; neighbor < retained; neighbor++) { if (separation < distances[neighbor]) { insertion = neighbor; break; } }
    if (insertion < NEIGHBOR_COUNT) {
      var shift = min(retained, NEIGHBOR_COUNT - 1u);
      loop { if (shift <= insertion) { break; } neighbors[shift] = neighbors[shift - 1u]; distances[shift] = distances[shift - 1u]; shift--; }
      neighbors[insertion] = sample; distances[insertion] = separation; retained = min(retained + 1u, NEIGHBOR_COUNT);
    }
  }
}`;
    // Grid candidates are the CANDIDATE_COUNT nearest usable samples by the index's float32
    // `dx^2 + dy^2`, in ascending row. A sample outside a full candidate list has an index d^2 at
    // least that of the farthest candidate, so its distance() is at least the farthest candidate's
    // within rounding. When the farthest candidate is not clearly beyond the retained boundary (the
    // NEIGHBOR_COUNT-th distance, or the search radius when fewer are retained), a tie or near-tie
    // may continue past the list: that query rescans every sample, giving exactly the scan's
    // neighbors. Lattice ties rarely reach past the extra candidates, so rescans are rare.
    const selection = grid
      ? `var farthestCandidate = -1.0;
var listed = 0u;
for (var candidate = 0u; candidate < CANDIDATE_COUNT; candidate++) {
  let sample = neighborhoods[neighborhoodsOffset + candidateQuery * CANDIDATE_COUNT + candidate];
  if (sample < SAMPLE_COUNT) { listed++; }
  ${retainSample}
}
let boundary = select(SEARCH_RADIUS, distances[NEIGHBOR_COUNT - 1u], retained == NEIGHBOR_COUNT);
if (listed == CANDIDATE_COUNT && farthestCandidate <= boundary * ${getWGSLFloatLiteral(1 + LOCAL_KRIGING_INDEX_TIE_TOLERANCE)}) {
  retained = 0u;
  for (var sample = 0u; sample < SAMPLE_COUNT; sample++) {
    ${retainSample}
  }
}`
      : `${candidateOffsets ? `let candidateBegin = neighborhoodOffsets[neighborhoodOffsetsOffset + index];\nlet candidateEnd = neighborhoodOffsets[neighborhoodOffsetsOffset + index + 1u];\nif (candidateBegin > candidateEnd || candidateEnd > ${candidateNeighbors!.length}u) { return; }` : ''}
for (var candidate = ${candidateOffsets ? 'candidateBegin' : '0u'}; candidate < ${candidateOffsets ? 'candidateEnd' : candidateNeighbors ? `${candidateCapacity}u` : 'SAMPLE_COUNT'}; candidate++) {
  let sample = ${candidateNeighbors ? `neighborhoods[neighborhoodsOffset + ${candidateOffsets ? '' : `index * ${candidateCapacity}u + `}candidate]` : 'candidate'};
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
const QUERY_START: u32 = ${grid.queryStart}u;
const POSITIONS_BASE: u32 = ${grid.queryCount * grid.candidateCapacity}u;
fn read_positions(row: u32) -> vec2<f32> { return vec2<f32>(bitcast<f32>(neighborhoods[neighborhoodsOffset + POSITIONS_BASE + row * 2u]), bitcast<f32>(neighborhoods[neighborhoodsOffset + POSITIONS_BASE + row * 2u + 1u])); }`
      : '';
    return createWGSLKernelNode(graph, {
      id: grid ? grid.nodeId : id,
      operation: 'GPULocalKriging',
      variant: props.mode,
      bindings,
      invocationCount: queryCount,
      workgroupSize: 32,
      declarations: `${supportAccessors}\n${gridDeclarations}\nconst SEARCH_RADIUS: f32 = ${searchRadius};\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${usableSampleWGSL}
${getEquilibratedSolveWGSL(neighborCount + driftCount)}\n${props.parameters ? getPerFramePredictionCovarianceWGSL(this.covarianceStructureCount, this.variableCount) : getPredictionCovarianceWGSL(props.covariance!)}
const NEIGHBOR_COUNT: u32 = ${neighborCount}u;
const DRIFT_COUNT: u32 = ${driftCount}u;
const SUPPORT_COUNT: u32 = ${supportCount}u;
const SAMPLE_COUNT: u32 = ${props.positions.length}u;
const TARGET_VARIABLE: u32 = ${targetVariable}u;
const MEANS: array<f32, ${means.length}> = array<f32, ${means.length}>(${means.map(getWGSLFloatLiteral).join(', ')});
${
  props.parameters
    ? ''
    : `const COVARIANCE_SCALE: f32 = ${getWGSLFloatLiteral(covarianceScale > 0 && Number.isFinite(covarianceScale) ? covarianceScale : 1)};
${crossModel}
fn crossCovariance(first: vec2<f32>, second: vec2<f32>, firstVariable: u32, secondVariable: u32) -> f32 {
  return ${props.mode === 'co-kriging' ? structures.map((_, structureIndex) => `covariance${structureIndex}(first, second) * CROSS${structureIndex}[firstVariable * ${this.variableCount}u + secondVariable]`).join(' + ') : 'covariance(first, second)'};
}`
}`,
      body: `${grid ? 'let candidateQuery = index;\n{\nlet index = index + QUERY_START;' : ''}
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
${props.output.weights ? 'for (var column = 0u; column < NEIGHBOR_COUNT; column++) { weights[weightsOffset + index * NEIGHBOR_COUNT + column] = 0.0; }' : ''}
${props.parameters ? 'if (!loadCovarianceParameters()) { return; }\nlet covarianceScale = params[paramsOffset];' : 'let covarianceScale = COVARIANCE_SCALE;'}
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
${grid ? '}' : ''}`
    });
  }

  /**
   * Grid-index candidates: one pass writes the support centers and a mask of the usable samples,
   * then, per query chunk, a `GPUNeighborSearch` kNN cross join lists the `neighborCount` plus
   * margin nearest usable samples of every center (ascending sample row, the order of the
   * exhaustive scan). The search targets are the samples followed by the centers (masked out), so
   * the derived bounds cover the queries and no query falls outside the lattice. No radius is
   * applied in the search: the nearest candidates followed by the kernel's `searchRadius` test
   * retain the same rows as a scan. A last pass per chunk copies the CSR into fixed-width rows
   * padded with 0xffffffff, followed by the sample positions, in one binding of at most the
   * device's `maxStorageBufferBindingSize`.
   */
  private getSampleIndexNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    usableSampleWGSL: string
  ): {
    nodes: GPUCommandNode<Parameters>[];
    chunks: {queryStart: number; queryCount: number; candidates: GraphDataView<'uint32'>}[];
    capacity: number;
  } {
    const {props, id, queryCount, supportCount} = this;
    const sampleCount = props.positions.length;
    const targetCount = sampleCount + queryCount;
    const k = getSampleIndexCandidateCount(this.neighborCount);
    const bindingByteLimit = graph.device.limits.maxStorageBufferBindingSize;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      "sampleIndex 'grid' search targets (samples plus queries, float32x2)",
      targetCount * 8,
      "use sampleIndex 'brute-force' or supply neighborhoods"
    );
    const deviceChunkQueryCount = getSampleIndexChunkQueryCount(bindingByteLimit, sampleCount, k);
    validateRequirement(
      id,
      deviceChunkQueryCount > 0,
      `sampleIndex 'grid' candidate binding needs ${sampleCount * 8 + k * 4} bytes for one query (sample positions plus ${k} candidates), above the device maxStorageBufferBindingSize of ${bindingByteLimit}`
    );
    const chunkQueryCount = Math.min(
      deviceChunkQueryCount,
      props.sampleIndexChunkQueryCount ?? deviceChunkQueryCount
    );
    const targets = createTransientView(graph, `${id}-index-targets`, 'float32x2', targetCount);
    const mask = createTransientView(graph, `${id}-index-mask`, 'uint32', targetCount);
    const centers = createTransientView(graph, `${id}-index-centers`, 'float32x2', queryCount);
    const prepare = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-index-prepare`,
      operation: 'GPULocalKriging',
      variant: 'sample-index-prepare',
      invocationCount: targetCount,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
        {name: 'supports', view: props.supports, type: 'f32', access: 'read'},
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
        {name: 'targets', view: targets, type: 'f32', access: 'read_write'},
        {name: 'mask', view: mask, type: 'u32', access: 'read_write'},
        {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
      ],
      declarations: `${getPackedPredictionAccessors({positions: 2, observations: 4, supports: props.supports.format === 'float32x4' ? 4 : 2})}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
${usableSampleWGSL}
const SAMPLE_COUNT: u32 = ${sampleCount}u;
const SUPPORT_COUNT: u32 = ${supportCount}u;
fn getSupportCenter(query: u32) -> vec2<f32> {
  // The same weighted mean as the kriging kernel; an unusable support gives NaN (an empty row).
  var center = vec2<f32>(0.0);
  var supportWeightSum = 0.0;
  for (var support = 0u; support < SUPPORT_COUNT; support++) {
    let supportValue = read_supports(query * SUPPORT_COUNT + support);
    let point = supportValue.xy;
    let weight = ${props.supports.format === 'float32x4' ? 'supportValue.z' : '1.0'};
    if (!isFiniteValue(weight) || weight < 0.0) { return vec2<f32>(getNaN()); }
    supportWeightSum += weight;
    if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return vec2<f32>(getNaN()); }
    center += point * weight;
  }
  if (!(supportWeightSum > 0.0) || !isFiniteValue(supportWeightSum)) { return vec2<f32>(getNaN()); }
  return center / supportWeightSum;
}`,
      body: `var point = vec2<f32>(getNaN());
  var usable = false;
  if (index < SAMPLE_COUNT) {
    usable = isUsableSample(index);
    point = read_positions(index);
  } else {
    point = getSupportCenter(index - SAMPLE_COUNT);
    centers[centersOffset + (index - SAMPLE_COUNT) * 2u] = point.x;
    centers[centersOffset + (index - SAMPLE_COUNT) * 2u + 1u] = point.y;
  }
  targets[targetsOffset + index * 2u] = point.x;
  targets[targetsOffset + index * 2u + 1u] = point.y;
  mask[maskOffset + index] = select(0u, 1u, usable);`
    });
    const parameters = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
      id: `${id}-index-parameters`,
      operation: 'GPULocalKriging',
      positions: targets,
      radius: Infinity
    });
    const nodes: GPUCommandNode<Parameters>[] = [prepare, ...parameters.nodes];
    const chunks: {
      queryStart: number;
      queryCount: number;
      candidates: GraphDataView<'uint32'>;
    }[] = [];
    for (let queryStart = 0; queryStart < queryCount; queryStart += chunkQueryCount) {
      const chunkIndex = chunks.length;
      const suffix = chunkIndex === 0 ? '' : `-chunk-${chunkIndex}`;
      const chunkCount = Math.min(chunkQueryCount, queryCount - queryStart);
      let chunkCenters = centers;
      if (chunkCount !== queryCount) {
        chunkCenters = createTransientView(
          graph,
          `${id}-index-centers${suffix || '-chunk-0'}`,
          'float32x2',
          chunkCount
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-index-centers${suffix || '-chunk-0'}`,
            operation: 'GPULocalKriging',
            variant: 'sample-index-chunk-centers',
            invocationCount: chunkCount * 2,
            bindings: [
              {name: 'centers', view: centers, type: 'f32', access: 'read'},
              {name: 'chunkCenters', view: chunkCenters, type: 'f32', access: 'read_write'}
            ],
            body: `chunkCenters[chunkCentersOffset + index] = centers[centersOffset + ${queryStart * 2}u + index];`
          })
        );
      }
      const offsets = createTransientView(
        graph,
        `${id}-index-offsets${suffix}`,
        'uint32',
        chunkCount + 1
      );
      const neighbors = createTransientView(
        graph,
        `${id}-index-neighbors${suffix}`,
        'uint32',
        chunkCount * k
      );
      const candidates = createTransientView(
        graph,
        `${id}-index-rows${suffix}`,
        'uint32',
        chunkCount * k + sampleCount * 2
      );
      const search = new GPUNeighborSearch({
        id: `${id}-index${suffix}`,
        spatialContext: this.spatialContext,
        mode: 'knn',
        k,
        positions: targets,
        queryPositions: chunkCenters,
        mask,
        parameters: parameters.parameters,
        gridSize: parameters.gridSize,
        output: {
          offsets,
          neighbors,
          weights: createTransientView(
            graph,
            `${id}-index-weights${suffix}`,
            'float32',
            chunkCount * k
          ),
          status: {
            overflow: createTransientView(graph, `${id}-index-overflow${suffix}`, 'uint32', 1)
          }
        }
      });
      // Fixed-width rows padded with 0xffffffff, then the sample positions as float32 bits: one
      // kriging binding for candidates and positions, so the grid binds no more than the scan.
      const pad = createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-index-rows${suffix}`,
        operation: 'GPULocalKriging',
        variant: 'sample-index-rows',
        invocationCount: chunkCount + sampleCount,
        bindings: [
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'positions', view: props.positions, type: 'u32', access: 'read'},
          {name: 'rows', view: candidates, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < ${chunkCount}u) {
    let begin = offsets[offsetsOffset + index];
    let count = min(offsets[offsetsOffset + index + 1u] - begin, ${k}u);
    for (var slot = 0u; slot < ${k}u; slot++) {
      var sample = 0xffffffffu;
      if (slot < count) { sample = neighbors[neighborsOffset + begin + slot]; }
      rows[rowsOffset + index * ${k}u + slot] = sample;
    }
  } else {
    let sample = index - ${chunkCount}u;
    rows[rowsOffset + ${chunkCount * k}u + sample * 2u] = positions[positionsOffset + sample * 2u];
    rows[rowsOffset + ${chunkCount * k}u + sample * 2u + 1u] = positions[positionsOffset + sample * 2u + 1u];
  }`
      });
      nodes.push(...search.getCommandNodes(graph), pad);
      chunks.push({queryStart, queryCount: chunkCount, candidates});
    }
    return {nodes, chunks, capacity: k};
  }
}
