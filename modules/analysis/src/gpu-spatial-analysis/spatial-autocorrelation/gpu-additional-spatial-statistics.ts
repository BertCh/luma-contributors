// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';

import {snapshotSpatialAutocorrelationProps} from './spatial-autocorrelation-construction';

import {clamp} from '@math.gl/core';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStatisticsKernelNode as createWGSLKernelNode} from '../spatial-weights/statistics-kernel-node';
import {
  validateCapacity,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from '../spatial-weights/float32-values';
import type {GPUPermutationStatisticAdapter} from '../../analysis-core/contracts/index';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  isGPUSpatialWeightsRowStandardized,
  resolveGPUIslandPolicy,
  type GPUChunkedSpatialWeights,
  type GPUIslandPolicy,
  type GPUSpatialWeights,
  type GPUSpatialWeightsChunkDispatch,
  validateGPUSpatialWeightsOrChunks
} from '../spatial-weights/index';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {createSpatialStatisticsCollectiveNode} from '../spatial-weights/spatial-statistics-collective-node';
import {createSpatialStatisticsRowNode} from '../spatial-weights/spatial-statistics-row-dispatch';
import {
  getSpatialAutocorrelationInputNodes,
  getSpatialWeightsNeighborLoopWGSL,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs,
  validateSpatialAutocorrelationBindings
} from './spatial-autocorrelation-kernels';

import {
  getSpatialAutocorrelationStatusNodes,
  validateSpatialAutocorrelationStatus,
  type GPUSpatialAutocorrelationStatusPort
} from './spatial-autocorrelation-status';

/** Lanes of every workgroup that reduces a global edge-association statistic. */
const EDGE_ASSOCIATION_WORKGROUP_SIZE = 256;

/** Upper bound of the workgroups that share the row reduction of an edge-association statistic. */
const EDGE_ASSOCIATION_MAXIMUM_GROUPS = 256;

/** Float words per workgroup in the partials: stage 1 at 0..2, stage 2 at 3..5. */
const EDGE_ASSOCIATION_PARTIAL_WORDS = 6;

/** Caller-owned output of {@link GPUSpatialPearson} and {@link GPUGammaStatistic}. */
export type GPUEdgeAssociationOutput = {
  /** One float32: the global statistic (NaN when undefined). There is no analytic p-value. */
  statistic: GraphDataView<'float32'>;
};

/** Inputs shared by global edge-association statistics. */
type GPUEdgeAssociationProps = {
  id?: string;
  /** Whole weights, or row chunks whose per-chunk partials combine in chunk order. */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** First variable, one per row, `float32` or `uint32` (converted to f32, exact below 2^24). */
  values: GPUFloat32OrUint32Values;
  /** Optional second variable, one per row, `float32` or `uint32`. */
  secondValues?: GPUFloat32OrUint32Values;
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned output. */
  output: GPUEdgeAssociationOutput;
};

/** Caller-owned outputs of {@link GPULocalGeary}. */
export type GPULocalGearyOutput = {
  /**
   * Optional input diagnostics, overwritten on every encoding. invalidCount counts unmasked
   * nonfinite values, excluding islands and undefined tests. overflow and candidateOverflow
   * separately forward input weights' producer-specific flags as 0 or 1
   * (0 when the corresponding input port is absent).
   */
  status?: GPUSpatialAutocorrelationStatusPort;
  /** One local Geary value per weights row. */
  statistic: GraphDataView<'float32'>;
  /** Optional `[n, mean, variance, standardDeviation, islandCount]` used this frame. */
  globalStatistics?: GraphDataView<'float32'>;
  /** Exact uint32 [populationCount, islandCount]; populationCount uses fixedMoments.count when supplied. */
  globalCounts?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPULocalGeary}. */
export type GPULocalGearyProps = {
  id?: string;
  /**
   * Square self-join weights. Used as given, except that when `weights.information.transform` is `'row'`
   * each row is re-standardized over the neighbors that remain after masking. A neighbor is a slot
   * with a positive weight that points at another included row. Either nonzero weights status
   * flag (overflow or candidateOverflow) suppresses the statistic with NaN for every row.
   * Chunk-local loss flags are rejected; aggregate them on the outer weights.status port.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Analysis values, one per row, `float32` or `uint32` (converted to f32, exact below 2^24). */
  values: GPUFloat32OrUint32Values;
  parameters: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /**
   * Compile-time. `'exclude'` (default) removes islands from the moments and writes NaN for them;
   * `'keep'` keeps them in the moments (esda) with a local Geary value of 0. The count is
   * reported in `globalStatistics[4]`.
   */
  islands?: GPUIslandPolicy;
  /** Caller-owned outputs. */
  output: GPULocalGearyOutput;
};

/**
 * Local Geary plugs into the shared conditional-permutation contributor. Small `c_i` means
 * similar neighbors (a cluster), so the tail that signals clustering is the lower one:
 * `'lesser'`.
 */
export const GPU_LOCAL_GEARY_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localGeary',
  permutedVariable: 'values',
  alternative: 'lesser'
};

/**
 * Local bivariate Moran (esda `Moran_Local_BV`) runs in `GPULocalPermutationTest` with
 * `statistic: 'localBivariateMoran'`: the focus keeps `x`, the neighbors' `y` are drawn.
 */
export const GPU_LOCAL_BIVARIATE_MORAN_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localBivariateMoran',
  permutedVariable: 'secondValues',
  alternative: 'two-sided'
};

/**
 * Local join counts (Anselin and Li 2019; esda `Join_Counts_Local`, `Join_Counts_Local_BV`) run in
 * `GPULocalPermutationTest` with `statistic: 'localJoinCount'` or `'localBivariateJoinCount'`.
 * Clustering is more joins than expected, so the tail is `'greater'`.
 */
export const GPU_LOCAL_JOIN_COUNT_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localJoinCount',
  permutedVariable: 'values',
  alternative: 'greater'
};

/** Bivariate local join counts permute `secondValues` (and the paired `values` of the drawn rows). */
export const GPU_LOCAL_BIVARIATE_JOIN_COUNT_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localBivariateJoinCount',
  permutedVariable: 'secondValues',
  alternative: 'greater'
};

/**
 * Lee's spatial Pearson (esda `Spatial_Pearson`, `Spatial_Pearson_Local`) runs in
 * `GPUGlobalPermutationTest` (`statistic: 'leesL'`) and `GPULocalPermutationTest`
 * (`statistic: 'localLeeL'`). Both permute the rows of `(x, y)` together, so the tests ask whether
 * the spatial pattern of the pair is unusual given their pairing; esda's own p-value is the
 * 'directed' one, which is anti-conservative. The default `'two-sided'` is kept.
 */
export const GPU_LEES_L_PERMUTATION_ADAPTERS: {
  global: GPUPermutationStatisticAdapter;
  local: GPUPermutationStatisticAdapter;
} = {
  global: {
    scope: 'global',
    statistic: 'leesL',
    permutedVariable: 'values',
    alternative: 'two-sided'
  },
  local: {
    scope: 'local',
    statistic: 'localLeeL',
    permutedVariable: 'values',
    alternative: 'two-sided'
  }
};

/**
 * Local Geary `c_i = sum_j w_ij (z_i - z_j)^2`, where `z` is standardized by the population
 * variance of the analysis rows. The sum runs over active neighbors (positive weight, included
 * row other than `i`); row-standardized weights are re-standardized over the neighbors that remain
 * after masking. Excluded rows return NaN, as do islands under `islands: 'exclude'` (default);
 * under `'keep'` an island returns 0. A constant variable (zero variance) returns NaN. CSR slot
 * order fixes the summation order. There is no analytic classification; use permutation inference
 * with `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER` (`GPULocalPermutationTest` with `statistic:
 * 'localGeary'` and its `classes` output gives the positive / negative association classes).
 */
export class GPULocalGeary implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocalGearyProps;

  /** Resolved island policy. */
  readonly islands: GPUIslandPolicy;

  constructor(props: GPULocalGearyProps) {
    props = snapshotSpatialAutocorrelationProps(props);
    this.id = props.id ?? 'local-geary';
    this.props = props;
    this.islands = resolveGPUIslandPolicy(this.id, props.islands);
    const rows = validateSpatialAutocorrelationInputs({
      ...props,
      id: this.id,
      globalCounts: props.output.globalCounts,
      globalStatistics: props.output.globalStatistics
    });
    validateSinglePackedView(props.output.statistic, ['float32'], `${this.id} output.statistic`);
    if (props.output.statistic.length !== rows) {
      throw new Error(`${this.id} output.statistic length must equal the row count`);
    }
    validateSpatialAutocorrelationStatus(this.id, props.output.status, [
      props.output.statistic,
      props.output.globalStatistics
    ]);
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        props.output.statistic,
        props.output.globalStatistics,
        props.output.globalCounts,
        ...Object.values(props.output.status ?? {})
      ],
      [
        ...getGPUSpatialWeightsOrChunksViews(props.weights),
        props.weights.status?.overflow,
        props.weights.status?.candidateOverflow,
        props.values,
        props.parameters,
        props.mask
      ]
    );
    validateDisjointWritableViews(this.id, {
      statistic: props.output.statistic,
      globalStatistics: props.output.globalStatistics,
      globalCounts: props.output.globalCounts,
      status: undefined,
      ...props.output.status
    });
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.weights.status?.candidateOverflow,
      props.values,
      props.parameters,
      props.mask,
      props.output.statistic,
      props.output.globalStatistics,
      props.output.globalCounts,
      ...Object.values(props.output.status ?? {})
    ]);
    validateSpatialAutocorrelationBindings(graph, {...props, id}, [
      props.output.statistic,
      props.output.globalStatistics,
      props.output.globalCounts,
      ...Object.values(props.output.status ?? {})
    ]);
    const statusNodes = getSpatialAutocorrelationStatusNodes(graph, id, 'GPULocalGeary', props);
    const rows = props.values.length;
    const inputs = getSpatialAutocorrelationInputNodes(graph, {
      ...props,
      globalCounts: props.output.globalCounts,
      globalStatistics: props.output.globalStatistics,
      id,
      islands: this.islands,
      operation: 'GPULocalGeary'
    });
    inputs.nodes.push(
      ...dispatches.map(dispatch =>
        createSpatialStatisticsRowNode(graph, {
          rowBase: dispatch.rowStart,
          id: `${id}-statistic${dispatch.idSuffix}`,
          operation: 'GPULocalGeary',
          variant: 'local-geary',
          invocationCount: dispatch.rowCount,
          bindings: [
            {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
            {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'}
          ],
          declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const ROWS: u32 = ${rows}u;
const MOMENTS: u32 = ${rows}u;
const STANDARDIZE: bool = ${isGPUSpatialWeightsRowStandardized(props.weights)};`,
          body: `let focus = statistics[statisticsOffset + index];
  if (!isFiniteFloat(focus)) {
    result[resultOffset + index] = getQuietNaN(index);
    return;
  }
  let variance = statistics[statisticsOffset + MOMENTS + 2u];
  var sum = 0.0;
  var weightSum = 0.0;
  ${getSpatialWeightsNeighborLoopWGSL(
    `let other = statistics[statisticsOffset + neighbor];
    let difference = focus - other;
    sum += weight * difference * difference;
    weightSum += weight;`,
    'WEIGHTS_ROW_START'
  )}
  if (STANDARDIZE && weightSum > 0.0) {
    // Row-standardized weights are divided by what remains of their row sum.
    sum = sum / weightSum;
  }
  // A constant variable has no standardized values: the statistic is undefined, not zero.
  result[resultOffset + index] = select(getQuietNaN(index), sum / variance, variance > 0.0 && isFiniteFloat(variance));`
        })
      )
    );
    return [...statusNodes, ...inputs.nodes];
  }
}

/** Properties for {@link GPUSpatialPearson}. */
export type GPUSpatialPearsonProps = GPUEdgeAssociationProps & {
  /** Second variable is required for the spatial cross-correlation. */
  secondValues: GPUFloat32OrUint32Values;
};

/**
 * Names the global permutation statistic closest to {@link GPUSpatialPearson}: bivariate Moran
 * with `secondValues` permuted. It is a test of bivariate Moran's I, not of the edge-weighted
 * Pearson coefficient. The two order every permutation the same way only when every column of the
 * weights has the same total (each row receives the same in-weight): then the edge-weighted mean
 * and variance of `y` do not change under permutation. On other graphs (binary weights with
 * unequal degrees, row-standardized weights) the orderings and so the pseudo p-values can differ.
 * Use an explicit joint complete-case mask when either variable has missing rows; the shared
 * bivariate engine excludes incomplete pairs. Column totals refer to the effective selected weights.
 */
export const GPU_SPATIAL_PEARSON_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'global',
  statistic: 'bivariateMoran',
  permutedVariable: 'secondValues',
  alternative: 'two-sided'
};

/**
 * Weighted Pearson correlation across directed spatial edges `(i,j)`: the weighted covariance of
 * `x_i` and `y_j`, divided by their weighted edge standard deviations. This definition is bounded
 * to `[-1, 1]`, is valid for asymmetric weights, and returns NaN for zero edge weight or variance.
 *
 * This is not `esda.Spatial_Pearson`, which is Lee's L (`'leesL'` in `GPUGlobalPermutationTest`).
 * For a permutation test see {@link GPU_SPATIAL_PEARSON_PERMUTATION_ADAPTER} and its limits.
 * Either nonzero outer weights.status loss flag suppresses the result with NaN. Chunk-local
 * loss flags must be aggregated on outer weights.status and are rejected at construction.
 * Exponent-scaled division by separate square roots preserves correlation when the product of
 * finite sums of squares would overflow or underflow float32, even with GPU reassociation;
 * the individual moment sums must still fit float32. Edge-end means stay in reference-offset
 * coordinates so a large common baseline does not erase small covariances.
 */
export class GPUSpatialPearson implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSpatialPearsonProps;

  constructor(props: GPUSpatialPearsonProps) {
    props = snapshotSpatialAutocorrelationProps(props);
    this.id = props.id ?? 'spatial-pearson';
    this.props = props;
    validateRequirement(this.id, Boolean(props.secondValues), 'secondValues is required');
    validateEdgeAssociationProps(this.id, props);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return createEdgeAssociationNodes(graph, this.id, this.props, 'pearson');
  }
}

/** Gamma statistic operation. */
export type GPUGammaOperation = 'cross-product' | 'squared-difference' | 'absolute-difference';

/** Properties for {@link GPUGammaStatistic}. */
export type GPUGammaStatisticProps = GPUEdgeAssociationProps & {
  operation?: GPUGammaOperation;
};

/**
 * Names the existing global permutation statistic to run for a Gamma operation.
 * These mappings require a univariate Gamma (`secondValues` omitted or identical to `values`).
 * With a distinct second variable they do not test the bivariate edge statistic.
 * Both calculations must use identical effective weights and the same donor population after
 * mask/missing/island selection. Gamma retains supplied edge weights; the permutation engine
 * re-standardizes rows tagged transform='row' after selection. Materialize selected weights,
 * apply any desired row normalization once, then supply both calculations the same weights
 * tagged transform='raw' with the same explicit mask. Retain isolated donors with engine
 * islands='keep', or remove them from both populations before materializing the weights.
 *
 * Under those conditions, `'squared-difference'` maps to `'geary'` exactly: Gamma is its numerator divided by
 * the constant total edge weight, so both order every permutation the same way.
 * Selected labels must have positive variance. Constant labels give Gamma=0 but undefined
 * Geary (its label-variance denominator is zero), so there is no ordering equivalence there.
 *
 * `'cross-product'` maps to `'moran'`, which is the same ordering only when every row of the
 * weights has the same total, or every column does (row-standardized weights, or binary weights
 * on a regular graph): {@link GPUGammaStatistic} centers on the edge-weighted means and divides
 * by the edge weight, and those means move under permutation when degrees differ. On such a
 * graph the adapter runs a test of Moran's I, not of this Gamma. Example: the star 0-1, 0-2, 0-3
 * with the edge 3-4, binary weights; the values (7, 2, 4, 1, 11) and (11, 2, 1, 7, 4) have Gamma
 * -10 and -11.0625 but Moran pair sums -80 and -64, the opposite order.
 *
 * `'absolute-difference'` has no matching permutation statistic and throws.
 */
export function getGPUGammaPermutationAdapter(
  operation: GPUGammaOperation
): GPUPermutationStatisticAdapter {
  validateRequirement(
    'getGPUGammaPermutationAdapter',
    ['cross-product', 'squared-difference', 'absolute-difference'].includes(operation),
    'operation must be cross-product, squared-difference or absolute-difference'
  );
  if (operation === 'absolute-difference') {
    throw new Error(
      'getGPUGammaPermutationAdapter operation absolute-difference has no matching shared permutation statistic; use GPUGammaStatistic descriptively or a separate absolute-difference randomization test'
    );
  }
  return {
    scope: 'global',
    statistic: operation === 'squared-difference' ? 'geary' : 'moran',
    permutedVariable: 'values',
    alternative: 'two-sided'
  };
}

/**
 * A Gamma-type statistic over directed weighted edges. The operation is either centered
 * cross-product, squared difference, or absolute difference. The result is the weighted mean over
 * valid edges; the cross-product is centered on the edge-weighted means of the two edge ends,
 * retained in reference-offset coordinates to preserve small covariances at large baselines.
 *
 * This is not `esda.Gamma`, which is the plain sum `sum_ij w_ij x_i x_j` (uncentered unless its
 * `standardize` option is set; or the same sum of squared or absolute differences) with no
 * division by the edge weight. For a permutation test
 * see {@link getGPUGammaPermutationAdapter} and its limits.
 * Either nonzero outer weights.status loss flag suppresses the result with NaN. Chunk-local
 * loss flags must be aggregated on outer weights.status and are rejected at construction.
 */
export class GPUGammaStatistic implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUGammaStatisticProps;
  readonly operation: GPUGammaOperation;

  constructor(props: GPUGammaStatisticProps) {
    props = snapshotSpatialAutocorrelationProps(props);
    this.id = props.id ?? 'gamma-statistic';
    this.props = props;
    this.operation = props.operation ?? 'cross-product';
    validateRequirement(
      this.id,
      ['cross-product', 'squared-difference', 'absolute-difference'].includes(this.operation),
      `operation must be cross-product, squared-difference or absolute-difference; received ${this.operation}`
    );
    validateEdgeAssociationProps(this.id, props);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return createEdgeAssociationNodes(graph, this.id, this.props, this.operation);
  }
}

function validateEdgeAssociationProps(id: string, props: GPUEdgeAssociationProps): void {
  const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
  if ('chunks' in props.weights) {
    for (const [index, chunk] of props.weights.chunks.entries()) {
      validateRequirement(
        id,
        !chunk.weights.status?.overflow && !chunk.weights.status?.candidateOverflow,
        `weights.chunks[${index}].weights.status loss flags must be aggregated on the outer weights.status port`
      );
    }
  }
  if (validateFloat32OrUint32Values(props.values, `${id} values`) !== rows) {
    throw new Error(`${id} values length must equal weights rows`);
  }
  if (
    props.secondValues &&
    validateFloat32OrUint32Values(props.secondValues, `${id} secondValues`) !== rows
  ) {
    throw new Error(`${id} secondValues length must equal weights rows`);
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) throw new Error(`${id} mask length must equal weights rows`);
  }
  validateSinglePackedView(props.output.statistic, ['float32'], `${id} output.statistic`);
  validateCapacity(id, 'output.statistic length', props.output.statistic.length);
  validateGraphOutputsDisjointFromInputs(
    id,
    [props.output.statistic],
    [
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.weights.status?.candidateOverflow,
      props.values,
      props.secondValues,
      props.mask
    ]
  );
}

/**
 * Four kernels reduce the edges deterministically (no float atomics): `-anchors` selects safe references; `-moments` (one partial of
 * weight, weighted x and weighted y per workgroup), `-centered` (every workgroup first combines the
 * moment partials in the same fixed tree order, so all derive bit-identical means, then writes a
 * partial of numerator and squared deviations) and `-statistic` (one workgroup combines the
 * partials in fixed tree order and writes the answer).
 */
function createEdgeAssociationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUEdgeAssociationProps,
  operation: 'pearson' | GPUGammaOperation
): GPUCommandNode<Parameters>[] {
  const partialWords = operation === 'pearson' ? 10 : EDGE_ASSOCIATION_PARTIAL_WORDS;
  const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
  const chunkGroups = dispatches.map(dispatch =>
    clamp(
      Math.ceil(dispatch.rowCount / EDGE_ASSOCIATION_WORKGROUP_SIZE),
      1,
      EDGE_ASSOCIATION_MAXIMUM_GROUPS
    )
  );
  const totalGroups = chunkGroups.reduce((sum, count) => sum + count, 0);
  validateSpatialStatisticsBindings(graph, id, props, {
    'converted values': props.values.length * 4,
    'edge moment partials': totalGroups * partialWords * 4,
    'edge anchors': (totalGroups * 4 + 1) * 4
  });
  validateGraphViewsBelongToGraph(id, graph, [
    ...getGPUSpatialWeightsOrChunksViews(props.weights),
    props.weights.status?.overflow,
    props.weights.status?.candidateOverflow,
    props.values,
    props.secondValues,
    props.mask,
    props.output.statistic
  ]);
  const rows = props.values.length;
  const valuesInput = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
  const secondInput = props.secondValues
    ? createFloat32ValuesNodes(graph, `${id}-second-values`, props.secondValues)
    : valuesInput;
  // Each chunk reduces its own rows with up to EDGE_ASSOCIATION_MAXIMUM_GROUPS workgroups; the
  // partial regions of the chunks are consecutive, so the final fixed-order tree combines them in
  // chunk order. A whole CSR is one chunk.
  const partials = createTransientView(
    graph,
    `${id}-partials`,
    'float32',
    totalGroups * partialWords
  );
  const getEdgeBindings = (dispatch: GPUSpatialWeightsChunkDispatch): WGSLKernelBinding[] => [
    {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
    {name: 'values', view: valuesInput.view, type: 'f32', access: 'read'},
    {name: 'secondValues', view: secondInput.view, type: 'f32', access: 'read'},
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : [])
  ];
  const term =
    operation === 'cross-product' || operation === 'pearson'
      ? 'centeredX * centeredY'
      : operation === 'squared-difference'
        ? '(x - y) * (x - y)'
        : operation === 'absolute-difference'
          ? 'abs(x - y)'
          : '0.0';
  const operationName = operation === 'pearson' ? 'GPUSpatialPearson' : 'GPUGammaStatistic';
  const getDeclarations = (includeExtrema = false) => `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const ROWS: u32 = ${rows}u;
var<private> CHUNK_ROWS: u32;
var<private> ROW_START: u32;
const LANES: u32 = ${EDGE_ASSOCIATION_WORKGROUP_SIZE}u;
var<private> GROUPS: u32;
var<private> GROUP_BASE: u32;
const TOTAL_GROUPS: u32 = ${totalGroups}u;
const PARTIAL_WORDS: u32 = ${partialWords}u;
var<workgroup> sharedA: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> sharedB: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> sharedC: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
${
  operation === 'pearson' && includeExtrema
    ? `var<workgroup> minimumX: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> maximumX: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> minimumY: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> maximumY: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;`
    : ''
}
`;
  const filters = (code: string) =>
    code
      .replaceAll(
        '%ROW_OK%',
        `${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteFloat(values[valuesOffset + row])`
      )
      .replaceAll(
        '%NEIGHBOR_OK%',
        `neighbor >= ROWS || neighbor == row || !(weights[weightsOffset + slot] > 0.0) || ${props.mask ? 'mask[maskOffset + neighbor] == 0u || ' : ''}!isFiniteFloat(secondValues[secondValuesOffset + neighbor])`
      );
  // Fixed-order tree reduction of the lane values staged in sharedA/B/C. Ends with a barrier.
  const reduceWGSL = `workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedA[lane] += sharedA[lane + stride];
      sharedB[lane] += sharedB[lane + stride];
      sharedC[lane] += sharedC[lane + stride];
    }
    workgroupBarrier();
  }`;
  // Lane `l` adds the three partial words at `wordOffset` of workgroups `l`, `l + LANES`, ... (all
  // chunks, in chunk order) before the tree; the block scope lets the loader be emitted twice; a lone chunk of at most LANES groups loads one each.
  const loadPartialsWGSL = (wordOffset: number) => `{
  var accumulatedA = 0.0;
  var accumulatedB = 0.0;
  var accumulatedC = 0.0;
  for (var partialGroup = lane; partialGroup < TOTAL_GROUPS; partialGroup += LANES) {
    accumulatedA += partials[partialsOffset + partialGroup * PARTIAL_WORDS + ${wordOffset}u];
    accumulatedB += partials[partialsOffset + partialGroup * PARTIAL_WORDS + ${wordOffset + 1}u];
    accumulatedC += partials[partialsOffset + partialGroup * PARTIAL_WORDS + ${wordOffset + 2}u];
  }
  sharedA[lane] = accumulatedA;
  sharedB[lane] = accumulatedB;
  sharedC[lane] = accumulatedC;
  ${reduceWGSL}
  }`;
  const partialsBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
    name: 'partials',
    view: partials,
    type: 'f32',
    access
  });
  const reduceExtremaWGSL = `workgroupBarrier();
for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
 if (lane < stride) {
  minimumX[lane] = min(minimumX[lane], minimumX[lane + stride]); maximumX[lane] = max(maximumX[lane], maximumX[lane + stride]);
  minimumY[lane] = min(minimumY[lane], minimumY[lane + stride]); maximumY[lane] = max(maximumY[lane], maximumY[lane + stride]);
 }
 workgroupBarrier();
}`;
  let groupBase = 0;
  const chunkPlans = dispatches.map((dispatch, chunkIndex) => {
    const plan = {dispatch, groups: chunkGroups[chunkIndex], groupBase};
    groupBase += plan.groups;
    return plan;
  });

  // Pick finite eligible edge ends before summing offsets. Each chunk writes disjoint partials;
  // every later workgroup combines the anchors in the same order, including empty chunks.
  // A stored zero fixes offset subtraction through integer bits against Metal reassociation.
  const anchors = createTransientView(graph, `${id}-anchors`, 'float32', totalGroups * 4 + 1);
  const anchorBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
    name: 'anchors',
    view: anchors,
    type: 'f32',
    access
  });
  // Sterbenz's interval makes reference subtraction exact. Outside it, reference zero
  // preserves observations far from the minimum instead of rounding their offsets together.
  const anchorDeclarations = `
fn getEdgeCenteringAnchor(minimum: f32, maximum: f32) -> f32 {
  return select(0.0, minimum, (minimum >= 0.0 && minimum >= maximum * 0.5) || (maximum <= 0.0 && maximum <= minimum * 0.5));
}`;
  const reduceRangeWGSL = `workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedB[lane] = min(sharedB[lane], sharedB[lane + stride]);
      sharedC[lane] = max(sharedC[lane], sharedC[lane + stride]);
    }
    workgroupBarrier();
  }`;
  const loadAnchorRangeWGSL = (word: number) => `{
  var low = 3.402823466e38;
  var high = -3.402823466e38;
  for (var partialGroup = lane; partialGroup < TOTAL_GROUPS; partialGroup += LANES) {
    low = min(low, anchors[anchorsOffset + partialGroup * 4u + ${word}u]);
    high = max(high, anchors[anchorsOffset + partialGroup * 4u + ${word + 1}u]);
  }
  sharedB[lane] = low;
  sharedC[lane] = high;
  ${reduceRangeWGSL}
  }`;
  const loadAnchorsWGSL = `${loadAnchorRangeWGSL(0)}
  let anchorX = getEdgeCenteringAnchor(sharedB[0], sharedC[0]);
  workgroupBarrier();
  ${loadAnchorRangeWGSL(2)}
  let anchorY = getEdgeCenteringAnchor(sharedB[0], sharedC[0]);
  let runtimeZero = bitcast<u32>(anchors[anchorsOffset + TOTAL_GROUPS * 4u]);
  workgroupBarrier();`;
  const anchorNodes = chunkPlans.map(({dispatch, groups, groupBase: base}) =>
    createSpatialStatisticsCollectiveNode(graph, {
      id: `${id}-anchors${dispatch.idSuffix}`,
      operation: operationName,
      variant: `${operation}-anchors`,
      workgroupSize: EDGE_ASSOCIATION_WORKGROUP_SIZE,
      invocationCount: groups * EDGE_ASSOCIATION_WORKGROUP_SIZE,
      parameterValues: [dispatch.rowCount, dispatch.rowStart, groups, base],
      bindings: [...getEdgeBindings(dispatch), anchorBinding('read_write')],
      declarations: `${getDeclarations()}\n${ROUNDED_DIFFERENCE_WGSL}\n${anchorDeclarations}`,
      body: filters(`CHUNK_ROWS = getChunkParameter(36u);
  ROW_START = getChunkParameter(37u);
  GROUPS = getChunkParameter(38u);
  GROUP_BASE = getChunkParameter(39u);
  let lane = localInvocationIndex;
  let group = GROUP_BASE + index / LANES;
  var lowAnchorX = 3.402823466e38;
  var lowAnchorY = 3.402823466e38;
  var highAnchorX = -3.402823466e38;
  var highAnchorY = -3.402823466e38;
  for (var localRow = index; localRow < CHUNK_ROWS; localRow += GROUPS * LANES) {
    let row = ROW_START + localRow;
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + localRow]; slot < offsets[offsetsOffset + localRow + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      lowAnchorX = min(lowAnchorX, values[valuesOffset + row]);
      highAnchorX = max(highAnchorX, values[valuesOffset + row]);
      lowAnchorY = min(lowAnchorY, secondValues[secondValuesOffset + neighbor]);
      highAnchorY = max(highAnchorY, secondValues[secondValuesOffset + neighbor]);
    }
  }
  sharedB[lane] = lowAnchorX;
  sharedC[lane] = highAnchorX;
  ${reduceRangeWGSL}
  if (lane == 0u) {
    anchors[anchorsOffset + group * 4u] = sharedB[0];
    anchors[anchorsOffset + group * 4u + 1u] = sharedC[0];
  }
  workgroupBarrier();
  sharedB[lane] = lowAnchorY;
  sharedC[lane] = highAnchorY;
  ${reduceRangeWGSL}
  if (lane == 0u) {
    anchors[anchorsOffset + group * 4u + 2u] = sharedB[0];
    anchors[anchorsOffset + group * 4u + 3u] = sharedC[0];
    if (group == 0u) { anchors[anchorsOffset + TOTAL_GROUPS * 4u] = 0.0; }
  }`)
    })
  );

  // Workgroup g of a chunk strides the chunk's rows g * LANES + lane, step GROUPS * LANES.
  const momentsNodes = chunkPlans.map(({dispatch, groups, groupBase: base}) =>
    createSpatialStatisticsCollectiveNode(graph, {
      id: `${id}-moments${dispatch.idSuffix}`,
      operation: operationName,
      variant: `${operation}-moments`,
      workgroupSize: EDGE_ASSOCIATION_WORKGROUP_SIZE,
      invocationCount: groups * EDGE_ASSOCIATION_WORKGROUP_SIZE,
      parameterValues: [dispatch.rowCount, dispatch.rowStart, groups, base],
      bindings: [
        ...getEdgeBindings(dispatch),
        partialsBinding('read_write'),
        anchorBinding('read')
      ],
      declarations: `${getDeclarations()}\n${ROUNDED_DIFFERENCE_WGSL}\n${anchorDeclarations}`,
      body: filters(`CHUNK_ROWS = getChunkParameter(36u);
  ROW_START = getChunkParameter(37u);
  GROUPS = getChunkParameter(38u);
  GROUP_BASE = getChunkParameter(39u);
  let lane = localInvocationIndex;
  let group = GROUP_BASE + index / LANES;
  ${loadAnchorsWGSL}
  var laneWeight = 0.0;
  var laneX = 0.0;
  var laneY = 0.0;
  for (var localRow = index; localRow < CHUNK_ROWS; localRow += GROUPS * LANES) {
    let row = ROW_START + localRow;
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + localRow]; slot < offsets[offsetsOffset + localRow + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      let weight = weights[weightsOffset + slot];
      laneWeight += weight;
      laneX += weight * getRoundedDifference(values[valuesOffset + row], anchorX, runtimeZero);
      laneY += weight * getRoundedDifference(secondValues[secondValuesOffset + neighbor], anchorY, runtimeZero);
    }
  }
  sharedA[lane] = laneWeight;
  sharedB[lane] = laneX;
  sharedC[lane] = laneY;
  ${reduceWGSL}
  if (lane == 0u) {
    partials[partialsOffset + group * PARTIAL_WORDS] = sharedA[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 1u] = sharedB[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 2u] = sharedC[0];
  }`)
    })
  );

  const centeredNodes = chunkPlans.map(({dispatch, groups, groupBase: base}) =>
    createSpatialStatisticsCollectiveNode(graph, {
      id: `${id}-centered${dispatch.idSuffix}`,
      operation: operationName,
      variant: `${operation}-centered`,
      workgroupSize: EDGE_ASSOCIATION_WORKGROUP_SIZE,
      invocationCount: groups * EDGE_ASSOCIATION_WORKGROUP_SIZE,
      parameterValues: [dispatch.rowCount, dispatch.rowStart, groups, base],
      // The moment words (0..2) are only read and the centered words (3..5) only written.
      bindings: [
        ...getEdgeBindings(dispatch),
        partialsBinding('read_write'),
        anchorBinding('read')
      ],
      declarations: `${getDeclarations(true)}\n${ROUNDED_DIFFERENCE_WGSL}\n${anchorDeclarations}`,
      body: filters(`CHUNK_ROWS = getChunkParameter(36u);
  ROW_START = getChunkParameter(37u);
  GROUPS = getChunkParameter(38u);
  GROUP_BASE = getChunkParameter(39u);
  let lane = localInvocationIndex;
  let group = GROUP_BASE + index / LANES;
  ${loadAnchorsWGSL}
  ${loadPartialsWGSL(0)}
  let weightTotal = sharedA[0];
  let meanX = sharedB[0] / weightTotal;
  let meanY = sharedC[0] / weightTotal;
  workgroupBarrier();
  var laneNumerator = 0.0;
  var laneSquareX = 0.0;
  var laneSquareY = 0.0;
  ${operation === 'pearson' ? 'var lowX = 3.402823466e38; var highX = -3.402823466e38; var lowY = 3.402823466e38; var highY = -3.402823466e38;' : ''}
  for (var localRow = index; localRow < CHUNK_ROWS; localRow += GROUPS * LANES) {
    let row = ROW_START + localRow;
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + localRow]; slot < offsets[offsetsOffset + localRow + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      let weight = weights[weightsOffset + slot];
      let x = values[valuesOffset + row];
      let y = secondValues[secondValuesOffset + neighbor];
      ${operation === 'pearson' ? 'lowX = min(lowX, x); highX = max(highX, x); lowY = min(lowY, y); highY = max(highY, y);' : ''}
      let centeredX = getRoundedDifference(x, anchorX, runtimeZero) - meanX;
      let centeredY = getRoundedDifference(y, anchorY, runtimeZero) - meanY;
      laneNumerator += weight * ${term};
      laneSquareX += weight * centeredX * centeredX;
      laneSquareY += weight * centeredY * centeredY;
    }
  }
  sharedA[lane] = laneNumerator;
  sharedB[lane] = laneSquareX;
  sharedC[lane] = laneSquareY;
  ${reduceWGSL}
  if (lane == 0u) {
    partials[partialsOffset + group * PARTIAL_WORDS + 3u] = sharedA[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 4u] = sharedB[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 5u] = sharedC[0];
  }
  ${
    operation === 'pearson'
      ? `minimumX[lane] = lowX; maximumX[lane] = highX; minimumY[lane] = lowY; maximumY[lane] = highY;
  ${reduceExtremaWGSL}
  if (lane == 0u) { partials[partialsOffset + group * PARTIAL_WORDS + 6u] = minimumX[0]; partials[partialsOffset + group * PARTIAL_WORDS + 7u] = maximumX[0]; partials[partialsOffset + group * PARTIAL_WORDS + 8u] = minimumY[0]; partials[partialsOffset + group * PARTIAL_WORDS + 9u] = maximumY[0]; }`
      : ''
  }`)
    })
  );

  const statisticNode = createWGSLKernelNode(graph, {
    id: `${id}-statistic`,
    operation: operationName,
    variant: operation,
    workgroupSize: EDGE_ASSOCIATION_WORKGROUP_SIZE,
    invocationCount: EDGE_ASSOCIATION_WORKGROUP_SIZE,
    guardIndex: false,
    bindings: [
      partialsBinding('read'),
      {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
      ...(props.weights.status?.overflow
        ? [
            {
              name: 'inputOverflow',
              view: props.weights.status.overflow,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : []),
      ...(props.weights.status?.candidateOverflow
        ? [
            {
              name: 'inputCandidateOverflow',
              view: props.weights.status.candidateOverflow,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ],
    declarations: `${getDeclarations(true)}\n${operation === 'pearson' ? SCALED_QUOTIENT_WGSL : ''}`,
    body: `let lane = localInvocationIndex;
  ${loadPartialsWGSL(0)}
  let weightTotal = sharedA[0];
  workgroupBarrier();
  ${loadPartialsWGSL(3)}
  ${
    operation === 'pearson'
      ? `minimumX[lane] = 3.402823466e38; maximumX[lane] = -3.402823466e38; minimumY[lane] = 3.402823466e38; maximumY[lane] = -3.402823466e38;
  for (var partialGroup = lane; partialGroup < TOTAL_GROUPS; partialGroup += LANES) {
   let base = partialsOffset + partialGroup * PARTIAL_WORDS;
   minimumX[lane] = min(minimumX[lane], partials[base + 6u]); maximumX[lane] = max(maximumX[lane], partials[base + 7u]);
   minimumY[lane] = min(minimumY[lane], partials[base + 8u]); maximumY[lane] = max(maximumY[lane], partials[base + 9u]);
  }
  ${reduceExtremaWGSL}`
      : ''
  }
  if (lane == 0u) {
    let numerator = sharedA[0];
    let squareX = sharedB[0];
    let squareY = sharedC[0];
    var answer = getQuietNaN(0u);
    ${
      operation === 'pearson'
        ? `if (weightTotal > 0.0 && squareX > 0.0 && squareY > 0.0 && maximumX[0] > minimumX[0] && maximumY[0] > minimumY[0]) {
      let squareProduct = squareX * squareY;
      // Preserve the ordinary arithmetic. The product can overflow or underflow even when
      // each sum of squares and the correlation are representable in f32.
      if (squareProduct >= 1.17549435e-38 && isFiniteFloat(squareProduct)) {
        answer = clamp(numerator / sqrt(squareProduct), -1.0, 1.0);
      } else {
        // Integer exponents prevent Metal from regrouping these divisions into the overflowing
        // or underflowing product. Only bounded significands enter the denominator product.
        answer = clamp(getScaledQuotient(numerator, sqrt(squareX), sqrt(squareY), 0), -1.0, 1.0);
      }
    }`
        : 'if (weightTotal > 0.0) { answer = numerator / weightTotal; }'
    }
    ${
      props.weights.status?.overflow || props.weights.status?.candidateOverflow
        ? `if (${[
            props.weights.status?.overflow ? 'inputOverflow[inputOverflowOffset] != 0u' : undefined,
            props.weights.status?.candidateOverflow
              ? 'inputCandidateOverflow[inputCandidateOverflowOffset] != 0u'
              : undefined
          ]
            .filter(Boolean)
            .join(' || ')}) { answer = getQuietNaN(0u); }`
        : ''
    }
    result[resultOffset] = answer;
  }`
  });
  const conversionNodes =
    secondInput === valuesInput ? valuesInput.nodes : [...valuesInput.nodes, ...secondInput.nodes];
  return [...conversionNodes, ...anchorNodes, ...momentsNodes, ...centeredNodes, statisticNode];
}
