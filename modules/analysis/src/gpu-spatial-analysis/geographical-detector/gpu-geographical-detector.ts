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
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPURegionPartitionEvaluation
} from '../spatial-regionalization/gpu-region-partition-evaluation';

const OPERATION = 'GPUGeographicalDetector';
/** Labels reduced by one thread of the tile pass. */
const LABEL_TILE = 256;

/** Layout of the `output.summary` words written by {@link GPUGeographicalDetector}. */
export const GPU_GEOGRAPHICAL_DETECTOR_LAYOUT = {
  /**
   * The q statistic `1 - SSW / SST` (Wang et al. 2010): the share of the variance of `values`
   * explained by the stratification, in [0, 1]. NaN when `SST` is 0 or no row is labelled.
   */
  q: 0,
  /**
   * `F = ((N - L) / (L - 1)) * q / (1 - q)`, distributed as a non-central F with `L - 1` and
   * `N - L` degrees of freedom and parameter `nonCentrality` under the stratified model.
   * `+Infinity` when `q` is 1. NaN when `L < 2` or `N <= L`.
   */
  fStatistic: 1,
  /**
   * `lambda = (sum_h mean_h^2 - (sum_h sqrt(N_h) mean_h)^2 / N) / sigma^2` with `sigma^2 = SST / N`
   * the population variance. This is the formula published by Wang et al. (2010) and implemented
   * by the R `GD` package and `geodetector` 0.2.0; it is not invariant to shifting `values`.
   */
  nonCentrality: 2,
  /** Number of non-empty strata `L`. */
  strataCount: 3,
  /** Number of labelled rows `N`. */
  observationCount: 4,
  /** `L - 1`. */
  numeratorDegreesOfFreedom: 5,
  /** `N - L`. */
  denominatorDegreesOfFreedom: 6,
  /** Total sum of squares `SST` of the labelled rows. */
  totalSumOfSquares: 7,
  /** Total word count. */
  length: 8
} as const;

/**
 * Properties for {@link GPUGeographicalDetector}.
 *
 * Compile-time: row count and `labelCapacity`. Per-frame: the contents of `values` and `labels`.
 */
export type GPUGeographicalDetectorProps = {
  /** Prefix for generated node IDs. Defaults to `'geographical-detector'`. */
  id?: string;
  /** Response variable `Y`, one value per row. */
  values: GraphDataView<'float32'>;
  /**
   * Stratum per row: the explanatory variable `X` already stratified (a classification, an
   * administrative code or a zoning). `0xffffffff` and labels at or above `labelCapacity` leave
   * the row out.
   */
  labels: GraphDataView<'uint32'>;
  /** Number of distinct label values `[0, labelCapacity)`. Defaults to the row count. */
  labelCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUGeographicalDetectorOutput;
};

/** Caller-owned outputs of {@link GPUGeographicalDetector}. */
export type GPUGeographicalDetectorOutput = {
  /** Summary words, see {@link GPU_GEOGRAPHICAL_DETECTOR_LAYOUT}. */
  summary: GraphDataView<'float32'>;
};

/**
 * Factor detector of the geographical detector (Wang, Li, Christakos et al. 2010; Wang, Zhang and
 * Fu 2016): the q statistic of a stratification of one response variable, with its non-central F
 * statistic and non-centrality parameter.
 *
 * `q = 1 - SSW / SST` equals the between-strata share of the total sum of squares, so the sums of
 * squares, strata sizes and strata means come from {@link GPURegionPartitionEvaluation} (a stable
 * sort by stratum, load-balanced segment sums, two-pass deviations about the mean). One extra
 * tiled pass reduces the strata means for the non-centrality parameter. Work is
 * `O(rows)` plus the sort and `O(labelCapacity)`; no float atomics are used, so results are
 * bitwise reproducible.
 *
 * **P-value.** The p-value `1 - F_ncf(F; L - 1, N - L, lambda)` needs a non-central F
 * distribution function and is computed on the host from the read-back summary with
 * `getGeographicalDetectorPValue`. It is not computed on the GPU.
 *
 * **Reference.** `geodetector` 0.2.0 (MIT) `q_statistic` and `q_significance_test`, which match
 * the R `GD` package. Like those, `lambda` uses the population variance. Unlike them, strata with
 * a single row are kept (they add nothing to SSW and count in `L` and `N`); the reference drops
 * them, so compare on data whose strata all have at least two rows.
 *
 * **float32 accuracy.** Sums of squares are float32 sums of squared deviations about a mean
 * computed in float32, so `q` agrees with a binary64 evaluation to about `1e-5` for well scaled
 * data. `lambda` subtracts two nearly equal sums when the strata means are large compared with
 * their spread, and loses digits in proportion; centre or scale `values` first when that
 * matters, noting the formula is not shift invariant.
 */
export class GPUGeographicalDetector implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeographicalDetectorProps;

  constructor(props: GPUGeographicalDetectorProps) {
    const id = props.id ?? 'geographical-detector';
    this.id = id;
    this.props = props;
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.labels, ['uint32'], `${id} labels`);
    validateRequirement(
      id,
      props.values.length === props.labels.length && props.labels.length > 0,
      'values and labels must have the same positive length'
    );
    validatePackedView(props.output.summary, ['float32'], `${id} output.summary`);
    validateRequirement(
      id,
      props.output.summary.length >= GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.length,
      `output.summary must hold ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.length} words`
    );
  }

  /** Returns the partition evaluation, strata mean tile and finalize nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.values, props.labels, props.output.summary]);
    const labelCapacity = props.labelCapacity ?? props.labels.length;
    const tileCount = Math.ceil(labelCapacity / LABEL_TILE);
    const evaluationSummary = createTransientView(
      graph,
      `${id}-evaluation-summary`,
      'float32',
      GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
    );
    const sizes = createTransientView(graph, `${id}-sizes`, 'uint32', labelCapacity);
    const means = createTransientView(graph, `${id}-means`, 'float32', labelCapacity);
    const tiles = createTransientView(graph, `${id}-tiles`, 'float32', tileCount * 2);
    const declarations = `const LABELS: u32 = ${labelCapacity}u;
const TILES: u32 = ${tileCount}u;
const LABEL_TILE: u32 = ${LABEL_TILE}u;`;
    return [
      ...new GPURegionPartitionEvaluation({
        id: `${id}-evaluation`,
        values: props.values,
        labels: props.labels,
        labelCapacity,
        output: {summary: evaluationSummary, regionSizes: sizes, regionMeans: means}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mean-tiles`,
        operation: OPERATION,
        variant: 'mean-tiles',
        bindings: [
          {name: 'sizes', view: sizes, type: 'u32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read'},
          {name: 'tiles', view: tiles, type: 'f32', access: 'read_write'}
        ],
        invocationCount: tileCount,
        declarations,
        body: `var squares = 0.0;
  var roots = 0.0;
  let first = index * LABEL_TILE;
  for (var label = first; label < min(first + LABEL_TILE, LABELS); label++) {
    let size = sizes[sizesOffset + label];
    if (size > 0u) {
      let mean = means[meansOffset + label];
      squares += mean * mean;
      roots += sqrt(f32(size)) * mean;
    }
  }
  tiles[tilesOffset + 2u * index] = squares;
  tiles[tilesOffset + 2u * index + 1u] = roots;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'evaluation', view: evaluationSummary, type: 'f32', access: 'read'},
          {name: 'tiles', view: tiles, type: 'f32', access: 'read'},
          {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${declarations}
const ROWS: u32 = ${props.labels.length}u;
${GET_NAN_WGSL}`,
        body: `var squares = 0.0;
  var roots = 0.0;
  for (var tile = 0u; tile < TILES; tile++) {
    squares += tiles[tilesOffset + 2u * tile];
    roots += tiles[tilesOffset + 2u * tile + 1u];
  }
  let strata = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount}u];
  let within = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares}u];
  let total = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares}u];
  let observations = f32(ROWS) - evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.ignoredRows}u];
  let nan = getNaN();
  let defined = observations > 0.0 && total > 0.0;
  let q = select(nan, clamp(1.0 - within / total, 0.0, 1.0), defined);
  let degreesNumerator = strata - 1.0;
  let degreesDenominator = observations - strata;
  let fDefined = defined && strata >= 2.0 && degreesDenominator > 0.0;
  var infinityBits = 0x7f800000u;
  let infinity = bitcast<f32>(infinityBits);
  let ratio = degreesDenominator * q / (degreesNumerator * (1.0 - q));
  let fStatistic = select(nan, select(ratio, infinity, q >= 1.0), fDefined);
  let variance = total / max(observations, 1.0);
  let nonCentrality = select(nan, (squares - roots * roots / observations) / variance, defined);
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.q}u] = q;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.fStatistic}u] = fStatistic;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.nonCentrality}u] = nonCentrality;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.strataCount}u] = strata;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.observationCount}u] = observations;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.numeratorDegreesOfFreedom}u] = degreesNumerator;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.denominatorDegreesOfFreedom}u] = degreesDenominator;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.totalSumOfSquares}u] = total;`
      })
    ];
  }
}
