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
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getRowTileMergeNodes} from '../spatial-regression/regression-row-tiles';
import {
  createDetectorNormalizationNodes,
  DETECTOR_NORMALIZATION_WGSL
} from './geographical-detector-normalization';
import {
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPURegionPartitionEvaluation,
  getGPURegionPartitionEvaluationMaximumRowCount
} from '../spatial-regionalization/gpu-region-partition-evaluation';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {
  validateCapacity,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

const OPERATION = 'GPUGeographicalDetector';
/** Labels reduced by one thread of the tile pass. */
const LABEL_TILE = 256;

/** Layout of the `output.summary` words written by {@link GPUGeographicalDetector}. */
export const GPU_GEOGRAPHICAL_DETECTOR_LAYOUT = {
  /**
   * The q statistic `1 - SSW / SST` (Wang et al. 2010): the share of the variance of `values`
   * explained by the stratification, in [0, 1], evaluated as directly accumulated SSB / SST
   * to retain small explained variation. Exactly 1 when SSW is zero and SSB is positive.
   * NaN when mathematical SST is 0 or no row is labelled. Ratios use normalized moments for
   * uniformly tiny responses even when the physical totalSumOfSquares rounds to zero.
   */
  q: 0,
  /**
   * `F = ((N - L) / (L - 1)) * q / (1 - q)`, distributed as a non-central F with `L - 1` and
   * `N - L` degrees of freedom and parameter `nonCentrality` under the stratified model.
   * Uses the partition's exponent-safe `((N-L)/(L-1)) * SSB/SSW` for positive SSW.
   * `+Infinity` only when SSW is zero and between-stratum SS is positive.
   * NaN when `L < 2`, `N <= L`, or a nonperfect F overflows float32.
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
  /** Number of labelled rows with a finite value `N`. */
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
  /**
   * Response variable `Y`, one value per row. A row whose value is not finite (NaN or infinite) is
   * left out of every statistic, like an unlabelled row.
   */
  values: GraphDataView<'float32'>;
  /**
   * Stratum per row: the explanatory variable `X` already stratified (a classification, an
   * administrative code or a zoning). `0xffffffff` and labels at or above `labelCapacity` leave
   * the row out.
   */
  labels: GraphDataView<'uint32'>;
  /** Optional scalar solver certificate, nonzero when the supplied partition is complete. */
  labelsComplete?: GraphDataView<'uint32'>;
  /** Number of distinct label values `[0, labelCapacity)`. Defaults to the row count. */
  labelCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUGeographicalDetectorOutput;
};

/** Caller-owned outputs of {@link GPUGeographicalDetector}. */
export type GPUGeographicalDetectorOutput = {
  /** Scalar copy of the optional input certificate; statistics still describe the incumbent. */
  labelsComplete?: GraphDataView<'uint32'>;
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
 * Uniformly tiny responses (largest usable magnitude below 2^-32) are normalized with one
 * shared power of two before moments are evaluated. Ratios retain variation before squaring;
 * totalSumOfSquares is restored to response-squared units with bit-safe subnormal rounding.
 *
 * **Missing values.** A row whose value is NaN or infinite is left out of every statistic (it is
 * given no stratum before the partition evaluation), so it does not count in `N`.
 * The label/response row limit is 16,777,216 at a 128 MiB storage binding limit. The
 * partition stage also bounds `labelCapacity` to 33,554,431 at that binding size.
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
    const captured = snapshotConstructionFields(props, [
      'id',
      'values',
      'labels',
      'labelsComplete',
      'labelCapacity',
      'output'
    ]);
    props = Object.freeze({
      ...captured,
      output: snapshotConstructionFields(captured.output, ['summary', 'labelsComplete'])
    });
    const id = props.id ?? 'geographical-detector';
    this.id = id;
    this.props = props;
    validateCapacity(id, 'labelCapacity', props.labelCapacity ?? props.labels.length);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.summary, props.output.labelsComplete],
      [props.values, props.labels, props.labelsComplete]
    );
    validateRequirement(
      id,
      Boolean(props.labelsComplete) === Boolean(props.output.labelsComplete),
      'labelsComplete input and output.labelsComplete must both be supplied to retain partition provenance'
    );
    for (const [name, view] of [
      ['labelsComplete', props.labelsComplete],
      ['output.labelsComplete', props.output.labelsComplete]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        validateCapacity(id, `${name} length`, view.length, 1, 1);
      }
    }
    validateRequirement(
      id,
      !props.output.labelsComplete ||
        props.output.labelsComplete.buffer !== props.output.summary.buffer,
      'output.labelsComplete and output.summary must use separate buffers'
    );
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    validateSinglePackedView(props.labels, ['uint32'], `${id} labels`);
    validateRequirement(
      id,
      props.values.length === props.labels.length && props.labels.length > 0,
      'values and labels must have the same positive length'
    );
    validateSinglePackedView(props.output.summary, ['float32'], `${id} output.summary`);
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
    validateSpatialStatisticsBindings(graph, id, props, {
      'label offsets scratch': ((props.labelCapacity ?? props.labels.length) + 1) * 4,
      'partition summary': GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4,
      'analysis labels': props.labels.length * 4,
      'normalization magnitudes': props.labels.length * 4,
      'normalized responses': props.labels.length * 4,
      'mean tiles': Math.ceil((props.labelCapacity ?? props.labels.length) / LABEL_TILE) * 8
    });
    validateMaximumCount(
      id,
      'labels rows',
      props.labels.length,
      getGPURegionPartitionEvaluationMaximumRowCount(graph.device)
    );
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.labels,
      props.labelsComplete,
      props.output.summary,
      props.output.labelsComplete
    ]);
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
    // Labels of the analysis rows: a row with a non-finite value is unlabelled, so one NaN cannot
    // reach the sums of the partition evaluation, which has no finiteness test of its own.
    const analysisLabels = createTransientView(
      graph,
      `${id}-analysis-labels`,
      'uint32',
      props.labels.length
    );
    const declarations = `const LABELS: u32 = ${labelCapacity}u;
const TILES: u32 = ${tileCount}u;
const LABEL_TILE: u32 = ${LABEL_TILE}u;`;
    const normalization = createDetectorNormalizationNodes(
      graph,
      id,
      props.values,
      props.labels,
      labelCapacity
    );
    return [
      ...normalization.nodes,
      ...(props.labelsComplete && props.output.labelsComplete
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-partition-provenance`,
              operation: OPERATION,
              variant: 'partition-provenance',
              invocationCount: 1,
              bindings: [
                {name: 'labelsComplete', view: props.labelsComplete, type: 'u32', access: 'read'},
                {
                  name: 'complete',
                  view: props.output.labelsComplete,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              body: 'complete[completeOffset] = u32(labelsComplete[labelsCompleteOffset] != 0u);'
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-analysis-labels`,
        operation: OPERATION,
        variant: 'analysis-labels',
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
          {name: 'analysisLabels', view: analysisLabels, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.labels.length,
        declarations: IS_FINITE_FLOAT_WGSL,
        body: `analysisLabels[analysisLabelsOffset + index] = select(
    0xffffffffu,
    labels[labelsOffset + index],
    isFiniteFloat(values[valuesOffset + index])
  );`
      }),
      ...new GPURegionPartitionEvaluation({
        id: `${id}-evaluation`,
        values: normalization.normalized,
        labels: analysisLabels,
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
      ...getRowTileMergeNodes<Parameters>(
        graph,
        {
          id: `${id}-summary`,
          operation: OPERATION,
          variant: 'summary',
          bindings: [
            {name: 'evaluation', view: evaluationSummary, type: 'f32', access: 'read'},
            {name: 'magnitude', view: normalization.magnitude, type: 'u32', access: 'read'},
            {name: 'tiles', view: tiles, type: 'f32', access: 'read'},
            {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'}
          ],
          // contract-allow: single-invocation -- REGRESSION_TILE_MERGE_GROUP_SIZE = 4096 bounds two-word partial merges; getRowTileMergeNodes adds parallel groups above that limit, with no input-row scan.
          invocationCount: 1,
          declarations: `${declarations}
const ROWS: u32 = ${props.labels.length}u;
${GET_NAN_WGSL}
${IS_FINITE_FLOAT_WGSL}
${DETECTOR_NORMALIZATION_WGSL}`,
          body: `var squares = 0.0;
  var roots = 0.0;
  for (var tile = 0u; tile < TILES; tile++) {
    squares += tiles[tilesOffset + 2u * tile];
    roots += tiles[tilesOffset + 2u * tile + 1u];
  }
  let strata = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount}u];
  let within = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares}u];
  let total = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares}u];
  let between = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares}u];
  let observations = f32(ROWS) - evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.ignoredRows}u];
  let nan = getNaN();
  let defined = observations > 0.0 && total > 0.0;
  let perfect = (bitcast<u32>(within) & 0x7fffffffu) == 0u && between > 0.0;
  let q = select(nan, select(clamp(between / total, 0.0, 1.0), 1.0, perfect), defined);
  let degreesNumerator = strata - 1.0;
  let degreesDenominator = observations - strata;
  let fDefined = defined && strata >= 2.0 && degreesDenominator > 0.0;
  var infinityBits = 0x7f800000u;
  let infinity = bitcast<f32>(infinityBits);
  let ratio = evaluation[evaluationOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF}u];
  let finiteRatio = select(nan, ratio, isFiniteFloat(ratio));
  let fStatistic = select(nan, select(finiteRatio, infinity, perfect), fDefined);
  let variance = total / max(observations, 1.0);
  let nonCentrality = select(nan, (squares - roots * roots / observations) / variance, defined);
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.q}u] = q;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.fStatistic}u] = fStatistic;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.nonCentrality}u] = nonCentrality;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.strataCount}u] = strata;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.observationCount}u] = observations;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.numeratorDegreesOfFreedom}u] = degreesNumerator;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.denominatorDegreesOfFreedom}u] = degreesDenominator;
  summary[summaryOffset + ${GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.totalSumOfSquares}u] = restoreDetectorSumOfSquares(total, detectorNormalizationShift(magnitude[magnitudeOffset]));`
        },
        {tileCount, partials: [{name: 'tiles', termCount: 2, countConstant: 'TILES'}]}
      )
    ];
  }
}
