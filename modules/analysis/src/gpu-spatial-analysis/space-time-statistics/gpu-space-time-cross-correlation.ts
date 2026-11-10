// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {
  GPUMultipleTestingAdjustment,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/gpu-multiple-testing-adjustment';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';

const OPERATION = 'GPUSpaceTimeCrossCorrelation';

/** Largest number of lags. */
export const GPU_SPACE_TIME_CROSS_CORRELATION_MAXIMUM_LAGS = 64;
/** Smallest overlap (pairs of periods) of a tested lag: Fisher's z needs `n > 3`. */
export const GPU_SPACE_TIME_CROSS_CORRELATION_MINIMUM_PAIRS = 4;

/** Caller-owned outputs of {@link GPUSpaceTimeCrossCorrelation}, row `cell * lags.length + lagIndex`. */
export type GPUSpaceTimeCrossCorrelationOutput = {
  /** Pearson correlation `r` per (cell, lag), NaN when untestable. */
  correlations: GraphDataView<'float32'>;
  /** Two-sided Fisher-z p-value per (cell, lag), NaN when untestable. */
  pValues: GraphDataView<'float32'>;
  /** Adjusted p-value over every tested (cell, lag), see `multipleTesting`. Must not alias `pValues`. */
  adjustedPValues?: GraphDataView<'float32'>;
  /** Optional overlap pair count `n` per (cell, lag), uint32. */
  pairCounts?: GraphDataView<'uint32'>;
  /** Optional one-row uint32: the size of the multiple-testing family. */
  testedCount?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUSpaceTimeCrossCorrelation}. */
export type GPUSpaceTimeCrossCorrelationProps = {
  /** Prefix for generated node IDs. Defaults to `'space-time-cross-correlation'`. */
  id?: string;
  /** First variable, `cellCount * periodCount` float32 values in `layout`. */
  first: GraphDataView<'float32'>;
  /** Second variable, same shape. */
  second: GraphDataView<'float32'>;
  /** Cube layout of both, see `GPUCubeLayoutTranspose`. */
  layout: GPUSpaceTimeWeightsLayout;
  /** Number of cells (compile-time). */
  cellCount: number;
  /** Number of periods (compile-time). */
  periodCount: number;
  /**
   * Distinct integer lags `l` (compile-time, at most 64). Lag `l` correlates `first[t]` with
   * `second[t + l]` over the periods where both exist, so a positive lag asks whether `second`
   * follows `first` by `l` periods. Every lag needs at least 4 overlapping periods.
   */
  lags: readonly number[];
  /**
   * Correction across every tested (cell, lag) pair, default `'benjamini-hochberg'` (FDR).
   * `'none'` copies the p-values. Applies to `adjustedPValues`.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /** Caller-owned outputs. */
  output: GPUSpaceTimeCrossCorrelationOutput;
};

/**
 * Time-series cross-correlation per location over a set of lags (STX-02), with false discovery
 * control across locations and lags.
 *
 * For cell `c` and lag `l` the statistic is the Pearson correlation of the `n` overlapping pairs
 * `(first[c, t], second[c, t + l])`, each series centred on its own overlap mean (two passes, f32).
 * The p-value is Fisher's z test, `p = erfc(|atanh r| sqrt(n - 3) / sqrt 2)`, a normal
 * approximation that assumes serially independent observations: autocorrelated series give
 * p-values that are too small, so prewhiten (or difference) the series first when that matters.
 * `erfc` is the Numerical Recipes Chebyshev fit (relative error below 1.2e-7). A pair with a
 * non-finite value in the overlap, fewer than 4 pairs, or a constant series is untested (NaN) and
 * not in the correction family. The family of `multipleTesting` is every tested (cell, lag), via
 * {@link GPUMultipleTestingAdjustment} (Benjamini-Hochberg by default).
 *
 * The cube layout is addressed in place, so both layouts work without a transpose.
 */
export class GPUSpaceTimeCrossCorrelation implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeCrossCorrelationProps;
  /** Number of (cell, lag) rows. */
  readonly rowCount: number;

  constructor(props: GPUSpaceTimeCrossCorrelationProps) {
    this.id = props.id ?? 'space-time-cross-correlation';
    this.props = props;
    const id = this.id;
    const {cellCount, periodCount, lags, output} = props;
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(id, 'periodCount', periodCount);
    validateCapacity(
      id,
      'lags length',
      lags.length,
      1,
      GPU_SPACE_TIME_CROSS_CORRELATION_MAXIMUM_LAGS
    );
    if (props.layout !== 'period-major' && props.layout !== 'cell-major') {
      throw new Error(`${id} layout must be 'period-major' or 'cell-major'`);
    }
    for (const lag of lags) {
      validateCapacity(
        id,
        'lag',
        Math.abs(lag),
        0,
        periodCount - GPU_SPACE_TIME_CROSS_CORRELATION_MINIMUM_PAIRS
      );
    }
    if (new Set(lags).size !== lags.length) {
      throw new Error(`${id} lags must be distinct`);
    }
    this.rowCount = cellCount * lags.length;
    for (const [name, view] of [
      ['first', props.first],
      ['second', props.second]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} ${name}`);
      validateViewLength(id, name, view, cellCount * periodCount);
    }
    validatePackedView(output.correlations, ['float32'], `${id} correlations`);
    validateViewLength(id, 'correlations', output.correlations, this.rowCount);
    validatePackedView(output.pValues, ['float32'], `${id} pValues`);
    validateViewLength(id, 'pValues', output.pValues, this.rowCount);
    if (output.adjustedPValues) {
      validatePackedView(output.adjustedPValues, ['float32'], `${id} adjustedPValues`);
      validateViewLength(id, 'adjustedPValues', output.adjustedPValues, this.rowCount);
    }
    if (output.pairCounts) {
      validatePackedView(output.pairCounts, ['uint32'], `${id} pairCounts`);
      validateViewLength(id, 'pairCounts', output.pairCounts, this.rowCount);
    }
    resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.correlations,
        output.pValues,
        output.adjustedPValues,
        output.pairCounts,
        output.testedCount
      ],
      [props.first, props.second]
    );
  }

  /** Returns the correlation kernel and, with an adjusted-p output, the adjustment nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount} = this;
    const {first, second, layout, cellCount, periodCount, lags, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      first,
      second,
      output.correlations,
      output.pValues,
      output.adjustedPValues,
      output.pairCounts,
      output.testedCount
    ]);
    const address = (cell: string, period: string) =>
      layout === 'period-major'
        ? `(${period}) * CELLS + ${cell}`
        : `${cell} * PERIODS + (${period})`;
    const bindings = [
      {name: 'first', view: first, type: 'f32' as const, access: 'read' as const},
      {name: 'second', view: second, type: 'f32' as const, access: 'read' as const},
      {
        name: 'correlations',
        view: output.correlations,
        type: 'f32' as const,
        access: 'read_write' as const
      },
      {name: 'pValues', view: output.pValues, type: 'f32' as const, access: 'read_write' as const},
      ...(output.pairCounts
        ? [
            {
              name: 'pairCounts',
              view: output.pairCounts,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-correlate`,
        operation: OPERATION,
        variant: 'correlate',
        bindings,
        invocationCount: rowCount,
        declarations: `const CELLS: u32 = ${cellCount}u;
const PERIODS: u32 = ${periodCount}u;
const LAG_COUNT: u32 = ${lags.length}u;
const MINIMUM_PAIRS: u32 = ${GPU_SPACE_TIME_CROSS_CORRELATION_MINIMUM_PAIRS}u;
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
// Numerical Recipes erfcc, relative error below 1.2e-7 for x >= 0.
fn erfcPositive(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.5 * x);
  let polynomial = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return t * exp(polynomial);
}`,
        body: `let cell = index / LAG_COUNT;
  let lagIndex = index % LAG_COUNT;
  var lagTable = array<i32, ${lags.length}>(${lags.map(lag => `${lag}`).join(', ')});
  let lag = lagTable[lagIndex];
  let start = max(0, -lag);
  let stop = min(i32(PERIODS), i32(PERIODS) - lag);
  let pairs = u32(max(stop - start, 0));
  ${output.pairCounts ? 'pairCounts[pairCountsOffset + index] = pairs;' : ''}
  correlations[correlationsOffset + index] = getNaN();
  pValues[pValuesOffset + index] = getNaN();
  if (pairs < MINIMUM_PAIRS) {
    return;
  }
  var firstSum = 0.0;
  var secondSum = 0.0;
  var firstMinimum = 3.4e38;
  var firstMaximum = -3.4e38;
  var secondMinimum = 3.4e38;
  var secondMaximum = -3.4e38;
  for (var period = start; period < stop; period++) {
    let a = first[firstOffset + ${address('cell', 'u32(period)')}];
    let b = second[secondOffset + ${address('cell', 'u32(period + lag)')}];
    if (!isFiniteValue(a) || !isFiniteValue(b)) {
      return;
    }
    firstMinimum = min(firstMinimum, a);
    firstMaximum = max(firstMaximum, a);
    secondMinimum = min(secondMinimum, b);
    secondMaximum = max(secondMaximum, b);
    firstSum += a;
    secondSum += b;
  }
  // A constant series has no correlation; testing the range avoids a rounding residue in the mean.
  if (firstMinimum == firstMaximum || secondMinimum == secondMaximum) {
    return;
  }
  let firstMean = firstSum / f32(pairs);
  let secondMean = secondSum / f32(pairs);
  var firstSquares = 0.0;
  var secondSquares = 0.0;
  var products = 0.0;
  for (var period = start; period < stop; period++) {
    let a = first[firstOffset + ${address('cell', 'u32(period)')}] - firstMean;
    let b = second[secondOffset + ${address('cell', 'u32(period + lag)')}] - secondMean;
    firstSquares += a * a;
    secondSquares += b * b;
    products += a * b;
  }
  let denominator = sqrt(firstSquares * secondSquares);
  if (!(denominator > 0.0)) {
    return;
  }
  let correlation = clamp(products / denominator, -1.0, 1.0);
  correlations[correlationsOffset + index] = correlation;
  if (abs(correlation) >= 1.0) {
    pValues[pValuesOffset + index] = 0.0;
    return;
  }
  let z = abs(0.5 * log((1.0 + correlation) / (1.0 - correlation))) * sqrt(f32(pairs) - 3.0);
  pValues[pValuesOffset + index] = clamp(erfcPositive(z * 0.70710678), 0.0, 1.0);`
      })
    ];
    if (output.adjustedPValues) {
      nodes.push(
        ...new GPUMultipleTestingAdjustment({
          id: `${id}-adjust`,
          pValues: output.pValues,
          method: props.multipleTesting ?? 'benjamini-hochberg',
          output: {adjustedPValues: output.adjustedPValues, testedCount: output.testedCount}
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
