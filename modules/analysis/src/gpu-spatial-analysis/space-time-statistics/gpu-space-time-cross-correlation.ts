// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  validateCapacity,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {
  GPUMultipleTestingAdjustment,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/gpu-multiple-testing-adjustment';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';

const OPERATION = 'GPUSpaceTimeCrossCorrelation';
// Overlap endpoints and lag arithmetic use signed 32-bit integers in the shader.
const SPACE_TIME_CORRELATION_MAXIMUM_PERIOD_COUNT = 0x7fffffff;

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
  /** Optional one-row uint32: tested (finite p-value) count, independent of adjustedPValues. */
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
 * `(first[c, t], second[c, t + l])`, each series scaled before centring on its own overlap mean (three passes, f32).
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
    const captured = snapshotConstructionFields(props, [
      'id',
      'first',
      'second',
      'layout',
      'cellCount',
      'periodCount',
      'lags',
      'multipleTesting',
      'output'
    ]);
    props = Object.freeze({
      ...captured,
      lags: Object.freeze([...captured.lags]),
      output: snapshotConstructionFields(captured.output, [
        'correlations',
        'pValues',
        'adjustedPValues',
        'pairCounts',
        'testedCount'
      ])
    });
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
    validateRequirement(
      id,
      props.layout === 'period-major' || props.layout === 'cell-major',
      `layout must be 'period-major' or 'cell-major'`
    );
    for (const lag of lags) {
      validateCapacity(
        id,
        'lag',
        Math.abs(lag),
        0,
        periodCount - GPU_SPACE_TIME_CROSS_CORRELATION_MINIMUM_PAIRS
      );
    }
    validateRequirement(id, new Set(lags).size === lags.length, `lags must be distinct`);
    this.rowCount = cellCount * lags.length;
    // All cube addresses are uint32; periods and signed lags are converted to i32 in WGSL.
    validateCapacity(id, 'cube rows', cellCount * periodCount);
    validateCapacity(id, 'correlation rows', this.rowCount);
    validateCapacity(
      id,
      'periodCount',
      periodCount,
      1,
      SPACE_TIME_CORRELATION_MAXIMUM_PERIOD_COUNT
    );
    for (const [name, view] of [
      ['first', props.first],
      ['second', props.second]
    ] as const) {
      validateSinglePackedView(view, ['float32'], `${id} ${name}`);
      validateViewLength(id, name, view, cellCount * periodCount);
    }
    validateSinglePackedView(output.correlations, ['float32'], `${id} correlations`);
    validateViewLength(id, 'correlations', output.correlations, this.rowCount);
    validateSinglePackedView(output.pValues, ['float32'], `${id} pValues`);
    validateViewLength(id, 'pValues', output.pValues, this.rowCount);
    if (output.adjustedPValues) {
      validateSinglePackedView(output.adjustedPValues, ['float32'], `${id} adjustedPValues`);
      validateViewLength(id, 'adjustedPValues', output.adjustedPValues, this.rowCount);
    }
    if (output.pairCounts) {
      validateSinglePackedView(output.pairCounts, ['uint32'], `${id} pairCounts`);
      validateViewLength(id, 'pairCounts', output.pairCounts, this.rowCount);
    }
    if (output.testedCount) {
      validateSinglePackedView(output.testedCount, ['uint32'], `${id} output.testedCount`);
      validateViewLength(id, 'output.testedCount', output.testedCount, 1);
    }
    resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    validateTrajectoryOutputRanges(this.id, output);
    validateTrajectoryWritableDispatches(this.id, [
      {correlations: output.correlations, pValues: output.pValues, pairCounts: output.pairCounts}
    ]);
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

  /** Returns correlation, optional family counting and optional adjustment nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
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
${SCALED_QUOTIENT_WGSL}
${ERFC_VALUE_WGSL}`,
        body: `let cell = index / LAG_COUNT;
  let lagIndex = index % LAG_COUNT;
  var lagTable = array<i32, ${lags.length}>(${lags.map(lag => `${lag}`).join(', ')});
  let lag = lagTable[lagIndex];
  let start = max(0, -lag);
  let stop = i32(PERIODS) - max(lag, 0);
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

  }
  // A constant series has no correlation; testing the range avoids a rounding residue in the mean.
  if (firstMinimum == firstMaximum || secondMinimum == secondMaximum) {
    return;
  }
  // Normalize before summing or squaring: finite observations can otherwise overflow the
  // mean, covariance or variance product even for a perfectly correlated series.
  let firstScale = max(abs(firstMinimum), abs(firstMaximum));
  let secondScale = max(abs(secondMinimum), abs(secondMaximum));
  let firstAnchor = getScaledQuotient(firstMinimum, firstScale, 1.0, 0);
  let secondAnchor = getScaledQuotient(secondMinimum, secondScale, 1.0, 0);
  for (var period = start; period < stop; period++) {
    firstSum += getScaledQuotient(first[firstOffset + ${address('cell', 'u32(period)')}], firstScale, 1.0, 0) - firstAnchor;
    secondSum += getScaledQuotient(second[secondOffset + ${address('cell', 'u32(period + lag)')}], secondScale, 1.0, 0) - secondAnchor;
  }
  let firstMean = firstSum / f32(pairs);
  let secondMean = secondSum / f32(pairs);
  var firstSquares = 0.0;
  var secondSquares = 0.0;
  var products = 0.0;
  for (var period = start; period < stop; period++) {
    let a = getScaledQuotient(first[firstOffset + ${address('cell', 'u32(period)')}], firstScale, 1.0, 0) - firstAnchor - firstMean;
    let b = getScaledQuotient(second[secondOffset + ${address('cell', 'u32(period + lag)')}], secondScale, 1.0, 0) - secondAnchor - secondMean;
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
  pValues[pValuesOffset + index] = clamp(erfcValue(z * 0.70710678), 0.0, 1.0);`
      })
    ];
    // Both branches count privately so pValues and testedCount may occupy disjoint ranges of
    // one caller buffer, while the adjustment contributor receives separate input/output storage.
    const testedCount = output.testedCount
      ? createTransientView(graph, `${id}-tested-count`, 'uint32', 1)
      : undefined;
    if (output.adjustedPValues) {
      nodes.push(
        ...new GPUMultipleTestingAdjustment({
          id: `${id}-adjust`,
          pValues: output.pValues,
          method: props.multipleTesting ?? 'benjamini-hochberg',
          output: {adjustedPValues: output.adjustedPValues, effectiveFamilySize: testedCount}
        }).getCommandNodes(graph)
      );
    } else if (testedCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-tested-count`,
          operation: OPERATION,
          view: testedCount,
          type: 'u32',
          value: '0u'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-tested-count`,
          operation: OPERATION,
          variant: 'tested-count',
          bindings: [
            {name: 'pValues', view: output.pValues, type: 'f32', access: 'read'},
            {name: 'testedCount', view: testedCount, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rowCount,
          declarations: IS_FINITE_VALUE_WGSL,
          body: `if (isFiniteValue(pValues[pValuesOffset + index])) {
  atomicAdd(&testedCount[testedCountOffset], 1u);
}`
        })
      );
    }
    if (testedCount && output.testedCount) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-tested-count`,
          operation: OPERATION,
          variant: 'publish-tested-count',
          bindings: [
            {name: 'count', view: testedCount, type: 'u32', access: 'read'},
            {name: 'testedCount', view: output.testedCount, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: 'testedCount[testedCountOffset] = count[countOffset];'
        })
      );
    }
    return nodes;
  }
}
