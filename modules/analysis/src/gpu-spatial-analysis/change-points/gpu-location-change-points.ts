// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {POISSON_OBSERVED_TERM_WGSL} from '../../analysis-core/kernels/wgsl-poisson-observed-term';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUOverflowStatusPort} from '../../analysis-core/contracts/status';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';
import {getChangePointSeriesReaderWGSL} from './change-point-series-wgsl';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPULocationChangePoints';

/** Elements of the `parameters` view: `[penalty, minimumSegmentLength]`. */
export const GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH = 2;
/**
 * Largest `sliceCount`. Every location keeps its standardized series and the dynamic program in
 * workgroup arrays and evaluates `sliceCount^2 / 2` segment costs, so the bound limits both.
 */
export const GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT = 512;
/**
 * Largest `sliceCount` of the `'slope'` cost: the time sums of a segment are exact unsigned
 * integers, and `n * sum(t^2)` with positions below 256 stays below `2^32`.
 */
export const GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT = 256;
/** Largest `maximumChangePoints` per location. */
export const GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS = 128;
/** Sentinel of unused `changePoints` rows. */
export const GPU_LOCATION_CHANGE_POINTS_NONE = 0xffffffff;

/** Segment cost of {@link GPULocationChangePoints}. */
export type GPULocationChangePointCost =
  | 'mean'
  | 'variance'
  | 'mean-variance'
  | 'poisson'
  | 'slope';

/** Smallest admissible segment length per cost: a cost must be defined on a segment that short. */
const MINIMUM_SEGMENT_LENGTHS: Record<GPULocationChangePointCost, number> = {
  mean: 1,
  poisson: 1,
  variance: 2,
  'mean-variance': 2,
  slope: 3
};

/** Per-frame settings of {@link GPULocationChangePoints}. */
export type GPULocationChangePointSettings = {
  /**
   * Penalty `beta` per added change point, finite and non-negative, on the scale of the cost: see
   * {@link GPULocationChangePoints}. For the `changepoint` package's `SIC`: `(p + 1) ln n` with
   * `p` the parameters added per segment (1 for mean, variance and Poisson, 2 for mean-variance and
   * slope, the Poisson and Normal likelihoods assume unit-variance scaling for mean and slope).
   */
  penalty: number;
  /**
   * Smallest number of finite slices per segment. Defaults to the smallest length the cost is
   * defined on: 1 for mean and Poisson, 2 for variance and mean-variance, 3 for slope.
   */
  minimumSegmentLength?: number;
};

/** Caller-owned outputs of {@link GPULocationChangePoints}. */
export type GPULocationChangePointsOutput = {
  /** Change points per cell, clamped to `maximumChangePoints`. */
  changePointCount: GraphDataView<'uint32'>;
  /**
   * `cellCount * maximumChangePoints` uint32 rows `cell * maximumChangePoints + index`: the first
   * slice of each segment after the first, ascending; unused rows hold
   * `GPU_LOCATION_CHANGE_POINTS_NONE`. The earliest change points are kept when a cell has too many.
   */
  changePoints: GraphDataView<'uint32'>;
  /** Optional per-cell unclamped change point count. */
  requiredChangePointCounts?: GraphDataView<'uint32'>;
  /**
   * Optional per-cell penalized optimum: the sum of the segment costs plus `penalty` per change
   * point (no penalty for the first segment). NaN for a cell without a defined series.
   */
  optimalCost?: GraphDataView<'float32'>;
  /** Optional scalar status: `overflow` is 1 when any cell had more than `maximumChangePoints`. */
  status?: Pick<GPUOverflowStatusPort, 'overflow'>;
};

/** Properties for {@link GPULocationChangePoints}. */
export type GPULocationChangePointsProps = {
  /** Prefix for generated node IDs. Defaults to `'location-change-points'`. */
  id?: string;
  /** `cellCount * sliceCount` float32 values in `layout`. NaN marks a missing slice, skipped. */
  values: GraphDataView<'float32'>;
  /** Cube layout, see `GPUCubeLayoutTranspose`. */
  layout: GPUSpaceTimeWeightsLayout;
  /** Compile-time number of cells. */
  cellCount: number;
  /** Compile-time slices per cell, at most `GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT`. */
  sliceCount: number;
  /** Compile-time segment cost. */
  cost: GPULocationChangePointCost;
  /** Compile-time capacity of change points per cell, at most `GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS`. */
  maximumChangePoints: number;
  /** Per-frame `[penalty, minimumSegmentLength]`, written by {@link GPULocationChangePoints.getParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPULocationChangePointsOutput;
};

/**
 * Per-location change points by unpruned optimal partitioning (Jackson et al. 2005): the
 * dynamic program `F(t) = min_s F(s) + C(s + 1 .. t) + beta` that PELT (Killick et al. 2012)
 * prunes, so the optimum is the same with `sliceCount^2 / 2` bounded segment evaluations. One
 * invocation per location; the workload is `cellCount * sliceCount^2 / 2` segment costs.
 *
 * Costs are minus twice the Normal or Poisson log-likelihood, the definitions of the
 * R `changepoint` package (Killick and Eckley 2014), with the additive terms that do not depend
 * on the segmentation removed:
 * - `'mean'`: `sum (y - mean)^2` over the segment, the unit-variance Normal likelihood (standardize
 *   `values`, or scale `penalty` by the variance).
 * - `'variance'`: `n (ln(2 pi) + ln(s^2) + 1)` with `s^2 = sum (y - m)^2 / n` and `m` the mean of the
 *   whole finite series (`cpt.var` with `know.mean = FALSE`).
 * - `'mean-variance'`: as `'variance'` with `s^2` the segment's own residual variance.
 * - `'poisson'`: `2 x (ln n - ln x)` for segment total `x` of non-negative counts.
 * - `'slope'`: the residual sum of squares of a line per segment (positions are the true slice
 *   indices, so missing slices keep their spacing); a unit-variance Normal cost like `'mean'`.
 *
 * Precision: values are centered and scaled by the series' own mean and standard deviation
 * (a deviance form for Poisson), so the partition does not depend on the unit of `values`. The
 * moments of a segment are running sums that begin at the segment's own start, taken about its
 * first value, so a quiet segment after a loud one keeps f32 relative accuracy over 512 slices;
 * no difference of two long prefix sums is formed. The series mean and variance use a sum
 * compensated by `getAdditionRoundingError`. Constant series and all-zero count series have no
 * change points. Minimum segments count finite slices.
 *
 * Variance floor: for `'variance'` and `'mean-variance'` a segment variance below `1e-11` times the
 * variance of the whole series is raised to that value, so a constant segment costs a finite
 * amount whatever the unit. The `changepoint` package instead substitutes `1e-11` in data units
 * when a segment variance is not positive; the two agree on which segments are affected only when
 * the series variance is near 1. A constant series has no scale to be relative to: its
 * `optimalCost` uses the package's `1e-11` in data units.
 *
 * Per-frame: the contents of `values` and `parameters` (`penalty`, `minimumSegmentLength`).
 * Compile-time: `cellCount`, `sliceCount`, `cost`, `layout`, `maximumChangePoints`, and which
 * optional outputs exist.
 *
 * The `changepoint` R package is a definition source only; no code of it is used.
 */
export class GPULocationChangePoints implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocationChangePointsProps;

  constructor(props: GPULocationChangePointsProps) {
    const id = props.id ?? 'location-change-points';
    this.id = id;
    this.props = props;
    const {values, cellCount, sliceCount, cost, maximumChangePoints, parameters, output} = props;
    validateRequirement(id, cost in MINIMUM_SEGMENT_LENGTHS, 'cost is not supported');
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(
      id,
      'sliceCount',
      sliceCount,
      1,
      GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT
    );
    if (cost === 'slope') {
      validateCapacity(
        id,
        "sliceCount of the 'slope' cost",
        sliceCount,
        1,
        GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT
      );
    }
    validateCapacity(
      id,
      'maximumChangePoints',
      maximumChangePoints,
      1,
      GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS
    );
    validateSinglePackedView(values, ['float32'], `${id} values`);
    validateViewLength(id, 'values', values, cellCount * sliceCount);
    validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      parameters.length >= GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH,
      `parameters needs ${GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH} rows`
    );
    validateSinglePackedView(output.changePointCount, ['uint32'], `${id} changePointCount`);
    validateViewLength(id, 'changePointCount', output.changePointCount, cellCount);
    validateSinglePackedView(output.changePoints, ['uint32'], `${id} changePoints`);
    validateViewLength(id, 'changePoints', output.changePoints, cellCount * maximumChangePoints);
    if (output.requiredChangePointCounts) {
      validateSinglePackedView(
        output.requiredChangePointCounts,
        ['uint32'],
        `${id} requiredChangePointCounts`
      );
      validateViewLength(
        id,
        'requiredChangePointCounts',
        output.requiredChangePointCounts,
        cellCount
      );
    }
    if (output.optimalCost) {
      validateSinglePackedView(output.optimalCost, ['float32'], `${id} optimalCost`);
      validateViewLength(id, 'optimalCost', output.optimalCost, cellCount);
    }
    if (output.status?.overflow) {
      validateSinglePackedView(output.status.overflow, ['uint32'], `${id} status.overflow`);
    }
    validateDisjointWritableViews(id, {...output, status: undefined, ...output.status});
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.changePointCount,
        output.changePoints,
        output.requiredChangePointCounts,
        output.optimalCost,
        output.status?.overflow
      ],
      [values, parameters]
    );
  }

  /** Packs `settings`, rejecting a penalty or segment length the cost does not allow. */
  getParameterValues(
    settings: GPULocationChangePointSettings,
    target: Float32Array = new Float32Array(GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH)
  ): Float32Array {
    const {id, props} = this;
    validateFiniteNumber(id, 'penalty', settings.penalty, 0);
    const smallest = MINIMUM_SEGMENT_LENGTHS[props.cost];
    const minimumSegmentLength = settings.minimumSegmentLength ?? smallest;
    validateCapacity(id, 'minimumSegmentLength', minimumSegmentLength, smallest, props.sliceCount);
    target.set([settings.penalty, minimumSegmentLength]);
    return target;
  }

  /** Returns the one kernel of this contributor. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateSpatialStatisticsBindings(graph, id, props);
    const {values, layout, cellCount, sliceCount, cost, maximumChangePoints, parameters, output} =
      props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      parameters,
      output.changePointCount,
      output.changePoints,
      output.requiredChangePointCounts,
      output.optimalCost,
      output.status?.overflow
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'values',
      cellCount * sliceCount * 4,
      'reduce cellCount * sliceCount'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'changePoints',
      cellCount * maximumChangePoints * 4,
      'lower maximumChangePoints or split the cells'
    );
    return [
      ...(output.status?.overflow
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-clear-status`,
              operation: OPERATION,
              variant: 'clear-status',
              bindings: [
                {
                  name: 'overflowStatus',
                  view: output.status.overflow,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              invocationCount: 1,
              body: 'overflowStatus[overflowStatusOffset] = 0u;'
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-dynamic-program`,
        operation: OPERATION,
        variant: cost,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {
            name: 'changePointCount',
            view: output.changePointCount,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'changePoints', view: output.changePoints, type: 'u32', access: 'read_write'},
          ...(output.requiredChangePointCounts
            ? [
                {
                  name: 'requiredChangePointCounts',
                  view: output.requiredChangePointCounts,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(output.optimalCost
            ? [
                {
                  name: 'optimalCost',
                  view: output.optimalCost,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(output.status?.overflow
            ? [
                {
                  name: 'overflowStatus',
                  view: output.status.overflow,
                  type: 'atomic<u32>' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: cellCount * 2,
        // One location per two-thread workgroup, the second thread idle: the standardized series and
        // the dynamic program live in workgroup memory (16 KiB by default), which unlike private arrays
        // does not grow the scratch of each invocation.
        workgroupSize: 2,
        declarations: getDeclarationsWGSL(cost, layout, cellCount, sliceCount, maximumChangePoints),
        body: getDynamicProgramBodyWGSL(cost, output)
      })
    ];
  }
}

function getDeclarationsWGSL(
  cost: GPULocationChangePointCost,
  layout: GPUSpaceTimeWeightsLayout,
  cellCount: number,
  sliceCount: number,
  maximumChangePoints: number
): string {
  const arraySize = sliceCount + 1;
  const isPoisson = cost === 'poisson';
  return /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
${GET_NAN_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
${isPoisson ? POISSON_OBSERVED_TERM_WGSL : ''}
var<workgroup> position: array<u32, ${sliceCount}>;
var<workgroup> series: array<f32, ${sliceCount}>;
var<workgroup> bestCost: array<f32, ${arraySize}>;
var<workgroup> previousEnd: array<u32, ${arraySize}>;
const CELL_COUNT: u32 = ${cellCount}u;
const SLICE_COUNT: u32 = ${sliceCount}u;
const MAXIMUM_CHANGE_POINTS: u32 = ${maximumChangePoints}u;
const NO_CHANGE_POINT: u32 = ${GPU_LOCATION_CHANGE_POINTS_NONE}u;
const TAU: f32 = 6.283185307179586;
// Smallest segment variance as a fraction of the series variance (segment variances are computed
// from the standardized values, so the comparison is in standardized units and does not depend on
// the unit of the data): a segment with a zero variance would otherwise cost minus infinity. The
// changepoint package substitutes 1e-11 in data units when a variance is not positive.
const VARIANCE_FLOOR: f32 = 1e-11;
${getChangePointSeriesReaderWGSL(layout)}
// Compensated sum of the whole series: (sum, accumulated rounding error). The error comes from the
// integer significands: the float form (sum - next) + value is algebraically zero and is folded away.
fn addCompensated(accumulator: vec2<f32>, value: f32) -> vec2<f32> {
  let sum = accumulator.x + value;
  return vec2<f32>(sum, accumulator.y + getAdditionRoundingError(accumulator.x, value, sum));
}`;
}

/**
 * WGSL statements computing `segmentCost` for the segment `[start, end)` from its running sums:
 * `total` and `squares` of the values less the segment's first value (less nothing for
 * `'variance'` and `'poisson'`), and for `'slope'` `timeValue`, `timeFirst` and `timeSecond` over
 * the slice positions less the segment's first position.
 */
function getSegmentCostWGSL(cost: GPULocationChangePointCost): string {
  switch (cost) {
    case 'mean':
      return `segmentCost = max(squares - total * total / segmentLength, 0.0);`;
    case 'variance':
      return `let variance = max(squares / segmentLength, VARIANCE_FLOOR);
      segmentCost = segmentLength * log(variance);`;
    case 'mean-variance':
      return `let variance = max((squares - total * total / segmentLength) / segmentLength, VARIANCE_FLOOR);
      segmentCost = segmentLength * log(variance);`;
    case 'poisson':
      return `segmentCost = -2.0 * getObservedTerm(max(total, 0.0), segmentLength * overallRate);`;
    case 'slope':
      return `let centeredSquares = squares - total * total / segmentLength;
      // Exact in u32: positions are below ${GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT}, so n * sum(t^2) stays below 2^32.
      let timeVariance = f32((end - start) * timeSecond - timeFirst * timeFirst) / segmentLength;
      let covariance = timeValue - f32(timeFirst) * total / segmentLength;
      var residual = centeredSquares;
      if (timeVariance > 0.0) {
        residual -= covariance * covariance / timeVariance;
      }
      segmentCost = max(residual, 0.0);`;
    default:
      return '';
  }
}

function getDynamicProgramBodyWGSL(
  cost: GPULocationChangePointCost,
  output: GPULocationChangePointsOutput
): string {
  const isPoisson = cost === 'poisson';
  const isSlope = cost === 'slope';
  const usesSquareScale = cost === 'mean' || isSlope;
  // 'variance' measures about the mean of the whole series (0 after standardizing) and 'poisson'
  // needs the raw total; the other costs are unchanged by subtracting the segment's first value.
  const usesSegmentOrigin = cost !== 'variance' && !isPoisson;
  const requiredWGSL = output.requiredChangePointCounts
    ? 'requiredChangePointCounts[requiredChangePointCountsOffset + cellIndex] = found;'
    : '';
  const overflowWGSL = output.status?.overflow
    ? `if (found > MAXIMUM_CHANGE_POINTS) {
    atomicOr(&overflowStatus[overflowStatusOffset], 1u);
  }`
    : '';
  const optimalWGSL = output.optimalCost
    ? `optimalCost[optimalCostOffset + cellIndex] = select(getNaN(), ${getOptimalCostExpression(cost)}, previousEnd[seriesLength] != 0xffffffffu);`
    : '';
  const invalidOptimalWGSL = output.optimalCost
    ? 'optimalCost[optimalCostOffset + cellIndex] = getNaN();'
    : '';
  return /* wgsl */ `if (localInvocationIndex != 0u) {
    return;
  }
  let cellIndex = index / 2u;
  var seriesLength = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = readSeriesValue(cellIndex, slice);
    if (isFiniteValue(value)) {
      position[seriesLength] = slice;
      seriesLength++;
    }
  }
  var sumAccumulator = vec2<f32>(0.0, 0.0);
  for (var row = 0u; row < seriesLength; row++) {
    sumAccumulator = addCompensated(sumAccumulator, readSeriesValue(cellIndex, position[row]));
  }
  let count = f32(seriesLength);
  let overallSum = sumAccumulator.x + sumAccumulator.y;
  let center = ${isPoisson ? '0.0' : 'select(0.0, overallSum / count, seriesLength > 0u)'};
  var squareAccumulator = vec2<f32>(0.0, 0.0);
  for (var row = 0u; row < seriesLength; row++) {
    let deviation = readSeriesValue(cellIndex, position[row]) - center;
    squareAccumulator = addCompensated(squareAccumulator, deviation * deviation);
  }
  let scale = select(0.0, (squareAccumulator.x + squareAccumulator.y) / count, seriesLength > 0u);
  let overallRate = select(0.0, overallSum / count, seriesLength > 0u);
  var found = 0u;
  let hasStructure = seriesLength >= 2u && ${isPoisson ? 'overallRate > 0.0' : 'scale > 0.0'};
  if (hasStructure) {
    let deviationScale = ${isPoisson ? '1.0' : 'sqrt(scale)'};
    for (var row = 0u; row < seriesLength; row++) {
      let observed = readSeriesValue(cellIndex, position[row]);
      series[row] = ${isPoisson ? 'observed' : '(observed - center) / deviationScale'};
    }
    let penalty = parameters[parametersOffset]${usesSquareScale ? ' / scale' : ''};
    let minimumLength = max(u32(parameters[parametersOffset + 1u]), 1u);
    bestCost[0] = -penalty;
    for (var end = 1u; end <= seriesLength; end++) {
      bestCost[end] = 0x1.fffffep+127f;
      previousEnd[end] = 0xffffffffu;
    }
    // Forward form of F(end) = min over start of F(start) + C(start, end) + penalty: starts are
    // visited in ascending order, so F(start) is final when its segments are relaxed, and the
    // strict comparison keeps the smallest start among equal candidates. Each segment's moments
    // are running sums that begin at its own start, about its own first value: no difference of
    // two long prefix sums is taken, so nothing has to survive a cancellation (a shader compiler
    // that reassociates float sums also reorders the difference of two compensated prefixes).
    for (var start = 0u; start < seriesLength; start++) {
      if (start > 0u && previousEnd[start] == 0xffffffffu) {
        continue;
      }
      let origin = ${usesSegmentOrigin ? 'series[start]' : '0.0'};
      var total = 0.0;
      var squares = 0.0;
      ${
        isSlope
          ? `let timeOrigin = position[start];
      var timeValue = 0.0;
      var timeFirst = 0u;
      var timeSecond = 0u;`
          : ''
      }
      for (var end = start + 1u; end <= seriesLength; end++) {
        let offsetValue = series[end - 1u] - origin;
        total += offsetValue;
        squares += offsetValue * offsetValue;
        ${
          isSlope
            ? `let relativeTime = position[end - 1u] - timeOrigin;
        timeValue += f32(relativeTime) * offsetValue;
        timeFirst += relativeTime;
        timeSecond += relativeTime * relativeTime;`
            : ''
        }
        if (end - start < minimumLength) {
          continue;
        }
        let segmentLength = f32(end - start);
        var segmentCost = 0.0;
        {
          ${getSegmentCostWGSL(cost)}
        }
        let candidate = bestCost[start] + segmentCost + penalty;
        if (candidate < bestCost[end]) {
          bestCost[end] = candidate;
          previousEnd[end] = start;
        }
      }
    }
    // Backtracking visits the partition from the end; the first walk counts the change points.
    var cursor = seriesLength;
    loop {
      if (cursor == 0u || previousEnd[cursor] == 0xffffffffu || previousEnd[cursor] == 0u) {
        break;
      }
      found++;
      cursor = previousEnd[cursor];
    }
    ${requiredWGSL}
    ${overflowWGSL}
    let kept = min(found, MAXIMUM_CHANGE_POINTS);
    changePointCount[changePointCountOffset + cellIndex] = kept;
    for (var slot = 0u; slot < MAXIMUM_CHANGE_POINTS; slot++) {
      changePoints[changePointsOffset + cellIndex * MAXIMUM_CHANGE_POINTS + slot] = NO_CHANGE_POINT;
    }
    // The second walk writes the earliest change points: the j-th from the end is slot found - 1 - j.
    cursor = seriesLength;
    for (var walked = 0u; walked < found; walked++) {
      let segmentStart = previousEnd[cursor];
      let slot = found - 1u - walked;
      if (slot < kept) {
        changePoints[changePointsOffset + cellIndex * MAXIMUM_CHANGE_POINTS + slot] = position[segmentStart];
      }
      cursor = segmentStart;
    }
    ${optimalWGSL}
  } else {
    changePointCount[changePointCountOffset + cellIndex] = 0u;
    for (var slot = 0u; slot < MAXIMUM_CHANGE_POINTS; slot++) {
      changePoints[changePointsOffset + cellIndex * MAXIMUM_CHANGE_POINTS + slot] = NO_CHANGE_POINT;
    }
    ${requiredWGSL}
    ${seriesLength0OptimalWGSL(output, cost)}
    ${seriesLength0InvalidWGSL(invalidOptimalWGSL)}
  }`;
}

function seriesLength0OptimalWGSL(
  output: GPULocationChangePointsOutput,
  cost: GPULocationChangePointCost
): string {
  // A series with no structure has one segment; its cost in the data scale.
  return output.optimalCost
    ? `if (seriesLength >= 1u) {
      optimalCost[optimalCostOffset + cellIndex] = ${getFlatCostExpression(cost)};
    }`
    : '';
}

function seriesLength0InvalidWGSL(invalidOptimalWGSL: string): string {
  return invalidOptimalWGSL
    ? `if (seriesLength == 0u) {
      ${invalidOptimalWGSL}
    }`
    : '';
}

function getFlatCostExpression(cost: GPULocationChangePointCost): string {
  switch (cost) {
    case 'mean':
    case 'slope':
      return '0.0';
    case 'poisson':
      return '0.0';
    default:
      // A constant series has zero variance, so the relative floor of the segment costs has nothing
      // to be relative to: the single segment takes the changepoint package's substitute variance
      // for a non-positive variance, 1e-11 in data units (the same number as VARIANCE_FLOOR).
      return 'count * (log(TAU) + log(VARIANCE_FLOOR) + 1.0)';
  }
}

function getOptimalCostExpression(cost: GPULocationChangePointCost): string {
  switch (cost) {
    case 'mean':
    case 'slope':
      return 'bestCost[seriesLength] * scale';
    case 'poisson':
      return 'bestCost[seriesLength] - 2.0 * overallSum * log(overallRate)';
    default:
      return 'bestCost[seriesLength] + count * (log(TAU) + log(scale) + 1.0)';
  }
}
