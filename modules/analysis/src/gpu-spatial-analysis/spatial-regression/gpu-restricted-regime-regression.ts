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
  getStorageBindingElementLimit,
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL,
  IS_NAN_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../analysis-core/kernels/incomplete-gamma-wgsl';
import {getCholeskyWGSL, getPivotLimitedCholeskyWGSL} from './spatial-regression-solve';
import {getRegimeMomentNodes, getRegimeMomentStride} from './regime-moments';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {
  getRegressionTileLayout,
  getRowTileSumNodes,
  REGRESSION_TILE_MAXIMUM_ROW_COUNT
} from './regression-row-tiles';

export type GPURestrictedRegimeRegressionProps = {
  id?: string;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  regimes: GraphDataView<'uint32'>;
  regimeCount: number;
  /** Intercept first. True shares a coefficient across regimes, false estimates one per regime. */
  common: readonly boolean[];
  mask?: GraphDataView<'uint32'>;
  output: {
    /** Source-column-major: one slot if common, otherwise regimeCount slots. */
    coefficients: GraphDataView<'float32'>;
    /** Full covariance of the restricted model, pooled homoskedastic SSE/(n-parameterCount). */
    covariance: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
    /** Joint Chow F statistic, numerator degrees, denominator degrees. NaN F if no restrictions. */
    chow: GraphDataView<'float32'>;
    /** Optional coefficient stability Wald tests [chi-square, degrees, upper-tail probability].
     * One row per source column (intercept first), then a joint row. Shared columns have df 0/NaN.
     * Tests equality of the varying coefficients under this model's pooled covariance, as spreg Chow.
     */
    stability?: GraphDataView<'float32'>;
    /** 0 fitted; 1 insufficient rows; 2 restricted singular; 3 unrestricted singular. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Restricted OLS with any common/varying intercept and slopes, plus the joint nested-model Chow
 * F statistic against fully varying regimes on exactly the same complete observations. The test
 * assumes independent homoskedastic normal errors; no spatial correction or p-value is implied.
 * Empty/singular regimes invalidate the unrestricted comparison explicitly. Total unrestricted
 * coefficients are bounded at 16. Design columns are norm-scaled, not implicitly ridge-regularized.
 *
 * Both nested models are solved in a shifted basis, never from raw cross-products, so a predictor
 * with a large mean (a year, an elevation, a projected northing) costs no precision:
 * - with a varying intercept (and always for the unrestricted model) every predictor and the
 *   response are centered on their regime means, the slopes come from the centered moments and
 *   each intercept is recovered as `mean(y) - mean(x)' slopes`;
 * - with a common intercept, the response and the predictors with a common slope are centered on
 *   their pooled means. A varying slope under a common intercept is not invariant to a shift,
 *   so it gets a change of basis instead: when the varying-slope predictor `x` with the largest
 *   `mean^2 / variance` has that ratio above 1, `x - mean` takes the place of the constant and
 *   the raw columns `x 1[g]` stay. The two bases span the same model
 *   (`1 = (sum_g x 1[g] - (x - mean)) / mean`), the squared pivot of the constant against the raw
 *   columns is about `variance / (variance + mean^2)` and that of `x - mean` about
 *   `mean^2 / (variance + mean^2)`, so the better conditioned one is always taken and the
 *   conditioning no longer depends on the size of the mean. Other varying-slope predictors enter
 *   as `x_c - (mean_c / mean) x`.
 * The published coefficients and covariance are those of the raw design (`T V T'` of the
 * solution in its basis).
 *
 * Two float32 details keep levels finer than a float32 step of a mean. The regime means are
 * float32, so the sums of the deviations from them (`regime-moments.ts`) are carried through
 * every moment, which makes the moments exact for the stored means. And each difference of a
 * value and its shift in the residual pass goes through `getRoundedDifference`, because the
 * shader compiler of the test GPU otherwise regroups `(y - mean) - fit` as `y - (mean + fit)`
 * and every residual comes out in float32 steps of the response (a Chow statistic was 12% off
 * for a response near 10,000 with residuals of 0.05). `fitted` is still evaluated from the raw
 * coefficients and carries the float32 rounding of the response magnitude. The residual sums of squares, the Chow numerator and the Wald stability tests are
 * evaluated in the shifted basis: the Chow numerator is the direct sum of squared differences of
 * the two fits (equal to `SSE_restricted - SSE_unrestricted` for nested least squares), and each
 * Wald contrast is formed from differences of shifts before it meets the coefficients.
 *
 * A system is singular (status 2 or 3) when a diagonal is not positive or a Cholesky pivot of the
 * unit-diagonal normal matrix is below `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (`1e-3`), the
 * limit of `GPUOrdinaryLeastSquares`.
 *
 * Rows are reduced in parallel and in a fixed order: per-regime means and centered moments come
 * from a stable regime sort and one workgroup per regime (see `GPUSpatialRegimeRegression`), one
 * invocation solves the small dense systems of both nested models from them, and the residual
 * sums of squares are row-tile sums with a fixed-order tile merge. Rows are limited by float32
 * counts (`2^24`) and by the predictor binding of the device.
 */
export class GPURestrictedRegimeRegression implements GPUCommandNodeProducer {
  readonly id: string;
  readonly coefficientCount: number;
  readonly coefficientOffsets: readonly number[];
  constructor(readonly props: GPURestrictedRegimeRegressionProps) {
    this.id = props.id ?? 'restricted-regime-regression';
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, 7);
    validateCapacity(this.id, 'regimeCount', props.regimeCount, 2, 8);
    validateCapacity(
      this.id,
      'unrestricted coefficients',
      (props.predictorCount + 1) * props.regimeCount,
      1,
      16
    );
    validateCapacity(this.id, 'rows', props.response.length, 1, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
    validateRequirement(
      this.id,
      props.common.length === props.predictorCount + 1 &&
        props.common.every(value => typeof value === 'boolean'),
      'common must hold one boolean for the intercept and one for each of predictorCount predictors'
    );
    let count = 0;
    this.coefficientOffsets = props.common.map(common => {
      const offset = count;
      count += common ? 1 : props.regimeCount;
      return offset;
    });
    this.coefficientCount = count;
    const rows = props.response.length;
    for (const [name, view, format, length] of [
      ['predictors', props.predictors, 'float32', rows * props.predictorCount],
      ['response', props.response, 'float32', rows],
      ['regimes', props.regimes, 'uint32', rows],
      ['mask', props.mask, 'uint32', rows],
      ['output.coefficients', props.output.coefficients, 'float32', count],
      ['output.covariance', props.output.covariance, 'float32', count * count],
      ['output.fitted', props.output.fitted, 'float32', rows],
      ['output.chow', props.output.chow, 'float32', 3],
      ['output.stability', props.output.stability, 'float32', (props.predictorCount + 2) * 3],
      ['output.status', props.output.status, 'uint32', 1]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        validateViewLength(this.id, name, view, length);
      }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.predictors,
      props.response,
      props.regimes,
      props.mask
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, coefficientCount: restricted} = this;
    const columns = props.predictorCount;
    const unrestricted = (columns + 1) * props.regimeCount;
    const rows = props.response.length;
    // The predictor view is the largest binding: rows * predictorCount float32 values.
    validateCapacity(
      id,
      'rows',
      rows,
      1,
      Math.min(
        REGRESSION_TILE_MAXIMUM_ROW_COUNT,
        getStorageBindingElementLimit(graph.device, 4 * columns)
      )
    );
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
const OFFSETS = array<u32, ${columns + 1}>(${this.coefficientOffsets.map(value => `${value}u`).join(', ')});
const COMMON = array<bool, ${columns + 1}>(${props.common.join(', ')});
fn included(row: u32) -> bool { if (regimes[regimesOffset + row] >= ${props.regimeCount}u || ${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteBits(response[responseOffset + row])) { return false; }
for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; } } return true; }
fn rawValue(row: u32, column: u32) -> f32 { if (column == 0u) { return 1.0; } return predictors[predictorsOffset + row * ${columns}u + column - 1u]; }
`;
    const stride = getRegimeMomentStride(columns);
    const regimeCount = props.regimeCount;
    // Words of one regime's level record: the shifted intercept, the response shift, then one
    // shift per predictor. A row residual is `(y - shift) - intercept - slopes'(x - shifts)`.
    const levelStride = columns + 2;
    // Workspace (float32): included row count, solve code, restricted estimate, its unscaled
    // inverse normal matrix (raw design), unrestricted estimate, the three row sums (restricted
    // and unrestricted residual squares, squared difference of the two fits), the level records
    // of both models, then the restricted solution in its shifted basis for the Wald tests: the
    // estimate, its unscaled inverse normal matrix, the map `T` to raw coefficients, the raw
    // origin `t0` (`raw = T shifted + t0`) and the design forms.
    const workspaceCount = 0;
    const workspaceStatus = 1;
    const workspaceRestricted = 2;
    const workspaceInverse = workspaceRestricted + restricted;
    const workspaceUnrestricted = workspaceInverse + restricted * restricted;
    const workspaceSquares = workspaceUnrestricted + unrestricted;
    const workspaceRestrictedLevels = workspaceSquares + 3;
    const workspaceUnrestrictedLevels = workspaceRestrictedLevels + regimeCount * levelStride;
    const workspaceShiftedEstimate = workspaceUnrestrictedLevels + regimeCount * levelStride;
    const workspaceShiftedInverse = workspaceShiftedEstimate + restricted;
    const workspaceTransform = workspaceShiftedInverse + restricted * restricted;
    const workspaceOrigin = workspaceTransform + restricted * restricted;
    // Design forms of the restricted model: the pivot column, then per source column the constant
    // term and the pivot term of its design value (see `fit` below).
    const workspaceForms = workspaceOrigin + restricted;
    const formLength = 1 + 2 * (columns + 1);
    // A float32 zero written at run time: its bits are the runtime zero of getRoundedDifference.
    const workspaceRuntimeZero = workspaceForms + formLength;
    const workspaceLength = workspaceRuntimeZero + 1;
    const workspace = createTransientView(graph, `${id}-workspace`, 'float32', workspaceLength);
    const tileLayout = getRegressionTileLayout(rows);
    const momentsWGSL = `const D: u32 = ${columns + 1}u;
const STRIDE: u32 = ${stride}u;
fn regimeRows(regime: u32) -> f32 { return f32(segmentOffsets[segmentOffsetsOffset + regime + 1u] - segmentOffsets[segmentOffsetsOffset + regime]); }
// Mean of a column (1..K predictors, K + 1 response) in one regime.
fn regimeMean(regime: u32, column: u32) -> f32 { return moments[momentsOffset + regime * STRIDE + column - 1u]; }
// Centered cross-product of two columns (1..K predictors, K + 1 response) in one regime.
fn centeredMoment(regime: u32, first: u32, second: u32) -> f32 {
  let low = min(first, second) - 1u;
  let high = max(first, second) - 1u;
  return moments[momentsOffset + regime * STRIDE + D + low * (2u * D - low + 1u) / 2u + (high - low)];
}
// Sum of the deviations of a column (1..K predictors, K + 1 response) from its float32 regime
// mean: what the float32 mean misses of the true mean, times the row count.
fn regimeDeviationSum(regime: u32, column: u32) -> f32 { return moments[momentsOffset + regime * STRIDE + D + D * (D + 1u) / 2u + column - 1u]; }
// Cross-product of two columns (0 the constant, 1..K predictors, K + 1 response) of one regime
// about the given shifts, exact for the stored float32 means: with offsets o = mean - shift,
// sum (a - sa)(b - sb) = C_ab + oa S_b + ob S_a + n oa ob, and sum (b - sb) = S_b + n ob.
fn momentAboutShifts(regime: u32, first: u32, second: u32, firstShift: f32, secondShift: f32) -> f32 {
  let count = regimeRows(regime);
  if (first == 0u && second == 0u) { return count; }
  if (first == 0u) { return regimeDeviationSum(regime, second) + count * (regimeMean(regime, second) - secondShift); }
  if (second == 0u) { return regimeDeviationSum(regime, first) + count * (regimeMean(regime, first) - firstShift); }
  let firstOffset = regimeMean(regime, first) - firstShift; let secondOffset = regimeMean(regime, second) - secondShift;
  return centeredMoment(regime, first, second) + firstOffset * regimeDeviationSum(regime, second) + secondOffset * regimeDeviationSum(regime, first) + count * firstOffset * secondOffset;
}
// Pooled mean of every column, the count-weighted mean of the regime means.
fn getPooledMeans() -> array<f32, ${columns + 2}> {
  var pooled: array<f32, ${columns + 2}>; var total = 0.0;
  for (var regime = 0u; regime < ${regimeCount}u; regime++) { let count = regimeRows(regime); total += count; for (var column = 1u; column < ${columns + 2}u; column++) { pooled[column] += count * regimeMean(regime, column); } }
  for (var column = 1u; column < ${columns + 2}u; column++) { pooled[column] /= max(total, 1.0); }
  return pooled;
}`;
    // Restricted model with a common intercept. Every column is taken about its pooled mean,
    // `d_c = x_c - mean_c`, and the design value of source column `c` is the form
    // `constant_c + d_c + pivot_c d_p` (`c = 0` has no `d_c`):
    // - common slope: `d_c`;
    // - no pivot: intercept `1`, varying slope `mean_c + d_c = x_c` (the raw column);
    // - pivot `p` (the varying-slope predictor with the largest `mean^2 / variance`, used when
    //   that ratio exceeds 1): intercept slot `d_p`, pivot column `mean_p + d_p = x_p`, another
    //   varying column `d_c - (mean_c / mean_p) d_p = x_c - (mean_c / mean_p) x_p`.
    // Both bases span the model `{1, x_c, x_c 1[g]}` (`1 = (sum_g x_p 1[g] - d_p) / mean_p`). The
    // squared Cholesky pivot of the constant against the raw columns is about
    // `variance / (variance + mean^2)`, that of `d_p` about `mean^2 / (variance + mean^2)`, so the
    // pivot basis is taken exactly when it is the better conditioned one. Its conditioning does
    // not depend on the size of the mean, which the raw basis loses as `(variance / mean^2)`.
    const getCommonInterceptFitWGSL = (name: string, size: number) => `
fn destination_${name}(column: u32, regime: u32) -> u32 { return OFFSETS[column] + select(regime, 0u, COMMON[column]); }
// Cross-product about the pooled means of the columns (0 the constant, 1..K predictors, K + 1
// response) in one regime: the centered moment plus the product of the mean offsets.
fn pooledMoment_${name}(regime: u32, first: u32, second: u32, pooled: ptr<function, array<f32, ${columns + 2}>>) -> f32 {
  return momentAboutShifts(regime, first, second, (*pooled)[first], (*pooled)[second]);
}
// Cross-product of the design values of two source columns (the response is column K + 1, whose
// design value is its pooled deviation) in one regime.
fn designMoment_${name}(regime: u32, first: u32, second: u32, pooled: ptr<function, array<f32, ${columns + 2}>>, forms: ptr<function, array<f32, ${formLength}>>) -> f32 {
  let pivotColumn = u32((*forms)[0]);
  var firstIndices = array<u32, 3>(0u, first, pivotColumn); var secondIndices = array<u32, 3>(0u, second, pivotColumn);
  var firstWeights = array<f32, 3>(0.0, select(1.0, 0.0, first == 0u), 0.0); var secondWeights = array<f32, 3>(0.0, select(1.0, 0.0, second == 0u), 0.0);
  if (first <= ${columns}u) { firstWeights[0] = (*forms)[1u + first]; firstWeights[2] = (*forms)[${columns + 2}u + first]; }
  if (second <= ${columns}u) { secondWeights[0] = (*forms)[1u + second]; secondWeights[2] = (*forms)[${columns + 2}u + second]; }
  var value = 0.0;
  for (var left = 0u; left < 3u; left++) { if (firstWeights[left] == 0.0) { continue; }
    for (var other = 0u; other < 3u; other++) { if (secondWeights[other] == 0.0) { continue; }
      value += firstWeights[left] * secondWeights[other] * pooledMoment_${name}(regime, firstIndices[left], secondIndices[other], pooled); } }
  return value;
}
fn fit_${name}(estimate: ptr<function, array<f32, ${size}>>, levels: ptr<function, array<f32, ${regimeCount * levelStride}>>, inverse: ptr<function, array<f32, ${size * size}>>, shiftedEstimate: ptr<function, array<f32, ${size}>>, shiftedInverse: ptr<function, array<f32, ${size * size}>>, transform: ptr<function, array<f32, ${size * size}>>, origin: ptr<function, array<f32, ${size}>>, forms: ptr<function, array<f32, ${formLength}>>) -> bool {
var pooled = getPooledMeans();
// Pivot: the varying-slope predictor whose pooled mean is largest against its pooled spread.
var pivot = 0u; var largestRatio = 1.0;
for (var column = 1u; column < ${columns + 1}u; column++) { if (COMMON[column]) { continue; }
  var squares = 0.0; var total = 0.0;
  for (var regime = 0u; regime < ${regimeCount}u; regime++) { squares += pooledMoment_${name}(regime, column, column, &pooled); total += regimeRows(regime); }
  if (!(squares > 0.0)) { continue; }
  let ratio = pooled[column] * pooled[column] * total / squares;
  if (ratio > largestRatio) { largestRatio = ratio; pivot = column; } }
for (var slot = 0u; slot < ${formLength}u; slot++) { (*forms)[slot] = 0.0; }
// With no pivot the pivot terms are all zero and column 1 only keeps the index in range.
(*forms)[0] = f32(max(pivot, 1u));
(*forms)[1] = select(1.0, 0.0, pivot != 0u); (*forms)[${columns + 2}u] = select(0.0, 1.0, pivot != 0u);
for (var column = 1u; column < ${columns + 1}u; column++) { if (COMMON[column]) { continue; }
  if (pivot == 0u || column == pivot) { (*forms)[1u + column] = pooled[column]; } else { (*forms)[${columns + 2}u + column] = -pooled[column] / pooled[pivot]; } }
var normal: array<f32, ${size * size}>; var right: array<f32, ${size}>;
for (var regime = 0u; regime < ${regimeCount}u; regime++) { for (var first = 0u; first < ${columns + 1}u; first++) { let row = destination_${name}(first, regime); right[row] += designMoment_${name}(regime, first, ${columns + 1}u, &pooled, forms);
 for (var second = 0u; second < ${columns + 1}u; second++) { normal[row * ${size}u + destination_${name}(second, regime)] += designMoment_${name}(regime, first, second, &pooled, forms); } } }
var scales: array<f32, ${size}>;
for (var column = 0u; column < ${size}u; column++) { let diagonal = normal[column * ${size + 1}u]; if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return false; } scales[column] = inverseSqrt(diagonal); right[column] *= scales[column]; }
for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) { normal[first * ${size}u + second] *= scales[first] * scales[second]; } }
if (!choleskyFactorLimited_${size}(&normal)) { return false; } choleskySolve_${size}(&normal, &right);
for (var column = 0u; column < ${size}u; column++) { (*shiftedEstimate)[column] = right[column] * scales[column]; (*origin)[column] = 0.0; var unit: array<f32, ${size}>; unit[column] = scales[column]; choleskySolve_${size}(&normal, &unit); for (var row = 0u; row < ${size}u; row++) { (*shiftedInverse)[row * ${size}u + column] = unit[row] * scales[row]; } }
// raw = T solution + t0. A slope is its own coefficient, and the pivot slope of a regime also
// collects the pivot terms of that regime's columns. The intercept collects the constant left by
// each form: 1 or -mean_p for the intercept slot, -mean_c for a common slope, and exactly 0 for a
// varying slope (mean_c - mean_c, or -mean_c + (mean_c / mean_p) mean_p).
for (var slot = 0u; slot < ${size * size}u; slot++) { (*transform)[slot] = 0.0; }
for (var column = 0u; column < ${size}u; column++) { (*transform)[column * ${size + 1}u] = 1.0; }
let interceptSlot = OFFSETS[0]; (*origin)[interceptSlot] = pooled[${columns + 1}u];
(*transform)[interceptSlot * ${size + 1}u] = select(1.0, -pooled[max(pivot, 1u)], pivot != 0u);
for (var column = 1u; column < ${columns + 1}u; column++) { if (COMMON[column]) { (*transform)[interceptSlot * ${size}u + OFFSETS[column]] = -pooled[column]; } }
if (pivot != 0u) { for (var regime = 0u; regime < ${regimeCount}u; regime++) { let pivotSlot = destination_${name}(pivot, regime);
  for (var column = 0u; column < ${columns + 1}u; column++) { if (column == pivot) { continue; } (*transform)[pivotSlot * ${size}u + destination_${name}(column, regime)] += (*forms)[${columns + 2}u + column]; } } }
for (var row = 0u; row < ${size}u; row++) { var value = (*origin)[row]; for (var column = 0u; column < ${size}u; column++) { value += (*transform)[row * ${size}u + column] * (*shiftedEstimate)[column]; } (*estimate)[row] = value; }
// Level records: every column of this model is taken about its pooled mean.
for (var regime = 0u; regime < ${regimeCount}u; regime++) { let level = regime * ${levelStride}u; (*levels)[level] = (*shiftedEstimate)[interceptSlot]; for (var column = 1u; column < ${columns + 2}u; column++) { (*levels)[level + select(1u + column, 1u, column == ${columns + 1}u)] = pooled[column]; } }
// Inverse normal matrix of the raw design: T V T'.
var product: array<f32, ${size * size}>;
for (var row = 0u; row < ${size}u; row++) { for (var column = 0u; column < ${size}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${size}u; inner++) { value += (*transform)[row * ${size}u + inner] * (*shiftedInverse)[inner * ${size}u + column]; } product[row * ${size}u + column] = value; } }
for (var row = 0u; row < ${size}u; row++) { for (var column = 0u; column < ${size}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${size}u; inner++) { value += product[row * ${size}u + inner] * (*transform)[column * ${size}u + inner]; } (*inverse)[row * ${size}u + column] = value; } }
return true;
}`;
    const fit = (name: string, size: number, constrained: boolean) => {
      // A varying intercept absorbs any per-regime shift, so the regime means are used. A common
      // intercept has its own basis (getCommonInterceptFitWGSL).
      const interceptVaries = !constrained || !props.common[0];
      if (!interceptVaries) {
        return getCommonInterceptFitWGSL(name, size);
      }
      return `
fn destination_${name}(column: u32, regime: u32) -> u32 { return ${constrained ? 'OFFSETS[column] + select(regime, 0u, COMMON[column])' : `column * ${regimeCount}u + regime`}; }
// Shift of a column (1..K predictors, K + 1 response) in one regime.
fn shift_${name}(regime: u32, column: u32, pooled: ptr<function, array<f32, ${columns + 2}>>) -> f32 {
  return regimeMean(regime, column);
}
// Cross-product about the shifts of design columns (0 intercept, 1..K predictors, K + 1 response)
// in one regime (momentAboutShifts with the shifts of this model).
fn shiftedMoment_${name}(regime: u32, first: u32, second: u32, pooled: ptr<function, array<f32, ${columns + 2}>>) -> f32 {
  var firstShift = 0.0; var secondShift = 0.0;
  if (first != 0u) { firstShift = shift_${name}(regime, first, pooled); }
  if (second != 0u) { secondShift = shift_${name}(regime, second, pooled); }
  return momentAboutShifts(regime, first, second, firstShift, secondShift);
}
fn fit_${name}(estimate: ptr<function, array<f32, ${size}>>, levels: ptr<function, array<f32, ${regimeCount * levelStride}>>${constrained ? `, inverse: ptr<function, array<f32, ${size * size}>>, shiftedEstimate: ptr<function, array<f32, ${size}>>, shiftedInverse: ptr<function, array<f32, ${size * size}>>, transform: ptr<function, array<f32, ${size * size}>>, origin: ptr<function, array<f32, ${size}>>, forms: ptr<function, array<f32, ${formLength}>>` : ''}) -> bool {
var pooled = getPooledMeans();
${
  constrained
    ? `// Design forms of the shifted basis: the intercept column is the constant 1, every predictor column its shifted value.
for (var slot = 0u; slot < ${formLength}u; slot++) { (*forms)[slot] = 0.0; } (*forms)[0] = 1.0; (*forms)[1] = 1.0;`
    : ''
}
var normal: array<f32, ${size * size}>; var right: array<f32, ${size}>;
for (var regime = 0u; regime < ${regimeCount}u; regime++) { for (var first = 0u; first < ${columns + 1}u; first++) { let row = destination_${name}(first, regime); right[row] += shiftedMoment_${name}(regime, first, ${columns + 1}u, &pooled);
 for (var second = 0u; second < ${columns + 1}u; second++) { normal[row * ${size}u + destination_${name}(second, regime)] += shiftedMoment_${name}(regime, first, second, &pooled); } } }
var scales: array<f32, ${size}>;
for (var column = 0u; column < ${size}u; column++) { let diagonal = normal[column * ${size + 1}u]; if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return false; } scales[column] = inverseSqrt(diagonal); right[column] *= scales[column]; }
for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) { normal[first * ${size}u + second] *= scales[first] * scales[second]; } }
if (!choleskyFactorLimited_${size}(&normal)) { return false; } choleskySolve_${size}(&normal, &right);
var solution: array<f32, ${size}>;
for (var column = 0u; column < ${size}u; column++) { solution[column] = right[column] * scales[column]; (*estimate)[column] = solution[column]; }
// Back to the raw design: slopes are unchanged, each intercept absorbs the shifts.
for (var regime = 0u; regime < ${regimeCount}u; regime++) {
  let interceptSlot = destination_${name}(0u, regime); let level = regime * ${levelStride}u;
  let responseShift = shift_${name}(regime, ${columns + 1}u, &pooled);
  (*levels)[level] = solution[interceptSlot]; (*levels)[level + 1u] = responseShift;
  var intercept = solution[interceptSlot] + responseShift;
  for (var column = 1u; column < ${columns + 1}u; column++) { let shift = shift_${name}(regime, column, &pooled); (*levels)[level + 1u + column] = shift; intercept -= shift * solution[destination_${name}(column, regime)]; }
  (*estimate)[interceptSlot] = intercept;
}
${
  constrained
    ? `for (var column = 0u; column < ${size}u; column++) { (*shiftedEstimate)[column] = solution[column]; (*origin)[column] = 0.0; var unit: array<f32, ${size}>; unit[column] = scales[column]; choleskySolve_${size}(&normal, &unit); for (var row = 0u; row < ${size}u; row++) { (*shiftedInverse)[row * ${size}u + column] = unit[row] * scales[row]; } }
// raw = T shifted + t0: T is the identity except that each intercept row carries minus the shifts.
for (var slot = 0u; slot < ${size * size}u; slot++) { (*transform)[slot] = 0.0; }
for (var column = 0u; column < ${size}u; column++) { (*transform)[column * ${size + 1}u] = 1.0; }
for (var regime = 0u; regime < ${regimeCount}u; regime++) { let interceptSlot = destination_${name}(0u, regime); (*origin)[interceptSlot] = shift_${name}(regime, ${columns + 1}u, &pooled);
  for (var column = 1u; column < ${columns + 1}u; column++) { (*transform)[interceptSlot * ${size}u + destination_${name}(column, regime)] = -shift_${name}(regime, column, &pooled); } }
// Inverse normal matrix of the raw design: T V T'.
var product: array<f32, ${size * size}>;
for (var row = 0u; row < ${size}u; row++) { for (var column = 0u; column < ${size}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${size}u; inner++) { value += (*transform)[row * ${size}u + inner] * (*shiftedInverse)[inner * ${size}u + column]; } product[row * ${size}u + column] = value; } }
for (var row = 0u; row < ${size}u; row++) { for (var column = 0u; column < ${size}u; column++) { var value = 0.0; for (var inner = 0u; inner < ${size}u; inner++) { value += product[row * ${size}u + inner] * (*transform)[column * ${size}u + inner]; } (*inverse)[row * ${size}u + column] = value; } }`
    : ''
}
return true;
}`;
    };
    const inputs = [
      {name: 'predictors', view: props.predictors, type: 'f32' as const, access: 'read' as const},
      {name: 'response', view: props.response, type: 'f32' as const, access: 'read' as const},
      {name: 'regimes', view: props.regimes, type: 'u32' as const, access: 'read' as const},
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
        : [])
    ];
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.regimes,
      props.mask,
      ...Object.values(props.output)
    ]);
    const moments = getRegimeMomentNodes<Parameters>(graph, {
      id,
      operation: 'GPURestrictedRegimeRegression',
      predictors: props.predictors,
      predictorCount: columns,
      response: props.response,
      regimes: props.regimes,
      regimeCount,
      mask: props.mask
    });
    const nodes = [
      ...moments.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'nested-models',
        // contract-allow: single-invocation -- <=16 coefficients, <=8 regimes; rows already reduced into moments.
        invocationCount: 1,
        bindings: [
          {name: 'segmentOffsets', view: moments.views.segmentOffsets, type: 'u32', access: 'read'},
          {name: 'moments', view: moments.views.moments, type: 'f32', access: 'read'},
          {name: 'workspace', view: workspace, type: 'f32', access: 'read_write'}
        ],
        // A small dense solve over (predictorCount + 1) * regimeCount <= 16 coefficients: the row
        // work is done by the regime moments above, so this one invocation never loops over rows.
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
const OFFSETS = array<u32, ${columns + 1}>(${this.coefficientOffsets.map(value => `${value}u`).join(', ')});
const COMMON = array<bool, ${columns + 1}>(${props.common.join(', ')});
${momentsWGSL}\n${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(restricted)}\n${getPivotLimitedCholeskyWGSL(restricted)}\n${unrestricted === restricted ? '' : `${getCholeskyWGSL(unrestricted)}\n${getPivotLimitedCholeskyWGSL(unrestricted)}`}\n${fit('restricted', restricted, true)}\n${fit('unrestricted', unrestricted, false)}`,
        body: `for (var slot = 0u; slot < ${workspaceLength}u; slot++) { workspace[workspaceOffset + slot] = getNaN(); }
workspace[workspaceOffset + ${workspaceRuntimeZero}u] = 0.0;
let count = segmentOffsets[segmentOffsetsOffset + ${regimeCount}u] - segmentOffsets[segmentOffsetsOffset];
workspace[workspaceOffset + ${workspaceCount}u] = f32(count);
workspace[workspaceOffset + ${workspaceStatus}u] = 1.0; if (count <= ${unrestricted}u) { return; }
var estimate: array<f32, ${restricted}>; var inverse: array<f32, ${restricted * restricted}>; var levels: array<f32, ${regimeCount * levelStride}>;
var shiftedEstimate: array<f32, ${restricted}>; var shiftedInverse: array<f32, ${restricted * restricted}>; var transform: array<f32, ${restricted * restricted}>; var origin: array<f32, ${restricted}>; var forms: array<f32, ${formLength}>;
workspace[workspaceOffset + ${workspaceStatus}u] = 2.0; if (!fit_restricted(&estimate, &levels, &inverse, &shiftedEstimate, &shiftedInverse, &transform, &origin, &forms)) { return; }
for (var slot = 0u; slot < ${formLength}u; slot++) { workspace[workspaceOffset + ${workspaceForms}u + slot] = forms[slot]; }
for (var slot = 0u; slot < ${restricted}u; slot++) { workspace[workspaceOffset + ${workspaceRestricted}u + slot] = estimate[slot]; workspace[workspaceOffset + ${workspaceShiftedEstimate}u + slot] = shiftedEstimate[slot]; workspace[workspaceOffset + ${workspaceOrigin}u + slot] = origin[slot]; }
for (var slot = 0u; slot < ${restricted * restricted}u; slot++) { workspace[workspaceOffset + ${workspaceInverse}u + slot] = inverse[slot]; workspace[workspaceOffset + ${workspaceShiftedInverse}u + slot] = shiftedInverse[slot]; workspace[workspaceOffset + ${workspaceTransform}u + slot] = transform[slot]; }
for (var slot = 0u; slot < ${regimeCount * levelStride}u; slot++) { workspace[workspaceOffset + ${workspaceRestrictedLevels}u + slot] = levels[slot]; }
var fullEstimate: array<f32, ${unrestricted}>; var fullLevels: array<f32, ${regimeCount * levelStride}>;
workspace[workspaceOffset + ${workspaceStatus}u] = 3.0; if (!fit_unrestricted(&fullEstimate, &fullLevels)) { return; }
for (var slot = 0u; slot < ${unrestricted}u; slot++) { workspace[workspaceOffset + ${workspaceUnrestricted}u + slot] = fullEstimate[slot]; }
for (var slot = 0u; slot < ${regimeCount * levelStride}u; slot++) { workspace[workspaceOffset + ${workspaceUnrestrictedLevels}u + slot] = fullLevels[slot]; }
workspace[workspaceOffset + ${workspaceStatus}u] = 0.0;`
      }),
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-residual-squares`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'residual-squares',
        rowCount: rows,
        layout: tileLayout,
        sumCount: 3,
        bindings: [...inputs, {name: 'workspace', view: workspace, type: 'f32', access: 'read'}],
        // Residuals in the shifted basis of each model: the large means are removed from the
        // response and the predictors before they meet the coefficients.
        declarations: `${common}
${ROUNDED_DIFFERENCE_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  let regime = regimes[regimesOffset + row];
  let restrictedLevel = workspaceOffset + ${workspaceRestrictedLevels}u + regime * ${levelStride}u;
  let unrestrictedLevel = workspaceOffset + ${workspaceUnrestrictedLevels}u + regime * ${levelStride}u;
  let observed = response[responseOffset + row];
  // Restricted model: the response about its shift minus, per source column, the shifted-basis
  // coefficient times the design value constant + deviation + pivotTerm * pivotDeviation (the
  // intercept column has no deviation of its own). The unrestricted model has regime-mean shifts
  // and a shifted intercept that carries what the float32 regime means miss.
  // Every difference of a value and its shift goes through getRoundedDifference: the compiler of
  // the test GPU otherwise regroups the chain as value - (shift + small terms) and the residual
  // comes out in float32 steps of the value.
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + ${workspaceRuntimeZero}u]);
  let formBase = workspaceOffset + ${workspaceForms}u;
  let estimateBase = workspaceOffset + ${workspaceShiftedEstimate}u;
  let pivotColumn = u32(workspace[formBase]);
  let pivotDeviation = getRoundedDifference(rawValue(row, pivotColumn), workspace[restrictedLevel + 1u + pivotColumn], runtimeZero);
  var restrictedResidual = getRoundedDifference(observed, workspace[restrictedLevel + 1u], runtimeZero);
  restrictedResidual -= workspace[estimateBase + OFFSETS[0] + select(regime, 0u, COMMON[0])] * (workspace[formBase + 1u] + workspace[formBase + ${columns + 2}u] * pivotDeviation);
  var unrestrictedResidual = getRoundedDifference(observed, workspace[unrestrictedLevel + 1u], runtimeZero);
  unrestrictedResidual -= workspace[unrestrictedLevel];
  for (var column = 1u; column < ${columns + 1}u; column++) {
    let value = rawValue(row, column);
    let designValue = workspace[formBase + 1u + column] + getRoundedDifference(value, workspace[restrictedLevel + 1u + column], runtimeZero) + workspace[formBase + ${columns + 2}u + column] * pivotDeviation;
    restrictedResidual -= designValue * workspace[estimateBase + OFFSETS[column] + select(regime, 0u, COMMON[column])];
    unrestrictedResidual -= getRoundedDifference(value, workspace[unrestrictedLevel + 1u + column], runtimeZero) * workspace[workspaceOffset + ${workspaceUnrestricted}u + column * ${regimeCount}u + regime];
  }
  (*terms)[0] += restrictedResidual * restrictedResidual;
  (*terms)[1] += unrestrictedResidual * unrestrictedResidual;
  // SSE_restricted - SSE_unrestricted of nested least squares is the squared distance of the fits.
  let fitDifference = restrictedResidual - unrestrictedResidual;
  (*terms)[2] += fitDifference * fitDifference;
}`,
        partial: createTransientView(
          graph,
          `${id}-residual-partials`,
          'float32',
          tileLayout.tileCount * 3
        ),
        output: workspace,
        outputSlot: workspaceSquares
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'finish',
        // contract-allow: single-invocation -- <=16 coefficients and <=256 covariance entries; no row scan.
        invocationCount: 1,
        bindings: [
          {name: 'workspace', view: workspace, type: 'f32', access: 'read'},
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read_write'},
          {name: 'chow', view: props.output.chow, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `for (var slot = 0u; slot < ${restricted}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${restricted * restricted}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
for (var slot = 0u; slot < 3u; slot++) { chow[chowOffset + slot] = getNaN(); }
var code = u32(workspace[workspaceOffset + ${workspaceStatus}u]);
let squares = workspace[workspaceOffset + ${workspaceSquares}u]; let fullSquares = workspace[workspaceOffset + ${workspaceSquares + 1}u]; let fitSquares = workspace[workspaceOffset + ${workspaceSquares + 2}u];
if (code == 0u || code == 3u) { if (!isFiniteBits(squares)) { code = 2u; } }
status[statusOffset] = code; if (code == 1u || code == 2u) { return; }
let count = u32(workspace[workspaceOffset + ${workspaceCount}u]);
let variance = squares / f32(count - ${restricted}u);
for (var slot = 0u; slot < ${restricted}u; slot++) { coefficients[coefficientsOffset + slot] = workspace[workspaceOffset + ${workspaceRestricted}u + slot]; }
for (var slot = 0u; slot < ${restricted * restricted}u; slot++) { covariance[covarianceOffset + slot] = workspace[workspaceOffset + ${workspaceInverse}u + slot] * variance; }
if (code == 0u && !isFiniteBits(fullSquares)) { code = 3u; status[statusOffset] = 3u; }
if (code != 0u) { return; }
chow[chowOffset + 1u] = ${unrestricted - restricted}.0; chow[chowOffset + 2u] = f32(count - ${unrestricted}u);
${unrestricted > restricted ? `if (fullSquares > 0.0) { chow[chowOffset] = fitSquares / ${unrestricted - restricted}.0 / (fullSquares/f32(count-${unrestricted}u)); }` : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-predict`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'predict',
        invocationCount: rows,
        bindings: [
          ...inputs,
          {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: `var prediction = getNaN(); if (included(index)) { prediction = 0.0; let regime = regimes[regimesOffset+index]; for (var column=0u; column<${columns + 1}u; column++) { let coefficient = OFFSETS[column]+select(regime,0u,COMMON[column]); prediction += rawValue(index,column)*coefficients[coefficientsOffset+coefficient]; } } fitted[fittedOffset+index]=prediction;`
      })
    ];

    if (props.output.stability) {
      const varying = props.common.flatMap((shared, column) => (shared ? [] : [column]));
      const contrasts = Math.max(1, varying.length * (props.regimeCount - 1));
      const contrastColumns = varying.flatMap(column =>
        Array.from({length: props.regimeCount - 1}, () => column)
      );
      const contrastOffsets = varying.flatMap(column =>
        Array.from(
          {length: props.regimeCount - 1},
          (_, regime) => this.coefficientOffsets[column] + regime + 1
        )
      );
      const contrastBases = varying.flatMap(column =>
        Array.from({length: props.regimeCount - 1}, () => this.coefficientOffsets[column])
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-stability`,
          operation: 'GPURestrictedRegimeRegression',
          variant: 'wald-stability',
          invocationCount: columns + 2,
          bindings: [
            {name: 'workspace', view: workspace, type: 'f32', access: 'read'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
            {name: 'stability', view: props.output.stability, type: 'f32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(contrasts)}
const COLUMNS = array<u32, ${contrasts}>(${(contrastColumns.length ? contrastColumns : [0]).map(value => `${value}u`).join(',')});
const OFFSETS = array<u32, ${contrasts}>(${(contrastOffsets.length ? contrastOffsets : [0]).map(value => `${value}u`).join(',')});
const BASES = array<u32, ${contrasts}>(${(contrastBases.length ? contrastBases : [0]).map(value => `${value}u`).join(',')});`,
          body: `stability[stabilityOffset + index * 3u] = getNaN(); stability[stabilityOffset + index * 3u + 1u] = 0.0; stability[stabilityOffset + index * 3u + 2u] = getNaN();
if (status[statusOffset] != 0u) { return; }
var selected: array<u32, ${contrasts}>; var count = 0u;
for (var contrast = 0u; contrast < ${contrastColumns.length}u; contrast++) { if (index == ${columns + 1}u || COLUMNS[contrast] == index) { selected[count] = contrast; count++; } }
if (count == 0u) { return; }
var normal: array<f32, ${contrasts * contrasts}>; var right: array<f32, ${contrasts}>; var difference: array<f32, ${contrasts}>;
for (var first = 0u; first < ${contrasts}u; first++) { normal[first * ${contrasts + 1}u] = 1.0; }
// Each contrast of raw coefficients is formed in the shifted basis (raw = T shifted + t0): the
// rows of T and the origins are differenced first, so the large shifts of two regimes cancel
// before they meet a coefficient or the covariance.
let variance = workspace[workspaceOffset + ${workspaceSquares}u] / (workspace[workspaceOffset + ${workspaceCount}u] - ${restricted}.0);
var contrastRows: array<f32, ${contrasts * restricted}>; var projected: array<f32, ${contrasts * restricted}>;
for (var first = 0u; first < count; first++) { let positive = OFFSETS[selected[first]]; let negative = BASES[selected[first]];
  difference[first] = workspace[workspaceOffset + ${workspaceOrigin}u + positive] - workspace[workspaceOffset + ${workspaceOrigin}u + negative];
  for (var slot = 0u; slot < ${restricted}u; slot++) { contrastRows[first * ${restricted}u + slot] = workspace[workspaceOffset + ${workspaceTransform}u + positive * ${restricted}u + slot] - workspace[workspaceOffset + ${workspaceTransform}u + negative * ${restricted}u + slot]; } }
// Joint tests: an intercept contrast and a slope contrast of the same regime are almost parallel
// when the predictor has a large mean (the intercept row holds minus the shift of each regime).
// Adding the base regime's shift times the slope contrast to the intercept contrast spans the same
// hypothesis, leaves the statistic unchanged and keeps only the difference of the two shifts.
for (var first = 0u; first < count; first++) { if (COLUMNS[selected[first]] != 0u) { continue; } let regime = OFFSETS[selected[first]] - BASES[selected[first]];
  for (var second = 0u; second < count; second++) { let other = selected[second]; if (COLUMNS[other] == 0u || OFFSETS[other] - BASES[other] != regime) { continue; }
    let weight = contrastRows[first * ${restricted}u + BASES[other]];
    for (var slot = 0u; slot < ${restricted}u; slot++) { contrastRows[first * ${restricted}u + slot] += weight * contrastRows[second * ${restricted}u + slot]; } } }
for (var first = 0u; first < count; first++) { var value = difference[first];
  for (var slot = 0u; slot < ${restricted}u; slot++) { value += contrastRows[first * ${restricted}u + slot] * workspace[workspaceOffset + ${workspaceShiftedEstimate}u + slot]; }
  difference[first] = value; right[first] = value; }
for (var first = 0u; first < count; first++) { for (var slot = 0u; slot < ${restricted}u; slot++) { var value = 0.0; for (var inner = 0u; inner < ${restricted}u; inner++) { value += contrastRows[first * ${restricted}u + inner] * workspace[workspaceOffset + ${workspaceShiftedInverse}u + inner * ${restricted}u + slot]; } projected[first * ${restricted}u + slot] = value; } }
for (var first = 0u; first < count; first++) { for (var second = 0u; second < count; second++) { var value = 0.0; for (var slot = 0u; slot < ${restricted}u; slot++) { value += projected[first * ${restricted}u + slot] * contrastRows[second * ${restricted}u + slot]; } normal[first * ${contrasts}u + second] = value * variance; } }
if (!choleskyFactor_${contrasts}(&normal)) { return; } choleskySolve_${contrasts}(&normal, &right);
var statistic = 0.0; for (var contrast = 0u; contrast < count; contrast++) { statistic += difference[contrast] * right[contrast]; }
statistic = max(0.0, statistic); stability[stabilityOffset + index * 3u] = statistic; stability[stabilityOffset + index * 3u + 1u] = f32(count); stability[stabilityOffset + index * 3u + 2u] = getChiSquareSurvivalValue(statistic, f32(count));`
        })
      );
    }
    return nodes;
  }
}
