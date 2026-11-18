// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {snapshotRegressionConstructionProps} from './regression-construction-snapshot';
import {getRegressionWeightsSuppressionNodes} from './regression-weights-completeness';
import {getRegressionTraceLayout} from './regression-trace-layout';

import {
  validateRegressionAllocations,
  getSpatialProbeAllocationLengths
} from './spatial-likelihood-allocations';
import {getOrdinaryLeastSquaresAllocationLengths} from './gpu-ordinary-least-squares';
import {
  getRegressionWeightBindings,
  getRegressionMaximumEdgeCount,
  getRegressionTraceAllocationLengths
} from './regression-binding-plans';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/index';
import {
  getGPUSpatialWeightsTilePlan,
  rewriteWeightsChunkTileBodyWGSL,
  wrapWeightsChunkRowBodyWGSL
} from './chunked-weights-kernels';
import type {GPUSolverStatusPort} from '../../analysis-core/contracts/index';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../../analysis-core/contracts/model-result';
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import {
  GPUSpatialWeightsTraces,
  validateGPUSpatialWeightsTracesParameters
} from './gpu-spatial-weights-traces';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from './ordinary-least-squares-parameters';
import {
  createKernelFactory,
  createSeriesProbeNodes,
  getF32Binding,
  getU32Binding,
  SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE
} from './spatial-maximum-likelihood-nodes';
import {
  GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_MINIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PRECISION_WARNING,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE,
  getSpatialMaximumLikelihoodSeriesLimit
} from './spatial-maximum-likelihood-parameters';
import {
  getParameterSearchWGSL,
  getTableRowWGSL,
  getTraceSeriesWGSL
} from './spatial-maximum-likelihood-wgsl';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {SPATIAL_SERIES_TAIL_MINIMUM_POWER, getSpatialSeriesTailWGSL} from './spatial-series-tail';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  getSpatialLogDeterminantWGSL,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';

import {
  getExactLogDeterminantWGSL,
  validateExactLogDeterminant,
  type GPUSpatialLogDeterminantMode
} from './spatial-exact-log-determinant';

const OPERATION = 'GPUSpatialLagMaximumLikelihood';
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
const TILE_WORKGROUP_SIZE = SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE;
const LN_TWO_PI = Math.log(2 * Math.PI);
/**
 * The first-order condition of an interior estimate, checked from the published residuals: with
 * `t = u'e1 / u'u`, `g = n t - tr(W A)` the derivative of the concentrated log-likelihood at
 * the published `rho` and `c = n (e1'e1 / u'u - 2 t^2) + tr((W A)^2)` its curvature,
 * `|g| <= sqrt(tolerance * c) + c d`, that is, `rho` is within half a standard error (of the
 * concentrated likelihood) plus `d` of a stationary point, with `d` the float32 allowance of
 * `STATIONARITY_PARAMETER_STEPS`. `u'u` and `u'e1` are summed per row from the published
 * residuals and the second OLS residual, not taken from the sums the search used, so a search
 * that ended away from a stationary point (a pole of a badly conditioned derivative, a wrong
 * direct sum) is status 3. It shares the trace series with the search and cannot tell a local
 * maximum from the global one. On the scenes of `math-review-lag-conditioning.spec.ts` the
 * float32 step of `rho` leaves `g^2 / c` at 0.015 or less.
 */
const STATIONARITY_TOLERANCE = 0.25;
/**
 * Float32 steps of `rho` the first-order condition allows on top of half a standard error. A
 * float32 `rho` cannot be nearer to the stationary point than half a step, and the sums it is
 * tested with carry the same rounding: the bisection of the search ends within one step of the
 * root of its own derivative; each product `rho0 e1` of the direct sums and each product
 * `rho e1` of the published residuals is rounded by at most `2^-24` of itself, which moves
 * `u'e1` by no more than a shift of `rho` by one step would (`sum |rho e1| |e1| 2^-24 <=
 * e1'e1 step`). Three steps in the worst case, four allowed. Where the standard error of `rho`
 * is many steps the allowance is nothing; where it is below one step (a response that is its
 * own lag times `rho` plus `X b` to float32 rounding) the nearest float32 `rho` failed the bare
 * test (`g^2 / c = 0.36` in float64 on the 64-row scene of
 * `math-review-lag-conditioning.spec.ts`).
 */
const STATIONARITY_PARAMETER_STEPS = 4;

/** Fixed pairwise sum of `count` items of `source` at `base + item * stride` (see `getPairwiseSumWGSL`). */
function getMergeSumWGSL(
  source: string,
  base: string,
  stride: string,
  count: string,
  name = 'sum'
): string {
  return getPairwiseSumWGSL({
    count,
    name,
    term: `${source}[${source}Offset + ${base} + ${name}Item * ${stride}]`
  });
}

/** Caller-owned outputs of {@link GPUSpatialLagMaximumLikelihood}. */
export type GPUSpatialLagMaximumLikelihoodOutput = {
  /**
   * One-row solver `code` (0 on success) and, optionally, `converged` (1 when the optimum is an
   * interior stationary point).
   */
  status: Pick<GPUSolverStatusPort, 'code'> & Partial<Pick<GPUSolverStatusPort, 'converged'>>;

  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then `rho` last), each row holding the estimate, standard error, z
   * statistic and two-sided normal p-value. The layout is the one `GPUSpatialLagImpacts` reads
   * with `coefficientStride: 4` and `rhoIndex: predictorCount + 1`.
   */
  table: GraphDataView<'float32'>;
  /** At least 22 float32 values; see the `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /**
   * Optional per-row residuals `u = y - rho W y - X b` (spreg `u`); NaN for a failed fit, except
   * a fit rejected by the first-order condition (status 3 with a finite search result), which
   * leaves the residuals of the rejected `rho`.
   */
  residuals?: GraphDataView<'float32'>;
  /**
   * Optional `maximumPower` float32 values `tr((W/s)^q)` the fit used, so that a downstream
   * `GPUSpatialLagImpacts` can share them. Provide `powerTraces`, `powerSums` and `infinityNorm`
   * together or none.
   */
  powerTraces?: GraphDataView<'float32'>;
  /** Optional `maximumPower` float32 values `1' (W/s)^q 1`, see `powerTraces`. */
  powerSums?: GraphDataView<'float32'>;
  /** Optional one float32 value `s = ||W||_inf` of the traces; provided with `powerTraces`. */
  infinityNorm?: GraphDataView<'float32'>;
  /**
   * Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`
   * across the Hutchinson probes (0 for exact traces); see
   * `GPUSpatialWeightsTracesOutput.powerTraceStandardErrors` and
   * `getSpatialLogDeterminantErrorBound`, which turns them into a bound on the log-determinant
   * (hence log-likelihood) sampling error at the estimate.
   */
  powerTraceStandardErrors?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialLagMaximumLikelihood}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, `maximumPower`, the trace options,
 * `tileRowCount` and which optional outputs are present.
 */
export type GPUSpatialLagMaximumLikelihoodProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-maximum-likelihood'`. */
  id?: string;
  /**
   * Square weights `W`. Any sparsity pattern works (no symmetry is required) and rows need not be
   * standardized: `rho` is searched within `min(0.99, seriesLimit) / ||W||_inf`.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 14. The intercept is implicit. */
  predictorCount: number;
  /**
   * Series length `Q` of the log-determinant, `tr1`, `tr2` and the probe pass, an integer in
   * `[2, 256]`. Defaults to 128. See the class documentation for the truncation bound.
   */
  maximumPower?: number;
  /**
   * `'exact'` (unit probes, no sampling error) or `'stochastic'` (Hutchinson). Defaults to exact
   * up to 4096 rows and stochastic above, where the likelihood, hence `rho`, carries stochastic
   * error and `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.
   */
  traceMode?: GPUSpatialWeightsTracesMode;
  /**
   * Extrapolates the omitted log-determinant and trace series tail with a geometric model of the
   * stored traces (`spatial-series-tail.ts`). The search interval then reaches the full
   * `0.99 / ||W||_inf` instead of stopping where the plain series loses accuracy (about 0.93 of
   * the bound for `Q = 128`), so strongly autocorrelated data can yield an interior estimate.
   * It is an approximation that assumes the traces have converged by `Q` (true for a
   * row-standardized `W`, whose traces tend to the number of components); with stochastic traces
   * the tail inherits the probe noise. Defaults to false (the plain truncated series), which is a
   * maintainer decision to revisit once enough measurements exist.
   */
  tailExtrapolation?: boolean;
  /**
   * `'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.
   * `'exact'` uses the eigenvalues of `W` instead (`eigenvalues`), which removes the truncation
   * error and the series limit on the search interval. It is limited to
   * `GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come
   * from a CPU dense decomposition, and needs a `W` that is similar to a symmetric matrix (for
   * example row-standardized symmetric weights). The third trace of the variance still comes from
   * the probe pass.
   */
  logDeterminant?: GPUSpatialLogDeterminantMode;
  /**
   * The `rowCount` eigenvalues of `W` as float32, from `getSpatialWeightsEigenvalues`. Required
   * with `logDeterminant: 'exact'`, rejected otherwise.
   */
  eigenvalues?: GraphDataView<'float32'>;
  /** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */
  traceProbeCount?: number;
  /**
   * Optional per-frame uint32 view of `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` words written
   * with `getGPUSpatialWeightsTracesParameterValues({seed})`: the Philox seed of the stochastic
   * probes, read on the GPU. Omitted means seed 0.
   */
  traceParameters?: GraphDataView<'uint32'>;
  /**
   * Rows reduced by one invocation of the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialLagMaximumLikelihoodOutput;
};

/**
 * Maximum-likelihood estimator of the spatial lag model `y = rho W y + X b + e`, `e ~ N(0, s2 I)`
 * (Anselin 1988; spreg `ML_Lag` with `method='full'`, whose exact log-determinant is replaced
 * here by a truncated trace series). Nothing is read back.
 *
 * 1. `e0` and `e1` are the OLS residuals of `y` and of the lag column on `[1, X]` (two
 *    `GPUOrdinaryLeastSquares` compositions), so `u(rho) = e0 - rho e1` and
 *    `s2(rho) = u'u / n`. The lag column is `v = W (y - ybar) + ybar (s_i - s)` from an inline
 *    sparse product, with `s_i` the exact sum of the float32 weights of row `i`
 *    (`getAdditionRoundingError`) and `s` the largest row sum: `W y = v + ybar s`, and the
 *    constant `ybar s` only moves the intercept, where it is added back. A sum of `w * y` over
 *    the raw response is rounded to a float32 step of the response, which for residuals of
 *    0.002 around 10,000 was up to 4% of `s2`; the lag about the mean is not.
 *    The search does not form `u'u` as `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1`: where the model
 *    fits well (`u'u` below 1e-7 of `e0'e0`) that difference of three float32 sums is rounding,
 *    and the search then ran to a pole of the derivative or found no candidate. It uses
 *    `u'u = S + e1'e1 (rho - rho*)^2`, with `rho0 = e0'e1 / e1'e1`, `r = e0 - rho0 e1` summed
 *    per row, `rho* = rho0 + r'e1 / e1'e1` and `S = r'r - (r'e1)^2 / e1'e1`: two non-negative
 *    terms.
 * 2. The concentrated log-likelihood `-n/2 ln(s2(rho)) + ln|I - rho W|` has the log-determinant
 *    `-sum_{q=1}^{Q} rho^q tr(W^q) / q` from `GPUSpatialWeightsTraces` (exact traces up to 4096
 *    rows, Hutchinson estimates above), evaluated by Horner's rule.
 * 3. `rho` is the global maximum of the likelihood on `[-B, B]`. A single kernel evaluates the
 *    analytic derivative on 1025 grid points, refines every positive-to-non-positive sign change
 *    by 32 bisection steps on the derivative and keeps the best candidate (end points count when
 *    the derivative points outward). It is deterministic and needs no iteration budget.
 * 4. `b = b0 - rho b1`, `s2 = u'u / n`, the full log-likelihood
 *    `-n/2 ln(s2) + ln|I - rho W| - n/2 (ln 2 pi + 1)`, spreg `pr2` (squared correlation of `y`
 *    and `y - u`) and `pr2_e` (of `y` and `(I - rho W)^-1 X b`).
 * 5. The variance is the inverse of spreg's information matrix of `(b, rho, s2)`, with
 *    `tr1 = tr(W A)`, `tr2 = tr((W A)^2)` as series in `rho` of the power traces, and
 *    `tr3 = tr((W A)' W A)` from a second probe pass at the estimated `rho`
 *    (`createSeriesProbeNodes`), with `A = (I - rho W)^-1`. The `s2` row is eliminated by its Schur
 *    complement (the `rho` entry becomes `tr2 + tr3 + (W X b~)'(W X b~) / s2 - 2 tr1^2 / n`), the
 *    matrix is built on centered predictors and on `W X b~` about its mean (a response with a
 *    large mean otherwise cancels the `rho` entry against the intercept to nothing in float32)
 *    and unit-scaled before the Cholesky factorization, and the intercept variance is mapped back
 *    to the uncentered model.
 *
 * Search interval. `B = min(0.99, seriesLimit) / ||W||_inf`, with `||W||_inf` the largest absolute
 * row sum computed on the GPU, which keeps `I - rho W` invertible for any `W` (spreg's fixed
 * `(-1, 1)` admits explosive `rho` for raw weights) and equals 0.99 for a row-standardized `W`
 * (spreg searches up to 1). The truncated series omits a tail below
 * `n s^(Q+1) / ((Q+1) (1 - s))` with `s = |rho| ||W||_inf` (the worst case, `|tr(W^q)| <= n s^q`),
 * and the omission biases the likelihood toward large positive `rho`, so the interval also stops
 * where this tail exceeds `1e-5 n` (`getSpatialMaximumLikelihoodSeriesLimit`): about `|rho| < 0.93`
 * for the default `Q = 128` and 0.96 for 256. Beyond it the fit reports a boundary solution
 * (`..._SUMMARY_AT_BOUNDARY`, `..._SERIES_LIMITED`). The default `Q = 128` keeps the worst-case
 * truncation below `1e-7 n` log-likelihood units at `|rho| <= 0.9` (below `1e-6 |LL|`), at the
 * price of about `Q * (1 + probe batches)` dispatches; raise it when `rho` is near the bound.
 * `..._SUMMARY_TRUNCATION_BOUND` reports the worst-case remainder at the estimate. The
 * hand-written spreg search uses an `xatol` of `1e-7`, so `rho` agrees with it to about that.
 *
 * The float32 traces `tr(W^q)` grow like `n ||W||_inf^q`: for a raw (unstandardized) `W` choose
 * `maximumPower` with `||W||_inf^maximumPower < 1e36`, else they overflow and the status is 3.
 * What float32 leaves (measured on the test GPU against float64, residuals of 0.002 to 0.004
 * around a response mean of 10,000; `math-review-lag-conditioning.spec.ts`). Weights whose rows
 * share a row sum, exactly or up to the rounding of the weights (row-standardized or not, with
 * or without rows that have no neighbours): `s2` within 1.1e-5 for `rho` from 0.2 to 0.8, the
 * log-likelihood within 0.004. With three rows without neighbours and `u'u` at 2e-10 of
 * `e0'e0`: `s2` within 4e-5, `rho` within 0.12 standard errors (a float32 step of `rho`).
 * Slopes two hundred times the residual deviation over the predictor range (`u'u` at 3e-9 of
 * `e0'e0`): `s2` within 3.4e-4, because the fitted values are rounded to 1% to 3% of a residual.
 * Row sums that differ by more than rounding with such a response (row sums 2 and 3): the lag
 * differs between rows by the response mean, the lag column is rounded to a float32 step of
 * that, and `s2` was within 2.5e-4.
 * Degenerate sums of squares. `S = 0` (the response residual is a multiple of the lag
 * residual) has a likelihood wherever `u'u = e1'e1 (rho - rho*)^2` is positive: when `rho*`
 * lies outside the search interval the search runs (spreg `ML_Lag` returns a fit there too);
 * when it lies inside, the likelihood has a pole and the status is 3. `e1'e1 = 0` (the lag
 * column lies in the span of `[1, X]`) makes `u'u = e0'e0` for every `rho`: the likelihood is
 * the log-determinant plus a constant and the estimate is its maximum (`rho = 0` for weights
 * with a zero diagonal).
 * Status 3 also covers an interior `rho` that fails the first-order condition. `u'u` and `u'e1`
 * are summed per row from the published residuals (not taken from the sums of the search), and
 * the derivative `g = n t - tr(W A)`, `t = u'e1 / u'u`, of the concentrated log-likelihood is
 * held against its curvature `c = n (e1'e1 / u'u - 2 t^2) + tr((W A)^2)`:
 * `|g| <= sqrt(0.25 c) + 4 c step`, that is, `rho` within half a standard error (of the
 * concentrated likelihood) plus four float32 steps of `rho` of a stationary point. Only ratios
 * of the sums enter, so the test is the same for a response multiplied by any factor that
 * keeps the sums in float32 range (measured identical status and `rho` within 6e-8 for
 * factors of 1e-10, 1 and 1e10; the products of two sums of the earlier form overflowed at
 * 1e10 and rejected the fit). The allowance of four steps is the float32 resolution of `rho`
 * and of the sums it is tested with (`STATIONARITY_PARAMETER_STEPS`): without it the float32
 * nearest to the maximum fails the test when the standard error of `rho` is below one step. A
 * search moved 1e-4 off on purpose is status 3 (pinned by
 * `math-review-lag-conditioning.spec.ts`, which offsets the stored `rho` between the search and
 * the rest of the graph). The check shares the trace series with the search, so it does not
 * detect a wrong log-determinant, and it cannot tell a local maximum from the global one. The
 * `residuals` output of a fit rejected by this test holds the residuals of the rejected `rho`,
 * not NaN (every other output is NaN).
 * Where the standard error of `rho` is below its float32 step (a response that is `rho` times
 * its lag plus `X b` to float32 rounding), `rho` is right to a step or two, but `s2`, the
 * log-likelihood and the standard errors are not those of the data: on the 64-row scene of the
 * same spec `s2` is 7.8e-13 against 2.4e-14 in float64 (33 times), the standard errors 5.7
 * times and the log-likelihood 111 low. The cause is the coefficients, not the residual
 * arithmetic. The residuals are `e0 - rho e1` of two float32 OLS fits, each with its own
 * rounded slope, so they are not orthogonal to `[1, X]`: 91% of the published residual sum of
 * squares lies in the span of `[1, x]` (the published slope is 1.00000012, one float32 step
 * above the float64 slope 1.0000000014, and the intercept -1.2e-7 against 0). The float32
 * floor of the residual arithmetic itself is far lower: `rho` rounded to float32 costs 1.006
 * times, `e0` and `e1` rounded 1.04, both with the product 1.65, and the published residuals
 * with their component in the span of `[1, x]` removed 2.9 (the decomposition of the re-review
 * of round three, in float64 from the GPU's outputs). A regression of the residuals on
 * `[1, X]` with the correction applied to `b` (as the two-stage kernel refines its estimate)
 * would remove it; the kernel does not do that. No output flags this.
 * Inputs must be finite: a non-finite predictor, response or weight gives status 3 (NaN outputs).
 * At most 14 predictors (the 16-wide Cholesky of the `(predictorCount + 2)`-wide information
 * matrix); no kernel binds more than 8 storage buffers. The `table` feeds `GPUSpatialLagImpacts`
 * directly.
 */
export class GPUSpatialLagMaximumLikelihood implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialLagMaximumLikelihoodProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Resolved series length `Q`. */
  readonly maximumPower: number;
  /** Whether the omitted series tail is extrapolated (see `props.tailExtrapolation`). */
  readonly tailExtrapolation: boolean;
  /** Whether the log-determinant comes from the eigenvalues (see `props.logDeterminant`). */
  readonly exactLogDeterminant: boolean;
  /** Resolved trace mode. */
  readonly traceMode: GPUSpatialWeightsTracesMode;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;

  private readonly traceTileRowCount: number;
  private readonly traceTileCount: number;

  constructor(props: GPUSpatialLagMaximumLikelihoodProps) {
    props = snapshotRegressionConstructionProps(
      props,
      [
        'id',
        'weights',
        'predictors',
        'response',
        'predictorCount',
        'maximumPower',
        'traceMode',
        'tailExtrapolation',
        'logDeterminant',
        'eigenvalues',
        'traceProbeCount',
        'traceParameters',
        'tileRowCount',
        'output'
      ],
      [
        'table',
        'summary',
        'residuals',
        'status',
        'powerTraces',
        'powerSums',
        'powerTraceStandardErrors',
        'infinityNorm'
      ],
      'spatial-lag-maximum-likelihood'
    );
    const id = props.id ?? 'spatial-lag-maximum-likelihood';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    validateCapacity(
      id,
      'predictorCount',
      predictorCount,
      1,
      GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT
    );
    const maximumPower = props.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER;
    validateCapacity(
      id,
      'maximumPower',
      maximumPower,
      GPU_SPATIAL_MAXIMUM_LIKELIHOOD_MINIMUM_POWER,
      256
    );
    this.maximumPower = maximumPower;
    this.tailExtrapolation = props.tailExtrapolation ?? false;
    if (this.tailExtrapolation) {
      validateCapacity(id, 'maximumPower', maximumPower, SPATIAL_SERIES_TAIL_MINIMUM_POWER, 256); // tail model needs three stored traces
    }
    const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
    this.exactLogDeterminant = validateExactLogDeterminant(id, props, rows);
    validateSinglePackedView(props.predictors, ['float32'], `${id} predictors`);
    validateSinglePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== rows) {
      throw new Error(`${id} response length must equal the weights row count`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal the weights row count * predictorCount`);
    }
    if (
      props.traceMode !== undefined &&
      props.traceMode !== 'exact' &&
      props.traceMode !== 'stochastic'
    ) {
      throw new Error(`${id} traceMode must be 'exact' or 'stochastic'`);
    }
    this.traceMode =
      props.traceMode ??
      (rows <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic');
    if (this.traceMode === 'exact' && rows > GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT) {
      throw new Error(`${id} exact traces support at most 16384 rows; use traceMode 'stochastic'`);
    }
    if (props.traceProbeCount !== undefined) {
      validateCapacity(id, 'traceProbeCount', props.traceProbeCount);
    }
    validateGPUSpatialWeightsTracesParameters(id, 'traceParameters', props.traceParameters);
    validateSinglePackedView(output.table, ['float32'], `${id} output.table`);
    if (output.table.length < (predictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE) {
      throw new Error(`${id} output.table must hold (predictorCount + 2) * 4 float32 values`);
    }
    validateSinglePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} float32 values`
      );
    }
    validateSinglePackedView(output.status.code, ['uint32'], `${id} output.status.code`);
    if (output.status.code.length < 1) {
      throw new Error(`${id} output.status.code must hold one uint32 row`);
    }
    if (output.status.converged) {
      validateSinglePackedView(
        output.status.converged,
        ['uint32'],
        `${id} output.status.converged`
      );
      if (output.status.converged.length < 1) {
        throw new Error(`${id} output.status.converged must hold one uint32 row`);
      }
    }
    if (output.residuals) {
      validateSinglePackedView(output.residuals, ['float32'], `${id} output.residuals`);
      if (output.residuals.length < rows) {
        throw new Error(`${id} output.residuals must hold one row per input row`);
      }
    }
    if (
      (output.powerTraces === undefined) !== (output.powerSums === undefined) ||
      (output.powerTraces === undefined) !== (output.infinityNorm === undefined)
    ) {
      throw new Error(
        `${id} output.powerTraces, output.powerSums and output.infinityNorm must be provided together`
      );
    }
    for (const [name, view] of [
      ['powerTraces', output.powerTraces],
      ['powerSums', output.powerSums],
      ['infinityNorm', output.infinityNorm],
      ['powerTraceStandardErrors', output.powerTraceStandardErrors]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length < (name === 'infinityNorm' ? 1 : maximumPower)) {
          throw new Error(`${id} output.${name} must hold maximumPower float32 values`);
        }
      }
    }
    if (props.tileRowCount !== undefined) {
      validateCapacity(id, 'tileRowCount', props.tileRowCount);
    }
    this.rowCount = rows;
    this.tileRowCount =
      props.tileRowCount ??
      Math.max(MINIMUM_DEFAULT_TILE_ROWS, Math.ceil(rows / MAXIMUM_DEFAULT_TILE_COUNT));
    this.tileCount = Math.ceil(rows / this.tileRowCount);
    const traceLayout = getRegressionTraceLayout(id, props.weights, rows);
    this.traceTileRowCount = traceLayout.tileRowCount;
    this.traceTileCount = traceLayout.tileCount;
    validateRegressionOutputDispatches(
      this.id,
      {...output, status: undefined, code: output.status.code, converged: output.status.converged},
      [
        ['table', 'summary', 'code', 'converged'],
        {read: ['infinityNorm'], write: ['powerTraces', 'powerTraceStandardErrors']},
        {
          read: [...(this.exactLogDeterminant ? [] : ['powerTraces']), 'infinityNorm'],
          write: ['table', 'summary', 'code', 'converged']
        }
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.table,
        output.summary,
        output.status.code,
        output.status.converged,
        output.residuals,
        output.powerTraces,
        output.powerSums,
        output.infinityNorm,
        output.powerTraceStandardErrors
      ],
      [
        props.predictors,
        props.response,
        ...Object.values(getRegressionWeightBindings(props.weights))
      ]
    );
  }

  /**
   * Returns the means pass, the lagged-response and weights-norm node, the row-sum merge and
   * the lag-column shift, the two `GPUOrdinaryLeastSquares` compositions,
   * `GPUSpatialWeightsTraces`, the residual moment passes (three sums and their merge, then the
   * direct sums at `rho0`), the `rho` search, the coefficient solve, the residual pass, the
   * `Q - 1` Horner steps of
   * `(I - rho W)^-1 X b`, the mean of `W (I - rho W)^-1 X b` (tiles and merge) and the final
   * moment passes, the `tr3` probe pass (per batch `Q` steps and a merge), the variance and the
   * finish kernel.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount, maximumPower, tailExtrapolation} = this;
    const tailWGSL = tailExtrapolation ? getSpatialSeriesTailWGSL(maximumPower, 'powerTraces') : '';
    const exact = this.exactLogDeterminant;
    const exactWGSL = getExactLogDeterminantWGSL('eigenvalues', rowCount);
    const logDeterminantWGSL = exact
      ? exactWGSL
      : `${getSpatialLogDeterminantWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]', tailExtrapolation)}${tailWGSL}`;
    const traceWGSL = exact
      ? ''
      : getTraceSeriesWGSL(
          maximumPower,
          'powerTraces',
          'infinityNorm[infinityNormOffset]',
          tailExtrapolation
        );
    const traceOnlyWGSL = exact
      ? exactWGSL
      : `${getTraceSeriesWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]', tailExtrapolation)}${tailWGSL}`;
    const {weights, output, predictorCount: k} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.eigenvalues,
      ...Object.values(getRegressionWeightBindings(weights)),
      output.table,
      output.summary,
      output.status.code,
      output.status.converged,
      output.residuals,
      output.powerTraces,
      output.powerSums,
      output.infinityNorm,
      output.powerTraceStandardErrors
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      eigenvalues: props.eigenvalues,
      traceParameters: props.traceParameters,
      ...getRegressionWeightBindings(weights),
      ...output,
      status: undefined,
      code: output.status.code,
      converged: output.status.converged
    });
    const p = k + 1;
    const informationSize = p + 1;
    const stochastic = this.traceMode === 'stochastic';
    const packedCount = (k * (k + 1)) / 2;
    // Final sums: [sum wp, sum xc_j wp (k), sum wp^2, Syy, Syu, Su, Suu, Sye, See, Se, gram (packed)].
    const sumWeighted = 0;
    const sumWeightedSquares = k + 1;
    const sumResponseSquares = k + 2;
    const sumResponseResidual = k + 3;
    const sumResidual = k + 4;
    const sumResidualSquares = k + 5;
    const sumResponsePrediction = k + 6;
    const sumPredictionSquares = k + 7;
    const sumPrediction = k + 8;
    // u'e1 of the published residuals and the second OLS residual (the first-order condition).
    const sumResidualLagged = k + 9;
    const gramBase = k + 10;
    const finalSumCount = gramBase + packedCount;

    let slotCount = 0;
    const slots: Record<string, number> = {};
    const allocate = (name: string, count = 1) => {
      slots[name] = slotCount;
      slotCount += count;
    };
    allocate('NORM');
    allocate('BOUND');
    allocate('PARAMETER');
    allocate('VALID');
    allocate('CANDIDATES');
    allocate('AT_BOUNDARY');
    allocate('SERIES_LIMITED');
    allocate('STATUS');
    allocate('BETA', p);
    allocate('MEANS', k);
    allocate('YBAR');
    allocate('CONDITIONING', 4);
    allocate('TRACES', 3);
    allocate('STANDARD_ERRORS', p + 1);
    allocate('VARIANCE_DEFINED');
    // The mean of W A X b, about which the rho column of the information matrix is centered, and
    // a zero written at run time (the runtime zero of getRoundedDifference).
    allocate('WEIGHTED_MEAN');
    allocate('RUNTIME_ZERO');
    // The row sum the lag is formed about (the largest row sum of W): W y = v + ybar * W_ROW_SUM
    // with v the column of the second OLS fit.
    allocate('ROW_SUM');
    // e0'e0, e0'e1, e1'e1 and rho0 = e0'e1 / e1'e1, the minimizer of the residual sum of squares.
    allocate('MOMENTS', 3);
    allocate('PARAMETER_ZERO');
    const slotConstants = Object.entries(slots)
      .map(([name, value]) => `const W_${name}: u32 = ${value}u;`)
      .join('\n');

    const traceTileCount = this.traceTileCount;
    const probeTotal = this.traceMode === 'exact' ? rowCount : (props.traceProbeCount ?? 64);
    validateRegressionAllocations(graph.device, id, {
      ...getOrdinaryLeastSquaresAllocationLengths(rowCount, k, tileRowCount),
      ...getRegressionTraceAllocationLengths({
        rowCount,
        edgeCount: getRegressionMaximumEdgeCount(weights),
        maximumPower,
        tileCount: traceTileCount,
        mode: this.traceMode,
        probeCount: props.traceProbeCount
      }),
      ...Object.fromEntries(
        Object.entries(getSpatialProbeAllocationLengths(rowCount, tileCount, probeTotal, 1)).map(
          ([name, length]) => [`information ${name}`, length]
        )
      ),
      'lag workspace': slotCount,
      'lag row sums': 2 * rowCount,
      'lag norm partials': 2 * getGPUSpatialWeightsTilePlan(weights, tileRowCount).tileCount,
      'lag row scratch': rowCount,
      'lag OLS outputs': Math.max(p, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
      'lag mean partials': tileCount * (k + 1),
      'lag moment partials': tileCount * 3,
      'lag final partials': tileCount * finalSumCount,
      'lag final sums': finalSumCount
    });

    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const transientUint = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const kernel = createKernelFactory(graph, id, OPERATION);
    const workspace = transient('workspace', slotCount);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const P: u32 = ${p}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const MAXIMUM_POWER: u32 = ${maximumPower}u;
${slotConstants}`;
    const finiteDeclarations = `${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}`;
    const f32 = getF32Binding;
    const getWeightBindings = (chunkWeights: GPUSpatialWeights): WGSLKernelBinding[] => [
      getU32Binding('offsets', chunkWeights.offsets, 'read'),
      getU32Binding('neighbors', chunkWeights.neighbors, 'read'),
      f32('weights', chunkWeights.weights, 'read')
    ];
    // The weights-reading norm tiles are laid out per chunk; every other tile kernel tiles all rows.
    const tilePlan = getGPUSpatialWeightsTilePlan(weights, tileRowCount);
    const normTileCount = tilePlan.tileCount;

    // ---- First stage: the lag column, ||W||_inf tiles and the two OLS fits. ----
    // `lagged` holds W (y - ybar) after the tile pass and the lag column v after the shift pass.
    const lagged = transient('lagged-response', rowCount);
    // Per row: the float32 row sum of W and what its additions dropped.
    const rowSums = transient('weights-row-sums', rowCount * 2);
    // Per tile: the largest absolute row sum and the largest row sum.
    const partialNorm = transient('partial-weights-norm', normTileCount * 2);
    const olsResiduals = [
      transient('ols-residuals-response', rowCount),
      transient('ols-residuals-lagged', rowCount)
    ];
    const olsCoefficients = [p, p].map((length, index) =>
      transient(`ols-coefficients-${index}`, length)
    );
    const olsSummaries = [0, 1].map(index =>
      transient(`ols-summary-${index}`, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH)
    );
    const olsStatuses = [0, 1].map(index => transientUint(`ols-status-${index}`, 1));
    const olsNodes = [0, 1].map(index =>
      new GPUOrdinaryLeastSquares({
        id: `${id}-ols-${index === 0 ? 'response' : 'lagged'}`,
        predictors: props.predictors,
        response: index === 0 ? props.response : lagged,
        predictorCount: k,
        tileRowCount,
        output: {
          coefficients: olsCoefficients[index],
          standardErrors: transient(`ols-standard-errors-${index}`, p),
          tStatistics: transient(`ols-t-statistics-${index}`, p),
          summary: olsSummaries[index],
          status: {code: olsStatuses[index]},
          residuals: olsResiduals[index]
        }
      }).getCommandNodes(graph)
    );

    const powerTraces = output.powerTraces ?? transient('power-traces', maximumPower);
    const powerSums = output.powerSums ?? transient('power-sums', maximumPower);
    const infinityNorm = output.infinityNorm ?? transient('traces-infinity-norm', 1);

    const partialMeans = transient('partial-means', tileCount * (k + 1));
    const partialMoments = transient('partial-moments', tileCount * 3);
    const partialDirect = transient('partial-direct-moments', tileCount * 2);
    const residualView = output.residuals ?? transient('residuals', rowCount);
    const reduced = transient('reduced-prediction', rowCount);
    const chain = [transient('chain-a', rowCount), transient('chain-b', rowCount)];
    const weightedPrediction = transient('weighted-prediction', rowCount);
    const partialWeightedMean = transient('partial-weighted-mean', tileCount);
    const partialFinal = transient('partial-final-sums', tileCount * finalSumCount);
    const finalSums = transient('final-sums', finalSumCount);

    const nodes: GPUCommandNode<Parameters>[] = [
      // The column means come first: the lag below is formed about the response mean.
      kernel(
        'means-tiles',
        [
          f32('design', props.predictors, 'read'),
          f32('response', props.response, 'read'),
          f32('partial', partialMeans, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${k + 1}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${k + 1}u; slot++) {
    partial[partialOffset + index * ${k + 1}u + slot] = sums[slot];
  }`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'means-merge',
        [f32('partial', partialMeans, 'read'), f32('workspace', workspace, 'read_write')],
        k + 1,
        constants,
        `${getMergeSumWGSL('partial', 'index', `${k + 1}u`, 'TILE_COUNT')}
  workspace[workspaceOffset + W_MEANS + index] = sum / f32(ROWS);
  if (index == 0u) {
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;
  }`
      ),
      // W (y - ybar), the exact row sums of W, and ||W||_inf. The lag is formed about the response
      // mean because a sum of w * y is rounded to a float32 step of the response: for residuals of
      // 0.002 around 10,000 that rounding was up to 4% of the residual variance. Each y - ybar is
      // exact or rounded once (getRoundedDifference keeps the compiler from summing the raw values
      // and subtracting the mean afterwards). The part left out, ybar * (row sum), needs the row
      // sum of the float32 weights exactly: three weights of float32(1/3) sum to 1.0000000298, and
      // 3e-8 of 10,000 is a tenth of such a residual. getAdditionRoundingError returns what each
      // float32 addition of the row sum drops.
      ...tilePlan.dispatches.map(dispatch =>
        kernel(
          `lagged-response-tiles${dispatch.idSuffix}`,
          [
            ...getWeightBindings(dispatch.weights),
            f32('response', props.response, 'read'),
            f32('workspace', workspace, 'read'),
            f32('lagged', lagged, 'read_write'),
            f32('rowSums', rowSums, 'read_write'),
            f32('partialNorm', partialNorm, 'read_write')
          ],
          dispatch.chunkTileCount,
          `${constants}${dispatch.tileConstantsWGSL ? `\n${dispatch.tileConstantsWGSL}` : ''}
${ROUNDED_DIFFERENCE_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}`,
          rewriteWeightsChunkTileBodyWGSL(
            dispatch,
            `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let responseMean = workspace[workspaceOffset + W_YBAR];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var largest = 0.0;
  var largestRowSum = -3.0e38;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    var absoluteSum = 0.0;
    var rowSum = 0.0;
    var rowSumError = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      sum += weight * getRoundedDifference(response[responseOffset + neighbor], responseMean, runtimeZero);
      absoluteSum += abs(weight);
      let nextRowSum = rowSum + weight;
      rowSumError += getAdditionRoundingError(rowSum, weight, nextRowSum);
      rowSum = nextRowSum;
    }
    lagged[laggedOffset + row] = sum;
    rowSums[rowSumsOffset + 2u * row] = rowSum;
    rowSums[rowSumsOffset + 2u * row + 1u] = rowSumError;
    largest = max(largest, absoluteSum);
    largestRowSum = max(largestRowSum, rowSum);
  }
  partialNorm[partialNormOffset + 2u * index] = largest;
  partialNorm[partialNormOffset + 2u * index + 1u] = largestRowSum;`
          ),
          {workgroupSize: TILE_WORKGROUP_SIZE}
        )
      ),
      // The row sum the lag column is formed about: the largest row sum, which is the common row
      // sum of row-standardized weights (rows without neighbours sum to 0 and do not set it).
      kernel(
        'row-sum-merge',
        [f32('partialNorm', partialNorm, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        constants,
        `var largestRowSum = -3.0e38;
  for (var tile = 0u; tile < ${normTileCount}u; tile++) {
    largestRowSum = max(largestRowSum, partialNorm[partialNormOffset + 2u * tile + 1u]);
  }
  workspace[workspaceOffset + W_ROW_SUM] = largestRowSum;`
      ),
      // The lag column of the second OLS fit: v = W (y - ybar) + ybar (s_i - s), with s_i the exact
      // row sum and s = W_ROW_SUM, so that W y = v + ybar s. The constant ybar s only moves the
      // intercept and is added back there. For rows that share the row sum s the second term is
      // ybar times the rounding of the row sum (a few 1e-8); for a row without neighbours v is
      // -ybar s, the lag 0 of that row on the same footing.
      kernel(
        'lagged-response-shift',
        [
          f32('rowSums', rowSums, 'read'),
          f32('workspace', workspace, 'read'),
          f32('lagged', lagged, 'read_write')
        ],
        rowCount,
        `${constants}
${ROUNDED_DIFFERENCE_WGSL}`,
        `let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  let rowSumDeviation = getRoundedDifference(rowSums[rowSumsOffset + 2u * index], workspace[workspaceOffset + W_ROW_SUM], runtimeZero)
    + rowSums[rowSumsOffset + 2u * index + 1u];
  lagged[laggedOffset + index] = lagged[laggedOffset + index] + workspace[workspaceOffset + W_YBAR] * rowSumDeviation;`
      ),
      ...olsNodes[0],
      ...olsNodes[1],
      ...new GPUSpatialWeightsTraces({
        id: `${id}-traces`,
        tileRowCount: this.traceTileRowCount,
        weights,
        maximumPower,
        mode: this.traceMode,
        probeCount: props.traceProbeCount,
        parameters: props.traceParameters,
        output: {
          powerTraces,
          powerSums,
          infinityNorm,
          powerTraceStandardErrors: output.powerTraceStandardErrors
        }
      }).getCommandNodes(graph),
      kernel(
        'moments-tiles',
        [
          f32('residual0', olsResiduals[0], 'read'),
          f32('residual1', olsResiduals[1], 'read'),
          f32('partial', partialMoments, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sumA = 0.0;
  var sumB = 0.0;
  var sumC = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e0 = residual0[residual0Offset + row];
    let e1 = residual1[residual1Offset + row];
    sumA += e0 * e0;
    sumB += e0 * e1;
    sumC += e1 * e1;
  }
  partial[partialOffset + index * 3u] = sumA;
  partial[partialOffset + index * 3u + 1u] = sumB;
  partial[partialOffset + index * 3u + 2u] = sumC;`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      // rho0 = e0'e1 / e1'e1, the minimizer of the residual sum of squares u(rho)'u(rho).
      kernel(
        'moments-merge',
        [f32('partialMoments', partialMoments, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        constants,
        `${getMergeSumWGSL('partialMoments', '0u', '3u', 'TILE_COUNT', 'momentA')}
  ${getMergeSumWGSL('partialMoments', '1u', '3u', 'TILE_COUNT', 'momentB')}
  ${getMergeSumWGSL('partialMoments', '2u', '3u', 'TILE_COUNT', 'momentC')}
  workspace[workspaceOffset + W_MOMENTS] = momentA;
  workspace[workspaceOffset + W_MOMENTS + 1u] = momentB;
  workspace[workspaceOffset + W_MOMENTS + 2u] = momentC;
  // e1'e1 = 0 (the lag column lies in the span of [1, X]): u'u does not depend on rho, and
  // rho0 is 0 by convention.
  workspace[workspaceOffset + W_PARAMETER_ZERO] = select(momentB / momentC, 0.0, momentC == 0.0);`
      ),
      // The residual sum of squares at rho0, summed per row from r = e0 - rho0 e1, and r'e1 (what
      // the float32 rho0 misses of the minimizer). The search needs u(rho)'u(rho) where it is
      // 1e-9 of e0'e0 or less (a lag model that fits well): e0'e0 - 2 rho e0'e1 + rho^2 e1'e1
      // of three float32 sums is rounding there, and the search then ran to a pole of the
      // derivative or found no candidate. u'u = r'r - 2 (rho - rho0) r'e1 + (rho - rho0)^2 e1'e1
      // has no such cancellation.
      kernel(
        'direct-moments-tiles',
        [
          f32('residual0', olsResiduals[0], 'read'),
          f32('residual1', olsResiduals[1], 'read'),
          f32('workspace', workspace, 'read'),
          f32('partial', partialDirect, 'read_write')
        ],
        tileCount,
        `${constants}
${ROUNDED_DIFFERENCE_WGSL}`,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let parameterZero = workspace[workspaceOffset + W_PARAMETER_ZERO];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var sumSquares = 0.0;
  var sumProducts = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e1 = residual1[residual1Offset + row];
    // Rounded as written, so the tile sums stay sums of the small differences.
    let residual = getRoundedDifference(residual0[residual0Offset + row], parameterZero * e1, runtimeZero);
    sumSquares += residual * residual;
    sumProducts += residual * e1;
  }
  partial[partialOffset + index * 2u] = sumSquares;
  partial[partialOffset + index * 2u + 1u] = sumProducts;`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'rho-search',
        [
          f32('partialNorm', partialNorm, 'read'),
          f32('partialDirect', partialDirect, 'read'),
          exact
            ? getF32Binding('eigenvalues', props.eigenvalues!, 'read')
            : f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}
${ROUNDED_DIFFERENCE_WGSL}
const ROWS_F: f32 = ${rowCount}.0;
${logDeterminantWGSL}
${traceWGSL}
// u(rho)'u(rho) = residualSquares + laggedSquares (rho - rho*)^2, a sum of two non-negative
// terms, with rho* = parameterZero + parameterCorrection its minimizer.
var<private> residualSquares: f32;
var<private> laggedSquares: f32;
var<private> parameterZero: f32;
var<private> parameterCorrection: f32;
var<private> searchRuntimeZero: u32;

fn getParameterOffset(rho: f32) -> f32 {
  return getRoundedDifference(rho, parameterZero, searchRuntimeZero) - parameterCorrection;
}

fn getSumOfSquares(rho: f32) -> f32 {
  let offset = getParameterOffset(rho);
  return residualSquares + laggedSquares * offset * offset;
}

fn getObjective(rho: f32) -> f32 {
  return -0.5 * ROWS_F * log(getSumOfSquares(rho) / ROWS_F) + getSpatialLogDeterminant(rho);
}

fn getDerivative(rho: f32) -> f32 {
  return -ROWS_F * laggedSquares * getParameterOffset(rho) / getSumOfSquares(rho) + getLogDeterminantDerivative(rho);
}
${getParameterSearchWGSL()}`,
        `var largest = 0.0;
  for (var tile = 0u; tile < ${normTileCount}u; tile++) {
    largest = max(largest, partialNorm[partialNormOffset + 2u * tile]);
  }
  let validNorm = largest > 0.0 && isFiniteValue(largest);
  let seriesLimit = ${getWGSLFloatLiteral(
    tailExtrapolation || exact
      ? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION
      : getSpatialMaximumLikelihoodSeriesLimit(maximumPower)
  )};
  let stationarity = ${getWGSLFloatLiteral(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION)};
  let bound = select(getNaN(), min(stationarity, seriesLimit) / largest, validNorm);
  workspace[workspaceOffset + W_NORM] = select(getNaN(), largest, validNorm);
  workspace[workspaceOffset + W_BOUND] = bound;
  workspace[workspaceOffset + W_SERIES_LIMITED] = select(0.0, 1.0, seriesLimit < stationarity);
  workspace[workspaceOffset + W_PARAMETER] = getNaN();
  workspace[workspaceOffset + W_VALID] = 0.0;
  workspace[workspaceOffset + W_CANDIDATES] = 0.0;
  workspace[workspaceOffset + W_AT_BOUNDARY] = 0.0;
  ${getMergeSumWGSL('partialDirect', '0u', '2u', 'TILE_COUNT', 'directSquares')}
  ${getMergeSumWGSL('partialDirect', '1u', '2u', 'TILE_COUNT', 'directProducts')}
  let momentA = workspace[workspaceOffset + W_MOMENTS];
  laggedSquares = workspace[workspaceOffset + W_MOMENTS + 2u];
  parameterZero = workspace[workspaceOffset + W_PARAMETER_ZERO];
  // e1'e1 = 0: a constant sum of squares e0'e0 (directSquares at rho0 = 0), no correction.
  parameterCorrection = select(directProducts / laggedSquares, 0.0, laggedSquares == 0.0);
  residualSquares = directSquares - directProducts * parameterCorrection;
  searchRuntimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  // The likelihood needs u'u > 0 on the whole search interval. S > 0 gives that for every rho.
  // S = 0 (the response residual is a multiple of the lag residual) gives it when the minimizer
  // rho* lies outside the interval, where u'u = e1'e1 (rho - rho*)^2 is positive throughout; S
  // is then a difference that rounds to zero either way, so a rounding-sized negative value
  // (below 1e-6 of r'r) is taken as zero. S = 0 with rho* inside the interval is a pole of the
  // likelihood: no candidate, status 3.
  let minimizer = parameterZero + parameterCorrection;
  let minimizerOutside = laggedSquares > 0.0 && abs(minimizer) > bound;
  let positiveSquares = residualSquares > 0.0 || (minimizerOutside && residualSquares >= -1.0e-6 * directSquares);
  residualSquares = max(residualSquares, 0.0);
  if (validNorm && isFiniteValue(momentA) && isFiniteValue(parameterZero) && isFiniteValue(parameterCorrection)
      && isFiniteValue(residualSquares) && isFiniteValue(laggedSquares) && momentA > 0.0 && laggedSquares >= 0.0 && positiveSquares) {
    let result = searchParameter(bound);
    workspace[workspaceOffset + W_PARAMETER] = result.parameter;
    workspace[workspaceOffset + W_VALID] = select(0.0, 1.0, isFiniteValue(result.parameter));
    workspace[workspaceOffset + W_CANDIDATES] = f32(result.localMaximumCount);
    workspace[workspaceOffset + W_AT_BOUNDARY] = select(0.0, 1.0, result.atBoundary);
  }`
      ),
      kernel(
        'solve-coefficients',
        [
          f32('olsCoefficients0', olsCoefficients[0], 'read'),
          f32('olsCoefficients1', olsCoefficients[1], 'read'),
          f32('olsSummary0', olsSummaries[0], 'read'),
          f32('olsSummary1', olsSummaries[1], 'read'),
          getU32Binding('olsStatus0', olsStatuses[0], 'read'),
          getU32Binding('olsStatus1', olsStatuses[1], 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}`,
        `let rho = workspace[workspaceOffset + W_PARAMETER];
  var code = 0u;
  if (!(f32(ROWS) > f32(K + 2u))) {
    code = 2u;
  } else if (olsSummary0[olsSummary0Offset] != f32(ROWS) || olsSummary1[olsSummary1Offset] != f32(ROWS)) {
    // A non-finite input: the fits skipped a row. Tested before the fit statuses because a
    // non-finite response mean or row sum makes the whole lag column non-finite.
    code = 3u;
  } else if (olsStatus0[olsStatus0Offset] != 0u) {
    code = select(1u, 2u, olsStatus0[olsStatus0Offset] == 2u);
  } else if (olsStatus1[olsStatus1Offset] != 0u) {
    code = select(1u, 2u, olsStatus1[olsStatus1Offset] == 2u);
  } else if (workspace[workspaceOffset + W_VALID] != 1.0 || !isFiniteValue(rho)) {
    code = 3u;
  }
  workspace[workspaceOffset + W_STATUS] = f32(code);
  // The second fit is of the lag column v = W y - ybar s (see lagged-response-shift), so its
  // intercept misses the constant ybar s of the lag.
  let lagShift = workspace[workspaceOffset + W_YBAR] * workspace[workspaceOffset + W_ROW_SUM];
  for (var column = 0u; column < P; column++) {
    var laggedCoefficient = olsCoefficients1[olsCoefficients1Offset + column];
    if (column == 0u) {
      laggedCoefficient += lagShift;
    }
    workspace[workspaceOffset + W_BETA + column] = olsCoefficients0[olsCoefficients0Offset + column] - rho * laggedCoefficient;
  }
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_CONDITIONING + slot] = olsSummary0[olsSummary0Offset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER}u + slot];
  }`
      ),
      kernel(
        'residuals',
        [
          f32('response', props.response, 'read'),
          f32('lagged', lagged, 'read'),
          f32('residual0', olsResiduals[0], 'read'),
          f32('residual1', olsResiduals[1], 'read'),
          f32('residualOut', residualView, 'read_write'),
          f32('reduced', reduced, 'read_write'),
          f32('workspace', workspace, 'read')
        ],
        rowCount,
        constants,
        `let rho = workspace[workspaceOffset + W_PARAMETER];
  residualOut[residualOutOffset + index] = residual0[residual0Offset + index] - rho * residual1[residual1Offset + index];
  // X b = (y - e0) - rho (W y - e1): the OLS fitted values of y and W y, with W y the lag column
  // plus the constant it was formed about.
  let lagShift = workspace[workspaceOffset + W_YBAR] * workspace[workspaceOffset + W_ROW_SUM];
  reduced[reducedOffset + index] = (response[responseOffset + index] - residual0[residual0Offset + index])
    - rho * ((lagged[laggedOffset + index] + lagShift) - residual1[residual1Offset + index]);`
      )
    ];

    // ---- (I - rho W)^-1 X b by Horner: u <- z + rho W u, Q - 1 times from u = z. ----
    for (let step = 1; step < maximumPower; step++) {
      const first = step === 1;
      nodes.push(
        ...tilePlan.dispatches.map(dispatch =>
          kernel(
            `inverse-series-${step}${dispatch.idSuffix}`,
            [
              ...getWeightBindings(dispatch.weights),
              f32('reduced', reduced, 'read'),
              ...(first ? [] : [f32('vectorIn', chain[step % 2], 'read')]),
              f32('vectorOut', chain[(step - 1) % 2], 'read_write'),
              f32('workspace', workspace, 'read')
            ],
            dispatch.rowCount,
            `${constants}${dispatch.constantsWGSL ? `\n${dispatch.constantsWGSL}` : ''}`,
            wrapWeightsChunkRowBodyWGSL(
              dispatch,
              `let rho = workspace[workspaceOffset + W_PARAMETER];
  var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * ${first ? 'reduced[reducedOffset + neighbor]' : 'vectorIn[vectorInOffset + neighbor]'};
  }
  vectorOut[vectorOutOffset + index] = reduced[reducedOffset + index] + rho * sum;`
            )
          )
        )
      );
    }
    const spatialPrediction = chain[(maximumPower - 2) % 2];
    nodes.push(
      ...tilePlan.dispatches.map(dispatch =>
        kernel(
          `weighted-prediction${dispatch.idSuffix}`,
          [
            ...getWeightBindings(dispatch.weights),
            f32('vectorIn', spatialPrediction, 'read'),
            f32('vectorOut', weightedPrediction, 'read_write')
          ],
          dispatch.rowCount,
          `${constants}${dispatch.constantsWGSL ? `\n${dispatch.constantsWGSL}` : ''}`,
          wrapWeightsChunkRowBodyWGSL(
            dispatch,
            `var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * vectorIn[vectorInOffset + neighbor];
  }
  vectorOut[vectorOutOffset + index] = sum;`
          )
        )
      ),
      // The mean of W A X b. Its column of the information matrix is accumulated about this mean:
      // for a response with a large mean, sum (W A X b)^2 - (sum W A X b)^2 / n of the raw column
      // cancels to nothing in float32 and the variance of rho was undefined or wrong.
      kernel(
        'weighted-mean-tiles',
        [
          f32('weighted', weightedPrediction, 'read'),
          f32('partial', partialWeightedMean, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    sum += weighted[weightedOffset + row];
  }
  partial[partialOffset + index] = sum;`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'weighted-mean-merge',
        [f32('partial', partialWeightedMean, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        constants,
        `${getMergeSumWGSL('partial', '0u', '1u', 'TILE_COUNT')}
  workspace[workspaceOffset + W_WEIGHTED_MEAN] = sum / f32(ROWS);
  workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;`
      ),
      kernel(
        'final-moments-tiles',
        [
          f32('design', props.predictors, 'read'),
          f32('response', props.response, 'read'),
          f32('residual', residualView, 'read'),
          f32('residual1', olsResiduals[1], 'read'),
          f32('prediction', spatialPrediction, 'read'),
          f32('weighted', weightedPrediction, 'read'),
          f32('workspace', workspace, 'read'),
          f32('partial', partialFinal, 'read_write')
        ],
        tileCount,
        `${constants}
${ROUNDED_DIFFERENCE_WGSL}`,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let ybar = workspace[workspaceOffset + W_YBAR];
  let weightedMean = workspace[workspaceOffset + W_WEIGHTED_MEAN];
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var sums: array<f32, ${finalSumCount}>;
  var centered: array<f32, ${k}>;
  for (var row = firstRow; row < lastRow; row++) {
    // Each difference from a mean is rounded as written (getRoundedDifference), so the tile sums
    // below cannot be regrouped into sums of the raw values less a multiple of the mean.
    let y = getRoundedDifference(response[responseOffset + row], ybar, runtimeZero);
    let u = residual[residualOffset + row];
    let e = getRoundedDifference(prediction[predictionOffset + row], ybar, runtimeZero);
    // W A X b about its mean, rounded as written so the sums below stay sums of deviations.
    let wp = getRoundedDifference(weighted[weightedOffset + row], weightedMean, runtimeZero);
    sums[${sumWeighted}] += wp;
    sums[${sumWeightedSquares}] += wp * wp;
    sums[${sumResponseSquares}] += y * y;
    sums[${sumResponseResidual}] += y * u;
    sums[${sumResidual}] += u;
    sums[${sumResidualSquares}] += u * u;
    sums[${sumResponsePrediction}] += y * e;
    sums[${sumPredictionSquares}] += e * e;
    sums[${sumPrediction}] += e;
    sums[${sumResidualLagged}] += u * residual1[residual1Offset + row];
    for (var column = 0u; column < K; column++) {
      centered[column] = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + W_MEANS + column], runtimeZero);
      sums[1u + column] += centered[column] * wp;
    }
    for (var first = 0u; first < K; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[${gramBase}u + first * (first + 1u) / 2u + second] += centered[first] * centered[second];
      }
    }
  }
  for (var slot = 0u; slot < ${finalSumCount}u; slot++) {
    partial[partialOffset + index * ${finalSumCount}u + slot] = sums[slot];
  }`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'final-merge',
        [f32('partial', partialFinal, 'read'), f32('finalSums', finalSums, 'read_write')],
        finalSumCount,
        constants,
        `${getMergeSumWGSL('partial', 'index', `${finalSumCount}u`, 'TILE_COUNT')}
  finalSums[finalSumsOffset + index] = sum;`
      )
    );

    // ---- tr3 = tr((W A)' W A) probe pass at the estimated rho. ----
    const probes = createSeriesProbeNodes(graph, {
      id: `${id}-tr3`,
      operation: OPERATION,
      weights,
      rowCount,
      maximumPower,
      mode: this.traceMode,
      probeCount: props.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
      seedParameters: props.traceParameters,
      tileRowCount,
      tileCount,
      workspace,
      parameterSlot: slots['PARAMETER']
    });
    nodes.push(...probes.nodes);
    const probeScale = stochastic ? `1.0 / f32(${probes.probeTotal}u)` : '1.0';

    nodes.push(
      kernel(
        'variance',
        [
          f32('finalSums', finalSums, 'read'),
          f32('dots', probes.dots, 'read'),
          exact
            ? getF32Binding('eigenvalues', props.eigenvalues!, 'read')
            : f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}
${traceOnlyWGSL}
${getCholeskyWGSL(informationSize)}`,
        `let n = f32(ROWS);
  let rho = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = finalSums[finalSumsOffset + ${sumResidualSquares}u] / n;
  let nan = getNaN();
  for (var slot = 0u; slot < P + 1u; slot++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + slot] = nan;
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  let traceOne = getTraceOne(rho);
  let traceTwo = getTraceTwo(rho);
  ${probes.dotsSumWGSL}
  let traceThree = sum * ${probeScale};
  workspace[workspaceOffset + W_TRACES] = traceOne;
  workspace[workspaceOffset + W_TRACES + 1u] = traceTwo;
  workspace[workspaceOffset + W_TRACES + 2u] = traceThree;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2) || !isFiniteValue(traceThree) || !isFiniteValue(rho)) {
    return;
  }
  // Information matrix on [intercept (centered), b (K), rho]; the sigma2 row is eliminated. The
  // rho column is W A X b about its mean (the sums of final-moments-tiles), which leaves the
  // variances of b and rho unchanged and moves the intercept, mapped back below.
  var a: array<f32, ${informationSize * informationSize}>;
  a[0] = n / sigma2;
  for (var first = 0u; first < K; first++) {
    for (var second = 0u; second <= first; second++) {
      let value = finalSums[finalSumsOffset + ${gramBase}u + first * (first + 1u) / 2u + second] / sigma2;
      a[(1u + first) * ${informationSize}u + 1u + second] = value;
      a[(1u + second) * ${informationSize}u + 1u + first] = value;
    }
  }
  for (var column = 0u; column < P; column++) {
    let value = finalSums[finalSumsOffset + ${sumWeighted}u + column] / sigma2;
    a[column * ${informationSize}u + P] = value;
    a[P * ${informationSize}u + column] = value;
  }
  a[P * ${informationSize}u + P] = traceTwo + traceThree + finalSums[finalSumsOffset + ${sumWeightedSquares}u] / sigma2 - 2.0 * traceOne * traceOne / n;
  var scale: array<f32, ${informationSize}>;
  for (var column = 0u; column < ${informationSize}u; column++) {
    let diagonal = a[column * ${informationSize}u + column];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return;
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < ${informationSize}u; row++) {
    for (var column = 0u; column < ${informationSize}u; column++) {
      a[row * ${informationSize}u + column] = a[row * ${informationSize}u + column] / (scale[row] * scale[column]);
    }
  }
  if (!choleskyFactor_${informationSize}(&a)) {
    return;
  }
  // Covariance V = D^-1 (scaled inverse) D^-1, one unit solve per column.
  var covariance: array<f32, ${informationSize * informationSize}>;
  for (var column = 0u; column < ${informationSize}u; column++) {
    var unit: array<f32, ${informationSize}>;
    unit[column] = 1.0;
    choleskySolve_${informationSize}(&a, &unit);
    for (var row = 0u; row < ${informationSize}u; row++) {
      covariance[row * ${informationSize}u + column] = unit[row] / (scale[row] * scale[column]);
    }
  }
  // Intercept of the uncentered model: b0 = a0 - m'b - mean(W A X b) rho, with a0 the constant
  // of the model on the centered predictors and the centered W A X b column.
  var shift: array<f32, ${informationSize}>;
  for (var column = 0u; column < K; column++) {
    shift[1u + column] = workspace[workspaceOffset + W_MEANS + column];
  }
  shift[P] = workspace[workspaceOffset + W_WEIGHTED_MEAN];
  var interceptVariance = covariance[0];
  for (var first = 1u; first < ${informationSize}u; first++) {
    interceptVariance -= 2.0 * shift[first] * covariance[first];
    for (var second = 1u; second < ${informationSize}u; second++) {
      interceptVariance += shift[first] * shift[second] * covariance[first * ${informationSize}u + second];
    }
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS] = sqrt(interceptVariance);
  for (var column = 1u; column < P; column++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + column] = sqrt(covariance[column * ${informationSize}u + column]);
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS + P] = sqrt(covariance[P * ${informationSize}u + P]);
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 1.0;`
      ),
      kernel(
        'finish',
        [
          f32('finalSums', finalSums, 'read'),
          exact
            ? getF32Binding('eigenvalues', props.eigenvalues!, 'read')
            : f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read'),
          f32('tableOut', output.table, 'read_write'),
          f32('summaryOut', output.summary, 'read_write'),
          getU32Binding('statusOut', output.status.code, 'read_write'),
          ...(output.status.converged
            ? [getU32Binding('convergedOut', output.status.converged, 'read_write')]
            : [])
        ],
        1,
        `${constants}
${finiteDeclarations}
${ERFC_VALUE_WGSL}
${logDeterminantWGSL}
${getTableRowWGSL()}
${getFinishWGSL({
  predictorCount: k,
  maximumPower,
  stochastic,
  sumResponseSquares,
  sumResponseResidual,
  sumResidual,
  sumResidualSquares,
  sumResponsePrediction,
  sumPredictionSquares,
  sumPrediction,
  sumResidualLagged,
  hasConverged: output.status.converged !== undefined,
  exact
})}`,
        'finish();'
      )
    );
    return [
      ...nodes,
      ...getRegressionWeightsSuppressionNodes<Parameters>(graph, id, props.weights, props.output)
    ];
  }
}

function getFinishWGSL(options: {
  predictorCount: number;
  maximumPower: number;
  stochastic: boolean;
  sumResponseSquares: number;
  sumResponseResidual: number;
  sumResidual: number;
  sumResidualSquares: number;
  sumResponsePrediction: number;
  sumPredictionSquares: number;
  sumPrediction: number;
  sumResidualLagged: number;
  hasConverged: boolean;
  exact: boolean;
}): string {
  const {predictorCount: k} = options;
  const rows = k + 2;
  const summary = (slot: number) => `summaryOut[summaryOutOffset + ${slot}u]`;
  const sum = (slot: number) => `finalSums[finalSumsOffset + ${slot}u]`;
  return /* wgsl */ `
fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${rows * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH}u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT)} = f32(ROWS);
  statusOut[statusOutOffset] = status;
${options.hasConverged ? '  convergedOut[convergedOutOffset] = 0u;' : ''}
}

fn finish() {
  let code = u32(workspace[workspaceOffset + W_STATUS]);
  if (code != 0u) {
    fail(code);
    return;
  }
  let n = f32(ROWS);
  let rho = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = ${sum(options.sumResidualSquares)} / n;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2)) {
    fail(3u);
    return;
  }
  // The first-order condition of an interior estimate, from the published residuals: the
  // derivative of the concentrated log-likelihood at rho against the square root of its
  // curvature (see STATIONARITY_TOLERANCE). u'u and u'e1 are per-row sums of the published
  // residuals, not the sums the search used; the traces are those of the variance kernel at
  // rho. Only ratios of the sums are formed (u'e1 / u'u, e1'e1 / u'u), so the test does not
  // depend on the scale of the response: the products of two sums overflowed float32 for a
  // response of 1e10. The allowance for the float32 spacing of rho is the change of the
  // derivative over STATIONARITY_PARAMETER_STEPS steps of rho, curvature times that distance.
  if (workspace[workspaceOffset + W_AT_BOUNDARY] != 1.0) {
    let residualSquares = sigma2 * n;
    let laggedRatio = ${sum(options.sumResidualLagged)} / residualSquares;
    let score = n * laggedRatio - workspace[workspaceOffset + W_TRACES];
    let curvature = n * (workspace[workspaceOffset + W_MOMENTS + 2u] / residualSquares - 2.0 * laggedRatio * laggedRatio)
      + workspace[workspaceOffset + W_TRACES + 1u];
    // One float32 step of rho, from its bit pattern.
    let parameterStep = bitcast<f32>(bitcast<u32>(abs(rho)) + 1u) - abs(rho);
    let allowance = sqrt(${getWGSLFloatLiteral(STATIONARITY_TOLERANCE)} * curvature)
      + curvature * ${getWGSLFloatLiteral(STATIONARITY_PARAMETER_STEPS)} * parameterStep;
    if (!(curvature > 0.0) || !(abs(score) <= allowance)) {
      fail(3u);
      return;
    }
  }
  let defined = workspace[workspaceOffset + W_VARIANCE_DEFINED] == 1.0;
  for (var row = 0u; row < P; row++) {
    writeTableRow(row, workspace[workspaceOffset + W_BETA + row], workspace[workspaceOffset + W_STANDARD_ERRORS + row]);
  }
  writeTableRow(P, rho, workspace[workspaceOffset + W_STANDARD_ERRORS + P]);
  let logDeterminant = getSpatialLogDeterminant(rho);
  let logLikelihood = -0.5 * n * log(sigma2) + logDeterminant - 0.5 * n * (${getWGSLFloatLiteral(LN_TWO_PI)} + 1.0);
  let parameterCount = f32(P + 1u);
  // Squared correlations of y with y - u and with (I - rho W)^-1 X b.
  let responseSquares = ${sum(options.sumResponseSquares)};
  let responseResidual = ${sum(options.sumResponseResidual)};
  let residualTotal = ${sum(options.sumResidual)};
  let predictionCovariance = responseSquares - responseResidual;
  let predictionVariance = responseSquares - 2.0 * responseResidual + sigma2 * n - residualTotal * residualTotal / n;
  let spatialCovariance = ${sum(options.sumResponsePrediction)};
  let spatialVariance = ${sum(options.sumPredictionSquares)} - ${sum(options.sumPrediction)} * ${sum(options.sumPrediction)} / n;
  let conditionNumber = workspace[workspaceOffset + W_CONDITIONING];
  let correlationCondition = workspace[workspaceOffset + W_CONDITIONING + 1u];
  let bound = workspace[workspaceOffset + W_BOUND];
  let norm = workspace[workspaceOffset + W_NORM];
  let scaled = abs(rho) * norm;
  let atBoundary = workspace[workspaceOffset + W_AT_BOUNDARY] == 1.0;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT)} = n;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER)} = rho;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED)} = sigma2;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD)} = logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC)} = 2.0 * parameterCount - 2.0 * logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ)} = parameterCount * log(n) - 2.0 * logLikelihood;
  // Ratios first: the product of two sums of squares overflows float32 for a response of 1e10.
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED)} = (predictionCovariance / responseSquares) * (predictionCovariance / predictionVariance);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL)} = (spatialCovariance / responseSquares) * (spatialCovariance / spatialVariance);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES)} = sigma2 * n;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER)} = conditionNumber;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER)} = correlationCondition;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR)} = workspace[workspaceOffset + W_CONDITIONING + 2u];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PRECISION_WARNING)} = select(0.0, 1.0, correlationCondition > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)});
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND)} = bound;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION)} = ${options.stochastic ? '1.0' : '0.0'};
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED)} = select(0.0, 1.0, !atBoundary);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY)} = select(0.0, 1.0, atBoundary);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED)} = workspace[workspaceOffset + W_SERIES_LIMITED];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND)} = select(
    1.0e38,
    n * pow(scaled, ${options.maximumPower + 1}.0) / (${options.maximumPower + 1}.0 * (1.0 - scaled)),
    scaled < 1.0
  );
  ${options.exact ? `${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND)} = 0.0;` : ''}
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED)} = select(0.0, 1.0, defined);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT)} = workspace[workspaceOffset + W_CANDIDATES];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT)} = logDeterminant;
  statusOut[statusOutOffset] = 0u;
${options.hasConverged ? '  convergedOut[convergedOutOffset] = select(0u, 1u, !atBoundary);' : ''}
}
`;
}
