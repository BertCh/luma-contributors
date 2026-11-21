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
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateRequirement,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  getGPUSpatialWeightsViews
} from '../spatial-weights/spatial-weights';
import {getCholeskyWGSL, getPivotLimitedCholeskyWGSL} from './spatial-regression-solve';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {
  getRegressionTileLayout,
  getRowTileSumDispatchShapes,
  getRowTileSumNodes,
  REGRESSION_TILE_MAXIMUM_ROW_COUNT
} from './regression-row-tiles';
import {
  getProbitSpatialLagNodes,
  getProbitWeightsFailureStatusEntries,
  getProbitStatusFailureComparison,
  getProbitWeightsStatusSummary,
  PROBIT_SPATIAL_LAG_COMMAND_NODE_COUNT,
  validateProbitSpatialLagStorage,
  validateProbitSpatialWeights,
  getProbitWeightsStatusNodeCount
} from './probit-spatial-lag-nodes';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  getGPUProbitConvergenceThreshold,
  PROBIT_STEP_VERDICT_WGSL,
  resolveProbitConvergenceTolerance
} from './probit-convergence';

const OPERATION = 'GPUProbitRegression';
/** Three statistic/probability pairs in the spatial diagnostic output. */
export const GPU_PROBIT_SPATIAL_DIAGNOSTIC_FIELD_COUNT = 6;
/** Largest predictor count: the 12 x 12 information matrix is factored in one invocation. */
const MAXIMUM_PREDICTOR_COUNT = 11;
/** Largest `maximumIterations`: bounds the unrolled Fisher-scoring nodes (6 per iteration). */
const MAXIMUM_ITERATION_COUNT = 256;
const DEFAULT_MAXIMUM_ITERATIONS = 64;
/** Step halvings evaluated by the line search: steps 1, 1/2, ..., 2^-23 (as the serial search). */
const LINE_SEARCH_STEP_COUNT = 24;
/**
 * Part of a Fisher direction is rounding once a coefficient has settled: a column, or the move
 * of the rows a separating column does not touch, whose contribution is below this fraction of
 * the largest contribution of any column is removed from the candidate direction. On the test
 * GPU every separated scene of `math-review-probit-bound.spec.ts`, from 400 to 1,000,000 rows,
 * is still certified with a fraction of 1e-6, so the rounding is below that; a coefficient
 * that is still moving contributes far more. This constant only shapes which direction is
 * tested: it is not a tolerance of the test, which accepts no move towards the wrong side.
 */
const SEPARATION_NOISE_FRACTION = 1e-4;
/**
 * Maxima the line-search pass reduces for the separation tests, after the likelihood and the
 * wrong-side bound of every step:
 * - 0 to 2: flags of the candidate direction, a row that moves towards the wrong side, a row
 *   whose sign is not decided by the float32 evaluation, a row that moves towards the right
 *   side (see `getSeparationErrorBound`);
 * - 3 to 8: for each of the three single-column thresholds, a row on the wrong side of it and
 *   a row on the right side of it (exact comparisons of the raw predictor);
 * - 9 to 13: the data values next to the current threshold: the largest value not above it
 *   (largest non-negative value, a found flag, the order key of the largest negative value) and
 *   the smallest value not below it (largest magnitude of a negative value, the order key of the
 *   smallest non-negative value). The row reduction is a maximum of non-negative terms, so a
 *   minimum is carried as the maximum of an order key (`getOrderKey`).
 */
const SEPARATION_TERM_COUNT = 14;
/** Single-column thresholds tested per pass: the current one and the two data values next to the previous one. */
const SEPARATION_THRESHOLD_COUNT = 3;
/**
 * Maxima of the initial pass per predictor column: the largest centered, scaled value and the
 * largest of its negation, then four flags: a row above the lowest raw value of the column with
 * a response of 1, with a response of 0; a row below the highest raw value with a response of
 * 1, with a response of 0.
 */
const COLUMN_EXTREME_TERM_COUNT = 6;
/** Maxima of the variance pass per predictor column: its raw extremes in order-key form. */
const RAW_EXTREME_TERM_COUNT = 5;

/**
 * Bound on the float32 evaluation error of a row's move `d_0 + sum_j z_j d_j` along a
 * direction, as a fraction of `|d_0| + sum_j |z_j d_j|`, with `z_j = (x_j - mean_j) scale_j`
 * the centered, scaled design value. In real arithmetic the move is that of the direction
 * `(d_0, d_j scale_j)` on the raw predictors about the float32 means. The float32 value differs
 * by the rounding of `x_j - mean_j` and of its product with the scale (two roundings a term),
 * of the product with `d_j` (one) and of the `predictorCount + 1` additions: at most
 * `(predictorCount + 4) 2^-24` of the sum of magnitudes for correctly rounded operations in any
 * order, with or without fused multiply-adds (Higham, Accuracy and Stability of Numerical
 * Algorithms, section 3.1). The bound used is `(predictorCount + 5) 2^-23`: twice that, which
 * covers operations that are only faithfully rounded (an error of a whole unit in the last
 * place), plus one more `2^-23` for the rounding of the float32 sum of magnitudes it is
 * multiplied with. A move beyond the bound has the sign of its float32 value; a move within it
 * is undecided, and an undecided row is never counted as moving to the right side or as
 * standing still.
 */
function getSeparationErrorBound(predictorCount: number): number {
  return (predictorCount + 5) * 2 ** -23;
}
/** Float32 words of one diagnostic residual row: naive, generalized, standardized, variance, weight. */
const DIAGNOSTIC_RESIDUAL_STRIDE = 5;

export type GPUProbitRegressionProps = {
  id?: string;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  maximumIterations?: number;
  /** Positive finite relative per-coefficient step threshold in centered/scaled coordinates;
   * default 1e-5. Requests must be at most 1e-2 (`MAXIMUM_PROBIT_TOLERANCE`).
   * Success requires min(tolerance, 1e-5). The kernel compares in float32:
   * the threshold is rounded to float32 and floored at 2^-126, and a request below about 1.2e-7
   * asks for more than a float32 step can show, so it usually ends at status 3.
   */
  tolerance?: number;
  /** One-step weighted linearized GMM for a latent spatial-lag probit, about rho = 0.
   * Complete rows induce W without renormalization; islands remain as zero spatial rows.
   * Instruments are [X, WX, W²X], including intercept lags when independent.
   */
  spatialLag?: {
    weights: GPUSpatialWeights;
    output: {
      /** Intercept, predictor slopes, then rho; NaN on failure except status 4. */
      coefficients: GraphDataView<'float32'>;
      /** HC0 structural-residual 2SLS sandwich errors, in coefficient order. */
      standardErrors: GraphDataView<'float32'>;
      /** GPU_PROBIT_SPATIAL_LAG_STATUS: 0 valid, 1 ordinary fit failed, 2 singular first
       * stage, 3 singular second stage/nonfinite result, 4 |rho| >= 1 (published),
       * 5 invalid/incomplete weights. */
      status: GraphDataView<'uint32'>;
    };
  };
  /** Pinkse, Kelejian-Prucha and Pinkse-Slade diagnostics under independent binary probit.
   * Supplied square weights have zero diagonal; complete cases induce the retained graph.
   * These are asymptotic model diagnostics, not latent spatial-lag coefficient estimates.
   */
  spatialDiagnostics?: {
    weights: GPUSpatialWeights;
    output: {
      /** [Pinkse chi², p, KP normal z, two-sided p, Pinkse-Slade chi², p]. */
      tests: GraphDataView<'float32'>;
      /** 0 valid; 1 failed model/no cases; 2 invalid weights; 3 no positive diagnostic variance. */
      status: GraphDataView<'uint32'>;
    };
  };
  output: {
    /**
     * Intercept first, of the model on the raw predictors. Status 3 writes the last iterate;
     * every other failure writes NaN. The raw intercept is `a - sum mean_j b_j` in float32: for
     * a predictor whose mean is millions of its spread it is rounded to a float32 step of that
     * product, which is why `fitted`, `marginalEffects` and the diagnostics are not computed
     * from these values (see the class documentation).
     */
    coefficients: GraphDataView<'float32'>;
    /** Model-based expected-information standard errors, NaN unless converged. */
    standardErrors: GraphDataView<'float32'>;
    /** Fitted probabilities of the centered fit (status 0 and 3), NaN otherwise. */
    fitted: GraphDataView<'float32'>;
    /** Average marginal effect per predictor (not an effect at the means). */
    marginalEffects: GraphDataView<'float32'>;
    /** Complete rows, log likelihood, deviance, iteration count. */
    summary: GraphDataView<'float32'>;
    /** 0 converged; 1 insufficient; 2 singular; 3 not converged (the iterate is written):
     * iterations or line-search steps exhausted, a step pinned by the bound on rows on the
     * wrong side, a linear predictor that overflowed, or a converged step on data the float32
     * evaluation could not tell from separated; 4 a nonbinary response, or separated data
     * with a certificate (see the class documentation). */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Nonspatial binary probit with bounded Fisher scoring and monotone likelihood backtracking.
 * Complete cases, centered/scaled predictors, implicit intercept. The log likelihood and score are
 * row-tile sums merged in a fixed pairwise order, and the line search tolerates likelihood noise
 * relative to |loglik| and sqrt(rows).
 *
 * Tails. The likelihood of a row is `log Phi(s eta)` with `s = +1` for a response of 1 and `-1`
 * for 0, and `Phi` is evaluated through `erfc`, so it keeps its relative accuracy in the lower
 * tail down to the smallest normal f32 (`s eta` near -13). The score is `s phi / Phi(s eta)` and
 * the information weight `(phi / Phi(eta)) (phi / Phi(-eta))`: no `1 - p` is formed, so a row
 * whose probability is within float32 rounding of 0 or 1 still pulls the fit as the likelihood
 * requires (a response of 0 at a linear predictor of 7.7 costs `-eta^2 / 2`, not a constant).
 * The `fitted` output and the spatial diagnostics still use probabilities floored at 1e-7.
 *
 * Outputs of the centered fit. The iterations run on centered, scaled predictors. `fitted`,
 * `marginalEffects` (their mean density) and the diagnostic residuals are computed from that
 * design and its estimate, as the likelihood is. The published raw intercept
 * `a - sum mean_j b_j` is a float32 difference of large numbers when a predictor has a large
 * offset; rebuilt from the raw coefficients, a predictor near 2^24 with a spread of five
 * float32 steps gave fitted probabilities of 0.185, 0.421, 0.691 where 0.136, 0.345, 0.618 are
 * right, and a marginal effect 2.7% low. The slopes are unaffected; the raw intercept is
 * correct to a float32 step of `mean * slope` and is kept as a published output.
 *
 * Status. Three invariants hold:
 * - Status 0 requires, for every coefficient, both the accepted step and the full unshortened
 *   scoring direction to be at most min(tolerance, 1e-5) * (1 + |that coefficient|) in
 *   centered/scaled coordinates (one large coefficient cannot certify another), on data
 *   none of whose tested directions was certified or left undecided (see below). Requests up to
 *   1e-2 are accepted; values below 1e-5 tighten the certificate.
 * - Status 4 for binary data means separated, and only with a certificate: a direction that
 *   provably moves no row towards the wrong side and some row towards the right side. Data that
 *   are not separated are never status 4.
 * - Everything else that is neither too few rows (1) nor a singular information matrix (2) is
 *   status 3, not converged, with the last iterate written. An ordinary well-conditioned fit
 *   is never status 3 at the default settings. Separated data whose certificate the search does
 *   not find are status 3 (or 2, when the information matrix of the runaway fit loses rank
 *   first), not 0.
 *
 * Bound. A row on the right side of the fit (`s eta > 0`) contributes `log(1) = 0` once its
 * probability rounds to 1, however large its linear predictor: such rows are not bounded, and a
 * maximum that puts a correctly classified high-leverage row at a linear predictor of 19 is
 * returned with status 0. A row on the wrong side leaves the float32 range of `Phi` near 13, so
 * trial steps are halved until the likelihood does not decrease and every row on the wrong side
 * has |linear predictor| <= 12. A fit is pinned when that bound, not the likelihood, shortened
 * the accepted step and the shortened step is below min(tolerance, 1e-5) while the full Fisher
 * step is not: its maximum needs a misclassified row beyond 12 (a probability below 2e-33), which this
 * kernel cannot evaluate. Such data are not separated: the status is 3 and the pinned iterate
 * is written (it was 4 with NaN coefficients).
 *
 * Separation. Separated data have no finite maximum: some direction `d` of the coefficients
 * moves every row to the right side or leaves it where it is (`s x'd >= 0` for every row, and
 * `> 0` for one), so the likelihood rises along it without bound. Data that are not separated
 * have no such direction: some row always moves the wrong way. The fit certifies separation by
 * exhibiting a direction, never by the size of a linear predictor, and a direction counts only
 * when the sign of every row's move is known. There is no tolerance: a move towards the wrong
 * side is disqualifying however small, and a row whose sign float32 cannot decide is not
 * counted as standing still. Three kinds of direction are tested, all reduced over the rows
 * with maxima of flags (exact, whatever the row count):
 * - A column that separates at its lowest or highest value: the rows above the lowest value of
 *   a column (or below its highest) all have the same response. The direction `x_j - lowest`
 *   moves exactly those rows, and the rows at the lowest value stand still whatever their
 *   response. Raw values are compared, which is exact, and the test does not depend on the fit:
 *   an indicator of a group of rows that all have a response of 1 is certified at any
 *   `maximumIterations` and any `tolerance`, whatever the share of the group (measured from 1
 *   row of 600 to 1 row of 100,000 and 20 of 1,000,000).
 * - A single-column threshold from the Fisher direction. When the direction, with its rounding
 *   taken out, has one column left (columns whose largest contribution to any row is below
 *   1e-4 of the largest contribution of any column are dropped, and the intercept is set so
 *   that the rows at an end of the kept columns do not move when their move is below that
 *   level), it is `sign * (x_j - value)` with `value` where the direction changes sign. Rows
 *   are compared with `value`, and in the next pass with the two data values next to it, so a
 *   column that separates at an interior value where both responses occur (quasi-complete
 *   separation, a three-valued predictor whose mixed rows are at its middle value) is certified
 *   one iteration after the direction points there.
 * - The Fisher direction itself (rounding taken out), for separation by a combination of
 *   columns. The move of a row, `d_0 + sum_j z_j d_j`, takes its sign from the float32 value
 *   only when that is beyond the evaluation error bound `(predictorCount + 5) 2^-23` of the
 *   sum of the magnitudes of its terms (`getSeparationErrorBound`); a row all of whose terms
 *   are exactly zero stands still; any other row is undecided and the direction is not a
 *   certificate. Complete separation by several predictors is certified this way.
 *   Quasi-complete separation by a combination of columns, whose tied rows have moves that
 *   cancel to rounding, is not: such data end as status 3 or 2.
 * A certificate is kept once found (the directions of a separated fit end as rounding), and a
 * certified fit is status 4 however it ends: the iterations run out, the rows that still carry
 * information no longer span the design (a singular information matrix; without a certificate
 * that is status 2), the line search fails, a linear predictor overflows, or a step falls below
 * `tolerance`. All-one and all-zero samples are status 4 as well. Measured on the test GPU:
 * complete separation by one predictor is certified from a budget of two iterations, by two
 * predictors from seven, quasi-complete separation at an interior value from five; with a
 * smaller `maximumIterations` those are status 3.
 * A candidate direction that moves no row towards the wrong side as far as float32 can tell but
 * leaves some row undecided is remembered: a later step below `tolerance` is then not accepted
 * as a maximum (status 3), because on separated data it may be rounding. No scene of the specs
 * needs this guard; it cannot fire on data with a row that provably moves the wrong way.
 * Not separated, never status 4: data within 1.5e-5 (of a row's own terms) of separation, for
 * which a tolerance of 1e-3 used to report status 4, converge to their maximum with status 0.
 *
 * Status 3 is a fit that did not converge on data without a certificate: iterations or
 * line-search steps exhausted, a step pinned by the bound, a linear predictor that overflowed,
 * or a small step after an undecided direction. Its coefficients are the last iterate, however
 * large its linear predictors are (a correctly classified row beyond 8 does not make an
 * unfinished fit status 4).
 *
 * Every row pass is parallel (the row-tile pattern of `GPUOrdinaryLeastSquares`: one invocation
 * per tile of consecutive rows, then a pairwise merge in a fixed tile order, so results are bitwise
 * reproducible). Each Fisher-scoring iteration is unrolled into six nodes:
 * 1. tiles and merge of the score `X'(y - p) phi / (p (1 - p))` and the expected information
 *    `X' W X` at the current coefficients (and the largest |linear predictor|);
 * 2. one invocation that factors the information, tests convergence and the iteration limit, and
 *    solves for the Fisher direction and the candidate of the separation test (a small `p x p`
 *    solve, no row loop);
 * 3. tiles and merge of the log likelihood at all 24 halved steps `1, 1/2, ..., 2^-23` in one pass
 *    (and the flags of the separation tests);
 * 4. one invocation that records a certificate, accepts the largest acceptable step, exactly as
 *    the serial backtracking did, and updates the coefficients.
 * The one-invocation nodes own a GPU dispatch gate (`createWGSLDispatchGate`): once the fit has
 * converged, failed or exhausted its iterations, every remaining iteration launches nothing.
 * `estimatedCommandNodeCount` is `6 * maximumIterations` plus a fixed prologue and epilogue.
 * Rows are limited by float32 counts (`2^24`) and by the predictor (and, with diagnostics, the
 * diagnostic residual) bindings of the device. Row and tile counts are pipeline overrides, so the
 * row passes of models that differ only in row count share shader modules.
 * The expected information is singular (status 2) when a Cholesky pivot, relative to its diagonal,
 * is below `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (`1e-3`), the limit of
 * `GPUOrdinaryLeastSquares`: near-collinear predictors report status 2, not an estimate.
 * Optional latent spatial lag. For y* = rho W y* + X beta + e with standard normal e,
 * differentiating Phi((I-rho W)^-1 X beta) at (beta0, 0) gives G_beta = phi(X beta0) X
 * and G_rho = phi(X beta0) W X beta0. Zero-diagonal W makes the derivative of the latent
 * variance zero there. Thus y - p + G_beta beta0 = [G_beta, G_rho] (beta, rho) + error
 * to first order. The Klier-McMillen probability-residual variant used here multiplies this
 * target and every gradient by 1/sqrt(p(1-p)), with p clamped as for fitted probabilities.
 * Project these weighted gradients on unweighted Z = [X, WX, W²X], including W1 and W²1,
 * then regress the weighted target on the projected gradients H. Complete the spatial blocks
 * on the centered/scaled ordinary design, then center every nonconstant instrument column
 * before its Gram matrix. This preserves the full span under predictor translation even on
 * masked graphs and islands. Redundant
 * instrument columns are dropped in declared order at the ordinary 1e-3 factor-pivot limit;
 * fewer independent instruments than coefficients is status 2. The second stage uses the
 * same pivot limit. HC0 covariance is (H'H)^-1 H'diag(r²)H (H'H)^-1 with structural
 * residual r = weightedTarget - weightedGradient * estimate; no degrees-of-freedom correction.
 * Computation uses the centered ordinary design and transforms beta and its covariance back
 * to raw units; instruments retain the span of raw X, WX and W²X even at masked rows/islands.
 * Masks and missing data induce W on complete cases without re-standardization, regardless of
 * weights metadata. Islands remain in the fit. Invalid CSR, negative/nonfinite weights, or a
 * nonzero loss/invalid/approximation flag (or complete or converged equal to zero) suppress the optional estimate.
 * Ordinary output remains the independent probit. |rho| >= 1 is published with status 4.
 * Before second-stage normalization, W eta must exceed a propagated ordinary-fit rounding
 * floor on at least one row: (predictorCount + 8) * 2^-23 times
 * [1 + sum_j |design_j| (1 + |beta0_j|)] in eta, propagated through nonnegative W.
 * Otherwise rho is numerically unidentified (status 3). This check scales with W.
 * Status vectors hold exactly one uint32. requiredCount exceeding count or edge capacity
 * proves loss without overflow. Empty CSR edges are accepted; all-island graphs cannot
 * identify rho (status 2). Writable output ranges must be disjoint. Copied configuration
 * and port maps are frozen; input buffer contents remain mutable per-frame data.
 * This is a local expansion about rho = 0, not constrained spatial maximum likelihood;
 * strong dependence can have substantial bias (see coverage-statistics.md for simulation).
 * The independent 2,000-row, 50-seed ring simulation gives mean rho 0.21738 (sample SD 0.06593)
 * at true rho 0.2 and mean 0.85474 (SD 0.08858) at rho 0.6. This variant has upward bias in
 * that experiment; a general bias towards zero or equivalence to McSpatial is not asserted.
 * Compile-time: all scalar props and port maps. Per-frame: contents of all input views.
 */
export class GPUProbitRegression implements GPUCommandNodeProducer {
  readonly id: string;
  /** Number of command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;
  readonly props: GPUProbitRegressionProps;
  constructor(props: GPUProbitRegressionProps) {
    this.props = props = {
      ...props,
      output: {...props.output},
      spatialDiagnostics: props.spatialDiagnostics
        ? {
            ...props.spatialDiagnostics,
            weights: {
              ...props.spatialDiagnostics.weights,
              status: {...props.spatialDiagnostics.weights.status}
            },
            output: {...props.spatialDiagnostics.output}
          }
        : undefined,
      spatialLag: props.spatialLag
        ? {
            ...props.spatialLag,
            weights: {...props.spatialLag.weights, status: {...props.spatialLag.weights.status}},
            output: {...props.spatialLag.output}
          }
        : undefined
    };
    this.id = props.id ?? 'probit-regression';
    const rows = props.response.length;
    const columns = props.predictorCount;
    if (props.spatialLag) {
      validateRequirement(
        this.id,
        validateProbitSpatialWeights(this.id, props.spatialLag.weights, 'spatialLag.weights') ===
          rows,
        'spatialLag.weights row count must equal response.length'
      );
      for (const [name, view] of Object.entries(props.spatialLag.output)) {
        validateSinglePackedView(view, [name === 'status' ? 'uint32' : 'float32'], this.id);
        validateRequirement(
          this.id,
          view.length === (name === 'status' ? 1 : columns + 2),
          `spatialLag.output.${name} length must equal ${name === 'status' ? 1 : columns + 2}`
        );
      }
    }
    if (props.spatialDiagnostics) {
      const diagnostics = props.spatialDiagnostics;
      if (
        validateProbitSpatialWeights(this.id, diagnostics.weights) !== rows ||
        diagnostics.output.tests.length !== GPU_PROBIT_SPATIAL_DIAGNOSTIC_FIELD_COUNT ||
        diagnostics.output.status.length !== 1
      )
        throw new Error(`${this.id}: diagnostic shape mismatch`);
      validateSinglePackedView(diagnostics.output.tests, ['float32'], this.id);
      validateSinglePackedView(diagnostics.output.status, ['uint32'], this.id);
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [...Object.values(props.output), ...Object.values(diagnostics.output)],
        [
          props.predictors,
          props.response,
          props.mask,
          ...getGPUSpatialWeightsViews(diagnostics.weights)
        ]
      );
    }
    validateCapacity(this.id, 'rows', rows, 1, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
    validateCapacity(this.id, 'predictorCount', columns, 1, MAXIMUM_PREDICTOR_COUNT);
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateCapacity(this.id, 'maximumIterations', maximumIterations, 1, MAXIMUM_ITERATION_COUNT);
    resolveProbitConvergenceTolerance(this.id, props.tolerance);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * columns],
      [props.response, 'float32', rows],
      [props.mask, 'uint32', rows],
      [props.output.coefficients, 'float32', columns + 1],
      [props.output.standardErrors, 'float32', columns + 1],
      [props.output.fitted, 'float32', rows],
      [props.output.marginalEffects, 'float32', columns],
      [props.output.summary, 'float32', 4],
      [props.output.status, 'uint32', 1]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.predictors,
      props.response,
      props.mask
    ]);
    validateDisjointWritableViews(
      this.id,
      Object.fromEntries([
        ...Object.entries(props.output).map(([name, view]) => [`output.${name}`, view]),
        ...Object.entries(props.spatialLag?.output ?? {}).map(([name, view]) => [
          `spatialLag.output.${name}`,
          view
        ]),
        ...Object.entries(props.spatialDiagnostics?.output ?? {}).map(([name, view]) => [
          `spatialDiagnostics.output.${name}`,
          view
        ])
      ])
    );
    if (props.spatialLag || props.spatialDiagnostics) {
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [
          ...Object.values(props.output),
          ...Object.values(props.spatialLag?.output ?? {}),
          ...Object.values(props.spatialDiagnostics?.output ?? {})
        ],
        [
          props.predictors,
          props.response,
          props.mask,
          ...(props.spatialLag
            ? [
                ...getGPUSpatialWeightsViews(props.spatialLag.weights),
                ...Object.values(props.spatialLag.weights.status ?? {})
              ]
            : []),
          ...(props.spatialDiagnostics
            ? [
                ...getGPUSpatialWeightsViews(props.spatialDiagnostics.weights),
                ...Object.values(props.spatialDiagnostics.weights.status ?? {})
              ]
            : [])
        ]
      );
    }
    // Prologue: two tile passes (4 nodes), initialization, initial likelihood (2). Per iteration:
    // information tiles and merge, solve, line-search tiles and merge, decide; the last iteration
    // stops after its solve. Epilogue: finish, predict, effect tiles (2), effects, and with
    // diagnostics the residual, tile (2) and finish nodes.
    this.estimatedCommandNodeCount =
      7 +
      6 * maximumIterations +
      3 +
      5 +
      (props.spatialDiagnostics
        ? 4 +
          (props.spatialDiagnostics.weights.status?.requiredCount ||
          props.spatialDiagnostics.weights.status?.count ||
          getProbitWeightsFailureStatusEntries(props.spatialDiagnostics.weights).length > 4
            ? getProbitWeightsStatusNodeCount(props.spatialDiagnostics.weights)
            : 0)
        : 0) +
      (props.spatialLag
        ? PROBIT_SPATIAL_LAG_COMMAND_NODE_COUNT +
          getProbitWeightsStatusNodeCount(props.spatialLag.weights)
        : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    for (const spatial of [props.spatialLag, props.spatialDiagnostics]) {
      if (spatial) {
        Object.freeze(spatial.output);
        Object.freeze(spatial.weights.status);
        Object.freeze(spatial.weights);
        Object.freeze(spatial);
      }
    }
    Object.freeze(props.output);
    Object.freeze(props);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.response.length;
    const columns = props.predictorCount;
    const size = columns + 1;
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    if (props.spatialLag) {
      validateProbitSpatialLagStorage(graph, id, rows, columns);
      validateGraphViewsBelongToGraph(id, graph, [
        ...Object.values(props.spatialLag.output),
        ...getGPUSpatialWeightsViews(props.spatialLag.weights),
        ...Object.values(props.spatialLag.weights.status ?? {})
      ]);
    }
    // The predictor view (and the diagnostic residual rows) are the largest bindings.
    validateCapacity(
      id,
      'rows',
      rows,
      1,
      Math.min(
        REGRESSION_TILE_MAXIMUM_ROW_COUNT,
        getStorageBindingElementLimit(
          graph.device,
          4 * Math.max(columns, props.spatialDiagnostics ? DIAGNOSTIC_RESIDUAL_STRIDE : 1)
        )
      )
    );
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.mask,
      ...Object.values(props.output)
    ]);
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'f32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const inputBindings = [
      binding('predictors', props.predictors),
      binding('response', props.response),
      ...(props.mask ? [binding('mask', props.mask, 'read', 'u32')] : [])
    ];
    const layout = getProbitStateLayout(columns);
    const tileLayout = getRegressionTileLayout(rows);
    const state = createTransientView(graph, `${id}-state`, 'float32', layout.length);
    const informationSumCount = size + (size * (size + 1)) / 2;
    const [tileShape, informationMergeShape] = getRowTileSumDispatchShapes(
      tileLayout,
      informationSumCount + 2
    );
    // Line-search terms: the likelihood at each step, the largest wrong-side |linear predictor| at
    // each step, and the maxima of the separation tests.
    const lineSearchTermCount = 2 * LINE_SEARCH_STEP_COUNT + SEPARATION_TERM_COUNT;
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: OPERATION,
      shapes: [tileShape, informationMergeShape, tileShape, [lineSearchTermCount]]
    });
    const partial = (name: string, termCount: number) =>
      createTransientView(
        graph,
        `${id}-${name}-partials`,
        'float32',
        tileLayout.tileCount * termCount
      );
    const informationPartial = partial('information', informationSumCount + 2);
    const lineSearchPartial = partial('line-search', lineSearchTermCount);
    // A positive float that decreases as |value| grows, exactly invertible
    // (getOrderKeyMagnitude): the row reductions are maxima of non-negative terms, so the
    // smallest magnitude of a set is carried as the largest key. Integer arithmetic on the bit
    // pattern; the key is positive and normal for every |value| below 1.7e38.
    const orderKeyWGSL = `fn getOrderKey(value: f32) -> f32 { return bitcast<f32>(0x7f7fffffu - (bitcast<u32>(value) & 0x7fffffffu)); }
fn getOrderKeyMagnitude(key: f32) -> f32 { return bitcast<f32>(0x7f7fffffu - bitcast<u32>(key)); }`;
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ERFC_VALUE_WGSL}
fn included(row: u32) -> bool {
 if (${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteBits(response[responseOffset + row])) { return false; }
 for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; } } return true;
}
fn probability(value: f32) -> f32 { return clamp(0.5 * erfcValue(-value * 0.7071067811865475), 1e-7, 1.0 - 1e-7); }
fn density(value: f32) -> f32 { return 0.3989422804014327 * exp(-0.5 * value * value); }
// Phi(value) through erfc, so the lower tail keeps its relative accuracy down to the smallest
// normal f32 (value near -13), where it is floored. Called with +eta for a response of 1 and
// -eta for a response of 0 it is the probability of the observed outcome.
fn tailProbability(value: f32) -> f32 { return max(0.5 * erfcValue(-value * 0.7071067811865475), 1.1754944e-38); }
${orderKeyWGSL}
`;
    const stateWGSL = `const C: u32 = ${columns}u;
const P: u32 = ${size}u;
${Object.entries(layout)
  .map(
    ([name, slot]) =>
      `const S_${name.replace(/[A-Z]/g, letter => `_${letter}`).toUpperCase()}: u32 = ${slot}u;`
  )
  .join('\n')}
fn stateAt(slot: u32) -> f32 { return state[stateOffset + slot]; }`;
    // Centered, scaled design row with the intercept first. Each x - mean is rounded as written
    // (getRoundedDifference, with the zero the initialize node writes), so a linear predictor
    // cannot be regrouped into a sum over the raw predictors less a sum over the means.
    const designWGSL = `${ROUNDED_DIFFERENCE_WGSL}
fn getDesign(row: u32) -> array<f32, ${size}> { var design: array<f32, ${size}>; design[0] = 1.0; let runtimeZero = bitcast<u32>(stateAt(S_RUNTIME_ZERO));
 for (var column = 0u; column < C; column++) { design[column + 1u] = getRoundedDifference(predictors[predictorsOffset + row * C + column], stateAt(S_MEANS + column), runtimeZero) * stateAt(S_SCALES + column); } return design; }`;
    const rowDeclarations = `${common}\n${stateWGSL}\n${designWGSL}`;
    const stateBindings = [...inputBindings, binding('state', state)];
    // Log likelihood at `stepCount` halved steps from the current coefficients along the stored
    // direction (largest step first), and for each step the largest |linear predictor| among the
    // rows it puts on the wrong side (a response of 1 with a negative linear predictor, or 0 with
    // a positive one). Only those rows can leave the float32 range of the likelihood: a row on
    // the right side contributes log(1) = 0 however large its linear predictor is.
    // The next SEPARATION_TERM_COUNT maxima are the separation tests (see that constant). All of
    // them are maxima of flags or of exact values over the rows, so no row is lost in them
    // however many rows there are, and none of them is a tolerance.
    // (1) The candidate direction the solve node stored (the Fisher direction with its rounding
    // taken out). The move of a row is d_0 + sum_j z_j d_j. Its sign is taken from the float32
    // value only when that value is beyond the evaluation error bound (getSeparationErrorBound,
    // a fraction of the sum of the magnitudes of the row's terms); a row all of whose terms are
    // exactly zero (d_j = 0, or the raw predictor equal to its float32 mean) stands still; any
    // other row is undecided.
    // (2) Up to three single-column thresholds (column, value, sign): the direction that moves a
    // row by sign * (x - value). Its move is decided by comparing two float32 values, which is
    // exact, and a row at the value itself stands still whatever its response.
    // (3) The data values next to the current threshold, tested in the next pass.
    // The initial pass (one step) also reduces the extremes of every predictor column, centered
    // and scaled (which the solve node needs to tell rounding from signal) and raw (the
    // thresholds of a column that separates at its lowest or highest value).
    const separationBase = (stepCount: number) => 2 * stepCount;
    const likelihoodWGSL = (stepCount: number) => `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; }
 let design = getDesign(row); let side = 2.0 * response[responseOffset + row] - 1.0;
 var estimate: array<f32, ${size}>; var direction: array<f32, ${size}>;
 for (var column = 0u; column < P; column++) { estimate[column] = stateAt(S_ESTIMATE + column); direction[column] = stateAt(S_DIRECTION + column); }
 var rowMove = 0.0; var rowScale = 0.0; var standsStill = true;
 for (var column = 0u; column < P; column++) {
  let candidate = stateAt(S_SEPARATION_DIRECTION + column); let term = design[column] * candidate; rowMove += term; rowScale += abs(term);
  if (candidate != 0.0) { if (column == 0u) { standsStill = false; } else if (predictors[predictorsOffset + row * C + column - 1u] != stateAt(S_MEANS + column - 1u)) { standsStill = false; } }
 }
 if (!standsStill) {
  let signedMove = side * rowMove; let errorBound = ${getWGSLFloatLiteral(getSeparationErrorBound(columns))} * rowScale;
  let decided = isFiniteBits(rowScale) && isFiniteBits(rowMove) && rowScale >= 1.0e-30;
  if (decided && signedMove > errorBound) { (*terms)[${separationBase(stepCount) + 2}u] = 1.0; }
  else if (decided && signedMove < -errorBound) { (*terms)[${separationBase(stepCount)}u] = 1.0; }
  else { (*terms)[${separationBase(stepCount) + 1}u] = 1.0; }
 }
 for (var threshold = 0u; threshold < ${SEPARATION_THRESHOLD_COUNT}u; threshold++) {
  let thresholdSign = stateAt(S_THRESHOLDS + 3u * threshold + 2u);
  if (thresholdSign == 0.0) { continue; }
  let value = predictors[predictorsOffset + row * C + u32(stateAt(S_THRESHOLDS + 3u * threshold))];
  let thresholdValue = stateAt(S_THRESHOLDS + 3u * threshold + 1u);
  // Exact: a comparison of two float32 values.
  let thresholdMove = thresholdSign * side * select(select(0.0, 1.0, value > thresholdValue), -1.0, value < thresholdValue);
  if (thresholdMove < 0.0) { (*terms)[${separationBase(stepCount) + 3}u + 2u * threshold] = 1.0; }
  if (thresholdMove > 0.0) { (*terms)[${separationBase(stepCount) + 4}u + 2u * threshold] = 1.0; }
  if (threshold == 0u) {
   if (value <= thresholdValue) { if (value >= 0.0) { (*terms)[${separationBase(stepCount) + 9}u] = max((*terms)[${separationBase(stepCount) + 9}u], value); (*terms)[${separationBase(stepCount) + 10}u] = 1.0; } else { (*terms)[${separationBase(stepCount) + 11}u] = max((*terms)[${separationBase(stepCount) + 11}u], getOrderKey(value)); } }
   if (value >= thresholdValue) { if (value < 0.0) { (*terms)[${separationBase(stepCount) + 12}u] = max((*terms)[${separationBase(stepCount) + 12}u], -value); } else { (*terms)[${separationBase(stepCount) + 13}u] = max((*terms)[${separationBase(stepCount) + 13}u], getOrderKey(value)); } }
  }
 }
${
  stepCount === 1
    ? ` for (var column = 0u; column < C; column++) {
  let extremes = ${separationBase(stepCount) + SEPARATION_TERM_COUNT}u + ${COLUMN_EXTREME_TERM_COUNT}u * column; let value = predictors[predictorsOffset + row * C + column];
  (*terms)[extremes] = max((*terms)[extremes], design[column + 1u]);
  (*terms)[extremes + 1u] = max((*terms)[extremes + 1u], -design[column + 1u]);
  // Exact comparisons with the lowest and the highest value of the column: is there a row
  // above the lowest value with a response of 1, of 0; below the highest value with 1, with 0.
  if (value > stateAt(S_RAW_MINIMUM + column)) { (*terms)[extremes + select(3u, 2u, side > 0.0)] = 1.0; }
  if (value < stateAt(S_RAW_MAXIMUM + column)) { (*terms)[extremes + select(5u, 4u, side > 0.0)] = 1.0; }
 }`
    : ''
}
 var scale = 1.0;
 for (var step = 0u; step < ${stepCount}u; step++) {
  var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * (estimate[column] + scale * direction[column]); }
  let signedLinear = side * linear;
  (*terms)[${stepCount}u + step] = max((*terms)[${stepCount}u + step], select(0.0, abs(linear), signedLinear < 0.0 || !isFiniteBits(linear)));
  (*terms)[step] += log(tailProbability(signedLinear));
  scale *= 0.5;
 }
}`;
    const tileSums = (
      name: string,
      sumCount: number,
      maximumCount: number,
      declarations: string,
      outputSlot: number,
      partialView: GraphDataView<'float32'>,
      bindings: readonly WGSLKernelBinding[] = stateBindings,
      gated = false
    ) =>
      getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name.replace(/-\d+/g, ''),
        rowCount: rows,
        layout: tileLayout,
        sumCount,
        maximumCount,
        bindings,
        declarations,
        partial: partialView,
        output: state,
        outputSlot,
        ...(gated
          ? {
              tileGate: gate.getGate(
                name.startsWith('iteration') && name.endsWith('information') ? 0 : 2
              ),
              mergeGate: gate.getGate(name.endsWith('information') ? 1 : 3)
            }
          : {})
      });
    const singleNode = (
      name: string,
      bindings: WGSLKernelBinding[],
      declarations: string,
      body: string
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name.replace(/iteration-\d+-/, ''),
        invocationCount: 1,
        bindings,
        declarations,
        body
      });
    const tolerance = getWGSLFloatLiteral(getGPUProbitConvergenceThreshold(id, props.tolerance));

    const nodes: GPUCommandNode<Parameters>[] = [
      // Complete-row count, successes, column sums and a nonbinary-response flag.
      ...tileSums(
        'prepare',
        columns + 2,
        1,
        `${common}
const C: u32 = ${columns}u;
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let outcome = response[responseOffset + row];
 // Binary means the stored bits are +0, -0 or exactly 1.0: a float comparison may flush a subnormal word to zero and admit it.
 let outcomeBits = bitcast<u32>(outcome);
 if (outcomeBits != 0u && outcomeBits != 0x80000000u && outcomeBits != 0x3f800000u) { (*terms)[SUM_COUNT] = 1.0; }
 (*terms)[0] += 1.0; (*terms)[1] += outcome;
 for (var column = 0u; column < C; column++) { (*terms)[2u + column] += predictors[predictorsOffset + row * C + column]; }
}`,
        layout.prepare,
        partial('prepare', columns + 3),
        inputBindings
      ),
      // Centered sums of squares about the column means (two-pass variance), and the lowest and
      // highest raw value of every column in order-key form (RAW_EXTREME_TERM_COUNT maxima a
      // column: the largest non-negative value, a found flag, the key of the largest negative
      // value; the largest magnitude of a negative value, the key of the smallest non-negative
      // value), which the initialize node decodes.
      ...tileSums(
        'variance',
        columns,
        RAW_EXTREME_TERM_COUNT * columns,
        `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let count = stateAt(S_PREPARE);
 for (var column = 0u; column < C; column++) {
  let value = predictors[predictorsOffset + row * C + column]; let difference = value - stateAt(S_PREPARE + 2u + column) / count; (*terms)[column] += difference * difference;
  let extremes = C + ${RAW_EXTREME_TERM_COUNT}u * column;
  if (value >= 0.0) { (*terms)[extremes] = max((*terms)[extremes], value); (*terms)[extremes + 1u] = 1.0; (*terms)[extremes + 4u] = max((*terms)[extremes + 4u], getOrderKey(value)); }
  else { (*terms)[extremes + 2u] = max((*terms)[extremes + 2u], getOrderKey(value)); (*terms)[extremes + 3u] = max((*terms)[extremes + 3u], -value); }
 }
}`,
        layout.variance,
        partial('variance', (1 + RAW_EXTREME_TERM_COUNT) * columns),
        stateBindings
      ),
      singleNode(
        'initialize',
        [binding('state', state, 'read_write'), gate.binding],
        `${stateWGSL}\n${orderKeyWGSL}`,
        `${gate.disarmWGSL}
state[stateOffset + S_DONE] = 1.0; state[stateOffset + S_CODE] = 1.0; state[stateOffset + S_COUNTED] = 0.0;
state[stateOffset + S_ITERATIONS] = 0.0; state[stateOffset + S_ITERATION] = 0.0; state[stateOffset + S_LARGEST] = 0.0;
// No direction has been tested for separation yet (no certificate, no undecided direction, no threshold); and a zero the compiler cannot see (the runtime zero of getRoundedDifference).
state[stateOffset + S_SEPARATING] = 0.0; state[stateOffset + S_UNDECIDED] = 0.0; state[stateOffset + S_RUNTIME_ZERO] = 0.0;
for (var slot = 0u; slot < ${3 * SEPARATION_THRESHOLD_COUNT}u; slot++) { state[stateOffset + S_THRESHOLDS + slot] = 0.0; }
// The lowest and highest raw value of every column, from the order-key maxima of the variance pass.
for (var column = 0u; column < C; column++) {
 let extremes = S_VARIANCE + C + ${RAW_EXTREME_TERM_COUNT}u * column;
 var highest = stateAt(extremes); if (stateAt(extremes + 1u) == 0.0) { highest = -getOrderKeyMagnitude(stateAt(extremes + 2u)); }
 var lowest = -stateAt(extremes + 3u); if (stateAt(extremes + 3u) == 0.0) { lowest = getOrderKeyMagnitude(stateAt(extremes + 4u)); }
 state[stateOffset + S_RAW_MINIMUM + column] = lowest; state[stateOffset + S_RAW_MAXIMUM + column] = highest;
}
for (var column = 0u; column < P; column++) { state[stateOffset + S_ESTIMATE + column] = 0.0; state[stateOffset + S_DIRECTION + column] = 0.0; state[stateOffset + S_SEPARATION_DIRECTION + column] = 0.0; }
let count = stateAt(S_PREPARE); let successes = stateAt(S_PREPARE + 1u);
for (var column = 0u; column < C; column++) { state[stateOffset + S_MEANS + column] = stateAt(S_PREPARE + 2u + column) / count; state[stateOffset + S_SCALES + column] = 0.0; }
if (stateAt(S_PREPARE + C + 2u) != 0.0) { state[stateOffset + S_CODE] = 4.0; return; }
state[stateOffset + S_COUNTED] = 1.0;
if (count <= f32(P)) { return; } if (successes == 0.0 || successes == count) { state[stateOffset + S_CODE] = 4.0; return; }
for (var column = 0u; column < C; column++) { let variance = stateAt(S_VARIANCE + column); if (!(variance > 0.0)) { state[stateOffset + S_CODE] = 2.0; return; } state[stateOffset + S_SCALES + column] = sqrt(count / variance); }
// Likelihood noise floor: the tiled pairwise sum is good to a few eps * |loglik|, the per-row log/erfc errors add roughly sqrt(count) of that.
state[stateOffset + S_NOISE] = 4.0 * 1.1920929e-7 * max(1.0, sqrt(count));
state[stateOffset + S_CODE] = 3.0; state[stateOffset + S_DONE] = 0.0;
${gate.armWGSL}`
      ),
      // Log likelihood at the zero start (one step of the zero direction).
      ...tileSums(
        'initial-likelihood',
        1,
        1 + SEPARATION_TERM_COUNT + COLUMN_EXTREME_TERM_COUNT * columns,
        likelihoodWGSL(1),
        layout.current,
        partial(
          'initial-likelihood',
          2 + SEPARATION_TERM_COUNT + COLUMN_EXTREME_TERM_COUNT * columns
        )
      )
    ];

    for (let iteration = 0; iteration <= maximumIterations; iteration++) {
      const prefix = `iteration-${iteration}`;
      nodes.push(
        ...tileSums(
          `${prefix}-information`,
          informationSumCount,
          2,
          `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let design = getDesign(row);
 var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
 if (!isFiniteBits(linear)) { (*terms)[SUM_COUNT + 1u] = 1.0; return; }
 (*terms)[SUM_COUNT] = max((*terms)[SUM_COUNT], abs(linear));
 // Score (y - p) phi / (p (1 - p)) and weight phi^2 / (p (1 - p)) as ratios of the density to
 // the two tail probabilities, which stay accurate where p or 1 - p is below float32 rounding.
 let side = 2.0 * response[responseOffset + row] - 1.0; let derivative = density(linear);
 let observedRatio = derivative / tailProbability(side * linear); let score = side * observedRatio; let weight = observedRatio * (derivative / tailProbability(-side * linear));
 var slot = P;
 for (var first = 0u; first < P; first++) { (*terms)[first] += design[first] * score; for (var second = first; second < P; second++) { (*terms)[slot] += design[first] * design[second] * weight; slot++; } }
}`,
          layout.information,
          informationPartial,
          stateBindings,
          true
        ),
        singleNode(
          `${prefix}-solve`,
          [binding('state', state, 'read_write'), gate.binding],
          `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${stateWGSL}\n${IS_FINITE_VALUE_WGSL}\n${orderKeyWGSL}\n${getCholeskyWGSL(size)}\n${getPivotLimitedCholeskyWGSL(size)}`,
          `if (stateAt(S_DONE) != 0.0) { return; }
let iteration = u32(stateAt(S_ITERATION)); state[stateOffset + S_ITERATION] = f32(iteration + 1u);
let informationSums = ${informationSumCount}u;
let extremes = S_CURRENT + ${2 + SEPARATION_TERM_COUNT}u;
// A column separates at its lowest (or highest) value when the rows above (below) that value all have the same response: the direction that moves exactly those rows moves none of them towards the wrong side, and the rows at the value itself stand still. The four flags of the initial pass are exact comparisons of raw values, so this is a certificate, and it does not depend on the fit (an indicator of a group of rows that all have a response of 1 is the usual case).
for (var column = 0u; column < C; column++) { let flags = extremes + ${COLUMN_EXTREME_TERM_COUNT}u * column + 2u; if ((stateAt(flags) != 0.0) != (stateAt(flags + 1u) != 0.0) || (stateAt(flags + 2u) != 0.0) != (stateAt(flags + 3u) != 0.0)) { state[stateOffset + S_SEPARATING] = 1.0; } }
// A linear predictor that overflowed: the fit cannot go on. Not converged (status 3, the iterate is written); separated only with a certificate.
if (stateAt(S_INFORMATION + informationSums + 1u) != 0.0) { state[stateOffset + S_CODE] = select(3.0, 4.0, stateAt(S_SEPARATING) == 1.0); state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
state[stateOffset + S_LARGEST] = stateAt(S_INFORMATION + informationSums);
var information: array<f32, ${size * size}>; var gradient: array<f32, ${size}>; var slot = P;
for (var first = 0u; first < P; first++) { gradient[first] = stateAt(S_INFORMATION + first); for (var second = first; second < P; second++) { let value = stateAt(S_INFORMATION + slot); slot++; information[first * P + second] = value; information[second * P + first] = value; } }
// A singular information matrix at the start is a collinear design (status 2). With a certificate of separation it means the rows that still carry information no longer span the design: the fit is running away (status 4).
if (!choleskyFactorLimited_${size}(&information)) { state[stateOffset + S_CODE] = select(2.0, 4.0, stateAt(S_SEPARATING) == 1.0); state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
for (var entry = 0u; entry < ${size * size}u; entry++) { state[stateOffset + S_FACTOR + entry] = information[entry]; }
if (stateAt(S_CODE) == 0.0 || iteration == ${maximumIterations}u) { state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
choleskySolve_${size}(&information, &gradient);
for (var column = 0u; column < P; column++) { state[stateOffset + S_DIRECTION + column] = gradient[column]; }
// The candidates of the separation tests, from the Fisher direction with its rounding taken out. Any direction that provably moves no row towards the wrong side certifies separation, however it was found, so nothing done here can produce a false certificate; it only decides whether a true one is seen.
// (a) A coefficient that has settled has a direction that is rounding of the score sums: a column whose largest contribution to any row is below ${SEPARATION_NOISE_FRACTION} of the largest contribution of any column is dropped.
var contributions: array<f32, ${size}>; var largestContribution = 0.0;
for (var column = 1u; column < P; column++) { let extent = max(stateAt(extremes + ${COLUMN_EXTREME_TERM_COUNT}u * (column - 1u)), stateAt(extremes + ${COLUMN_EXTREME_TERM_COUNT}u * (column - 1u) + 1u)); contributions[column] = abs(gradient[column]) * extent; largestContribution = max(largestContribution, contributions[column]); }
let noiseLevel = ${getWGSLFloatLiteral(SEPARATION_NOISE_FRACTION)} * largestContribution;
var candidate: array<f32, ${size}>; candidate[0] = gradient[0]; var lowCorner = gradient[0]; var highCorner = gradient[0]; var keptCount = 0u; var keptColumn = 1u;
for (var column = 1u; column < P; column++) { if (contributions[column] >= noiseLevel && largestContribution > 0.0) { candidate[column] = gradient[column]; keptCount++; keptColumn = column; lowCorner += gradient[column] * -stateAt(extremes + ${COLUMN_EXTREME_TERM_COUNT}u * (column - 1u) + 1u); highCorner += gradient[column] * stateAt(extremes + ${COLUMN_EXTREME_TERM_COUNT}u * (column - 1u)); } }
// (b) The intercept of the centered model compensates the kept columns for the rows they do not touch (the rows outside a small group that an indicator separates): for those rows the move is a difference of two equal terms, each a fraction of the group's share of the rows, and its rounding grows against them as the group shrinks. When the move of a row at the low (or the high) end of every kept column is itself below the noise level, the intercept is set so that such a row does not move at all.
if (abs(lowCorner) <= noiseLevel && abs(lowCorner) <= abs(highCorner)) { candidate[0] = gradient[0] - lowCorner; } else if (abs(highCorner) <= noiseLevel) { candidate[0] = gradient[0] - highCorner; }
for (var column = 0u; column < P; column++) { state[stateOffset + S_SEPARATION_DIRECTION + column] = candidate[column]; }
// (c) A candidate with one column is the direction sign * (x - value) of that predictor: the value is where the candidate changes sign, and the move of a row is then decided by comparing its predictor with the value, which is exact, where the float32 sum of two terms that cancel is not. The line-search pass also reports the data values next to it, which are the thresholds of the following pass.
var thresholdSign = 0.0; var thresholdValue = 0.0;
if (keptCount == 1u) {
 let predictor = keptColumn - 1u; let slope = candidate[keptColumn] * stateAt(S_SCALES + predictor);
 thresholdSign = select(select(0.0, 1.0, slope > 0.0), -1.0, slope < 0.0);
 thresholdValue = stateAt(S_MEANS + predictor) - candidate[0] / slope;
 if (!isFiniteValue(thresholdValue)) { thresholdSign = 0.0; thresholdValue = 0.0; }
 state[stateOffset + S_THRESHOLDS] = f32(predictor);
}
state[stateOffset + S_THRESHOLDS + 1u] = thresholdValue; state[stateOffset + S_THRESHOLDS + 2u] = thresholdSign;`
        )
      );
      if (iteration === maximumIterations) {
        break;
      }
      nodes.push(
        ...tileSums(
          `${prefix}-line-search`,
          LINE_SEARCH_STEP_COUNT,
          LINE_SEARCH_STEP_COUNT + SEPARATION_TERM_COUNT,
          likelihoodWGSL(LINE_SEARCH_STEP_COUNT),
          layout.lineSearch,
          lineSearchPartial,
          stateBindings,
          true
        ),
        singleNode(
          `${prefix}-decide`,
          [binding('state', state, 'read_write'), gate.binding],
          `${IS_FINITE_BITS_WGSL}\n${stateWGSL}\n${orderKeyWGSL}\n${PROBIT_STEP_VERDICT_WGSL}`,
          `if (stateAt(S_DONE) != 0.0) { return; }
// Separation is certified by a direction that provably moves some row towards the right side and no row towards the wrong side: the candidate direction when the sign of every row's move was decided, or a single-column threshold (exact comparisons). Data that are not separated have no such direction. The certificate is kept once found, because the directions of a separated fit end as rounding (the score of the separated rows falls below the float32 noise of the score sums near a linear predictor of 6).
let separation = S_LINE_SEARCH + ${2 * LINE_SEARCH_STEP_COUNT}u;
let movesWrong = stateAt(separation) != 0.0; let undecided = stateAt(separation + 1u) != 0.0; let movesRight = stateAt(separation + 2u) != 0.0;
var certified = movesRight && !movesWrong && !undecided;
for (var threshold = 0u; threshold < ${SEPARATION_THRESHOLD_COUNT}u; threshold++) { if (stateAt(S_THRESHOLDS + 3u * threshold + 2u) != 0.0 && stateAt(separation + 3u + 2u * threshold) == 0.0 && stateAt(separation + 4u + 2u * threshold) != 0.0) { certified = true; } }
if (certified) { state[stateOffset + S_SEPARATING] = 1.0; }
// A candidate that moves no row towards the wrong side as far as float32 can tell, but leaves the sign of some row undecided, is not a certificate. It is remembered, because a step that later falls below the tolerance on such data may be rounding and not a maximum: the fit is then status 3, not 0.
if (movesRight && !movesWrong && undecided) { state[stateOffset + S_UNDECIDED] = 1.0; }
// The data values next to the current threshold become the thresholds of the next pass: a column that separates at a value where both outcomes occur needs that value exactly.
let thresholdSign = stateAt(S_THRESHOLDS + 2u); var lowerSign = 0.0; var lowerValue = 0.0; var upperSign = 0.0; var upperValue = 0.0;
if (thresholdSign != 0.0) {
 if (stateAt(separation + 10u) != 0.0) { lowerValue = stateAt(separation + 9u); lowerSign = thresholdSign; } else if (stateAt(separation + 11u) > 0.0) { lowerValue = -getOrderKeyMagnitude(stateAt(separation + 11u)); lowerSign = thresholdSign; }
 if (stateAt(separation + 12u) > 0.0) { upperValue = -stateAt(separation + 12u); upperSign = thresholdSign; } else if (stateAt(separation + 13u) > 0.0) { upperValue = getOrderKeyMagnitude(stateAt(separation + 13u)); upperSign = thresholdSign; }
}
state[stateOffset + S_THRESHOLDS + 3u] = stateAt(S_THRESHOLDS); state[stateOffset + S_THRESHOLDS + 4u] = lowerValue; state[stateOffset + S_THRESHOLDS + 5u] = lowerSign;
state[stateOffset + S_THRESHOLDS + 6u] = stateAt(S_THRESHOLDS); state[stateOffset + S_THRESHOLDS + 7u] = upperValue; state[stateOffset + S_THRESHOLDS + 8u] = upperSign;
let current = stateAt(S_CURRENT); let noiseScale = stateAt(S_NOISE);
var scale = 1.0; var accepted = false; var chosen = 0u;
for (var attempt = 0u; attempt < ${LINE_SEARCH_STEP_COUNT}u; attempt++) {
 let candidateLikelihood = stateAt(S_LINE_SEARCH + attempt); let lastLargestLinear = stateAt(S_LINE_SEARCH + ${LINE_SEARCH_STEP_COUNT}u + attempt);
 // Halve the step until the likelihood does not decrease beyond its noise and every row the step puts on the wrong side stays bounded (|eta| <= 12, the largest value the line-search pass reports); rows on the right side are not bounded.
 let likelihoodAccepts = candidateLikelihood >= current - noiseScale * (abs(current) + 1.0) && isFiniteBits(candidateLikelihood);
 if (likelihoodAccepts && lastLargestLinear <= 12.0) { accepted = true; chosen = attempt; break; }
 // A step shortened here, by the bound or by the likelihood, is tested against its full direction below (verdict 1).
 scale *= 0.5; }
if (!accepted) { state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
// Per coefficient, so that one large coefficient cannot certify an unrelated small one: the verdict is the minimum of getProbitStepVerdict over the coefficients (0 step too large, 1 step small but full direction large, 2 both small).
var verdict = 2u;
for (var column = 0u; column < P; column++) { let direction = stateAt(S_DIRECTION + column); let candidate = stateAt(S_ESTIMATE + column) + scale * direction; verdict = min(verdict, getProbitStepVerdict(scale, direction, candidate, ${tolerance})); state[stateOffset + S_ESTIMATE + column] = candidate; }
state[stateOffset + S_CURRENT] = stateAt(S_LINE_SEARCH + chosen); state[stateOffset + S_ITERATIONS] = stateAt(S_ITERATIONS) + 1.0;
if (verdict != 0u) {
 // A small accepted step means convergence only when the full Fisher step is small too. A step shortened by the bound or by likelihood backtracking cannot certify a large full direction: not converged (the code stays 3 and the iterate is written).
 if (verdict == 1u) { state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
 // Separated data have no maximum to converge to: a small step there is a step along the separating direction below the tolerance, or rounding. Status 4, not 0.
 if (stateAt(S_SEPARATING) == 1.0) { state[stateOffset + S_CODE] = 4.0; state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
 // A direction of this fit could not be told from a separating one: the small step is not accepted as a maximum (the code stays 3).
 if (stateAt(S_UNDECIDED) == 1.0) { state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
 state[stateOffset + S_CODE] = 0.0; }`
        )
      );
    }

    nodes.push(
      singleNode(
        'finish',
        [
          binding('state', state),
          binding('coefficients', props.output.coefficients, 'read_write'),
          binding('errors', props.output.standardErrors, 'read_write'),
          binding('summary', props.output.summary, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32')
        ],
        `${GET_NAN_WGSL}\n${stateWGSL}\n${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(size)}`,
        `for (var column = 0u; column < P; column++) { coefficients[coefficientsOffset + column] = getNaN(); errors[errorsOffset + column] = getNaN(); }
for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); }
let code = u32(stateAt(S_CODE)); status[statusOffset] = code;
if (stateAt(S_COUNTED) == 0.0) { return; }
summary[summaryOffset] = stateAt(S_PREPARE); summary[summaryOffset + 3u] = 0.0;
if (code == 1u || code == 2u || code == 4u) { return; }
// Separation: a direction that provably moves no row towards the wrong side was found, so there is no maximum, however the fit ended. An unfinished fit of data without a certificate stays status 3 and publishes its iterate, however large its linear predictors are.
if (stateAt(S_SEPARATING) == 1.0) { status[statusOffset] = 4u; return; }
var information: array<f32, ${size * size}>; for (var entry = 0u; entry < ${size * size}u; entry++) { information[entry] = stateAt(S_FACTOR + entry); }
var intercept = stateAt(S_ESTIMATE); var interceptTransform: array<f32, ${size}>; interceptTransform[0] = 1.0;
for (var column = 0u; column < C; column++) { let scale = stateAt(S_SCALES + column); let mean = stateAt(S_MEANS + column); let coefficient = stateAt(S_ESTIMATE + column + 1u) * scale; coefficients[coefficientsOffset + column + 1u] = coefficient; intercept -= mean * coefficient; interceptTransform[column + 1u] = -mean * scale; if (code == 0u) { errors[errorsOffset + column + 1u] = scale * sqrt(choleskyInverseDiagonal_${size}(&information, column + 1u)); } }
coefficients[coefficientsOffset] = intercept;
if (code == 0u) { var solved = interceptTransform; choleskySolve_${size}(&information, &solved); var variance = 0.0; for (var column = 0u; column < P; column++) { variance += solved[column] * interceptTransform[column]; } errors[errorsOffset] = sqrt(max(variance, 0.0)); }
let current = stateAt(S_CURRENT);
summary[summaryOffset + 1u] = current; summary[summaryOffset + 2u] = -2.0 * current; summary[summaryOffset + 3u] = stateAt(S_ITERATIONS);`
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-predict`,
        operation: OPERATION,
        variant: 'predict',
        invocationCount: rows,
        // The linear predictor of the centered fit, the one the iterations and the likelihood
        // used: sum_j z_j estimate_j with z the centered, scaled design. Rebuilding it from the
        // published raw coefficients loses the fit when a predictor has a large offset (a
        // predictor near 2^24 with a spread of a few float32 steps: the raw intercept is
        // rounded to a step of mean * slope and a fitted probability of 0.42 came out where
        // 0.345 is right). Coefficients are published for status 0 and 3 only.
        bindings: [
          ...stateBindings,
          binding('status', props.output.status, 'read', 'u32'),
          binding('fitted', props.output.fitted, 'read_write')
        ],
        declarations: rowDeclarations,
        body: `var prediction = getNaN(); let code = status[statusOffset]; if ((code == 0u || code == 3u) && included(index)) { let design = getDesign(index); var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); } if (isFiniteBits(linear)) { prediction = probability(linear); } } fitted[fittedOffset + index] = prediction;`
      }),
      // Average marginal effects: mean density of the linear predictor (of the centered fit, as
      // above) times each slope.
      ...tileSums(
        'effects',
        2,
        0,
        `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let design = getDesign(row); var linear = 0.0;
 for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
 (*terms)[0] += density(linear); (*terms)[1] += 1.0;
}`,
        layout.effects,
        partial('effects', 2),
        stateBindings
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-effects`,
        operation: OPERATION,
        variant: 'average-marginal-effects',
        invocationCount: columns,
        bindings: [
          binding('state', state),
          binding('coefficients', props.output.coefficients),
          binding('effects', props.output.marginalEffects, 'read_write')
        ],
        declarations: stateWGSL,
        body: `effects[effectsOffset + index] = stateAt(S_EFFECTS) / stateAt(S_EFFECTS + 1u) * coefficients[coefficientsOffset + index + 1u];`
      })
    );

    if (props.spatialDiagnostics) {
      const diagnostics = props.spatialDiagnostics;
      const statusEntries = getProbitWeightsFailureStatusEntries(diagnostics.weights);
      const statusSummary =
        diagnostics.weights.status?.requiredCount ||
        diagnostics.weights.status?.count ||
        statusEntries.length > 4
          ? getProbitWeightsStatusSummary(graph, `${id}-diagnostic`, diagnostics.weights)
          : {nodes: [], entries: statusEntries};
      const failureStatuses = statusSummary.entries;
      const neighborCount = diagnostics.weights.neighbors.length;
      const residuals = createTransientView(
        graph,
        `${id}-diagnostic-residuals`,
        'float32',
        rows * DIAGNOSTIC_RESIDUAL_STRIDE
      );
      validateGraphViewsBelongToGraph(id, graph, [
        ...Object.values(diagnostics.output),
        ...getGPUSpatialWeightsViews(diagnostics.weights),
        ...Object.values(diagnostics.weights.status ?? {})
      ]);
      nodes.push(
        ...statusSummary.nodes,
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-diagnostic-residuals`,
          operation: OPERATION,
          variant: 'generalized-residuals',
          invocationCount: rows,
          // The density is that of the linear predictor of the centered fit, as `fitted` is.
          bindings: [
            ...stateBindings,
            binding('fitted', props.output.fitted),
            binding('residuals', residuals, 'read_write')
          ],
          declarations: rowDeclarations,
          body: `for (var slot = 0u; slot < 5u; slot++) { residuals[residualsOffset + index * 5u + slot] = getNaN(); }
let fittedProbability = fitted[fittedOffset + index]; let observed = response[responseOffset + index];
if (!isFiniteBits(observed) || !isFiniteBits(fittedProbability) || !(fittedProbability > 0.0 && fittedProbability < 1.0) ${props.mask ? '|| mask[maskOffset + index] == 0u' : ''}) { return; }
let design = getDesign(index); var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
let rowDensity = density(linear); let variance = fittedProbability * (1.0 - fittedProbability); let naive = observed - fittedProbability;
residuals[residualsOffset + index * 5u] = naive; residuals[residualsOffset + index * 5u + 1u] = rowDensity * naive / variance; residuals[residualsOffset + index * 5u + 2u] = naive / sqrt(variance); residuals[residualsOffset + index * 5u + 3u] = variance; residuals[residualsOffset + index * 5u + 4u] = rowDensity * rowDensity / variance;`
        }),
        // One invocation per row visits its own CSR neighbors: O(nnz) work in row tiles.
        // Sums: count, information, trace, variance trace, naive, generalized and standardized
        // cross-products; maximum: the invalid-weights flag.
        ...tileSums(
          'diagnostics',
          7,
          1,
          `${IS_FINITE_BITS_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 let retained = isFiniteBits(residuals[residualsOffset + row * 5u]);
 if (retained) { (*terms)[0] += 1.0; (*terms)[1] += residuals[residualsOffset + row * 5u + 4u]; }
 let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u]; if (offsets[offsetsOffset] != 0u || end < begin || end > ${neighborCount}u) { (*terms)[SUM_COUNT] = 1.0; return; }
 for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ROW_COUNT || neighbor == row || !isFiniteBits(weight) || weight < 0.0 || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { (*terms)[SUM_COUNT] = 1.0; return; }
 if (!retained || !isFiniteBits(residuals[residualsOffset + neighbor * 5u])) { continue; }
 var reciprocal = 0.0; for (var other = offsets[offsetsOffset + neighbor]; other < min(offsets[offsetsOffset + neighbor + 1u], ${neighborCount}u); other++) { if (neighbors[neighborsOffset + other] == row) { reciprocal = weights[weightsOffset + other]; } }
 let contribution = weight * (weight + reciprocal); (*terms)[2] += contribution; (*terms)[3] += contribution * residuals[residualsOffset + row * 5u + 3u] * residuals[residualsOffset + neighbor * 5u + 3u];
 (*terms)[4] += weight * residuals[residualsOffset + row * 5u] * residuals[residualsOffset + neighbor * 5u]; (*terms)[5] += weight * residuals[residualsOffset + row * 5u + 1u] * residuals[residualsOffset + neighbor * 5u + 1u]; (*terms)[6] += weight * residuals[residualsOffset + row * 5u + 2u] * residuals[residualsOffset + neighbor * 5u + 2u]; }
}`,
          layout.diagnostics,
          partial('diagnostics', 8),
          [
            binding('offsets', diagnostics.weights.offsets, 'read', 'u32'),
            binding('neighbors', diagnostics.weights.neighbors, 'read', 'u32'),
            binding('weights', diagnostics.weights.weights),
            binding('residuals', residuals)
          ]
        ),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-spatial-diagnostics`,
          operation: OPERATION,
          variant: 'pinkse-kp-pinkse-slade',
          // contract-allow: single-invocation -- GPU_PROBIT_SPATIAL_DIAGNOSTIC_FIELD_COUNT = 6 diagnostic fields and fixed scalar arithmetic; no row loop.
          invocationCount: 1,
          bindings: [
            binding('state', state),
            ...failureStatuses.map(([, view], index) =>
              binding(`weightStatus${index}`, view, 'read', 'u32')
            ),
            binding('modelStatus', props.output.status, 'read', 'u32'),
            binding('tests', diagnostics.output.tests, 'read_write'),
            binding('diagnosticStatus', diagnostics.output.status, 'read_write', 'u32')
          ],
          declarations: `${GET_NAN_WGSL}\n${ERFC_VALUE_WGSL}\n${stateWGSL}`,
          body: `diagnosticStatus[diagnosticStatusOffset] = 1u; ${Array.from({length: GPU_PROBIT_SPATIAL_DIAGNOSTIC_FIELD_COUNT}, (_, slot) => `tests[testsOffset + ${slot}u] = getNaN();`).join(' ')} if (modelStatus[modelStatusOffset] != 0u) { return; }
${failureStatuses.map(([name], index) => `if (weightStatus${index}[weightStatus${index}Offset] ${getProbitStatusFailureComparison(name)} 0u) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }`).join('\n')}
if (stateAt(S_DIAGNOSTICS + 7u) != 0.0) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }
let count = stateAt(S_DIAGNOSTICS); var information = stateAt(S_DIAGNOSTICS + 1u); let trace = stateAt(S_DIAGNOSTICS + 2u); let varianceTrace = stateAt(S_DIAGNOSTICS + 3u);
let naiveProduct = stateAt(S_DIAGNOSTICS + 4u); let generalizedProduct = stateAt(S_DIAGNOSTICS + 5u); let standardizedProduct = stateAt(S_DIAGNOSTICS + 6u);
if (!(count > 0.0)) { return; } diagnosticStatus[diagnosticStatusOffset] = 3u; information /= count; if (!(trace > 0.0 && varianceTrace > 0.0 && information > 0.0)) { return; }
let pinkse = generalizedProduct * generalizedProduct / (information * information * trace); let kelejian = naiveProduct / sqrt(varianceTrace); let slade = standardizedProduct * standardizedProduct / trace;
tests[testsOffset] = pinkse; tests[testsOffset + 1u] = erfcValue(sqrt(pinkse * 0.5)); tests[testsOffset + 2u] = kelejian; tests[testsOffset + 3u] = erfcValue(abs(kelejian) * 0.7071067811865476); tests[testsOffset + 4u] = slade; tests[testsOffset + 5u] = erfcValue(sqrt(slade * 0.5)); diagnosticStatus[diagnosticStatusOffset] = 0u;`
        })
      );
    }
    if (props.spatialLag) {
      nodes.push(
        ...getProbitSpatialLagNodes(graph, {
          id,
          rows,
          columns,
          predictors: props.predictors,
          modelStatus: props.output.status,
          weights: props.spatialLag.weights,
          output: props.spatialLag.output,
          rowBindings: stateBindings,
          rowDeclarations,
          state,
          meansSlot: layout.means,
          scalesSlot: layout.scales
        })
      );
    }
    return nodes;
  }
}

/** Float32 word offsets of the probit state buffer. */
function getProbitStateLayout(columns: number) {
  const size = columns + 1;
  let length = 0;
  const take = (count: number) => {
    const slot = length;
    length += count;
    return slot;
  };
  return {
    /** Complete-row count, successes, column sums, nonbinary flag. */
    prepare: take(columns + 3),
    /** Centered column sums of squares, then `RAW_EXTREME_TERM_COUNT` order-key maxima a column. */
    variance: take((1 + RAW_EXTREME_TERM_COUNT) * columns),
    /** Lowest and highest raw value of every predictor column (exact). */
    rawMinimum: take(columns),
    rawMaximum: take(columns),
    means: take(columns),
    scales: take(columns),
    /** Status code in progress (as the serial fit), done flag, whether the count is reported. */
    code: take(1),
    done: take(1),
    counted: take(1),
    /** Accepted steps, unrolled iterations entered, current log likelihood and its scratch maximum. */
    iterations: take(1),
    iteration: take(1),
    /**
     * Current log likelihood, the scratch maxima of the initial likelihood pass (the wrong-side
     * bound and the separation terms, unused), then `COLUMN_EXTREME_TERM_COUNT` extremes of
     * every predictor column.
     */
    current: take(2 + SEPARATION_TERM_COUNT + COLUMN_EXTREME_TERM_COUNT * columns),
    noise: take(1),
    /** Largest |linear predictor| of the latest information pass. */
    largest: take(1),
    /** 1 once a direction has provably moved no row towards the wrong side (kept from then on). */
    separating: take(1),
    /**
     * 1 once a candidate direction has moved no row towards the wrong side as far as float32
     * can tell, with the sign of some row undecided (kept from then on): no status 0 after it.
     */
    undecided: take(1),
    /**
     * `SEPARATION_THRESHOLD_COUNT` single-column thresholds, each a predictor column, a value
     * and a sign (0 when unused): the current one, then the data values next to the previous one.
     */
    thresholds: take(3 * SEPARATION_THRESHOLD_COUNT),
    /** A zero written at run time by the initialize node (see `getRoundedDifference`). */
    runtimeZero: take(1),
    estimate: take(size),
    direction: take(size),
    /** The Fisher direction with its rounding taken out: the candidate of the separation test. */
    separationDirection: take(size),
    factor: take(size * size),
    /** Score, packed information, largest |linear predictor|, nonfinite flag. */
    information: take(size + (size * (size + 1)) / 2 + 2),
    /** Likelihood and wrong-side bound of each step, then the maxima of the separation tests. */
    lineSearch: take(2 * LINE_SEARCH_STEP_COUNT + SEPARATION_TERM_COUNT),
    effects: take(2),
    diagnostics: take(8),
    length
  };
}
