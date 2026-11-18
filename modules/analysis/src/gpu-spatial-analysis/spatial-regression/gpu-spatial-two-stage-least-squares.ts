// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {snapshotRegressionConstructionProps} from './regression-construction-snapshot';
import {getRegressionWeightsSuppressionNodes} from './regression-weights-completeness';
import {getRegressionLossViews} from './regression-weights-completeness';

import {getInstrumentalVariablesCovarianceAllocationLengths} from './instrumental-variables-covariance-parameters';

import {getRegressionTwoStageAllocationLengths} from './regression-binding-plans';

import {validateRegressionAllocations} from './spatial-likelihood-allocations';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {GPUSpatialWeightsTranspose} from '../spatial-weights/gpu-spatial-weights-transpose';
import {
  getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PRECISION_WARNING,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
} from './spatial-two-stage-least-squares-parameters';
import {
  getBelsleyConditionNumberWGSL,
  getCholeskyWGSL,
  getSymmetricEigenvalueRangeWGSL
} from './spatial-regression-solve';
import type {GPUSolverStatusPort} from '../../analysis-core/contracts/index';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../../analysis-core/contracts/model-result';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';

import {
  GPUInstrumentalVariablesCovariance,
  type GPUInstrumentalVariablesCovarianceProps
} from './gpu-instrumental-variables-covariance';
import {
  getInstrumentalVariablesCovarianceStatusView,
  getInstrumentalVariablesCovarianceCommandNodeEstimate,
  validateInstrumentalVariablesCovarianceOptions
} from './instrumental-variables-covariance-parameters';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

const OPERATION = 'GPUSpatialTwoStageLeastSquares';
// Fixed regression/refinement stages plus a conservative allowance for the published transpose
// radix sort, scan and gather internals (32-bit CSR keys). Tested against the generated graph.
const SPATIAL_TWO_STAGE_LEAST_SQUARES_BASE_NODE_ALLOWANCE = 512;
const TILE_WORKGROUP_SIZE = 64;
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
// Bounds the serial norm merge to the regression family's 4096-partial merge budget.
const MAXIMUM_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
/** Residual sums per tile: `u'u`, `sum u`, `sum (y - ybar) u`, `sum (y - ybar)^2`. */
const RESIDUAL_SUM_COUNT = 4;
/** Anselin-Kelejian sums besides the `k + 1` products `u'W z`: `u'Wu`, `u'W1`, `S0`, `tr W'W`, `tr WW`. */
const ANSELIN_KELEJIAN_FIXED_SUM_COUNT = 5;
/**
 * A column is constant at float32 resolution when its sum of squares about the true mean is
 * below this fraction of its sum of squares about the stored float32 mean. The first is the
 * second less a correction of the same size for such a column, and both carry the rounding of a
 * tile sum (a few 1e-6 at worst), so their difference is noise below this. A column that really
 * varies loses the fraction only when its deviation is under 0.003 float32 steps of its mean.
 */
const CONSTANT_COLUMN_TOLERANCE = 1e-5;

/** Caller-owned outputs of {@link GPUSpatialTwoStageLeastSquares}. */
export type GPUSpatialTwoStageLeastSquaresOutput = {
  /** One-row solver `code` (0 on success; see the class documentation for failure codes). */
  status: Pick<GPUSolverStatusPort, 'code'>;

  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then the spatial lag `rho` of the response last), each row holding the
   * coefficient, standard error, z statistic and two-sided normal p-value.
   */
  table: GraphDataView<'float32'>;
  /** At least 16 float32 values; see the `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /** Optional per-row structural residuals `u = y - rho W y - X b`; NaN for a failed fit. */
  residuals?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialTwoStageLeastSquares}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, whether `residuals` is present and
 * `tileRowCount`.
 */
export type GPUSpatialTwoStageLeastSquaresProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-two-stage-least-squares'`. */
  id?: string;
  /**
   * Square weights `W`; rows index the regression rows. Any sparsity pattern is accepted,
   * including directed kNN weights; the Anselin-Kelejian test forms `W'` with
   * `GPUSpatialWeightsTranspose`.
   */
  weights: GPUSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /**
   * Number of predictor columns, compile-time, 1 to 8 with `instrumentOrder: 1` and 1 to 5 with
   * `instrumentOrder: 2` (`(order + 1) * predictorCount <= 16`, the 16-wide Cholesky of the
   * instrument system; the constant is centered out). The intercept is implicit.
   */
  predictorCount: number;
  /**
   * Rows reduced by one invocation of the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`.
   */
  tileRowCount?: number;
  /**
   * Instrument order (spreg `w_lags`), compile-time: 1 (the default) instruments `W y` with
   * `[1, X, W X]`, 2 with `[1, X, W X, W^2 X]`, where `W^2 X = W (W X)` is evaluated per row from
   * the two-hop neighborhood, so no extra storage buffers are bound. The instrument count is
   * `(order + 1) * predictorCount` columns.
   */
  instrumentOrder?: 1 | 2;
  /** Optional robust inference; point estimates and classical diagnostics remain unchanged.
   * HAC kernel is separate from model weights. The returned covariance is certified PSD to 1e-5
   * relative tolerance. HC2/HC3 use projected-design leverage; CLUSTER uses occupied-group CR1.
   * See GPUInstrumentalVariablesCovariance for formulas, limits and covariance status codes.
   * The robust status is separate from the point-estimate solver status. */
  robust?: {
    method: NonNullable<GPUInstrumentalVariablesCovarianceProps['method']>;
    kernel?: GPUSpatialWeights;
    clusters?: GraphDataView<'uint32'>;
    clusterCount?: number;
    output: GPUInstrumentalVariablesCovarianceProps['output'];
  };
  /** Caller-owned outputs. */
  output: GPUSpatialTwoStageLeastSquaresOutput;
};

/**
 * Spatial two-stage least squares for the lag model `y = rho W y + X b + e` (spreg `GM_Lag` with
 * `w_lags = 1`). The endogenous regressor `W y` is
 * instrumented with `[1, X, W X]` (the lag of every predictor, never of the intercept).
 *
 * With `Z = [1, X, W y]`, `H = [1, X, W X]` and `P = H (H'H)^-1 H'` the estimate is
 * `delta = (Z'PZ)^-1 Z'P y`, the structural residuals are `u = y - Z delta`, `sigma2 = u'u / n`
 * (spreg `sig2n`, its `GM_Lag` default), and the covariance is `sigma2 (Z'PZ)^-1`. z statistics and
 * p-values are standard normal (spreg reports z for `GM_Lag`).
 *
 * Because `1` is in both `Z` and `H`, the problem is solved on columns centered by their means
 * (Frisch-Waugh), which removes the cancellation of raw moments; the intercept is recovered as
 * `ybar - mean(Z)' delta` with variance `sigma2 / n + mean(Z)' Var(delta) mean(Z)`.
 * The stored means are float32, so the centered columns do not sum to zero: the moment pass also
 * sums the deviations of the regressors and of the response from their stored means, and the
 * residual pass subtracts what they leave (`(sum (y - ybar) - sum (Z - mean) delta) / n`), so the
 * structural residuals sum to zero. Every difference of a value and its mean is fixed with
 * `getRoundedDifference`, because the shader compiler of the test GPU otherwise regroups the
 * residual around the mean. Without both, residuals of 0.002 around a response mean of 10,000
 * carried a common offset of 0.0009: `sigma2` 19.6% high and an Anselin-Kelejian statistic of 16
 * against 2.
 *
 * The lag of the response is formed about the response mean: the regressor column is
 * `v = W (y - c) + c (s_i - s)`, with `c` the float32 response mean, `s_i` the exact sum of the
 * float32 weights of row `i` (`getAdditionRoundingError`) and `s` the largest row sum, so that
 * `W y = v + c s`; the constant `c s` belongs to the intercept and to the regressor mean. A sum
 * of `w * y` over the raw response is rounded to a float32 step of the response: for residuals
 * of 0.002 to 0.004 around 10,000 that was up to 4.2% of `sigma2` and 45% of the
 * Anselin-Kelejian statistic.
 *
 * One step of iterative refinement follows the solve. The float32 solve lands a few float32
 * steps of `rho` from the estimate (3 and 5 steps on the scenes below). Where the lag differs
 * between rows by the response mean (rows without neighbours, row sums that differ) the
 * two-stage `rho` is not at the minimum of `u'u`, `sigma2` moves by
 * `2 (rho - rho_ML) var(W y) / sigma2` per unit of `rho`, and those few steps were 0.8% and 3%
 * of `sigma2`; the float32 value nearest to the estimate would have cost 0.03% and 0.07%. The
 * refinement pass forms the residuals at the float32 `delta` without the rounding of their
 * products (`getStructuralResidual`: each product and each subtraction keeps its rounding
 * error, and the lag enters in its two parts), sums `H'u`, and solves the same system for the
 * correction `(Z'PZ)^-1 Z'P u`. The correction is kept beside `delta`, not added to it: the
 * residuals, `sigma2`, the standard errors and the Anselin-Kelejian statistic are those of
 * `delta + correction`, and `table` publishes the float32 nearest to it.
 *
 * What float32 leaves (measured on the test GPU against spreg and a centered float64 solve,
 * `math-review-lag-conditioning.spec.ts`), for residuals of 0.002 to 0.004 around 10,000: rows
 * that share a row sum, exactly or up to the rounding of the weights: `sigma2` within 2.8e-6
 * and the statistic within 2.4e-4 for `rho` from 0.2 to 0.8; three rows without neighbours:
 * 9.2e-6 and 9.1e-6; row sums of 2 and 3: 9.5e-7 and 4.9e-7. Slopes two hundred times the
 * residual deviation: `sigma2` within 1.6e-4. `W X` is still summed from the raw predictors
 * (instruments only).
 *
 * The moments are sums about the stored float32 means, corrected to moments about the true
 * means by `n` times the product of two mean remainders, as in `GPUOrdinaryLeastSquares`. A
 * predictor that is constant at float32 resolution (its corrected sum of squares below 1e-5 of
 * the uncorrected one) is collinear with the intercept: status 1, as the other estimators
 * report it. Without the correction a constant such as 12345.678, whose float32 mean over 600
 * rows is not the constant, was fitted with status 0.
 * The accumulation is the same fixed-order tile and pairwise-merge scheme as
 * `GPUOrdinaryLeastSquares`: tile moments `H'H`, `H'Z`, `H'y` and `diag(Z'Z)`, a single-invocation
 * solve (Cholesky of the scaled `H'H`, then of the scaled `Z'PZ`), a per-row residual pass and a
 * finish kernel for the table. Nothing is read back.
 *
 * The Anselin-Kelejian test of the structural residuals (spreg `AKtest(case='gen')`) is written to
 * `summary`: `AK = n I^2 / phi2` with `I = (n / S0) u'Wu / u'u`,
 * `phi2 = (T + 4 a / sigma2) / ((S0 / n)^2 n)` and `a = (u'WZ) (Z'PZ)^-1 (u'WZ)'`. `W'u` is summed
 * over the rows of the transposed CSR, so the sparsity pattern of `weights` may be asymmetric
 * (`T = sum w_ij^2 + sum w_ij w_ji` only involves pairs present in both directions). The total
 * weight `S0` cancels from the statistic, which is computed as
 * `(u'Wu / sigma2)^2 / (T + 4 a / sigma2)`: signed weights whose total is zero have a defined
 * statistic (it was NaN) and an undefined Moran's I, written as NaN.
 *
 * `instrumentOrder: 2` adds `W^2 X` to the instruments (`w_lags = 2`). By default standard
 * errors use classical covariance. `robust` replaces table standard errors, z statistics and
 * normal p-values using HC0/HC1/HC2/HC3, clustered CR1 or spatial HAC covariance and exposes the full covariance.
 * Existing summary diagnostics retain their classical definitions; robust covariance does not
 * make the classical Anselin-Kelejian reference distribution robust to heteroskedasticity.
 *
 * Diagnostics written to `summary` (none changes `status`):
 * - Conditioning of the second-stage system `Z'PZ` (Belsley condition number, correlation
 *   condition number, largest variance inflation factor) and of the instrument system `H'H`
 *   (correlation condition number), with a precision warning when either correlation condition
 *   number exceeds `GPU_REGRESSION_PRECISION_CONDITION_LIMIT` (1e4).
 * - Admissibility of `rho`: `||W||_inf` (largest absolute row sum) and a flag for
 *   `|rho| ||W||_inf >= 1`. The estimate is reported either way: 2SLS is unconstrained, and the
 *   flag is information for the reader (impacts and the reduced form need `|rho| ||W||_inf < 1`),
 *   not a numerical failure.
 * - First-stage strength of `W y` on the instruments: the classical joint F statistic and the
 *   partial R squared of the excluded instruments (`W X`, `W^2 X`), the quantities to compare with
 *   the weak-instrument rule of thumb `F < 10`.
 *
 * Capacity: with the constant centered out the instrument solve factors `(order + 1) k` columns
 * in a 16-wide Cholesky, so `k <= 8` for `instrumentOrder: 1` and `k <= 5` for `instrumentOrder:
 * 2` (a previous version accepted `k` up to 8 with order 2 and failed when building the graph).
 * No kernel binds more than 8 storage buffers.
 */
export class GPUSpatialTwoStageLeastSquares implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialTwoStageLeastSquaresProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;
  /** Instrument order, 1 or 2. */
  readonly instrumentOrder: 1 | 2;
  /** Upper bound including robust covariance and published transpose/sort internals. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUSpatialTwoStageLeastSquaresProps) {
    props = snapshotRegressionConstructionProps(
      props,
      [
        'id',
        'weights',
        'predictors',
        'response',
        'predictorCount',
        'tileRowCount',
        'instrumentOrder',
        'robust',
        'output'
      ],
      ['table', 'summary', 'residuals', 'status'],
      'spatial-two-stage-least-squares'
    );
    const id = props.id ?? 'spatial-two-stage-least-squares';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    const instrumentOrder = props.instrumentOrder ?? 1;
    if (instrumentOrder !== 1 && instrumentOrder !== 2) {
      throw new Error(`${id} instrumentOrder must be 1 or 2`);
    }
    const maximumPredictorCount =
      getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(instrumentOrder);
    if (
      !Number.isInteger(predictorCount) ||
      predictorCount < 1 ||
      predictorCount > maximumPredictorCount
    ) {
      throw new Error(
        `${id} predictorCount must be an integer in [1, ${maximumPredictorCount}] with instrumentOrder ${instrumentOrder}`
      );
    }
    const rows = validateGPUSpatialWeights(id, props.weights);
    validateSinglePackedView(props.predictors, ['float32'], `${id} predictors`);
    validateSinglePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== rows) {
      throw new Error(`${id} response length must equal the weights row count`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal the weights row count * predictorCount`);
    }
    validateSinglePackedView(output.table, ['float32'], `${id} output.table`);
    if (
      output.table.length <
      (predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
    ) {
      throw new Error(`${id} output.table must hold (predictorCount + 2) * 4 float32 values`);
    }
    validateSinglePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} float32 values`
      );
    }
    validateSinglePackedView(output.status.code, ['uint32'], `${id} output.status.code`);
    if (output.status.code.length < 1) {
      throw new Error(`${id} output.status.code must hold one uint32 row`);
    }
    if (output.residuals) {
      validateSinglePackedView(output.residuals, ['float32'], `${id} output.residuals`);
      if (output.residuals.length < rows) {
        throw new Error(`${id} output.residuals must hold one row per input row`);
      }
    }
    if (props.robust) {
      const robust = props.robust;
      validateInstrumentalVariablesCovarianceOptions(id, rows, robust);
      if (robust.kernel)
        validateRequirement(
          id,
          validateGPUSpatialWeights(id, robust.kernel) === rows,
          'robust.kernel row count must equal response.length'
        );
      const robustStatus = getInstrumentalVariablesCovarianceStatusView(robust.output.status);
      for (const [name, view, format, length] of [
        [
          'robust.output.covariance',
          robust.output.covariance,
          'float32',
          (predictorCount + 2) ** 2
        ],
        ['robust.output.status.code', robustStatus, 'uint32', 1],
        ...(robust.clusters ? [['robust.clusters', robust.clusters, 'uint32', rows] as const] : [])
      ] as const) {
        validateSinglePackedView(view, [format], `${id} ${name}`);
        validateRequirement(id, view.length === length, `${name} length must equal ${length}`);
      }
    }
    this.instrumentOrder = instrumentOrder;
    if (props.tileRowCount !== undefined) {
      validateCapacity(id, 'tileRowCount', props.tileRowCount);
    }
    this.estimatedCommandNodeCount =
      SPATIAL_TWO_STAGE_LEAST_SQUARES_BASE_NODE_ALLOWANCE +
      (props.robust
        ? 2 +
          getInstrumentalVariablesCovarianceCommandNodeEstimate(
            predictorCount + 1,
            Boolean(props.robust.clusters)
          )
        : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.rowCount = rows;
    this.tileRowCount =
      props.tileRowCount ??
      Math.max(MINIMUM_DEFAULT_TILE_ROWS, Math.ceil(rows / MAXIMUM_DEFAULT_TILE_COUNT));
    this.tileCount = Math.ceil(rows / this.tileRowCount);
    validateMaximumCount(id, 'tileCount', this.tileCount, MAXIMUM_TILE_COUNT);
    validateRegressionOutputDispatches(
      this.id,
      {
        ...output,
        status: undefined,
        code: output.status.code,
        covariance: props.robust?.output.covariance,
        covarianceStatus: props.robust
          ? getInstrumentalVariablesCovarianceStatusView(props.robust.output.status)
          : undefined
      },
      [
        ['table', 'summary', 'code'],
        ['covariance', 'covarianceStatus'],
        {read: ['covariance', 'covarianceStatus'], write: ['table']}
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.table,
        output.summary,
        output.status.code,
        output.residuals,
        props.robust?.output.covariance,
        props.robust
          ? getInstrumentalVariablesCovarianceStatusView(props.robust.output.status)
          : undefined
      ],
      [
        props.predictors,
        props.response,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        ...getRegressionLossViews(props.weights),
        props.robust?.kernel?.offsets,
        props.robust?.kernel?.neighbors,
        props.robust?.kernel?.weights,
        props.robust?.kernel?.status?.overflow,
        props.robust?.clusters
      ]
    );
  }

  /**
   * Returns the nodes in order: weights-norm tiles and merge (`||W||_inf`, and the response mean
   * and row sum the lag is formed about), mean tiles and merge, moment tiles and merge, the
   * solve, the refinement tiles, merge and solve, the residual tiles and merge, the transpose of
   * `W`, the `W'u` pass, the two
   * Anselin-Kelejian tile kernels and merge, and the finish that writes `table`, `summary` and
   * `status`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount, instrumentOrder} = this;
    const {weights, output, predictorCount: k} = props;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      offsets: weights.offsets,
      neighbors: weights.neighbors,
      weights: weights.weights,
      ...Object.fromEntries(
        getRegressionLossViews(weights).map((view, index) => [`weights loss ${index}`, view])
      ),
      table: output.table,
      summary: output.summary,
      code: output.status.code,
      residuals: output.residuals,
      robustCovariance: props.robust?.output.covariance,
      robustStatus: props.robust
        ? getInstrumentalVariablesCovarianceStatusView(props.robust.output.status)
        : undefined,
      robustClusters: props.robust?.clusters,
      robustKernelOffsets: props.robust?.kernel?.offsets,
      robustKernelNeighbors: props.robust?.kernel?.neighbors,
      robustKernelWeights: props.robust?.kernel?.weights,
      robustKernelOverflow: props.robust?.kernel?.status?.overflow
    });
    validateRegressionAllocations(
      graph.device,
      id,
      getRegressionTwoStageAllocationLengths(
        rowCount,
        k,
        weights.neighbors.length,
        instrumentOrder,
        tileRowCount
      )
    );
    if (props.robust)
      validateRegressionAllocations(graph.device, id, {
        'robust design': rowCount * (k + 1),
        'robust instruments': rowCount * (instrumentOrder + 1) * k,
        ...getInstrumentalVariablesCovarianceAllocationLengths(
          rowCount,
          k + 1,
          (instrumentOrder + 1) * k,
          props.robust
        )
      });
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.table,
      output.summary,
      output.status.code,
      output.residuals
    ]);
    const m = (instrumentOrder + 1) * k;
    const q = k + 1;
    const momentCount = m * m + m * q + m + q;
    // After the moments: the sums of the deviations of the regressors (q), of the response and of
    // the lagged instruments (m - k) from their stored float32 means, which do not sum to zero.
    const momentStride = momentCount + q + 1 + (m - k);
    // Means: x (k), W x (k), the lag column of y, y, then W^2 x (k) for instrument order 2.
    const meanCount = (instrumentOrder + 1) * k + 2;
    // Workspace: the means, then delta (q), then ||W||_inf, the residual offset (what the stored
    // means miss of the true means, carried through the fit), a zero written at run time (the
    // runtime zero of getRoundedDifference), the response mean the lag is formed about and the
    // row sum it is formed about (W y = lag column + center * row sum), then the refinement of
    // delta (q) and of the residual offset, which are kept beside the float32 values.
    const workspaceLength = meanCount + 2 * q + 6;
    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const partialMeans = transient('partial-means', tileCount * meanCount);
    const partialMoments = transient('partial-moments', tileCount * momentStride);
    const moments = transient('moments', momentStride);
    const workspace = transient('workspace', workspaceLength);
    const partialResiduals = transient('partial-residuals', tileCount * RESIDUAL_SUM_COUNT);
    const residualSums = transient('residual-sums', RESIDUAL_SUM_COUNT);
    // Refinement sums: H'u (m) and the sum of u, of the residuals at the float32 delta.
    const partialRefinement = transient('partial-refinement', tileCount * (m + 1));
    const refinement = transient('refinement-sums', m + 1);
    const akCount = ANSELIN_KELEJIAN_FIXED_SUM_COUNT + q;
    // Per tile: the largest absolute row sum, the largest row sum and the sum of the response.
    const partialNorm = transient('partial-weights-norm', tileCount * 3);
    const partialAk = transient('partial-anselin-kelejian', tileCount * akCount);
    const akSums = transient('anselin-kelejian-sums', akCount);
    const residualView = output.residuals ?? transient('residuals', rowCount);
    const transposed: GPUSpatialWeights = {
      offsets: createTransientView(graph, `${id}-transpose-offsets`, 'uint32', rowCount + 1),
      neighbors: createTransientView(
        graph,
        `${id}-transpose-neighbors`,
        'uint32',
        weights.neighbors.length
      ),
      weights: transient('transpose-weights', weights.neighbors.length)
    };
    const transposedLagResidual = transient('transposed-lag-residual', rowCount);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const ORDER: u32 = ${instrumentOrder}u;
const M: u32 = ${m}u;
const Q: u32 = ${q}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const MEAN_COUNT: u32 = ${meanCount}u;
const MOMENTS: u32 = ${momentCount}u;
const MOMENT_STRIDE: u32 = ${momentStride}u;
const W_DELTA: u32 = ${meanCount}u;
const W_NORM: u32 = ${meanCount + q}u;
const W_RESIDUAL_OFFSET: u32 = ${meanCount + q + 1}u;
const W_RUNTIME_ZERO: u32 = ${meanCount + q + 2}u;
const W_LAG_CENTER: u32 = ${meanCount + q + 3}u;
const W_ROW_SUM: u32 = ${meanCount + q + 4}u;
const W_CORRECTION: u32 = ${meanCount + q + 5}u;
const W_CORRECTION_OFFSET: u32 = ${meanCount + 2 * q + 5}u;`;
    const mergeLoop = (stride: number) =>
      getPairwiseSumWGSL({
        count: 'TILE_COUNT',
        term: `partial[partialOffset + sumItem * ${stride}u + index]`
      });
    const f32 = (name: string, view: GraphDataView, access: 'read' | 'read_write') =>
      ({name, view, type: 'f32', access}) as WGSLKernelBinding;
    const rowBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      f32('weights', weights.weights, 'read'),
      f32('design', props.predictors, 'read'),
      f32('response', props.response, 'read')
    ];
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      declarations: string,
      body: string,
      workgroupSize?: number
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name,
        bindings,
        invocationCount,
        declarations,
        body,
        ...(workgroupSize ? {workgroupSize} : {})
      });
    const lagDeclarations = `${constants}
${getLagWGSL()}`;
    const rowDeclarations = `${lagDeclarations}
${getCenteredRowWGSL()}`;
    const residualDeclarations = `${rowDeclarations}
${getStructuralResidualWGSL()}`;
    const transposeNodes = new GPUSpatialWeightsTranspose({
      id: `${id}-transpose`,
      weights,
      output: transposed
    }).getCommandNodes(graph);
    const nodes: GPUCommandNode<Parameters>[] = [
      // ||W||_inf, the largest absolute row sum (the admissible interval of rho is
      // |rho| ||W||_inf < 1), and the two constants the lag of the response is formed about: the
      // response mean and the largest row sum (see getLags).
      kernel(
        'weights-norm-tiles',
        [
          rowBindings[0],
          rowBindings[1],
          rowBindings[2],
          rowBindings[4],
          f32('partial', partialNorm, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  var largestRowSum = -3.0e38;
  var responseSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var absoluteSum = 0.0;
    var rowSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] < ROWS) {
        absoluteSum += abs(weights[weightsOffset + slot]);
        rowSum += weights[weightsOffset + slot];
      }
    }
    largest = max(largest, absoluteSum);
    largestRowSum = max(largestRowSum, rowSum);
    responseSum += response[responseOffset + row];
  }
  partial[partialOffset + 3u * index] = largest;
  partial[partialOffset + 3u * index + 1u] = largestRowSum;
  partial[partialOffset + 3u * index + 2u] = responseSum;`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'weights-norm-merge',
        [f32('partial', partialNorm, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        constants,
        `var largest = 0.0;
  var largestRowSum = -3.0e38;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partial[partialOffset + 3u * tile]);
    largestRowSum = max(largestRowSum, partial[partialOffset + 3u * tile + 1u]);
  }
  ${getPairwiseSumWGSL({
    count: 'TILE_COUNT',
    term: 'partial[partialOffset + 3u * sumItem + 2u]'
  })}
  workspace[workspaceOffset + W_NORM] = largest;
  workspace[workspaceOffset + W_ROW_SUM] = largestRowSum;
  workspace[workspaceOffset + W_LAG_CENTER] = sum / f32(ROWS);
  // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
  workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;`
      ),
      kernel(
        'means-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('partial', partialMeans, 'read_write')
        ],
        tileCount,
        lagDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${meanCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${k}>;
    var twiceLagged: array<f32, ${k}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, ORDER == 2u);
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
      sums[K + column] = sums[K + column] + lagged[column];
      if (ORDER == 2u) {
        sums[2u * K + 2u + column] = sums[2u * K + 2u + column] + twiceLagged[column];
      }
    }
    sums[2u * K] = sums[2u * K] + lagResponse;
    sums[2u * K + 1u] = sums[2u * K + 1u] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < MEAN_COUNT; slot++) {
    partial[partialOffset + index * MEAN_COUNT + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'means-merge',
        [f32('partial', partialMeans, 'read'), f32('workspace', workspace, 'read_write')],
        meanCount,
        constants,
        `${mergeLoop(meanCount)}
  workspace[workspaceOffset + index] = sum / f32(ROWS);
  if (index == 0u) {
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    workspace[workspaceOffset + W_RUNTIME_ZERO] = 0.0;
  }`
      ),
      kernel(
        'moments-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('partial', partialMoments, 'read_write')
        ],
        tileCount,
        rowDeclarations,
        `let base = partialOffset + index * MOMENT_STRIDE;
  for (var slot = 0u; slot < MOMENT_STRIDE; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${m}>;
    var z: array<f32, ${q}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, true);
    for (var a = 0u; a < M; a++) {
      for (var b = 0u; b < M; b++) {
        partial[base + a * M + b] += h[a] * h[b];
      }
      for (var b = 0u; b < Q; b++) {
        partial[base + M * M + a * Q + b] += h[a] * z[b];
      }
      partial[base + M * M + M * Q + a] += h[a] * centeredResponse;
    }
    for (var b = 0u; b < Q; b++) {
      partial[base + M * M + M * Q + M + b] += z[b] * z[b];
      partial[base + MOMENTS + b] += z[b];
    }
    partial[base + MOMENTS + Q] += centeredResponse;
    // The lagged instruments; the first K instruments are the first K regressors.
    for (var a = K; a < M; a++) {
      partial[base + MOMENTS + Q + 1u + (a - K)] += h[a];
    }
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'moments-merge',
        [f32('partial', partialMoments, 'read'), f32('moments', moments, 'read_write')],
        momentStride,
        constants,
        `${mergeLoop(momentStride)}
  moments[momentsOffset + index] = sum;`
      ),
      kernel(
        'solve',
        [f32('moments', moments, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        `${constants}
${getSolveWGSL(m, q)}`,
        `var delta: array<f32, ${q}>;
  var inverseDiagonal: array<f32, ${q}>;
  var meanQuadratic = 0.0;
  var unusedVector: array<f32, ${q}>;
  var unusedQuadratic = 0.0;
  var unusedDiagnostics: array<f32, 6>;
  let ok = solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &unusedVector, &unusedQuadratic, &unusedDiagnostics, false);
  // The stored means are float32, so the centered columns do not sum to zero. The residuals of a
  // model with an intercept do: this is the mean they would otherwise carry.
  var residualOffset = moments[momentsOffset + MOMENTS + Q];
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_DELTA + column] = select(getNaN(), delta[column], ok);
    residualOffset -= moments[momentsOffset + MOMENTS + column] * delta[column];
  }
  workspace[workspaceOffset + W_RESIDUAL_OFFSET] = select(getNaN(), residualOffset / f32(ROWS), ok);`
      ),
      // One step of iterative refinement. The float32 solve above lands a few float32 steps of
      // rho from the estimate; where the lag differs between rows by the response mean (rows
      // without neighbours, row sums that differ) the two-stage estimate is not at the minimum
      // of u'u, and those few steps were 0.8% to 3% of sigma2. This pass forms the residuals at
      // the float32 delta without the rounding of their products (getStructuralResidual) and
      // sums H'u; the refine node solves the same system for the correction of delta.
      kernel(
        'refinement-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('partial', partialRefinement, 'read_write')
        ],
        tileCount,
        residualDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${m + 1}>;
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${m}>;
    var z: array<f32, ${q}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, true);
    let value = getStructuralResidual(&z, centeredResponse);
    let residual = value.x + value.y;
    for (var a = 0u; a < M; a++) {
      sums[a] += h[a] * residual;
    }
    sums[M] += residual;
  }
  for (var slot = 0u; slot < ${m + 1}u; slot++) {
    partial[partialOffset + index * ${m + 1}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'refinement-merge',
        [f32('partial', partialRefinement, 'read'), f32('refinement', refinement, 'read_write')],
        m + 1,
        constants,
        `${mergeLoop(m + 1)}
  refinement[refinementOffset + index] = sum;`
      ),
      // correction = (Z'PZ)^-1 Z'P u with the factors of the solve. It is kept beside delta, not
      // added to it: a float32 delta cannot hold it (it is a few 1e-8 of rho), and sigma2, the
      // standard errors and the Anselin-Kelejian statistic are those of delta + correction.
      kernel(
        'refine',
        [
          f32('moments', moments, 'read'),
          f32('refinement', refinement, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${getSolveWGSL(m, q, 'refinement')}`,
        `var correction: array<f32, ${q}>;
  var inverseDiagonal: array<f32, ${q}>;
  var meanQuadratic = 0.0;
  var unusedVector: array<f32, ${q}>;
  var unusedQuadratic = 0.0;
  var unusedDiagnostics: array<f32, 6>;
  let ok = solveSystem(&correction, &inverseDiagonal, &meanQuadratic, &unusedVector, &unusedQuadratic, &unusedDiagnostics, false);
  // The corrected residuals sum to zero: the mean of u less the mean of (Z - mean) correction.
  var offsetCorrection = refinement[refinementOffset + M];
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_CORRECTION + column] = select(getNaN(), correction[column], ok);
    offsetCorrection -= moments[momentsOffset + MOMENTS + column] * correction[column];
  }
  workspace[workspaceOffset + W_CORRECTION_OFFSET] = select(getNaN(), offsetCorrection / f32(ROWS), ok);`
      ),
      kernel(
        'residuals-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('residualsOut', residualView, 'read_write'),
          f32('partial', partialResiduals, 'read_write')
        ],
        tileCount,
        residualDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${RESIDUAL_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${m}>;
    var z: array<f32, ${q}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, false);
    // The residual at delta (a float32 value and its correction), then the refinement.
    let value = getStructuralResidual(&z, centeredResponse);
    var lowPart = value.y - workspace[workspaceOffset + W_CORRECTION_OFFSET];
    for (var b = 0u; b < Q; b++) {
      lowPart = lowPart - z[b] * workspace[workspaceOffset + W_CORRECTION + b];
    }
    let residual = value.x + lowPart;
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += residual * residual;
    sums[1] += residual;
    sums[2] += centeredResponse * residual;
    sums[3] += centeredResponse * centeredResponse;
  }
  for (var slot = 0u; slot < ${RESIDUAL_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${RESIDUAL_SUM_COUNT}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'residuals-merge',
        [f32('partial', partialResiduals, 'read'), f32('residualSums', residualSums, 'read_write')],
        RESIDUAL_SUM_COUNT,
        constants,
        `${mergeLoop(RESIDUAL_SUM_COUNT)}
  residualSums[residualSumsOffset + index] = sum;`
      ),
      ...transposeNodes,
      kernel(
        'anselin-kelejian-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('residuals', residualView, 'read'),
          f32('partial', partialAk, 'read_write')
        ],
        tileCount,
        lagDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${akCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    let residual = residuals[residualsOffset + row];
    var rowSum = 0.0;
    var lagResidual = 0.0;
    var squareSum = 0.0;
    var crossSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      // Reverse weight w_ji by binary search of this row in the neighbor's sorted row.
      var reverse = 0.0;
      var low = offsets[offsetsOffset + neighbor];
      var high = offsets[offsetsOffset + neighbor + 1u];
      while (low < high) {
        let middle = (low + high) / 2u;
        let candidate = neighbors[neighborsOffset + middle];
        if (candidate == row) {
          reverse = weights[weightsOffset + middle];
          break;
        }
        if (candidate < row) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      let neighborResidual = residuals[residualsOffset + neighbor];
      rowSum += weight;
      lagResidual += weight * neighborResidual;
      squareSum += weight * weight;
      crossSum += weight * reverse;
    }
    sums[0] += residual * lagResidual;
    sums[1] += residual * rowSum;
    sums[2] += rowSum;
    sums[3] += squareSum;
    sums[4] += crossSum;
  }
  // The u'W z products (slots after the fixed ones) come from the next kernel.
  for (var slot = 0u; slot < ${ANSELIN_KELEJIAN_FIXED_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${akCount}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      // (W'u)_i over the transposed CSR, so one-way links are included.
      kernel(
        'transposed-lag-residual',
        [
          {name: 'offsets', view: transposed.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: transposed.neighbors, type: 'u32', access: 'read'},
          f32('weights', transposed.weights, 'read'),
          f32('residuals', residualView, 'read'),
          f32('lagOut', transposedLagResidual, 'read_write')
        ],
        rowCount,
        constants,
        `var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += weights[weightsOffset + slot] * residuals[residualsOffset + neighbors[neighborsOffset + slot]];
  }
  lagOut[lagOutOffset + index] = sum;`
      ),
      kernel(
        'anselin-kelejian-products',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('transposedLagResidual', transposedLagResidual, 'read'),
          f32('partial', partialAk, 'read_write')
        ],
        tileCount,
        lagDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${q}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${k}>;
    var twiceLagged: array<f32, ${k}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, false);
    let product = transposedLagResidual[transposedLagResidualOffset + row];
    // Differences from the means, rounded as written so the tile sums are not regrouped.
    let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
    for (var column = 0u; column < K; column++) {
      let centered = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + column], runtimeZero);
      sums[column] += product * centered;
    }
    sums[K] += product * getRoundedDifference(lagResponse, workspace[workspaceOffset + 2u * K], runtimeZero);
  }
  for (var column = 0u; column < Q; column++) {
    partial[partialOffset + index * ${akCount}u + ${ANSELIN_KELEJIAN_FIXED_SUM_COUNT}u + column] = sums[column];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'anselin-kelejian-merge',
        [f32('partial', partialAk, 'read'), f32('akSums', akSums, 'read_write')],
        akCount,
        constants,
        `${mergeLoop(akCount)}
  akSums[akSumsOffset + index] = sum;`
      ),
      kernel(
        'finish',
        [
          f32('moments', moments, 'read'),
          f32('workspace', workspace, 'read'),
          f32('residualSums', residualSums, 'read'),
          f32('akSums', akSums, 'read'),
          f32('tableOut', output.table, 'read_write'),
          f32('summaryOut', output.summary, 'read_write'),
          {name: 'statusOut', view: output.status.code, type: 'u32', access: 'read_write'}
        ],
        1,
        `${constants}
${getSolveWGSL(m, q)}
${getFinishWGSL(k)}`,
        'finish();'
      )
    ];
    if (props.robust) {
      const design = transient('robust-design', rowCount * q);
      const instruments = transient('robust-instruments', rowCount * m);
      nodes.push(
        kernel(
          'robust-inputs',
          [
            ...rowBindings,
            f32('workspace', workspace, 'read'),
            f32('robustDesign', design, 'read_write'),
            f32('robustInstruments', instruments, 'read_write')
          ],
          rowCount,
          lagDeclarations,
          `var lagged: array<f32, ${k}>; var twiceLagged: array<f32, ${k}>;
let lagResponse = getLags(index, &lagged, &twiceLagged, ORDER == 2u);
for (var column = 0u; column < K; column++) {
 let value = design[designOffset + index * K + column];
 robustDesign[robustDesignOffset + index * Q + column] = value;
 robustInstruments[robustInstrumentsOffset + index * M + column] = value;
 robustInstruments[robustInstrumentsOffset + index * M + K + column] = lagged[column];
 if (ORDER == 2u) { robustInstruments[robustInstrumentsOffset + index * M + 2u * K + column] = twiceLagged[column]; }
}
// The raw lag W y: the lag column plus the constant it was formed about.
robustDesign[robustDesignOffset + index * Q + K] = lagResponse + workspace[workspaceOffset + W_LAG_CENTER] * workspace[workspaceOffset + W_ROW_SUM];`
        )
      );
      nodes.push(
        ...new GPUInstrumentalVariablesCovariance({
          id: `${id}-robust`,
          design,
          coefficientCount: q,
          instruments,
          instrumentCount: m,
          residuals: residualView,
          ...props.robust
        }).getCommandNodes(graph)
      );
      nodes.push(
        kernel(
          'robust-table',
          [
            f32('covariance', props.robust.output.covariance, 'read'),
            {
              name: 'robustStatus',
              view: getInstrumentalVariablesCovarianceStatusView(props.robust.output.status),
              type: 'u32',
              access: 'read'
            },
            f32('table', output.table, 'read_write')
          ],
          q + 1,
          `${GET_NAN_WGSL}\n${ERFC_VALUE_WGSL}`,
          `let variance = covariance[covarianceOffset + index * ${q + 2}u];
var standardError = getNaN();
if (robustStatus[robustStatusOffset] == 0u && variance >= 0.0) { standardError = sqrt(variance); }
let statistic = table[tableOffset + index * 4u] / standardError;
table[tableOffset + index * 4u + 1u] = standardError;
table[tableOffset + index * 4u + 2u] = statistic;
table[tableOffset + index * 4u + 3u] = erfcValue(abs(statistic) * 0.70710678);`
        )
      );
    }
    return [
      ...nodes,
      ...getRegressionWeightsSuppressionNodes<Parameters>(
        graph,
        id,
        props.weights,
        props.output,
        props.robust?.output
      )
    ];
  }
}

/**
 * `getLags`: `W x` for every predictor and the lag column of `y` in one slot-order loop over the
 * row. With `needSecond`, also `W^2 x = W (W x)`, summed over the two-hop neighborhood in slot
 * order. Kernels that include it bind `workspace` (the lag center, the row sum and the runtime
 * zero, written by the weights-norm merge).
 *
 * The lag column is `v = W (y - c) + c (s_i - s)`, so that `W y = v + c s`: `c` is the float32
 * response mean, `s_i` the exact sum of the float32 weights of the row and `s` the largest row
 * sum. A sum of `w * y` is rounded to a float32 step of the response (0.001 near 10,000), which
 * for residuals of 0.002 was up to 4% of the residual variance and 45% of the Anselin-Kelejian
 * statistic; the differences `y - c` are exact or rounded once. The constant `c s` belongs to
 * the intercept (`regressorMean` adds it back). `getAdditionRoundingError` returns what each
 * float32 addition of the row sum drops: three weights of float32(1/3) sum to 1.0000000298, and
 * 3e-8 of 10,000 is a tenth of such a residual.
 */
function getLagWGSL(): string {
  return /* wgsl */ `
${ROUNDED_DIFFERENCE_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
// The parts of the lag column of the last getLags call: W (y - c), and s_i - s as a rounded
// difference and what the row sum dropped. getStructuralResidual reads them, because their sum
// times c is rounded to a float32 step of the response mean when the row sums differ.
var<private> lagCenteredPart: f32;
var<private> lagRowSumDeviationHigh: f32;
var<private> lagRowSumDeviationLow: f32;

fn getLags(
  row: u32,
  lagged: ptr<function, array<f32, K>>,
  twiceLagged: ptr<function, array<f32, K>>,
  needSecond: bool
) -> f32 {
  let lagCenter = workspace[workspaceOffset + W_LAG_CENTER];
  let lagRuntimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var lagResponse = 0.0;
  var rowSum = 0.0;
  var rowSumError = 0.0;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    lagResponse += weight * getRoundedDifference(response[responseOffset + neighbor], lagCenter, lagRuntimeZero);
    let nextRowSum = rowSum + weight;
    rowSumError += getAdditionRoundingError(rowSum, weight, nextRowSum);
    rowSum = nextRowSum;
    for (var column = 0u; column < K; column++) {
      (*lagged)[column] += weight * design[designOffset + neighbor * K + column];
    }
    if (needSecond) {
      var inner: array<f32, K>;
      for (var second = offsets[offsetsOffset + neighbor]; second < offsets[offsetsOffset + neighbor + 1u]; second++) {
        let farther = neighbors[neighborsOffset + second];
        if (farther >= ROWS) {
          continue;
        }
        let farWeight = weights[weightsOffset + second];
        for (var column = 0u; column < K; column++) {
          inner[column] += farWeight * design[designOffset + farther * K + column];
        }
      }
      for (var column = 0u; column < K; column++) {
        (*twiceLagged)[column] += weight * inner[column];
      }
    }
  }
  lagCenteredPart = lagResponse;
  lagRowSumDeviationHigh = getRoundedDifference(rowSum, workspace[workspaceOffset + W_ROW_SUM], lagRuntimeZero);
  lagRowSumDeviationLow = rowSumError;
  return lagResponse + lagCenter * (lagRowSumDeviationHigh + lagRowSumDeviationLow);
}
`;
}

/**
 * `getCenteredRow`: centered instruments `h`, regressors `z` and response of one row. Kernels
 * that include it bind `workspace` (the means).
 */
function getCenteredRowWGSL(): string {
  return /* wgsl */ `
// h = [x - mean(x), W x - mean(W x)], z = [x - mean(x), v - mean(v)] with v the lag column of
// getLags (W y less a constant, so its deviations are those of W y). Every difference of a
// value and its mean goes through getRoundedDifference: the shader compiler of the test GPU
// otherwise regroups (y - mean) - z delta - offset around the mean, where the small terms are
// lost below the float32 step of the response.
fn getCenteredRow(
  row: u32,
  h: ptr<function, array<f32, M>>,
  z: ptr<function, array<f32, Q>>,
  centeredResponse: ptr<function, f32>,
  needSecond: bool
) {
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var lagged: array<f32, K>;
  var twiceLagged: array<f32, K>;
  let lagResponse = getLags(row, &lagged, &twiceLagged, needSecond);
  for (var column = 0u; column < K; column++) {
    let centered = getRoundedDifference(design[designOffset + row * K + column], workspace[workspaceOffset + column], runtimeZero);
    (*h)[column] = centered;
    (*z)[column] = centered;
    (*h)[K + column] = getRoundedDifference(lagged[column], workspace[workspaceOffset + K + column], runtimeZero);
    if (ORDER == 2u && needSecond) {
      (*h)[2u * K + column] = getRoundedDifference(twiceLagged[column], workspace[workspaceOffset + 2u * K + 2u + column], runtimeZero);
    }
  }
  (*z)[K] = getRoundedDifference(lagResponse, workspace[workspaceOffset + 2u * K], runtimeZero);
  *centeredResponse = getRoundedDifference(response[responseOffset + row], workspace[workspaceOffset + 2u * K + 1u], runtimeZero);
}
`;
}

/**
 * `getStructuralResidual`: the residual `(y - ybar) - (Z - mean) delta - offset` of one row at the
 * float32 `delta`, as a float32 value (x) and a float32 correction (y) whose sum is the residual.
 * Call it after `getCenteredRow` of the same row (it reads the parts of the lag that `getLags`
 * left). Kernels that include it bind `workspace`.
 *
 * Why two parts. Where the lag differs between rows by the response mean (rows without
 * neighbours, row sums that differ), a regressor times its coefficient is hundreds to thousands
 * while the residual is 0.01: each product is rounded to 1e-5 to 1e-4, and the lag column itself
 * is rounded to a float32 step of the response mean. That rounding is random over rows and
 * small in `sigma2`, but it is all the refinement step has to work from. Here every product
 * keeps its rounding error (`getMultiplicationRoundingError`), every subtraction keeps its own
 * (`getAdditionRoundingError`), and the lag enters as `W (y - c) + c (s_i - s)` with its two
 * parts multiplied separately, so the residual is that of the float32 inputs and `delta`, to the
 * rounding of the residual itself. Each product is passed through an integer exclusive-or with
 * the runtime zero first, so that the compiler cannot fuse it with the subtraction (the error
 * terms are those of the two separate operations).
 */
function getStructuralResidualWGSL(): string {
  return /* wgsl */ `
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
// value - first * second, where value.x + value.y is the running value.
fn subtractProduct(value: vec2<f32>, first: f32, second: f32, runtimeZero: u32) -> vec2<f32> {
  let product = bitcast<f32>(bitcast<u32>(first * second) ^ runtimeZero);
  let next = value.x - product;
  return vec2<f32>(
    next,
    value.y + getAdditionRoundingError(value.x, -product, next) - getMultiplicationRoundingError(first, second, product)
  );
}

fn getStructuralResidual(z: ptr<function, array<f32, Q>>, centeredResponse: f32) -> vec2<f32> {
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  var value = vec2<f32>(centeredResponse, 0.0);
  for (var b = 0u; b < K; b++) {
    value = subtractProduct(value, (*z)[b], workspace[workspaceOffset + W_DELTA + b], runtimeZero);
  }
  let lagCoefficient = workspace[workspaceOffset + W_DELTA + K];
  let lagCenter = workspace[workspaceOffset + W_LAG_CENTER];
  // The lag column times rho in its parts: W (y - c), then c (s_i - s) as c times the product of
  // the row sum deviation and rho, then the mean of the lag column, which the centered regressor
  // subtracts.
  value = subtractProduct(value, lagCenteredPart, lagCoefficient, runtimeZero);
  let scaledDeviation = bitcast<f32>(bitcast<u32>(lagRowSumDeviationHigh * lagCoefficient) ^ runtimeZero);
  let scaledDeviationError = getMultiplicationRoundingError(lagRowSumDeviationHigh, lagCoefficient, scaledDeviation);
  value = subtractProduct(value, lagCenter, scaledDeviation, runtimeZero);
  value = subtractProduct(value, -workspace[workspaceOffset + 2u * K], lagCoefficient, runtimeZero);
  let lowPart = value.y
    - lagCenter * (scaledDeviationError + lagRowSumDeviationLow * lagCoefficient)
    - workspace[workspaceOffset + W_RESIDUAL_OFFSET];
  return vec2<f32>(value.x, lowPart);
}
`;
}

/**
 * `solveSystem` for the solve and finish kernels: scaled Cholesky of `H'H`, `P`-projected normal
 * matrix `Z'PZ = (H'Z)' (H'H)^-1 (H'Z)`, its solve for `delta`, the diagonal of its inverse and the
 * quadratic form `m' (Z'PZ)^-1 m` of the regressor means `m` used by the intercept variance.
 * With `responseSource: 'refinement'` the right-hand side is `H'u` of the `refinement` binding
 * (the residuals at the float32 `delta`) instead of `H'y`, so `delta` is the refinement step.
 */
function getSolveWGSL(
  m: number,
  q: number,
  responseSource: 'moments' | 'refinement' = 'moments'
): string {
  return /* wgsl */ `
${GET_NAN_WGSL}

${IS_FINITE_VALUE_WGSL}

// Mean of regressor column: a predictor, or W y, which is the lag column plus the constant it
// was formed about (see getLags).
fn regressorMean(column: u32) -> f32 {
  if (column == K) {
    return workspace[workspaceOffset + 2u * K] + workspace[workspaceOffset + W_LAG_CENTER] * workspace[workspaceOffset + W_ROW_SUM];
  }
  return workspace[workspaceOffset + column];
}

// The moment pass sums products of deviations from the stored float32 means, which miss the true
// means by up to a float32 step. Each accessor subtracts n times the product of the two mean
// remainders (the deviation sums over n), which gives the moment about the true means, as
// GPUOrdinaryLeastSquares does. Without it a constant predictor whose float32 mean is not the
// constant kept a sum of squares above zero and was fitted with status 0.
fn getInstrumentDeviationSum(a: u32) -> f32 {
  if (a < K) {
    return moments[momentsOffset + MOMENTS + a];
  }
  return moments[momentsOffset + MOMENTS + Q + 1u + (a - K)];
}

fn getInstrumentMoment(a: u32, c: u32) -> f32 {
  return moments[momentsOffset + a * M + c] - getInstrumentDeviationSum(a) * (getInstrumentDeviationSum(c) / f32(ROWS));
}

fn getCrossMoment(a: u32, b: u32) -> f32 {
  return moments[momentsOffset + M * M + a * Q + b] - getInstrumentDeviationSum(a) * (moments[momentsOffset + MOMENTS + b] / f32(ROWS));
}

fn getResponseMoment(a: u32) -> f32 {
  ${
    responseSource === 'refinement'
      ? 'return refinement[refinementOffset + a] - getInstrumentDeviationSum(a) * (refinement[refinementOffset + M] / f32(ROWS));'
      : 'return moments[momentsOffset + M * M + M * Q + a] - getInstrumentDeviationSum(a) * (moments[momentsOffset + MOMENTS + Q] / f32(ROWS));'
  }
}

fn getRegressorSquares(b: u32) -> f32 {
  let deviationSum = moments[momentsOffset + MOMENTS + b];
  return moments[momentsOffset + M * M + M * Q + M + b] - deviationSum * (deviationSum / f32(ROWS));
}

// Whether a corrected sum of squares is above the rounding of the two sums it is the difference
// of: below, the column is constant at float32 resolution.
fn isColumnVarying(corrected: f32, uncorrected: f32) -> bool {
  return corrected > ${getWGSLFloatLiteral(CONSTANT_COLUMN_TOLERANCE)} * uncorrected && isFiniteValue(corrected);
}

${getCholeskyWGSL(m)}
${q === m ? '' : getCholeskyWGSL(q)}
${[...new Set([m, q])].map(size => getSymmetricEigenvalueRangeWGSL(size)).join('\n')}
${getBelsleyConditionNumberWGSL(q)}

// With wantDiagnostics, diagnostics = [Belsley condition number of [1, PZ], correlation condition
// number of Z'PZ, largest variance inflation factor of Z'PZ, correlation condition number of the
// instruments, first-stage F of the excluded instruments, their partial R squared].
fn solveSystem(
  delta: ptr<function, array<f32, ${q}>>,
  inverseDiagonal: ptr<function, array<f32, ${q}>>,
  meanQuadratic: ptr<function, f32>,
  second: ptr<function, array<f32, ${q}>>,
  secondQuadratic: ptr<function, f32>,
  diagnostics: ptr<function, array<f32, 6>>,
  wantDiagnostics: bool
) -> bool {
  var instrumentScale: array<f32, ${m}>;
  var regressorScale: array<f32, ${q}>;
  for (var a = 0u; a < M; a++) {
    let diagonal = getInstrumentMoment(a, a);
    if (!isColumnVarying(diagonal, moments[momentsOffset + a * M + a])) {
      return false;
    }
    instrumentScale[a] = 1.0 / sqrt(diagonal);
  }
  for (var b = 0u; b < Q; b++) {
    let diagonal = getRegressorSquares(b);
    if (!isColumnVarying(diagonal, moments[momentsOffset + M * M + M * Q + M + b])) {
      return false;
    }
    regressorScale[b] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${m * m}>;
  for (var a = 0u; a < M; a++) {
    for (var c = 0u; c < M; c++) {
      factor[a * M + c] = getInstrumentMoment(a, c) * instrumentScale[a] * instrumentScale[c];
    }
  }
  if (wantDiagnostics) {
    var instrumentCorrelation = factor;
    let instrumentRange = getSymmetricEigenvalueRange_${m}(&instrumentCorrelation);
    (*diagnostics)[3] = instrumentRange.y / max(instrumentRange.x, 1.0e-30);
  }
  if (!choleskyFactor_${m}(&factor)) {
    return false;
  }
  if (wantDiagnostics) {
    // First stage of W y on the instruments: explained sums of squares from the forward
    // substitution y = L^-1 (scaled H'z); the first K instruments are the included X.
    var reduced: array<f32, ${m}>;
    for (var a = 0u; a < M; a++) {
      reduced[a] = getCrossMoment(a, K) * instrumentScale[a];
    }
    var explainedFull = 0.0;
    var explainedRestricted = 0.0;
    for (var row = 0u; row < M; row++) {
      var value = reduced[row];
      for (var j = 0u; j < row; j++) {
        value = value - factor[row * M + j] * reduced[j];
      }
      value = value / factor[row * M + row];
      reduced[row] = value;
      explainedFull += value * value;
      if (row < K) {
        explainedRestricted += value * value;
      }
    }
    let total = getRegressorSquares(K);
    let excluded = explainedFull - explainedRestricted;
    (*diagnostics)[4] = (excluded / f32(M - K)) / ((total - explainedFull) / (f32(ROWS) - f32(M) - 1.0));
    (*diagnostics)[5] = excluded / (total - explainedRestricted);
  }
  // x1 = (H'H)^-1 H'Z column by column, x2 = (H'H)^-1 H'y (scaled instrument space).
  var projectedInstruments: array<f32, ${m * q}>;
  for (var b = 0u; b < Q; b++) {
    var rhs: array<f32, ${m}>;
    for (var a = 0u; a < M; a++) {
      rhs[a] = getCrossMoment(a, b) * instrumentScale[a] * regressorScale[b];
    }
    choleskySolve_${m}(&factor, &rhs);
    for (var a = 0u; a < M; a++) {
      projectedInstruments[a * Q + b] = rhs[a];
    }
  }
  var projectedResponse: array<f32, ${m}>;
  for (var a = 0u; a < M; a++) {
    projectedResponse[a] = getResponseMoment(a) * instrumentScale[a];
  }
  choleskySolve_${m}(&factor, &projectedResponse);
  var normal: array<f32, ${q * q}>;
  var rhs: array<f32, ${q}>;
  for (var b = 0u; b < Q; b++) {
    var rhsValue = 0.0;
    for (var a = 0u; a < M; a++) {
      rhsValue += getCrossMoment(a, b) * instrumentScale[a] * regressorScale[b] * projectedResponse[a];
    }
    rhs[b] = rhsValue;
    for (var c = 0u; c < Q; c++) {
      var sum = 0.0;
      for (var a = 0u; a < M; a++) {
        sum += getCrossMoment(a, b) * instrumentScale[a] * regressorScale[b] * projectedInstruments[a * Q + c];
      }
      normal[b * Q + c] = sum;
    }
  }
  var normalDiagonal: array<f32, ${q}>;
  if (wantDiagnostics) {
    for (var b = 0u; b < Q; b++) {
      normalDiagonal[b] = normal[b * Q + b];
    }
    // Correlation condition number of Z'PZ (unit diagonal) and Belsley's of the raw [1, PZ]
    // design with unit columns, from the Cholesky factor of the correlation matrix and the column
    // means (never the raw Gram matrix; see getBelsleyConditionNumberWGSL).
    var secondCorrelation: array<f32, ${q * q}>;
    var alpha: array<f32, ${q}>;
    var beta: array<f32, ${q}>;
    let sampleSize = f32(ROWS);
    for (var b = 0u; b < Q; b++) {
      let meanB = regressorMean(b);
      let centered = normal[b * Q + b] / (regressorScale[b] * regressorScale[b]);
      let rawNorm = sqrt(sampleSize * meanB * meanB + centered);
      alpha[b] = sqrt(sampleSize) * meanB / rawNorm;
      beta[b] = sqrt(centered) / rawNorm;
      for (var c = 0u; c < Q; c++) {
        secondCorrelation[b * Q + c] = select(normal[b * Q + c] / sqrt(normal[b * Q + b] * normal[c * Q + c]), 1.0, b == c);
      }
    }
    var correlationEigen = secondCorrelation;
    let correlationRange = getSymmetricEigenvalueRange_${q}(&correlationEigen);
    (*diagnostics)[1] = correlationRange.y / max(correlationRange.x, 1.0e-30);
    var belsley = 3.0e38;
    if (choleskyFactor_${q}(&secondCorrelation)) {
      belsley = getBelsleyConditionNumber_${q}(&secondCorrelation, &alpha, &beta);
    }
    (*diagnostics)[0] = belsley;
  }
  if (!choleskyFactor_${q}(&normal)) {
    return false;
  }
  if (wantDiagnostics) {
    var inflation = 0.0;
    for (var b = 0u; b < Q; b++) {
      inflation = max(inflation, choleskyInverseDiagonal_${q}(&normal, b) * normalDiagonal[b]);
    }
    (*diagnostics)[2] = inflation;
  }
  choleskySolve_${q}(&normal, &rhs);
  for (var b = 0u; b < Q; b++) {
    (*delta)[b] = rhs[b] * regressorScale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${q}(&normal, b) * regressorScale[b] * regressorScale[b];
  }
  // Quadratic form of the regressor means, in the scaled space.
  var meansScaled: array<f32, ${q}>;
  for (var b = 0u; b < Q; b++) {
    meansScaled[b] = regressorMean(b) * regressorScale[b];
  }
  var solved = meansScaled;
  choleskySolve_${q}(&normal, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    quadratic += meansScaled[b] * solved[b];
  }
  *meanQuadratic = quadratic;
  var secondScaled: array<f32, ${q}>;
  for (var b = 0u; b < Q; b++) {
    secondScaled[b] = (*second)[b] * regressorScale[b];
  }
  var secondSolved = secondScaled;
  choleskySolve_${q}(&normal, &secondSolved);
  var secondSum = 0.0;
  for (var b = 0u; b < Q; b++) {
    secondSum += secondScaled[b] * secondSolved[b];
  }
  *secondQuadratic = secondSum;
  return true;
}
`;
}

/** The finish kernel: coefficient table, summary and status. */
function getFinishWGSL(k: number): string {
  return /* wgsl */ `
${ERFC_VALUE_WGSL}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${(k + 2) * 4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH}u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  let degrees = n - f32(Q + 1u);
  if (!(degrees > 0.0)) {
    fail(2u);
    return;
  }
  let residualSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let responseResidual = residualSums[residualSumsOffset + 2u];
  let responseSquares = residualSums[residualSumsOffset + 3u];
  if (!isFiniteValue(residualSquares) || !isFiniteValue(responseResidual) || !isFiniteValue(responseSquares)) {
    fail(select(3u, 1u, !isFiniteValue(workspace[workspaceOffset + W_DELTA])));
    return;
  }
  var delta: array<f32, ${k + 1}>;
  var inverseDiagonal: array<f32, ${k + 1}>;
  var meanQuadratic = 0.0;
  var productVector: array<f32, ${k + 1}>;
  for (var b = 0u; b < Q; b++) {
    productVector[b] = akSums[akSumsOffset + ${ANSELIN_KELEJIAN_FIXED_SUM_COUNT}u + b];
  }
  var productQuadratic = 0.0;
  var diagnostics: array<f32, 6>;
  if (!solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &productVector, &productQuadratic, &diagnostics, true)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSquares / n;
  // The published coefficients: the float32 nearest to the solve's delta plus its refinement
  // (the residuals, sigma2 and the statistics are those of the unrounded sum).
  for (var b = 0u; b < Q; b++) {
    delta[b] = workspace[workspaceOffset + W_DELTA + b] + workspace[workspaceOffset + W_CORRECTION + b];
  }
  // Intercept: ybar - mean(Z) delta. Its variance is sigma2 / n plus the delta part.
  var intercept = workspace[workspaceOffset + 2u * K + 1u];
  for (var b = 0u; b < Q; b++) {
    intercept -= regressorMean(b) * delta[b];
  }
  intercept += workspace[workspaceOffset + W_RESIDUAL_OFFSET] + workspace[workspaceOffset + W_CORRECTION_OFFSET];
  let interceptVariance = sigmaSquared * (1.0 / n + meanQuadratic);
  for (var row = 0u; row < Q + 1u; row++) {
    // Rows 0 is the intercept; rows 1..K are the predictors; row K + 1 is rho (delta index K).
    let index = max(row, 1u) - 1u;
    let coefficient = select(delta[index], intercept, row == 0u);
    let variance = select(sigmaSquared * inverseDiagonal[index], interceptVariance, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = residualSquares;
  summaryOut[summaryOutOffset + 3u] = covariance * covariance / (responseSquares * predictedVariance);
  // Anselin-Kelejian (1997) as in spreg akTest: Moran's I of u, corrected by the estimation variance.
  let residualLagProduct = akSums[akSumsOffset];
  let rowSumProduct = akSums[akSumsOffset + 1u];
  let weightsSum = akSums[akSumsOffset + 2u];
  let traceT = akSums[akSumsOffset + 3u] + akSums[akSumsOffset + 4u];
  // Moran's I divides by the total weight S0 and is undefined (NaN) when signed weights sum
  // to zero. The statistic n I^2 / phi^2, with phi^2 = (T + 4 a / sigma2) / ((S0 / n)^2 n),
  // does not depend on S0: it is (u'Wu / sigma2)^2 / (T + 4 a / sigma2), formed that way so
  // that a zero total does not turn it into infinity over infinity.
  let moranI = select((n * residualLagProduct) / (weightsSum * residualSquares), getNaN(), weightsSum == 0.0);
  let estimationTerm = rowSumProduct * rowSumProduct / n + productQuadratic;
  let residualLagRatio = residualLagProduct / sigmaSquared;
  let anselinKelejian = residualLagRatio * residualLagRatio / (traceT + 4.0 * estimationTerm / sigmaSquared);
  summaryOut[summaryOutOffset + 4u] = moranI;
  summaryOut[summaryOutOffset + 5u] = anselinKelejian;
  summaryOut[summaryOutOffset + 6u] = erfcValue(sqrt(max(anselinKelejian, 0.0) * 0.5));
  // Conditioning of the second-stage and instrument systems; a warning never changes the status.
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER}u] = diagnostics[0];
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER}u] = diagnostics[1];
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR}u] = diagnostics[2];
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER}u] = diagnostics[3];
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PRECISION_WARNING}u] =
    select(0.0, 1.0, diagnostics[1] > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)} || diagnostics[3] > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)});
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F}u] = diagnostics[4];
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED}u] = diagnostics[5];
  // rho is the last delta; admissible when |rho| ||W||_inf < 1 (NaN compares false: flagged).
  let weightsNorm = workspace[workspaceOffset + W_NORM];
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM}u] = weightsNorm;
  summaryOut[summaryOutOffset + ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE}u] =
    select(1.0, 0.0, abs(delta[K]) * weightsNorm < 1.0);
  statusOut[statusOutOffset] = 0u;
}
`;
}
