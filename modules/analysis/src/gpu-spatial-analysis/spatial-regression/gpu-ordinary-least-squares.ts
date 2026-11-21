// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  validateRequirement,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveOrdinaryLeastSquaresPredictorCount,
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_PRECISION_WARNING,
  GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from './ordinary-least-squares-parameters';
import {
  CHI_SQUARE_MAXIMUM_ITERATIONS,
  getLogGammaOfHalfInteger
} from './ordinary-least-squares-statistics';
import {
  getBelsleyConditionNumberWGSL,
  getCholeskyWGSL,
  getSymmetricEigenvalueRangeWGSL,
  SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
} from './spatial-regression-solve';
import type {GPUSolverStatusPort} from '../../analysis-core/contracts/index';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../../analysis-core/contracts/model-result';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {
  getRegressionTileLayout,
  getRowTileMergeNodes,
  getRowTileMergeScratchLength,
  REGRESSION_TILE_MERGE_GROUP_SIZE
} from './regression-row-tiles';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {LOCAL_REGRESSION_PAIRS_WGSL} from './local-regression-pairs-wgsl';
import {
  CENTERED_REGRESSION_PREDICTION_WGSL,
  getCenteredRegressionContributionWGSL
} from './centered-regression-prediction';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {
  createOrdinaryLeastSquaresDesign,
  createOrdinaryLeastSquaresJoinedTableMask,
  type OrdinaryLeastSquaresJoinedTable
} from './ordinary-least-squares-composition';

import {
  getOrdinaryLeastSquaresResponseScaleNodes,
  getOrdinaryLeastSquaresResponseRestoreNodes,
  ORDINARY_LEAST_SQUARES_RESPONSE_SCALE_WGSL
} from './ordinary-least-squares-response-scale';

const OPERATION = 'GPUOrdinaryLeastSquares';
const TILE_WORKGROUP_SIZE = 64;

/** Caller-owned outputs of {@link GPUOrdinaryLeastSquares}. */
export type GPUOrdinaryLeastSquaresOutput = {
  /** One-row solver `code` (0 on success; see the class documentation for failure codes). */
  status: Pick<GPUSolverStatusPort, 'code'>;

  /** `predictorCount + 1` coefficients, intercept first, then one per predictor column. */
  coefficients: GraphDataView<'float32'>;
  /** Coefficient standard errors, same layout as `coefficients`. */
  standardErrors: GraphDataView<'float32'>;
  /** Coefficient t statistics, same layout as `coefficients`. */
  tStatistics: GraphDataView<'float32'>;
  /**
   * At least 22 float32 values; see the `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*` slot constants:
   * row count, R2, adjusted R2, sigma squared, log-likelihood, AIC, BIC, Jarque-Bera and p-value,
   * Koenker-Bassett and p-value, RSS, TSS, skewness, kurtosis, ridge lambda, Breusch-Pagan and
   * p-value, Belsley condition number, correlation condition number, maximum variance inflation
   * factor and the precision warning. `getGPUOrdinaryLeastSquaresModelResult` turns the read-back
   * arrays into a `GPURegressionModelResult`.
   */
  summary: GraphDataView<'float32'>;
  /** Optional per-row residuals `y - fitted`; NaN for excluded rows and for a failed fit. */
  residuals?: GraphDataView<'float32'>;
  /** Optional per-row fitted values; NaN for excluded rows and for a failed fit. */
  fitted?: GraphDataView<'float32'>;
  /** Optional predictions for every finite predictor row, including masked rows or missing responses.
   * Uses training-only centers and their mean correction; NaN for a failed fit or invalid predictors.
   */
  predictions?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUOrdinaryLeastSquares}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Topology (needs
 * a new graph): `predictorCount`, the row count, whether `mask`, `parameters`, `residuals` and
 * `fitted` and `predictions` are present, and `tileRowCount`.
 */
export type GPUOrdinaryLeastSquaresProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'ordinary-least-squares'`. */
  id?: string;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept column. */
  predictors: GraphDataView<'float32'>;
  /** Response, one value per row. */
  response: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional per-frame parameters: a float32 view of at least 1 element written with
   * `getGPUOrdinaryLeastSquaresParameterValues` (ridge lambda). Omit for a plain OLS fit.
   */
  parameters?: GraphDataView<'float32'>;
  /** Number of predictor columns, 1 to 15. Defaults to the exact predictors.length / response.length for nonempty rows. */
  predictorCount: number;
  /**
   * Rows reduced by one invocation in the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`. The tile layout fixes the summation order, so results are
   * bitwise reproducible for a given value but differ in the last bits between values.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUOrdinaryLeastSquaresOutput;
};

/** Construction input allowing predictor-count inference from exact nonempty views.
 * The operation form retains an explicit count because its graph names carry no view lengths.
 */
export type GPUOrdinaryLeastSquaresInputProps = Omit<
  GPUOrdinaryLeastSquaresProps,
  'predictorCount'
> & {
  predictorCount?: number;
};

/**
 * Ordinary least squares with an intercept and a regression report: coefficients, standard errors,
 * t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC, BIC, residual and fitted
 * columns, the Jarque-Bera normality test, the Koenker-Bassett and Breusch-Pagan
 * heteroskedasticity tests, and collinearity and precision diagnostics.
 *
 * The number of predictors is capped at 15 (`predictorCount` 1 to 15, 16 coefficients with the
 * intercept) by the 16-wide function-local Cholesky factorization of the solve.
 *
 * Rows with a non-finite predictor or response, or with a zero `mask`, are excluded. An optional
 * per-frame ridge penalty shrinks slopes (never the intercept). Use the parameter packer: positive
 * penalties must remain normal float32 values, at least 2^-126; smaller requests are rejected.
 * Response magnitudes outside binary exponents [-32,32] are normalized by an exact power of two
 * before centering. Reports and row outputs are restored to caller units; exponent arithmetic
 * carries the original units through the likelihood even when original residuals are subnormal.
 *
 * Accumulation is deterministic and well-conditioned, with no float atomics:
 * 1. One origin pass selects the first complete training row of each tile. A tile pass sums
 *    differences from that row, fixing each difference with an integer bit cast, then a merge
 *    shifts the partials to the first nonempty tile's origin and sums them in a fixed pairwise
 *    order (`getPairwiseSumWGSL`). Only then is the origin added back to obtain the stored means.
 *    This avoids accumulating large raw values before recovering small predictor variances.
 * 2. A second tile and merge pass accumulates the centered cross-products `X'X`, `X'y`, `y'y`
 *    about those means, which removes the cancellation of the raw-moment formulation. A float32
 *    mean misses the true mean of its column by up to a float32 step, so the same pass sums the
 *    deviations of every column from its stored mean: each sum over `n` is the part of the mean
 *    that float32 cannot hold. The cross-products are corrected by `n` times the product of two
 *    such parts, which makes them moments about the true means.
 * 3. One single-invocation kernel scales the centered normal matrix to a correlation matrix (so
 *    columns of different units are comparable), adds the scaled ridge penalty, factors it with
 *    `choleskyFactor_p`, and solves for the slopes. A pivot below
 *    `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` reports status 1.
 * 4. A per-row pass writes residuals and fitted values from the centered form
 *    `e = (y - ybar) - sum(b (x - xbar)) - offset`. Tile/merge passes retain raw RSS and the largest
 *    absolute residual, then sum squared/cubed/fourth-power residuals normalized by that maximum.
 *    This keeps standardized diagnostics independent of response units without overflowing or
 *    underflowing raw fourth powers. The raw RSS remains accurate near `R2 = 1`. `offset` is
 *    what the stored means miss of the true means, carried through the fit (the response part
 *    minus the slopes times the predictor parts), so the residuals sum to zero as least squares
 *    with an intercept requires, also for a response whose mean is thousands of residual
 *    deviations (residuals of 0.002 around 10,000 match float64 to 1e-4 in the residual sum of
 *    squares). `y - ybar` is fixed with `getRoundedDifference` first: the shader compiler of the
 *    test GPU otherwise adds the offset to the mean, where it is lost. Each `x - xbar` of the
 *    fitted value is fixed the same way, so the sum over the predictors cannot be regrouped
 *    into `sum(b x) - sum(b xbar)` (not observed on the test GPU; the moment pass guards the
 *    same difference). What remains is the float32 resolution of the inputs themselves.
 * 5. A single-invocation finish kernel computes the report. Standard errors are
 *    `sqrt(sigma2 * diag((X'X + lambda I)^-1))` (for a ridge fit they ignore the shrinkage bias),
 *    the intercept variance is `sigma2 (1/n + xbar' (X'X)^-1 xbar)`. The log-likelihood is
 *    Gaussian with `sigma2 = RSS / n`, evaluated from scaled residual squares and their
 *    scale in log space for every positive RSS. Exactly zero RSS alone uses variance
 *    `1.17549435e-38`, preserving the finite exact-fit convention. AIC and BIC count
 *    `p = predictorCount + 1` parameters (the GeoDa convention): the
 *    intercept and the predictors, not the error variance. The other estimators follow their own
 *    reference: `GPUSpatialLagMaximumLikelihood` counts `p + 1` (it adds `rho`),
 *    `GPUSpatialErrorMaximumLikelihood` counts `p` (not `lambda`), and the GWR family counts
 *    `tr(S) + 1` (the error variance). Before comparing criteria across these models, put them
 *    on one count: each uncounted parameter adds 2 to AIC and `ln(n)` to the Schwarz criterion.
 * 6. Jarque-Bera uses the residual moments about zero (the mean is zero with an intercept):
 *    `JB = n/6 (S^2 + (K - 3)^2 / 4)` with p-value `exp(-JB / 2)`. The same accumulation, solve and
 *    a small finish kernel run again with `e^2` as the response (no ridge) and give `R2_aux` of
 *    `e^2 ~ [1, X]`. Koenker-Bassett (studentized, no normality assumption) is `n R2_aux`; the
 *    classic Breusch-Pagan (normal errors) is `0.5 R2_aux n (K - 1)` with the residual kurtosis
 *    `K`, which equals the explained sum of squares over two of `e^2 / s2 - 1` on `[1, X]` with
 *    `s2 = RSS / n`. Both use a chi-square p-value on `predictorCount` degrees of freedom
 *    (regularized upper incomplete gamma with a fixed GPU iteration cap; the host helper uses
 *    a converged shape-dependent budget). Classic BP is zero with p=1 for positive constant
 *    squared residuals; KB is undefined there. The auxiliary response is normalized e² minus one,
 *    formed from differences of magnitudes to retain small deviations. This constant shift leaves
 *    both statistics unchanged. Positive normalized residual sums, rather than raw squares that
 *    can underflow, distinguish a nonzero residual fit. The tests differ for non-normal errors, when
 *    Jarque-Bera rejects: prefer Koenker-Bassett then.
 * 7. Conditioning is computed in the finish kernel from the quantities already in the workspace:
 *    the Belsley condition number of the raw `[1, X]` design with unit-length columns (spreg
 *    `condition_index`), computed by one-sided Jacobi from the correlation Cholesky factor and the
 *    column means (`getBelsleyConditionNumberWGSL`; the raw Gram eigenvalue route loses the small
 *    eigenvalue to float32 rounding beyond a condition number of a few hundred), the condition number
 *    of the predictor correlation matrix that the float32 Cholesky factors, and the largest
 *    variance inflation factor (largest diagonal of its inverse). Eigenvalues come from the
 *    cyclic Jacobi routine of `spatial-regression-solve`. The precision warning is 1 when the
 *    correlation condition number exceeds `GPU_REGRESSION_PRECISION_CONDITION_LIMIT` (1e4, where
 *    float32 normal equations may have lost over three of seven digits). A warning never changes
 *    `status` and never NaNs the fit. With a ridge penalty the correlation condition number and
 *    the variance inflation factors describe the penalized matrix that was factored.
 *
 * Inputs must be single packed views. Skipped-row masking, NaN propagation and the status word
 * make a failed fit leave every statistic NaN, never partial garbage.
 * Construction captures scalar options and input/output view identities, including nested status;
 * changing caller props afterward cannot change the validated topology. Buffer contents stay dynamic.
 */
export class GPUOrdinaryLeastSquares implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOrdinaryLeastSquaresProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;
  /** Fixed main/BP stages, including the extra hierarchical reduction stages when needed. */
  readonly estimatedCommandNodeCount: number;

  /** Adapts predictor columns in coefficient order to the row-major design. */
  static createDesign = createOrdinaryLeastSquaresDesign;

  /** Adds a completeness-aware joined-table fit and retains join loss status beside its outputs. */
  static addJoinedTableFit<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUOrdinaryLeastSquaresInputProps,
    joinedTable: OrdinaryLeastSquaresJoinedTable
  ) {
    // Construction validates and snapshots every declared field without allocating graph resources.
    const captured = new GPUOrdinaryLeastSquares(props).props;
    const capturedTable = {
      matched: joinedTable.matched,
      status: snapshotConstructionFields(joinedTable.status, ['complete', 'candidateOverflow'])
    };
    validateCapacity(
      captured.id ?? 'ordinary-least-squares',
      'joinedTable.matched.length',
      capturedTable.matched.length,
      captured.response.length,
      captured.response.length
    );
    for (const view of [
      captured.predictors,
      captured.response,
      captured.mask,
      captured.parameters,
      captured.output.coefficients,
      captured.output.standardErrors,
      captured.output.tStatistics,
      captured.output.summary,
      captured.output.status.code,
      captured.output.residuals,
      captured.output.fitted,
      captured.output.predictions,
      capturedTable.matched,
      capturedTable.status.complete,
      capturedTable.status.candidateOverflow
    ]) {
      if (view)
        validateRequirement(
          captured.id ?? 'ordinary-least-squares',
          view.buffer.graph === graph,
          'joined fit input and output views must belong to the supplied graph'
        );
    }
    validateGraphChunkBindingRanges(graph, captured.id ?? 'ordinary-least-squares', {
      predictors: captured.predictors,
      response: captured.response,
      mask: captured.mask,
      parameters: captured.parameters,
      coefficients: captured.output.coefficients,
      standardErrors: captured.output.standardErrors,
      tStatistics: captured.output.tStatistics,
      summary: captured.output.summary,
      code: captured.output.status.code,
      residuals: captured.output.residuals,
      fitted: captured.output.fitted,
      predictions: captured.output.predictions,
      joinedMatched: capturedTable.matched,
      joinedComplete: capturedTable.status.complete,
      joinedCandidateOverflow: capturedTable.status.candidateOverflow
    });
    validateRegressionAllocations(
      graph.device,
      captured.id ?? 'ordinary-least-squares',
      getOrdinaryLeastSquaresAllocationLengths(
        captured.response.length,
        captured.predictorCount,
        captured.tileRowCount
      )
    );
    const composition = createOrdinaryLeastSquaresJoinedTableMask(
      graph,
      capturedTable,
      captured.mask,
      `${captured.id ?? 'ordinary-least-squares'}-join`
    );
    const fit = new GPUOrdinaryLeastSquares({...captured, mask: composition.mask});
    graph.add(fit);
    return {fit, mask: composition.mask, status: composition.status};
  }

  constructor(props: GPUOrdinaryLeastSquaresInputProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'predictors',
      'response',
      'mask',
      'parameters',
      'predictorCount',
      'tileRowCount',
      'output'
    ]);
    const outputSnapshot = snapshotConstructionFields(captured.output, [
      'status',
      'coefficients',
      'standardErrors',
      'tStatistics',
      'summary',
      'residuals',
      'fitted',
      'predictions'
    ]);
    props = Object.freeze({
      ...captured,
      output: Object.freeze({
        ...outputSnapshot,
        status: snapshotConstructionFields(outputSnapshot.status, ['code'])
      })
    });
    this.id = props.id ?? 'ordinary-least-squares';
    const id = this.id;
    const {output} = props;
    const predictorCount = resolveOrdinaryLeastSquaresPredictorCount(
      id,
      props.predictors.length,
      props.response.length,
      props.predictorCount,
      SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT - 1
    );
    this.props = Object.freeze({...props, predictorCount});
    validateSinglePackedView(props.predictors, ['float32'], `${id} predictors`);
    validateSinglePackedView(props.response, ['float32'], `${id} response`);
    const rows = props.response.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal response length * predictorCount`);
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal response length`);
      }
    }
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH} float32 value`
        );
      }
    }
    const coefficientCount = predictorCount + 1;
    for (const name of ['coefficients', 'standardErrors', 'tStatistics'] as const) {
      validateSinglePackedView(output[name], ['float32'], `${id} output.${name}`);
      if (output[name].length < coefficientCount) {
        throw new Error(`${id} output.${name} must hold predictorCount + 1 rows`);
      }
    }
    validateSinglePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} float32 values`
      );
    }
    validateSinglePackedView(output.status.code, ['uint32'], `${id} output.status.code`);
    if (output.status.code.length < 1) {
      throw new Error(`${id} output.status.code must hold one uint32 row`);
    }
    for (const name of ['residuals', 'fitted', 'predictions'] as const) {
      const view = output[name];
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length < rows) {
          throw new Error(`${id} output.${name} must hold one row per input row`);
        }
      }
    }
    if (props.tileRowCount !== undefined) {
      validateCapacity(id, 'tileRowCount', props.tileRowCount);
    }
    this.rowCount = rows;
    const tileLayout = getRegressionTileLayout(rows, props.tileRowCount);
    this.tileRowCount = tileLayout.tileRowCount;
    this.tileCount = tileLayout.tileCount;
    this.estimatedCommandNodeCount =
      20 +
      (this.tileCount > REGRESSION_TILE_MERGE_GROUP_SIZE ? 10 : 0) +
      Math.max(1, Math.ceil(Math.log(rows) / Math.log(256))) +
      4 +
      (output.residuals || output.fitted || output.predictions ? 1 : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateRegressionOutputDispatches(
      id,
      {
        coefficients: output.coefficients,
        standardErrors: output.standardErrors,
        tStatistics: output.tStatistics,
        summary: output.summary,
        code: output.status.code,
        residuals: output.residuals,
        fitted: output.fitted,
        predictions: output.predictions
      },
      [
        ['coefficients', 'standardErrors', 'tStatistics', 'summary', 'code'],
        ['residuals', 'fitted', 'predictions']
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.coefficients,
        output.standardErrors,
        output.tStatistics,
        output.summary,
        output.status.code,
        output.residuals,
        output.fitted,
        output.predictions
      ],
      [props.predictors, props.response, props.mask, props.parameters]
    );
  }

  /**
   * Returns the main chain (origin selection, tile and merge means, tile and merge moments, solve, residuals,
   * residual-moment tile and merge, finish) followed by the heteroskedasticity chain (the same
   * five accumulation nodes on `e^2`, a solve, a finish for Koenker-Bassett and Breusch-Pagan),
   * in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    useRefinedScalarSlope = false
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, tileRowCount, tileCount} = this;
    const {output, predictorCount} = props;
    const refineScalarSlope = useRefinedScalarSlope && predictorCount === 1 && !props.parameters;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      mask: props.mask,
      parameters: props.parameters,
      coefficients: output.coefficients,
      standardErrors: output.standardErrors,
      tStatistics: output.tStatistics,
      summary: output.summary,
      code: output.status.code,
      residuals: output.residuals,
      fitted: output.fitted,
      predictions: output.predictions
    });
    validateRegressionAllocations(
      graph.device,
      id,
      getOrdinaryLeastSquaresAllocationLengths(rowCount, predictorCount, tileRowCount)
    );
    const k = predictorCount;
    const layout = getWorkspaceLayout(k);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const residualView = output.residuals ?? f32('residuals', rowCount);
    const squaredView = f32('squared-residuals', rowCount);
    const responseScale = getOrdinaryLeastSquaresResponseScaleNodes(graph, props);
    const nodes: GPUCommandNode<Parameters>[] = [...responseScale.nodes];
    const accumulationProps = {
      id,
      operation: OPERATION,
      predictors: props.predictors,
      mask: props.mask,
      rowCount,
      k,
      tileRowCount,
      tileCount
    };

    const mainWorkspace = u32('workspace', layout.length);
    nodes.push(
      ...getAccumulationNodes(graph, {
        ...accumulationProps,
        prefix: id,
        response: responseScale.response,
        workspace: mainWorkspace,
        partialMeans: u32('partial-means', tileCount * getMeanPartialStride(k)),
        partialMoments: u32('partial-moments', tileCount * (layout.momentCount + k + 1))
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-solve`,
        variant: 'solve',
        bindings: [
          ...(props.parameters ? [read('params', props.parameters, 'f32')] : []),
          writeWorkspace(mainWorkspace)
        ],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, true)}
${getCholeskyWGSL(k)}${refineScalarSlope ? `\n${LOCAL_REGRESSION_PAIRS_WGSL}` : ''}`,
        body: getSolveBody(k, Boolean(props.parameters), refineScalarSlope)
      })
    );

    nodes.push(
      createKernel(graph, {
        id: `${id}-residuals`,
        variant: 'residuals',
        bindings: [
          read('design', props.predictors, 'f32'),
          read('response', responseScale.response, 'f32'),
          ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
          readWorkspace(mainWorkspace),
          write('residualOut', residualView, 'f32'),
          ...(output.fitted ? [write('fittedOut', output.fitted, 'f32')] : []),
          ...(output.predictions ? [write('predictionOut', output.predictions, 'f32')] : []),
          write('squaredOut', squaredView, 'f32')
        ],
        invocationCount: rowCount,
        declarations: `${getSharedWGSL(layout, false)}
${getRowWGSL(rowCount, k, tileRowCount, Boolean(props.mask))}`,
        body: `let nan = getNaN();
  residualOut[residualOutOffset + index] = nan;
  ${output.fitted ? 'fittedOut[fittedOutOffset + index] = nan;' : ''}
  ${output.predictions ? 'predictionOut[predictionOutOffset + index] = nan;' : ''}
  squaredOut[squaredOutOffset + index] = nan;
  if (ws[wsOffset + W_STATUS] != 0u) {
    return;
  }
  for (var column = 0u; column < K; column++) {
    if (!isFiniteValue(design[designOffset + index * K + column])) { return; }
  }
  // Each x - xbar is rounded as written, as in the moment pass: a compiler that regrouped the sum
  // into sum(b x) - sum(b xbar) would round every fitted value to a float32 step of b x.
  ${getCenteredRegressionContributionWGSL({predictorCount: k, predictor: 'design[designOffset + index * K + column]', mean: 'wsGet(W_MEANS + column)', slope: 'wsGet(W_SLOPES + column)', runtimeZero: 'ws[wsOffset + W_RUNTIME_ZERO]'})}
  let fit = centeredContribution;
  let meanResponse = wsGet(W_MEANS + K);
  // The stored means are float32: W_RESIDUAL_OFFSET is what they miss of the true means, carried
  // through the fit, so the residuals sum to zero as least squares with an intercept requires.
  let offset = wsGet(W_RESIDUAL_OFFSET);
  ${output.predictions ? 'predictionOut[predictionOutOffset + index] = getCenteredRegressionPrediction(meanResponse, fit, offset, ws[wsOffset + W_RUNTIME_ZERO]);' : ''}
  if (!isRowValid(index)) { return; }
  // getRoundedDifference: without it the shader compiler of the test GPU adds the offset to the
  // mean first, where it is lost below the float32 step of the mean.
  let centeredResponse = getRoundedDifference(response[responseOffset + index], meanResponse, ws[wsOffset + W_RUNTIME_ZERO]);
  let residual = (centeredResponse - fit) - offset;
  residualOut[residualOutOffset + index] = residual;
  ${output.fitted ? 'fittedOut[fittedOutOffset + index] = getCenteredRegressionPrediction(meanResponse, fit, offset, ws[wsOffset + W_RUNTIME_ZERO]);' : ''}
  squaredOut[squaredOutOffset + index] = residual * residual;`
      })
    );

    const partialResiduals = f32('partial-residuals', tileCount * RESIDUAL_SUM_COUNT);
    nodes.push(
      createKernel(graph, {
        id: `${id}-residual-tiles`,
        variant: 'residual-tiles',
        workgroupSize: TILE_WORKGROUP_SIZE,
        bindings: [
          read('residualIn', residualView, 'f32'),
          write('partial', partialResiduals, 'f32')
        ],
        invocationCount: tileCount,
        declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
${FINITE_WGSL}`,
        body: `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sumSquares = 0.0;
  var maximumResidual = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e = residualIn[residualInOffset + row];
    if (!isFiniteValue(e)) {
      continue;
    }
    let e2 = e * e;
    sumSquares = sumSquares + e2;
    maximumResidual = max(maximumResidual, abs(e));
  }
  let base = partialOffset + index * ${RESIDUAL_SUM_COUNT}u;
  partial[base] = sumSquares;
  partial[base + 1u] = maximumResidual;
  partial[base + 2u] = 0.0;`
      })
    );
    nodes.push(
      ...getRowTileMergeNodes(
        graph,
        {
          id: `${id}-residual-merge`,
          operation: OPERATION,
          variant: 'residual-merge',
          bindings: [read('partial', partialResiduals, 'f32'), writeWorkspace(mainWorkspace)],
          invocationCount: RESIDUAL_SUM_COUNT,
          declarations: `${getSharedWGSL(layout, true)}
const TILE_COUNT: u32 = ${tileCount}u;`,
          body: `if (index == 1u) {
    var largest = 0.0;
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      largest = max(largest, partial[partialOffset + tile * ${RESIDUAL_SUM_COUNT}u + 1u]);
    }
    wsSet(W_RESIDUAL_SCALE, largest);
    return;
  }
  ${getPairwiseSumWGSL({count: 'TILE_COUNT', term: `partial[partialOffset + sumItem * ${RESIDUAL_SUM_COUNT}u + index]`})}
  wsSet(W_RESIDUAL_SUMS + index, sum);`
        },
        {
          tileCount,
          partials: [{name: 'partial', termCount: RESIDUAL_SUM_COUNT, maxima: {1: 0}}]
        }
      )
    );

    // Normalize by the largest actual residual before cubing/squaring diagnostic responses.
    // Reuse the partial allocation after the raw RSS/scale merge; raw RSS stays untouched.
    nodes.push(
      createKernel(graph, {
        id: `${id}-standardized-residual-tiles`,
        variant: 'standardized-residual-tiles',
        workgroupSize: TILE_WORKGROUP_SIZE,
        invocationCount: tileCount,
        bindings: [
          read('residualIn', residualView, 'f32'),
          readWorkspace(mainWorkspace),
          write('partial', partialResiduals, 'f32'),
          write('squaredOut', squaredView, 'f32')
        ],
        declarations: `${getSharedWGSL(layout, false)}
const ROW_COUNT:u32=${rowCount}u;
const TILE_ROWS:u32=${tileRowCount}u;`,
        body: `let scale=wsGet(W_RESIDUAL_SCALE);
  var sumSquares=0.0;var sumCubes=0.0;var sumFourth=0.0;
  let firstRow=index*TILE_ROWS;let lastRow=min(firstRow+TILE_ROWS,ROW_COUNT);
  for(var row=firstRow;row<lastRow;row++) {
    let residual=residualIn[residualInOffset+row];
    squaredOut[squaredOutOffset+row]=getNaN();
    if(!isFiniteValue(residual)){continue;}
    var standardized=0.0;
    if(scale>0.0){standardized=residual/scale;}
    let squared=standardized*standardized;
    // Center e²/scale² at one before storing it. Direct squaring near one discards the
    // differences that the auxiliary regression needs (and can reverse the KB decision).
    var auxiliary=0.0;
    if(scale>0.0){
      let difference=getRoundedDifference(abs(residual),scale,ws[wsOffset+W_RUNTIME_ZERO]);
      auxiliary=(difference/scale)*(abs(standardized)+1.0);
    }
    squaredOut[squaredOutOffset+row]=auxiliary;
    sumSquares+=squared;sumCubes+=squared*standardized;sumFourth+=squared*squared;
  }
  let base=partialOffset+index*${RESIDUAL_SUM_COUNT}u;
  partial[base]=sumSquares;partial[base+1u]=sumCubes;partial[base+2u]=sumFourth;`
      })
    );
    nodes.push(
      ...getRowTileMergeNodes(
        graph,
        {
          id: `${id}-standardized-residual-merge`,
          operation: OPERATION,
          variant: 'standardized-residual-merge',
          bindings: [read('partial', partialResiduals, 'f32'), writeWorkspace(mainWorkspace)],
          invocationCount: RESIDUAL_SUM_COUNT,
          declarations: `${getSharedWGSL(layout, true)}\nconst TILE_COUNT: u32 = ${tileCount}u;`,
          body: `${getPairwiseSumWGSL({count: 'TILE_COUNT', term: `partial[partialOffset+sumItem*${RESIDUAL_SUM_COUNT}u+index]`})}
  wsSet(W_STANDARDIZED_SUMS+index,sum);`
        },
        {tileCount, partials: [{name: 'partial', termCount: RESIDUAL_SUM_COUNT}]}
      )
    );

    nodes.push(
      createKernel(graph, {
        id: `${id}-finish`,
        variant: 'finish',
        bindings: [
          readWorkspace(mainWorkspace),
          write('coefficientsOut', output.coefficients, 'f32'),
          write('standardErrorsOut', output.standardErrors, 'f32'),
          write('tStatisticsOut', output.tStatistics, 'f32'),
          write('summaryOut', output.summary, 'f32'),
          write('statusOut', output.status.code, 'u32')
        ],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, false)}
${getCholeskyWGSL(k)}
${getSymmetricEigenvalueRangeWGSL(k)}
${getBelsleyConditionNumberWGSL(k)}`,
        body: getFinishBody(k)
      })
    );

    // Each invocation owns one criterion. Reuse the normalized residual accumulation so positive
    // subnormal raw RSS never enters a logarithm (Metal may flush raw squares to zero).
    nodes.push(
      createKernel(graph, {
        id: `${id}-scaled-likelihood`,
        variant: 'scaled-likelihood',
        invocationCount: 3,
        bindings: [
          readWorkspace(mainWorkspace),
          read('responseMaximum', responseScale.maximum, 'u32'),
          write('summaryOut', output.summary, 'f32')
        ],
        declarations: `${getSharedWGSL(layout, false)}\n${ORDINARY_LEAST_SQUARES_RESPONSE_SCALE_WGSL}`,
        body: `if (ws[wsOffset + W_STATUS] != 0u) { return; }
  let n = f32(ws[wsOffset + W_COUNT]);
  let scale = wsGet(W_RESIDUAL_SCALE);
  var logVariance = log(1.17549435e-38);
  if ((bitcast<u32>(scale) & 0x7fffffffu) != 0u) {
    logVariance = log(wsGet(W_STANDARDIZED_SUMS) / n) + 2.0 * log(scale) +
      1.3862943611198906 * f32(getOLSResponseExponent(responseMaximum[responseMaximumOffset]));
  }
  let likelihood = -0.5 * n * (1.8378770664093453 + logVariance + 1.0);
  var value = likelihood;
  if (index == 1u) { value = 2.0 * ${k + 1}.0 - 2.0 * likelihood; }
  if (index == 2u) { value = ${k + 1}.0 * log(n) - 2.0 * likelihood; }
  summaryOut[summaryOutOffset + 4u + index] = value;`
      })
    );

    // Koenker-Bassett and Breusch-Pagan: regress e^2 on the same predictors with no ridge.
    const bpWorkspace = u32('bp-workspace', layout.length);
    nodes.push(
      ...getAccumulationNodes(graph, {
        ...accumulationProps,
        prefix: `${id}-bp`,
        response: squaredView,
        workspace: bpWorkspace,
        partialMeans: u32('bp-partial-means', tileCount * getMeanPartialStride(k)),
        partialMoments: u32('bp-partial-moments', tileCount * (layout.momentCount + k + 1))
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-bp-solve`,
        variant: 'solve',
        bindings: [writeWorkspace(bpWorkspace)],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, true)}
${getCholeskyWGSL(k)}`,
        body: getSolveBody(k, false, false)
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-bp-finish`,
        variant: 'breusch-pagan',
        bindings: [
          readWorkspace(bpWorkspace),
          read('mainWs', mainWorkspace, 'u32'),
          write('summaryOut', output.summary, 'f32')
        ],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, false)}
${getChiSquareWGSL(k)}`,
        body: getBreuschPaganBody(k)
      })
    );
    nodes.push(...getOrdinaryLeastSquaresResponseRestoreNodes(graph, props, responseScale.maximum));
    return nodes;
  }
}

const RESIDUAL_SUM_COUNT = 3;

type WorkspaceLayout = {
  /** Number of packed upper-triangular centered moments of `[predictors, response]`. */
  momentCount: number;
  means: number;
  count: number;
  status: number;
  lambda: number;
  residualSums: number;
  residualScale: number;
  standardizedSums: number;
  scale: number;
  slopes: number;
  factor: number;
  moments: number;
  /** Sums of the deviations of each column from its stored float32 mean (`k + 1` words). */
  deviationSums: number;
  /** `(true mean - stored mean)` of the response minus the same for the predictors times the slopes. */
  residualOffset: number;
  /** A `uint32` zero written at run time by the means merge (see `getRoundedDifference`). */
  runtimeZero: number;
  /** Workspace length in 32-bit words. */
  length: number;
};

/**
 * Word layout of one chain's `uint32` workspace (float32 values are stored with bitcast so counts
 * and statuses survive denormal handling):
 * means (k + 1), count, status, lambda, residual sums (3), scale (k), slopes (k), factor (k*k),
 * packed upper-triangular moments about the stored means (d (d + 1) / 2 with d = k + 1), the sums
 * of the deviations from the stored means (d), the residual offset, the runtime zero,
 * the maximum absolute residual, and three normalized residual sums.
 */
/** Allocation-free scratch plan for OLS and composite panel/SUR callers. @internal */
export function getOrdinaryLeastSquaresAllocationLengths(
  rowCount: number,
  predictorCount: number,
  tileRowCount = Math.max(64, Math.ceil(rowCount / 4096))
): Readonly<Record<string, number>> {
  const layout = getWorkspaceLayout(predictorCount);
  const tileCount = Math.ceil(rowCount / tileRowCount);
  return {
    'OLS residuals/squared residuals': rowCount,
    'OLS normalized response/magnitude scratch': rowCount,
    'OLS workspace': layout.length,
    'OLS mean partials': tileCount * getMeanPartialStride(predictorCount),
    'OLS moment partials': tileCount * (layout.momentCount + predictorCount + 1),
    'OLS residual partials': tileCount * RESIDUAL_SUM_COUNT,
    'OLS mean metadata groups': getRowTileMergeScratchLength(tileCount, 2),
    'OLS merge groups': getRowTileMergeScratchLength(
      tileCount,
      layout.momentCount + predictorCount + 1
    )
  };
}

function getWorkspaceLayout(k: number): WorkspaceLayout {
  const d = k + 1;
  const momentCount = (d * (d + 1)) / 2;
  const means = 0;
  const count = means + d;
  const status = count + 1;
  const lambda = status + 1;
  const residualSums = lambda + 1;
  const scale = residualSums + RESIDUAL_SUM_COUNT;
  const slopes = scale + k;
  const factor = slopes + k;
  const moments = factor + k * k;
  const deviationSums = moments + momentCount;
  const residualOffset = deviationSums + d;
  const residualScale = residualOffset + 2;
  const standardizedSums = residualScale + 1;
  return {
    momentCount,
    means,
    count,
    status,
    lambda,
    residualSums,
    scale,
    slopes,
    factor,
    moments,
    deviationSums,
    residualOffset,
    residualScale,
    standardizedSums,
    runtimeZero: residualOffset + 1,
    length: standardizedSums + RESIDUAL_SUM_COUNT
  };
}

/** Per tile: training-row origins, sums of differences, count, and a separately written zero. */
function getMeanPartialStride(predictorCount: number): number {
  return 2 * (predictorCount + 1) + 2;
}

const FINITE_WGSL = /* wgsl */ `
${GET_NAN_WGSL}

${IS_FINITE_VALUE_WGSL}
`;

/** Layout constants, workspace accessors, and the packed moment index. */
function getSharedWGSL(layout: WorkspaceLayout, writable: boolean): string {
  const k = layout.slopes - layout.scale;
  const d = k + 1;
  return /* wgsl */ `
const K: u32 = ${k}u;
const D: u32 = ${d}u;
const KK: u32 = ${k * k}u;
const W_MEANS: u32 = ${layout.means}u;
const W_COUNT: u32 = ${layout.count}u;
const W_STATUS: u32 = ${layout.status}u;
const W_LAMBDA: u32 = ${layout.lambda}u;
const W_RESIDUAL_SUMS: u32 = ${layout.residualSums}u;
const W_RESIDUAL_SCALE: u32 = ${layout.residualScale}u;
const W_STANDARDIZED_SUMS: u32 = ${layout.standardizedSums}u;
const W_SCALE: u32 = ${layout.scale}u;
const W_SLOPES: u32 = ${layout.slopes}u;
const W_FACTOR: u32 = ${layout.factor}u;
const W_MOMENTS: u32 = ${layout.moments}u;
const W_DEVIATION_SUMS: u32 = ${layout.deviationSums}u;
const W_RESIDUAL_OFFSET: u32 = ${layout.residualOffset}u;
const W_RUNTIME_ZERO: u32 = ${layout.runtimeZero}u;
${ROUNDED_DIFFERENCE_WGSL}
${CENTERED_REGRESSION_PREDICTION_WGSL}
${FINITE_WGSL}
fn wsGet(slot: u32) -> f32 {
  return bitcast<f32>(ws[wsOffset + slot]);
}
${
  writable
    ? `fn wsSet(slot: u32, value: f32) {
  ws[wsOffset + slot] = bitcast<u32>(value);
}`
    : ''
}
// The stored float32 mean of column c misses the true mean by getMeanCorrection(c), which is
// exact to the rounding of a sum of small deviations.
fn getMeanCorrection(column: u32) -> f32 {
  return getRoundedDifference(wsGet(W_DEVIATION_SUMS + column) / max(f32(ws[wsOffset + W_COUNT]), 1.0), 0.0, ws[wsOffset + W_RUNTIME_ZERO]);
}
// Cross-product about the true means: the sum about the stored means less n times the product of
// the two mean corrections.
fn momentAt(first: u32, second: u32) -> f32 {
  let lo = min(first, second);
  let hi = max(first, second);
  return wsGet(W_MOMENTS + lo * (2u * D - lo + 1u) / 2u + (hi - lo))
    - wsGet(W_DEVIATION_SUMS + first) * getMeanCorrection(second);
}
`;
}

/** Row access: validity (mask and finiteness) for the `design`, `response` and optional `rowMask` bindings. */
function getRowWGSL(rowCount: number, k: number, tileRowCount: number, hasMask: boolean): string {
  return /* wgsl */ `
const ROW_COUNT: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const ROW_K: u32 = ${k}u;

fn isRowValid(row: u32) -> bool {
  ${hasMask ? 'if (rowMask[rowMaskOffset + row] == 0u) {\n    return false;\n  }' : ''}
  if (!isFiniteValue(response[responseOffset + row])) {
    return false;
  }
  for (var j = 0u; j < ROW_K; j++) {
    if (!isFiniteValue(design[designOffset + row * ROW_K + j])) {
      return false;
    }
  }
  return true;
}
`;
}

type AccumulationProps = {
  id: string;
  operation: string;
  prefix: string;
  predictors: GraphDataView<'float32'>;
  response: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  rowCount: number;
  k: number;
  tileRowCount: number;
  tileCount: number;
  workspace: GraphDataView<'uint32'>;
  partialMeans: GraphDataView<'uint32'>;
  partialMoments: GraphDataView<'uint32'>;
};

/** Select training-row origins, sum and merge shifted means, then tile and merge centered moments. */
function getAccumulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: AccumulationProps
): GPUCommandNode<Parameters>[] {
  const {prefix, k, rowCount, tileRowCount, tileCount, workspace} = props;
  const d = k + 1;
  const layout = getWorkspaceLayout(k);
  const momentCount = layout.momentCount;
  const meanPartialStride = getMeanPartialStride(k);
  const hasMask = Boolean(props.mask);
  const rowBindings = [
    read('design', props.predictors, 'f32'),
    read('response', props.response, 'f32'),
    ...(props.mask ? [read('rowMask', props.mask, 'u32')] : [])
  ];
  const rowDeclarations = `${getSharedWGSL(layout, false)}
${getRowWGSL(rowCount, k, tileRowCount, hasMask)}`;
  const mergeDeclarations = `${getSharedWGSL(layout, true)}
const TILE_COUNT: u32 = ${tileCount}u;`;
  const mergeLoop = (stride: string) =>
    `${getPairwiseSumWGSL({count: 'TILE_COUNT', term: `bitcast<f32>(partial[partialOffset + sumItem * ${stride} + index])`})}`;
  const nodes = [
    createKernel(graph, {
      id: `${prefix}-means-origins`,
      variant: 'means-origins',
      workgroupSize: TILE_WORKGROUP_SIZE,
      bindings: [...rowBindings, write('partial', props.partialMeans, 'u32')],
      invocationCount: tileCount,
      declarations: `${FINITE_WGSL}\n${getRowWGSL(rowCount, k, tileRowCount, hasMask)}`,
      body: `let base = partialOffset + index * ${meanPartialStride}u;
  for (var column = 0u; column < ${d}u; column++) { partial[base + column] = bitcast<u32>(0.0); }
  // Separate dispatch: later difference bit casts read a zero the compiler cannot know.
  partial[base + ${2 * d + 1}u] = 0u;
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) { continue; }
    for (var column = 0u; column < ROW_K; column++) {
      partial[base + column] = bitcast<u32>(design[designOffset + row * ROW_K + column]);
    }
    partial[base + ROW_K] = bitcast<u32>(response[responseOffset + row]);
    break;
  }`
    }),
    createKernel(graph, {
      id: `${prefix}-means-tiles`,
      variant: 'means-tiles',
      workgroupSize: TILE_WORKGROUP_SIZE,
      bindings: [...rowBindings, write('partial', props.partialMeans, 'u32')],
      invocationCount: tileCount,
      declarations: `${FINITE_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}
${getRowWGSL(rowCount, k, tileRowCount, hasMask)}`,
      body: `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  let base = partialOffset + index * ${meanPartialStride}u;
  let runtimeZero = partial[base + ${2 * d + 1}u];
  var sums: array<f32, ${d}>;
  var count = 0u;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    count = count + 1u;
    for (var j = 0u; j < ROW_K; j++) {
      sums[j] += getRoundedDifference(design[designOffset + row * ROW_K + j], bitcast<f32>(partial[base + j]), runtimeZero);
    }
    sums[ROW_K] += getRoundedDifference(response[responseOffset + row], bitcast<f32>(partial[base + ROW_K]), runtimeZero);
  }
  for (var column = 0u; column < ${d}u; column++) {
    partial[base + ${d}u + column] = bitcast<u32>(sums[column]);
  }
  partial[base + ${2 * d}u] = count;`
    }),
    createKernel(graph, {
      id: `${prefix}-means-merge`,
      variant: 'means-merge',
      bindings: [read('partial', props.partialMeans, 'u32'), writeWorkspace(workspace)],
      invocationCount: d,
      declarations: mergeDeclarations,
      body: `var rows = 0u;
  var referenceTile = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileRows = partial[partialOffset + tile * ${meanPartialStride}u + ${2 * d}u];
    if (rows == 0u && tileRows > 0u) { referenceTile = tile; }
    rows += tileRows;
  }
  let origin = bitcast<f32>(partial[partialOffset + referenceTile * ${meanPartialStride}u + index]);
  let runtimeZero = partial[partialOffset + ${2 * d + 1}u];
  ${getPairwiseSumWGSL({
    count: 'TILE_COUNT',
    setup: `let base = partialOffset + sumItem * ${meanPartialStride}u;
    let tileRows = partial[base + ${2 * d}u];
    var contribution = 0.0;
    if (tileRows > 0u) {
      let shift = getRoundedDifference(bitcast<f32>(partial[base + index]), origin, runtimeZero);
      contribution = bitcast<f32>(partial[base + ${d}u + index]) + shift * f32(tileRows);
    }`,
    term: 'contribution'
  })}
  wsSet(W_MEANS + index, select(0.0, origin + sum / max(f32(rows), 1.0), rows > 0u));
  if (index == 0u) {
    ws[wsOffset + W_COUNT] = rows;
    // A zero the compiler cannot see: the runtime zero of getRoundedDifference.
    ws[wsOffset + W_RUNTIME_ZERO] = 0u;
  }`
    }),
    createKernel(graph, {
      id: `${prefix}-moments-tiles`,
      variant: 'moments-tiles',
      workgroupSize: TILE_WORKGROUP_SIZE,
      bindings: [
        ...rowBindings,
        readWorkspace(workspace),
        write('partial', props.partialMoments, 'u32')
      ],
      invocationCount: tileCount,
      declarations: rowDeclarations,
      body: `var means: array<f32, ${d}>;
  for (var column = 0u; column < D; column++) {
    means[column] = wsGet(W_MEANS + column);
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  // Each deviation is rounded as written (getRoundedDifference): the shader compiler of the test
  // GPU otherwise sums the raw values of a tile and subtracts the mean once, which loses the
  // deviation sums to the float32 step of a tile total.
  let runtimeZero = ws[wsOffset + W_RUNTIME_ZERO];
  var accumulator: array<f32, ${momentCount + d}>;
  var centered: array<f32, ${d}>;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    for (var j = 0u; j < ROW_K; j++) {
      centered[j] = getRoundedDifference(design[designOffset + row * ROW_K + j], means[j], runtimeZero);
    }
    centered[ROW_K] = getRoundedDifference(response[responseOffset + row], means[ROW_K], runtimeZero);
    var slot = 0u;
    for (var first = 0u; first < D; first++) {
      for (var second = first; second < D; second++) {
        accumulator[slot] = accumulator[slot] + centered[first] * centered[second];
        slot = slot + 1u;
      }
    }
    // The deviations from the stored float32 means do not sum to zero: their sums give the part
    // of each true mean that float32 cannot hold.
    for (var column = 0u; column < D; column++) {
      accumulator[${momentCount}u + column] = accumulator[${momentCount}u + column] + centered[column];
    }
  }
  for (var slot = 0u; slot < ${momentCount + d}u; slot++) {
    partial[partialOffset + index * ${momentCount + d}u + slot] = bitcast<u32>(accumulator[slot]);
  }`
    }),
    createKernel(graph, {
      id: `${prefix}-moments-merge`,
      variant: 'moments-merge',
      bindings: [read('partial', props.partialMoments, 'u32'), writeWorkspace(workspace)],
      // The deviation sums follow the moments in the partials and in the workspace.
      invocationCount: momentCount + d,
      declarations: mergeDeclarations,
      body: `${mergeLoop(`${momentCount + d}u`)}
  wsSet(W_MOMENTS + index, sum);`
    })
  ];
  if (tileCount <= REGRESSION_TILE_MERGE_GROUP_SIZE) return nodes;
  return nodes.flatMap(node => {
    if (node.id === `${prefix}-means-merge`) return getHierarchicalMeanMergeNodes(graph, props);
    if (node.id === `${prefix}-moments-merge`)
      return getRowTileMergeNodes(
        graph,
        {
          id: node.id,
          operation: OPERATION,
          variant: 'moments-merge',
          bindings: [read('partial', props.partialMoments, 'u32'), writeWorkspace(workspace)],
          invocationCount: momentCount + d,
          declarations: mergeDeclarations,
          body: `${mergeLoop(`${momentCount + d}u`)}\nwsSet(W_MOMENTS + index, sum);`
        },
        {tileCount, partials: [{name: 'partial', termCount: momentCount + d, bitcastSource: true}]}
      );
    return [node];
  });
}

/** Preserve the first usable global origin and exact integer counts before grouped float sums. */
function getHierarchicalMeanMergeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: AccumulationProps
): GPUCommandNode<Parameters>[] {
  const {prefix, k: predictorCount, tileCount, workspace, partialMeans} = props;
  const dimension = predictorCount + 1;
  const stride = getMeanPartialStride(predictorCount);
  const groupCount = Math.ceil(tileCount / REGRESSION_TILE_MERGE_GROUP_SIZE);
  const metadata = createTransientView(
    graph,
    `${prefix}-mean-group-metadata`,
    'uint32',
    groupCount * 2
  );
  const partial = createTransientView(
    graph,
    `${prefix}-mean-group-sums`,
    'float32',
    groupCount * dimension
  );
  const shared = getSharedWGSL(getWorkspaceLayout(predictorCount), true);
  return [
    createKernel(graph, {
      id: `${prefix}-means-metadata`,
      variant: 'means-metadata',
      invocationCount: groupCount,
      bindings: [read('source', partialMeans, 'u32'), write('metadata', metadata, 'u32')],
      body: `let firstTile = index * 4096u;
  let lastTile = min(firstTile + 4096u, ${tileCount}u);
  var rows = 0u; var referenceTile = firstTile;
  for (var tile = firstTile; tile < lastTile; tile++) {
    let count = source[sourceOffset + tile * ${stride}u + ${2 * dimension}u];
    if (rows == 0u && count > 0u) { referenceTile = tile; }
    rows += count;
  }
  metadata[metadataOffset + index * 2u] = rows;
  metadata[metadataOffset + index * 2u + 1u] = referenceTile;`
    }),
    createKernel(graph, {
      id: `${prefix}-means-reference`,
      variant: 'means-reference',
      invocationCount: dimension,
      bindings: [
        read('source', partialMeans, 'u32'),
        read('metadata', metadata, 'u32'),
        writeWorkspace(workspace)
      ],
      declarations: shared,
      body: `var rows = 0u; var referenceTile = 0u;
  for (var group = 0u; group < ${groupCount}u; group++) {
    let count = metadata[metadataOffset + group * 2u];
    if (rows == 0u && count > 0u) { referenceTile = metadata[metadataOffset + group * 2u + 1u]; }
    rows += count;
  }
  wsSet(W_MEANS + index, bitcast<f32>(source[sourceOffset + referenceTile * ${stride}u + index]));
  if (index == 0u) { ws[wsOffset + W_COUNT] = rows; ws[wsOffset + W_RUNTIME_ZERO] = 0u; }`
    }),
    createKernel(graph, {
      id: `${prefix}-means-groups`,
      variant: 'means-groups',
      invocationCount: groupCount * dimension,
      bindings: [
        read('source', partialMeans, 'u32'),
        readWorkspace(workspace),
        write('partial', partial, 'f32')
      ],
      declarations: getSharedWGSL(getWorkspaceLayout(predictorCount), false),
      body: `let group = index / ${dimension}u; let column = index % ${dimension}u;
  let firstTile = group * 4096u; let lastTile = min(firstTile + 4096u, ${tileCount}u);
  let origin = wsGet(W_MEANS + column);
  let runtimeZero = ws[wsOffset + W_RUNTIME_ZERO];
  ${getPairwiseSumWGSL({
    count: 'lastTile - firstTile',
    setup: `let base = sourceOffset + (firstTile + sumItem) * ${stride}u;
    let rows = source[base + ${2 * dimension}u];
    var contribution = 0.0;
    if (rows > 0u) { contribution = bitcast<f32>(source[base + ${dimension}u + column]) + getRoundedDifference(bitcast<f32>(source[base + column]), origin, runtimeZero) * f32(rows); }`,
    term: 'contribution'
  })}
  partial[partialOffset + index] = sum;`
    }),
    createKernel(graph, {
      id: `${prefix}-means-merge`,
      variant: 'means-merge',
      invocationCount: dimension,
      bindings: [read('partial', partial, 'f32'), writeWorkspace(workspace)],
      declarations: `${shared}\nconst TILE_COUNT: u32 = ${groupCount}u;`,
      body: `${getPairwiseSumWGSL({count: 'TILE_COUNT', term: `partial[partialOffset + sumItem * ${dimension}u + index]`})}
  let rows = ws[wsOffset + W_COUNT];
  wsSet(W_MEANS + index, select(0.0, wsGet(W_MEANS + index) + sum / max(f32(rows), 1.0), rows > 0u));`
    })
  ];
}

/** Single-invocation solve of the (optionally ridge-penalized) scaled normal equations. */
function getSolveBody(k: number, hasRidge: boolean, refineScalarSlope: boolean): string {
  return /* wgsl */ `let rows = ws[wsOffset + W_COUNT];
  ${
    hasRidge
      ? `let requested = params[paramsOffset];
  let lambda = select(0.0, requested, requested > 0.0 && isFiniteValue(requested));`
      : 'let lambda = 0.0;'
  }
  wsSet(W_LAMBDA, lambda);
  if (rows <= K + 1u) {
    ws[wsOffset + W_STATUS] = 2u;
    return;
  }
  var a: array<f32, ${k * k}>;
  var rhs: array<f32, ${k}>;
  var scale: array<f32, ${k}>;
  for (var i = 0u; i < K; i++) {
    let variance = momentAt(i, i);
    if (!(variance > 0.0) || !isFiniteValue(variance)) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
    scale[i] = sqrt(variance);
  }
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j <= i; j++) {
      var entry = momentAt(i, j) / (scale[i] * scale[j]);
      if (i == j) {
        entry = entry + lambda / (scale[i] * scale[i]);
      }
      a[i * K + j] = entry;
    }
    rhs[i] = momentAt(i, K) / scale[i];
  }
  if (!choleskyFactor_${k}(&a)) {
    ws[wsOffset + W_STATUS] = 1u;
    return;
  }
  for (var i = 0u; i < K; i++) {
    if (!(a[i * K + i] >= ${getWGSLFloatLiteral(GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE)})) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
  }
  for (var i = 0u; i < KK; i++) {
    wsSet(W_FACTOR + i, a[i]);
  }
  choleskySolve_${k}(&a, &rhs);
  var residualOffset = getMeanCorrection(K);
  for (var i = 0u; i < K; i++) {
    wsSet(W_SCALE + i, scale[i]);
    ${
      refineScalarSlope
        ? `var slope = rhs[i] / scale[i];
    let variance = momentAt(i, i);
    if (variance >= 1.1754943508222875e-38 && variance <= 8.507059173023462e37) {
      let refined = divideRegressionPair(vec2f(momentAt(i, K), 0.0), vec2f(variance, 0.0)).x;
      if (isFiniteValue(refined)) { slope = refined; }
    }
    wsSet(W_SLOPES + i, slope);
    residualOffset = residualOffset - slope * getMeanCorrection(i);`
        : `wsSet(W_SLOPES + i, rhs[i] / scale[i]);
    residualOffset = residualOffset - (rhs[i] / scale[i]) * getMeanCorrection(i);`
    }
  }
  wsSet(W_RESIDUAL_OFFSET, residualOffset);
  ws[wsOffset + W_STATUS] = 0u;`;
}

function getFinishBody(k: number): string {
  const p = k + 1;
  return /* wgsl */ `let rows = ws[wsOffset + W_COUNT];
  let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  statusOut[statusOutOffset] = status;
  if (status != 0u) {
    for (var i = 0u; i < ${p}u; i++) {
      coefficientsOut[coefficientsOutOffset + i] = nan;
      standardErrorsOut[standardErrorsOutOffset + i] = nan;
      tStatisticsOut[tStatisticsOutOffset + i] = nan;
    }
    for (var slot = 0u; slot < ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH}u; slot++) {
      summaryOut[summaryOutOffset + slot] = nan;
    }
    summaryOut[summaryOutOffset] = f32(rows);
    return;
  }
  let n = f32(rows);
  let freedom = n - ${p}.0;
  let residualSumSquares = wsGet(W_RESIDUAL_SUMS);
  let totalSumSquares = momentAt(K, K);
  var a: array<f32, ${k * k}>;
  for (var i = 0u; i < KK; i++) {
    a[i] = wsGet(W_FACTOR + i);
  }
  let sigmaSquared = residualSumSquares / freedom;
  var intercept = wsGet(W_MEANS + K);
  var u: array<f32, ${k}>;
  for (var i = 0u; i < K; i++) {
    intercept = getRoundedDifference(intercept, wsGet(W_SLOPES + i) * wsGet(W_MEANS + i), ws[wsOffset + W_RUNTIME_ZERO]);
    u[i] = wsGet(W_MEANS + i) / wsGet(W_SCALE + i);
  }
  intercept = intercept + wsGet(W_RESIDUAL_OFFSET);
  coefficientsOut[coefficientsOutOffset] = intercept;
  for (var i = 0u; i < K; i++) {
    coefficientsOut[coefficientsOutOffset + 1u + i] = wsGet(W_SLOPES + i);
  }
  // Intercept variance: sigma2 (1 / n + u' R^-1 u) with u = xbar / scale.
  choleskySolve_${k}(&a, &u);
  var quadratic = 0.0;
  for (var i = 0u; i < K; i++) {
    quadratic = quadratic + (wsGet(W_MEANS + i) / wsGet(W_SCALE + i)) * u[i];
  }
  standardErrorsOut[standardErrorsOutOffset] = sqrt(max(sigmaSquared * (1.0 / n + quadratic), 0.0));
  var maximumInflation = 0.0;
  for (var i = 0u; i < K; i++) {
    let scale = wsGet(W_SCALE + i);
    // Diagonal of the inverse correlation matrix is the variance inflation factor of column i.
    let correlationInverse = choleskyInverseDiagonal_${k}(&a, i);
    maximumInflation = max(maximumInflation, correlationInverse);
    let inverse = correlationInverse / (scale * scale);
    standardErrorsOut[standardErrorsOutOffset + 1u + i] = sqrt(max(sigmaSquared * inverse, 0.0));
  }
  for (var i = 0u; i < ${p}u; i++) {
    let standardError = standardErrorsOut[standardErrorsOutOffset + i];
    tStatisticsOut[tStatisticsOutOffset + i] = select(
      nan,
      coefficientsOut[coefficientsOutOffset + i] / standardError,
      standardError > 0.0
    );
  }
  let rSquared = select(nan, 1.0 - residualSumSquares / totalSumSquares, totalSumSquares > 0.0);
  let meanSquare = max(residualSumSquares / n, 1.17549435e-38);
  let logLikelihood = -0.5 * n * (1.8378770664093453 + log(meanSquare) + 1.0);
  var jarqueBera = nan;
  var jarqueBeraP = nan;
  var skewness = nan;
  var kurtosis = nan;
  let m2 = wsGet(W_STANDARDIZED_SUMS) / n;
  if (m2 > 0.0) {
    let m3 = wsGet(W_STANDARDIZED_SUMS + 1u) / n;
    let m4 = wsGet(W_STANDARDIZED_SUMS + 2u) / n;
    skewness = m3 / (m2 * sqrt(m2));
    kurtosis = m4 / (m2 * m2);
    let excess = kurtosis - 3.0;
    jarqueBera = n / 6.0 * (skewness * skewness + 0.25 * excess * excess);
    jarqueBeraP = exp(-0.5 * jarqueBera);
  }
  for (var slot = 0u; slot < ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH}u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset + 0u] = n;
  summaryOut[summaryOutOffset + 1u] = rSquared;
  summaryOut[summaryOutOffset + 2u] = 1.0 - (1.0 - rSquared) * (n - 1.0) / freedom;
  summaryOut[summaryOutOffset + 3u] = sigmaSquared;
  summaryOut[summaryOutOffset + 4u] = logLikelihood;
  summaryOut[summaryOutOffset + 5u] = 2.0 * ${p}.0 - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 6u] = ${p}.0 * log(n) - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 7u] = jarqueBera;
  summaryOut[summaryOutOffset + 8u] = jarqueBeraP;
  summaryOut[summaryOutOffset + 11u] = residualSumSquares;
  summaryOut[summaryOutOffset + 12u] = totalSumSquares;
  summaryOut[summaryOutOffset + 13u] = skewness;
  summaryOut[summaryOutOffset + 14u] = kurtosis;
  summaryOut[summaryOutOffset + 15u] = wsGet(W_LAMBDA);

  // Conditioning. Belsley (spreg condition_index) of the raw [1, X] design, from the Cholesky
  // factor of the unpenalized predictor correlation matrix and the column means, never from the
  // raw Gram matrix (its smallest eigenvalue is below float32 resolution once the condition
  // number passes a few hundred). See getBelsleyConditionNumberWGSL.
  var unpenalized: array<f32, ${k * k}>;
  var alpha: array<f32, ${k}>;
  var beta: array<f32, ${k}>;
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j < K; j++) {
      unpenalized[i * K + j] = momentAt(i, j) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + j));
    }
    let centered = momentAt(i, i);
    let meanI = wsGet(W_MEANS + i);
    let rawNorm = sqrt(n * meanI * meanI + centered);
    alpha[i] = sqrt(n) * meanI / rawNorm;
    beta[i] = sqrt(centered) / rawNorm;
  }
  var belsley = 3.0e38;
  if (choleskyFactor_${k}(&unpenalized)) {
    belsley = getBelsleyConditionNumber_${k}(&unpenalized, &alpha, &beta);
  }
  // The factored (optionally ridge-penalized) predictor correlation matrix.
  var correlation: array<f32, ${k * k}>;
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j < K; j++) {
      var entry = momentAt(i, j) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + j));
      if (i == j) {
        entry = entry + wsGet(W_LAMBDA) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + i));
      }
      correlation[i * K + j] = entry;
    }
  }
  let correlationRange = getSymmetricEigenvalueRange_${k}(&correlation);
  let correlationCondition = correlationRange.y / max(correlationRange.x, 1.0e-30);
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER}u] =
    belsley;
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER}u] =
    correlationCondition;
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR}u] =
    maximumInflation;
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_PRECISION_WARNING}u] =
    select(0.0, 1.0, correlationCondition > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)});`;
}

/** Regularized upper incomplete gamma for the chi-square survival function, fixed iteration cap. */
function getChiSquareWGSL(degreesOfFreedom: number): string {
  return /* wgsl */ `
const DEGREES_OF_FREEDOM: u32 = ${degreesOfFreedom}u;
const LOG_GAMMA: f32 = ${getWGSLFloatLiteral(getLogGammaOfHalfInteger(degreesOfFreedom))};

fn getChiSquareSurvival(value: f32) -> f32 {
  if (value <= 0.0) {
    return 1.0;
  }
  if (!isFiniteValue(value)) {
    return select(getNaN(), 0.0, value > 0.0);
  }
  let a = f32(DEGREES_OF_FREEDOM) * 0.5;
  let x = value * 0.5;
  let logPrefactor = -x + a * log(x) - LOG_GAMMA;
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= ${CHI_SQUARE_MAXIMUM_ITERATIONS}u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (term < sum * 1.0e-7) {
        break;
      }
    }
    return clamp(1.0 - sum * exp(logPrefactor), 0.0, 1.0);
  }
  let tiny = 1.0e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= ${CHI_SQUARE_MAXIMUM_ITERATIONS}u; iteration++) {
    let fi = f32(iteration);
    let an = -fi * (fi - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < 1.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`;
}

function getBreuschPaganBody(k: number): string {
  return /* wgsl */ `let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  var koenker = nan;
  var koenkerP = nan;
  var breusch = nan;
  var breuschP = nan;
  let totalSumSquares = momentAt(K, K);
  if (status == 0u) {
    var explained = 0.0;
    for (var i = 0u; i < ${k}u; i++) {
      explained = explained + wsGet(W_SLOPES + i) * momentAt(i, K);
    }
    let n = f32(ws[wsOffset + W_COUNT]);
    if (totalSumSquares > 0.0) {
      koenker = n * clamp(explained / totalSumSquares, 0.0, 1.0);
      koenkerP = getChiSquareSurvival(koenker);
    }
    // Classic BP uses explained SS, even when the squared residuals have zero variance.
    let normalizedSquaredSum = bitcast<f32>(mainWs[mainWsOffset + W_STANDARDIZED_SUMS]);
    if (normalizedSquaredSum > 0.0) {
      let meanSquare = normalizedSquaredSum / n;
      breusch = max(0.5 * (explained / meanSquare) / meanSquare, 0.0);
      breuschP = getChiSquareSurvival(breusch);
    }
  }
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT}u] = koenker;
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE}u] = koenkerP;
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN}u] = breusch;
  summaryOut[summaryOutOffset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE}u] = breuschP;`;
}

function read(name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding {
  return {name, view, type, access: 'read'};
}

function write(name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding {
  return {name, view, type, access: 'read_write'};
}

function readWorkspace(view: GraphDataView<'uint32'>): WGSLKernelBinding {
  return read('ws', view, 'u32');
}

function writeWorkspace(view: GraphDataView<'uint32'>): WGSLKernelBinding {
  return write('ws', view, 'u32');
}

type KernelProps = {
  id: string;
  variant: string;
  bindings: readonly WGSLKernelBinding[];
  invocationCount: number;
  declarations?: string;
  body: string;
  workgroupSize?: number;
};

function createKernel<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KernelProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    operation: OPERATION,
    ...props
  });
}
