// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUInferenceMetadata} from './inference';
import {validateRequirement} from '../kernels/gpu-contributor-utils';

/** Regression estimators that report a {@link GPURegressionModelResult}. */
export type GPURegressionModelKind =
  | 'ordinary-least-squares'
  | 'spatial-lag-two-stage-least-squares'
  | 'spatial-error-generalized-moments'
  | 'spatial-lag-maximum-likelihood'
  | 'spatial-error-maximum-likelihood'
  | 'geographically-weighted-regression'
  | 'geographically-temporally-weighted-regression'
  | 'spatial-panel-error-maximum-likelihood';

/** Penalty applied to the fitted coefficients; this metadata does not alter their interpretation. */
export type GPURegressionRegularization = {
  kind: 'ridge';
  /** Finite, non-negative ridge penalty in the estimator's documented design scale. */
  lambda: number;
};

/** Reference distribution of a coefficient test statistic. */
export type GPURegressionStatisticKind = 't' | 'z';

/** What a coefficient row estimates. */
export type GPURegressionCoefficientRole =
  | 'intercept'
  | 'predictor'
  | 'spatial-lag'
  | 'spatial-error'
  | 'lagged-predictor';

/** One row of a regression coefficient table. NaN marks a quantity the estimator does not define. */
export type GPURegressionCoefficient = {
  /** Display name, for example `'CONSTANT'`, a predictor name, `'W_y'` or `'lambda'`. */
  name: string;
  role: GPURegressionCoefficientRole;
  estimate: number;
  standardError: number;
  /** `estimate / standardError`. */
  statistic: number;
  statisticKind: GPURegressionStatisticKind;
  /** Two-sided p-value of `statistic` under its reference distribution. */
  pValue: number;
};

/**
 * Residual-variance conventions. Every model result reports both: `sigmaSquared` is always
 * `RSS / (n - k)` and `sigmaSquaredMaximumLikelihood` is always `RSS / n`.
 * `standardErrorSigmaSquared` names the coefficient-error scale, or `'robust'` for a sandwich
 * covariance that is not a scalar residual variance times an inverse normal matrix.
 */
export type GPUSigmaSquaredConvention = 'degrees-of-freedom' | 'maximum-likelihood' | 'robust';

/**
 * Collinearity and float32 precision evidence of a model's normal equations.
 *
 * - `conditionNumber` is the Belsley multicollinearity condition number of the design including
 *   the intercept, `sqrt(lambdaMax / lambdaMin)` of `X'X` scaled to unit diagonal (spreg and
 *   GeoDa `condition_index`). Values above 30 indicate harmful collinearity.
 * - `correlationConditionNumber` is `lambdaMax / lambdaMin` of the predictor correlation matrix
 *   that the float32 Cholesky solve actually factors (intercept centered out). Relative coefficient
 *   error is about `correlationConditionNumber * 6e-8`.
 * - `maximumVarianceInflationFactor` is the largest diagonal entry of the inverse correlation
 *   matrix, `max_j 1 / (1 - R_j^2)`.
 * - `illConditioned` is true when `correlationConditionNumber` exceeds
 *   `GPU_REGRESSION_PRECISION_CONDITION_LIMIT`, that is when float32 normal equations may have
 *   lost more than about three of their seven significant digits. The fit is still reported.
 */
export type GPURegressionConditioning = {
  conditionNumber: number;
  correlationConditionNumber: number;
  maximumVarianceInflationFactor: number;
  illConditioned: boolean;
};

/**
 * Largest correlation-matrix condition number at which a float32 normal-equation solve is trusted
 * to about 1e-3 relative accuracy (`1e4 * 2^-24` is `6e-4`). Above it a model result sets
 * `conditioning.illConditioned`.
 */
export const GPU_REGRESSION_PRECISION_CONDITION_LIMIT = 1e4;

/** A specification test reported with a model (Jarque-Bera, Breusch-Pagan, Anselin-Kelejian, ...). */
export type GPURegressionTest = {
  statistic: number;
  /** Degrees of freedom of a chi-square test, or 0 for a standard normal test. */
  degreesOfFreedom: number;
  pValue: number;
};

/** Outcome of a model fit. */
export type GPURegressionModelStatus = {
  /** Estimator-specific status word read back from the GPU; 0 is success. */
  code: number;
  /** True when `code` is 0 and the estimates are defined. */
  ok: boolean;
  /** Iterative estimators: true when the optimizer met its stopping rule; null for closed forms. */
  converged: boolean | null;
  /** Copy of `conditioning.illConditioned`, false when conditioning is not reported. */
  illConditioned: boolean;
  /** Estimator-specific name supplied by its named status decoder. */
  name?: string;
  /** Optional explanation of the estimator-specific status. */
  reason?: string;
  /** Named decoder flags; these do not change `code` or the existing `ok` predicate. */
  excluded?: boolean;
  failed?: boolean;
};

/**
 * CPU-side summary of one regression fit, built from read-back GPU outputs by the estimator's
 * `get...ModelResult` function. The same fields mean the same thing for every estimator, so a UI
 * can render or compare OLS, spatial lag and spatial error fits without per-model slot tables.
 *
 * Conventions:
 * - `parameterCount` (k) counts every estimated coefficient row, including the intercept and the
 *   spatial parameter (`rho` or `lambda`), but not `sigma2`; for GWR it is the effective number
 *   of parameters `tr(S)` and `coefficients` is empty (also for GTWR). Panel-error ML counts
 *   all absorbed entity intercepts, predictors and lambda: `k = entityCount + predictorCount + 1`,
 *   even though the coefficient table contains only one absorbed-intercept placeholder.
 *   `degreesOfFreedom` is `n - k`.
 * - `sigmaSquared` is `RSS / (n - k)`; `sigmaSquaredMaximumLikelihood` is `RSS / n`.
 * - `logLikelihood`, `akaikeInformationCriterion` and `schwarzCriterion` are null where the
 *   estimator does not define a likelihood (2SLS, GM). Where defined, AIC is
 *   `2 (k + extra) - 2 logLikelihood` with the estimator's documented parameter count.
 * - p-values are two-sided; `inference.alternative` is `'two-sided'` and `permutationCount` 0.
 */
export type GPURegressionModelResult = {
  kind: GPURegressionModelKind;
  coefficients: GPURegressionCoefficient[];
  /** Rows used by the fit (finite, unmasked). */
  rowCount: number;
  parameterCount: number;
  degreesOfFreedom: number;
  sigmaSquared: number;
  sigmaSquaredMaximumLikelihood: number;
  standardErrorSigmaSquared: GPUSigmaSquaredConvention;
  residualSumOfSquares: number;
  /** `1 - RSS / TSS` for OLS; NaN for estimators that only define a pseudo R squared. */
  rSquared: number;
  /** Squared correlation of observed and predicted response; NaN where not reported. */
  pseudoRSquared: number;
  logLikelihood: number | null;
  akaikeInformationCriterion: number | null;
  schwarzCriterion: number | null;
  /**
   * Corrected AIC, `AIC + 2 k (k + 1) / (n - k - 1)` for global models; for GWR/GTWR the Hurvich form
   * with `k = tr(S)` that bandwidth selection minimizes. The derived global correction is null
   * when no likelihood is defined or `n <= k + 1`, where `k` is the information-criterion
   * parameter count. Estimator-specific overrides are preserved.
   */
  correctedAkaikeInformationCriterion: number | null;
  conditioning: GPURegressionConditioning | null;
  /** Named specification tests; keys are camelCase test names such as `jarqueBera`. */
  tests: Record<string, GPURegressionTest>;
  status: GPURegressionModelStatus;
  inference: GPUInferenceMetadata;
  /** Present when a penalty was applied; omission does not assert an unpenalized fit. */
  regularization?: GPURegressionRegularization;
  /**
   * Estimator-supplied description of coefficient inference, for example `'ridge-approximate'`
   * for the penalized inverse-normal-matrix uncertainty convention, rather than ridge sandwich
   * covariance. Omission preserves the estimator's existing documented inference convention.
   */
  inferenceMethod?: string;
};

/** Inputs of {@link createGPURegressionModelResult}; derived fields are computed from these. */
export type GPURegressionModelResultInput = {
  kind: GPURegressionModelKind;
  coefficients: GPURegressionCoefficient[];
  rowCount: number;
  residualSumOfSquares: number;
  standardErrorSigmaSquared: GPUSigmaSquaredConvention;
  statusCode: number;
  converged?: boolean | null;
  rSquared?: number;
  pseudoRSquared?: number;
  logLikelihood?: number | null;
  /**
   * Parameter count `k` for degrees of freedom and sigma squared. Required for panel-error ML
   * (including every absorbed entity intercept) and GTWR (effective count `tr(S)`). Otherwise
   * defaults to `coefficients.length`; GWR passes its effective number of parameters `tr(S)`.
   */
  parameterCount?: number;
  /** Parameters counted by AIC and Schwarz: GTWR requires `tr(S) + 1`; otherwise defaults to k. */
  informationCriterionParameterCount?: number;
  /** Overrides the derived corrected AIC (GWR reports its own). */
  correctedAkaikeInformationCriterion?: number | null;
  conditioning?: Omit<GPURegressionConditioning, 'illConditioned'> | null;
  tests?: Record<string, GPURegressionTest>;
  /** Penalty evidence carried into the result without changing fitted quantities. */
  regularization?: GPURegressionRegularization;
  /** Estimator-supplied description of the inference convention, including approximations. */
  inferenceMethod?: string;
};

/**
 * Builds a {@link GPURegressionModelResult} with every derived field (degrees of freedom, both
 * sigma squared conventions, information criteria, the ill-conditioning flag) computed in one
 * place, so all estimators apply the same conventions.
 */
export function createGPURegressionModelResult(
  input: GPURegressionModelResultInput
): GPURegressionModelResult {
  const id = 'createGPURegressionModelResult';
  const suppliedRegularization = input.regularization;
  const regularization =
    suppliedRegularization === undefined ? undefined : {...suppliedRegularization};
  if (regularization !== undefined) {
    validateRequirement(
      id,
      regularization.kind === 'ridge',
      `regularization.kind must be 'ridge'; received ${String(regularization.kind)}`
    );
    validateRequirement(
      id,
      typeof regularization.lambda === 'number' &&
        Number.isFinite(regularization.lambda) &&
        regularization.lambda >= 0,
      `regularization.lambda must be finite and >= 0; received ${String(regularization.lambda)}`
    );
  }
  const panelError = input.kind === 'spatial-panel-error-maximum-likelihood';
  const geographicallyTemporallyWeighted =
    input.kind === 'geographically-temporally-weighted-regression';
  const suppliedParameterCount = input.parameterCount;
  if (panelError) {
    validateRequirement(
      id,
      suppliedParameterCount !== undefined &&
        Number.isSafeInteger(suppliedParameterCount) &&
        suppliedParameterCount > 0 &&
        suppliedParameterCount >= input.coefficients.length,
      `parameterCount must be a positive safe integer >= coefficients.length (${input.coefficients.length}) including all absorbed entity intercepts, predictors and lambda for panel-error ML; received ${String(suppliedParameterCount)}`
    );
  }
  if (geographicallyTemporallyWeighted) {
    validateRequirement(
      id,
      suppliedParameterCount !== undefined &&
        Number.isFinite(suppliedParameterCount) &&
        suppliedParameterCount >= 0,
      `parameterCount must explicitly supply finite non-negative tr(S) for GTWR; received ${String(suppliedParameterCount)}`
    );
  }
  const parameterCount = suppliedParameterCount ?? input.coefficients.length;
  const rowCount = input.rowCount;
  const degreesOfFreedom = rowCount - parameterCount;
  const residualSumOfSquares = input.residualSumOfSquares;
  const logLikelihood = input.logLikelihood ?? null;
  const suppliedCriterionParameterCount = input.informationCriterionParameterCount;
  if (geographicallyTemporallyWeighted) {
    validateRequirement(
      id,
      suppliedCriterionParameterCount === undefined ||
        suppliedCriterionParameterCount === parameterCount + 1,
      `informationCriterionParameterCount must equal tr(S) + 1 (${parameterCount + 1}) for GTWR; received ${String(suppliedCriterionParameterCount)}`
    );
  }
  const criterionParameterCount =
    suppliedCriterionParameterCount ??
    (geographicallyTemporallyWeighted ? parameterCount + 1 : parameterCount);
  const correctionDenominator = geographicallyTemporallyWeighted
    ? rowCount - parameterCount - 2
    : rowCount - criterionParameterCount - 1;
  const conditioning = input.conditioning
    ? {
        ...input.conditioning,
        illConditioned:
          input.conditioning.correlationConditionNumber > GPU_REGRESSION_PRECISION_CONDITION_LIMIT
      }
    : null;
  const ok = input.statusCode === 0;
  return {
    kind: input.kind,
    coefficients: input.coefficients,
    rowCount,
    parameterCount,
    degreesOfFreedom,
    sigmaSquared: degreesOfFreedom > 0 ? residualSumOfSquares / degreesOfFreedom : NaN,
    sigmaSquaredMaximumLikelihood: rowCount > 0 ? residualSumOfSquares / rowCount : NaN,
    standardErrorSigmaSquared: input.standardErrorSigmaSquared,
    residualSumOfSquares,
    rSquared: input.rSquared ?? NaN,
    pseudoRSquared: input.pseudoRSquared ?? NaN,
    logLikelihood,
    akaikeInformationCriterion:
      panelError || logLikelihood === null ? null : 2 * criterionParameterCount - 2 * logLikelihood,
    schwarzCriterion:
      panelError || logLikelihood === null
        ? null
        : criterionParameterCount * Math.log(rowCount) - 2 * logLikelihood,
    correctedAkaikeInformationCriterion: panelError
      ? null
      : input.correctedAkaikeInformationCriterion !== undefined
        ? input.correctedAkaikeInformationCriterion
        : logLikelihood === null || correctionDenominator <= 0
          ? null
          : geographicallyTemporallyWeighted
            ? -2 * logLikelihood + (2 * rowCount * (parameterCount + 1)) / correctionDenominator
            : 2 * criterionParameterCount -
              2 * logLikelihood +
              (2 * criterionParameterCount * (criterionParameterCount + 1)) / correctionDenominator,
    conditioning,
    tests: input.tests ?? {},
    status: {
      code: input.statusCode,
      ok,
      converged: input.converged ?? null,
      illConditioned: conditioning?.illConditioned ?? false
    },
    inference: {
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: 'none',
      testedRowsDefinition: 'model coefficients'
    },
    ...(regularization === undefined ? {} : {regularization}),
    ...(input.inferenceMethod === undefined ? {} : {inferenceMethod: input.inferenceMethod})
  };
}

/** Default coefficient names: `'CONSTANT'` then `'x0'`, `'x1'`, ... (spreg's default naming). */
export function getGPURegressionPredictorNames(
  predictorCount: number,
  predictorNames?: readonly string[]
): string[] {
  const names = ['CONSTANT'];
  for (let column = 0; column < predictorCount; column++) {
    names.push(predictorNames?.[column] ?? `x${column}`);
  }
  return names;
}
