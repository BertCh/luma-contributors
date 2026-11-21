// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateRegressionStatusReadback,
  type RegressionStatusReadback
} from './regression-status-readback';

import {clamp} from '@math.gl/core';
import {
  createGPURegressionModelResult,
  getGPURegressionPredictorNames,
  type GPURegressionCoefficient,
  type GPURegressionModelResult
} from '../../analysis-core/contracts/model-result';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT
} from './ordinary-least-squares-parameters';
import {getBetaContinuedFraction} from './incomplete-beta';
import {getStudentTPositiveTail} from './student-t-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Natural log of the gamma function at `degreesOfFreedom / 2` for a positive integer
 * `degreesOfFreedom`, by the half-integer recurrence for small arguments and the Stirling
 * expansion for arguments at least 32 (avoids a row-count-sized recurrence).
 */
export function getLogGammaOfHalfInteger(degreesOfFreedom: number): number {
  if (!Number.isInteger(degreesOfFreedom) || degreesOfFreedom < 1) {
    throw new Error('Chi-square degrees of freedom must be a positive integer');
  }
  const target = degreesOfFreedom / 2;
  if (target >= 32) {
    return (
      (target - 0.5) * Math.log(target) -
      target +
      0.5 * Math.log(2 * Math.PI) +
      getStirlingCorrection(target)
    );
  }
  let argument = degreesOfFreedom % 2 === 0 ? 1 : 0.5;
  let logGamma = degreesOfFreedom % 2 === 0 ? 0 : 0.5 * Math.log(Math.PI);
  while (argument < target) {
    logGamma += Math.log(argument);
    argument += 1;
  }
  return logGamma;
}

/** Fixed iteration cap of the GPU's small-shape chi-square series and continued fraction. */
export const CHI_SQUARE_MAXIMUM_ITERATIONS = 200;

function getStirlingCorrection(argument: number): number {
  const inverse = 1 / argument;
  const squared = inverse * inverse;
  return (
    inverse *
    (1 / 12 + squared * (-1 / 360 + squared * (1 / 1260 + squared * (-1 / 1680 + squared / 1188))))
  );
}

/**
 * Survival function `P(X > value)` of a chi-square distribution with `degreesOfFreedom` degrees of
 * freedom, computed as the regularized upper incomplete gamma `Q(df / 2, value / 2)`.
 *
 * Uses a converged power series or modified Lentz continued fraction, with a shape-dependent
 * host budget. For a >= 1e8, the uniform large-shape incomplete-gamma expansion avoids work
 * proportional to sqrt(a). The GPU uses its separate 200-step budget only for df <= 15.
 * Returns NaN for a NaN value, 1 for `value <= 0`, and 0 for positive infinity.
 */
export function getChiSquareSurvival(value: number, degreesOfFreedom: number): number {
  const logGamma = getLogGammaOfHalfInteger(degreesOfFreedom);
  if (Number.isNaN(value)) {
    return NaN;
  }
  if (!(value > 0)) {
    return 1;
  }
  if (value === Infinity) {
    return 0;
  }
  const a = degreesOfFreedom / 2;
  const x = value / 2;
  if (x === 0) return 1;
  if (a >= 1e8) {
    // Uniform gamma expansion Q(a,x) = .5 erfc(eta sqrt(a/2))
    // + exp(-a eta²/2)/sqrt(2 pi a) * (1/sigma - 1/eta) + O(a^-3/2).
    // sigma=x/a-1, eta=sign(sigma)*sqrt(2*(sigma-log1p(sigma))).
    const sigma = (x - a) / a;
    const difference =
      Math.abs(sigma) < 1e-3
        ? sigma *
          sigma *
          (0.5 +
            sigma * (-1 / 3 + sigma * (1 / 4 + sigma * (-1 / 5 + sigma * (1 / 6 - sigma / 7)))))
        : sigma - Math.log1p(sigma);
    const eta = Math.sign(sigma) * Math.sqrt(2 * difference);
    const normal = eta * Math.sqrt(a);
    const halfTail = 0.5 * getChiSquareSurvival(normal * normal, 1);
    const leading = normal >= 0 ? halfTail : 1 - halfTail;
    const correction =
      Math.abs(sigma) < 1e-3
        ? -1 / 3 + sigma / 12 - (23 * sigma * sigma) / 540
        : 1 / sigma - 1 / eta;
    return clamp(
      leading +
        (Math.exp(-0.5 * normal * normal) / Math.sqrt(2 * Math.PI) / Math.sqrt(a)) * correction,
      0,
      1
    );
  }
  const relative = (x - a) / a;
  const logPrefactor =
    a >= 32
      ? a * (Math.log1p(relative) - relative) +
        0.5 * Math.log(a) -
        0.5 * Math.log(2 * Math.PI) -
        getStirlingCorrection(a)
      : -x + a * Math.log(x) - logGamma;
  const maximumIterations = Math.max(
    CHI_SQUARE_MAXIMUM_ITERATIONS,
    Math.ceil(64 * Math.sqrt(a) + 1000)
  );
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let iteration = 1; iteration <= maximumIterations; iteration++) {
      term *= x / (a + iteration);
      sum += term;
      if (term < sum * 1e-15) {
        return clamp(1 - sum * Math.exp(logPrefactor), 0, 1);
      }
    }
    throw new Error(
      'getChiSquareSurvival series did not converge within the host iteration budget'
    );
  }
  return clamp(
    Math.exp(logPrefactor) * getChiSquareContinuedFraction(a, x, maximumIterations),
    0,
    1
  );
}

function getChiSquareContinuedFraction(
  shape: number,
  argument: number,
  maximumIterations: number
): number {
  const tiny = 1e-300;
  let b = argument + 1 - shape;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let iteration = 1; iteration <= maximumIterations; iteration++) {
    const an = -iteration * (iteration - shape);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (Math.abs(c) < tiny) {
      c = tiny;
    }
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) {
      return h;
    }
  }
  throw new Error(
    'getChiSquareSurvival continued fraction did not converge within the host iteration budget'
  );
}

/**
 * Two-sided p-value `P(|T| > |t|)` of a Student t statistic with an integer `degreesOfFreedom`,
 * the regularized incomplete beta `I_x(df / 2, 1 / 2)` with `x = df / (df + t^2)`, in float64
 * (Numerical Recipes continued fraction). Returns NaN for a NaN statistic or non-positive
 * degrees of freedom and 0 for an infinite statistic.
 * At df >= 1e10, integrates the normal-density expansion through df^-2 to retain tails
 * that a rounded near-one beta argument cannot represent.
 */
export function getStudentTTwoSidedPValue(statistic: number, degreesOfFreedom: number): number {
  return getStudentTTail(statistic, degreesOfFreedom, 1);
}

function getStudentTTail(
  statistic: number,
  degreesOfFreedom: number,
  tailFraction: 0.5 | 1
): number {
  if (Number.isNaN(statistic) || !Number.isInteger(degreesOfFreedom) || degreesOfFreedom < 1) {
    return NaN;
  }
  if (!Number.isFinite(statistic)) {
    return 0;
  }
  const magnitude = Math.abs(statistic);
  if (magnitude === 0) {
    return tailFraction;
  }
  if (degreesOfFreedom >= 1e10) {
    // Integrating the t-density's expansion about the normal density avoids a beta argument
    // rounded to one. Through order df^-2, for 0<=|t|<=40 the omitted relative tail term is
    // below about 3e-14 at this threshold. Beyond 40 the tail is below float64's range.
    if (magnitude > 40) return 0;
    const squared = magnitude * magnitude;
    const first = (magnitude * (squared + 1)) / 4;
    const second = (magnitude * (3 * squared ** 3 - 7 * squared ** 2 - 5 * squared - 3)) / 96;
    const inverse = 1 / degreesOfFreedom;
    if (magnitude >= 37) {
      // Factor out the shared Gaussian exponential before adding the two-sided terms.
      // Split the exponential so only the final product can round to a subnormal or zero.
      const normalFactor =
        magnitude * getChiSquareContinuedFraction(0.5, squared / 2, CHI_SQUARE_MAXIMUM_ITERATIONS);
      const correction = 2 * inverse * (first + inverse * second);
      const halfExponential = Math.exp(-0.25 * squared);
      const tailFactor = (tailFraction * (normalFactor + correction)) / Math.sqrt(2 * Math.PI);
      const scaledTail = halfExponential * tailFactor;
      return scaledTail * halfExponential;
    }
    const normalTail = 0.5 * getChiSquareSurvival(squared, 1);
    const density = Math.exp(-0.5 * squared) / Math.sqrt(2 * Math.PI);
    return (
      tailFraction * clamp(2 * (normalTail + density * inverse * (first + inverse * second)), 0, 1)
    );
  }
  const twoSided = getStudentTPositiveTail(magnitude, degreesOfFreedom);
  if (tailFraction === 0.5 && twoSided < 2 ** -1021) {
    // A half that can enter the subnormal range needs the beta prefactor and the
    // one-sided factor combined before exponentiation. This small tail uses the
    // direct (rather than complementary) continued fraction.
    return getStudentTSmallSurvival(magnitude, degreesOfFreedom);
  }
  return tailFraction * twoSided;
}

function getStudentTSmallSurvival(magnitude: number, degreesOfFreedom: number): number {
  const shape = degreesOfFreedom / 2;
  const logSquaredRatio = 2 * Math.log(magnitude / Math.sqrt(degreesOfFreedom));
  const logScale = Math.log1p(Math.exp(-Math.abs(logSquaredRatio)));
  const logArgument = -Math.max(logSquaredRatio, 0) - logScale;
  const logComplement = Math.min(logSquaredRatio, 0) - logScale;
  const logGammaRatio =
    shape >= 32
      ? 0.5 * Math.log(shape) +
        shape * Math.log1p(0.5 / shape) -
        0.5 +
        getStirlingCorrection(shape + 0.5) -
        getStirlingCorrection(shape)
      : getLogGammaOfHalfInteger(degreesOfFreedom + 1) - getLogGammaOfHalfInteger(degreesOfFreedom);
  const fraction = getBetaContinuedFraction(shape, 0.5, Math.exp(logArgument));
  return Math.exp(
    logGammaRatio -
      0.5 * Math.log(Math.PI) +
      shape * logArgument +
      0.5 * logComplement +
      Math.log(fraction / shape) -
      Math.LN2
  );
}

/**
 * Survival function `P(T > t)` of a Student t distribution with an integer `degreesOfFreedom`
 * (scipy `stats.t.sf`). The one-sided factor is included before rounding a subnormal tail.
 */
export function getStudentTSurvival(statistic: number, degreesOfFreedom: number): number {
  const positiveTail = getStudentTTail(statistic, degreesOfFreedom, 0.5);
  return statistic >= 0 ? positiveTail : 1 - positiveTail;
}

/** Read-back arrays of one `GPUOrdinaryLeastSquares` fit. */
export type GPUOrdinaryLeastSquaresReadback = {
  coefficients: Float32Array;
  standardErrors: Float32Array;
  tStatistics: Float32Array;
  /** At least `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH` values. */
  summary: Float32Array;
  status: Uint32Array;
};

/** Status may also be supplied as an exact buffer read. */
export type GPUOrdinaryLeastSquaresReadbackInput = Omit<
  GPUOrdinaryLeastSquaresReadback,
  'status'
> & {status: RegressionStatusReadback};

/** Named OLS report with the original packed summary retained as an independent snapshot. */
export type GPUOrdinaryLeastSquaresModelResult = GPURegressionModelResult & {summary: Float32Array};

/** Options of {@link getGPUOrdinaryLeastSquaresModelResult}. */
export type GPUOrdinaryLeastSquaresModelResultOptions = {
  /** Active predictor count for readbacks with spare coefficient capacity; defaults to coefficients.length - 1. */
  predictorCount?: number;
  /** Predictor names after `'CONSTANT'`; missing entries default to `'x0'`, `'x1'`, ... */
  predictorNames?: readonly string[];
};

/**
 * Builds the shared {@link GPURegressionModelResult} from the read-back outputs of
 * `GPUOrdinaryLeastSquares`.
 *
 * The predictor count defaults to `coefficients.length - 1`; pass `options.predictorCount` when
 * arrays have spare capacity. All three coefficient arrays must hold the active intercept-first
 * prefix of `predictorCount + 1` entries. Array references are captured once, so accessor-backed
 * readbacks validate and decode the same arrays. Coefficient p-values are two-sided Student t
 * on `n - p` degrees of freedom (float64, {@link getStudentTTwoSidedPValue}) where `p` counts the
 * intercept. `sigmaSquared` is `RSS / (n - p)` (`'degrees-of-freedom'`, the convention of the
 * standard errors) and `sigmaSquaredMaximumLikelihood` is `RSS / n`. The log-likelihood is the
 * Gaussian one with `p` counted parameters, so AIC and the Schwarz criterion equal spreg
 * `diagnostics.akaike` and `diagnostics.schwarz`. Tests are `jarqueBera` (2 df),
 * `breuschPagan` and `koenkerBassett` (`predictorCount` df each). Conditioning is null when the
 * fit failed. A failed fit (`status` not 0) carries NaN estimates. Positive applied lambda adds
 * `regularization: {kind: 'ridge', lambda}` and `inferenceMethod: 'ridge-approximate'`: the
 * penalized inverse-matrix uncertainty ignores shrinkage bias and is not sandwich covariance.
 * Unpenalized fits use `ordinary-student-t`. `summary` retains a copy of the original packed report.
 */
export function getGPUOrdinaryLeastSquaresModelResult(
  readback: GPUOrdinaryLeastSquaresReadbackInput,
  options: GPUOrdinaryLeastSquaresModelResultOptions = {}
): GPUOrdinaryLeastSquaresModelResult {
  const statusCode = validateRegressionStatusReadback(
    'getGPUOrdinaryLeastSquaresModelResult',
    readback.status
  );
  const {summary, coefficients: coefficientValues, standardErrors, tStatistics} = readback;
  if (summary.length < GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH) {
    throw new Error(
      `Ordinary least squares summary must hold ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} values`
    );
  }
  const id = 'getGPUOrdinaryLeastSquaresModelResult';
  const predictorCount = options.predictorCount ?? coefficientValues.length - 1;
  validateCapacity(id, 'predictorCount', predictorCount);
  const coefficientCount = predictorCount + 1;
  for (const [name, values] of [
    ['coefficients', coefficientValues],
    ['standardErrors', standardErrors],
    ['tStatistics', tStatistics]
  ] as const) {
    validateRequirement(
      id,
      values.length >= coefficientCount,
      `${name} length must hold predictorCount + 1 active entries`
    );
  }
  const rowCount = summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT];
  const degreesOfFreedom = rowCount - coefficientCount;
  const names = getGPURegressionPredictorNames(predictorCount, options.predictorNames);
  const coefficients: GPURegressionCoefficient[] = names.map((name, index) => {
    const statistic = tStatistics[index];
    return {
      name,
      role: index === 0 ? 'intercept' : 'predictor',
      estimate: coefficientValues[index],
      standardError: standardErrors[index],
      statistic,
      statisticKind: 't',
      pValue: getStudentTTwoSidedPValue(statistic, degreesOfFreedom)
    };
  });
  const conditionNumber = summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER];
  const correlationConditionNumber =
    summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER];
  const hasConditioning =
    Number.isFinite(conditionNumber) && Number.isFinite(correlationConditionNumber);
  const chiSquareTest = (statisticSlot: number, pValueSlot: number, freedom: number) => ({
    statistic: summary[statisticSlot],
    degreesOfFreedom: freedom,
    pValue: summary[pValueSlot]
  });
  const ridgeLambda = summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA];
  return {
    ...createGPURegressionModelResult({
      ...(Number.isFinite(ridgeLambda) && ridgeLambda > 0
        ? {
            regularization: {kind: 'ridge' as const, lambda: ridgeLambda},
            inferenceMethod: 'ridge-approximate'
          }
        : ridgeLambda === 0
          ? {inferenceMethod: 'ordinary-student-t'}
          : {}),
      kind: 'ordinary-least-squares',
      coefficients,
      rowCount,
      residualSumOfSquares: summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES],
      standardErrorSigmaSquared: 'degrees-of-freedom',
      statusCode,
      rSquared: summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED],
      logLikelihood: Number.isNaN(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD])
        ? null
        : summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD],
      conditioning: hasConditioning
        ? {
            conditionNumber,
            correlationConditionNumber,
            maximumVarianceInflationFactor:
              summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR]
          }
        : null,
      tests: {
        jarqueBera: chiSquareTest(
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA,
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE,
          2
        ),
        breuschPagan: chiSquareTest(
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
          predictorCount
        ),
        koenkerBassett: chiSquareTest(
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE,
          predictorCount
        )
      }
    }),
    summary: summary.slice()
  };
}
