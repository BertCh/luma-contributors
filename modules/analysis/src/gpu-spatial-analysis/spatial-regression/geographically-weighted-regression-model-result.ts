// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPURegressionModelResult,
  type GPURegressionModelResult
} from '../../analysis-core/contracts/model-result';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
} from './geographically-weighted-regression-parameters';
import {getPackedFloat32Values} from '../../analysis-core/kernels/packed-readback';

/** Read-back arrays of a `GPUGeographicallyWeightedRegression` fit. */
export type GPUGeographicallyWeightedRegressionReadback = {
  /** `output.summary`: numeric words or logical float32 bytes/buffer, with at least 12 words.
   * Numeric inputs retain their precision; only the required prefix is read. Byte lengths must
   * be whole float32 words, including for unaligned views and foreign JavaScript realms. */
  summary: ArrayLike<number> | ArrayBuffer;
  /** Optional `output.selectedBandwidth`, `[candidate index, ladder value]`: at least two numeric
   * words or logical float32 bytes/buffer, with the same byte rules as summary. */
  selectedBandwidth?: ArrayLike<number> | ArrayBuffer;
};

/**
 * {@link GPURegressionModelResult} of a GWR fit plus the quantities specific to local models.
 *
 * `coefficients` is empty (the local coefficient surface is a per-row output) and `parameterCount`
 * is the effective number of parameters `tr(S)`.
 */
export type GPUGeographicallyWeightedRegressionModelResult = GPURegressionModelResult & {
  /** Effective number of parameters `tr(S)` (mgwr `ENP`, `tr_S`); same as `parameterCount`. */
  effectiveParameterCount: number;
  /** Included rows whose local system was singular at the selected bandwidth (NaN rows). */
  singularRowCount: number;
  /** Included rows, singular or not; `rowCount` excludes the singular ones. */
  includedRowCount: number;
  /** Da Silva-Fotheringham adjusted alpha `alpha * k / tr(S)` (mgwr `adj_alpha`). */
  adjustedAlpha: number;
  /** Critical `|t|` of the adjusted alpha with `rowCount - 1` degrees of freedom (mgwr `critical_tval`). */
  criticalT: number;
  /** Selected `[candidate index, ladder value]`; null when `selectedBandwidth` was not read back. */
  selectedBandwidth: {index: number; value: number} | null;
};

/**
 * Builds a {@link GPUGeographicallyWeightedRegressionModelResult} from the read-back summary of a
 * `GPUGeographicallyWeightedRegression`, with the conventions of mgwr 2.2.1 `GWRResults`:
 *
 * - `rowCount` is `n`, the non-singular included rows; `parameterCount` is `ENP = tr(S)`;
 * - `sigmaSquared` is `RSS / (n - tr(S))` (mgwr `sigma2` with `sigma2_v1`, which the standard
 *   errors use: `standardErrorSigmaSquared` is `'degrees-of-freedom'`) and
 *   `sigmaSquaredMaximumLikelihood` is `RSS / n`;
 * - `logLikelihood` is mgwr `llf` and `akaikeInformationCriterion` mgwr `aic`
 *   (`-2 llf + 2 (tr(S) + 1)`): the count is the effective parameters plus the error variance.
 *   `GPUOrdinaryLeastSquares` and the spatial maximum-likelihood estimators do not count the
 *   variance (and the error model does not count `lambda`), so add 2 per uncounted parameter
 *   before comparing an AIC from this result with theirs;
 * - `correctedAkaikeInformationCriterion` is the selected AICc (mgwr `aicc`);
 * - `rSquared` is the global `1 - RSS / TSS` (mgwr `R2`).
 *
 * `status.code` is 0 when a bandwidth candidate was valid and 1 otherwise (all numbers are then
 * NaN). Singular rows do not change the status; read `singularRowCount`.
 *
 * @param readback Read-back `summary` (and optionally `selectedBandwidth`).
 */
export function getGPUGeographicallyWeightedRegressionModelResult(
  readback: GPUGeographicallyWeightedRegressionReadback
): GPUGeographicallyWeightedRegressionModelResult {
  const summary = readSummary(readback.summary, 'GWR');
  const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
  const hasValid = summary[SUMMARY.HAS_VALID_CANDIDATE] > 0.5;
  const traceOfHat = summary[SUMMARY.TRACE_OF_HAT];
  const result = createGPURegressionModelResult({
    kind: 'geographically-weighted-regression',
    coefficients: [],
    rowCount: summary[SUMMARY.OBSERVATION_COUNT],
    residualSumOfSquares: summary[SUMMARY.RESIDUAL_SUM_OF_SQUARES],
    standardErrorSigmaSquared: 'degrees-of-freedom',
    statusCode: hasValid ? 0 : 1,
    rSquared: summary[SUMMARY.R_SQUARED],
    logLikelihood: hasValid ? summary[SUMMARY.LOG_LIKELIHOOD] : null,
    parameterCount: traceOfHat,
    // mgwr counts the error variance: aic = -2 llf + 2 (tr(S) + 1).
    informationCriterionParameterCount: traceOfHat + 1,
    correctedAkaikeInformationCriterion: hasValid ? summary[SUMMARY.AICC] : null
  });
  const selected =
    readback.selectedBandwidth === undefined
      ? undefined
      : readLocalValues(readback.selectedBandwidth, 'GWR selectedBandwidth', 2);
  return {
    ...result,
    effectiveParameterCount: traceOfHat,
    singularRowCount: summary[SUMMARY.SINGULAR_ROW_COUNT],
    includedRowCount: summary[SUMMARY.INCLUDED_ROW_COUNT],
    adjustedAlpha: summary[SUMMARY.ADJUSTED_ALPHA],
    criticalT: summary[SUMMARY.CRITICAL_T],
    selectedBandwidth: selected ? {index: selected[0], value: selected[1]} : null
  };
}

/** Dense and indexed GTWR share the published GWR statistics slots, but do not publish slot 5. */
export type GPUGeographicallyTemporallyWeightedRegressionReadback = Pick<
  GPUGeographicallyWeightedRegressionReadback,
  'summary'
>;

/** GTWR summary status describes usable rows, independently of bandwidth selection or AICc. */
export type GPUGeographicallyTemporallyWeightedRegressionModelResult = Omit<
  GPUGeographicallyWeightedRegressionModelResult,
  'kind'
> & {kind: 'geographically-temporally-weighted-regression'};

/**
 * Reads either dense or indexed GTWR summaries, including historical summaries with NaN in slot 5.
 * `status.code` is 0 when at least one row is usable, otherwise 1. Read `singularRowCount` for
 * included rows with unsuccessful local fits. A valid fit may have undefined AICc (`null`);
 * its published likelihood is retained. Only NaN AICc maps to null; published infinite values
 * (including exact-fit -Infinity) are preserved. The Gaussian variance and AIC conventions match GWR.
 * GTWR does not publish adjusted alpha, critical t or a selected bandwidth in this summary.
 */
export function getGPUGeographicallyTemporallyWeightedRegressionModelResult(
  readback: GPUGeographicallyTemporallyWeightedRegressionReadback
): GPUGeographicallyTemporallyWeightedRegressionModelResult {
  const summary = readSummary(readback.summary, 'GTWR');
  const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
  const hasUsableRows = summary[SUMMARY.OBSERVATION_COUNT] > 0;
  // Adapt only the host copy: the GWR reader's candidate-selection contract remains unchanged.
  const normalizedSummary = Float64Array.from(
    {length: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH},
    (_, index) => summary[index]
  );
  normalizedSummary[SUMMARY.HAS_VALID_CANDIDATE] = hasUsableRows ? 1 : 0;
  const result = getGPUGeographicallyWeightedRegressionModelResult({summary: normalizedSummary});
  return {
    ...result,
    kind: 'geographically-temporally-weighted-regression',
    correctedAkaikeInformationCriterion:
      hasUsableRows && !Number.isNaN(summary[SUMMARY.AICC]) ? summary[SUMMARY.AICC] : null
  };
}

/** Preserve numeric-element callers; buffer reads contain float32 words, not numeric bytes. */
function readSummary(input: ArrayLike<number> | ArrayBuffer, producer: string): ArrayLike<number> {
  return readLocalValues(
    input,
    `${producer} summary`,
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
  );
}

function readLocalValues(
  input: ArrayLike<number> | ArrayBuffer,
  label: string,
  minimumLength: number
): ArrayLike<number> {
  let summary: ArrayLike<number>;
  if (isLocalReadbackBytes(input)) {
    try {
      summary = getPackedFloat32Values(input, minimumLength);
    } catch (error) {
      throw new Error(`${label}: ${String(error)}`);
    }
  } else {
    summary = input;
  }
  if (!Number.isInteger(summary.length) || summary.length < minimumLength) {
    throw new Error(`${label} must hold ${minimumLength} values`);
  }
  for (let index = 0; index < minimumLength; index++) {
    if (typeof summary[index] !== 'number') {
      throw new Error(`${label} is missing numeric value ${index}`);
    }
  }
  return summary;
}

/** Realm-independent runtime detection also narrows the numeric-word branch for TypeScript. */
function isLocalReadbackBytes(
  input: ArrayLike<number> | ArrayBuffer
): input is Uint8Array | ArrayBuffer {
  const typedArrayTag = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(Uint8Array.prototype),
    Symbol.toStringTag
  )!.get!;
  return (
    (ArrayBuffer.isView(input) && typedArrayTag.call(input) === 'Uint8Array') ||
    Object.prototype.toString.call(input) === '[object ArrayBuffer]'
  );
}
