// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Numeric storage-format narrowing diagnostics and cause-specific rate invalid-value diagnostics. */

import type {NumericNarrowingReport} from '@luma-contributors/analysis/dataframe-analysis';
export {
  diagnoseFloat32Narrowing,
  mergeNumericNarrowingReports
} from '@luma-contributors/analysis/dataframe-analysis';
export type {
  NumericNarrowingReport,
  WideNumericFormat
} from '@luma-contributors/analysis/dataframe-analysis';

/** Explicit policy: Float64/Int64 columns narrow only when exact, or when the user accepts rounding. */
export function decideNarrowing(
  report: NumericNarrowingReport,
  acceptLossy: boolean
): {allowed: boolean; message?: string} {
  if (report.verdict === 'exact') return {allowed: true};
  if (report.verdict === 'refused')
    return {
      allowed: false,
      message: `${report.overflowCount} ${report.format} values overflow Float32`
    };
  return acceptLossy
    ? {
        allowed: true,
        message: `${report.inexactCount} values round (max error ${report.maximumAbsoluteError})`
      }
    : {
        allowed: false,
        message: `${report.inexactCount} ${report.format} values are not exact in Float32; accept rounding to continue`
      };
}

export {
  classifyRateInvalidCauses,
  KEPLER_RATE_CAUSES as RATE_INVALID_CAUSES
} from '@luma-contributors/kepler';
export type {RateInvalidCause, RateInvalidDiagnostics} from '@luma-contributors/kepler';
