// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH} from './geographically-weighted-regression-parameters';

/** Shared publication slots, pinned against the public model layout. @internal */
export const GWR_SUMMARY_RESIDUAL_SUM_OF_SQUARES = 0;
export const GWR_SUMMARY_TRACE_OF_HAT = 1;
export const GWR_SUMMARY_AICC = 2;
export const GWR_SUMMARY_R_SQUARED = 3;
export const GWR_SUMMARY_OBSERVATION_COUNT = 4;
export const GWR_SUMMARY_HAS_VALID_CANDIDATE = 5;
export const GWR_SUMMARY_SINGULAR_ROW_COUNT = 6;
export const GWR_SUMMARY_SIGMA_SQUARED = 7;
export const GWR_SUMMARY_ADJUSTED_ALPHA = 8;
export const GWR_SUMMARY_CRITICAL_T = 9;
export const GWR_SUMMARY_LOG_LIKELIHOOD = 10;
export const GWR_SUMMARY_INCLUDED_ROW_COUNT = 11;
/** GWR selection retains its variance floor and historical arithmetic order. @internal */
export const GWR_SELECTED_LOG_LIKELIHOOD_EXPRESSION =
  '-0.5 * count * (LN_TWO_PI + log(max(residualSum / count, MINIMUM_VARIANCE)) + 1.0)';

/**
 * Shared dense/indexed GTWR publication in GWR summary slots. Slot 5 stays unpublished (NaN).
 * Requires count, included, singular, residualSquares, trace, totalSquares, summary and getNaN.
 * Unfloored likelihood preserves exact fits; AICc separately requires n-trace-2 > 0.
 * @internal
 */
export const GTWR_SUMMARY_PUBLICATION_WGSL = /* wgsl */ `
  let correctedDenominator = count - trace - 2.0;
  var logLikelihood = getNaN();
  var corrected = getNaN();
  if (count > 0.0) {
    if (residualSquares == 0.0) {
      logLikelihood = bitcast<f32>(0x7f800000u | (index & 0u));
      if (correctedDenominator > 0.0) { corrected = bitcast<f32>(0xff800000u | (index & 0u)); }
    } else if (residualSquares > 0.0) {
      let logVariance = log(residualSquares) - log(count);
      logLikelihood = -0.5 * count * (logVariance + 1.8378770664093453 + 1.0);
      if (correctedDenominator > 0.0) {
        corrected = ${getGWRAICcExpression('count', 'trace', 'correctedDenominator', 'logVariance', '1.8378770664093453')};
      }
    }
  }
  for (var slot = 0u; slot < ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH}u; slot++) {
    summary[summaryOffset + slot] = getNaN();
  }
  summary[summaryOffset] = residualSquares;
  summary[summaryOffset + ${GWR_SUMMARY_TRACE_OF_HAT}u] = trace;
  summary[summaryOffset + ${GWR_SUMMARY_AICC}u] = select(getNaN(), corrected, count > 0.0 && correctedDenominator > 0.0);
  summary[summaryOffset + ${GWR_SUMMARY_R_SQUARED}u] = select(getNaN(), 1.0 - residualSquares / totalSquares, totalSquares > 0.0);
  summary[summaryOffset + ${GWR_SUMMARY_OBSERVATION_COUNT}u] = count;
  summary[summaryOffset + ${GWR_SUMMARY_SINGULAR_ROW_COUNT}u] = singular;
  summary[summaryOffset + ${GWR_SUMMARY_SIGMA_SQUARED}u] = select(getNaN(), residualSquares / (count - trace), count > trace);
  summary[summaryOffset + ${GWR_SUMMARY_LOG_LIKELIHOOD}u] = select(getNaN(), logLikelihood, count > 0.0);
  summary[summaryOffset + ${GWR_SUMMARY_INCLUDED_ROW_COUNT}u] = included;`;

/**
 * mgwr AICc, algebraically `-2 llf + 2 n (trace + 1) / (n - trace - 2)`.
 * Callers guard the denominator and count. Keeping log variance explicit preserves GWR's
 * selection floor and GTWR's unfloored exact-fit likelihood/AICc (+Infinity/-Infinity).
 * @internal
 */
export function getGWRAICcExpression(
  count: string,
  trace: string,
  denominator: string,
  logVariance: string,
  logTwoPi: string
): string {
  return `${count} * ${logVariance} + ${count} * ${logTwoPi} + ${count} * (${count} + ${trace}) / ${denominator}`;
}
