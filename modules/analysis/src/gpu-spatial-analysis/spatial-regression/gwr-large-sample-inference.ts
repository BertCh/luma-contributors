// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ERFC_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getStudentTWGSL} from './gwr-student-t';

/** Above this df the third-order Student quantile expansion is accurate to float32 precision. */
const GWR_QUANTILE_EXPANSION_MINIMUM_DEGREES_OF_FREEDOM = 4096;
/** Keeps the normal quantile below 6.5; smaller tails retain the historical beta evaluator. */
const GWR_QUANTILE_EXPANSION_MINIMUM_TAIL = 1e-10;
/** Bisection resolves the normal quantile to float32 precision on the fixed [0, 8] bracket. */
const GWR_NORMAL_QUANTILE_BISECTION_COUNT = 32;

/**
 * Avoid forming x = df / (df + t²) in float32 at large df: its rounding is amplified by
 * the incomplete beta fraction. Invert the shared erfc tail without subtracting from one,
 * then apply the Student quantile expansion through df^-3. Keep the historical df clamp
 * and the beta evaluator for small samples and extreme tails. @internal
 */
export function getGWRLargeSampleInferenceWGSL(): string {
  return `${getStudentTWGSL()}
${ERFC_VALUE_WGSL}
const GWR_QUANTILE_MINIMUM_DF: f32 = ${GWR_QUANTILE_EXPANSION_MINIMUM_DEGREES_OF_FREEDOM}.0;
const GWR_QUANTILE_MINIMUM_TAIL: f32 = ${GWR_QUANTILE_EXPANSION_MINIMUM_TAIL};
fn gwrLargeSampleCriticalValue(tail: f32, degreesOfFreedom: f32) -> f32 {
  if (!(degreesOfFreedom >= GWR_QUANTILE_MINIMUM_DF) ||
      !(tail >= GWR_QUANTILE_MINIMUM_TAIL) || !(tail < 1.0)) {
    return gwrStudentCriticalValue(tail, degreesOfFreedom);
  }
  var lower = 0.0;
  var upper = 8.0;
  for (var iteration = 0u; iteration < ${GWR_NORMAL_QUANTILE_BISECTION_COUNT}u; iteration++) {
    let middle = 0.5 * (lower + upper);
    if (erfcValue(middle * 0.7071067811865476) < tail) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  let normal = 0.5 * (lower + upper);
  let squared = normal * normal;
  let inverse = 1.0 / min(degreesOfFreedom, GWR_T_MAXIMUM_DF);
  let first = normal * (squared + 1.0) / 4.0;
  let second = normal * (5.0 * squared * squared + 16.0 * squared + 3.0) / 96.0;
  let third = normal * (3.0 * squared * squared * squared + 19.0 * squared * squared + 17.0 * squared - 15.0) / 384.0;
  return normal + inverse * (first + inverse * (second + inverse * third));
}
`;
}
