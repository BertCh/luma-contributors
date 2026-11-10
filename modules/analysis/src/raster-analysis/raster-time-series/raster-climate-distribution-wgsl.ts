// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../analysis-core/kernels/incomplete-gamma-wgsl';

/** Term cap of the incomplete gamma series and continued fraction (float32 converges well before it). */
const REGULARIZED_GAMMA_ITERATIONS = 256;

/**
 * WGSL for `getRegularizedGammaTails(a, x)`, `vec2(P(a, x), Q(a, x))`. Each returned tail is
 * computed directly from the series (`P`, for `x < a + 1`) or the continued fraction (`Q`, for
 * `x >= a + 1`) rather than as one minus the other, so both tails keep relative accuracy down to
 * the float32 range. Declares nothing that collides with `GPU_UPPER_INCOMPLETE_GAMMA_WGSL`, whose
 * `getLogGammaValue` it reuses.
 *
 * @internal
 */
export const RASTER_REGULARIZED_GAMMA_TAILS_WGSL = `${GET_NAN_WGSL}
${IS_NAN_BITS_WGSL}
${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}
fn getRegularizedGammaTails(a: f32, x: f32) -> vec2<f32> {
  if (!(x > 0.0)) { return vec2<f32>(0.0, 1.0); }
  let logPrefactor = -x + a * log(x) - getLogGammaValue(a);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= ${REGULARIZED_GAMMA_ITERATIONS}u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (abs(term) < abs(sum) * 1e-7) { break; }
    }
    let lower = clamp(sum * exp(logPrefactor), 0.0, 1.0);
    return vec2<f32>(lower, 1.0 - lower);
  }
  let tiny = 1e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= ${REGULARIZED_GAMMA_ITERATIONS}u; iteration++) {
    let stepIndex = f32(iteration);
    let an = -stepIndex * (stepIndex - a);
    b += 2.0;
    d = an * d + b;
    if (abs(d) < tiny) { d = tiny; }
    c = b + an / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < 2e-7) { break; }
  }
  let upper = clamp(exp(logPrefactor) * h, 0.0, 1.0);
  return vec2<f32>(1.0 - upper, upper);
}
`;
