// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Poisson half-deviance with a cancellation-free series near an exact fit. @internal */
export const INTERACTION_DEVIANCE_WGSL = `
fn getPoissonHalfDeviance(flow: f32, prediction: f32) -> f32 {
  if (!(flow > 0.0)) { return prediction; }
  var relative = (prediction - flow) / flow;
  // Apple Metal divides as a * (1 / b): at flow >= 2^126 the reciprocal is subnormal and
  // flushes to zero. Divide by flow / 8 (an exact exponent decrement) and scale the numerator.
  let flowBits = bitcast<u32>(flow);
  if (flowBits >= 0x7e800000u) {
    let scaledFlow = bitcast<f32>(flowBits - (3u << 23u));
    relative = ((prediction - flow) * 0.125) / scaledFlow;
  }
  if (abs(relative) < 0.125) {
    // r-log(1+r) = r^2(1/2-r/3+r^2/4-...). At |r|<1/8,
    // the remainder from degree 13 is below 3e-11 relative to the leading term.
    var polynomial = 0.5;
    var power = 1.0;
    for (var degree = 3u; degree <= 12u; degree++) {
      power *= -relative;
      polynomial += power / f32(degree);
    }
    return flow * relative * relative * polynomial;
  }
  return flow * (log(flow) - log(prediction)) - flow + prediction;
}`;
