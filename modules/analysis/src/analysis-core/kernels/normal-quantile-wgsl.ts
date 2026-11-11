// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL for the standard normal quantile `getNormalQuantileValue(p)`, Wichura's Algorithm AS 241
 * (PPND16), evaluated in float32. Its rational pieces carry the published 16-digit coefficients
 * (pinned digit for digit by `test/utils/normal-quantile-wgsl.node.spec.ts`), so the float32
 * result is limited by the float32 evaluation: within about 3e-7 relative of the float64 quantile
 * of the float32 `p` it is given, in all three branches, down to `p = 1.5e-38` (2.1e-7 at most on
 * the rows of `test/utils/normal-quantile-wgsl.spec.ts`).
 * Callers needing a precise upper tail should pass the small tail probability directly and negate
 * the result, because `1 - p` cannot be represented near 1 in float32.
 *
 * Returns a large finite value, not an infinity: `-1e15` for `p <= 0` and `1e15` for `p >= 1`.
 */
export const GET_NORMAL_QUANTILE_WGSL = /* wgsl */ `
fn getNormalQuantileValue(probability: f32) -> f32 {
  if (probability <= 0.0) { return -1.0 / 0.000000000000001; }
  if (probability >= 1.0) { return 1.0 / 0.000000000000001; }
  let centered = probability - 0.5;
  if (abs(centered) <= 0.425) {
    let radius = 0.180625 - centered * centered;
    return centered * (((((((radius * 2509.0809287301226727 + 33430.575583588128105) * radius + 67265.770927008700853) * radius + 45921.953931549871457) * radius + 13731.693765509461125) * radius + 1971.5909503065514427) * radius + 133.14166789178437745) * radius + 3.387132872796366608)
      / (((((((radius * 5226.495278852854561 + 28729.085735721942674) * radius + 39307.89580009271061) * radius + 21213.794301586595867) * radius + 5394.1960214247511077) * radius + 687.1870074920579083) * radius + 42.313330701600911252) * radius + 1.0);
  }
  let tail = select(1.0 - probability, probability, centered < 0.0);
  var radius = sqrt(-log(tail));
  var value: f32;
  if (radius <= 5.0) {
    radius -= 1.6;
    value = (((((((radius * 7.7454501427834140764e-4 + 0.0227238449892691845833) * radius + 0.24178072517745061177) * radius + 1.27045825245236838258) * radius + 3.64784832476320460504) * radius + 5.7694972214606914055) * radius + 4.6303378461565452959) * radius + 1.42343711074968357734)
      / (((((((radius * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * radius + 0.0151986665636164571966) * radius + 0.14810397642748007459) * radius + 0.68976733498510000455) * radius + 1.6763848301838038494) * radius + 2.05319162663775882187) * radius + 1.0);
  } else {
    radius -= 5.0;
    value = (((((((radius * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * radius + 0.0012426609473880784386) * radius + 0.026532189526576123093) * radius + 0.29656057182850489123) * radius + 1.7848265399172913358) * radius + 5.4637849111641143699) * radius + 6.6579046435011037772)
      / (((((((radius * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * radius + 1.8463183175100546818e-5) * radius + 7.868691311456132591e-4) * radius + 0.0148753612908506148525) * radius + 0.13692988092273580531) * radius + 0.59983220655588793769) * radius + 1.0);
  }
  return select(value, -value, centered < 0.0);
}
`;
