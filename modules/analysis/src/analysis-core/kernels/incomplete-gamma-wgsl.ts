// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Iteration cap shared by the series and the continued fraction of {@link GPU_UPPER_INCOMPLETE_GAMMA_WGSL}. */
export const GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS = 500;

/**
 * Shape `a = dof / 2` above which the Wilson-Hilferty normal approximation replaces the series and
 * continued fraction. There the term count approaches the iteration cap and the f32 cancellation
 * in `-x + a ln x - lnGamma(a)` loses most digits.
 */
export const GPU_UPPER_INCOMPLETE_GAMMA_NORMAL_APPROXIMATION_SHAPE = 1000;

/**
 * WGSL for the chi-square survival function `P(X > value)` for any positive (integer or real)
 * degrees of freedom, as the regularized upper incomplete gamma `Q(dof / 2, value / 2)`.
 *
 * Declares `getLogGammaValue(z: f32) -> f32` (Stirling series after an upward shift, `z > 0`) and
 * `getChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32`. The survival function
 * uses the power series for `x < a + 1` and a modified Lentz continued fraction otherwise, each
 * capped at {@link GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}. It returns 1 for `value <= 0`
 * and a NaN for a NaN value or `degreesOfFreedom <= 0`.
 *
 * Above `dof = 2000` ({@link GPU_UPPER_INCOMPLETE_GAMMA_NORMAL_APPROXIMATION_SHAPE}) the
 * Wilson-Hilferty cube-root normal approximation is used instead; its absolute error is below
 * about 2e-4 there, which is smaller than the f32 cancellation error of the exact series.
 *
 * Accuracy (f32, checked against `scipy.stats.chi2.sf` for 1 to 500 degrees of freedom): relative
 * error below 6e-5 for p-values above 1e-6 and absolute error below 1.5e-5 for p-values above
 * 1e-3. The error is dominated by f32 cancellation in `-x + a ln x - lnGamma(a)`, so it grows
 * roughly with `6e-8 * max(x, a ln x)`; p-values below the f32 normal range flush to 0.
 *
 * The declaration requires `getNaN()` and `isNanBits(x)` (`GET_NAN_WGSL`, `IS_NAN_BITS_WGSL` of
 * `utils/wgsl-common.ts`) to be declared once in the same module. It is kept here because it
 * belongs in `utils/wgsl-common.ts` next to those helpers; move it there when a second consumer
 * appears.
 *
 * @internal
 */
export const GPU_UPPER_INCOMPLETE_GAMMA_WGSL = /* wgsl */ `
fn getLogGammaValue(z: f32) -> f32 {
  var shifted = z;
  var product = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (shifted >= 8.0) {
      break;
    }
    product = product * shifted;
    shifted = shifted + 1.0;
  }
  let inverse = 1.0 / shifted;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (shifted - 0.5) * log(shifted) - shifted + 0.9189385332 + series - log(product);
}

// Standard normal survival function via the Numerical Recipes erfc fit (relative error under 1.2e-7).
fn getNormalSurvivalValue(z: f32) -> f32 {
  let magnitude = abs(z) * 0.7071067811865476;
  let t = 1.0 / (1.0 + 0.5 * magnitude);
  let r = t * exp(-magnitude * magnitude - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return 0.5 * select(2.0 - r, r, z >= 0.0);
}

fn getChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32 {
  if (isNanBits(value) || !(degreesOfFreedom > 0.0)) {
    return getNaN();
  }
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = degreesOfFreedom * 0.5;
  let x = value * 0.5;
  if (a > ${GPU_UPPER_INCOMPLETE_GAMMA_NORMAL_APPROXIMATION_SHAPE}.0) {
    // Wilson-Hilferty: (X / dof)^(1/3) is approximately normal with mean 1 - 2 / (9 dof).
    let variance = 2.0 / (9.0 * degreesOfFreedom);
    let z = (pow(value / degreesOfFreedom, 1.0 / 3.0) - (1.0 - variance)) / sqrt(variance);
    return getNormalSurvivalValue(z);
  }
  let logPrefactor = -x + a * log(x) - getLogGammaValue(a);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= ${GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}u; iteration++) {
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
  for (var iteration = 1u; iteration <= ${GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}u; iteration++) {
    let stepIndex = f32(iteration);
    let an = -stepIndex * (stepIndex - a);
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
    if (abs(delta - 1.0) < 2.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`;

/** Centered lower-tail series cap, preserving the point-pattern indices route. */
export const GPU_CENTERED_GAMMA_MAXIMUM_ITERATIONS = 2048;

/**
 * Optional centered exact gamma route; include GPU_UPPER_INCOMPLETE_GAMMA_WGSL first.
 * Adds getCenteredGammaTails(a,x) -> (P,Q) and getCenteredChiSquareSurvivalValue(value,df).
 * Preserves the indices route: centered log1p series through order 20, lower-tail series at
 * most 2048 iterations and continued fraction at most 499 iterations. No Wilson-Hilferty
 * substitution. The directly evaluated tail retains relative precision; its complement can
 * cancel near zero. Exhausting either iteration budget returns NaN (both paired tails), never
 * a partial probability. Positive finite shape/df is required; NaN/nonfinite shape returns NaN.
 * Nonpositive arguments return (P,Q)=(0,1); positive infinity returns (1,0), using bit checks.
 * Results below the normal f32 range may flush to zero. This does not change
 * getChiSquareSurvivalValue or its existing approximation selection. No bindings are added.
 * @internal
 */
export const GPU_CENTERED_INCOMPLETE_GAMMA_WGSL = /* wgsl */ `
// Stirling's log-gamma expansion (DLMF 5.11.1), centered at x = a to avoid subtracting
// terms of order a * log(a). The log1p(delta) - delta series also avoids cancellation.
fn getCenteredGammaPrefix(shape: f32, value: f32) -> f32 {
  if (shape <= 50.0) {
    return exp(shape * log(value) - value - getLogGammaValue(shape));
  }
  let delta = (value - shape) / shape;
  var centeredLog: f32;
  if (abs(delta) <= 0.25) {
    var power = -delta * delta;
    centeredLog = 0.5 * power;
    for (var order = 3; order <= 20; order++) {
      power *= -delta;
      centeredLog += power / f32(order);
    }
  } else {
    centeredLog = log(value / shape) - delta;
  }
  let inverse = 1.0 / shape;
  let correction = inverse * (0.08333333333333 - inverse * inverse * 0.00277777777778);
  return exp(shape * centeredLog + 0.5 * log(shape) - 0.91893853320467 - correction);
}

fn getCenteredGammaTails(a: f32, x: f32) -> vec2<f32> {
  let shapeBits = bitcast<u32>(a);
  let valueBits = bitcast<u32>(x);
  if ((shapeBits & 0x7fffffffu) >= 0x7f800000u || !(a > 0.0) || isNanBits(x)) {
    return vec2<f32>(getNaN());
  }
  if (valueBits == 0x7f800000u) { return vec2<f32>(1.0, 0.0); }
  if (valueBits == 0xff800000u || x <= 0.0) {
    return vec2<f32>(0.0, 1.0);
  }
  let prefix = getCenteredGammaPrefix(a, x);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    var denominator = a;
    var seriesConverged = false;
    for (var iteration = 0; iteration < ${GPU_CENTERED_GAMMA_MAXIMUM_ITERATIONS}; iteration++) {
      denominator += 1.0;
      term *= x / denominator;
      sum += term;
      if (abs(term) < abs(sum) * 1.0e-7) { seriesConverged = true; break; }
    }
    if (!seriesConverged) { return vec2<f32>(getNaN()); }
    let lower = clamp(sum * prefix, 0.0, 1.0);
    return vec2<f32>(lower, 1.0 - lower);
  }
  var b = x + 1.0 - a;
  var c = 1.0e30;
  var d = 1.0 / b;
  var h = d;
  var fractionConverged = false;
  for (var iteration = 1; iteration < ${GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}; iteration++) {
    let an = -f32(iteration) * (f32(iteration) - a);
    b += 2.0;
    d = an * d + b;
    if (abs(d) < 1.0e-30) { d = 1.0e-30; }
    c = b + an / c;
    if (abs(c) < 1.0e-30) { c = 1.0e-30; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < 1.0e-7) { fractionConverged = true; break; }
  }
  if (!fractionConverged) { return vec2<f32>(getNaN()); }
  let upper = clamp(prefix * h, 0.0, 1.0);
  return vec2<f32>(1.0 - upper, upper);
}

fn getCenteredChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32 {
  let degreesBits = bitcast<u32>(degreesOfFreedom);
  if (isNanBits(value) || (degreesBits & 0x7fffffffu) >= 0x7f800000u || !(degreesOfFreedom > 0.0)) { return getNaN(); }
  let valueBits = bitcast<u32>(value);
  if (valueBits == 0x7f800000u) { return 0.0; }
  if (valueBits == 0xff800000u || !(value > 0.0)) { return 1.0; }
  let tails = getCenteredGammaTails(degreesOfFreedom * 0.5, value * 0.5);
  return tails.y;
}
`;
