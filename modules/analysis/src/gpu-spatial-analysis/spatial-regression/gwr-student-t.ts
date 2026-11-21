// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getStudentTPositiveTail} from './student-t-core';
import {ERFC_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
/**
 * Student t two-sided tail probability and critical value, as WGSL for the GWR significance mask
 * and as a float64 CPU reference.
 *
 * The two-sided tail of a t variate with `df` degrees of freedom is the regularized incomplete
 * beta `I_x(df / 2, 1 / 2)` with `x = df / (df + t^2)`. It is evaluated with the Lentz continued
 * fraction (Numerical Recipes `betacf`), on the side of the symmetry relation where the fraction
 * converges quickly. The beta-function prefactor needs `Gamma(a + 1/2) / Gamma(a)`; it is formed
 * from a recurrence up to `a >= 10` and the asymptotic series there, so no `lgamma` difference of
 * large numbers appears (a plain `lgamma(a + 1/2) - lgamma(a)` would lose all float32 digits for
 * `df` in the thousands). `x^a` uses `log1p(t^2 / df)` for the same reason. The critical value
 * inverts the tail by doubling and bisection (the tail is monotone in `t`).
 *
 * Degrees of freedom above `GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM` are clamped: the quantile
 * differs from the normal one by about `(z^3 + z) / (4 df)` there (about 1.2e-5 at a 0.05
 * two-sided tail). This is a deliberate approximation, not float32 rounding; the error grows
 * toward the tails.
 * For df >= 4096 and t <= 16 the WGSL tail instead inverts the third-order Cornish-Fisher
 * t quantile and evaluates erfc, avoiding cancellation in a beta argument close to one.
 */

/** Largest degrees of freedom used; larger values are clamped (see above). */
export const GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM = 200000;

/** Continued-fraction iteration cap; the fraction needs about `sqrt(df)` iterations. */
const MAXIMUM_ITERATIONS = 3000;

/** Bisection steps; the bracket is at most a factor 2 wide before them. */
const BISECTION_COUNT = 40;

/**
 * Returns WGSL declaring `gwrStudentTwoSidedTail(t, df)` and `gwrStudentCriticalValue(tail, df)`.
 *
 * - `gwrStudentTwoSidedTail(t: f32, df: f32) -> f32` is `P(|T| >= t)`.
 * - `gwrStudentCriticalValue(tail: f32, df: f32) -> f32` is the `t >= 0` whose two-sided tail is
 *   `tail`: 0 for `tail >= 1`, NaN for a non-positive or non-finite `tail` or `df < 1`, and
 *   `scipy.stats.t.ppf(1 - tail / 2, df)` otherwise.
 *
 * NaN t inputs propagate NaN. The declaration uses `log`, `exp`, `sqrt` and arithmetic,
 * plus `getNaN()` (`GET_NAN_WGSL`),
 * which the including kernel must declare.
 */
export function getStudentTWGSL(): string {
  return /* wgsl */ `
${ERFC_VALUE_WGSL.replaceAll('erfcValue', 'gwrStudentErfcValue')}
const GWR_T_EPSILON: f32 = 3.0e-7;
const GWR_T_MINIMUM: f32 = 1.0e-30;
const GWR_T_SQRT_PI: f32 = 1.7724538509055159;
const GWR_T_MAXIMUM_DF: f32 = ${GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM}.0;

// log(1 + z) without the rounding of 1 + z (z small and positive).
fn gwrLog1p(z: f32) -> f32 {
  // The relative 1/z correction is negligible here; log(u) * z would overflow first.
  if (z > 1.0e30) {
    return log(z);
  }
  let u = 1.0 + z;
  if (u == 1.0) {
    return z;
  }
  return log(u) * z / (u - 1.0);
}

// Gamma(a + 1/2) / Gamma(a) for a > 0: recurrence up to a >= 10, then the asymptotic series.
fn gwrGammaHalfRatio(a: f32) -> f32 {
  var s = a;
  var factor = 1.0;
  for (var step = 0u; step < 12u; step++) {
    if (s >= 10.0) {
      break;
    }
    factor = factor * s / (s + 0.5);
    s = s + 1.0;
  }
  let inverse = 1.0 / s;
  let series = 1.0 - 0.125 * inverse +
    inverse * inverse * (1.0 / 128.0 + inverse * (5.0 / 1024.0 -
      inverse * (21.0 / 32768.0 + inverse * (399.0 / 262144.0 - inverse * (869.0 / 4194304.0)))));
  return factor * sqrt(s) * series;
}

// Lentz continued fraction of the incomplete beta function (Numerical Recipes betacf).
fn gwrBetaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < GWR_T_MINIMUM) {
    d = GWR_T_MINIMUM;
  }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= ${MAXIMUM_ITERATIONS}u; m++) {
    let fm = f32(m);
    let m2 = 2.0 * fm;
    var aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    h = h * d * c;
    aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < GWR_T_EPSILON) {
      break;
    }
  }
  return h;
}

// P(|T| >= t) for a Student t variate with df degrees of freedom.
fn gwrStudentTwoSidedTail(t: f32, degreesOfFreedom: f32) -> f32 {
  if ((bitcast<u32>(t) & 0x7fffffffu) > 0x7f800000u) { return getNaN(); }
  if (!(t > 0.0)) {
    return 1.0;
  }
  let df = min(degreesOfFreedom, GWR_T_MAXIMUM_DF);
  // Invert the large-df Cornish-Fisher t quantile. This avoids forming a beta argument
  // indistinguishable from one in f32, while erfc retains small tails without subtraction.
  if (df >= 4096.0 && t <= 16.0) {
    var normal = t;
    let inverse = 1.0 / df;
    for (var step = 0u; step < 4u; step++) {
      let squared = normal * normal;
      let first = normal * (squared + 1.0) / 4.0;
      let second = normal * (5.0 * squared * squared + 16.0 * squared + 3.0) / 96.0;
      let third = normal * (3.0 * squared * squared * squared + 19.0 * squared * squared + 17.0 * squared - 15.0) / 384.0;
      let value = normal + inverse * (first + inverse * (second + inverse * third));
      let derivative = 1.0 + inverse * ((3.0 * squared + 1.0) / 4.0 + inverse * ((25.0 * squared * squared + 48.0 * squared + 3.0) / 96.0 + inverse * (21.0 * squared * squared * squared + 95.0 * squared * squared + 51.0 * squared - 15.0) / 384.0));
      normal -= (value - t) / derivative;
    }
    return clamp(gwrStudentErfcValue(normal * 0.7071067811865476), 0.0, 1.0);
  }
  let a = 0.5 * df;
  let ratio = t * t / df;
  // Once t^2 overflows, x is tiny enough that betacf(a, 1/2, x) is 1 to f32 precision.
  // Evaluate x^a in log space instead of creating inf/inf in y = ratio / (1 + ratio).
  if (ratio > 3.0e38) {
    let logRatio = 2.0 * log(t) - log(df);
    return exp(-a * logRatio) * gwrGammaHalfRatio(a) / (GWR_T_SQRT_PI * a);
  }
  let x = 1.0 / (1.0 + ratio);
  let y = ratio / (1.0 + ratio);
  let front = exp(-a * gwrLog1p(ratio)) * sqrt(y) * gwrGammaHalfRatio(a) / GWR_T_SQRT_PI;
  if (x < (a + 1.0) / (a + 2.5)) {
    return front * gwrBetaContinuedFraction(a, 0.5, x) / a;
  }
  return 1.0 - 2.0 * front * gwrBetaContinuedFraction(0.5, a, y);
}

// The t >= 0 with P(|T| >= t) = tail.
fn gwrStudentCriticalValue(tail: f32, degreesOfFreedom: f32) -> f32 {
  if (!(tail > 0.0) || !(degreesOfFreedom >= 1.0) || tail > 3.0e38) {
    return getNaN();
  }
  if (tail >= 1.0) {
    return 0.0;
  }
  var low = 0.0;
  var high = 1.0;
  // A Cauchy quantile for an f32 tail can require more than 60 doublings.
  for (var step = 0u; step < 128u; step++) {
    if (gwrStudentTwoSidedTail(high, degreesOfFreedom) < tail) {
      break;
    }
    low = high;
    high = min(high * 2.0, 3.4028234663852886e38);
  }
  if (high == 3.4028234663852886e38 && gwrStudentTwoSidedTail(high, degreesOfFreedom) > tail) {
    return bitcast<f32>(0x7f800000u | (bitcast<u32>(tail) & 0u));
  }
  for (var step = 0u; step < ${BISECTION_COUNT}u; step++) {
    let middle = select(0.5 * (low + high), low + 0.5 * (high - low), high > 1.0e38);
    if (gwrStudentTwoSidedTail(middle, degreesOfFreedom) < tail) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return select(0.5 * (low + high), low + 0.5 * (high - low), high > 1.0e38);
}
`;
}

/**
 * Float64 CPU mirror of `gwrStudentTwoSidedTail`: `P(|T| >= t)` for `df` degrees of freedom
 * (clamped to `GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM`).
 */
export function getStudentTTwoSidedTail(t: number, degreesOfFreedom: number): number {
  if (Number.isNaN(t)) return NaN;
  if (!(t > 0)) {
    return 1;
  }
  const df = Math.min(degreesOfFreedom, GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM);
  if (t === Infinity && Number.isFinite(df) && df > 0) return 0;
  return getStudentTPositiveTail(t, df, MAXIMUM_ITERATIONS);
}

/**
 * Float64 CPU mirror of `gwrStudentCriticalValue`: the `t >= 0` with two-sided tail `tail`, that
 * is `scipy.stats.t.ppf(1 - tail / 2, df)`. Returns 0 for `tail >= 1` and NaN for a non-positive
 * or non-finite `tail` or `df < 1`.
 */
export function getStudentTCriticalValue(tail: number, degreesOfFreedom: number): number {
  if (!(tail > 0) || !(degreesOfFreedom >= 1) || !Number.isFinite(tail)) {
    return Number.NaN;
  }
  if (tail >= 1) {
    return 0;
  }
  let low = 0;
  let high = 1;
  for (let step = 0; step < 1024; step++) {
    if (getStudentTTwoSidedTail(high, degreesOfFreedom) < tail) {
      break;
    }
    low = high;
    high = Math.min(high * 2, Number.MAX_VALUE);
  }
  if (high === Number.MAX_VALUE && getStudentTTwoSidedTail(high, degreesOfFreedom) > tail) {
    return Infinity;
  }
  for (let step = 0; step < 100; step++) {
    const middle = high > 1e308 ? low + 0.5 * (high - low) : 0.5 * (low + high);
    if (getStudentTTwoSidedTail(middle, degreesOfFreedom) < tail) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return high > 1e308 ? low + 0.5 * (high - low) : 0.5 * (low + high);
}
