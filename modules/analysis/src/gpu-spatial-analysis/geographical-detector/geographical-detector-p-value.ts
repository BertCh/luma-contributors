// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_GEOGRAPHICAL_DETECTOR_LAYOUT} from './gpu-geographical-detector';

/** Series terms and continued fraction iterations are cut at this many steps. */
const MAXIMUM_ITERATIONS = 100000;
const EPSILON = 1e-15;

/** Natural logarithm of the gamma function for positive `x` (Lanczos approximation, g = 7). */
function getLogGamma(x: number): number {
  const coefficients = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - getLogGamma(1 - x);
  }
  const shifted = x - 1;
  let sum = coefficients[0];
  for (let index = 1; index < 9; index++) {
    sum += coefficients[index] / (shifted + index);
  }
  const t = shifted + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Continued fraction of the regularized incomplete beta function (modified Lentz). */
function getBetaContinuedFraction(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  d = 1 / (Math.abs(d) < tiny ? tiny : d);
  let result = d;
  for (let m = 1; m <= MAXIMUM_ITERATIONS; m++) {
    const even = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + even * d;
    d = 1 / (Math.abs(d) < tiny ? tiny : d);
    c = 1 + even / c;
    if (Math.abs(c) < tiny) c = tiny;
    result *= d * c;
    const odd = (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + odd * d;
    d = 1 / (Math.abs(d) < tiny ? tiny : d);
    c = 1 + odd / c;
    if (Math.abs(c) < tiny) c = tiny;
    const delta = d * c;
    result *= delta;
    if (Math.abs(delta - 1) < EPSILON) break;
  }
  return result;
}

/** Regularized incomplete beta function `I_x(a, b)` in binary64. */
function getRegularizedIncompleteBeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const logFront =
    getLogGamma(a + b) - getLogGamma(a) - getLogGamma(b) + a * Math.log(x) + b * Math.log1p(-x);
  const front = Math.exp(logFront);
  return x < (a + 1) / (a + b + 2)
    ? (front * getBetaContinuedFraction(a, b, x)) / a
    : 1 - (front * getBetaContinuedFraction(b, a, 1 - x)) / b;
}

/**
 * Upper tail `P(F' > value)` of a non-central F distribution with `numeratorDegreesOfFreedom`,
 * `denominatorDegreesOfFreedom` and non-centrality `nonCentrality`, in binary64.
 *
 * It is the Poisson(`nonCentrality / 2`) mixture of central incomplete beta functions,
 * `P(F' <= x) = sum_j Pois(j) I_y(d1 / 2 + j, d2 / 2)` with `y = d1 x / (d1 x + d2)`, summed
 * outward from the Poisson mode until the weights are below `1e-16`. Agrees with
 * `scipy.stats.ncf.sf` to about `1e-10` for the ranges of the geographical detector.
 */
export function getNonCentralFUpperTail(
  value: number,
  numeratorDegreesOfFreedom: number,
  denominatorDegreesOfFreedom: number,
  nonCentrality: number
): number {
  if (Number.isNaN(value) || Number.isNaN(nonCentrality)) return NaN;
  if (value === Infinity) return 0;
  if (value <= 0) return 1;
  const halfNumerator = numeratorDegreesOfFreedom / 2;
  const halfDenominator = denominatorDegreesOfFreedom / 2;
  const y =
    (numeratorDegreesOfFreedom * value) /
    (numeratorDegreesOfFreedom * value + denominatorDegreesOfFreedom);
  const rate = Math.max(nonCentrality, 0) / 2;
  if (rate === 0) return 1 - getRegularizedIncompleteBeta(halfNumerator, halfDenominator, y);
  const mode = Math.floor(rate);
  const getLogWeight = (index: number) => -rate + index * Math.log(rate) - getLogGamma(index + 1);
  let cumulative = 0;
  for (let index = mode; index < mode + MAXIMUM_ITERATIONS; index++) {
    const weight = Math.exp(getLogWeight(index));
    cumulative += weight * getRegularizedIncompleteBeta(halfNumerator + index, halfDenominator, y);
    if (weight < 1e-17 && index > rate) break;
  }
  for (let index = mode - 1; index >= 0; index--) {
    const weight = Math.exp(getLogWeight(index));
    cumulative += weight * getRegularizedIncompleteBeta(halfNumerator + index, halfDenominator, y);
    if (weight < 1e-17) break;
  }
  return Math.min(1, Math.max(0, 1 - cumulative));
}

/**
 * P-value of the geographical detector q statistic, `1 - F_ncf(F; L - 1, N - L, lambda)`, from the
 * read-back `output.summary` of `GPUGeographicalDetector` (`geodetector` 0.2.0
 * `q_significance_test`). NaN where the summary is undefined (`F` is NaN); 0 for `q = 1`.
 *
 * @param summary Read-back summary words, laid out by `GPU_GEOGRAPHICAL_DETECTOR_LAYOUT`.
 */
export function getGeographicalDetectorPValue(summary: ArrayLike<number>): number {
  const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
  return getNonCentralFUpperTail(
    summary[layout.fStatistic],
    summary[layout.numeratorDegreesOfFreedom],
    summary[layout.denominatorDegreesOfFreedom],
    summary[layout.nonCentrality]
  );
}
