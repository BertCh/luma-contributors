// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_GEOGRAPHICAL_DETECTOR_LAYOUT} from './gpu-geographical-detector';

/** Series terms and continued fraction iterations are cut at this many steps. */
const MAXIMUM_ITERATIONS = 100000;
const EPSILON = 1e-15;
/** Keeps the Poisson mode at most 5000, within the bounded binary64 mixture evaluation. */
const MAXIMUM_NON_CENTRALITY = 10000;

/** Stirling's log-gamma correction for arguments >= 16; next term is below 1.1e-16. */
function getStirlingCorrection(value: number): number {
  const inverse = 1 / value;
  const inverseSquared = inverse * inverse;
  return (
    inverse *
    (1 / 12 -
      inverseSquared *
        (1 / 360 -
          inverseSquared * (1 / 1260 - inverseSquared * (1 / 1680 - inverseSquared / 1188))))
  );
}

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

/** Log beta without differencing large, nearly equal log-gamma values. */
function getLogBeta(firstShape: number, secondShape: number): number {
  const smaller = Math.min(firstShape, secondShape);
  const larger = Math.max(firstShape, secondShape);
  if (larger < 16)
    return getLogGamma(smaller) + getLogGamma(larger) - getLogGamma(smaller + larger);
  const logRatio = Math.log1p(smaller / larger);
  if (smaller < 16) {
    // Stirling difference logGamma(larger) - logGamma(larger + smaller), formed directly.
    return (
      getLogGamma(smaller) -
      smaller * Math.log(larger) +
      smaller -
      (larger + smaller - 0.5) * logRatio +
      getStirlingCorrection(larger) -
      getStirlingCorrection(larger + smaller)
    );
  }
  return (
    0.5 * Math.log(2 * Math.PI) -
    0.5 * (Math.log(larger) + logRatio) +
    (smaller - 0.5) * (Math.log(smaller) - Math.log(larger) - logRatio) -
    (larger - 0.5) * logRatio +
    getStirlingCorrection(smaller) +
    getStirlingCorrection(larger) -
    getStirlingCorrection(larger + smaller)
  );
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
    if (Math.abs(delta - 1) < EPSILON) return result;
  }
  throw new Error('getNonCentralFUpperTail beta continued fraction did not converge');
}

/** Log regularized incomplete beta; both arguments are formed directly by the caller. */
function getLogRegularizedIncompleteBeta(
  firstShape: number,
  secondShape: number,
  argument: number,
  complement: number
): number {
  if (argument <= 0) return -Infinity;
  if (complement <= 0) return 0;
  const logArgument = argument < 0.5 ? Math.log(argument) : Math.log1p(-complement);
  const logComplement = complement < 0.5 ? Math.log(complement) : Math.log1p(-argument);
  if (secondShape === 1) return firstShape * logArgument;
  if (firstShape === 1) return Math.log(-Math.expm1(secondShape * logComplement));
  const logFront =
    -getLogBeta(firstShape, secondShape) + firstShape * logArgument + secondShape * logComplement;
  if (argument < 1 / (1 + (secondShape + 1) / (firstShape + 1))) {
    return Math.min(
      0,
      logFront +
        Math.log(getBetaContinuedFraction(firstShape, secondShape, argument)) -
        Math.log(firstShape)
    );
  }
  const logOpposite =
    logFront +
    Math.log(getBetaContinuedFraction(secondShape, firstShape, complement)) -
    Math.log(secondShape);
  return Math.log(-Math.expm1(Math.min(0, logOpposite)));
}

/** Log of a sum, retaining terms whose products would underflow in binary64. */
function addLogProbabilities(first: number, second: number): number {
  if (first === -Infinity) return second;
  if (second === -Infinity) return first;
  const larger = Math.max(first, second);
  return larger + Math.log1p(Math.exp(Math.min(first, second) - larger));
}

/**
 * Upper tail `P(F' > value)` of a non-central F distribution with `numeratorDegreesOfFreedom`,
 * `denominatorDegreesOfFreedom` and non-centrality `nonCentrality`, in binary64.
 *
 * It is the Poisson(`nonCentrality / 2`) mixture of central incomplete beta functions,
 * `P(F' > x) = sum_j Pois(j) I_z(d2 / 2, d1 / 2 + j)` with `z = d2 / (d1 x + d2)`.
 * The complementary argument and survival terms are evaluated directly to retain small tails.
 * Both beta arguments are formed directly and weighted terms are summed in log space before
 * rounding the result. The mixture is summed outward from its mode with a relative remaining-weight
 * bound. For finite positive F, non-centrality above 10000 (or nonfinite) throws. For valid
 * degrees and finite non-centrality, exact support endpoints return 0 (F = Infinity) or 1
 * (F <= 0) before the mixture limit. Each direction and each beta evaluation has at most
 * 100000 iterations and throws if it cannot converge. Negative non-centrality is treated
 * as zero, as in the detector's roundoff clamp.
 */
export function getNonCentralFUpperTail(
  value: number,
  numeratorDegreesOfFreedom: number,
  denominatorDegreesOfFreedom: number,
  nonCentrality: number
): number {
  if (Number.isNaN(value) || Number.isNaN(nonCentrality)) return NaN;
  if (
    !Number.isFinite(numeratorDegreesOfFreedom) ||
    numeratorDegreesOfFreedom <= 0 ||
    !Number.isFinite(denominatorDegreesOfFreedom) ||
    denominatorDegreesOfFreedom <= 0
  )
    return NaN;
  if (Number.isFinite(nonCentrality)) {
    if (value === Infinity) return 0;
    if (value <= 0) return 1;
  }
  if (!Number.isFinite(nonCentrality) || nonCentrality > MAXIMUM_NON_CENTRALITY) {
    throw new Error('getNonCentralFUpperTail nonCentrality must be finite and at most 10000');
  }
  const halfNumerator = numeratorDegreesOfFreedom / 2;
  const halfDenominator = denominatorDegreesOfFreedom / 2;
  const degreeRatio = denominatorDegreesOfFreedom / numeratorDegreesOfFreedom;
  // Divide the smaller quantity by the larger, preserving subnormal complementary arguments
  // even when value / degreeRatio would overflow (for example F(4,2) at 1e308).
  const ratio = value > degreeRatio ? degreeRatio / value : value / degreeRatio;
  const smallerArgument = ratio / (1 + ratio);
  const largerArgument = 1 / (1 + ratio);
  const complement = value > degreeRatio ? smallerArgument : largerArgument;
  const argument = value > degreeRatio ? largerArgument : smallerArgument;
  const rate = Math.max(nonCentrality, 0) / 2;
  const getLogSurvival = (index: number) =>
    getLogRegularizedIncompleteBeta(halfDenominator, halfNumerator + index, complement, argument);
  if (rate === 0) return Math.exp(getLogSurvival(0));
  const mode = Math.floor(rate);
  const logRate = Math.log(rate);
  const logModeWeight = -rate + mode * logRate - getLogGamma(mode + 1);
  let logCumulative = logModeWeight + getLogSurvival(mode);
  let logWeight = logModeWeight;
  let upperConverged = false;
  for (let iteration = 1; iteration <= MAXIMUM_ITERATIONS; iteration++) {
    const index = mode + iteration;
    logWeight += logRate - Math.log(index);
    logCumulative = addLogProbabilities(logCumulative, logWeight + getLogSurvival(index));
    // Survival terms are at most one; the remaining Poisson weights are bounded by this
    // geometric series. Compare to the tail, rather than an absolute weight threshold.
    const nextRatio = rate / (index + 1);
    const logRemainingWeight = logWeight + Math.log(nextRatio) - Math.log1p(-nextRatio);
    if (logRemainingWeight <= Math.log(EPSILON) + logCumulative) {
      upperConverged = true;
      break;
    }
  }
  if (!upperConverged) throw new Error('getNonCentralFUpperTail upper mixture did not converge');
  logWeight = logModeWeight;
  for (let iteration = 1; iteration <= Math.min(mode, MAXIMUM_ITERATIONS); iteration++) {
    const index = mode - iteration;
    logWeight += Math.log(index + 1) - logRate;
    const logSurvival = getLogSurvival(index);
    logCumulative = addLogProbabilities(logCumulative, logWeight + logSurvival);
    // Both the Poisson weights and survival terms decrease in this direction.
    const nextRatio = index / rate;
    if (
      logWeight + logSurvival + Math.log(nextRatio) - Math.log1p(-nextRatio) <=
      Math.log(EPSILON) + logCumulative
    )
      break;
  }
  return Math.exp(Math.min(0, logCumulative));
}

/**
 * P-value of the geographical detector q statistic, `1 - F_ncf(F; L - 1, N - L, lambda)`, from the
 * read-back `output.summary` of `GPUGeographicalDetector` (`geodetector` 0.2.0
 * `q_significance_test`). NaN where the summary is undefined (`F` is NaN); 0 for `q = 1`.
 * The supported non-centrality range and convergence errors are those of
 * {@link getNonCentralFUpperTail}.
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

/**
 * Reads an optional derived probability without discarding the detector's other summary values.
 * Numerical range/convergence failures and undefined statistics return a null p-value with a
 * reason; unrelated errors still throw. The strict {@link getGeographicalDetectorPValue} retains
 * its throwing contract for callers that require a probability.
 */
export function readGeographicalDetectorPValue(summary: ArrayLike<number>): {
  pValue: number | null;
  unavailableReason: string | null;
} {
  try {
    const pValue = getGeographicalDetectorPValue(summary);
    return Number.isFinite(pValue)
      ? {pValue, unavailableReason: null}
      : {pValue: null, unavailableReason: 'Geographical detector summary has undefined statistics'};
  } catch (error) {
    if (!(error instanceof Error) || !error.message.startsWith('getNonCentralFUpperTail ')) {
      throw error;
    }
    return {pValue: null, unavailableReason: error.message};
  }
}
