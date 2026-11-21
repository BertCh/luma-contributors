// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A planar track on the CPU, in binary64. */
export type BrownianBridgeTrack = {
  /** Fix x coordinates. */
  x: ArrayLike<number>;
  /** Fix y coordinates. */
  y: ArrayLike<number>;
  /** Strictly increasing fix times, in the unit the variance is per. */
  times: ArrayLike<number>;
};

/** Options of {@link estimateBrownianMotionVariance}. */
export type BrownianMotionVarianceOptions = {
  /**
   * Standard deviation of the location error, in the unit of `x` and `y`: one number for every fix,
   * or one per fix. Defaults to 0.
   */
  locationError?: number | ArrayLike<number>;
};

/** Options of {@link estimateWindowedBrownianMotionVariance}. */
export type WindowedBrownianMotionVarianceOptions = BrownianMotionVarianceOptions & {
  /** Number of fixes per window. Defaults to 31. Windows are centred and clamped to the track ends. */
  windowSize?: number;
};

/** A motion variance estimate and its optimization diagnostic. */
export type BrownianMotionVarianceEstimate = {
  motionVariance: number;
  status: {
    /** True when the returned estimate is exactly the nonnegative lower bound, zero. */
    lowerBoundReached: boolean;
  };
};

/** Sliding-window estimates and the diagnostic for each segment's window. */
export type WindowedBrownianMotionVarianceEstimate = {
  motionVariances: Float64Array;
  segmentStatus: BrownianMotionVarianceEstimate['status'][];
};

const GRID_POINT_COUNT = 160;
const REFINEMENT_COUNT = 80;

function getError(locationError: number | ArrayLike<number> | undefined, index: number): number {
  if (locationError === undefined) {
    return 0;
  }
  return typeof locationError === 'number' ? locationError : locationError[index];
}

/**
 * Log-likelihood terms of a leave-one-out Brownian bridge variance estimator in the manner of
 * Horne, Garton, Krone and Lewis (Ecology 88, 2007): every second fix `i` (1, 3, 5, ...) is
 * predicted from the fixes `i - 1` and `i + 1`. With `T = t[i+1] - t[i-1]` and
 * `alpha = (t[i] - t[i-1]) / T` the prediction is the point at fraction `alpha` of the chord, and
 * the observed fix has the isotropic bivariate normal density with variance
 * `s^2 = T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta[i-1]^2 + alpha^2 delta[i+1]^2 + delta[i]^2`.
 *
 * The last term, `delta[i]^2`, is the location error of the predicted fix itself: the residual is
 * between two observed positions, and the observed fix `i` carries its own error. The bridge
 * variance as usually written for Horne et al. and the R packages `BBMM` and `move` (not run or
 * read here) stops at the first three terms, which describe the true position. With a location
 * error above 0 this estimator therefore returns a smaller motion variance than those packages
 * would for the same input; with a location error of 0 the two forms are the same.
 */
function getVarianceTerms(
  track: BrownianBridgeTrack,
  locationError: number | ArrayLike<number> | undefined,
  first: number,
  last: number
) {
  const terms: {weight: number; errorVariance: number; squaredResidual: number}[] = [];
  for (let index = first + 1; index + 1 <= last; index += 2) {
    const span = track.times[index + 1] - track.times[index - 1];
    const alpha = (track.times[index] - track.times[index - 1]) / span;
    const expectedX = track.x[index - 1] + alpha * (track.x[index + 1] - track.x[index - 1]);
    const expectedY = track.y[index - 1] + alpha * (track.y[index + 1] - track.y[index - 1]);
    const errorBefore = getError(locationError, index - 1);
    const errorAfter = getError(locationError, index + 1);
    const errorHere = getError(locationError, index);
    terms.push({
      weight: span * alpha * (1 - alpha),
      errorVariance:
        (1 - alpha) ** 2 * errorBefore ** 2 + alpha ** 2 * errorAfter ** 2 + errorHere ** 2,
      squaredResidual: (track.x[index] - expectedX) ** 2 + (track.y[index] - expectedY) ** 2
    });
  }
  return terms;
}

function getLogLikelihood(
  terms: ReturnType<typeof getVarianceTerms>,
  motionVariance: number
): number {
  let total = 0;
  for (const {weight, errorVariance, squaredResidual} of terms) {
    const variance = weight * motionVariance + errorVariance;
    if (!(variance > 0) || !Number.isFinite(variance)) {
      return -Infinity;
    }
    total += -Math.log(2 * Math.PI * variance) - squaredResidual / (2 * variance);
  }
  return total;
}

/** L(second) - L(first), retaining small changes that disappear in absolute log-likelihoods. */
function getLogLikelihoodDifference(
  terms: ReturnType<typeof getVarianceTerms>,
  first: number,
  second: number
): number {
  let total = 0;
  for (const {weight, errorVariance, squaredResidual} of terms) {
    const variance = weight * first + errorVariance;
    const nextVariance = weight * second + errorVariance;
    if (!(variance > 0) || !(nextVariance > 0)) {
      return getLogLikelihood(terms, second) - getLogLikelihood(terms, first);
    }
    const change = (weight * (second - first)) / variance;
    if (Math.abs(change) >= 0.01) {
      // Do not cancel x against -x for large variance ratios: the remaining logarithm is
      // the entire penalty for a zero residual with a very small location error.
      total +=
        Math.log(variance) -
        Math.log(nextVariance) +
        (squaredResidual === 0
          ? 0
          : (squaredResidual / variance - squaredResidual / nextVariance) / 2);
      continue;
    }
    // x - log1p(x) starts at x^2 / 2. Subtracting the two directly would erase the
    // likelihood gain of a positive optimum very near the nonnegative boundary.
    const logarithmRemainder =
      Math.abs(change) < 0.01
        ? change *
          change *
          (1 / 2 +
            change *
              (-1 / 3 +
                change *
                  (1 / 4 +
                    change * (-1 / 5 + change * (1 / 6 + change * (-1 / 7 + change * (1 / 8)))))))
        : change - Math.log1p(change);
    const residualExcess = (squaredResidual - 2 * variance) / (2 * variance);
    total += logarithmRemainder + ((residualExcess - change) * change) / (1 + change);
  }
  return total;
}

/** Maximum-likelihood motion variance of the given terms, by a log-spaced scan then golden section. */
function maximizeLikelihood(
  terms: ReturnType<typeof getVarianceTerms>
): BrownianMotionVarianceEstimate {
  if (
    terms.length === 0 ||
    terms.some(
      ({weight, errorVariance, squaredResidual}) =>
        !Number.isFinite(weight) ||
        weight <= 0 ||
        !Number.isFinite(errorVariance) ||
        errorVariance < 0 ||
        !Number.isFinite(squaredResidual) ||
        squaredResidual < 0
    )
  ) {
    return {motionVariance: NaN, status: {lowerBoundReached: false}};
  }
  // A zero-error, zero-residual bridge has unbounded density as motion tends to
  // zero. Other zero-error bridges must also have zero residual: any positive
  // residual instead contributes a dominating negative inverse-variance term.
  const zeroErrorTerms = terms.filter(term => term.errorVariance === 0);
  if (zeroErrorTerms.length > 0 && zeroErrorTerms.every(term => term.squaredResidual === 0)) {
    return {motionVariance: 0, status: {lowerBoundReached: true}};
  }
  // Scale of the answer: the closed form of the error-free estimator,
  // sigma^2 = mean(r^2 / (2 w)), which is exact when every location error is 0.
  let scale = 0;
  for (const {weight, squaredResidual} of terms) {
    scale += squaredResidual / (2 * weight);
  }
  scale = scale / terms.length;
  if (!Number.isFinite(scale)) {
    return {motionVariance: NaN, status: {lowerBoundReached: false}};
  }
  if (scale === 0) {
    return {motionVariance: 0, status: {lowerBoundReached: true}};
  }
  if (terms.every(term => term.errorVariance === 0)) {
    return {motionVariance: scale, status: {lowerBoundReached: false}};
  }
  // Heterogeneous errors can produce separate peaks many orders of magnitude apart. Include
  // each term's error transition and positive optimum instead of bracketing only the mean scale.
  const scales = new Set([scale]);
  for (const {weight, errorVariance, squaredResidual} of terms) {
    for (const candidate of [
      errorVariance / weight,
      (squaredResidual / 2 - errorVariance) / weight
    ]) {
      if (candidate > 0 && Number.isFinite(candidate)) scales.add(candidate);
    }
  }
  const candidates = new Set<number>([0]);
  const intervals = [...scales]
    .map(value => [Math.log(value) - 12, Math.log(value) + 6])
    .sort((first, second) => first[0] - second[0]);
  const merged: number[][] = [];
  for (const interval of intervals) {
    const previous = merged[merged.length - 1];
    if (previous && interval[0] <= previous[1]) previous[1] = Math.max(previous[1], interval[1]);
    else merged.push([...interval]);
  }
  for (const [lowest, highest] of merged) {
    const count = Math.ceil((highest - lowest) / (18 / (GRID_POINT_COUNT - 1)));
    for (let index = 0; index <= count; index++) {
      const candidate = Math.exp(lowest + ((highest - lowest) * index) / count);
      if (Number.isFinite(candidate)) candidates.add(candidate);
    }
  }
  const grid = [...candidates].sort((first, second) => first - second);
  let best = NaN;
  const consider = (candidate: number) => {
    if (
      Number.isFinite(getLogLikelihood(terms, candidate)) &&
      (Number.isNaN(best) || getLogLikelihoodDifference(terms, best, candidate) > 0)
    )
      best = candidate;
  };
  for (const candidate of grid) consider(candidate);
  const ratio = (Math.sqrt(5) - 1) / 2;
  for (let index = 0; index < grid.length - 1; index++) {
    // The first interval can contain a maximum even when its positive sample loses to zero.
    // Always refine it before declaring that the nonnegative boundary won.
    if (
      index !== 0 &&
      (getLogLikelihoodDifference(terms, grid[index - 1], grid[index]) < 0 ||
        getLogLikelihoodDifference(terms, grid[index], grid[index + 1]) > 0)
    )
      continue;
    const fromZero = index === 0 || grid[index - 1] === 0;
    let low = fromZero ? 0 : Math.log(grid[index - 1]);
    let high = fromZero ? grid[index + 1] : Math.log(grid[index + 1]);
    const getVariance = (value: number) => (fromZero ? value : Math.exp(value));
    for (let iteration = 0; iteration < REFINEMENT_COUNT; iteration++) {
      const left = high - ratio * (high - low);
      const right = low + ratio * (high - low);
      if (getLogLikelihoodDifference(terms, getVariance(left), getVariance(right)) > 0) low = left;
      else high = right;
    }
    consider(getVariance(0.5 * (low + high)));
  }
  return {motionVariance: best, status: {lowerBoundReached: best === 0}};
}

/**
 * Maximum-likelihood Brownian motion variance of one track (units of length squared per time
 * unit), in the manner of Horne et al. (2007): the variance that best predicts every second fix
 * from its two neighbours. The estimate is the input of {@link GPUBrownianBridgeUtilization}
 * (`motionVariances`). With a zero location error it equals
 * `mean(r^2 / (2 T alpha (1 - alpha)))` over the predicted fixes, which the tests use as a closed
 * form.
 *
 * With a location error the likelihood variance includes the predicted fix's own error
 * (`+ delta[i]^2`, see the terms above), so the estimate is lower than one from the three-term
 * bridge variance of `BBMM` and `move`. On a simulated random walk of true variance 15 with a
 * location error of 20 on every fix this form returned 15.6 and the three-term form 28.5 (review
 * of 2026-11-10, a binary64 script).
 *
 * This is a CPU, binary64 helper: the optimizer is a log-spaced scan followed by golden-section
 * refinement. The R packages `move` and `BBMM` (licences not verified) were not consulted or run,
 * so no parity with their estimates or optimizers is claimed.
 *
 * @returns The variance, or NaN when there are fewer than three fixes, unusable terms (including
 * non-finite coordinates, times or location errors), or no finite likelihood candidate.
 */
export function estimateBrownianMotionVariance(
  track: BrownianBridgeTrack,
  options: BrownianMotionVarianceOptions = {}
): number {
  return estimateBrownianMotionVarianceWithStatus(track, options).motionVariance;
}

/**
 * The same estimator as {@link estimateBrownianMotionVariance}, including the exact zero
 * candidate. `status.lowerBoundReached` reports whether zero was selected. With fewer than
 * three fixes or unusable terms/likelihoods the estimate is NaN and the flag is false.
 */
export function estimateBrownianMotionVarianceWithStatus(
  track: BrownianBridgeTrack,
  options: BrownianMotionVarianceOptions = {}
): BrownianMotionVarianceEstimate {
  return maximizeLikelihood(
    getVarianceTerms(track, options.locationError, 0, track.times.length - 1)
  );
}

/**
 * Brownian motion variance per segment from a sliding window, the variance input of a dynamic
 * Brownian bridge (Kranstauber et al., 2012) without its change-point step: segment `j` (from fix
 * `j` to fix `j + 1`) takes the variance estimated by {@link estimateBrownianMotionVariance} on a
 * window of `windowSize` fixes centred on it. Windows are shifted to stay inside the track, and a
 * track shorter than the window uses the whole track.
 *
 * The behavioural change point analysis of the dynamic Brownian bridge (a BIC comparison of split
 * windows) is not implemented, so the variance changes smoothly with the window instead of
 * jumping at detected behavioural changes.
 *
 * @returns One value per segment (`times.length - 1`), NaN where a window has fewer than three fixes.
 */
export function estimateWindowedBrownianMotionVariance(
  track: BrownianBridgeTrack,
  options: WindowedBrownianMotionVarianceOptions = {}
): Float64Array {
  return computeWindowedMotionVariances(track, options);
}

/** Windowed estimates with the zero-bound diagnostic for each segment, in segment order. */
export function estimateWindowedBrownianMotionVarianceWithStatus(
  track: BrownianBridgeTrack,
  options: WindowedBrownianMotionVarianceOptions = {}
): WindowedBrownianMotionVarianceEstimate {
  const segmentStatus: BrownianMotionVarianceEstimate['status'][] = [];
  return {
    motionVariances: computeWindowedMotionVariances(track, options, segmentStatus),
    segmentStatus
  };
}

function computeWindowedMotionVariances(
  track: BrownianBridgeTrack,
  options: WindowedBrownianMotionVarianceOptions,
  segmentStatus?: BrownianMotionVarianceEstimate['status'][]
): Float64Array {
  const fixCount = track.times.length;
  const windowSize = Math.min(options.windowSize ?? 31, fixCount);
  const variances = new Float64Array(Math.max(fixCount - 1, 0));
  for (let segment = 0; segment < variances.length; segment++) {
    const first = Math.min(
      Math.max(segment + 1 - Math.floor(windowSize / 2), 0),
      fixCount - windowSize
    );
    const estimate = maximizeLikelihood(
      getVarianceTerms(track, options.locationError, first, first + windowSize - 1)
    );
    variances[segment] = estimate.motionVariance;
    segmentStatus?.push(estimate.status);
  }
  return variances;
}
