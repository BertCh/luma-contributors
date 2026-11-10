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

const GRID_POINT_COUNT = 160;
const REFINEMENT_COUNT = 80;

function getError(locationError: number | ArrayLike<number> | undefined, index: number): number {
  if (locationError === undefined) {
    return 0;
  }
  return typeof locationError === 'number' ? locationError : locationError[index];
}

/**
 * Log-likelihood terms of the Brownian bridge variance estimator of Horne, Garton, Krone and
 * Lewis (Ecology 88, 2007): every second fix `i` (1, 3, 5, ...) is predicted from the fixes
 * `i - 1` and `i + 1`. With `T = t[i+1] - t[i-1]` and `alpha = (t[i] - t[i-1]) / T` the prediction
 * is the point at fraction `alpha` of the chord, with variance
 * `s^2 = T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta[i-1]^2 + alpha^2 delta[i+1]^2 + delta[i]^2`,
 * and the observed fix has the isotropic bivariate normal density with that variance.
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
    if (!(variance > 0)) {
      return -Infinity;
    }
    total += -Math.log(2 * Math.PI * variance) - squaredResidual / (2 * variance);
  }
  return total;
}

/** Maximum-likelihood motion variance of the given terms, by a log-spaced scan then golden section. */
function maximizeLikelihood(terms: ReturnType<typeof getVarianceTerms>): number {
  if (terms.length === 0) {
    return NaN;
  }
  // Scale of the answer: the closed form of the error-free estimator,
  // sigma^2 = mean(r^2 / (2 w)), which is exact when every location error is 0.
  let scale = 0;
  for (const {weight, squaredResidual} of terms) {
    scale += squaredResidual / (2 * weight);
  }
  scale = scale / terms.length;
  if (!(scale > 0) || !Number.isFinite(scale)) {
    return 0;
  }
  const lowest = Math.log(scale) - 12;
  const highest = Math.log(scale) + 6;
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let index = 0; index < GRID_POINT_COUNT; index++) {
    const logVariance = lowest + ((highest - lowest) * index) / (GRID_POINT_COUNT - 1);
    const value = getLogLikelihood(terms, Math.exp(logVariance));
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  const step = (highest - lowest) / (GRID_POINT_COUNT - 1);
  let low = lowest + Math.max(bestIndex - 1, 0) * step;
  let high = lowest + Math.min(bestIndex + 1, GRID_POINT_COUNT - 1) * step;
  const ratio = (Math.sqrt(5) - 1) / 2;
  for (let iteration = 0; iteration < REFINEMENT_COUNT; iteration++) {
    const left = high - ratio * (high - low);
    const right = low + ratio * (high - low);
    if (getLogLikelihood(terms, Math.exp(left)) < getLogLikelihood(terms, Math.exp(right))) {
      low = left;
    } else {
      high = right;
    }
  }
  return Math.exp(0.5 * (low + high));
}

/**
 * Maximum-likelihood Brownian motion variance of one track (units of length squared per time
 * unit), after Horne et al. (2007): the variance that best predicts every second fix from its two
 * neighbours. The estimate is the input of {@link GPUBrownianBridgeUtilization} (`motionVariances`).
 * With a zero location error it equals `mean(r^2 / (2 T alpha (1 - alpha)))` over the predicted
 * fixes, which the tests use as a closed form.
 *
 * This is a CPU, binary64 helper: the optimizer is a log-spaced scan followed by golden-section
 * refinement. It is written from the published definition; the R package `move` (licence not
 * verified) was not consulted or run, so no claim of parity with its optimizer is made.
 *
 * @returns The variance, or NaN when the track has fewer than three fixes.
 */
export function estimateBrownianMotionVariance(
  track: BrownianBridgeTrack,
  options: BrownianMotionVarianceOptions = {}
): number {
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
  const fixCount = track.times.length;
  const windowSize = Math.min(options.windowSize ?? 31, fixCount);
  const variances = new Float64Array(Math.max(fixCount - 1, 0));
  for (let segment = 0; segment < variances.length; segment++) {
    const first = Math.min(
      Math.max(segment + 1 - Math.floor(windowSize / 2), 0),
      fixCount - windowSize
    );
    variances[segment] = maximizeLikelihood(
      getVarianceTerms(track, options.locationError, first, first + windowSize - 1)
    );
  }
  return variances;
}
