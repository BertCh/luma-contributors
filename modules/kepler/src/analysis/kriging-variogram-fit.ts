// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  fitVariogramModel,
  type VariogramModelType
} from '@luma-contributors/analysis/dataframe-analysis';
import {toDegrees, toRadians} from '@math.gl/core';
import {EARTH_RADIUS_METERS, getHaversineDistanceMeters} from './catalog-geodesy';

export type KeplerEmpiricalVariogramOptions = {
  /** Equal-width lag bins over `(0, maximumLagMeters]`. Defaults to 20. */
  binCount?: number;
  /** Largest pair separation in spherical meters. Defaults to a third of the data diagonal. */
  maximumLagMeters?: number;
  /**
   * Largest number of pairs evaluated. Candidate pairs beyond it are thinned by a deterministic
   * hash of the two row indices, so the result is independent of row order and chunking.
   * Defaults to 4 million.
   */
  maximumPairs?: number;
  /** Seed of the thinning hash. Defaults to 0. */
  seed?: number;
};

export type KeplerEmpiricalVariogram = {
  /** Mean pair distance per bin in meters (NaN for empty bins). */
  distances: Float64Array;
  /** Matheron semivariance per bin (NaN for empty bins). */
  semivariances: Float64Array;
  pairCounts: Float64Array;
  maximumLagMeters: number;
  /** Rows with finite coordinates inside the geographic bounds and a finite value. */
  validRowCount: number;
  /** Pairs within the maximum lag that were accumulated. */
  accumulatedPairCount: number;
  /** Pairs within the maximum lag before thinning (an estimate when thinned). */
  candidatePairCount: number;
  /** True when every candidate pair was used. */
  exhaustive: boolean;
  /** Variance of the valid values, the usual sill reference. */
  valueVariance: number;
};

/**
 * Binned empirical semivariogram of geographic point values (Matheron estimator, great-circle
 * meters, float64 arithmetic). Pairs are found through a latitude/longitude bucket grid at least
 * one lag wide, so cost follows the number of pairs within the lag rather than all N squared pairs.
 * The antimeridian and poles are not wrapped, matching the kriging adapter's supported bounds.
 */
export function estimateKeplerEmpiricalVariogram(
  longitudes: ArrayLike<number>,
  latitudes: ArrayLike<number>,
  values: ArrayLike<number>,
  options: KeplerEmpiricalVariogramOptions = {}
): KeplerEmpiricalVariogram {
  const binCount = options.binCount ?? 20;
  const maximumPairs = options.maximumPairs ?? 4_000_000;
  const seed = (options.seed ?? 0) >>> 0;
  if (!Number.isSafeInteger(binCount) || binCount < 2 || binCount > 512 || !(maximumPairs >= 1))
    throw new Error('Invalid empirical variogram options');
  const rowIndices: number[] = [];
  let minimumLongitude = Infinity;
  let maximumLongitude = -Infinity;
  let minimumLatitude = Infinity;
  let maximumLatitude = -Infinity;
  let valueSum = 0;
  for (let row = 0; row < values.length; row++) {
    const longitude = longitudes[row];
    const latitude = latitudes[row];
    const value = values[row];
    if (
      Number.isFinite(longitude) &&
      Number.isFinite(latitude) &&
      Number.isFinite(value) &&
      Math.abs(longitude) <= 180 &&
      Math.abs(latitude) <= 90
    ) {
      rowIndices.push(row);
      valueSum += value;
      minimumLongitude = Math.min(minimumLongitude, longitude);
      maximumLongitude = Math.max(maximumLongitude, longitude);
      minimumLatitude = Math.min(minimumLatitude, latitude);
      maximumLatitude = Math.max(maximumLatitude, latitude);
    }
  }
  const validRowCount = rowIndices.length;
  const mean = validRowCount ? valueSum / validRowCount : 0;
  let squaredSum = 0;
  for (const row of rowIndices) squaredSum += (values[row] - mean) ** 2;
  const valueVariance = validRowCount ? squaredSum / validRowCount : 0;
  const distances = new Float64Array(binCount).fill(NaN);
  const semivariances = new Float64Array(binCount).fill(NaN);
  const pairCounts = new Float64Array(binCount);
  if (validRowCount < 2) {
    return {
      distances,
      semivariances,
      pairCounts,
      maximumLagMeters: options.maximumLagMeters ?? 0,
      validRowCount,
      accumulatedPairCount: 0,
      candidatePairCount: 0,
      exhaustive: true,
      valueVariance
    };
  }
  // Multiplying by toRadians(1) is exactly toRadians(x) without per-call overhead in the pair loop.
  const degreesToRadians = toRadians(1);
  const diagonalMeters =
    getHaversineDistanceMeters(
      (maximumLatitude - minimumLatitude) * degreesToRadians,
      (maximumLongitude - minimumLongitude) * degreesToRadians,
      minimumLatitude * degreesToRadians,
      maximumLatitude * degreesToRadians
    ) || 1;
  const maximumLagMeters = options.maximumLagMeters ?? diagonalMeters / 3;
  if (!(maximumLagMeters > 0) || !Number.isFinite(maximumLagMeters))
    throw new Error('Invalid empirical variogram maximum lag');
  // Bucket cells at least one lag wide in both directions (longitude width at the highest latitude).
  const latitudeStep = toDegrees(maximumLagMeters / EARTH_RADIUS_METERS) * 1.0001;
  const highestLatitude = Math.min(
    89,
    Math.max(Math.abs(minimumLatitude), Math.abs(maximumLatitude))
  );
  const longitudeStep = latitudeStep / Math.cos(highestLatitude * degreesToRadians);
  const columnCount = Math.ceil(360 / longitudeStep) + 2;
  const cells = new Map<number, number[]>();
  const cellOf = (row: number) =>
    Math.floor((latitudes[row] + 90) / latitudeStep) * columnCount +
    Math.floor((longitudes[row] + 180) / longitudeStep);
  for (const row of rowIndices) {
    const key = cellOf(row);
    const members = cells.get(key);
    if (members) members.push(row);
    else cells.set(key, [row]);
  }
  const neighborOffsets = [
    -columnCount - 1,
    -columnCount,
    -columnCount + 1,
    -1,
    0,
    1,
    columnCount - 1,
    columnCount,
    columnCount + 1
  ];
  // Candidate-pair estimate (cells overlap pairs), used to choose the thinning probability.
  let candidateEstimate = 0;
  for (const [key, members] of cells) {
    for (const offset of neighborOffsets) {
      const other = cells.get(key + offset);
      if (other) candidateEstimate += members.length * other.length;
    }
  }
  candidateEstimate = Math.max(1, candidateEstimate / 2);
  const keepProbability = Math.min(1, maximumPairs / candidateEstimate);
  const exhaustive = keepProbability >= 1;
  const keepThreshold = Math.floor(keepProbability * 4294967296);
  const binSums = new Float64Array(binCount);
  const distanceSums = new Float64Array(binCount);
  let accumulatedPairCount = 0;
  let withinLagCount = 0;
  for (const [key, members] of cells) {
    for (const offset of neighborOffsets) {
      const other = cells.get(key + offset);
      if (!other) continue;
      for (const first of members) {
        const firstLongitude = longitudes[first] * degreesToRadians;
        const firstLatitude = latitudes[first] * degreesToRadians;
        for (const second of other) {
          // Each unordered pair once: lower row index first.
          if (second <= first) continue;
          const secondLongitude = longitudes[second] * degreesToRadians;
          const secondLatitude = latitudes[second] * degreesToRadians;
          const distance = getHaversineDistanceMeters(
            secondLatitude - firstLatitude,
            secondLongitude - firstLongitude,
            firstLatitude,
            secondLatitude
          );
          if (!(distance <= maximumLagMeters)) continue;
          withinLagCount++;
          if (!exhaustive && hashPair(first, second, seed) >= keepThreshold) continue;
          const bin = Math.min(binCount - 1, Math.floor((distance / maximumLagMeters) * binCount));
          const difference = values[first] - values[second];
          binSums[bin] += 0.5 * difference * difference;
          distanceSums[bin] += distance;
          pairCounts[bin]++;
          accumulatedPairCount++;
        }
      }
    }
  }
  for (let bin = 0; bin < binCount; bin++) {
    if (pairCounts[bin] > 0) {
      semivariances[bin] = binSums[bin] / pairCounts[bin];
      distances[bin] = distanceSums[bin] / pairCounts[bin];
    }
  }
  // Pairs of two points in the same cell are visited once per offset 0 only; neighbors twice-safe:
  // the lower-index rule above counts each unordered pair exactly once across cell pairs.
  return {
    distances,
    semivariances,
    pairCounts,
    maximumLagMeters,
    validRowCount,
    accumulatedPairCount,
    candidatePairCount: exhaustive ? withinLagCount : Math.round(withinLagCount),
    exhaustive,
    valueVariance
  };
}

/** Order-independent 32-bit hash of an unordered row pair. */
function hashPair(first: number, second: number, seed: number): number {
  let hash = (Math.imul(first, 0x9e3779b1) ^ Math.imul(second, 0x85ebca6b) ^ seed) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

export type KeplerVariogramCandidate = {
  model: VariogramModelType;
  nugget: number;
  sill: number;
  rangeMeters: number;
  /** Weighted sum of squares with the common weights `N_h / gamma_h^2` (comparable across models). */
  criterion: number;
};

export type KeplerVariogramFit = KeplerVariogramCandidate & {
  method: 'weighted-least-squares';
  weighting: 'cressie';
  /** Every attempted family, ranked best first. */
  candidates: KeplerVariogramCandidate[];
  binCount: number;
  pairCount: number;
};

/**
 * Automatic variogram selection: each requested family is fitted to the binned empirical
 * semivariogram by Cressie-weighted least squares (nugget and partial sill by closed-form
 * non-negative least squares, range by ladder plus golden-section search) and the family with the
 * smallest common-weight criterion is chosen. Parameters are rounded to Float32 so the result is
 * a valid kriging variogram that replays exactly.
 */
export function fitKeplerVariogram(
  empirical: Pick<KeplerEmpiricalVariogram, 'distances' | 'semivariances' | 'pairCounts'>,
  models: readonly VariogramModelType[] = ['spherical', 'exponential', 'gaussian']
): KeplerVariogramFit {
  const candidates: KeplerVariogramCandidate[] = [];
  for (const model of models) {
    try {
      const fitted = fitVariogramModel(
        {
          distances: empirical.distances,
          semivariances: empirical.semivariances,
          pairCounts: empirical.pairCounts
        },
        {model, weighting: 'cressie'}
      );
      const nugget = Math.fround(fitted.nugget);
      const sill = Math.fround(fitted.sill);
      const rangeMeters = Math.fround(fitted.range);
      if (
        !Number.isFinite(nugget) ||
        !Number.isFinite(sill) ||
        !Number.isFinite(rangeMeters) ||
        !(rangeMeters > 0) ||
        !(Math.fround(nugget + sill) > 0)
      )
        continue;
      candidates.push({
        model,
        nugget,
        sill,
        rangeMeters,
        criterion: getCommonCriterion(empirical, {model, nugget, sill, range: rangeMeters})
      });
    } catch {
      // A family that cannot be fitted to these bins (too few usable bins) is skipped.
    }
  }
  candidates.sort((first, second) => first.criterion - second.criterion);
  if (!candidates.length)
    throw new Error('Variogram fitting needs at least two populated lag bins');
  let pairCount = 0;
  for (const count of empirical.pairCounts) pairCount += count;
  return {
    ...candidates[0],
    method: 'weighted-least-squares',
    weighting: 'cressie',
    candidates,
    binCount: empirical.pairCounts.length,
    pairCount
  };
}

function getCommonCriterion(
  empirical: Pick<KeplerEmpiricalVariogram, 'distances' | 'semivariances' | 'pairCounts'>,
  model: {model: VariogramModelType; nugget: number; sill: number; range: number}
): number {
  let total = 0;
  for (let bin = 0; bin < empirical.distances.length; bin++) {
    const distance = empirical.distances[bin];
    const gamma = empirical.semivariances[bin];
    const count = empirical.pairCounts[bin];
    if (!(count > 0) || !(gamma > 0) || !(distance > 0)) continue;
    const scaled = distance / model.range;
    const shape =
      model.model === 'spherical'
        ? scaled >= 1
          ? 1
          : 1.5 * scaled - 0.5 * scaled ** 3
        : model.model === 'exponential'
          ? 1 - Math.exp(-3 * scaled)
          : 1 - Math.exp(-3 * scaled * scaled);
    const predicted = model.nugget + model.sill * shape;
    total += (count * (gamma - predicted) ** 2) / (gamma * gamma);
  }
  return total;
}

/** Standard normal quantile (Acklam's rational approximation, relative error below 1.2e-9). */
export function getStandardNormalQuantile(probability: number): number {
  if (!(probability > 0 && probability < 1)) throw new Error('Probability must be in (0, 1)');
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const low = 0.02425;
  if (probability < low) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (probability > 1 - low) return -getStandardNormalQuantile(1 - probability);
  const q = probability - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

export type KeplerKrigingInterval = {
  /** Two-sided standard normal multiplier, `z_{(1 + level) / 2}`. */
  multiplier: number;
  lower: Float32Array;
  upper: Float32Array;
};

/**
 * Two-sided prediction interval `prediction +/- z * sqrt(variance)` per cell. The variance is the
 * model-based ordinary kriging variance under the supplied (or fitted) variogram, so the interval
 * is conditional on that model and assumes Gaussian errors; it ignores variogram estimation
 * uncertainty. Cells with NaN prediction or negative/NaN variance yield NaN bounds.
 */
export function getKrigingConfidenceInterval(
  prediction: ArrayLike<number>,
  variance: ArrayLike<number>,
  level: number
): KeplerKrigingInterval {
  if (!(level > 0 && level < 1)) throw new Error('Confidence level must be in (0, 1)');
  const multiplier = getStandardNormalQuantile((1 + level) / 2);
  const lower = new Float32Array(prediction.length);
  const upper = new Float32Array(prediction.length);
  for (let cell = 0; cell < prediction.length; cell++) {
    const spread = multiplier * Math.sqrt(variance[cell]);
    lower[cell] = prediction[cell] - spread;
    upper[cell] = prediction[cell] + spread;
  }
  return {multiplier, lower, upper};
}

export type KeplerKrigingCrossValidationSummary = {
  /** Samples with a finite observation and a finite leave-one-out prediction. */
  count: number;
  meanError: number;
  meanAbsoluteError: number;
  rootMeanSquareError: number;
  /** Samples that also have a positive finite variance. */
  standardizedCount: number;
  meanStandardizedError: number;
  /** Close to 1 for a well-calibrated model. */
  rootMeanSquareStandardizedError: number;
  /** Fraction of standardized samples whose observation lies inside the `level` interval. */
  coverage: number;
  level: number;
};

/**
 * Leave-one-out diagnostics. Error is `prediction - observed`; the standardized error is
 * `error / sqrt(variance)`. `coverage` should approximate `level` when the variogram is adequate.
 */
export function summarizeKrigingCrossValidation(
  observed: ArrayLike<number>,
  prediction: ArrayLike<number>,
  variance: ArrayLike<number>,
  level = 0.95
): KeplerKrigingCrossValidationSummary {
  const multiplier = getStandardNormalQuantile((1 + level) / 2);
  let count = 0;
  let errorSum = 0;
  let absoluteSum = 0;
  let squaredSum = 0;
  let standardizedCount = 0;
  let standardizedSum = 0;
  let standardizedSquaredSum = 0;
  let covered = 0;
  for (let row = 0; row < observed.length; row++) {
    const error = prediction[row] - observed[row];
    if (!Number.isFinite(error)) continue;
    count++;
    errorSum += error;
    absoluteSum += Math.abs(error);
    squaredSum += error * error;
    const rowVariance = variance[row];
    if (rowVariance > 0 && Number.isFinite(rowVariance)) {
      const standardized = error / Math.sqrt(rowVariance);
      standardizedCount++;
      standardizedSum += standardized;
      standardizedSquaredSum += standardized * standardized;
      if (Math.abs(standardized) <= multiplier) covered++;
    }
  }
  return {
    count,
    meanError: count ? errorSum / count : NaN,
    meanAbsoluteError: count ? absoluteSum / count : NaN,
    rootMeanSquareError: count ? Math.sqrt(squaredSum / count) : NaN,
    standardizedCount,
    meanStandardizedError: standardizedCount ? standardizedSum / standardizedCount : NaN,
    rootMeanSquareStandardizedError: standardizedCount
      ? Math.sqrt(standardizedSquaredSum / standardizedCount)
      : NaN,
    coverage: standardizedCount ? covered / standardizedCount : NaN,
    level
  };
}
