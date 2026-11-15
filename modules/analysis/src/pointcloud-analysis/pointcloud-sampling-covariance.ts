// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';

/** Exact empirical pair fitting is quadratic on the CPU; bound it to about eight million pairs. */
export const POINT_CLOUD_MAXIMUM_COVARIANCE_FIT_POINTS = 4096;
/** Each candidate length adds one model evaluation per pair; bound the reference fitter's work. */
export const POINT_CLOUD_MAXIMUM_COVARIANCE_FIT_LENGTHS = 32;

export type PointCloudSamplingCovarianceFit = {
  variance: number;
  correlationLength: number;
  /** Mean squared residual in squared-semivariance units across all distinct unordered pairs. */
  meanSquaredError: number;
};

/**
 * Fit a zero-nugget Gaussian covariance to detrended normal-direction residuals on the CPU.
 *
 * At each supplied correlation length, fit the nonnegative variance by ordinary least squares to
 * pair semivariances `(residual[first] - residual[second])² / 2`, with model
 * `variance * (1 - exp(-distance² / (2 * correlationLength²)))`. Choose the least squared-error
 * length, breaking exact ties by the smaller length. With two distinct points every length fits
 * the single semivariance exactly; its error is analytically zero, so the smallest length wins.
 * This finite-grid fit does not certify a
 * stochastic model, infer detrending, or fit registration error. Fit epochs separately; pass the
 * resulting variances and a shared length to GPUPointCloudChange.samplingCovariance (or fit both
 * with a common fixed candidate length). Uses float64 and all unordered pairs, at most 4096 points
 * and 32 length candidates. No GPU readback is performed by this helper.
 */
export function fitPointCloudSamplingCovariance(
  positions: ArrayLike<number>,
  residuals: ArrayLike<number>,
  correlationLengths: readonly number[]
): PointCloudSamplingCovarianceFit {
  const identifier = 'fit-pointcloud-sampling-covariance';
  validateCapacity(
    identifier,
    'residuals.length',
    residuals.length,
    2,
    POINT_CLOUD_MAXIMUM_COVARIANCE_FIT_POINTS
  );
  validateRequirement(
    identifier,
    positions.length === residuals.length * 3,
    'positions.length must equal residuals.length * 3'
  );
  validateCapacity(
    identifier,
    'correlationLengths.length',
    correlationLengths.length,
    1,
    POINT_CLOUD_MAXIMUM_COVARIANCE_FIT_LENGTHS
  );
  for (let component = 0; component < positions.length; component++) {
    validateRequirement(
      identifier,
      Number.isFinite(positions[component]),
      'positions must contain finite numbers'
    );
  }
  for (let row = 0; row < residuals.length; row++) {
    validateRequirement(
      identifier,
      Number.isFinite(residuals[row]),
      'residuals must contain finite numbers'
    );
  }
  correlationLengths.forEach(length =>
    validateRequirement(
      identifier,
      length > 0 && length * length > 0 && Number.isFinite(length * length),
      'correlationLengths must have positive finite nonzero squares'
    )
  );
  let bestFit: PointCloudSamplingCovarianceFit | undefined;
  for (const correlationLength of correlationLengths) {
    let modelSquares = 0;
    let modelProducts = 0;
    let observedSquares = 0;
    let pairs = 0;
    for (let first = 0; first < residuals.length; first++) {
      for (let second = first + 1; second < residuals.length; second++) {
        let squaredDistance = 0;
        for (let axis = 0; axis < 3; axis++) {
          squaredDistance += (positions[first * 3 + axis] - positions[second * 3 + axis]) ** 2;
        }
        const model = -Math.expm1((-0.5 * squaredDistance) / correlationLength ** 2);
        const observed = 0.5 * (residuals[first] - residuals[second]) ** 2;
        modelSquares += model * model;
        modelProducts += model * observed;
        observedSquares += observed * observed;
        pairs++;
      }
    }
    validateRequirement(
      identifier,
      modelSquares > 0 &&
        Number.isFinite(modelSquares) &&
        Number.isFinite(modelProducts) &&
        Number.isFinite(observedSquares),
      'positions and residuals must yield finite semivariances with nonzero spatial separation at every correlationLength'
    );
    const variance = Math.max(modelProducts / modelSquares, 0);
    // Evaluate residuals directly: differencing the fitted and observed moments cancels the
    // small errors of long-range models and can turn distinct fits into false zero-error ties.
    let squaredError = 0;
    for (let first = 0; first < residuals.length; first++) {
      for (let second = first + 1; second < residuals.length; second++) {
        let squaredDistance = 0;
        for (let axis = 0; axis < 3; axis++) {
          squaredDistance += (positions[first * 3 + axis] - positions[second * 3 + axis]) ** 2;
        }
        const model = -Math.expm1((-0.5 * squaredDistance) / correlationLength ** 2);
        const observed = 0.5 * (residuals[first] - residuals[second]) ** 2;
        squaredError += (observed - variance * model) ** 2;
      }
    }
    // One nonzero design value and one observation have an exact fit for every length.
    // Multiplication roundoff cannot identify a length in this case; retain the analytic tie
    // without applying an error tolerance to the meaningful residuals of multi-pair fits.
    const meanSquaredError = pairs === 1 ? 0 : squaredError / pairs;
    validateRequirement(
      identifier,
      Number.isFinite(variance) && Number.isFinite(meanSquaredError),
      'residuals must yield finite fitted variance and meanSquaredError'
    );
    if (
      !bestFit ||
      meanSquaredError < bestFit.meanSquaredError ||
      (meanSquaredError === bestFit.meanSquaredError &&
        correlationLength < bestFit.correlationLength)
    ) {
      bestFit = {variance, correlationLength, meanSquaredError};
    }
  }
  return bestFit!;
}
