// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPULocalKrigingParameterValues,
  GPU_LOCAL_KRIGING_PARAMETER_LENGTH,
  validatePredictionCovariance,
  type GPUPredictionCovariance
} from './prediction-covariance';

/** Number of float32 elements in a Vecchia parameter view (the local kriging layout). */
export const GPU_VECCHIA_PARAMETER_LENGTH = GPU_LOCAL_KRIGING_PARAMETER_LENGTH;

/**
 * Smallest ratio of nugget to partial sill accepted for Matern 3/2 and smoother covariances. The
 * float32 experiment and its GPU repeat (dev-docs/research/spatiotemporal-review-2026-11-09) find
 * the log-likelihood error below 0.5 units at 10,000 sites at this ratio, and float32 factorizations
 * failing at thousands of sites below it for smooth covariances.
 */
export const VECCHIA_NUGGET_FLOOR_FRACTION = 1e-3;

/** Rounding slack of the floor comparison, so a nugget of exactly 1e-3 of the sill is accepted. */
const NUGGET_FLOOR_ROUNDING_SLACK = 1e-12;

/** Covariance families offered by the Vecchia likelihood, per the float32 experiment. */
export type GPUVecchiaCovarianceModel = 'exponential' | 'matern-3/2';

/** Covariance of a Vecchia likelihood. `range` is the effective range of `GPUPredictionCovariance`. */
export type GPUVecchiaCovariance = {
  model: GPUVecchiaCovarianceModel;
  /** Effective range, positive. */
  range: number;
  /** Partial sill, at least 0. */
  sill: number;
  /** Process microscale variance. Matern 3/2 needs at least `VECCHIA_NUGGET_FLOOR_FRACTION * sill`. */
  nugget?: number;
};

/**
 * Smallest accepted nugget of a Vecchia covariance: 0 for the exponential, which float32 handles at
 * any nugget, and `VECCHIA_NUGGET_FLOOR_FRACTION * sill` for Matern 3/2.
 */
export function getVecchiaNuggetFloor(
  covariance: Pick<GPUVecchiaCovariance, 'model' | 'sill'>
): number {
  return covariance.model === 'exponential' ? 0 : VECCHIA_NUGGET_FLOOR_FRACTION * covariance.sill;
}

/**
 * Packs per-frame covariance parameters of `GPUVecchiaLogLikelihood`, so a parameter sweep writes
 * one buffer instead of recompiling. Layout is that of `getGPULocalKrigingParameterValues`.
 *
 * @param covariance Exponential or Matern 3/2 covariance.
 * @param target Optional destination of at least `GPU_VECCHIA_PARAMETER_LENGTH` elements.
 * @throws If the model is not offered, or the nugget is below `getVecchiaNuggetFloor`.
 */
export function getGPUVecchiaLogLikelihoodParameterValues(
  covariance: GPUVecchiaCovariance,
  target?: Float32Array
): Float32Array {
  validatePredictionCovariance(covariance as GPUPredictionCovariance);
  validateRequirement(
    'GPUVecchiaLogLikelihood',
    covariance.model === 'exponential' || covariance.model === 'matern-3/2',
    'model must be exponential or matern-3/2'
  );
  validateRequirement(
    'GPUVecchiaLogLikelihood',
    (covariance.nugget ?? 0) >=
      getVecchiaNuggetFloor(covariance) * (1 - NUGGET_FLOOR_ROUNDING_SLACK),
    `nugget must be at least ${VECCHIA_NUGGET_FLOOR_FRACTION} of the sill for ${covariance.model}`
  );
  return getGPULocalKrigingParameterValues({covariance}, target);
}
