// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getRasterGridCorners} from './spatial-interpolation-parameters';
import {
  getGPULocalKrigingParameterValues,
  type GPUPredictionCovariance
} from './prediction-covariance';

/** Float32 parameter layout length, including the eight-value local-kriging prefix. */
export const GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH = 16;

/** Per-frame settings of {@link GPUGaussianRandomField}. */
export type GPUGaussianRandomFieldSettings = {
  /** Isotropic variogram, with the effective-range convention of fitVariogramModel. */
  variogram: GPUPredictionCovariance;
  /** Full uint32 Philox seed. Packed as bits, never rounded to float32. */
  seed: number;
  /** Full uint32 realization counter; defaults to zero. */
  realization?: number;
  /** Constant field mean; defaults to zero. */
  mean?: number;
  /** Axis-aligned planar grid. Must match the contributor dimensions; omit for points. */
  grid?: GPURasterMetadata;
  /** Point-mode phase origin, subtracted before evaluating modes; defaults to [0, 0]. */
  origin?: readonly [number, number];
};

/**
 * Packs parameters for GPUGaussianRandomField.
 * Layout: local-kriging prefix [variance,0,0,0,model,nugget,sill,range], then
 * [mean, seed bits, realization bits, originX, originY, stepX, stepY, 0].
 * Grid origins are cell centers. Point coordinates should be supplied in a centered planar frame.
 */
export function getGPUGaussianRandomFieldParameterValues(
  settings: GPUGaussianRandomFieldSettings,
  target: Float32Array = new Float32Array(GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH)
): Float32Array {
  const id = 'GPUGaussianRandomField';
  validateRequirement(
    id,
    target.length >= GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH,
    'target length must be at least 16'
  );
  validateCapacity(id, 'seed', settings.seed, 0, 0xffffffff);
  validateCapacity(id, 'realization', settings.realization ?? 0, 0, 0xffffffff);
  validateFiniteNumber(id, 'mean', settings.mean ?? 0);
  validateRequirement(
    id,
    ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2', 'matern'].includes(
      settings.variogram.model
    ),
    'variogram.model must be spherical, exponential, gaussian, matern-3/2, matern-5/2 or matern'
  );
  const destination = target;
  target = new Float32Array(GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH);
  getGPULocalKrigingParameterValues({covariance: settings.variogram}, target.subarray(0, 8));
  target[8] = settings.mean ?? 0;
  const words = new Uint32Array(target.buffer, target.byteOffset, target.length);
  words[9] = settings.seed;
  words[10] = settings.realization ?? 0;
  if (settings.grid) {
    const [startX, startY] = getRasterGridCorners(id, settings.grid, false);
    const stepX = settings.grid.affine[0];
    const stepY = settings.grid.affine[4];
    target.set([startX + stepX / 2, startY + stepY / 2, stepX, stepY], 11);
  } else {
    target.set([...(settings.origin ?? [0, 0]), 1, 1], 11);
  }
  validateRequirement(
    id,
    [0, 4, 5, 6, 7, 8, 11, 12, 13, 14].every(index => Number.isFinite(target[index]!)) &&
      target[0]! > 0 &&
      target[7]! > 0 &&
      target[13]! !== 0 &&
      target[14]! !== 0,
    'settings must remain finite with positive variance/range and nonzero cell steps in float32'
  );
  destination.set(target);
  return destination;
}
