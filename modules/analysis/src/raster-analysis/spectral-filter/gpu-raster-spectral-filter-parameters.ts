// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateParameterTarget} from '../raster-band-utils';

/** Eight float32 values, including one reserved slot for alignment. */
export const GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH = 8;
/** Largest radial frequency in a sampled 2D grid: the Nyquist corner, in cycles/pixel. */
export const MAXIMUM_RASTER_SPECTRAL_FREQUENCY = Math.SQRT1_2;
/** Avoid an underflowing squared Gaussian denominator in float32. */
export const MINIMUM_RASTER_SPECTRAL_SIGMA = Math.fround(1e-6);

export type GPURasterSpectralFilterSettings = {
  /** Low-pass includes this radius; high-pass excludes it. Default 0.25 cycles/pixel. */
  cutoff?: number;
  lowerCutoff?: number;
  upperCutoff?: number;
  /** Gaussian standard deviation in frequency space, default 0.1 cycles/pixel. */
  sigma?: number;
  /** Notch center (column,row) frequency in [-0.5,0.5]; its conjugate is also removed. */
  notchCenter?: readonly [number, number];
  notchRadius?: number;
};

/**
 * Packs [cutoff, lowerCutoff, upperCutoff, sigma, notchColumn, notchRow, notchRadius, 0].
 * Write to a GPUParameterBuffer to change the response between encodings without recompiling.
 * All distances are cycles/pixel, independent of the grid's affine and ground cell sizes.
 */
export function getGPURasterSpectralFilterParameterValues(
  settings: GPURasterSpectralFilterSettings = {},
  target = new Float32Array(GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterSpectralFilterParameterValues';
  validateParameterTarget(id, target, GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH);
  const cutoff = settings.cutoff ?? 0.25;
  const lowerCutoff = settings.lowerCutoff ?? 0;
  const upperCutoff = settings.upperCutoff ?? 0.25;
  const sigma = settings.sigma ?? 0.1;
  const [notchColumn, notchRow] = settings.notchCenter ?? [0.25, 0];
  const notchRadius = settings.notchRadius ?? 0.05;
  for (const [name, value] of Object.entries({cutoff, lowerCutoff, upperCutoff, notchRadius})) {
    validateFiniteNumber(id, name, value, 0, MAXIMUM_RASTER_SPECTRAL_FREQUENCY);
  }
  validateRequirement(id, lowerCutoff <= upperCutoff, 'lowerCutoff must be <= upperCutoff');
  validateFiniteNumber(
    id,
    'sigma',
    sigma,
    MINIMUM_RASTER_SPECTRAL_SIGMA,
    MAXIMUM_RASTER_SPECTRAL_FREQUENCY
  );
  validateFiniteNumber(id, 'notchCenter[0]', notchColumn, -0.5, 0.5);
  validateFiniteNumber(id, 'notchCenter[1]', notchRow, -0.5, 0.5);
  target.set([cutoff, lowerCutoff, upperCutoff, sigma, notchColumn, notchRow, notchRadius, 0]);
  return target;
}
