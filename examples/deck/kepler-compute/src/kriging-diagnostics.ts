// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  estimateKeplerEmpiricalVariogram,
  fitKeplerVariogram,
  fitKeplerVecchiaVariogram,
  type KeplerEmpiricalVariogramOptions,
  type KeplerKrigingSettings,
  type KeplerVariogramFit
} from '@luma-contributors/kepler';

/** Reads a scalar Float32 vector in source order, respecting chunk offsets, row stride and null validity. */
export async function readFloat32Vector(vector: GPUVector<'float32'>): Promise<Float32Array> {
  const result = new Float32Array(vector.length);
  let rowOffset = 0;
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
    const bytes = await buffer.readAsync(
      chunk.byteOffset,
      (chunk.length - 1) * chunk.byteStride + chunk.rowByteLength
    );
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let row = 0; row < chunk.length; row++)
      result[rowOffset + row] =
        chunk.nullBitmap && !(chunk.nullBitmap[row >>> 3] & (1 << (row & 7)))
          ? NaN
          : view.getFloat32(row * chunk.byteStride, true);
    rowOffset += chunk.length;
  }
  return result;
}

export type KrigingVariogramSuggestion = {
  fit: KeplerVariogramFit;
  /** Kriging settings with the fitted variogram and its provenance filled in. */
  settings: KeplerKrigingSettings;
  validRowCount: number;
  accumulatedPairCount: number;
  exhaustive: boolean;
};

/**
 * Fits a variogram to the declared longitude, latitude and response vectors and returns settings
 * that replay the fit exactly (explicit parameters plus provenance). Empirical weighted-least-squares
 * fitting uses clipped pair discovery; wrapped settings require a manual variogram for that route.
 */
export async function suggestKrigingVariogram(
  vectors: {
    device?: Device;
    longitude: GPUVector<'float32'>;
    latitude: GPUVector<'float32'>;
    values: GPUVector<'float32'>;
  },
  settings: KeplerKrigingSettings,
  options: KeplerEmpiricalVariogramOptions & {
    fitMethod?: 'weighted-least-squares' | 'vecchia-likelihood' | 'vecchia-fisher';
  } = {}
): Promise<KrigingVariogramSuggestion> {
  const usesVecchia = Boolean(options.fitMethod && options.fitMethod !== 'weighted-least-squares');
  validateRequirement(
    'suggestKrigingVariogram',
    usesVecchia || settings.longitudeSearch !== 'wrapped',
    'longitudeSearch wrapped is unsupported by empirical weighted-least-squares fitting; supply a manual variogram instead'
  );
  const [longitudes, latitudes, values] = await Promise.all([
    readFloat32Vector(vectors.longitude),
    readFloat32Vector(vectors.latitude),
    readFloat32Vector(vectors.values)
  ]);
  const empirical = usesVecchia
    ? undefined
    : estimateKeplerEmpiricalVariogram(longitudes, latitudes, values, {
        binCount: 20,
        ...options
      });
  const fit =
    options.fitMethod && options.fitMethod !== 'weighted-least-squares'
      ? await (() => {
          if (!vectors.device) throw new Error('Vecchia variogram fitting requires the GPU device');
          return fitKeplerVecchiaVariogram(vectors.device, longitudes, latitudes, values, {
            method: options.fitMethod === 'vecchia-fisher' ? 'fisher' : 'likelihood',
            initial: {
              model: settings.variogram.model === 'matern-3/2' ? 'matern-3/2' : 'exponential',
              nugget: settings.variogram.nugget,
              sill: settings.variogram.sill,
              range: settings.variogram.rangeMeters
            }
          });
        })()
      : fitKeplerVariogram(empirical!);
  return {
    fit,
    validRowCount: empirical?.validRowCount ?? fit.diagnostics!['valid_rows'],
    accumulatedPairCount: empirical?.accumulatedPairCount ?? fit.pairCount,
    exhaustive: empirical?.exhaustive ?? false,
    settings: {
      ...settings,
      variogram: {
        model: fit.model,
        nugget: fit.nugget,
        sill: fit.sill,
        rangeMeters: fit.rangeMeters
      },
      variogramFit: {
        method: fit.method,
        weighting: fit.weighting,
        binCount: fit.binCount,
        pairCount: fit.pairCount,
        maximumLagMeters: empirical?.maximumLagMeters ?? settings.variogram.rangeMeters,
        criterion: fit.criterion
      }
    }
  };
}
