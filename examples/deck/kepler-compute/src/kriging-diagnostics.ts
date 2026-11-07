// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  estimateKeplerEmpiricalVariogram,
  fitKeplerVariogram,
  type KeplerEmpiricalVariogramOptions,
  type KeplerKrigingSettings,
  type KeplerVariogramFit
} from '@luma-contributors/kepler';

/** Reads a scalar Float32 vector in source order, respecting chunk offsets and row stride. */
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
      result[rowOffset + row] = view.getFloat32(row * chunk.byteStride, true);
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
 * that replay the fit exactly (explicit parameters plus provenance).
 */
export async function suggestKrigingVariogram(
  vectors: {
    longitude: GPUVector<'float32'>;
    latitude: GPUVector<'float32'>;
    values: GPUVector<'float32'>;
  },
  settings: KeplerKrigingSettings,
  options: KeplerEmpiricalVariogramOptions = {}
): Promise<KrigingVariogramSuggestion> {
  const [longitudes, latitudes, values] = await Promise.all([
    readFloat32Vector(vectors.longitude),
    readFloat32Vector(vectors.latitude),
    readFloat32Vector(vectors.values)
  ]);
  const empirical = estimateKeplerEmpiricalVariogram(longitudes, latitudes, values, {
    binCount: 20,
    ...options
  });
  const fit = fitKeplerVariogram(empirical);
  return {
    fit,
    validRowCount: empirical.validRowCount,
    accumulatedPairCount: empirical.accumulatedPairCount,
    exhaustive: empirical.exhaustive,
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
        pairCount: empirical.accumulatedPairCount,
        maximumLagMeters: empirical.maximumLagMeters,
        criterion: fit.criterion
      }
    }
  };
}
