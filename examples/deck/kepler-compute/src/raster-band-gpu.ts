// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import type {RasterBandUploader} from './raster-band-model';

/**
 * Uploads each band as an r32float value texture and an r8unorm validity texture.
 * Float64, Int32 and Uint32 samples are converted to float32; `hasPrecisionLoss` reports when
 * that conversion changes any valid sample. Not yet exercised on a device (see backlog).
 */
export function createRasterBandGpuUploader(device: Device): RasterBandUploader {
  return (tile, band) => {
    const floatValues =
      band.values instanceof Float32Array ? band.values : Float32Array.from(band.values);
    let hasPrecisionLoss = false;
    if (!(band.values instanceof Float32Array))
      for (let sampleIndex = 0; sampleIndex < floatValues.length; sampleIndex++)
        if (band.validity[sampleIndex] && floatValues[sampleIndex] !== band.values[sampleIndex]) {
          hasPrecisionLoss = true;
          break;
        }
    const usage = Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST;
    const valueTexture = device.createTexture({
      id: `raster-${tile.id}-b${band.index}-values`,
      format: 'r32float',
      width: tile.width,
      height: tile.height,
      usage,
      data: floatValues,
      sampler: {minFilter: 'nearest', magFilter: 'nearest'}
    });
    const validityTexture = device.createTexture({
      id: `raster-${tile.id}-b${band.index}-validity`,
      format: 'r8unorm',
      width: tile.width,
      height: tile.height,
      usage,
      data: band.validity.map(value => (value ? 255 : 0)),
      sampler: {minFilter: 'nearest', magFilter: 'nearest'}
    });
    return {
      byteLength: tile.width * tile.height * 5,
      hasPrecisionLoss,
      valueTexture,
      validityTexture,
      destroy: () => {
        valueTexture.destroy();
        validityTexture.destroy();
      }
    };
  };
}
