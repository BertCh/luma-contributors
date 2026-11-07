// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUElementwise} from '@luma.gl/graph/gpu-core';
import {CatalogGpuSession, CatalogGpuUnsupportedError} from './catalog-gpu-session';
import {
  addTerrainGraph,
  getTerrainOutputScale,
  type TerrainGpuInput
} from './analysis3-terrain-gpu';

/** GPU-resident terrain tile: nothing was read back. The caller owns and destroys both buffers. */
export type TerrainResidentTile = Readonly<{
  /** Output tile width and height (the halo is cropped when `cropHalo` is set). */
  width: number;
  height: number;
  /** Row-major float32 values in output units (hillshade 0..255, curvature ESRI sign). */
  values: Buffer;
  /** uint32 words, 1 where the cell holds a valid value. */
  validity: Buffer;
  contributors: readonly string[];
}>;

const RESIDENT_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;

/**
 * Runs the terrain contributors for one tile and keeps the result on the GPU. The output unit
 * scale is applied by GPUElementwise and, with `cropHalo`, the one-sample neighbor halo of the
 * input is cropped by GPU buffer copies, so the tile never visits the CPU. Cells that the
 * contributor marks invalid are invalid in `validity`; unlike the CPU path, a finite check of valid
 * cells is not applied because it would need the values on the CPU.
 */
export async function computeTerrainGridResident(
  device: Device,
  input: TerrainGpuInput,
  options: {cropHalo?: boolean} = {}
): Promise<TerrainResidentTile> {
  const {width: inputWidth, height: inputHeight} = input;
  const cropHalo = options.cropHalo ?? false;
  const width = cropHalo ? inputWidth - 2 : inputWidth;
  const height = cropHalo ? inputHeight - 2 : inputHeight;
  const pixelCount = inputWidth * inputHeight;
  // Unsupported tile size: a storage binding cannot hold the elevation band.
  if (pixelCount * 4 > device.limits.maxStorageBufferBindingSize)
    throw new CatalogGpuUnsupportedError('Terrain tile exceeds the storage binding limit');
  const session = new CatalogGpuSession(device, `terrain-resident-${input.operation}`);
  const finalBuffers: Buffer[] = [];
  try {
    const {values, outputValidity, contributor} = addTerrainGraph(session, input);
    const scale = getTerrainOutputScale(input.operation);
    let scaled = values;
    if (scale !== 1) {
      const scaleView = session.input('scale', new Float32Array(pixelCount).fill(scale), 'float32');
      scaled = session.output('scaled-values', 'float32', pixelCount);
      session.graph.add(
        new GPUElementwise({
          id: `terrain-scale-${input.operation}`,
          input: values.view,
          inputB: scaleView,
          output: scaled.view,
          operation: 'multiply'
        })
      );
    }
    session.run();
    const finalValues = device.createBuffer({
      id: `terrain-resident-values-${input.operation}`,
      byteLength: Math.max(4, width * height * 4),
      usage: RESIDENT_USAGE
    });
    finalBuffers.push(finalValues);
    const finalValidity = device.createBuffer({
      id: `terrain-resident-validity-${input.operation}`,
      byteLength: Math.max(4, width * height * 4),
      usage: RESIDENT_USAGE
    });
    finalBuffers.push(finalValidity);
    const encoder = device.createCommandEncoder({id: 'terrain-resident-crop'});
    if (!cropHalo) {
      encoder.copyBufferToBuffer({
        sourceBuffer: scaled.buffer,
        destinationBuffer: finalValues,
        size: pixelCount * 4
      });
      encoder.copyBufferToBuffer({
        sourceBuffer: outputValidity.buffer,
        destinationBuffer: finalValidity,
        size: pixelCount * 4
      });
    } else
      for (let row = 0; row < height; row++) {
        const sourceOffset = ((row + 1) * inputWidth + 1) * 4;
        const destinationOffset = row * width * 4;
        encoder.copyBufferToBuffer({
          sourceBuffer: scaled.buffer,
          sourceOffset,
          destinationBuffer: finalValues,
          destinationOffset,
          size: width * 4
        });
        encoder.copyBufferToBuffer({
          sourceBuffer: outputValidity.buffer,
          sourceOffset,
          destinationBuffer: finalValidity,
          destinationOffset,
          size: width * 4
        });
      }
    device.submit(encoder.finish());
    return {
      width,
      height,
      values: finalValues,
      validity: finalValidity,
      contributors: [contributor]
    };
  } catch (failure) {
    for (const buffer of finalBuffers) buffer.destroy();
    throw failure;
  } finally {
    // Submitted work keeps its resources alive until it completes; destroy only drops the handles.
    session.destroy();
  }
}
