// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Texture} from '@luma.gl/core';
import type {RasterBandGpuResource} from './raster-band-model';
import type {RasterIdentifyResult} from './raster-identify';

export type RasterGpuSample = Readonly<{
  value: number;
  isValid: boolean;
}>;

/** Reads one pixel of the resident r32float values texture and its r8unorm validity texture. */
export async function readRasterBandSample(
  resource: Pick<RasterBandGpuResource, 'valueTexture' | 'validityTexture'>,
  column: number,
  row: number
): Promise<RasterGpuSample | undefined> {
  const {valueTexture, validityTexture} = resource;
  if (!valueTexture || !validityTexture) return undefined;
  const [valueBuffer, validityBuffer] = [
    readTexel(valueTexture, column, row),
    readTexel(validityTexture, column, row)
  ];
  try {
    const [valueBytes, validityBytes] = await Promise.all([
      valueBuffer.readAsync(),
      validityBuffer.readAsync()
    ]);
    const value = new DataView(valueBytes.buffer, valueBytes.byteOffset, 4).getFloat32(0, true);
    return {value, isValid: validityBytes[0] > 127};
  } finally {
    valueBuffer.destroy();
    validityBuffer.destroy();
  }
}

function readTexel(texture: Texture, column: number, row: number) {
  return texture.readBuffer({x: column, y: row, width: 1, height: 1});
}

export type RasterGpuIdentifyResult = Readonly<{
  cpu: RasterIdentifyResult;
  /** Present when the winning tile's band 1 is resident and was read back from the GPU. */
  gpu?: RasterGpuSample;
  /** True when the GPU sample agrees with the CPU sample after float32 rounding. */
  isConsistent?: boolean;
}>;

/**
 * Augments a CPU identify result with a readback of the resident texture pixel. The CPU result
 * stays authoritative (the texture may be evicted or hold float32-rounded values); the GPU sample
 * is reported alongside and compared.
 */
export async function augmentIdentifyWithGpuSample(
  cpu: RasterIdentifyResult,
  getResource: (tileId: string, bandIndex: number) => RasterBandGpuResource | undefined
): Promise<RasterGpuIdentifyResult> {
  if (
    cpu.tileId === undefined ||
    cpu.column === undefined ||
    cpu.row === undefined ||
    (cpu.status !== 'value' && cpu.status !== 'nodata')
  )
    return {cpu};
  const resource = getResource(cpu.tileId, 1);
  const gpu = resource ? await readRasterBandSample(resource, cpu.column, cpu.row) : undefined;
  if (!gpu) return {cpu};
  const cpuValue = cpu.bandValues?.[0];
  const isConsistent =
    cpu.status === 'nodata'
      ? !gpu.isValid
      : gpu.isValid &&
        typeof cpuValue === 'number' &&
        Math.abs(gpu.value - cpuValue) <= Math.abs(cpuValue) * 1e-6 + 1e-6;
  return {cpu, gpu, isConsistent};
}

/**
 * Pixel cache for hover: reads are asynchronous, so the first hover over a pixel starts a
 * readback and a later hover (the tooltip re-renders on move) shows the cached sample.
 */
export class RasterGpuSampleCache {
  private readonly samples = new Map<string, RasterGpuSample>();
  private readonly pending = new Set<string>();

  constructor(
    private readonly getResource: (
      tileId: string,
      bandIndex: number
    ) => RasterBandGpuResource | undefined,
    private readonly maximumEntries = 256
  ) {}

  getSample(tileId: string, column: number, row: number): RasterGpuSample | undefined {
    return this.samples.get(`${tileId}:${column}:${row}`);
  }

  /** Starts a readback when the pixel is neither cached nor in flight. */
  request(tileId: string, column: number, row: number, onReady?: () => void): void {
    const key = `${tileId}:${column}:${row}`;
    if (this.samples.has(key) || this.pending.has(key)) return;
    const resource = this.getResource(tileId, 1);
    if (!resource) return;
    this.pending.add(key);
    void readRasterBandSample(resource, column, row)
      .then(sample => {
        if (!sample) return;
        if (this.samples.size >= this.maximumEntries)
          this.samples.delete(this.samples.keys().next().value as string);
        this.samples.set(key, sample);
        onReady?.();
      })
      .catch(() => {
        // The texture was evicted or the device was lost between request and copy.
      })
      .finally(() => this.pending.delete(key));
  }
}
