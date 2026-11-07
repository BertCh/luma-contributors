// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reliable pixel capture from a WebGPU canvas. A WebGPU canvas texture is only valid until the
 * browser presents it, so reading it in a later task returns a blank frame. Capture therefore runs
 * synchronously in the task that submitted the frame (callers invoke it in the continuation of the
 * host's render wait). When drawImage still yields a blank canvas, the helper copies the current
 * canvas texture into a staging buffer (copyTextureToBuffer) on the same queue, which is ordered
 * after Deck's submit and so sees the finished frame. The context is reconfigured once with COPY_SRC
 * usage for that path, because luma.gl configures bgra8unorm canvases for RENDER_ATTACHMENT only.
 */

const COPY_DST_USAGE = 0x08;
const MAP_READ_USAGE = 0x0001;
const TEXTURE_COPY_SRC = 0x01;
const MAP_MODE_READ = 1;

type GpuLike = {
  device: {
    createBuffer(descriptor: {size: number; usage: number}): GpuBufferLike;
    createCommandEncoder(): {
      copyTextureToBuffer(
        source: {texture: unknown},
        destination: {buffer: unknown; bytesPerRow: number; rowsPerImage: number},
        size: {width: number; height: number}
      ): void;
      finish(): unknown;
    };
    queue: {submit(commands: unknown[]): void};
  };
  format: string;
  usage: number;
};
type GpuBufferLike = {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
};
type GpuContextLike = {
  getConfiguration?(): GpuLike | null;
  configure(configuration: Record<string, unknown>): void;
  getCurrentTexture(): {width: number; height: number};
};

export type CanvasFrameMethod = 'draw-image' | 'gpu-readback';
export type CanvasFrame = {source: CanvasImageSource; method: CanvasFrameMethod};

/** Row pitch WebGPU requires for texture-to-buffer copies (multiple of 256). */
export function getAlignedBytesPerRow(width: number, bytesPerPixel: number = 4): number {
  return Math.ceil((width * bytesPerPixel) / 256) * 256;
}

/** True when every sampled RGBA value is zero: the transparent black a stale WebGPU canvas yields. */
export function isFrameBlank(rgba: ArrayLike<number>): boolean {
  for (let offset = 0; offset < rgba.length; offset++) if (rgba[offset] !== 0) return false;
  return true;
}

/** Strip row padding and swizzle bgra8unorm to RGBA (rgba8unorm is copied as is). */
export function unpackTextureRows(
  bytes: Uint8Array,
  width: number,
  height: number,
  bytesPerRow: number,
  format: string
): Uint8ClampedArray {
  const output = new Uint8ClampedArray(width * height * 4);
  const swap = format.startsWith('bgra');
  for (let row = 0; row < height; row++) {
    let source = row * bytesPerRow;
    let target = row * width * 4;
    for (let column = 0; column < width; column++, source += 4, target += 4) {
      output[target] = bytes[source + (swap ? 2 : 0)];
      output[target + 1] = bytes[source + 1];
      output[target + 2] = bytes[source + (swap ? 0 : 2)];
      output[target + 3] = bytes[source + 3];
    }
  }
  return output;
}

function getGpuContext(canvas: HTMLCanvasElement): GpuContextLike | null {
  try {
    // Returns the existing context when luma.gl already created the WebGPU one, else null/other type.
    const context = canvas.getContext('webgpu') as unknown as GpuContextLike | null;
    return context && typeof context.getCurrentTexture === 'function' ? context : null;
  } catch {
    return null;
  }
}

/** Sample the canvas through a 32x32 scratch copy to decide whether drawImage produced pixels. */
function sampleCanvasIsBlank(source: CanvasImageSource): boolean {
  const scratch = document.createElement('canvas');
  scratch.width = 32;
  scratch.height = 32;
  const context = scratch.getContext('2d', {willReadFrequently: true})!;
  context.drawImage(source, 0, 0, 32, 32);
  return isFrameBlank(context.getImageData(0, 0, 32, 32).data);
}

/**
 * Enable the GPU readback path. Reconfiguring invalidates the current texture, so this nudges nothing
 * itself: callers render again (view-state change) before the next capture. Returns true when the
 * canvas was reconfigured, false when it already allows COPY_SRC or is not a WebGPU canvas.
 */
export function prepareCanvasReadback(canvas: HTMLCanvasElement): boolean {
  const context = getGpuContext(canvas);
  const configuration = context?.getConfiguration?.();
  if (!context || !configuration || configuration.usage & TEXTURE_COPY_SRC) return false;
  context.configure({...configuration, usage: configuration.usage | TEXTURE_COPY_SRC});
  return true;
}

/** Submit a copy of the canvas texture now (same task) and resolve when the pixels are mapped. */
export function readCanvasTextureViaGpu(canvas: HTMLCanvasElement): Promise<HTMLCanvasElement> {
  const context = getGpuContext(canvas);
  const configuration = context?.getConfiguration?.();
  // Readback needs a configured WebGPU context with COPY_SRC (call prepareCanvasReadback first).
  if (!context || !configuration || !(configuration.usage & TEXTURE_COPY_SRC))
    return Promise.reject(new Error('The WebGPU canvas does not allow pixel readback.'));
  const texture = context.getCurrentTexture();
  const {width, height} = texture;
  const bytesPerRow = getAlignedBytesPerRow(width);
  const {device} = configuration;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: COPY_DST_USAGE | MAP_READ_USAGE
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    {texture},
    {buffer, bytesPerRow, rowsPerImage: height},
    {width, height}
  );
  device.queue.submit([encoder.finish()]);
  return buffer.mapAsync(MAP_MODE_READ).then(() => {
    try {
      const pixels = unpackTextureRows(
        new Uint8Array(buffer.getMappedRange()),
        width,
        height,
        bytesPerRow,
        configuration.format
      );
      const target = document.createElement('canvas');
      target.width = width;
      target.height = height;
      target
        .getContext('2d')!
        .putImageData(new ImageData(pixels as ImageDataArray, width, height), 0, 0);
      return target;
    } finally {
      buffer.unmap();
      buffer.destroy();
    }
  });
}

export type CanvasFrameHost = {
  getViewState(): {zoom: number};
  setViewState(viewState: never): void;
  waitForRender(): Promise<void>;
};

/**
 * Frame capturer for PNG tiles and video frames. Call capture() in the continuation of the host
 * render wait so the presented texture is still current. The first frame uses drawImage; when that
 * is blank on a WebGPU canvas the capturer arms COPY_SRC readback, nudges the view so Deck renders
 * again, and uses GPU readback for the rest of the session. Results are copies that outlive the frame.
 */
export function createCanvasFrameCapturer(host: CanvasFrameHost) {
  let armed = false;
  return {
    get method(): CanvasFrameMethod {
      return armed ? 'gpu-readback' : 'draw-image';
    },
    async capture(canvas: HTMLCanvasElement): Promise<CanvasFrame> {
      if (armed) return {source: await readCanvasTextureViaGpu(canvas), method: 'gpu-readback'};
      const copy = document.createElement('canvas');
      copy.width = canvas.width;
      copy.height = canvas.height;
      copy.getContext('2d')!.drawImage(canvas, 0, 0);
      if (!sampleCanvasIsBlank(copy) || !prepareCanvasReadback(canvas))
        return {source: copy, method: 'draw-image'};
      // The reconfigure dropped the stale texture; a tiny zoom change forces a fresh presented frame.
      armed = true;
      const original = host.getViewState();
      host.setViewState({...original, zoom: original.zoom + 1e-6} as never);
      await host.waitForRender();
      try {
        return {source: await readCanvasTextureViaGpu(canvas), method: 'gpu-readback'};
      } finally {
        host.setViewState(original as never);
      }
    }
  };
}
