// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser-only luma.gl device creation. luma.gl is imported dynamically so it never reaches the
// prerendered HTML or the initial bundle.

import type {Device} from '@luma.gl/core';

export type GpuBackend = 'webgpu' | 'webgl';

// Kept alive for the page lifetime: dropping the probe adapter can invalidate later devices.
let probeAdapter: unknown = null;

async function hasWebGpuAdapter(): Promise<boolean> {
  try {
    probeAdapter ??= await navigator.gpu?.requestAdapter();
    return Boolean(probeAdapter);
  } catch {
    return false;
  }
}

/**
 * Creates a device that renders to `canvas`, preferring WebGPU and falling back to WebGL2.
 * Resolves null on the server and when neither backend is available.
 */
export async function createGpuDevice(canvas: HTMLCanvasElement): Promise<Device | null> {
  if (typeof window === 'undefined') return null;
  const [{luma}, {webgpuAdapter}, {webgl2Adapter}] = await Promise.all([
    import('@luma.gl/core'),
    import('@luma.gl/core/webgpu'),
    import('@luma.gl/core/webgl')
  ]);
  const createCanvasContext = {canvas, autoResize: true, useDevicePixels: true};
  if (await hasWebGpuAdapter()) {
    try {
      return await luma.createDevice({
        type: 'webgpu',
        adapters: [webgpuAdapter],
        createCanvasContext
      });
    } catch (error) {
      // biome-ignore lint/suspicious/noConsole: diagnostic output
      console.warn('WebGPU device creation failed, trying WebGL2', error);
    }
  }
  try {
    return await luma.createDevice({type: 'webgl', adapters: [webgl2Adapter], createCanvasContext});
  } catch {
    return null;
  }
}

const MAXIMUM_LIVE_DEVICES = 12;
let liveDeviceCount = 0;
const waitingGrants: (() => void)[] = [];

/**
 * Browsers cap simultaneous GPU contexts, so only a few canvases are live at once. Resolves with a
 * release function when a slot is free; later canvases wait their turn and show their fallback.
 */
export function acquireGpuSlot(): Promise<() => void> {
  return new Promise(resolve => {
    const grant = () => {
      liveDeviceCount++;
      let isReleased = false;
      resolve(() => {
        if (isReleased) return;
        isReleased = true;
        liveDeviceCount--;
        waitingGrants.shift()?.();
      });
    };
    if (liveDeviceCount < MAXIMUM_LIVE_DEVICES) grant();
    else waitingGrants.push(grant);
  });
}
