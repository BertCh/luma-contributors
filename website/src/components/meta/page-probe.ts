// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Reads real numbers from a live canvas for the this-page topic. `probeScene` wraps any scene loader:
// it notes when the device and the scene are created and measures every frame (interval, CPU time,
// draw calls from the native pass encoder, GPU memory from luma.gl's stats). The store works with
// useSyncExternalStore; snapshots are replaced, never mutated.

import type {GpuScene, GpuSceneInstance, GpuSceneLoader} from '../../gpu/types';

export type PageProbeMilestone = 'hydrated' | 'device' | 'scene' | 'ready';

export type PageProbeSnapshot = {
  backend: 'webgpu' | 'webgl' | null;
  adapter: string | null;
  canvasWidth: number;
  canvasHeight: number;
  devicePixelRatio: number;
  frameMilliseconds: number;
  renderMilliseconds: number;
  drawCalls: number;
  renderPasses: number;
  frameCount: number;
  gpuMemoryBytes: number | null;
  bufferCount: number | null;
  /** Milliseconds since navigation start when each step happened. */
  times: Partial<Record<PageProbeMilestone, number>>;
  /** True when the server-rendered fallback was found in the DOM at hydration. */
  fallbackInHtml: boolean | null;
};

const INITIAL_SNAPSHOT: PageProbeSnapshot = {
  backend: null,
  adapter: null,
  canvasWidth: 0,
  canvasHeight: 0,
  devicePixelRatio: 1,
  frameMilliseconds: 0,
  renderMilliseconds: 0,
  drawCalls: 0,
  renderPasses: 0,
  frameCount: 0,
  gpuMemoryBytes: null,
  bufferCount: null,
  times: {},
  fallbackInHtml: null
};

export class PageProbe {
  private snapshot: PageProbeSnapshot = INITIAL_SNAPSHOT;
  private readonly listeners = new Set<() => void>();

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): PageProbeSnapshot => this.snapshot;

  getServerSnapshot = (): PageProbeSnapshot => INITIAL_SNAPSHOT;

  update(patch: Partial<PageProbeSnapshot>): void {
    this.snapshot = {...this.snapshot, ...patch};
    for (const listener of this.listeners) listener();
  }

  mark(milestone: PageProbeMilestone): void {
    if (this.snapshot.times[milestone] !== undefined) return;
    this.update({times: {...this.snapshot.times, [milestone]: performance.now()}});
  }

  /** Forgets a finished run, for example when the canvas is released offscreen. */
  reset(): void {
    this.update({
      backend: null,
      frameCount: 0,
      times: {hydrated: this.snapshot.times.hydrated},
      drawCalls: 0,
      renderPasses: 0
    });
  }
}

type FrameCounters = {draws: number; passes: number};

type DeviceLike = {
  type: string;
  info?: {renderer?: string; vendor?: string; gpu?: string};
  handle: unknown;
  beginRenderPass: (...args: never[]) => unknown;
  statsManager?: {getStats(name: string): {get(name: string): {count: number}}};
};

/** Counts native draw calls without changing what is drawn. Failures are ignored. */
function instrumentDevice(device: DeviceLike, counters: FrameCounters): void {
  try {
    const original = device.beginRenderPass.bind(device) as (...args: unknown[]) => {
      handle?: Record<string, unknown>;
    };
    device.beginRenderPass = ((...args: unknown[]) => {
      const pass = original(...args);
      counters.passes++;
      const encoder = pass.handle;
      if (encoder) {
        for (const name of ['draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']) {
          const method = encoder[name];
          if (typeof method === 'function') {
            encoder[name] = (...drawArgs: unknown[]) => {
              counters.draws++;
              return (method as (...values: unknown[]) => unknown).apply(encoder, drawArgs);
            };
          }
        }
      }
      return pass;
    }) as never;
    if (device.type === 'webgl') {
      const gl = device.handle as Record<string, unknown>;
      for (const name of [
        'drawArrays',
        'drawArraysInstanced',
        'drawElements',
        'drawElementsInstanced'
      ]) {
        const method = gl[name];
        if (typeof method === 'function') {
          gl[name] = (...drawArgs: unknown[]) => {
            counters.draws++;
            return (method as (...values: unknown[]) => unknown).apply(gl, drawArgs);
          };
        }
      }
    }
  } catch {
    // Counting is optional; the numbers stay at zero.
  }
}

function readDeviceStats(device: DeviceLike): Partial<PageProbeSnapshot> {
  try {
    const memory = device.statsManager?.getStats('GPU Time and Memory').get('GPU Memory').count;
    const buffers = device.statsManager?.getStats('Resource Counts').get('Buffers Active').count;
    return {gpuMemoryBytes: memory ?? null, bufferCount: buffers ?? null};
  } catch {
    return {};
  }
}

/** Wraps a scene loader so `probe` sees the device, the scene and every frame. */
export function probeScene(loader: GpuSceneLoader, probe: PageProbe): GpuSceneLoader {
  return async () => {
    const loaded = await loader();
    const scene: GpuScene = async context => {
      const device = context.device as unknown as DeviceLike;
      const counters: FrameCounters = {draws: 0, passes: 0};
      instrumentDevice(device, counters);
      probe.update({
        backend: device.type === 'webgpu' ? 'webgpu' : 'webgl',
        adapter: device.info?.renderer || device.info?.vendor || device.info?.gpu || null,
        canvasWidth: context.width,
        canvasHeight: context.height,
        devicePixelRatio: window.devicePixelRatio || 1
      });
      probe.mark('device');
      const instance = await loaded.default(context);
      probe.mark('scene');

      let previousTime = -1;
      let renderTotal = 0;
      let intervalTotal = 0;
      let sampleCount = 0;
      let lastPublished = performance.now();
      const wrapped: GpuSceneInstance = {
        render(time) {
          counters.draws = 0;
          counters.passes = 0;
          const started = performance.now();
          instance.render(time);
          renderTotal += performance.now() - started;
          if (previousTime >= 0 && time > previousTime) intervalTotal += time - previousTime;
          previousTime = time;
          sampleCount++;
          const now = performance.now();
          if (now - lastPublished >= 400 || probe.getSnapshot().frameCount === 0) {
            probe.update({
              frameMilliseconds: intervalTotal / Math.max(1, sampleCount - 1),
              renderMilliseconds: renderTotal / sampleCount,
              drawCalls: counters.draws,
              renderPasses: counters.passes,
              frameCount: probe.getSnapshot().frameCount + sampleCount,
              ...readDeviceStats(device)
            });
            renderTotal = 0;
            intervalTotal = 0;
            sampleCount = 0;
            lastPublished = now;
          }
        },
        resize(width, height) {
          instance.resize?.(width, height);
          probe.update({canvasWidth: width, canvasHeight: height});
        },
        setPointer: instance.setPointer?.bind(instance),
        setParameters: instance.setParameters?.bind(instance),
        destroy() {
          instance.destroy();
        }
      };
      return wrapped;
    };
    return {default: scene, parameters: loaded.parameters};
  };
}
