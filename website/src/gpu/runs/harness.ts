// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser side of run-harness.html (development only). Mounts one run by id, `?id=<entry id>`, on a
// fixed 960 x 540 canvas and exposes `window.runHarness` for scripts/verify-runs.mjs: set parameter
// values, toggle the inputs-only render, settle, and read GPU errors.

import type {Device} from '@luma.gl/core';
import {createGpuDevice} from '../device';
import type {GpuSceneInstance} from '../types';
import type {RunDebug} from './define-run';
import {getRunParameters, getRunScene, listRunIds} from './registry';

type HarnessState = {
  status: 'loading' | 'ready' | 'error';
  backend: string;
  error: string | null;
  gpuErrors: string[];
  consoleErrors: string[];
  parameters: {id: string; label: string; min: number; max: number; default: number}[];
  layerKinds: string[];
  runCount: number;
  rebuildCount: number;
};

type RunHarness = {
  state: HarnessState;
  setParameters(values: Record<string, number>): Promise<void>;
  setInputsOnly(isInputsOnly: boolean): Promise<void>;
  settle(): Promise<void>;
  listRunIds(): string[];
};

declare global {
  interface Window {
    runHarness: RunHarness;
  }
}

const state: HarnessState = {
  status: 'loading',
  backend: '',
  error: null,
  gpuErrors: [],
  consoleErrors: [],
  parameters: [],
  layerKinds: [],
  runCount: 0,
  rebuildCount: 0
};

// biome-ignore lint/suspicious/noConsole: the harness wraps console.error to report page errors
const originalError = console.error;
console.error = (...messages: unknown[]) => {
  state.consoleErrors.push(messages.map(String).join(' ').slice(0, 400));
  originalError(...messages);
};
window.addEventListener('error', event => state.consoleErrors.push(String(event.message)));

let instance: (GpuSceneInstance & {debug: RunDebug}) | null = null;

async function mount(): Promise<void> {
  const query = new URLSearchParams(location.search);
  const id = query.get('id') ?? '';
  const theme = query.get('theme') === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  const loader = getRunScene(id);
  if (!loader) throw new Error(`No run for entry id "${id}"`);
  const device: Device | null = await createGpuDevice(canvas);
  if (!device) throw new Error('No GPU device');
  state.backend = device.type;
  const handle = (device as unknown as {handle?: GPUDevice}).handle;
  handle?.addEventListener?.('uncapturederror', (event: Event) => {
    state.gpuErrors.push((event as GPUUncapturedErrorEvent).error.message.slice(0, 600));
  });
  const canvasContext = device.getDefaultCanvasContext();
  await canvasContext.initialized;
  canvasContext.setDrawingBufferSize(canvas.width, canvas.height);
  const module = await loader();
  const parameters = getRunParameters(id) ?? [];
  state.parameters = parameters.map(({id: parameterId, label, min, max, default: value}) => ({
    id: parameterId,
    label,
    min,
    max,
    default: value
  }));
  const values: Record<string, number> = {};
  for (const parameter of parameters) values[parameter.id] = parameter.default;
  const created = await module.default({
    device,
    canvas,
    width: canvas.width,
    height: canvas.height,
    theme,
    reducedMotion: false,
    seed: Number(query.get('seed') ?? 1),
    parameters: values
  });
  instance = created as GpuSceneInstance & {debug: RunDebug};
  let frame = 0;
  const loop = (time: number) => {
    instance?.render(time);
    frame = requestAnimationFrame(loop);
  };
  frame = requestAnimationFrame(loop);
  void frame;
  await instance.debug.settle();
  refresh();
  state.status = 'ready';
}

function refresh(): void {
  if (!instance) return;
  state.layerKinds = instance.debug.getLayerKinds();
  state.runCount = instance.debug.getRunCount();
  state.rebuildCount = instance.debug.getRebuildCount();
}

window.runHarness = {
  state,
  async setParameters(values) {
    instance?.setParameters?.(values);
    await instance?.debug.settle();
    refresh();
  },
  async setInputsOnly(isInputsOnly) {
    instance?.debug.setInputsOnly(isInputsOnly);
    await instance?.debug.settle();
    refresh();
  },
  async settle() {
    await instance?.debug.settle();
    refresh();
  },
  listRunIds
};

if (!new URLSearchParams(location.search).get('id')) {
  state.status = 'error';
  state.error = 'Missing ?id=';
} else {
  mount().catch(error => {
    state.status = 'error';
    state.error = String(error?.stack ?? error).slice(0, 800);
  });
}
