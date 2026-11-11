// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser side of run-harness.html (development only). Mounts one run by id, `?id=<entry id>`, on a
// fixed 960 x 540 canvas and exposes `window.runHarness` for scripts/verify-runs.mjs: set parameter
// values, toggle the inputs-only render, settle, and read GPU errors.

import type {Device} from '@luma.gl/core';
import {getDemo} from '../demos';
import {createGpuDevice} from '../device';
import type {GpuSceneInstance} from '../types';
import type {RunDebug} from './define-run';
import {getRunParameters, getRunScene, listRunIds} from './registry';
import {CARD_DISPLAY_WIDTH} from './text/display';

type HarnessState = {
  status: 'loading' | 'ready' | 'error';
  backend: string;
  error: string | null;
  gpuErrors: string[];
  consoleErrors: string[];
  parameters: {
    id: string;
    label: string;
    min: number;
    max: number;
    step: number;
    default: number;
  }[];
  layerKinds: string[];
  runCount: number;
  rebuildCount: number;
  /** The run's pointer gesture, or null when it only has the generic slider scrub. */
  gesture: {kind: string; hint: string} | null;
  readout: string | null;
  restValues: Record<string, number>;
  pointerRebuildCount: number;
  /** Ids of the parameters the pointer gesture has written. */
  pointerParameterIds: string[];
};

type RunHarness = {
  state: HarnessState;
  setParameters(values: Record<string, number>): Promise<void>;
  setInputsOnly(isInputsOnly: boolean): Promise<void>;
  /** Moves the pointer to a position across the canvas (0 to 1, y down) and settles. */
  setPointer(position: [number, number]): Promise<void>;
  /** Returns the gesture to its rest state and settles. */
  resetPointer(): Promise<void>;
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
  rebuildCount: 0,
  gesture: null,
  readout: null,
  restValues: {},
  pointerRebuildCount: 0,
  pointerParameterIds: []
};

// biome-ignore lint/suspicious/noConsole: the harness wraps console.error to report page errors
const originalError = console.error;
console.error = (...messages: unknown[]) => {
  state.consoleErrors.push(messages.map(String).join(' ').slice(0, 400));
  originalError(...messages);
};
window.addEventListener('error', event => state.consoleErrors.push(String(event.message)));
window.addEventListener('unhandledrejection', event =>
  state.consoleErrors.push(String(event.reason))
);

let instance: (GpuSceneInstance & {debug?: RunDebug}) | null = null;
let reportedReadout: string | null = null;

/** Scenes without a debug handle (the flagship demos) settle by drawing a few frames. */
async function settleInstance(): Promise<void> {
  if (instance?.debug) {
    await instance.debug.settle();
    return;
  }
  for (let frame = 0; frame < 30; frame++) {
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
}

async function mount(): Promise<void> {
  const query = new URLSearchParams(location.search);
  const id = query.get('id') ?? '';
  const theme = query.get('theme') === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  // The verifier exercises run-specific debug controls. Posters follow the public page's
  // scene priority, including the flagship demos which have no RunDebug interface.
  const poster = query.has('poster');
  // `?demo` loads the flagship demo live, for scripts/verify-demos.mjs.
  const demo = poster || query.has('demo') ? getDemo(id) : undefined;
  const loader = demo?.loader ?? getRunScene(id);
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
  const parameters = demo?.parameters ?? getRunParameters(id) ?? [];
  state.parameters = parameters.map(({id: parameterId, label, min, max, step, default: value}) => ({
    id: parameterId,
    label,
    min,
    max,
    step,
    default: value
  }));
  const values: Record<string, number> = {};
  for (const parameter of parameters) values[parameter.id] = parameter.default;
  const created = await module.default({
    device,
    canvas,
    width: canvas.width,
    height: canvas.height,
    // Posters are shown on catalog cards: size their text for the card, as the live card does.
    displayWidth: poster ? CARD_DISPLAY_WIDTH : undefined,
    theme,
    reducedMotion: poster,
    seed: Number(query.get('seed') ?? 1),
    parameters: values,
    reportReadout: text => {
      reportedReadout = text;
    }
  });
  instance = created as GpuSceneInstance & {debug?: RunDebug};
  if (poster) {
    // Use the scene's own reduced-motion frame at a fixed timestamp. Waiting for a
    // wall-clock animation frame makes photographs vary with shader compilation time.
    instance.render(0);
  } else {
    const loop = (time: number) => {
      instance?.render(time);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
  await instance.debug?.settle();
  refresh();
  state.status = 'ready';
}

function refresh(): void {
  state.gesture = instance?.gesture ?? null;
  state.readout = reportedReadout;
  if (!instance?.debug) return;
  state.layerKinds = instance.debug.getLayerKinds();
  state.runCount = instance.debug.getRunCount();
  state.rebuildCount = instance.debug.getRebuildCount();
  state.readout = instance.debug.getReadout();
  state.restValues = instance.debug.getRestValues();
  state.pointerRebuildCount = instance.debug.getPointerRebuildCount();
  state.pointerParameterIds = instance.debug.getPointerParameterIds();
}

window.runHarness = {
  state,
  async setParameters(values) {
    instance?.setParameters?.(values);
    await settleInstance();
    refresh();
  },
  async setInputsOnly(isInputsOnly) {
    instance?.debug?.setInputsOnly(isInputsOnly);
    await instance?.debug?.settle();
    refresh();
  },
  async setPointer(position) {
    const canvas = document.querySelector('canvas') as HTMLCanvasElement;
    instance?.setPointer?.([position[0] * canvas.width, position[1] * canvas.height]);
    await settleInstance();
    refresh();
  },
  async resetPointer() {
    instance?.resetPointer?.();
    await settleInstance();
    refresh();
  },
  async settle() {
    await settleInstance();
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
