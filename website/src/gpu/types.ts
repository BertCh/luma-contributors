// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Contract for live luma.gl visuals. Pages render <GpuCanvas> (src/gpu/gpu-canvas.tsx). During
// prerendering and on browsers without a GPU, its `fallback` is shown; in the browser the scene is
// lazily imported and drawn with luma.gl, and paused while offscreen.

import type {Device} from '@luma.gl/core';

export type GpuTheme = 'light' | 'dark';

export type GpuSceneContext = {
  device: Device;
  canvas: HTMLCanvasElement;
  /** Size in device pixels. */
  width: number;
  height: number;
  theme: GpuTheme;
  /** True when the user prefers reduced motion: render one still frame. */
  reducedMotion: boolean;
  /** Deterministic seed shared with the static fallback. */
  seed: number;
};

export type GpuSceneInstance = {
  /** Called every animation frame while visible. */
  render(timeMilliseconds: number): void;
  resize?(width: number, height: number): void;
  /** Pointer position in canvas pixels, or null when the pointer leaves. */
  setPointer?(position: [number, number] | null): void;
  destroy(): void;
};

export type GpuScene = (context: GpuSceneContext) => GpuSceneInstance | Promise<GpuSceneInstance>;

/** A lazily imported module whose default export is a scene. Keeps luma.gl out of the HTML. */
export type GpuSceneLoader = () => Promise<{default: GpuScene}>;
