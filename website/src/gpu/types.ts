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
  /** Initial values of the scene's declared parameters, keyed by parameter id. */
  parameters: Record<string, number>;
  /** The scene calls this when its pointer readout line changes (null clears it). */
  reportReadout?(text: string | null): void;
  /** The scene calls this with parameter values its pointer gesture wrote, so sliders can follow. */
  reportParameters?(values: Record<string, number>): void;
};

/**
 * A user-adjustable scene parameter. <GpuCanvas controls> renders one slider per parameter and
 * calls `setParameters` as it moves, so the picture answers the reader live.
 */
export type GpuSceneParameter = {
  id: string;
  /** Short label, for example `Bandwidth`. */
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  /** Prefer this control when scrubbing a catalog card with the pointer or arrow keys. */
  cardScrub?: boolean;
  /** Optional unit shown after the value, for example `px` or `%`. */
  unit?: string;
  /**
   * Enumerated choices. The control becomes a segmented selector and the value is the index of the
   * chosen option (`min` 0, `max` options.length - 1, `step` 1).
   */
  options?: readonly string[];
};

export type GpuSceneInstance = {
  /** Called every animation frame while visible. */
  render(timeMilliseconds: number): void;
  resize?(width: number, height: number): void;
  /** Pointer position in canvas pixels, or null when the pointer leaves. */
  setPointer?(position: [number, number] | null): void;
  /** The scene's own pointer gesture: its kind and a two or three word hint for the chip. */
  gesture?: {
    kind: string;
    hint: string;
    /** Where the gesture rests (0 to 1 across the canvas, y down); arrow keys start from here. */
    rest?: readonly [number, number];
  };
  /** Returns the pointer gesture to its rest state, the picture of the static poster. */
  resetPointer?(): void;
  /** Preferred parameter for horizontal card scrubbing when there is no native pointer interaction. */
  scrubParameter?: GpuSceneParameter;
  /** New parameter values from the controls; only changed ids need to be present. */
  setParameters?(values: Record<string, number>): void;
  destroy(): void;
};

export type GpuScene = (context: GpuSceneContext) => GpuSceneInstance | Promise<GpuSceneInstance>;

/** A scene module: the scene plus the parameters it exposes, if any. */
export type GpuSceneModule = {default: GpuScene; parameters?: GpuSceneParameter[]};

/** A lazily imported scene module. Keeps luma.gl out of the HTML. */
export type GpuSceneLoader = () => Promise<GpuSceneModule>;

/**
 * Parameters known without loading the scene, so controls render in the prerendered HTML. Pages pass
 * them to <GpuCanvas parameters={...}>; they must match the module's `parameters`.
 */
export type GpuSceneParameters = readonly GpuSceneParameter[];
