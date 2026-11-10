// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Pointer gesture driver for hand-written scenes (the flagship demos). It gives them the same
// behaviour defineOperationRun gives runs: the pointer rests at the poster position, glides and
// follows with a short damped lag, resolves an interaction at most once per frame, draws its glyphs
// over the picture and reports a readout. The scene keeps its own buffers, graph and draw loop.

import type {Device} from '@luma.gl/core';
import {getWorldPosition, type DemoLayer, type DemoTheme, type DemoView} from '../demos/kit';
import type {InteractionKind, Point, RunGlyph, RunPointer} from './interaction';
import {getGlyphMetrics, getPixelOfWorld, PointerOverlay, type Spotlight} from './pointer-overlay';

/** What a scene's `resolve` returns: glyphs and a readout. The scene applies its own state. */
export type PointerDriverResult = {
  glyphs?: readonly RunGlyph[];
  readout?: string;
  /**
   * The bright circle of a probe. Left out, a `probe` gets one around its glyphs; null turns it
   * off; other gestures have none unless they set it.
   */
  spotlight?: Spotlight | null;
};

export type PointerDriverContext = {
  /** World units per glyph pixel, for glyph sizes and pick tolerances. */
  pixelSize: number;
  /** World position to a position across the canvas (0 to 1, y down). */
  toUnit(point: Point): Point;
};

export type PointerDriverOptions = {
  device: Device;
  /** The scene's view; the driver reads it on every call, so the scene may resize it. */
  view: DemoView;
  theme: DemoTheme;
  reducedMotion: boolean;
  kind: InteractionKind;
  /** Two or three words for the chip: 'move the query point'. */
  hint: string;
  /** Pointer position (0 to 1 across the canvas, y down) of the poster. Default: the center. */
  rest?: Point | ((context: PointerDriverContext) => Point);
  /**
   * Called at most once per frame with the damped pointer. The scene updates its own state here
   * (and marks itself dirty) and returns the glyphs and readout for that position.
   */
  resolve(pointer: RunPointer, context: PointerDriverContext): PointerDriverResult;
  reportReadout?(text: string | null): void;
  /** Asks the scene to draw a frame. */
  requestDraw(): void;
};

export type PointerDriver = {
  readonly gesture: {kind: string; hint: string; rest: readonly [number, number]};
  /** Pointer position in canvas pixels; null (off the picture) holds the last state. */
  setPointer(position: [number, number] | null): void;
  /** Glides back to the rest position: the poster. */
  resetPointer(): void;
  /**
   * A slider took over: stops following the pointer, drops the glyphs and the readout, and makes
   * `refresh` a no-op until the pointer moves again, so the slider's value is not overwritten.
   */
  release(): void;
  /** Call every animation frame before drawing. Returns true while the pointer is still moving. */
  advance(timeMilliseconds: number): boolean;
  /** Resolves again at the current position (after a resize or a state change). */
  refresh(): void;
  /** Glyph layers to draw last, already updated for the current view. */
  getLayers(): DemoLayer[];
  destroy(): void;
};

/** Time constant of the glide from the rest position to a pointer that has just arrived. */
const GLIDE_MILLISECONDS = 45;

export function createPointerDriver(options: PointerDriverOptions): PointerDriver {
  const {device, view, theme} = options;
  const overlay = new PointerOverlay(device);
  let target: [number, number] | null = null;
  let effective: [number, number] = [0.5, 0.5];
  let rest: [number, number] = [0.5, 0.5];
  let lastTime = 0;
  let readout: string | null = null;
  let isDestroyed = false;
  /** True once the reader has pointed: the spotlight belongs to probing, not to the poster. */
  let isProbing = false;
  /** True while the position glides to a pointer that has just arrived. */
  let isGliding = false;
  /** True after `release`: a slider owns the scene's values until the pointer moves again. */
  let isReleased = false;

  const getContext = (): PointerDriverContext => ({
    pixelSize: getGlyphMetrics(view).pixelSize,
    toUnit: point => {
      const [pixelX, pixelY] = getPixelOfWorld(view, point[0], point[1]);
      return [pixelX / view.width, pixelY / view.height];
    }
  });

  const apply = () => {
    if (isDestroyed) return;
    const [unitX, unitY] = effective;
    const [x, y] = getWorldPosition(view, [unitX * view.width, unitY * view.height]);
    const result = options.resolve({x, y, unitX, unitY}, getContext()) ?? {};
    const spotlight =
      result.spotlight === undefined
        ? options.kind === 'probe'
          ? 'auto'
          : null
        : result.spotlight;
    overlay.set(result.glyphs ?? [], view, spotlight);
    if (options.reducedMotion) overlay.setSpotlightOn(isProbing);
    const nextReadout = result.readout ?? null;
    if (nextReadout !== readout) {
      readout = nextReadout;
      options.reportReadout?.(readout);
    }
  };

  const initialRest =
    typeof options.rest === 'function' ? options.rest(getContext()) : (options.rest ?? [0.5, 0.5]);
  rest = [initialRest[0], initialRest[1]];
  effective = [rest[0], rest[1]];
  apply();

  return {
    gesture: {kind: options.kind, hint: options.hint, rest},
    setPointer(position) {
      if (!position) return;
      // A pointer that arrives (first hover, or after a slider took over) glides in.
      if (!target && !options.reducedMotion) isGliding = true;
      target = [
        Math.min(1, Math.max(0, position[0] / view.width)),
        Math.min(1, Math.max(0, position[1] / view.height))
      ];
      isProbing = true;
      isReleased = false;
      if (options.reducedMotion) {
        effective = [target[0], target[1]];
        apply();
      }
      options.requestDraw();
    },
    resetPointer() {
      // Straight back to rest: a glide would write every value on the way over the defaults.
      target = null;
      isGliding = false;
      isProbing = false;
      isReleased = false;
      effective = [rest[0], rest[1]];
      overlay.setSpotlightOn(false);
      apply();
      options.requestDraw();
    },
    release() {
      target = null;
      isGliding = false;
      isProbing = false;
      isReleased = true;
      overlay.set([], view, null);
      if (readout !== null) {
        readout = null;
        options.reportReadout?.(null);
      }
      options.requestDraw();
    },
    advance(time) {
      const elapsed = Math.min(100, Math.max(0, time - lastTime));
      lastTime = time;
      const isFading = overlay.fadeSpotlight(isProbing, elapsed);
      if (isFading) options.requestDraw();
      if (!target) return isFading;
      const deltaX = target[0] - effective[0];
      const deltaY = target[1] - effective[1];
      if (deltaX === 0 && deltaY === 0) return isFading;
      // After the glide the position is the pointer itself, so handles stay under the cursor.
      if (!isGliding || Math.hypot(deltaX, deltaY) < 0.004) {
        isGliding = false;
        effective = [target[0], target[1]];
      } else {
        const amount = 1 - Math.exp(-elapsed / GLIDE_MILLISECONDS);
        effective = [effective[0] + deltaX * amount, effective[1] + deltaY * amount];
      }
      apply();
      return true;
    },
    refresh() {
      if (!isReleased) apply();
    },
    getLayers: () => overlay.getLayers(view, theme),
    destroy() {
      isDestroyed = true;
      overlay.destroy();
    }
  };
}
