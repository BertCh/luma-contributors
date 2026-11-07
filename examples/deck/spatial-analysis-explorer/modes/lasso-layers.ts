// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Viewport} from '@deck.gl/core';

import {clamp} from '@math.gl/core';
import {PICK_TARGET_SIZE} from '../../spatial-analysis-shared/rack-pick';

export {
  createRackPickBuffers,
  addRackPickPasses,
  getPickMatrix,
  type RackPickBuffers,
  PICK_RESULT_PAIRS,
  PICK_TARGET_SIZE
} from '../../spatial-analysis-shared/rack-pick';

/** Edge of the pick window around a click, in target pixels. */
export const PICK_WINDOW_SIZE = 44;

/** Pick window in target pixels around a CSS pixel position of a viewport. */
export function getPickWindow(
  viewport: Viewport,
  pixel: readonly [number, number]
): Uint32Array<ArrayBuffer> {
  const targetX = (pixel[0] / viewport.width) * PICK_TARGET_SIZE;
  const targetY = (pixel[1] / viewport.height) * PICK_TARGET_SIZE;
  const half = PICK_WINDOW_SIZE / 2;
  const x = clamp(Math.round(targetX - half), 0, PICK_TARGET_SIZE - 1);
  const y = clamp(Math.round(targetY - half), 0, PICK_TARGET_SIZE - 1);
  return Uint32Array.of(x, y, PICK_WINDOW_SIZE, PICK_WINDOW_SIZE);
}
