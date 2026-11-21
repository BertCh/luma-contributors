// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of uint32 elements in a {@link GPUDotDensity} or {@link GPURandomPointsInPolygon} parameter view. */
export const GPU_DOT_DENSITY_PARAMETER_LENGTH = 8;

/** Per-frame settings of {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensitySettings = {
  /** Random seed. Every dot position is a pure function of the seed and its (slot, rank). */
  seed: number;
  /**
   * Dots per unit of value (the reciprocal of "people per dot"), usually zoom dependent. Ignored by
   * {@link GPURandomPointsInPolygon}. Zero or a non-finite value draws no dots. Defaults to 1.
   */
  dotsPerUnit?: number;
  /** Mask origin and signed cell steps in polygon position units. Defaults to origin [0,0], cellSize [1,1]. */
  maskPlacement?: {
    origin: readonly [number, number];
    cellSize: readonly [number, number];
  };
};

/**
 * Packs per-frame parameters of {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}.
 *
 * Layout (uint32 words; floats stored as their IEEE-754 bits so one view carries both):
 * `[seed, f32 dotsPerUnit, f32 maskOriginX, f32 maskOriginY, f32 maskCellWidth,
 * f32 maskCellHeight, 0, 0]`.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least {@link GPU_DOT_DENSITY_PARAMETER_LENGTH} elements.
 */
export function getGPUDotDensityParameterValues(
  settings: GPUDotDensitySettings,
  target: Uint32Array = new Uint32Array(GPU_DOT_DENSITY_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_DOT_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `Dot density parameter target must hold ${GPU_DOT_DENSITY_PARAMETER_LENGTH} elements`
    );
  }
  const {seed} = settings;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('Dot density seed must be a uint32 integer');
  }
  const floats = new Float32Array(target.buffer, target.byteOffset, target.length);
  validateRequirement(
    'GPUDotDensity',
    !('maskExtent' in settings),
    'maskExtent was removed; use maskPlacement with origin and cellSize'
  );
  const {origin, cellSize} = settings.maskPlacement ?? {origin: [0, 0], cellSize: [1, 1]};
  const [originX, originY] = origin;
  const [cellWidth, cellHeight] = cellSize;
  const dotsPerUnit = settings.dotsPerUnit ?? 1;
  target[0] = seed;
  floats[1] = dotsPerUnit;
  floats[2] = originX;
  floats[3] = originY;
  floats[4] = cellWidth;
  floats[5] = cellHeight;
  target[6] = 0;
  target[7] = 0;
  return target;
}
