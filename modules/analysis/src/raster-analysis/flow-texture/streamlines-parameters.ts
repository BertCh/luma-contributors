// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveFlowRasterPlacement,
  resolveFlowLocalOrigin,
  type FlowRasterPlacement
} from './flow-raster-placement';

import {
  validateNonZeroFiniteNumber,
  validateRequirement,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';

/** Philox key word 1 of the seed jitter and priority stream. @internal */
export const STREAMLINES_SEED_PURPOSE = PHILOX_PURPOSES.streamlinesJitter.id;

/**
 * Boolean candidate-loss value when pruning rounds left undecided seeds. Final overflow reports
 * only line or point capacity.
 */
export const GPU_STREAMLINES_CANDIDATE_UNCONVERGED = 1;

/** Number of float32 elements in a {@link GPUStreamlines} parameter view. */
export const GPU_STREAMLINES_PARAMETER_LENGTH = 12;

/** Number of uint32 elements in a {@link GPUStreamlines} uint32 parameter view. */
export const GPU_STREAMLINES_UINT32_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUStreamlines}. */
export type GPUStreamlinesSettings = {
  /** Planar origin subtracted before packing both placements. All returned positions, including
   * candidates, are local; add this origin in binary64 on readback. Defaults to [0, 0]. */
  localOrigin?: readonly [number, number];
  /**
   * Field placement uses an outer-corner `origin` and signed `cellSize`.
   * Sample `(column, row)` is at origin + `(column + 0.5, row + 0.5)` times
   * cell size. Positive cellHeight puts row zero at the smallest y; signed cell sizes determine
   * the direction of increasing indices. Zero or non-finite sizes throw. Sampling is bilinear
   * between centers, clamped inside the outer bounds. No grid affine changes this placement.
   */
  fieldPlacement: FlowRasterPlacement;
  /**
   * Occupancy grid placement. The cell size is the
   * separation distance between streamlines; seeds and traced points stay inside the grid.
   * Cell `(column, row)` is the half-open rectangle from origin + `(column, row)` times cell
   * size to origin + `(column + 1, row + 1)` times cell size. Both cell sizes must be positive.
   * Both cell sizes must also have finite float32 reciprocals.
   */
  gridPlacement: FlowRasterPlacement;
  /** Integration step in world units. */
  stepLength: number;
  /** A streamline stops where the field is slower than this. Defaults to 0. */
  minimumSpeed?: number;
  /** Seed of the jittered seed lattice and the line priorities. Defaults to 0. */
  seed?: number;
  /** Streamlines with fewer points after trimming are rejected. Defaults to 2. */
  minimumPoints?: number;
};

/** Settings accepted by the float parameter packer. */
export type GPUStreamlinesParameterSettings = GPUStreamlinesSettings;

/**
 * Packs float parameters of {@link GPUStreamlines}.
 *
 * Layout: `[fieldOriginX, fieldOriginY, fieldCellWidth, fieldCellHeight, gridOriginX,
 * gridOriginY, gridCellWidth, gridCellHeight, 1 / gridCellWidth, 1 / gridCellHeight, stepLength,
 * minimumSpeed]`. The reciprocals are rounded once here so the GPU and the CPU oracle assign
 * points to grid cells with the same correctly rounded subtraction and multiplication.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPUStreamlinesParameterValues(
  settings: GPUStreamlinesParameterSettings,
  target: Float32Array = new Float32Array(GPU_STREAMLINES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_STREAMLINES_PARAMETER_LENGTH) {
    throw new Error(
      `Streamlines parameter target must hold ${GPU_STREAMLINES_PARAMETER_LENGTH} elements`
    );
  }
  const fieldPlacementWords = resolveFlowRasterPlacement(
    'GPUStreamlines',
    settings,
    'fieldPlacement',
    'fieldExtent'
  );
  const gridPlacementWords = resolveFlowRasterPlacement(
    'GPUStreamlines',
    settings,
    'gridPlacement',
    'gridExtent'
  );
  const [localOriginX, localOriginY] = resolveFlowLocalOrigin(
    'GPUStreamlines',
    settings.localOrigin
  );
  const [, , cellWidth, cellHeight] = gridPlacementWords;
  // Tested after rounding to float32, the precision the kernel receives: 1e-50 rounds to 0 and
  // 1e39 to Infinity, and the reciprocals packed below would then be Infinity or 0.
  for (const gridCellSize of [Math.fround(cellWidth), Math.fround(cellHeight)]) {
    if (!(gridCellSize > 0) || !Number.isFinite(gridCellSize)) {
      throw new Error(
        'GPUStreamlines gridPlacement cells must have a positive size that is finite in float32'
      );
    }
  }
  for (const [name, value] of [
    ['gridPlacement cellWidth', cellWidth],
    ['gridPlacement cellHeight', cellHeight]
  ] as const) {
    const reciprocal = Math.fround(1 / Math.fround(value));
    validateRequirement(
      'GPUStreamlines',
      Number.isFinite(reciprocal) && reciprocal > 0,
      `${name} reciprocal must be positive and finite in float32; received cell size ${value}, reciprocal ${reciprocal}`
    );
  }
  validateNonZeroFiniteNumber('GPUStreamlines', 'fieldPlacement cellWidth', fieldPlacementWords[2]);
  validateNonZeroFiniteNumber(
    'GPUStreamlines',
    'fieldPlacement cellHeight',
    fieldPlacementWords[3]
  );
  for (const [name, value] of [
    ['fieldPlacement originX - localOrigin', fieldPlacementWords[0] - localOriginX],
    ['fieldPlacement originY - localOrigin', fieldPlacementWords[1] - localOriginY],
    ['gridPlacement originX - localOrigin', gridPlacementWords[0] - localOriginX],
    ['gridPlacement originY - localOrigin', gridPlacementWords[1] - localOriginY]
  ] as const) {
    validateFiniteNumber('GPUStreamlines', `${name} (received ${value})`, value);
  }
  target.set([
    fieldPlacementWords[0] - localOriginX,
    fieldPlacementWords[1] - localOriginY,
    fieldPlacementWords[2],
    fieldPlacementWords[3],
    gridPlacementWords[0] - localOriginX,
    gridPlacementWords[1] - localOriginY,
    cellWidth,
    cellHeight,
    1 / Math.fround(cellWidth),
    1 / Math.fround(cellHeight),
    settings.stepLength,
    settings.minimumSpeed ?? 0
  ]);
  return target;
}

/**
 * Packs integer parameters of {@link GPUStreamlines}. Layout: `[seed, minimumPoints, 0, 0]`.
 *
 * @param settings Per-frame settings; reads `seed` and `minimumPoints`.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPUStreamlinesUint32ParameterValues(
  settings: Pick<GPUStreamlinesSettings, 'seed' | 'minimumPoints'> = {},
  target: Uint32Array = new Uint32Array(GPU_STREAMLINES_UINT32_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_STREAMLINES_UINT32_PARAMETER_LENGTH) {
    throw new Error(
      `Streamlines uint32 parameter target must hold ${GPU_STREAMLINES_UINT32_PARAMETER_LENGTH} elements`
    );
  }
  const seed = settings.seed ?? 0;
  const minimumPoints = settings.minimumPoints ?? 2;
  for (const [name, value] of [
    ['seed', seed],
    ['minimumPoints', minimumPoints]
  ] as const) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error(`Streamlines ${name} must be a uint32 integer`);
    }
  }
  target.set([seed, minimumPoints, 0, 0]);
  return target;
}
