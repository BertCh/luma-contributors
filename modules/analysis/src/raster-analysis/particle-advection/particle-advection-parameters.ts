// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveFlowRasterPlacement,
  resolveFlowLocalOrigin,
  type FlowRasterPlacement
} from '../flow-texture/flow-raster-placement';

import {
  validateNonZeroFiniteNumber,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';

/** Number of float32 elements in a {@link GPUParticleAdvection} parameter view. */
export const GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH = 12;

/** Number of uint32 elements in a {@link GPUParticleAdvection} uint32 parameter view. */
export const GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH = 4;

/** Philox key word 1 of the respawn position and staggered-age stream, shared by GPU and CPU. @internal */
export const PARTICLE_ADVECTION_SPAWN_PURPOSE = PHILOX_PURPOSES.particleAdvectionSpawn.id;
/** Philox key word 1 of the per-frame drop test stream, shared by GPU and CPU. @internal */
export const PARTICLE_ADVECTION_DROP_PURPOSE = PHILOX_PURPOSES.particleAdvectionDrop.id;

/** Per-frame float settings of {@link GPUParticleAdvection}. */
export type GPUParticleAdvectionSettings = {
  /** Planar origin subtracted in binary64 before packing. State, previous positions and trails
   * stay local across frames; upload initial positions as world minus this origin before narrowing.
   * Add it back in binary64 on readback. Keep it fixed until reset. Defaults to [0, 0]. */
  localOrigin?: readonly [number, number];
  /**
   * Field raster placement with an outer-corner `origin` and `cellSize`.
   * Positive cellHeight puts row zero on the smallest-y side; signed sizes set index direction.
   * Cell `(i, j)` is centered at `origin + (i + 0.5, j + 0.5) * cellSize`. A cell size
   * that is zero or not finite throws.
   */
  fieldPlacement: FlowRasterPlacement;
  /** Time step of this frame, in the time unit of the field velocities. */
  timeStep: number;
  /** Multiplier on the time step, for example a playback speed. Defaults to 1. */
  speedScale?: number;
  /** Probability in `[0, 1]` that a particle respawns this frame regardless of age. Defaults to 0. */
  dropRate?: number;
  /** Particles slower than this (world units per time unit, after `speedScale`) respawn. Defaults to 0. */
  minimumSpeed?: number;
  /**
   * Respawn rectangle `[minX, minY, maxX, maxY]`. Defaults to the field bounds. Particles spawned
   * outside the field respawn again on the next frame.
   */
  spawnBounds?: readonly [number, number, number, number];
};

/** Settings accepted by the float parameter packer. */
export type GPUParticleAdvectionParameterSettings = GPUParticleAdvectionSettings;

/** Per-frame integer settings of {@link GPUParticleAdvection}. */
export type GPUParticleAdvectionUint32Settings = {
  /** Random seed. Every random decision is a pure function of the seed and particle counters. */
  seed: number;
  /** Frame counter. Keys the per-frame drop test and selects the trail ring slot `frame % L`. */
  frame: number;
  /** Particles respawn when their age reaches this many frames. `0` disables ageing. */
  maximumAge: number;
  /**
   * When true, every particle respawns with generation 0 and a staggered age, which initialises
   * the state buffers. Pass it on the first frame and whenever the scene is reset.
   */
  reset?: boolean;
};

/**
 * Packs float parameters of {@link GPUParticleAdvection}.
 *
 * Layout: `[originX, originY, cellWidth, cellHeight, timeStep, speedScale, dropRate,
 * minimumSpeed, spawnMinX, spawnMinY, spawnMaxX, spawnMaxY]` as float32.
 *
 * @param settings Per-frame settings.
 * @param fieldSize `[width, height]` of the field in cells, used for the default spawn bounds.
 * @param target Optional destination of at least {@link GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH} elements.
 */
export function getGPUParticleAdvectionParameterValues(
  settings: GPUParticleAdvectionParameterSettings,
  fieldSize: readonly [number, number],
  target: Float32Array = new Float32Array(GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH) {
    throw new Error(
      `Particle advection parameter target must hold ${GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH} elements`
    );
  }
  const [originX, originY, cellWidth, cellHeight] = resolveFlowRasterPlacement(
    'GPUParticleAdvection',
    settings,
    'fieldPlacement',
    'fieldExtent'
  );
  validateNonZeroFiniteNumber('GPUParticleAdvection', 'fieldPlacement cellWidth', cellWidth);
  validateNonZeroFiniteNumber('GPUParticleAdvection', 'fieldPlacement cellHeight', cellHeight);
  const [localOriginX, localOriginY] = resolveFlowLocalOrigin(
    'GPUParticleAdvection',
    settings.localOrigin
  );
  const localFieldOriginX = originX - localOriginX;
  const localFieldOriginY = originY - localOriginY;
  const spawnBounds = settings.spawnBounds
    ? [
        settings.spawnBounds[0] - localOriginX,
        settings.spawnBounds[1] - localOriginY,
        settings.spawnBounds[2] - localOriginX,
        settings.spawnBounds[3] - localOriginY
      ]
    : [
        localFieldOriginX,
        localFieldOriginY,
        localFieldOriginX + cellWidth * fieldSize[0],
        localFieldOriginY + cellHeight * fieldSize[1]
      ];
  for (const [name, value] of [
    ['fieldPlacement originX - localOrigin', originX - localOriginX],
    ['fieldPlacement originY - localOrigin', originY - localOriginY],
    ['spawnBounds minX - localOrigin', spawnBounds[0]],
    ['spawnBounds minY - localOrigin', spawnBounds[1]],
    ['spawnBounds maxX - localOrigin', spawnBounds[2]],
    ['spawnBounds maxY - localOrigin', spawnBounds[3]]
  ] as const) {
    validateFiniteNumber('GPUParticleAdvection', `${name} (received ${value})`, value);
  }
  target.set([
    originX - localOriginX,
    originY - localOriginY,
    cellWidth,
    cellHeight,
    settings.timeStep,
    settings.speedScale ?? 1,
    settings.dropRate ?? 0,
    settings.minimumSpeed ?? 0,
    spawnBounds[0],
    spawnBounds[1],
    spawnBounds[2],
    spawnBounds[3]
  ]);
  return target;
}

/**
 * Packs integer parameters of {@link GPUParticleAdvection}.
 *
 * Layout: `[seed, frame, maximumAge, reset]` as uint32.
 *
 * @param settings Per-frame integer settings.
 * @param target Optional destination of at least {@link GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH} elements.
 */
export function getGPUParticleAdvectionUint32ParameterValues(
  settings: GPUParticleAdvectionUint32Settings,
  target: Uint32Array = new Uint32Array(GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH) {
    throw new Error(
      `Particle advection uint32 parameter target must hold ${GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH} elements`
    );
  }
  for (const [name, value] of [
    ['seed', settings.seed],
    ['frame', settings.frame],
    ['maximumAge', settings.maximumAge]
  ] as const) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error(`Particle advection ${name} must be a uint32 integer`);
    }
  }
  target[0] = settings.seed;
  target[1] = settings.frame;
  target[2] = settings.maximumAge;
  target[3] = settings.reset ? 1 : 0;
  return target;
}
