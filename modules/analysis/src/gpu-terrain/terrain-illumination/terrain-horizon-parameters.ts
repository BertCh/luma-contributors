// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateParameterTarget,
  validateRasterGridOrthogonal
} from '../../raster-analysis/raster-band-utils';
import {validateTerrainIlluminationCellSize} from './terrain-illumination-utils';
import {writeRasterGridParameterValues} from '../../analysis-core/contracts/raster-grid';
import {
  getTerrainSweepSourceParameterValues,
  TERRAIN_SWEEP_GRID_SLOTS
} from './terrain-horizon-sweep';

/** Number of float32 values read from `GPUTerrainHorizonProps.parameters`. */
export const GPU_TERRAIN_HORIZON_PARAMETER_LENGTH = 8;

/**
 * Number of float32 values read from `GPUTerrainHorizonProps.parameters` when
 * `anisotropicSkyViewFactor` is bound: the 8 base values plus
 * `[rvtDirectionDegrees, anisotropyLevel, anisotropyMinimumWeight, 0]`.
 */
export const GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH = 12;

/** Explicit sweep layout: base/anisotropy, reference cells/origin phase, shared source georeference. */
export const GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH = 24;

/**
 * Default direction of maximum anisotropic sky-view weight in RVT's convention: RVT's default
 * `asvf_dir = 315`. RVT direction angles run counterclockwise from north (90 is west, 270 east),
 * so 315 is compass azimuth 45 (north-east). See
 * {@link GPUTerrainHorizonSettings.rvtDirectionDegrees}.
 */
export const GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES = 315;

/** Default anisotropic sky-view exponent (RVT "low" preset: 4). */
export const GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL = 4;

/** Default anisotropic sky-view minimum weight (RVT "low" preset: 0.4). */
export const GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT = 0.4;

/** CPU-side description packed by {@link getGPUTerrainHorizonParameterValues}. */
export type GPUTerrainHorizonSettings = {
  /** Elevation multiplier. Defaults to 1. */
  zFactor?: number;
  /**
   * Earth curvature and refraction drop in 1/meters: a sample at ground distance `d` is lowered by
   * `c * d^2`. See `getGPUTerrainCurvatureCoefficient`. Defaults to 0.
   */
  curvatureCoefficient?: number;
  /** Ground search radius in meters; `<= 0` searches the full pixel radius. Defaults to 0. */
  maximumDistance?: number;
  /**
   * Shared ground metres per overview pixel `[x, y]` for sweep geometry and curvature decisions.
   * Use the same positive finite pair for every window of a mosaic, including the whole-grid
   * evaluation. Required for a seamless sweep halo in latitude-dependent cell-size modes.
   * With `sweepSourceGrid`, packs the explicit 24-value sweep layout; this pair occupies 12..13.
   */
  sweepReferenceCellSize?: readonly [number, number];
  /** Source-window origin in overview pixels, relative to `sweepSourceGrid`; per frame. Defaults to [0, 0]. */
  sweepOrigin?: readonly [number, number];
  /**
   * Shared full overview grid for row spacing and distance cutoffs; required with sweep settings.
   * Every window uses the same source grid per frame. Its height is at most 16777215 so global
   * source rows can be combined exactly as integers before conversion to float32.
   */
  sweepSourceGrid?: GPURasterMetadata;
  /**
   * Direction of maximum weight for `anisotropicSkyViewFactor`, in RVT's convention: the same
   * number as RVT's `asvf_dir` / `a_main_direction` means the same direction. RVT direction angles
   * run counterclockwise from north (0 north, 90 west, 180 south, 270 east), so `A` is compass
   * azimuth `(360 - A) mod 360`. Defaults to
   * {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES} (315, RVT's default, which is
   * compass north-east). This is the only counterclockwise angle in gpu-terrain: sector azimuths,
   * sun azimuths and aspect are compass azimuths, clockwise from north.
   */
  rvtDirectionDegrees?: number;
  /**
   * Weight exponent of `anisotropicSkyViewFactor`: a non-negative integer, RVT's `a_poly_level`.
   * RVT's presets are low (level 4, minimum weight 0.4) and high (level 8, minimum weight 0.1).
   * The power is signed as in RVT, `cos((a - A) / 2)^level`, so an odd level gives a weight below
   * the minimum weight on the far side of the main direction, exactly like RVT. Defaults to
   * {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL}.
   */
  anisotropyLevel?: number;
  /**
   * Weight floor of `anisotropicSkyViewFactor` in `[0, 1]` (RVT: low 0.4, high 0.1). Defaults to
   * {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT}.
   */
  anisotropyMinimumWeight?: number;
};

/**
 * Packs settings into the float layout read by {@link GPUTerrainHorizon}:
 * `[cellSizeX, cellSizeY, zFactor, curvatureCoefficient, northEdge, southEdge, maximumDistance, 0]`
 * (cell size and row-edge world y taken from `grid`), followed by `[rvtDirectionDegrees, anisotropyLevel, anisotropyMinimumWeight, 0]` (12 values,
 * see {@link GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH}) when any anisotropy field is
 * given or `target` holds at least 12 values. Otherwise exactly 8 values are written and returned.
 * Sweep settings explicitly select a 24-value layout: reference ground cell x/y at 12..13,
 * reduced origin x/y at 14..15, shared source cell size at 16..17, source north/south edges at
 * 18..19, source height and window row offset at 20..21, reserved zeros at 22..23. Both
 * `sweepReferenceCellSize` and `sweepSourceGrid` are required. The origin defaults to [0, 0]
 * relative to that source overview grid, and the window must lie inside it. Oversized targets
 * with ordinary settings retain the ordinary layout; padding never selects sweep semantics.
 *
 * @throws If the grid cell size is not finite and positive, an anisotropy field is out of range, or
 *   `target` is too short.
 */
export function getGPUTerrainHorizonParameterValues(
  settings: GPUTerrainHorizonSettings,
  grid: GPURasterMetadata,
  target?: Float32Array
): Float32Array {
  validateRequirement(
    'getGPUTerrainHorizonParameterValues',
    !('anisotropyAzimuthDegrees' in settings),
    'anisotropyAzimuthDegrees was removed; use rvtDirectionDegrees'
  );
  validateRasterGridOrthogonal('getGPUTerrainHorizonParameterValues', grid);
  if (settings.sweepSourceGrid)
    validateRasterGridOrthogonal(
      'getGPUTerrainHorizonParameterValues sweepSourceGrid',
      settings.sweepSourceGrid
    );
  const wantsSweepReference =
    settings.sweepReferenceCellSize !== undefined ||
    settings.sweepOrigin !== undefined ||
    settings.sweepSourceGrid !== undefined;
  validateRequirement(
    'getGPUTerrainHorizonParameterValues',
    !wantsSweepReference ||
      (settings.sweepReferenceCellSize !== undefined && settings.sweepSourceGrid !== undefined),
    'sweepReferenceCellSize and sweepSourceGrid are required for sweep settings'
  );
  const hasAnisotropy =
    settings.rvtDirectionDegrees !== undefined ||
    settings.anisotropyLevel !== undefined ||
    settings.anisotropyMinimumWeight !== undefined;
  const wantsAnisotropy =
    wantsSweepReference || hasAnisotropy || (target !== undefined && target.length >= 12);
  const values =
    target ??
    new Float32Array(
      wantsSweepReference
        ? GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH
        : wantsAnisotropy
          ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
          : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
  validateTerrainIlluminationCellSize('getGPUTerrainHorizonParameterValues', grid);
  validateParameterTarget(
    'getGPUTerrainHorizonParameterValues',
    values,
    wantsSweepReference
      ? GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH
      : hasAnisotropy
        ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
        : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
  );
  const anisotropyAzimuth =
    settings.rvtDirectionDegrees ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES;
  const anisotropyLevel = settings.anisotropyLevel ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL;
  const anisotropyMinimumWeight =
    settings.anisotropyMinimumWeight ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT;
  if (wantsAnisotropy) {
    validateRequirement(
      'getGPUTerrainHorizonParameterValues',
      Number.isFinite(Math.fround(anisotropyAzimuth)),
      `rvtDirectionDegrees (anisotropy azimuth) must remain finite after float32 conversion; received ${anisotropyAzimuth}, encoded ${Math.fround(anisotropyAzimuth)}`
    );
    if (!Number.isSafeInteger(anisotropyLevel) || anisotropyLevel < 0) {
      throw new Error('Terrain horizon anisotropy level must be a non-negative integer');
    }
    if (
      !Number.isFinite(anisotropyMinimumWeight) ||
      anisotropyMinimumWeight < 0 ||
      anisotropyMinimumWeight > 1
    ) {
      throw new Error('Terrain horizon anisotropy minimum weight must be in [0, 1]');
    }
  }
  values.set([
    0,
    0,
    settings.zFactor ?? 1,
    settings.curvatureCoefficient ?? 0,
    0,
    0,
    settings.maximumDistance ?? 0,
    0
  ]);
  writeRasterGridParameterValues(values, grid, TERRAIN_SWEEP_GRID_SLOTS);
  if (wantsAnisotropy) {
    values.set([anisotropyAzimuth, anisotropyLevel, anisotropyMinimumWeight, 0], 8);
  }
  if (settings.sweepReferenceCellSize) {
    values.set(
      getTerrainSweepSourceParameterValues('getGPUTerrainHorizonParameterValues', settings, grid),
      12
    );
  }
  return values;
}
