// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  type GPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

/** Number of float32 elements in a {@link GPUInverseDistanceWeighting} parameter view. */
export const GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH = 9;

/** Number of float32 elements in a {@link GPUFocalStatistics} parameter view. */
export const GPU_FOCAL_STATISTICS_PARAMETER_LENGTH = 4;

/**
 * Weight taper of {@link GPUInverseDistanceWeighting}: `'none'` is `w = 1 / d^p` inside the
 * search radius (a hard cutoff); `'modified-shepard'` is `w = ((R - d)+ / (R d))^p` with `R` the
 * search radius (Franke and Nielson 1980), which falls to zero at `R`.
 */
export type GPUInverseDistanceWeightingTaper = 'none' | 'modified-shepard';

/**
 * Returns the two outer corners `[startX, startY, endX, endY]` of an axis-aligned raster grid in
 * grid order: `(startX, startY)` is the outer corner of cell `(0, 0)` and `(endX, endY)` is the
 * opposite outer corner. Point pixels are centered on their coordinates, so their start corner
 * moves by half a cell. The end corner is smaller than the start corner on an axis whose cell
 * step is negative (the usual north-up rows).
 *
 * @param name Name of the calling function, for the error text.
 * @param validateEncodedCorners Whether the consuming route derives steps from float32 corners.
 * @throws If the grid is rotated or sheared, or its dimensions or affine are invalid.
 * @internal
 */
export function getRasterGridCorners(
  name: string,
  grid: GPURasterMetadata,
  validateEncodedCorners: boolean = true
): [number, number, number, number] {
  validateRasterGrid(name, grid);
  const [stepX, shearX, originX, shearY, stepY, originY] = grid.affine;
  validateRequirement(name, !(shearX !== 0 || shearY !== 0), `${name} needs an axis-aligned grid`);
  const cornerShift = grid.pixelInterpretation === 'point' ? -0.5 : 0;
  const startX = originX + stepX * cornerShift;
  const startY = originY + stepY * cornerShift;
  const endX = startX + stepX * grid.width;
  const endY = startY + stepY * grid.height;
  // Origin/step layouts (Gaussian fields) never derive cell steps from narrowed corners.
  if (!validateEncodedCorners) return [startX, startY, endX, endY];
  for (const [axis, start, end, count, step] of [
    ['x', startX, endX, grid.width, stepX],
    ['y', startY, endY, grid.height, stepY]
  ] as const) {
    const encodedStart = Math.fround(start);
    const encodedEnd = Math.fround(end);
    const encodedStep = Math.fround(Math.fround(encodedEnd - encodedStart) / count);
    validateRequirement(
      name,
      Number.isFinite(encodedStart) &&
        Number.isFinite(encodedEnd) &&
        Number.isFinite(encodedStep) &&
        encodedStep !== 0 &&
        Math.sign(encodedStep) === Math.sign(step),
      `${axis} grid original endpoints [${start}, ${end}], encoded endpoints [${encodedStart}, ${encodedEnd}] must preserve direction and a finite nonzero float32 cell step; subtract a common tile origin from grid and points`
    );
  }
  return [startX, startY, endX, endY];
}

/** Per-frame settings of {@link GPUInverseDistanceWeighting}. */
export type GPUInverseDistanceWeightingSettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Search radius in position units. `Infinity` searches the whole index domain, except with the
   * `'modified-shepard'` taper, which needs a finite radius.
   */
  searchRadius: number;
  /** Distance power `p` in `w = 1 / d^p`. Zero gives an unweighted mean. Defaults to 2. */
  power?: number;
  /** Weight taper at the search radius. Defaults to `'none'` (hard cutoff). */
  taper?: GPUInverseDistanceWeightingTaper;
  /**
   * Nearest-neighbor limit `k`; zero (default) uses every sample within the radius. Values above
   * the contributor's compile-time `maximumNeighborCount` are clamped by raw GPU parameters; use
   * the instance's `getParameterValues` method to reject requests beyond its capacity.
   */
  neighborCount?: number;
  /** Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 1. */
  minimumNeighborCount?: number;
};

/**
 * Packs per-frame {@link GPUInverseDistanceWeighting} parameters.
 *
 * Layout (float32): `[startX, startY, endX, endY, searchRadius, power, neighborCount,
 * minimumNeighborCount, taper (0 none, 1 modified-shepard)]`. The first four values are the outer
 * corners of the output grid in grid order: `(startX, startY)` is the outer corner of cell
 * `(0, 0)` and `(endX, endY)` the opposite corner, so a north-up grid (`grid.affine[4] < 0`) has
 * `endY < startY` and row 0 at the top, and a grid whose rows advance north has row 0 at `minY`.
 *
 * @param settings Search radius, power, taper, and neighbor limits.
 * @param grid The output grid: axis-aligned; its width and height must match the contributor's.
 * @param target Optional destination of at least 9 elements.
 * @throws If the grid is rotated or sheared or invalid, the target is too short, a count is not a
 *   non-negative integer, the taper is unknown, or the `'modified-shepard'` taper is requested
 *   with a non-finite radius.
 */
export function getGPUInverseDistanceWeightingParameterValues(
  settings: GPUInverseDistanceWeightingSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH)
): Float32Array {
  validateRequirement(
    'getGPUInverseDistanceWeightingParameterValues',
    !(target.length < GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH),
    `Inverse distance weighting parameter target must hold ${GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH} elements`
  );
  const neighborCount = settings.neighborCount ?? 0;
  const minimumNeighborCount = settings.minimumNeighborCount ?? 1;
  for (const [name, count] of [
    ['neighborCount', neighborCount],
    ['minimumNeighborCount', minimumNeighborCount]
  ] as const) {
    validateCapacity('Inverse distance weighting', name, count, 0);
  }
  const taper = settings.taper ?? 'none';
  validateRequirement(
    'getGPUInverseDistanceWeightingParameterValues',
    !(taper !== 'none' && taper !== 'modified-shepard'),
    "Inverse distance weighting taper must be 'none' or 'modified-shepard'"
  );
  validateRequirement(
    'getGPUInverseDistanceWeightingParameterValues',
    !(taper === 'modified-shepard' && !Number.isFinite(settings.searchRadius)),
    'Inverse distance weighting taper needs a finite searchRadius'
  );
  warnGPUShortLongitudeLatitudeDistance(
    'GPUInverseDistanceWeighting',
    settings.spatialContext,
    settings.searchRadius
  );
  const radius = Math.fround(settings.searchRadius);
  const power = Math.fround(settings.power ?? 2);
  validateRequirement(
    'getGPUInverseDistanceWeightingParameterValues',
    typeof settings.searchRadius === 'number' &&
      (Number.isNaN(settings.searchRadius) ||
        settings.searchRadius < 0 ||
        (settings.searchRadius === Infinity && taper === 'none') ||
        (Number.isFinite(radius) && radius >= 0 && (settings.searchRadius === 0 || radius > 0))),
    `searchRadius must preserve its finite nonnegative domain after float32 encoding (or Infinity with taper 'none'); negative/NaN nodata controls remain supported; received ${settings.searchRadius}`
  );
  validateRequirement(
    'getGPUInverseDistanceWeightingParameterValues',
    typeof (settings.power ?? 2) === 'number' &&
      (!Number.isFinite(settings.power ?? 2) ||
        (settings.power ?? 2) < 0 ||
        Number.isFinite(power)),
    `finite nonnegative power must remain finite after float32 encoding; nodata controls remain supported; received ${settings.power ?? 2}`
  );
  const corners = getRasterGridCorners('getGPUInverseDistanceWeightingParameterValues', grid);
  target.set(corners, 0);
  // Negative controls are documented nodata requests. Do not let narrowing to -0 enable them.
  target[4] = settings.searchRadius < 0 && radius === 0 ? NaN : settings.searchRadius;
  target[5] = (settings.power ?? 2) < 0 && power === 0 ? NaN : (settings.power ?? 2);
  target[6] = neighborCount;
  target[7] = minimumNeighborCount;
  target[8] = taper === 'modified-shepard' ? 1 : 0;
  return target;
}

/** Focal window shape. */
export type GPUFocalStatisticsShape = 'square' | 'circle';

/** Per-frame settings of {@link GPUFocalStatistics}. */
export type GPUFocalStatisticsSettings = {
  /**
   * Window radius in cells. A square window covers `|dx|, |dy| <= floor(radius)`; a circle covers
   * offsets with `dx^2 + dy^2 <= radius^2` inside that square. Clamped to the contributor's
   * compile-time `maximumRadius`.
   */
  radius: number;
  /** Window shape. Defaults to `'square'`. */
  shape?: GPUFocalStatisticsShape;
  /** Cells whose window has fewer valid cells are nodata. Defaults to 1. */
  minimumCount?: number;
  /** When true, a nodata center cell yields nodata statistics. Defaults to false. */
  propagateCenterNoData?: boolean;
};

/**
 * Packs per-frame {@link GPUFocalStatistics} parameters.
 *
 * Layout (float32): `[radius, shape (0 square, 1 circle), minimumCount, propagateCenterNoData]`.
 *
 * @param settings Radius, shape, and nodata policy.
 * @param target Optional destination of at least 4 elements.
 * @throws If the target is too short or `minimumCount` is not a non-negative integer.
 */
export function getGPUFocalStatisticsParameterValues(
  settings: GPUFocalStatisticsSettings,
  target: Float32Array = new Float32Array(GPU_FOCAL_STATISTICS_PARAMETER_LENGTH)
): Float32Array {
  validateRequirement(
    'getGPUFocalStatisticsParameterValues',
    !(target.length < GPU_FOCAL_STATISTICS_PARAMETER_LENGTH),
    `Focal statistics parameter target must hold ${GPU_FOCAL_STATISTICS_PARAMETER_LENGTH} elements`
  );
  const minimumCount = settings.minimumCount ?? 1;
  validateCapacity('Focal statistics', 'minimumCount', minimumCount, 0);
  const shape = settings.shape ?? 'square';
  validateRequirement(
    'getGPUFocalStatisticsParameterValues',
    !(shape !== 'square' && shape !== 'circle'),
    "Focal statistics shape must be 'square' or 'circle'"
  );
  target[0] = settings.radius;
  target[1] = shape === 'circle' ? 1 : 0;
  target[2] = minimumCount;
  target[3] = settings.propagateCenterNoData ? 1 : 0;
  return target;
}
