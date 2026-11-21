// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Number of float32 elements in a DBSCAN parameter buffer. */
export const GPU_DBSCAN_PARAMETER_LENGTH = 8;

/** Label and root value of points that belong to no cluster. */
export const GPU_DBSCAN_NOISE = 0xffffffff;

/** Largest accepted `minimumSamples`, the largest integer a float32 holds exactly. */
const MAXIMUM_MINIMUM_SAMPLES = 2 ** 24;
/** Squaring stays normal at the lower bound; two squared components stay finite at the upper. */
export const SPATIAL_CLUSTERING_MINIMUM_PLANAR_EPSILON = 2 ** -63;
export const SPATIAL_CLUSTERING_MAXIMUM_PLANAR_EPSILON = 2 ** 63;

/**
 * CPU settings of the per-frame parameters of one {@link GPUDBSCAN}.
 *
 * Distances use planar position units or the longitude/latitude context's length unit.
 */
export type GPUDBSCANSettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Inclusive `[minX, minY, maxX, maxY]` extent of the data. Points outside it, or with a
   * non-finite coordinate, are excluded and labeled noise. Bounds should cover the data: they
   * size the active cell lattice, and a tighter extent makes the neighbor search cheaper.
   * Bounds with `maxX < minX` or `maxY < minY` exclude every point.
   */
  bounds: readonly [number, number, number, number];
  /** Neighbor radius. Positive; on the planar route its float32 value must lie in [2^-63, 2^63]. */
  epsilon: number;
  /**
   * Integer `>= 1`. A point is a core point when its epsilon-neighborhood, including the point
   * itself, holds at least this many points.
   */
  minimumSamples: number;
  /**
   * Space-time clustering (ST-DBSCAN): two points are neighbors only when, besides the spatial
   * test, their relative times differ by at most this many ticks of the contributor's
   * `temporalContext` unit. Positive. Required when the contributor has `times`, and ignored
   * otherwise. Stored at index 6.
   */
  timeEpsilon?: number;
};

/**
 * Packs DBSCAN parameters into the 8-element float32 layout read by
 * `GPUDBSCAN`.
 *
 * Layout: `[minX, minY, maxX, maxY, epsilon, minimumSamples, timeEpsilon, 0]`; `timeEpsilon` is 0
 * when omitted.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `epsilon <= 0`, `timeEpsilon` is given and not positive and
 * finite (each also as its encoded float32 word, so a value beyond the float32 range or a positive
 * radius that rounds to zero is rejected), the planar encoded radius is outside `[2^-63, 2^63]`,
 * `minimumSamples` is not an integer in `[1, 2^24]`, or `target` is too short.
 */
export function getGPUDBSCANParameterValues(
  settings: GPUDBSCANSettings,
  target: Float32Array = new Float32Array(GPU_DBSCAN_PARAMETER_LENGTH)
): Float32Array {
  const id = 'GPUDBSCAN';
  validateRequirement(
    id,
    !('minimumPoints' in settings),
    'use minimumSamples instead of minimumPoints'
  );
  validateRequirement(
    id,
    target.length >= GPU_DBSCAN_PARAMETER_LENGTH,
    `target must hold ${GPU_DBSCAN_PARAMETER_LENGTH} elements`
  );
  validateRequirement(
    id,
    settings.bounds?.length === 4,
    `bounds must contain exactly 4 elements; received ${settings.bounds?.length}`
  );
  // Each check also applies to the float32 word the shader reads: a finite input can round to
  // infinity (above the float32 maximum) or to zero (below its smallest subnormal).
  // Index the bounds: JavaScript callers also supply typed arrays, whose map coerces tuples.
  for (let index = 0; index < settings.bounds.length; index++) {
    const value = settings.bounds[index];
    validateRequirement(
      id,
      Number.isFinite(value) && Number.isFinite(Math.fround(value)),
      `bounds[${index}] must be finite in float32`
    );
  }
  validateRequirement(
    id,
    Number.isFinite(settings.epsilon) && Number.isFinite(Math.fround(settings.epsilon)),
    'epsilon must be finite in float32'
  );
  validateRequirement(
    id,
    settings.epsilon > 0 && Math.fround(settings.epsilon) > 0,
    'epsilon must be positive in float32'
  );
  if (settings.spatialContext?.coordinateSpace !== 'longitude-latitude') {
    const epsilon = Math.fround(settings.epsilon);
    validateRequirement(
      id,
      epsilon >= SPATIAL_CLUSTERING_MINIMUM_PLANAR_EPSILON &&
        epsilon <= SPATIAL_CLUSTERING_MAXIMUM_PLANAR_EPSILON,
      'epsilon on the planar route must have a float32 value in [2^-63, 2^63] for squared distances'
    );
  }
  validateRequirement(
    id,
    settings.timeEpsilon === undefined ||
      (Number.isFinite(Math.fround(settings.timeEpsilon)) &&
        settings.timeEpsilon > 0 &&
        Math.fround(settings.timeEpsilon) > 0),
    'timeEpsilon must be positive and finite in float32'
  );
  validateCapacity(id, 'minimumSamples', settings.minimumSamples, 1, MAXIMUM_MINIMUM_SAMPLES);
  warnGPUShortLongitudeLatitudeDistance(id, settings.spatialContext, settings.epsilon);
  target.set([
    ...settings.bounds,
    settings.epsilon,
    settings.minimumSamples,
    settings.timeEpsilon ?? 0,
    0
  ]);
  return target;
}
