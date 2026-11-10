// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance
} from '../../analysis-core/contracts/spatial-context';

/** Number of float32 elements in a spatial-clustering parameter buffer. */
export const GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH = 8;

/** Label and root value of points that belong to no cluster. */
export const GPU_SPATIAL_CLUSTERING_NOISE = 0xffffffff;

/** Largest accepted `minimumPoints`, the largest integer a float32 holds exactly. */
const MAXIMUM_MINIMUM_POINTS = 2 ** 24;

/**
 * CPU settings of the per-frame parameters of one {@link GPUSpatialClustering}.
 *
 * Distances use planar position units or the longitude/latitude context's length unit.
 */
export type GPUSpatialClusteringSettings = {
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
  /** Neighbor radius. Two points are neighbors when their distance is `<= epsilon`. Positive. */
  epsilon: number;
  /**
   * Integer `>= 1`. A point is a core point when its epsilon-neighborhood, including the point
   * itself, holds at least this many points.
   */
  minimumPoints: number;
  /**
   * Space-time clustering (ST-DBSCAN): two points are neighbors only when, besides the spatial
   * test, their relative times differ by at most this many ticks of the contributor's
   * `temporalContext` unit. Positive. Required when the contributor has `times`, and ignored
   * otherwise. Stored at index 6.
   */
  timeEpsilon?: number;
};

/**
 * Packs spatial-clustering parameters into the 8-element float32 layout read by
 * `GPUSpatialClustering`.
 *
 * Layout: `[minX, minY, maxX, maxY, epsilon, minimumPoints, timeEpsilon, 0]`; `timeEpsilon` is 0
 * when omitted.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `epsilon <= 0`, `timeEpsilon` is given and not positive and
 * finite, `minimumPoints` is not an integer in `[1, 2^24]`, or `target` is too short.
 */
export function getGPUSpatialClusteringParameterValues(
  settings: GPUSpatialClusteringSettings,
  target: Float32Array = new Float32Array(GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH) {
    throw new Error(
      `Spatial clustering target must hold ${GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH} elements`
    );
  }
  for (const value of [...settings.bounds, settings.epsilon]) {
    if (!Number.isFinite(value)) {
      throw new Error('Spatial clustering bounds and epsilon must be finite');
    }
  }
  if (settings.epsilon <= 0) {
    throw new Error('Spatial clustering epsilon must be positive');
  }
  if (
    settings.timeEpsilon !== undefined &&
    (!Number.isFinite(settings.timeEpsilon) || settings.timeEpsilon <= 0)
  ) {
    throw new Error('Spatial clustering timeEpsilon must be positive and finite');
  }
  if (
    !Number.isInteger(settings.minimumPoints) ||
    settings.minimumPoints < 1 ||
    settings.minimumPoints > MAXIMUM_MINIMUM_POINTS
  ) {
    throw new Error(
      `Spatial clustering minimumPoints must be an integer from 1 to ${MAXIMUM_MINIMUM_POINTS}`
    );
  }
  warnGPUShortLongitudeLatitudeDistance(
    'GPUSpatialClustering',
    settings.spatialContext,
    settings.epsilon
  );
  target.set([
    ...settings.bounds,
    settings.epsilon,
    settings.minimumPoints,
    settings.timeEpsilon ?? 0,
    0
  ]);
  return target;
}
