// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPUSpatialContext,
  GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';

/**
 * Smallest angular tolerance (radians) the spherical predicates accept: f32 unit vectors carry
 * about 6e-8 of rounding per component, so a tighter band is below the noise of the input.
 */
export const GPU_SPHERICAL_MINIMUM_TOLERANCE = 1e-7;

/** Largest angular tolerance (radians) of the containment and convex-clipping predicates, about 64 km. */
export const GPU_SPHERICAL_MAXIMUM_TOLERANCE = 1e-2;

/** Largest angular tolerance (radians) of the arc overlay and S2 polygon covering, about 6 km. */
export const GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE = 1e-3;

/** Fewest vertices of a convex spherical polygon: a spherical triangle. */
export const GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES = 3;

/**
 * Most vertices of one convex spherical polygon. The kernels keep the polygon in private
 * fixed-size arrays, and the clip of two such polygons has at most twice as many vertices.
 */
export const GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES = 32;

/**
 * Largest per-row output `capacity` of `GPUSphericalConvexIntersection`: the intersection of two
 * convex polygons has at most the sum of their vertex counts, 2 * 32.
 */
export const GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY = 64;

/**
 * Largest total vertex count of the two operands of `GPUSphericalArcOverlay`. The arrangement
 * runs in one invocation with O(n^3) work and private scratch arrays sized for this many vertices.
 */
export const GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_INPUT_VERTICES = 128;

/**
 * Largest output vertex capacity of `GPUSphericalArcOverlay`: the private segment scratch holds
 * seven floats per output vertex and is allocated per kernel.
 */
export const GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES = 1024;

/**
 * Smallest horizon cosine of `GPUSphericalPolygonOverlay`: the gnomonic chart scales by the
 * inverse cosine, so a vertex closer to the horizon than this leaves f32 chart coordinates unusable.
 */
export const GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE = 1e-3;

/** Largest `maximumLevel` of the S2 polygon and cap coverings in their level-synchronous form. */
export const GPU_S2_COVERING_MAXIMUM_LEVEL = 15;

/**
 * Support declaration of the spherical family. The classes read unit direction vectors, which are
 * the embedding of longitude-latitude positions on the sphere, and join their edges by minor
 * great-circle arcs. Planar contexts are rejected; they measure no metric lengths, so only the
 * `'none'` and `'great-circle'` metrics apply.
 */
export const GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'great-circle'] as const),
  edgeModels: Object.freeze(['great-circle'] as const)
});

/** The context the spherical family assumes when none is given: great-circle edges, no metric. */
export const GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT: GPUSpatialContext = Object.freeze({
  coordinateSpace: 'longitude-latitude',
  metric: 'none',
  edgeModel: 'great-circle'
});

/**
 * Throws unless a tolerance prop is a finite number in `[minimum, maximum]`, compared as the
 * float64 prop value (not rounded to float32), so the bounds themselves are accepted.
 *
 * @param id Contributor ID, for the message.
 * @param value Tolerance in radians.
 * @param minimum Smallest accepted tolerance.
 * @param maximum Largest accepted tolerance.
 * @param name Prop name, for the message.
 */
export function validateAngularTolerance(
  id: string,
  value: number,
  minimum: number,
  maximum: number,
  name: string = 'tolerance'
): void {
  validateRequirement(
    id,
    Number.isFinite(value) && value >= minimum && value <= maximum,
    `${name} must be a finite number of radians in [${minimum}, ${maximum}]`
  );
}
