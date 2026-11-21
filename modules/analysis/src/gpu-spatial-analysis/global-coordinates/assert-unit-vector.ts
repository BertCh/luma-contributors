// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {vec3} from '@math.gl/core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Strict host bound: finite `abs(length(vector) - 1) < 1e-6`. See {@link validateUnitVector}. */
export const GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE = 1e-6;

/**
 * Inclusive GPU input bound: finite `abs(dot(vector, vector) - 1) <= 1e-4`, in f32 arithmetic.
 * Used for buffer vertices/queries by convex contains/intersection, polygon contains,
 * hemisphere polygon overlay, intrinsic arc overlay and S2 polygon covering. This differs
 * intentionally from the stricter host check on supplied hemisphere/chart/reference poles.
 * The CPU arc-overlay and S2-covering twins apply this squared-length bound in binary64.
 */
export const GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE = 1e-4;

/**
 * Throws unless `abs(length(vector) - 1) < 1e-6`; equality and non-finite lengths fail.
 * Construction-time check for `hemisphereCenter`, `chartCenter` and `referencePoint` props.
 * Buffer vertices and queries use the separate inclusive squared-length GPU bound.
 *
 * @param vector Direction vector to check.
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 */
export function validateUnitVector(
  vector: ArrayLike<number>,
  id: string = 'spherical',
  name: string = 'vector'
): void {
  validateRequirement(
    id,
    Math.abs(vec3.length(vector as number[]) - 1) < GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE,
    `${name} must be a finite unit vector (length 1 within ${GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE})`
  );
}
