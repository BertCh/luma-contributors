// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {vec3} from '@math.gl/core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Largest deviation of a direction vector's length from one that counts as unit length. */
export const GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE = 1e-6;

/**
 * Throws unless a 3-vector has unit length within {@link GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE}.
 * Non-finite lengths fail.
 *
 * @param vector Direction vector to check.
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 */
export function assertUnitVector(
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
