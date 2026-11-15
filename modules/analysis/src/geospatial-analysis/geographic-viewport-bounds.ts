// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../analysis-core/kernels/gpu-contributor-utils';

/**
 * Bounds for fitting a geographic viewport to admitted, packed Float32 XY/XYZ positions.
 * Visits every vertex in every batch, independently of primitive boundaries or source rows.
 *
 * Includes a 5% margin on each axis so boundary strokes remain inside the viewport, including
 * hosts which center latitude arithmetically rather than in Mercator space. The margin is at
 * least one Float32 relative epsilon: the source coordinate can lie outside its rounded upload.
 * Padding is clipped to longitude [-180,180] and latitude [-90,90] so native geographic fit
 * validation still accepts boundary/global extents. These are viewport bounds, not an exact
 * geometry envelope. Does not unwrap the antimeridian.
 */
export function getGeographicViewportBounds(
  positionBatches: Iterable<Float32Array>,
  positionSize: 2 | 3 = 3
): [number, number, number, number] | null {
  const identifier = 'GeographicViewportBounds';
  validateRequirement(
    identifier,
    positionSize === 2 || positionSize === 3,
    'positionSize must be 2 or 3'
  );
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  let vertexCount = 0;
  for (const positions of positionBatches) {
    validateRequirement(
      identifier,
      positions.length % positionSize === 0,
      'positionBatches lengths must be multiples of positionSize'
    );
    for (let offset = 0; offset < positions.length; offset += positionSize) {
      bounds[0] = Math.min(bounds[0], positions[offset]);
      bounds[1] = Math.min(bounds[1], positions[offset + 1]);
      bounds[2] = Math.max(bounds[2], positions[offset]);
      bounds[3] = Math.max(bounds[3], positions[offset + 1]);
      vertexCount++;
    }
  }
  if (vertexCount === 0) return null;
  validateRequirement(
    identifier,
    bounds.every(Number.isFinite),
    'positionBatches XY coordinates must be finite'
  );
  for (let axis = 0; axis < 2; axis++) {
    const margin = Math.max(
      (bounds[axis + 2] - bounds[axis]) * 0.05,
      Math.max(Math.abs(bounds[axis]), Math.abs(bounds[axis + 2])) * 2 ** -23,
      2 ** -149
    );
    const geographicLimit = axis === 0 ? 180 : 90;
    bounds[axis] = Math.max(-geographicLimit, Math.min(geographicLimit, bounds[axis] - margin));
    bounds[axis + 2] = Math.max(
      -geographicLimit,
      Math.min(geographicLimit, bounds[axis + 2] + margin)
    );
  }
  return bounds;
}
