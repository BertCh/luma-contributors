// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Bounds for fitting a geographic viewport to admitted, packed Float32 XY/XYZ positions.
 * Visits every vertex in every batch, independently of primitive boundaries or source rows.
 *
 * Includes a 5% margin on each axis so boundary strokes remain inside the viewport, including
 * hosts which center latitude arithmetically rather than in Mercator space. The margin is at
 * least one Float32 relative epsilon: the source coordinate can lie outside its rounded upload.
 * Padding is clipped to longitude [-180,180] and latitude [-90,90] so native geographic fit
 * validation still accepts boundary/global extents. These are viewport bounds, not an exact
 * geometry envelope. Inputs must have canonical longitudes in [-180, 180]. Does not unwrap the
 * antimeridian; use getGeographicViewportFit for a compact center/span fit.
 */
export function getGeographicViewportBounds(
  positionBatches: Iterable<Float32Array>,
  positionSize: 2 | 3
): [number, number, number, number] | null {
  const identifier = 'GeographicViewportBounds';
  validateRequirement(
    identifier,
    positionSize === 2 || positionSize === 3,
    `positionSize received ${positionSize}; pass 2 or 3 explicitly`
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
  validateRequirement(
    identifier,
    bounds[0] >= -180 && bounds[2] <= 180,
    `longitude received ${bounds[0] < -180 ? bounds[0] : bounds[2]}; must be canonical in [-180, 180]; use getGeographicViewportFit for continuous branches`
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

/** A compact spherical fit, in degrees; center is canonical and span includes viewport padding. */
export type GeographicViewportFit = {
  longitudeCenterDegrees: number;
  longitudeSpanDegrees: number;
  southDegrees: number;
  northDegrees: number;
  wrapsAntimeridian: boolean;
};

/**
 * Fits the shortest longitude arc, accepting continuous longitude branches such as 181 degrees.
 * All XY/XYZ vertices participate; empty batches return null. Longitude span and latitude bounds
 * include the same 5%/Float32 margin as getGeographicViewportBounds. Longitude center is in
 * [-180, 180). Hosts requiring ordered canonical west <= east should use
 * getGeographicViewportBounds with canonical inputs (a world fit at the dateline); hosts with
 * center/span support can consume this fit directly without clipping a wrapped interval.
 */
export function getGeographicViewportFit(
  positionBatches: Iterable<Float32Array>,
  positionSize: 2 | 3
): GeographicViewportFit | null {
  const identifier = 'GeographicViewportFit';
  validateRequirement(
    identifier,
    positionSize === 2 || positionSize === 3,
    `positionSize received ${positionSize}; accepted values are 2, 3`
  );
  const longitudes: number[] = [];
  let south = Infinity;
  let north = -Infinity;
  let longitudeMagnitude = 0;
  for (const positions of positionBatches) {
    validateRequirement(
      identifier,
      positions.length % positionSize === 0,
      `positionBatches length ${positions.length} must be a multiple of positionSize ${positionSize}`
    );
    for (let offset = 0; offset < positions.length; offset += positionSize) {
      const longitude = positions[offset];
      const latitude = positions[offset + 1];
      validateRequirement(
        identifier,
        Number.isFinite(longitude) && Number.isFinite(latitude) && Math.abs(latitude) <= 90,
        `positionBatches XY received [${longitude}, ${latitude}]; must be finite with latitude in [-90, 90]`
      );
      longitudes.push(normalizeLongitude(longitude));
      longitudeMagnitude = Math.max(longitudeMagnitude, Math.abs(longitude));
      south = Math.min(south, latitude);
      north = Math.max(north, latitude);
    }
  }
  if (longitudes.length === 0) return null;
  longitudes.sort((left, right) => left - right);
  let largestGap = -1;
  let arcStartIndex = 0;
  for (let index = 0; index < longitudes.length; index++) {
    const next = index + 1 === longitudes.length ? longitudes[0] + 360 : longitudes[index + 1];
    const gap = next - longitudes[index];
    if (gap > largestGap) {
      largestGap = gap;
      arcStartIndex = (index + 1) % longitudes.length;
    }
  }
  const start = longitudes[arcStartIndex];
  const end = longitudes[(arcStartIndex + longitudes.length - 1) % longitudes.length];
  // Subtract endpoints directly when the selected arc does not wrap. Subtracting its
  // complementary gap from 360 would erase tiny spans and can even make them negative.
  const span = arcStartIndex === 0 ? end - start : 180 - start + (end + 180);
  const longitudeCenterDegrees = normalizeLongitude(start + span / 2);
  const longitudeMargin = Math.max(span * 0.05, longitudeMagnitude * 2 ** -23, 2 ** -149);
  const longitudeSpanDegrees = Math.min(360, span + 2 * longitudeMargin);
  const latitudeMargin = Math.max(
    (north - south) * 0.05,
    Math.max(Math.abs(south), Math.abs(north)) * 2 ** -23,
    2 ** -149
  );
  return {
    longitudeCenterDegrees,
    longitudeSpanDegrees,
    southDegrees: Math.max(-90, south - latitudeMargin),
    northDegrees: Math.min(90, north + latitudeMargin),
    wrapsAntimeridian:
      longitudeSpanDegrees === 360 ||
      longitudeCenterDegrees - longitudeSpanDegrees / 2 < -180 ||
      longitudeCenterDegrees + longitudeSpanDegrees / 2 > 180
  };
}

/** Keep canonical coordinates untouched, particularly sub-ulp offsets about Greenwich. */
function normalizeLongitude(longitude: number): number {
  if (longitude >= -180 && longitude < 180) return longitude;
  const remainder = longitude % 360;
  return remainder >= 180 ? remainder - 360 : remainder < -180 ? remainder + 360 : remainder;
}
