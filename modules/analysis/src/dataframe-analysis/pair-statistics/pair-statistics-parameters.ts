// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Rectangle bounds or the bounding box of a compile-time polygon study window. */
export type GPUPairStatisticsBounds = readonly [number, number, number, number] | 'study-window';

/**
 * Validates the bounds and maximum distance shared by the pair-statistics parameter packers.
 *
 * @internal
 */
export function validatePairStatisticsBounds<Bounds extends GPUPairStatisticsBounds>(
  name: string,
  bounds: Bounds,
  maximumDistance: number
): void {
  if (bounds !== 'study-window' && (typeof bounds === 'string' || bounds.length !== 4)) {
    throw new Error(`${name} bounds must be [minX, minY, maxX, maxY] or 'study-window'`);
  }
  for (const value of [...(bounds === 'study-window' ? [] : bounds), maximumDistance]) {
    if (!Number.isFinite(value)) {
      throw new Error(`${name} bounds and maximumDistance must be finite`);
    }
  }
  if (bounds !== 'study-window' && (bounds[2] < bounds[0] || bounds[3] < bounds[1])) {
    throw new Error(`${name} bounds must satisfy minX <= maxX and minY <= maxY`);
  }
  if (maximumDistance <= 0) {
    throw new Error(`${name} maximumDistance must be positive`);
  }
  // The packers return float32: rounding keeps the non-strict bound order but can overflow a
  // finite value to infinity or flush a positive distance to zero.
  if (bounds !== 'study-window' && !bounds.every(value => Number.isFinite(Math.fround(value)))) {
    throw new Error(`${name} bounds must be finite in float32 (magnitude below 3.4e38)`);
  }
  validatePositiveFloat32(name, 'maximumDistance', maximumDistance);
}

/**
 * Throws unless `value` stays positive and finite once rounded to float32 (`Math.fround`), the
 * representation of every packed pair-statistics parameter.
 *
 * @internal
 */
export function validatePositiveFloat32(name: string, label: string, value: number): void {
  const packed = Math.fround(value);
  if (!(Number.isFinite(packed) && packed > 0)) {
    throw new Error(
      `${name} ${label} must be positive and finite in float32, got ${value} which packs to ${packed}`
    );
  }
}

/**
 * Throws unless `value` stays finite once rounded to float32 (`Math.fround`).
 *
 * @internal
 */
export function validateFiniteFloat32(name: string, label: string, value: number): void {
  if (!Number.isFinite(Math.fround(value))) {
    throw new Error(
      `${name} ${label} must be finite in float32 (magnitude below 3.4e38), got ${value}`
    );
  }
}
