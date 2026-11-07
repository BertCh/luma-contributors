// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Local-origin helpers for point clouds.
 *
 * Absolute float32 coordinates have a 0.06 m ulp at 1e6 m (projected eastings and northings), which
 * destroys millimetre registration and change detection. Subtract a shared float64 origin on the
 * CPU before uploading, run every operation in that local frame, and convert transforms back with
 * {@link getPointCloudAbsoluteTransform}. All clouds compared or registered together must use the
 * same origin.
 */

/** Largest number of point-pair tests one dispatch of an exhaustive (non-indexed) scan may perform. */
export const GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK = 2 ** 32;

/** Midpoint of the float64 bounding box of interleaved xyz positions; `[0, 0, 0]` when empty. */
export function getPointCloudLocalOrigin(positions: ArrayLike<number>): [number, number, number] {
  if (positions.length < 3) return [0, 0, 0];
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index + 2 < positions.length; index += 3)
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], positions[index + axis]);
      maximum[axis] = Math.max(maximum[axis], positions[index + axis]);
    }
  return [0, 1, 2].map(axis => (minimum[axis] + maximum[axis]) / 2) as [number, number, number];
}

/** Float32 positions relative to `origin`; the subtraction is done in float64 before rounding. */
export function makePointCloudLocalPositions(
  positions: ArrayLike<number>,
  origin: readonly [number, number, number]
): Float32Array {
  const local = new Float32Array(positions.length);
  for (let index = 0; index < positions.length; index++)
    local[index] = positions[index] - origin[index % 3];
  return local;
}

/** Column-major rigid transform conjugated by the translation to `origin`: `T_local = S^-1 T S`. */
export function getPointCloudLocalTransform(
  absolute: readonly number[],
  origin: readonly [number, number, number]
): number[] {
  return shiftTransform(absolute, origin, -1);
}

/** Inverse of {@link getPointCloudLocalTransform}: converts a local-frame result to absolute. */
export function getPointCloudAbsoluteTransform(
  local: ArrayLike<number>,
  origin: readonly [number, number, number]
): number[] {
  return shiftTransform(Array.from(local), origin, 1);
}

function shiftTransform(
  transform: readonly number[],
  origin: readonly [number, number, number],
  sign: 1 | -1
): number[] {
  const result = Array.from(transform);
  for (let row = 0; row < 3; row++) {
    // Absolute translation = local translation - R * origin + origin (sign flips for the inverse).
    let rotated = 0;
    for (let column = 0; column < 3; column++)
      rotated += transform[column * 4 + row] * origin[column];
    result[12 + row] = transform[12 + row] + sign * (origin[row] - rotated);
  }
  return result;
}
