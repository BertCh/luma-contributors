// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PreparedH3Source} from './h3-source';

/**
 * High/low Float32 position adapter for H3 cells that Float32 longitude/latitude cannot hold.
 *
 * Each canonical Float64 coordinate becomes `high = fround(value)` and `low = fround(value - high)`.
 * Deck's `project_position_vec3_f64(position, position64Low)` (project32) subtracts the viewport
 * center from the high part and adds the low part in Float32, which reconstructs roughly 48
 * mantissa bits near the viewport instead of 24. The pair maps to a `float32x2` position vector
 * plus a `float32x2` "position64Low" vector; renderer wiring (a GPU layer accepting the low
 * vector) is a separate step, see the capability gap.
 */
export type SplitCoordinates = {high: Float32Array; low: Float32Array};

export function splitFloat64Coordinates(values: ArrayLike<number>): SplitCoordinates {
  const high = new Float32Array(values.length);
  const low = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    high[index] = value;
    low[index] = value - high[index];
  }
  return {high, low};
}

/**
 * CPU reference of the shader's emulated double subtraction: `(high - centerHigh) + (low - centerLow)`
 * with every operation rounded to Float32, exactly as the GPU evaluates it.
 */
export function subtractSplitCoordinate(
  high: number,
  low: number,
  centerHigh: number,
  centerLow: number
): number {
  const highDifference = Math.fround(Math.fround(high) - Math.fround(centerHigh));
  const lowDifference = Math.fround(Math.fround(low) - Math.fround(centerLow));
  return Math.fround(highDifference + lowDifference);
}

/** Worst-case absolute error (degrees) of storing the given coordinates as a single Float32. */
export function getFloat32CoordinateError(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index++)
    maximum = Math.max(maximum, Math.abs(values[index] - Math.fround(values[index])));
  return maximum;
}

/** Whether Float32 storage is inadequate for this source and the split representation is required. */
export function requiresH3Fp64Split(source: PreparedH3Source): boolean {
  return source.precision.collapsedRows.length > 0;
}

/**
 * Ring-normalized canonical boundary coordinates (longitude unwrapped relative to each ring's
 * first vertex, like the Float32 upload) as interleaved longitude/latitude, then split.
 */
export function splitH3BoundaryPositions(
  boundaryPositions: Float64Array,
  boundaryOffsets: Uint32Array
): SplitCoordinates {
  const normalized = new Float64Array(boundaryPositions.length);
  for (let row = 0; row + 1 < boundaryOffsets.length; row++) {
    const start = boundaryOffsets[row];
    const end = boundaryOffsets[row + 1];
    if (end <= start) continue;
    const reference = boundaryPositions[start * 2];
    for (let vertex = start; vertex < end; vertex++) {
      const longitude = boundaryPositions[vertex * 2];
      const delta = longitude - reference;
      normalized[vertex * 2] =
        delta > 180 ? longitude - 360 : delta < -180 ? longitude + 360 : longitude;
      normalized[vertex * 2 + 1] = boundaryPositions[vertex * 2 + 1];
    }
  }
  return splitFloat64Coordinates(normalized);
}

/** Splits every batch's canonical cell centers (interleaved longitude/latitude). */
export function splitH3CenterPositions(source: PreparedH3Source): SplitCoordinates[] {
  return source.batches.map(batch => splitFloat64Coordinates(batch.centers));
}

/**
 * CPU reference of the H3 polygon coverage kernel in fp64 mode: scales a split coordinate about a
 * split center with an error-free two-sum, rounding every operation to Float32 like the GPU.
 */
export function scaleSplitCoordinateAboutCenter(
  high: number,
  low: number,
  centerHigh: number,
  centerLow: number,
  coverage: number
): {high: number; low: number} {
  const round = Math.fround;
  const scaled = round(
    round(round(round(high) - round(centerHigh)) + round(round(low) - round(centerLow))) *
      round(coverage)
  );
  const sum = round(round(centerHigh) + scaled);
  const virtual = round(sum - round(centerHigh));
  const error = round(round(round(centerHigh) - round(sum - virtual)) + round(scaled - virtual));
  return {high: sum, low: round(error + round(centerLow))};
}
