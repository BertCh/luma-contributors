// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCPUHostOffsets} from '../../analysis-core/operations/compact-host-offsets';

/** One float32 point represented without a further rounding. @internal */
export type CanonicalPoint = readonly [number, number];

/** Numeric lexicographic comparison; signed zeros compare equal. @internal */
export function compareCanonicalPoints(first: CanonicalPoint, second: CanonicalPoint): number {
  return first[0] < second[0]
    ? -1
    : first[0] > second[0]
      ? 1
      : first[1] < second[1]
        ? -1
        : first[1] > second[1]
          ? 1
          : 0;
}

/** Exact extreme-vertex winding for finite float32 coordinates, including degenerate rings. @internal */
export function computeCanonicalWinding(points: readonly CanonicalPoint[]): number {
  if (points.length < 3) return 0;
  let extreme = 0;
  for (let index = 1; index < points.length; index++)
    if (compareCanonicalPoints(points[index], points[extreme]) < 0) extreme = index;
  const center = points[extreme];
  let previous = center;
  let next = center;
  for (let step = 1; step < points.length; step++) {
    if (compareCanonicalPoints(previous, center) === 0)
      previous = points[(extreme + points.length - step) % points.length];
    if (compareCanonicalPoints(next, center) === 0) next = points[(extreme + step) % points.length];
    if (
      compareCanonicalPoints(previous, center) !== 0 &&
      compareCanonicalPoints(next, center) !== 0
    )
      break;
  }
  // Every finite float32 is an integer multiple of 2^-149. Scaling by this exact power
  // of two and converting to BigInt gives an unbounded exact predicate, even for slivers.
  const coordinateScale = 2 ** 149;
  const previousX = BigInt(previous[0] * coordinateScale);
  const previousY = BigInt(previous[1] * coordinateScale);
  const centerX = BigInt(center[0] * coordinateScale);
  const centerY = BigInt(center[1] * coordinateScale);
  const nextX = BigInt(next[0] * coordinateScale);
  const nextY = BigInt(next[1] * coordinateScale);
  const determinant =
    (centerX - previousX) * (nextY - previousY) - (centerY - previousY) * (nextX - previousX);
  return determinant < 0n ? -1 : determinant > 0n ? 1 : 0;
}

/** Checks a compact host hierarchy level, including exact child coverage and empty parents. @internal */
export function validateCanonicalOffsets(name: string, offsets: Uint32Array, limit: number): void {
  validateCPUHostOffsets('validateCanonicalOffsets', name, offsets, limit);
}
