// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Pointer probes shared by the weights runs: validation flags under the pointer and arrows between
// a focus row and its neighbors.

import type {TractsInput} from '../../../inputs';
import type {Point, RunGlyph} from '../../../interaction';
import {findPolygonAt} from '../../../pick';

/** Violation bits of GPUSpatialWeightsValidation, in the order a row reports them. */
const VIOLATION_LABELS: readonly (readonly [number, string])[] = [
  [1, 'bad offsets'],
  [2, 'neighbor id out of range'],
  [4, 'duplicate neighbor'],
  [8, 'self neighbor'],
  [16, 'negative weight']
];

/** The name of the first violation set in a row's flags. */
export function describeViolation(flags: number): string {
  return VIOLATION_LABELS.find(([bit]) => (flags & bit) !== 0)?.[1] ?? 'clean';
}

/** The flagged row nearest to (x, y), or -1 when no row is flagged. */
export function findNearestFlagged(
  flags: ArrayLike<number>,
  positions: Float32Array,
  x: number,
  y: number
): number {
  let nearest = -1;
  let nearestDistance = Infinity;
  for (let row = 0; row < flags.length; row++) {
    if (flags[row] === 0) continue;
    const distance = Math.hypot(positions[row * 2] - x, positions[row * 2 + 1] - y);
    if (distance < nearestDistance) {
      nearest = row;
      nearestDistance = distance;
    }
  }
  return nearest;
}

/** Arrows from `from` to each target (outgoing links), or from each target to `from` (incoming). */
export function getArrowGlyphs(
  from: Point,
  targets: readonly Point[],
  direction: 'outgoing' | 'incoming'
): RunGlyph[] {
  return targets.map(target =>
    direction === 'outgoing'
      ? {shape: 'arrow' as const, from, to: target}
      : {shape: 'arrow' as const, from: target, to: from}
  );
}

/** The tract under a world position on a map drawn at `offsetX`, or -1. */
export function findTractUnder(tracts: TractsInput, x: number, y: number, offsetX = 0): number {
  return findPolygonAt(
    tracts.vertices,
    tracts.triangleIndices,
    tracts.triangleOwners,
    x - offsetX,
    y,
    tracts.triangleCount
  );
}

/** The outline of one tract on a map drawn at `offsetX`. */
export function getTractOutline(
  tracts: TractsInput,
  tract: number,
  offsetX = 0,
  dashed = false
): RunGlyph {
  const ring = tracts.vertices.subarray(
    tracts.ringOffsets[tract] * 2,
    tracts.ringOffsets[tract + 1] * 2
  );
  return {
    shape: 'polyline',
    points:
      offsetX === 0
        ? ring
        : ring.map((value, index) => (index % 2 === 0 ? value + offsetX : value)),
    closed: true,
    dashed
  };
}

/** The contiguity neighbors of a tract. */
export function getTractNeighbors(tracts: TractsInput, tract: number): Uint32Array {
  return tracts.contiguity.neighbors.subarray(
    tracts.contiguity.offsets[tract],
    tracts.contiguity.offsets[tract + 1]
  );
}

/** Center of a tract on a map drawn at `offsetX`. */
export function getTractCenter(tracts: TractsInput, tract: number, offsetX = 0): Point {
  return [tracts.centers[tract * 2] + offsetX, tracts.centers[tract * 2 + 1]];
}

/** The weight a gaussian or bisquare kernel gives a point at `distance` for a bandwidth. */
export function getKernelWeight(
  kernel: 'gaussian' | 'bisquare',
  distance: number,
  bandwidth: number
) {
  const ratio = distance / bandwidth;
  if (kernel === 'gaussian') return Math.exp(-0.5 * ratio * ratio);
  return ratio < 1 ? (1 - ratio * ratio) ** 2 : 0;
}
