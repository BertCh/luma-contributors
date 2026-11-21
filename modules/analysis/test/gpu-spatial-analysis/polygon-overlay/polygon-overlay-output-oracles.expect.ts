// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Intersection of [0,4]^2 and [2,6]^2 has four boundary edges. Replacing both operands'
 * vertices by NaN leaves no valid segment, sets candidateOverflow and leaves ZERO uncertain
 * retained pairs. On every classified pair the coordinates are finite integers in [0,2^22+1].
 * Nonzero coordinate significand exponents lie in [-23,-1], so raw-product exponent spread
 * is at most 44, below the predicate's 200-bit limit. A nonzero uncertainty expectation would
 * therefore require bypassing the public snap-rounding graph, rather than an extreme world
 * coordinate. The corrected third submission must clear input loss back to zero. */
export function expectPolygonOverlayNodingDiagnostics(
  actual: Record<string, number[]>,
  usable: boolean
) {
  expect(actual['boundaryCount']).toEqual([usable ? 4 : 0]);
  expect(actual['boundaryRequiredCount']).toEqual([usable ? 4 : 0]);
  expect(actual['boundaryOverflow']).toEqual([0]);
  expect(actual['boundaryCandidateOverflow']).toEqual([usable ? 0 : 1]);
  if ('uncertainCount' in actual) expect(actual['uncertainCount']).toEqual([0]);
}

/** [0,4]^2 intersect [2,6]^2 = [2,4]^2. Bottom/left belong to right row 0;
 * top/right to left row 1. Retained surface is on the left of every directed edge. */
export function expectPolygonOverlayBoundaryOracle(actual: Record<string, number[]>) {
  const owners = Array.from({length: 4}, (_, row) => [
    ...actual['endpoints'].slice(row * 4, row * 4 + 4),
    actual['operandIds'][row],
    actual['sourceFeatureIds'][row]
  ]);
  expect(owners.sort()).toEqual(
    [
      [2, 2, 4, 2, 1, 0],
      [4, 2, 4, 4, 0, 1],
      [4, 4, 2, 4, 0, 1],
      [2, 4, 2, 2, 1, 0]
    ].sort()
  );
}
