// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Nonfinite vertices are rejected before pair classification, so they produce input loss,
 * not uncertain predicates. With both line endpoints nonfinite there are no retained edges.
 * Every classified endpoint is instead an integer in [0,2^22+1]: its nonzero f32 significand
 * exponent is in [-23,-1], so raw-product exponent spread is at most 44, below the exact
 * orientation accumulator's 200-bit span limit. Thus even this invalid input has counter ZERO.
 * Repeating the finite rectangle after it requires the candidate flag to clear back to ZERO. */
export function expectBufferNodingDiagnostics(actual: Record<string, number[]>, usable: boolean) {
  expect(actual['boundaryCount']).toEqual([usable ? 4 : 0]);
  expect(actual['boundaryRequiredCount']).toEqual([usable ? 4 : 0]);
  expect(actual['boundaryOverflow']).toEqual([0]);
  expect(actual['boundaryCandidateOverflow']).toEqual([usable ? 0 : 1]);
  if ('uncertainCount' in actual) expect(actual['uncertainCount']).toEqual([0]);
}

/** Hand derivation: flat buffer of (0,0)--(4,0) at d=1 is [0,4] x [-1,1].
 * Four directed edges retain the interior on the left. No crossings or uncertain predicates;
 * four noded pieces, one shell, one polygon, one feature. All coordinates are exact in f32. */
export function expectBufferOutputOracle(actual: Record<string, number[]>) {
  for (const [name, expected] of Object.entries({
    count: 1,
    overflow: 0,
    candidateOverflow: 0,
    boundaryCount: 4,
    boundaryRequiredCount: 4,
    boundaryOverflow: 0,
    boundaryCandidateOverflow: 0,
    uncertainCount: 0,
    ringCount: 1,
    requiredRingCount: 1,
    nodedSegmentRequiredCount: 4
  })) {
    if (name in actual) expect(actual[name], name).toEqual([expected]);
  }
  expect(actual['featureOffsets']).toEqual([0, 1]);
  expect(actual['polygonOffsets']).toEqual([0, 1]);
  expect(actual['ringOffsets']).toEqual([0, 4]);
  expectBufferBoundaryOracle(actual);
  const corners = Array.from({length: 4}, (_, row) =>
    actual['positions'].slice(row * 2, row * 2 + 2)
  );
  expect(corners.sort()).toEqual(
    [
      [0, -1],
      [4, -1],
      [4, 1],
      [0, 1]
    ].sort()
  );
}

/** The same four directed rectangle edges are observable before ring assembly. */
export function expectBufferBoundaryOracle(actual: Record<string, number[]>) {
  const edges = Array.from({length: 4}, (_, row) =>
    actual['endpoints'].slice(row * 4, row * 4 + 4)
  );
  expect(edges.sort()).toEqual(
    [
      [0, -1, 4, -1],
      [4, -1, 4, 1],
      [4, 1, 0, 1],
      [0, 1, 0, -1]
    ].sort()
  );
  expect(actual['operandIds']).toEqual([0, 0, 0, 0]);
  // Logical feature 0 is empty; the only line part belongs to feature 1.
  expect(actual['sourceFeatureIds']).toEqual([1, 1, 1, 1]);
}
