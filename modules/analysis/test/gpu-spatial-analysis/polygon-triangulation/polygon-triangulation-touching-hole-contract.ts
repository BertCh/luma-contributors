// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {checkTriangulation, trianglesOverlap} from './polygon-triangulation-monotone-cases';
import type {ModelPoint} from './polygon-triangulation-monotone-model';

// Hand area: 10 * 10 - 2 * 2 - 2 * 2 = 92. Independently checked with
// /usr/bin/python3: Polygon(rings[0], rings[1:]).is_valid == True, .area == 92.
export const TOUCHING_HOLE_RINGS: ModelPoint[][] = [
  [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10]
  ],
  [
    [2, 2],
    [2, 4],
    [4, 4],
    [4, 2]
  ],
  [
    [4, 4],
    [4, 6],
    [6, 6],
    [6, 4]
  ]
];

export function expectTouchingHoleTiling(triangles: [number, number, number][]): void {
  expect(triangles).toHaveLength(14);
  const check = checkTriangulation(TOUCHING_HOLE_RINGS, triangles);
  expect(Number.isFinite(check.area)).toBe(true);
  expect(check.area).toBe(92);
  expect(check.minimumSignedArea).toBe(0);
  expect(check.degenerateCount).toBe(2);
  // Directed edge cancellation proves coverage, including hole exclusion, independently of
  // decomposition order. Also check each positive-area triangle against both rectangular holes.
  expect(check.boundaryMatches).toBe(true);
  const positions = TOUCHING_HOLE_RINGS.flat();
  const degenerateRows: string[] = [];
  for (const rows of triangles) {
    const triangle = rows.map(row => positions[row]) as [ModelPoint, ModelPoint, ModelPoint];
    const [first, second, third] = triangle;
    const determinant =
      (second[0] - first[0]) * (third[1] - first[1]) -
      (second[1] - first[1]) * (third[0] - first[0]);
    expect(Number.isFinite(determinant)).toBe(true);
    if (determinant === 0) {
      degenerateRows.push(rows.join(','));
      continue;
    }
    for (const hole of TOUCHING_HOLE_RINGS.slice(1)) {
      expect(trianglesOverlap(triangle, [hole[0], hole[1], hole[2]])).toBe(false);
      expect(trianglesOverlap(triangle, [hole[0], hole[2], hole[3]])).toBe(false);
    }
  }
  // Rows 6 and 8 coincide at (4,4) but retain distinct input identities. These counted
  // triangles follow (4,6)->(4,4)->(4,4) and (6,4)->(4,4)->(4,4), not unused padding.
  // Only output row order may vary; source indices within each triple remain pinned.
  expect(degenerateRows.sort()).toEqual(['11,8,6', '9,6,8']);
}
