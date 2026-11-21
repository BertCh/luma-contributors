// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Independent geometric oracle for the four-sample square enclosure lens fixture. */
export function expectLocalTessellationQuadrants(
  cells: readonly number[],
  positions: readonly number[]
): void {
  expect(cells).toHaveLength(16);
  expect(positions).toHaveLength(128);
  // Boundary sampling starts at each segment's first endpoint. For sites (-1,-1), (1,-1),
  // (1,1), (-1,1), the non-redundant bisectors are x=0 and y=0 in the [-2,2]^2 enclosure.
  const quadrants = [
    [
      [-2, -2],
      [0, -2],
      [0, 0],
      [-2, 0]
    ],
    [
      [0, -2],
      [2, -2],
      [2, 0],
      [0, 0]
    ],
    [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2]
    ],
    [
      [-2, 0],
      [0, 0],
      [0, 2],
      [-2, 2]
    ]
  ];
  const sampleRows = new Set<number>();
  let area = 0;
  for (let piece = 0; piece < 4; piece++) {
    const count = cells[piece * 4];
    expect(Number.isInteger(count)).toBe(true);
    expect(count).toBeGreaterThanOrEqual(3);
    expect(count).toBeLessThanOrEqual(16);
    expect(cells[piece * 4 + 1]).toBe(count);
    expect(cells[piece * 4 + 2]).toBe(0);
    const sampleRow = cells[piece * 4 + 3];
    expect(Number.isInteger(sampleRow)).toBe(true);
    expect(sampleRow).toBeGreaterThanOrEqual(0);
    expect(sampleRow).toBeLessThan(4);
    expect(sampleRows.has(sampleRow)).toBe(false);
    sampleRows.add(sampleRow);
    const corners: number[] = [];
    for (let vertex = 0; vertex < count; vertex++) {
      const current = (piece * 16 + vertex) * 2;
      const next = (piece * 16 + ((vertex + 1) % count)) * 2;
      const x = positions[current];
      const y = positions[current + 1];
      expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true);
      const corner = quadrants[sampleRow].findIndex(
        ([expectedX, expectedY]) =>
          Math.abs(x - expectedX) <= 1e-4 && Math.abs(y - expectedY) <= 1e-4
      );
      expect(corner).toBeGreaterThanOrEqual(0);
      // A redundant diagonal bisector can repeat a boundary corner. Collapse only consecutive
      // copies; require the same complete CCW quadrant up to a cyclic rotation.
      if (corners.at(-1) !== corner) corners.push(corner);
      area +=
        0.5 * (positions[current] * positions[next + 1] - positions[next] * positions[current + 1]);
    }
    if (corners.at(-1) === corners[0]) corners.pop();
    expect(corners).toHaveLength(4);
    expect(corners).toEqual(Array.from({length: 4}, (_, index) => (corners[0] + index) % 4));
  }
  expect(sampleRows.size).toBe(4);
  expect(Number.isFinite(area)).toBe(true);
  expect(Math.abs(area)).toBeCloseTo(16, 4);
}
