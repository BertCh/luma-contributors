// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Preserve the cocircular device observation before checking any analytic expectation. */
export function expectCocircularVoronoiReadback(
  fixture: {name: string; triangles: number[]; diagonal: string},
  result: {
    written: number;
    required: number;
    invalidCount: number;
    segments: number[];
    siteIds: number[];
  }
): void {
  const rows = Array.from({length: result.written}, (_, row) => ({
    row,
    coordinates: result.segments.slice(4 * row, 4 * row + 4),
    ids: result.siteIds
      .slice(2 * row, 2 * row + 2)
      .sort((first, second) => first - second)
      .join(',')
  }));
  const zeroRows = rows.filter(
    ({coordinates}) =>
      coordinates.length === 4 &&
      coordinates[0] === coordinates[2] &&
      coordinates[1] === coordinates[3]
  );
  // Capture raw evidence even if the first count, finite-coordinate or bisector check fails.
  // Preserve nonfinite values by name; ordinary JSON serialization would erase them as null.
  console.info(
    'GEO-123 cocircular active-prefix probe',
    JSON.stringify(
      {
        triangulation: fixture.name,
        triangles: fixture.triangles,
        written: result.written,
        required: result.required,
        invalidCount: result.invalidCount,
        siteIds: result.siteIds,
        segments: result.segments,
        zeroRows
      },
      (_key, value: unknown) =>
        typeof value === 'number' && !Number.isFinite(value) ? String(value) : value
    )
  );

  expect(result.invalidCount).toBe(0);
  expect(result.written).toBe(result.required);
  expect(result.segments.length).toBe(4 * result.written);
  expect(result.siteIds.length).toBe(2 * result.written);
  const expected: Record<string, number[]> = {
    '0,1': [0.5, 0.5, 0.5, -1],
    '1,2': [0.5, 0.5, 2, 0.5],
    '2,3': [0.5, 0.5, 0.5, 2],
    '0,3': [0.5, 0.5, -1, 0.5]
  };
  const positivePairs: string[] = [];
  for (const {coordinates, ids} of rows) {
    coordinates.forEach(value => expect(Number.isFinite(value)).toBe(true));
    if (coordinates[0] !== coordinates[2] || coordinates[1] !== coordinates[3]) {
      positivePairs.push(ids);
      expect(coordinates).toEqual(expected[ids]);
    }
  }
  expect(positivePairs.sort()).toEqual(['0,1', '0,3', '1,2', '2,3']);
  // Diagnostic probe: no suppression or row-identity change is inferred from a zero row.
  expect(zeroRows.length).toBeLessThanOrEqual(1);
  for (const zero of zeroRows) {
    expect(zero.coordinates).toEqual([0.5, 0.5, 0.5, 0.5]);
    expect(zero.ids).toBe(fixture.diagonal);
  }
  expect(result.written).toBe(4 + zeroRows.length);
}
