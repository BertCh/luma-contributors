// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {buildKeplerClusterIndex, queryKeplerClusterRanges} from '../src/cluster/kepler-cluster';
import {
  KEPLER_CLUSTER_BUDGET_MESSAGE,
  buildKeplerCompactClusterIndex,
  queryKeplerCompactClusterRanges
} from '../src/cluster/kepler-compact-cluster';

function makePositions(rowCount: number, spreadLongitude: number, spreadLatitude: number) {
  const positions = new Float32Array(rowCount * 2);
  let seed = 987654321;
  const next = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  for (let row = 0; row < rowCount; row++) {
    // Every 50th row duplicates the previous position; a few rows are non finite.
    if (row % 50 === 49) {
      positions[row * 2] = positions[row * 2 - 2];
      positions[row * 2 + 1] = positions[row * 2 - 1];
      continue;
    }
    positions[row * 2] = -spreadLongitude + next() * 2 * spreadLongitude;
    positions[row * 2 + 1] = -spreadLatitude + next() * 2 * spreadLatitude;
  }
  positions[10] = Number.NaN;
  return positions;
}

describe('Kepler compact cluster hierarchy', () => {
  for (const [label, spreadLongitude, spreadLatitude] of [
    ['dense city', 0.3, 0.2],
    ['global', 179, 80]
  ] as const) {
    it(`matches the supercluster-backed query exactly for ${label} data`, () => {
      const rowCount = 12000;
      const positions = makePositions(rowCount, spreadLongitude, spreadLatitude);
      const selectionMask = Uint32Array.from({length: rowCount}, (_, row) =>
        row % 9 === 0 ? 0 : 1
      );
      for (const mask of [undefined, selectionMask]) {
        const reference = buildKeplerClusterIndex({positions, selectionMask: mask}, 40);
        const compact = buildKeplerCompactClusterIndex({positions, selectionMask: mask}, 40);
        for (const [zoom, longitude] of [
          [0, 0],
          [2, 100],
          [5, -120],
          [8, 0.1],
          [11, 0.05],
          [14, 0],
          [19, 0.1],
          [20, 0]
        ] as const) {
          const view = {longitude, latitude: 0, zoom, width: 1000, height: 700};
          for (const minimum of [1, 3]) {
            const expected = queryKeplerClusterRanges(reference, view, minimum);
            const actual = queryKeplerCompactClusterRanges(compact, view, minimum);
            expect(actual.clusterCount).toBe(expected.clusterCount);
            expect(Array.from(actual.offsets)).toEqual(Array.from(expected.offsets));
            expect(Array.from(actual.centers)).toEqual(Array.from(expected.centers));
            // Member rows must match in merge order, not only as sets.
            expect(Array.from(actual.rows)).toEqual(Array.from(expected.rows));
          }
        }
      }
    });
  }

  it('retains far less than 512 bytes per row and enforces the byte budget', () => {
    const rowCount = 50000;
    const positions = makePositions(rowCount, 0.5, 0.3);
    const compact = buildKeplerCompactClusterIndex({positions}, 40);
    expect(compact.byteLength / rowCount).toBeLessThan(400);
    expect(() => buildKeplerCompactClusterIndex({positions}, 40, rowCount * 8)).toThrow(
      KEPLER_CLUSTER_BUDGET_MESSAGE
    );
  });

  for (const [label, spreadLongitude, spreadLatitude] of [
    ['dense city', 0.3, 0.2],
    ['global', 179, 80],
    ['sparse', 179, 80]
  ] as const) {
    it(`builds coarse levels on demand with identical output for ${label} data`, () => {
      const rowCount = label === 'sparse' ? 60 : 8000;
      const positions = makePositions(rowCount, spreadLongitude, spreadLatitude);
      const reference = buildKeplerClusterIndex({positions}, 40);
      // Only zoom 12 and finer exist after the build; each query below extends the index.
      const lazy = buildKeplerCompactClusterIndex({positions}, 40, Infinity, 12);
      const initialBytes = lazy.byteLength;
      expect(lazy.levels[12]).toBeDefined();
      for (const zoom of [14, 12, 9, 5, 0]) {
        const view = {longitude: 0.1, latitude: 0, zoom, width: 1000, height: 700};
        const expected = queryKeplerClusterRanges(reference, view, 1);
        const actual = queryKeplerCompactClusterRanges(lazy, view, 1);
        expect(Array.from(actual.offsets)).toEqual(Array.from(expected.offsets));
        expect(Array.from(actual.centers)).toEqual(Array.from(expected.centers));
        expect(Array.from(actual.rows)).toEqual(Array.from(expected.rows));
      }
      // Zooming out built the missing coarse levels (dense data) or shared the leaf level (sparse).
      expect(lazy.byteLength).toBeGreaterThanOrEqual(initialBytes);
      if (label !== 'sparse') expect(lazy.byteLength).toBeGreaterThan(initialBytes);
    });
  }

  it('reports the byte budget when a coarser level no longer fits', () => {
    const rowCount = 20000;
    const positions = makePositions(rowCount, 0.5, 0.3);
    const full = buildKeplerCompactClusterIndex({positions}, 40);
    const fineOnly = buildKeplerCompactClusterIndex({positions}, 40, Infinity, 20);
    const budget = Math.floor((fineOnly.byteLength + full.byteLength) / 2);
    const lazy = buildKeplerCompactClusterIndex({positions}, 40, budget, 20);
    const view = {longitude: 0, latitude: 0, zoom: 0, width: 1000, height: 700};
    expect(() => queryKeplerCompactClusterRanges(lazy, view, 1)).toThrow(
      KEPLER_CLUSTER_BUDGET_MESSAGE
    );
  });
});
