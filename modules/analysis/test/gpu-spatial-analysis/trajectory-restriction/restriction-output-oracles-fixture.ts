// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {RestrictionScene} from './trajectory-restriction-operation-cases';

export function createRestrictionOracleScene(compact = true): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(-2, 0, 10, 0, 0.5, 1.5, 1.5, 1.5),
      timestamps: Float32Array.of(0, 12, 0, 3),
      trackOffsets: Uint32Array.of(0, 2, 4),
      parameters: Float32Array.of(
        0,
        12,
        0,
        0,
        0,
        0,
        0,
        0,
        -2,
        8,
        -2,
        8,
        2,
        0,
        2,
        2,
        -1,
        6,
        -1,
        6,
        1,
        2,
        1
      )
    },
    options: {
      polygonRingVertexCounts: [4, 4],
      intervalsPerSegment: 4,
      ...(compact ? {compactCapacity: 4, mergeIntervals: true} : {})
    }
  };
}

export function expectRestrictionCompact(
  result: Record<string, ArrayLike<number> | undefined>
): void {
  // GEOS intersections from restriction-output-oracles.oracle.py, ordered by track then time.
  // Compact source rows are 0 for both split pieces, then rows 2..3 for the wholly inside track.
  for (const [port, expected] of Object.entries({
    compactIntervals: [2, 4, 8, 10, 0, 3],
    compactPositions: [0, 0, 2, 0, 6, 0, 8, 0, 0.5, 1.5, 1.5, 1.5]
  })) {
    const actual = Array.from(result[port]!);
    expected.forEach((value, index) => {
      // Fewer than 20 float32 operations per interpolation; absolute error bound 2e-5 here.
      expect(Number.isFinite(actual[index]), `${port}[${index}]`).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-5);
    });
    actual.slice(expected.length).forEach(value => expect(value).toBeNaN());
  }
  expect(Array.from(result['compactSources']!)).toEqual([0, 0, 0, 0, 2, 3, 0xffffffff, 0xffffffff]);
  // First slot: interval|temporalSupport=33; later piece: interval=1;
  // wholly inside support: interval|temporalSupport|fullTemporalSupportInside=49. Tail: 0.
  expect(Array.from(result['compactStatus']!)).toEqual([33, 1, 49, 0]);
  expect(Array.from(result['compactSummary']!)).toEqual([3, 3, 0]);
}
