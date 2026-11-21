// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export const SIMILARITY_MEASURES = [
  'frechetWithin',
  'dynamicTimeWarping',
  'editDistanceOnRealSequence',
  'editDistanceWithRealPenalty',
  'longestCommonSubsequence'
] as const;
// Exhaustively enumerated paths, edit alignments and common subsequences in the adjacent
// similarity-output-oracles.oracle.py; these literals never come from a library implementation.
export const SIMILARITY_EXPECTED: Record<string, number[]> = {
  frechetWithin: [1, 0],
  dynamicTimeWarping: [4.23606797749979, 7.35917360311745],
  editDistanceOnRealSequence: [1, 2],
  editDistanceWithRealPenalty: [3.23606797749979, 7.35917360311745],
  longestCommonSubsequence: [1, 0.6666666666666666],
  maximumDistance: [4.123105625617661, 8.06225774829855],
  pairStatus: [0, 0]
};
export const SIMILARITY_OPTIONS = {
  editDistanceTolerance: 1.1,
  editDistanceGapPoint: [0, 0] as const,
  lcssTolerance: 1.1
};

export function expectSimilarityOracleOutputs(
  result: Record<string, ArrayLike<number>>,
  expectedKeys: readonly string[]
): void {
  expect(Object.keys(result).sort()).toEqual([...expectedKeys].sort());
  for (const port of expectedKeys) {
    const values = Array.from(result[port]);
    expect(values.length, port).toBe(SIMILARITY_EXPECTED[port].length);
    if (['frechetWithin', 'editDistanceOnRealSequence', 'pairStatus'].includes(port))
      expect(values, port).toEqual(SIMILARITY_EXPECTED[port]);
    else
      SIMILARITY_EXPECTED[port].forEach((expected, index) => {
        expect(Number.isFinite(values[index]), port).toBe(true);
        // At most five path steps: bound float32 square roots and accumulated costs.
        expect(Math.abs(values[index] - expected), port).toBeLessThanOrEqual(
          3e-5 * Math.max(1, Math.abs(expected))
        );
      });
  }
}
export function getSimilarityOracleInput() {
  return {
    linesA: {positions: Float32Array.of(0, 0, 2, 0, 4, 0), lineOffsets: Uint32Array.of(0, 3)},
    linesB: {
      positions: Float32Array.of(0, 1, 4, 1, 0, 1, 4, 1, 8, 1),
      lineOffsets: Uint32Array.of(0, 2, 5)
    },
    pairA: Uint32Array.of(0, 0),
    pairB: Uint32Array.of(0, 1),
    frechetEpsilon: Float32Array.of(2.5)
  };
}
