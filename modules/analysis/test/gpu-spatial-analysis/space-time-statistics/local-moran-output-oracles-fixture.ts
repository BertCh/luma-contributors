// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export const MORAN_OPTIONAL = [
  'analysisCount',
  'significant',
  'quadrants',
  'adjustedPValues',
  'permutationCounts',
  'islandCount',
  'invalidCount'
];
export const MORAN_CASES = [
  MORAN_OPTIONAL,
  ...MORAN_OPTIONAL.flatMap(name => [[name], MORAN_OPTIONAL.filter(field => field !== name)])
];
export const TIED_CSR = {
  offsets: [0, 1, 2, 2, 2],
  neighbors: [1, 0],
  weights: [1, 1],
  distances: []
};
export const CYCLE_CSR = {
  offsets: [0, 1, 2, 3, 3],
  neighbors: [1, 2, 0],
  weights: [1, 1, 1],
  distances: []
};

export function expectMoranOracle(
  result: Record<string, number[]>,
  cycle = false,
  significanceLevel = 0.75
): void {
  const missing = 0xffffffff;
  // Tied fixture: finite row 2 is an island, row 3 is missing. Excluding islands leaves two
  // rows [1,3]; each conditional null can draw only the other row. Every statistic ties, so
  // h=3 is reached in exactly 3 draws, p=h/L=1, BY=1, and scaled I=-1/2 for both rows.
  // Cycle fixture: independent oracle.py gives seed=11 draws/exceedances and exact-rational BY.
  const expected: Record<string, number[]> = cycle
    ? {
        exceedances: [0, 7, 7, missing],
        permutationCounts: [7, 7, 7, missing],
        pseudoPValues: [0.125, 1, 1, NaN],
        adjustedPValues: [0.6875, 1, 1, NaN],
        significant: [significanceLevel >= 0.6875 ? 1 : 0, 0, 0, 0],
        quadrants: [3, 2, 4, 0],
        analysisCount: [3],
        islandCount: [0],
        invalidCount: [0],
        overflow: [0],
        observed: [0.19047619047619047, -0.23809523809523808, -0.9523809523809523, NaN]
      }
    : {
        exceedances: [3, 3, missing, missing],
        permutationCounts: [3, 3, missing, missing],
        pseudoPValues: [1, 1, NaN, NaN],
        adjustedPValues: [1, 1, NaN, NaN],
        significant: [0, 0, 0, 0],
        quadrants: [2, 4, 0, 0],
        analysisCount: [2],
        islandCount: [1],
        invalidCount: [0],
        overflow: [0],
        observed: [-0.5, -0.5, NaN, NaN]
      };
  for (const [name, actual] of Object.entries(result)) {
    expect(actual.length, name).toBe(expected[name].length);
    for (const [index, value] of expected[name].entries()) {
      if (Number.isNaN(value)) expect(Number.isNaN(actual[index]), `${name}[${index}]`).toBe(true);
      else if (name === 'observed') {
        expect(Number.isFinite(actual[index])).toBe(true);
        // Small centered integer values, fewer than ten f32 operations: 2e-6 absolute.
        expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-6);
      } else expect(actual[index], `${name}[${index}]`).toBe(value);
    }
  }
}
