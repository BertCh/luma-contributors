// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Ordered [1-sqrt(u), sqrt(u)*(1-v), sqrt(u)*v], independently calculated
// by dot-density-output-oracles.oracle.py with integer Philox and binary64 sqrt.
export const TRIANGLE_BARYCENTRIC_WEIGHTS = [
  [0.013698102336637152, 0.344985349620352, 0.6413165480430109],
  [0.02831819620794218, 0.6642731381141669, 0.307408665677891],
  [0.3210400935087182, 0.3473267622769506, 0.3316331442143312]
];

export function matchesTriangleBarycentricOracle(actual: readonly (readonly number[])[]): boolean {
  const permutations = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0]
  ];
  return (
    actual.length === 3 &&
    permutations.some(permutation =>
      actual.every(
        (row, index) =>
          row.length === 3 &&
          row.every(
            (value, column) =>
              Number.isFinite(value) &&
              Math.abs(value - TRIANGLE_BARYCENTRIC_WEIGHTS[index][permutation[column]]) <= 2e-6
          )
      )
    )
  );
}
