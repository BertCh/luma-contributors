// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSpatialExactLogDeterminant,
  getSpatialWeightsEigenvalues,
  getSymmetricEigenvalues
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-exact-log-determinant';
import {createLatticeWeights} from './spatial-regression-diagnostics-oracle';

/** Brute-force `ln|A|` by LU decomposition with partial pivoting (float64). */
function getBruteForceLogDeterminant(matrix: number[][]): number {
  const n = matrix.length;
  const a = matrix.map(row => [...row]);
  let logDeterminant = 0;
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];
    logDeterminant += Math.log(Math.abs(a[column][column]));
    for (let row = column + 1; row < n; row++) {
      const factor = a[row][column] / a[column][column];
      for (let k = column; k < n; k++) a[row][k] -= factor * a[column][k];
    }
  }
  return logDeterminant;
}

function getDenseIdentityMinus(
  rho: number,
  weights: ReturnType<typeof createLatticeWeights>,
  n: number
) {
  const matrix = Array.from({length: n}, (_, row) =>
    Array.from({length: n}, (_, column) => (row === column ? 1 : 0))
  );
  for (let row = 0; row < n; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      matrix[row][weights.neighbors[slot]] -= rho * weights.weights[slot];
    }
  }
  return matrix;
}

for (const [side, rowStandardized] of [
  [3, true],
  [6, true],
  [7, false],
  [12, true]
] as const) {
  it(`getSpatialWeightsEigenvalues gives the exact log-determinant on a ${side}x${side} lattice (${rowStandardized ? 'row-standardized' : 'binary'})`, () => {
    const weights = createLatticeWeights(side, rowStandardized);
    const n = side * side;
    const eigenvalues = getSpatialWeightsEigenvalues(
      weights.offsets,
      weights.neighbors,
      weights.weights,
      n
    );
    const maximum = Math.max(...eigenvalues.map(Math.abs));
    // trace(W) = 0 (no self links).
    expect(Math.abs(eigenvalues.reduce((sum, value) => sum + value, 0))).toBeLessThan(1e-9);
    for (const fraction of [-0.9, -0.5, -0.1, 0, 0.3, 0.7, 0.95]) {
      const rho = fraction / maximum;
      const exact = getBruteForceLogDeterminant(getDenseIdentityMinus(rho, weights, n));
      expect(getSpatialExactLogDeterminant(rho, eigenvalues)).toBeCloseTo(exact, 8);
    }
  });
}

it('getSpatialWeightsEigenvalues handles a disconnected matrix and rejects asymmetric patterns', () => {
  // Two components: a row-standardized pair and an isolated row.
  const offsets = [0, 1, 2, 2];
  const eigenvalues = getSpatialWeightsEigenvalues(offsets, [1, 0], [1, 1], 3);
  expect(Array.from(eigenvalues)).toEqual([-1, 0, 1].map(value => expect.closeTo(value, 10)));
  expect(() => getSpatialWeightsEigenvalues([0, 1, 1], [1], [1], 2)).toThrow(/asymmetric/);
  // Symmetric pattern but not reversible: a directed 3-cycle weighted around the loop.
  expect(() =>
    getSpatialWeightsEigenvalues(
      [0, 2, 4, 6],
      [1, 2, 0, 2, 0, 1],
      [0.5, 0.5, 0.9, 0.1, 0.2, 0.8],
      3
    )
  ).toThrow(/similar to a symmetric/);
});

it('getSymmetricEigenvalues matches a known spectrum', () => {
  // Path graph adjacency: eigenvalues 2 cos(k pi / (n + 1)).
  const n = 9;
  const matrix = Array.from({length: n}, (_, row) =>
    Float64Array.from({length: n}, (_, column) => (Math.abs(row - column) === 1 ? 1 : 0))
  );
  const values = getSymmetricEigenvalues(matrix);
  const expected = Array.from(
    {length: n},
    (_, index) => 2 * Math.cos(((index + 1) * Math.PI) / (n + 1))
  ).sort((a, b) => a - b);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 10));
});
