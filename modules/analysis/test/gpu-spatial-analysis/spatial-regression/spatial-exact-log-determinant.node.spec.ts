// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getExactLogDeterminantWGSL,
  getSpatialExactLogDeterminant,
  getSpatialWeightsEigenvalues,
  getSymmetricEigenvalues
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-exact-log-determinant';
import {findConstantNonFiniteBitcasts} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createLatticeWeights} from './spatial-regression-diagnostics-oracle';

it('exact determinant WGSL creates nonfinite values at runtime without invalid constant expressions', () => {
  expect(findConstantNonFiniteBitcasts(getExactLogDeterminantWGSL('eigenvalues', 2))).toEqual([]);
});

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

it('rejects nonreversible weights independently of their magnitude', () => {
  for (const magnitude of [1, 1e-3, 1e-6, 1e6]) {
    expect(() =>
      getSpatialWeightsEigenvalues(
        [0, 2, 4, 6],
        [1, 2, 0, 2, 0, 1],
        [0.5, 0.5, 0.9, 0.1, 0.2, 0.8].map(value => value * magnitude),
        3
      )
    ).toThrow(/similar to a symmetric/);
  }
});

it('preserves the determinant of reversible weights under joint weight and rho rescaling', () => {
  const offsets = [0, 2, 4, 6];
  const neighbors = [1, 2, 0, 2, 0, 1];
  // W = D^-1 C, C symmetric, D = diag(2, 4, 8).
  const weights = [0.5, 1, 0.25, 0.75, 0.25, 0.375];
  const expected = Math.log(0.8125); // det(I - 0.5 W), expanded directly.
  for (const magnitude of [1, 1e-6, 1e6]) {
    const eigenvalues = getSpatialWeightsEigenvalues(
      offsets,
      neighbors,
      weights.map(value => value * magnitude),
      3
    );
    const actual = getSpatialExactLogDeterminant(0.5 / magnitude, eigenvalues);
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(expected, 12);
  }
});

it('rejects nonpositive or nonfinite similarity scales', () => {
  for (const weights of [
    [1, -1],
    [1, Infinity],
    [1, NaN],
    [1e308, 1e-308]
  ]) {
    expect(() => getSpatialWeightsEigenvalues([0, 1, 2], [1, 0], weights, 2)).toThrow(
      /positive finite/
    );
  }
});

it('computes the log absolute determinant for negative factors and singular matrices', () => {
  expect(getSpatialExactLogDeterminant(2, [1, -1])).toBeCloseTo(Math.log(3), 14);
  expect(getSpatialExactLogDeterminant(1, [1, -1])).toBe(-Infinity);
  expect(getSpatialExactLogDeterminant(2, [1, 1])).toBe(0);
});

it('retains small nonzero log determinant terms', () => {
  expect(getSpatialExactLogDeterminant(1e-20, [1])).toBeCloseTo(-1e-20, 30);
});

it('retains a nonzero determinant when the rounded product equals one (R1)', () => {
  const actual = getSpatialExactLogDeterminant(1 + 2 ** -27, [1 - 2 ** -27]);
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBeCloseTo(-54 * Math.LN2, 12);
  expect(getSpatialExactLogDeterminant(2, [0.5])).toBe(-Infinity);
});

it('preserves representable similarity scales across overflowing intermediate products (R2)', () => {
  for (const magnitude of [1, 2 ** 24]) {
    const eigenvalues = getSpatialWeightsEigenvalues(
      [0, 1, 3, 4],
      [1, 0, 2, 1],
      [1, 2 ** -1000, 1, 1].map(value => magnitude * value),
      3
    );
    const actual = getSpatialExactLogDeterminant(0.5 / magnitude, eigenvalues);
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(Math.log(0.75), 12);
  }
});

it('keeps joint rescaling invariant on a twelve-row path of float32 weights (R2)', () => {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < 12; row++) {
    if (row > 0) {
      neighbors.push(row - 1);
      weights.push(row === 11 ? 1 : 2 ** -100);
    }
    if (row < 11) {
      neighbors.push(row + 1);
      weights.push(1);
    }
    offsets.push(neighbors.length);
  }
  for (const magnitude of [1, 2 ** 24]) {
    const eigenvalues = getSpatialWeightsEigenvalues(
      offsets,
      neighbors,
      Float32Array.from(weights, value => magnitude * value),
      12
    );
    const actual = getSpatialExactLogDeterminant(0.5 / magnitude, eigenvalues);
    // Tridiagonal determinant recurrence: the first ten edge products are 2^-100,
    // below float64 spacing at one; the final edge contributes 1 - rho^2 = 3/4.
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(Math.log(0.75), 12);
  }
});

it('evaluates overflowing finite factors and gives true singularity priority in either order (R3)', () => {
  const largestPower = 2 ** 1023;
  const actual = getSpatialExactLogDeterminant(2, [largestPower]);
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBeCloseTo(1024 * Math.LN2, 12);
  expect(getSpatialExactLogDeterminant(2, [0.5, largestPower])).toBe(-Infinity);
  expect(getSpatialExactLogDeterminant(2, [largestPower, 0.5])).toBe(-Infinity);
});
