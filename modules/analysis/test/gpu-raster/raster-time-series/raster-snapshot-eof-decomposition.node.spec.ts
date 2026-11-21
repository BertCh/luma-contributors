// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPURasterSnapshotEOFDecomposition} from '../../../src/raster-analysis/raster-time-series/raster-snapshot-eof-decomposition';

// NumPy reference (/usr/bin/python3):
//   X = np.array([[1, 0.002, 0], [0, 0.8, 0.003], [0.001, 0, 0.5]]); C = X.T @ X
//   w = np.linalg.eigh(C)[0][::-1]; w / np.trace(C)
// gives the same fractions for (X * 1e-6) and (X * 1e6), within 2e-16.
const DATA = [
  [1, 0.002, 0],
  [0, 0.8, 0.003],
  [0.001, 0, 0.5]
];
const NUMPY_EXPLAINED_VARIANCE_RATIO = [
  0.5291032035760214, 0.33862586402645606, 0.1322709323975226
];
const NUMPY_SINGULAR_VALUES = [1.000006231082352, 0.8000047648433716, 0.49999391398733173];

/** Gram matrix `X' X` of `DATA * scale`, row-major. */
function getScaledGramMatrix(scale: number): Float64Array {
  const size = DATA[0].length;
  const matrix = new Float64Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      let total = 0;
      for (const cell of DATA) total += cell[row] * scale * (cell[column] * scale);
      matrix[row * size + column] = total;
    }
  }
  return matrix;
}

function expectClose(actual: ArrayLike<number>, expected: readonly number[], tolerance: number) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(Math.abs(actual[index] - expected[index])).toBeLessThan(tolerance);
  }
}

it('RT-120 EOF explained-variance fractions do not depend on the units of the data', () => {
  for (const scale of [1e-6, 1, 1e6]) {
    const decomposition = getGPURasterSnapshotEOFDecomposition(getScaledGramMatrix(scale), 3, 3);
    expectClose(decomposition.explainedVarianceRatio, NUMPY_EXPLAINED_VARIANCE_RATIO, 1e-12);
    expectClose(
      Array.from(decomposition.singularValues, value => value / scale),
      NUMPY_SINGULAR_VALUES,
      1e-6
    );
  }
});

it('RT-120 keeps a small-magnitude rank-one mode', () => {
  // Exact eigenvalues of a * [[1, -1], [-1, 1]] are [2a, 0], leading direction (1, -1) / sqrt(2).
  const amplitude = 1e-16;
  const decomposition = getGPURasterSnapshotEOFDecomposition(
    [amplitude, -amplitude, -amplitude, amplitude],
    2,
    2
  );
  expectClose(decomposition.explainedVarianceRatio, [1, 0], 1e-12);
  expectClose(decomposition.singularValues, [Math.sqrt(2e-16), 0], 1e-14);
  expectClose(decomposition.temporalModes.slice(0, 2), [Math.SQRT1_2, -Math.SQRT1_2], 1e-6);
});

// For a * [[1, 1], [1, 1]], the sum and difference directions have eigenvalues 2a and 0.
// /usr/bin/python3: eigenvalues = [2 * Fraction(scale_text), Fraction(0)];
// [float(value / sum(eigenvalues)) for value in eigenvalues] == [1.0, 0.0].
it.each([
  [1e-200, 2e-200, 0],
  [1, 2, 1.4142135381698608],
  [1e200, 2e200, Infinity]
])('W2 N2 resolves a rank-one EOF with Gram entries %s', (amplitude, variance, singularValue) => {
  const decomposition = getGPURasterSnapshotEOFDecomposition(
    [amplitude, amplitude, amplitude, amplitude],
    2,
    2
  );
  expect(Array.from(decomposition.explainedVarianceRatio)).toEqual([1, 0]);
  expect(Array.from(decomposition.explainedVariance)).toEqual([variance, 0]);
  expect(Array.from(decomposition.singularValues)).toEqual([singularValue, 0]);
  expect(decomposition.temporalModes).toHaveLength(4);
  expectClose(decomposition.temporalModes.slice(0, 2), [Math.SQRT1_2, Math.SQRT1_2], 1e-6);
});

it.each([
  [1e200, 1e-300],
  [2 ** 1023, Number.MIN_VALUE]
])('W2 N2 retains variances %s and %s of a diagonal Gram matrix', (large, small) => {
  const decomposition = getGPURasterSnapshotEOFDecomposition([large, 0, 0, small], 2, 2);
  // The diagonal entries are the exact eigenvalues, and the sample divisor is 1.
  expect(Array.from(decomposition.explainedVariance)).toEqual([large, small]);
});

it('W2 N2 preserves and orders isolated subnormal modes alongside a rotated block', () => {
  const large = 2 ** 1023;
  const offDiagonal = 2 ** 1022;
  const small = Number.MIN_VALUE;
  const matrix = [
    [large, offDiagonal, 0, 0],
    [offDiagonal, large, 0, 0],
    [0, 0, small, 0],
    [0, 0, 0, 4 * small]
  ].flat();
  const decomposition = getGPURasterSnapshotEOFDecomposition(matrix, 4, 4);
  // /usr/bin/python3: float(Fraction(value) / 3) for eigenvalues
  // (3 * 2**1022, 2**1022, 4 * Fraction(1, 2**1074), Fraction(1, 2**1074)).
  expect(Array.from(decomposition.explainedVariance)).toEqual([
    4.49423283715579e307,
    1.4980776123852632e307,
    Number.MIN_VALUE,
    0
  ]);
});

it('W2 N2 restores sample variance when the unscaled eigenvalue would overflow', () => {
  const decomposition = getGPURasterSnapshotEOFDecomposition(
    [1e308, 1e308, 0, 1e308, 1e308, 0, 0, 0, 0],
    3,
    3
  );
  // /usr/bin/python3: float(2 * Fraction('1e308') / (3 - 1)) == 1e308.
  expect(Array.from(decomposition.explainedVariance)).toEqual([1e308, 0, 0]);
  expect(Array.from(decomposition.explainedVarianceRatio)).toEqual([1, 0, 0]);
});

it('RT-120 returns a zero matrix without iterating', () => {
  const decomposition = getGPURasterSnapshotEOFDecomposition([0, 0, 0, 0], 2, 2);
  expect(Array.from(decomposition.explainedVarianceRatio)).toEqual([0, 0]);
  expect(Array.from(decomposition.singularValues)).toEqual([0, 0]);
});
