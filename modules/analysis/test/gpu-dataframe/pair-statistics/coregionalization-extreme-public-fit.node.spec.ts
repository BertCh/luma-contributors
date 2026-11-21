// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeCoregionalizationFitOnCPU} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';

const PARAMETERS = getGPUCoregionalizationFitParameterValues([
  {model: 'spherical', range: 1, sill: 1}
]);
const OPTIONS = {variableCount: 2, structureCount: 1, maximumIterations: 1};

function expectSuppressedCoefficients(matrices: Float32Array, loadings: Float32Array): void {
  expect(matrices).toHaveLength(4);
  expect(loadings).toHaveLength(4);
  expect(Array.from(matrices).every(Number.isNaN)).toBe(true);
  expect(Array.from(loadings).every(Number.isNaN)).toBe(true);
}

function expectProjectedCoefficients(matrices: Float32Array, loadings: Float32Array): void {
  expect(matrices).toHaveLength(4);
  expect(loadings).toHaveLength(4);
  const expected = [0.0008331019803633, 0.05, 0.05, 3.0008331019803633];
  for (let row = 0; row < 2; row++)
    for (let column = 0; column < 2; column++) {
      const index = row * 2 + column;
      const actual = matrices[index] / 1e18;
      const reconstructed =
        (loadings[row * 2] * loadings[column * 2] +
          loadings[row * 2 + 1] * loadings[column * 2 + 1]) /
        1e18;
      expect(Number.isFinite(actual)).toBe(true);
      expect(Number.isFinite(reconstructed)).toBe(true);
      expect(Math.abs(actual - expected[index])).toBeLessThan(2e-6);
      expect(Math.abs(reconstructed - expected[index])).toBeLessThan(2e-6);
    }
}

it('B4 coefficient assertions reject empty and extra arrays on both publication paths', () => {
  const nan = new Float32Array(4).fill(Number.NaN);
  for (const length of [0, 5]) {
    expect(() =>
      expectSuppressedCoefficients(new Float32Array(length).fill(Number.NaN), nan)
    ).toThrow();
    expect(() =>
      expectSuppressedCoefficients(nan, new Float32Array(length).fill(Number.NaN))
    ).toThrow();
  }
  const result = computeCoregionalizationFitOnCPU(
    {
      distances: Float32Array.of(1),
      pairCounts: Uint32Array.of(1),
      semivariances: Float32Array.of(-3e18, 1e17, 3e18),
      parameters: PARAMETERS
    },
    OPTIONS
  );
  expect(() =>
    expectProjectedCoefficients(Float32Array.of(...result.matrices, 99), result.loadings)
  ).toThrow();
  expect(() =>
    expectProjectedCoefficients(result.matrices, Float32Array.of(...result.loadings, 99))
  ).toThrow();
});

it('CORE-106 legal unit-shape lag reaches the extreme conditional matrix but cannot publish a finite residual', () => {
  const semivariances = Float32Array.of(-3e38, 1e37, 3e38);
  // One usable bin, count=1 and shape(1/1)=1: normalized weight=1, Gram=1,
  // target=[-3e38,1e37;1e37,3e38], initial other-structure contribution=0.
  // No preceding product or sum overflows; the diagonal difference in Jacobi does.
  expect(Array.from(semivariances).every(Number.isFinite)).toBe(true);
  expect(Math.fround(semivariances[2] - semivariances[0])).toBe(Number.POSITIVE_INFINITY);
  // A has eigenvalues +/-sqrt(a^2+b^2). The negative diagonal has magnitude >=3e38;
  // every PSD fit has a nonnegative diagonal, so the squared residual is >=9e76.
  expect(Math.fround(semivariances[0] ** 2)).toBe(Number.POSITIVE_INFINITY);
  const result = computeCoregionalizationFitOnCPU(
    {
      distances: Float32Array.of(1),
      pairCounts: Uint32Array.of(1),
      semivariances,
      parameters: PARAMETERS
    },
    OPTIONS
  );
  expect(Array.from(result.validity)).toEqual([0]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
  expect(Array.from(result.converged)).toEqual([0]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  expect(Array.from(result.invalidCount)).toEqual([0]);
  expectSuppressedCoefficients(result.matrices, result.loadings);
  expect(Array.from(result.diagnostics.slice(0, 2)).every(Number.isNaN)).toBe(true);
  expect(Array.from(result.diagnostics.slice(2))).toEqual([1, 1]);
});

it('CORE-106 the representable-scale public fit matches a closed-form binary64 PSD projection', () => {
  const result = computeCoregionalizationFitOnCPU(
    {
      distances: Float32Array.of(1),
      pairCounts: Uint32Array.of(1),
      semivariances: Float32Array.of(-3e18, 1e17, 3e18),
      parameters: PARAMETERS
    },
    OPTIONS
  );
  // For normalized A=[[-3,.1],[.1,3]], A_+=(A+sqrt(9.01)*I)/2.
  // Constants are the closed form evaluated in binary64; no shared eigensolver oracle.
  expect(Array.from(result.validity)).toEqual([1]);
  expect(Array.from(result.iterationCount)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([1]);
  expectProjectedCoefficients(result.matrices, result.loadings);
});
