// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {fitPointCloudSamplingCovariance} from '@luma-contributors/analysis/pointcloud-analysis';
import oracle from '../oracles/fixtures/pointcloud-covariance-fit.json';

it('fits a Gaussian covariance against independent NumPy vectorized least squares', () => {
  const actual = fitPointCloudSamplingCovariance(
    oracle.positions,
    oracle.residuals,
    oracle.correlationLengths
  );
  expect(actual.correlationLength).toBe(oracle.expected.correlationLength);
  expect(actual.variance).toBeCloseTo(oracle.expected.variance, 12);
  expect(actual.meanSquaredError).toBeCloseTo(oracle.expected.meanSquaredError, 12);
});

it('resolves zero-residual ties by the smallest length and fits the two-point closed form', () => {
  expect(fitPointCloudSamplingCovariance([0, 0, 0, 1, 0, 0], [2, 2], [2, 1])).toEqual({
    variance: 0,
    correlationLength: 1,
    meanSquaredError: 0
  });
  expect(fitPointCloudSamplingCovariance([0, 0, 0, 1, 0, 0], [0, 1], [1]).variance).toBeCloseTo(
    0.5 / (1 - Math.exp(-0.5)),
    12
  );
});

it('rejects shape, finite-number, zero-separation and named work-limit violations', () => {
  for (const [positions, residuals, lengths] of [
    [[0, 0, 0], [1, 2], [1]],
    [[0, 0, 0, 0, 0, 0], [1, 2], [1]],
    [[0, 0, 0, 1, 0, 0], [NaN, 2], [1]],
    [[0, 0, 0, Infinity, 0, 0], [1, 2], [1]],
    [[0, 0, 0, 1, 0, 0], [1, 2], []],
    [[0, 0, 0, 1, 0, 0], [1, 2], [0]],
    [[0, 0, 0, 1, 0, 0], [1, 2], new Array(33).fill(1)],
    [new Array(4097 * 3).fill(0), new Array(4097).fill(1), [1]]
  ]) {
    expect(() => fitPointCloudSamplingCovariance(positions, residuals, lengths)).toThrow(
      /^fit-pointcloud-sampling-covariance/
    );
  }
});

it('preserves small residual errors when selecting large Gaussian correlation lengths', () => {
  // Independent NumPy lstsq + direct residuals selects 1e8, with variance about 1e16.
  // The smaller lengths have errors about 8.33e-18 and 8.33e-26, not zero.
  const positions = [0, 0, 0, 1, 0, 0, 2, 0, 0];
  const residuals = [-1, 0, 1];
  const fits = [1e4, 1e6, 1e8].map(length =>
    fitPointCloudSamplingCovariance(positions, residuals, [length])
  );
  expect(fits[0].meanSquaredError).toBeGreaterThan(8e-18);
  expect(fits[0].meanSquaredError).toBeLessThan(9e-18);
  expect(fits[1].meanSquaredError).toBeGreaterThan(8e-26);
  expect(fits[1].meanSquaredError).toBeLessThan(9e-26);
  expect(fits[2].meanSquaredError).toBeLessThan(1e-28);
  const actual = fitPointCloudSamplingCovariance(positions, residuals, [1e4, 1e6, 1e8]);
  expect(actual.correlationLength).toBe(1e8);
  expect(actual.variance / 1e16).toBeCloseTo(1, 14);
});

it('keeps the mathematical two-point tie for nonzero residuals', () => {
  // One nonzero Gaussian design value fits the single observed semivariance exactly.
  for (const correlationLengths of [
    [0.13, 0.14],
    [0.14, 0.13]
  ]) {
    const actual = fitPointCloudSamplingCovariance([0, 0, 0, 1, 0, 0], [0, 1], correlationLengths);
    expect(actual.correlationLength).toBe(0.13);
    expect(actual.meanSquaredError).toBe(0);
  }
});

for (const correlationLengths of [
  [0.5, 10],
  [10, 0.5]
]) {
  it(`resolves the single-pair exact fit analytically for candidate order ${correlationLengths}`, () => {
    // A single observed semivariance fixes variance = 0.5 / (1 - exp(-2)) at length 0.5.
    const actual = fitPointCloudSamplingCovariance([0, 0, 0, 1, 0, 0], [0, 1], correlationLengths);
    expect(actual.correlationLength).toBe(0.5);
    expect(actual.variance).toBeCloseTo(0.5782588213748328, 14);
    expect(actual.meanSquaredError).toBe(0);
  });
}
