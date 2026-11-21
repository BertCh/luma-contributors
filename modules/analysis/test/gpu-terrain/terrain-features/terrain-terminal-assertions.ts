// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export function expectProfileSingletonStatus(
  rows: ArrayLike<number>,
  terminal: 'converged' | 'iterationLimitReached',
  maximumSuppressionIterations: 0 | 16
): void {
  // With a candidate, zero rounds leave it undecided; sixteen rounds complete suppression.
  const converged = maximumSuppressionIterations === 16;
  expect(Array.from(rows)).toEqual([Number(terminal === 'converged' ? converged : !converged)]);
}

export function expectSingleProfilePeak(
  rows: {
    mask: ArrayLike<number>;
    count: ArrayLike<number>;
    ids: ArrayLike<number>;
    prominence: ArrayLike<number>;
  },
  expectedMask: readonly number[]
): void {
  expect(Array.from(rows.mask)).toEqual(expectedMask);
  expect(Array.from(rows.count)).toEqual([1]);
  expect(Array.from(rows.ids)).toEqual([1]);
  expect(rows.prominence.length).toBe(expectedMask.length);
  expect(Number.isFinite(rows.prominence[1])).toBe(true);
  expect(rows.prominence[1]).toBe(1);
}

export function expectCapturedSolarAltitude(rows: ArrayLike<number>): void {
  expect(rows.length).toBe(1);
  // NREL SPA Appendix A, allowing NOAA's standard-refraction model error.
  expect(Number.isFinite(rows[0])).toBe(true);
  expect(Math.abs(rows[0] - 39.88838)).toBeLessThan(0.01);
}

export function expectSolarAngle(rows: ArrayLike<number>, expected: number): void {
  expect(rows.length).toBe(1);
  expect(Number.isFinite(rows[0])).toBe(true);
  expect(Math.abs(rows[0] - expected)).toBeLessThan(0.05);
}

export function expectFlatRelief(rows: ArrayLike<number>, expected: number): void {
  expect(rows.length).toBe(9);
  for (const value of Array.from(rows)) {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - expected)).toBeLessThan(1e-6);
  }
}

export function expectReliefColor(rows: ArrayLike<number>): void {
  expect(rows.length).toBe(9);
  for (const value of Array.from(rows)) {
    // WGSL pack4x8unorm permits either rounding at a half-step.
    const expected = [51, 26, 13, 255];
    for (let channel = 0; channel < 4; channel++)
      expect(Math.abs(((value >>> (channel * 8)) & 255) - expected[channel])).toBeLessThanOrEqual(
        channel === 3 ? 0 : 1
      );
  }
}
