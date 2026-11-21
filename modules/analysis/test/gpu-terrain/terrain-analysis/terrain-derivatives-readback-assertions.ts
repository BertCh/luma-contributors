// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export function expectDerivativeArraysClose(
  actual: number[],
  expected: number[],
  angular: boolean = false
): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index])).toBe(true);
      continue;
    }
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(Number.isFinite(value)).toBe(true);
    if (angular) {
      if (value === -1) {
        expect(actual[index]).toBe(-1); // Flat-cell sentinel is not an angle.
        continue;
      }
      expect(actual[index]).toBeGreaterThanOrEqual(0);
      expect(actual[index]).toBeLessThan(360);
    }
    let difference = Math.abs(actual[index] - value);
    if (angular) {
      difference = Math.min(difference, 360 - difference);
    }
    expect(difference).toBeLessThan(Math.max(5e-4, Math.abs(value) * 1e-4));
  }
}
