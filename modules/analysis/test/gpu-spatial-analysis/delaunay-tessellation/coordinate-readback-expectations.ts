// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Require every expected live coordinate to be returned, finite and numerically correct. */
export function expectFiniteCoordinatePrefix(
  actual: readonly number[],
  publishedRowCount: number,
  components: number,
  expected: readonly number[],
  tolerance: number
): void {
  expect(publishedRowCount).toBe(expected.length / components);
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - expected[index])).toBeLessThan(tolerance);
  });
}
