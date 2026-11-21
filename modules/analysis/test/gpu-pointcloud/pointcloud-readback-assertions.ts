// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Compare a numeric readback with independent expected values and an explicit error budget. */
export function expectPointCloudReadback(
  actual: readonly number[],
  expected: readonly number[],
  allowedError: (expectedValue: number) => number = () => 0
): void {
  expect(actual).toHaveLength(expected.length);
  for (const [slot, value] of actual.entries()) {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - expected[slot])).toBeLessThanOrEqual(allowedError(expected[slot]));
  }
}
