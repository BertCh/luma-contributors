// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Shape and untouched-axis checks for the independently specified R_z(.05) device fixture. */
export function expectExtremeICPReadback(
  transform: readonly number[],
  history: readonly number[]
): void {
  expect(transform).toHaveLength(16);
  expect(history).toHaveLength(3);
  expect(transform.every(Number.isFinite)).toBe(true);
  expect(history.every(Number.isFinite)).toBe(true);
  for (const [index, value] of [0, 0, 1].entries())
    expect(Math.abs(transform[8 + index] - value)).toBeLessThan(2e-5);
}
