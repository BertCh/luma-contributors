// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Check the primary prediction/scalar while retaining its complete readback cardinality. */
export function expectPredictionReadback(
  values: ArrayLike<number>,
  expected: number,
  expectedLength = 1
): void {
  expect(values.length).toBe(expectedLength);
  if (Number.isNaN(expected)) expect(Number.isNaN(values[0])).toBe(true);
  else {
    expect(Number.isFinite(values[0])).toBe(true);
    expect(values[0]).toBeCloseTo(expected, 4);
  }
}
