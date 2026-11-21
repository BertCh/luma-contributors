// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Device metric vector: three exact finite determinants, then three unsupported cases. */
export function expectMetricReadback(result: readonly number[]): void {
  expect(result).toHaveLength(6);
  expect(Array.from(result.slice(0, 3)).every(Number.isFinite)).toBe(true);
  expect(Array.from(result.slice(0, 3))).toEqual([-4, 4, 0]);
  for (const value of result.slice(3)) expect(Number.isNaN(value)).toBe(true);
}
