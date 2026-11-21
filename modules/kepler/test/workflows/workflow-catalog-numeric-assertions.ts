// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Expected indices must all be present; forEach on sparse readbacks can skip missing rows. */
export function expectCatalogNumericRows(
  values: readonly (string | number | boolean | null)[],
  expected: readonly number[]
): void {
  expect(values).toHaveLength(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(Number.isFinite(values[index])).toBe(true);
    expect(values[index]).toBeCloseTo(expected[index], 6);
  }
}
