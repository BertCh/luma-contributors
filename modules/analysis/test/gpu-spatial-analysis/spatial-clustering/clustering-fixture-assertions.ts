// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Shared by the external-fixture comparison and its truncated-output negative controls. */
export function expectRelocationCenters(actual: number[], expected: number[]): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((center, index) => {
    expect(Number.isFinite(center)).toBe(true);
    expect(center).toBeCloseTo(expected[index], 10);
  });
}
