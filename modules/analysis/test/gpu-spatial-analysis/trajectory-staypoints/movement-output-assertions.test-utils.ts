// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Split rows are readable up to the retained segment terminal, not the allocation capacity. */
export function expectSplitRowPrefix(
  rowIds: ArrayLike<number>,
  segmentOffsets: ArrayLike<number>,
  segmentCount: number,
  rowCapacity: number,
  expected: readonly number[]
): void {
  expect(rowIds.length).toBe(rowCapacity);
  expect(Number.isInteger(segmentCount)).toBe(true);
  expect(segmentCount).toBeGreaterThanOrEqual(0);
  expect(segmentCount).toBeLessThan(segmentOffsets.length);
  const rowCount = segmentOffsets[segmentCount];
  expect(rowCount).toBe(expected.length);
  expect(Array.from(rowIds).slice(0, rowCount)).toEqual(expected);
}
