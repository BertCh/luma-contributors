// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSortedTimeRangeMaximumSearchSteps,
  SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT,
  SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS
} from '../../../src/dataframe-analysis/temporal-reduction/sorted-range-bounds';

it('CORE-118 unsigned row count admits at most 32 midpoint steps, including empty and endpoint ranges', () => {
  expect(SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT).toBe(0xffffffff);
  expect(SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS).toBe(32);
  for (const rows of [0, 1, 2, 3, 2 ** 31, 0xfffffffe, 0xffffffff]) {
    const bound = getSortedTimeRangeMaximumSearchSteps(rows);
    // Independent half-open lower/upper bound searches, without constructing any row payload.
    // The predicates can place the answer at either endpoint or any interior split.
    for (const target of [0, Math.floor(rows / 2), rows]) {
      let low = 0;
      let high = rows;
      let steps = 0;
      while (low < high) {
        const middle = low + Math.floor((high - low) / 2);
        if (middle < target) low = middle + 1;
        else high = middle;
        steps++;
      }
      expect(low).toBe(target);
      expect(steps).toBeLessThanOrEqual(bound);
    }
    expect(bound).toBeLessThanOrEqual(32);
  }
  expect(getSortedTimeRangeMaximumSearchSteps(0)).toBe(0);
  expect(getSortedTimeRangeMaximumSearchSteps(0xffffffff)).toBe(32);
  expect(() => getSortedTimeRangeMaximumSearchSteps(2 ** 32)).toThrow(/rowCount.*4294967295/);
});
