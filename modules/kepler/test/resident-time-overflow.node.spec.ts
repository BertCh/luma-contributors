// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  MAXIMUM_RESIDENT_TIME_GROUPS,
  getResidentTimeGroupCapacity
} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';
import {
  ResidentCalendarOverflowError,
  getNextCoarserResidentInterval
} from '../../../examples/deck/kepler-compute/src/resident-time-overflow';
import {makeResidentTimeBarPeriods} from '../../../examples/deck/kepler-compute/src/resident-time-bar-periods';

test('calendar capacity is the native 1,000 periods while the series cross-product fits', () => {
  expect(MAXIMUM_RESIDENT_TIME_GROUPS).toBe(1000);
  expect(getResidentTimeGroupCapacity()).toBe(1000);
  expect(getResidentTimeGroupCapacity(3)).toBe(1000);
  expect(getResidentTimeGroupCapacity(4)).toBe(819);
  expect(getResidentTimeGroupCapacity(7)).toBe(512);
});

test('automatic overflow coarsens like native then falls back to equal-width bins', () => {
  expect(getNextCoarserResidentInterval('hour')).toBe('day');
  expect(getNextCoarserResidentInterval('month')).toBe('year');
  expect(getNextCoarserResidentInterval('year')).toBeUndefined();
  expect(new ResidentCalendarOverflowError('x')).toBeInstanceOf(Error);
});

test('the 1,000-bin fallback builds native equal-width periods', () => {
  const layout = makeResidentTimeBarPeriods(0, 1000 * 86400000, 1000);
  expect(layout.periods.length).toBe(1000);
  expect(layout.thresholds.length).toBe(999);
});
