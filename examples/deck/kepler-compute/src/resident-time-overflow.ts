// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentCalendarInterval} from './resident-chart-plan';

/** Raised when a calendar interval has more than the native 1,000 periods. */
export class ResidentCalendarOverflowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ResidentCalendarOverflowError';
  }
}

const NATIVE_INTERVAL_ORDER: readonly ResidentCalendarInterval[] = [
  'hour',
  'day',
  'week',
  'month',
  'year'
];

/**
 * Native timeGroupMap coarsens an automatic interval one step at a time while it has more than
 * 1,000 periods; past year it uses 1,000 equal-width bins. Returns undefined for that final step.
 */
export function getNextCoarserResidentInterval(
  interval: ResidentCalendarInterval
): ResidentCalendarInterval | undefined {
  return NATIVE_INTERVAL_ORDER[NATIVE_INTERVAL_ORDER.indexOf(interval) + 1];
}
