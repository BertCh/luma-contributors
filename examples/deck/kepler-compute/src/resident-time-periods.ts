// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentCalendarInterval} from './resident-chart-plan';

export type ResidentTimePeriod = {label: string; start: number; end: number};

/** Decode only bounded calendar keys; UTC construction preserves years 0–99 and pre-epoch dates. */
export function makeResidentTimePeriods(
  keys: readonly number[],
  interval: Exclude<ResidentCalendarInterval, 'hour'>
): ResidentTimePeriod[] {
  return keys.map(key => {
    const ordinal = (key ^ 0x80000000) | 0;
    const daily = interval === 'day' || interval === 'week';
    const start = new Date(daily ? ordinal * 86400000 : 0);
    const end = new Date(daily ? (ordinal + (interval === 'week' ? 7 : 1)) * 86400000 : 0);
    if (!daily) {
      const year = interval === 'year' ? ordinal : Math.floor(ordinal / 12);
      const month = interval === 'year' ? 0 : ordinal - year * 12;
      start.setUTCFullYear(year, month, 1);
      end.setUTCFullYear(
        interval === 'year' ? year + 1 : year,
        interval === 'year' ? 0 : month + 1,
        1
      );
    }
    if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()))
      throw new Error(
        'Calendar period crosses the supported Date boundary; narrow the selected time range.'
      );
    const year = start.getUTCFullYear();
    const month = String(start.getUTCMonth() + 1).padStart(2, '0');
    const date = `${year}-${month}-${String(start.getUTCDate()).padStart(2, '0')}`;
    return {
      label:
        interval === 'year'
          ? String(year)
          : interval === 'month'
            ? `${year}-${month}`
            : interval === 'week'
              ? `Week of ${date}`
              : date,
      start: start.getTime(),
      end: end.getTime()
    };
  });
}
