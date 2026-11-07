// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {ResidentTimePeriod} from './resident-time-periods';

/** Decode bounded dense-day/hour keys without narrowing full-range epoch hours to Int32. */
export function makeResidentHourPeriods(
  hourKeys: readonly number[],
  dayKeys: readonly number[]
): ResidentTimePeriod[] {
  return hourKeys.map(hourKey => {
    const dayKey = dayKeys[Math.floor(hourKey / 24)];
    // A discovered hour key must reference an entry in the same read revision's day domain.
    assert(Number.isInteger(hourKey) && hourKey >= 0 && dayKey !== undefined);
    const epochDay = (dayKey ^ 0x80000000) | 0;
    const hour = hourKey % 24;
    const start = new Date(epochDay * 86400000 + hour * 3600000);
    const end = new Date(start.getTime() + 3600000);
    if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()))
      throw new Error(
        'Calendar period crosses the supported Date boundary; narrow the selected time range.'
      );
    const year = start.getUTCFullYear();
    const month = String(start.getUTCMonth() + 1).padStart(2, '0');
    const day = String(start.getUTCDate()).padStart(2, '0');
    return {
      label: `${year}-${month}-${day} ${String(hour).padStart(2, '0')}:00`,
      start: start.getTime(),
      end: end.getTime()
    };
  });
}
