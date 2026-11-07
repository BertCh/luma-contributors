// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentTimePeriod} from './resident-time-periods';

const pad2 = (value: number) => String(value).padStart(2, '0');
const formatUtcDate = (milliseconds: number) => {
  const date = new Date(milliseconds);
  return `${date.getUTCFullYear()}-${pad2(date.getUTCMonth() + 1)}-${pad2(date.getUTCDate())}`;
};

/** Native formatTimeBinRange: date-only for spans of a day or more, otherwise date and minute. */
function formatTimeBinRange(start: number, end: number): string {
  const crossesDay = formatUtcDate(start) !== formatUtcDate(end);
  if (Math.max(0, end - start) >= 86400000 || crossesDay)
    return `${formatUtcDate(start)} \u2013 ${formatUtcDate(end)}`;
  const time = (milliseconds: number) => {
    const date = new Date(milliseconds);
    return `${formatUtcDate(milliseconds)} ${pad2(date.getUTCHours())}:${pad2(date.getUTCMinutes())}`;
  };
  return `${time(start)} \u2013 ${time(end)}`;
}

/**
 * Native equalWidthTimeBins: exactBinThresholds followed by d3 histogram bins over [minimum, maximum].
 * Thresholds at or below the minimum or above the maximum are dropped, bin edges keep their real
 * (non-integer) values, and a repeated label replaces the earlier bin's data but keeps its position.
 * `thresholds` are the retained real edges; `sourceBins[index]` is the raw bin supplying each period.
 */
export function makeResidentTimeBarPeriods(
  minimum: number,
  maximum: number,
  binCount: number
): {periods: ResidentTimePeriod[]; thresholds: number[]; sourceBins: number[]} {
  const count = Math.max(2, binCount);
  const thresholds: number[] = [];
  if (Number.isFinite(minimum) && Number.isFinite(maximum) && maximum > minimum) {
    const step = (maximum - minimum) / count;
    for (let index = 1; index < count; index++) thresholds.push(minimum + step * index);
  }
  let first = 0;
  let last = thresholds.length;
  while (first < last && thresholds[first] <= minimum) first++;
  while (last > first && thresholds[last - 1] > maximum) last--;
  const retained = thresholds.slice(first, last);
  const labels = new Map<string, number>();
  const periods: ResidentTimePeriod[] = [];
  const sourceBins: number[] = [];
  for (let index = 0; index <= retained.length; index++) {
    const start = index > 0 ? retained[index - 1] : minimum;
    const end = index < retained.length ? retained[index] : maximum;
    const label = formatTimeBinRange(start, end);
    const existing = labels.get(label);
    if (existing === undefined) {
      labels.set(label, periods.length);
      periods.push({label, start, end});
      sourceBins.push(index);
    } else {
      periods[existing] = {label, start, end};
      sourceBins[existing] = index;
    }
  }
  return {periods, thresholds: retained, sourceBins};
}
