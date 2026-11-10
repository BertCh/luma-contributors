// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ArrowAnalyticsTimeUnit} from '../../../src/analysis-core/graph-support/arrow-analytics-normalization';
import type {GPUTemporalCalendarStep} from '../../../src/analysis-core/contracts/temporal-context';
import type {GPUPeriodIndexCalendarFields} from '../../../src/dataframe-analysis/period-index/index';

/** Ticks per millisecond; seconds are 1/1000. */
const TICKS_PER_MILLISECOND: Record<ArrowAnalyticsTimeUnit, bigint> = {
  second: 1n,
  millisecond: 1n,
  microsecond: 1000n,
  nanosecond: 1_000_000n
};

/** Epoch milliseconds of `ticks`, floored toward the earlier millisecond. */
export function getEpochMilliseconds(ticks: bigint, unit: ArrowAnalyticsTimeUnit): number {
  if (unit === 'second') {
    return Number(ticks * 1000n);
  }
  const divisor = TICKS_PER_MILLISECOND[unit];
  const quotient = ticks / divisor;
  return Number(ticks % divisor < 0n ? quotient - 1n : quotient);
}

/**
 * Period of a time under a calendar step, from `Date` arithmetic only (not Hinnant's formula), or
 * `-1` for a time before the origin or past the last period. A week is 7 days from the origin day.
 */
export function getExpectedCalendarPeriod(
  step: GPUTemporalCalendarStep,
  origin: GPUPeriodIndexCalendarFields,
  count: number,
  milliseconds: number
): number {
  const interval = step.interval ?? 1;
  const date = new Date(milliseconds);
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();
  const originMonth = (origin.month ?? 1) - 1;
  const originDayStart = Date.UTC(origin.year, originMonth, origin.day ?? 1);
  const dayStart = Date.UTC(year, month, date.getUTCDate());
  let difference: number;
  let stepsPerPeriod = interval;
  switch (step.calendarUnit) {
    case 'year':
      difference = year - origin.year;
      break;
    case 'quarter':
      difference =
        year * 4 + Math.floor(month / 3) - (origin.year * 4 + Math.floor(originMonth / 3));
      break;
    case 'month':
      difference = year * 12 + month - (origin.year * 12 + originMonth);
      break;
    case 'week':
      difference = Math.round((dayStart - originDayStart) / 86_400_000);
      stepsPerPeriod = interval * 7;
      break;
    case 'day':
      difference = Math.round((dayStart - originDayStart) / 86_400_000);
      break;
    case 'hour': {
      const originHourStart = originDayStart + (origin.hour ?? 0) * 3_600_000;
      difference = Math.floor((milliseconds - originHourStart) / 3_600_000);
      break;
    }
  }
  const period = Math.floor(difference / stepsPerPeriod);
  return difference >= 0 && period < count ? period : -1;
}

/** Period of Int64 ticks under a fixed width, in exact bigint arithmetic, or `-1` when out of range. */
export function getExpectedFixedPeriod(
  ticks: bigint,
  width: bigint,
  origin: bigint,
  count: number
): number {
  if (ticks < origin) {
    return -1;
  }
  const period = (ticks - origin) / width;
  return period < BigInt(count) ? Number(period) : -1;
}

/** Deterministic pseudo-random numbers in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}
