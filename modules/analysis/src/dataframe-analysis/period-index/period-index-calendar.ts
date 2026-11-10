// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUTemporalCalendarStep,
  GPUTemporalCalendarUnit
} from '../../analysis-core/contracts/temporal-context';

/**
 * Local calendar time written as fields. Unset fields default to the start of the step: month 1,
 * day 1, hour 0.
 */
export type GPUPeriodIndexCalendarFields = {
  /** Proleptic Gregorian year, astronomical numbering (year 0 exists). */
  year: number;
  /** 1 to 12. Read by every step finer than `'year'`. */
  month?: number;
  /** Day of the month, 1 to 31. Read by `'week'`, `'day'` and `'hour'`; a week starts on this day. */
  day?: number;
  /** 0 to 23. Read by `'hour'`. */
  hour?: number;
};

/**
 * Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's `days_from_civil`), in integer
 * arithmetic with truncating division, the same formula the {@link GPUPeriodIndex} kernel evaluates.
 */
export function getGPUPeriodIndexDaysFromCivil(year: number, month: number, day: number): number {
  const adjustedYear = month <= 2 ? year - 1 : year;
  const era = Math.trunc((adjustedYear >= 0 ? adjustedYear : adjustedYear - 399) / 400);
  const yearOfEra = adjustedYear - era * 400;
  const shiftedMonth = (month + 9) % 12;
  const dayOfYear = Math.trunc((153 * shiftedMonth + 2) / 5) + day - 1;
  const dayOfEra =
    yearOfEra * 365 + Math.trunc(yearOfEra / 4) - Math.trunc(yearOfEra / 100) + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}

/**
 * Whole-step index of a local calendar time: `year`, `year * 4 + quarter - 1`,
 * `year * 12 + month - 1`, days since the epoch for `'day'` and `'week'`, hours since the epoch for
 * `'hour'`.
 */
export function getGPUPeriodIndexStepIndex(
  calendarUnit: GPUTemporalCalendarUnit,
  fields: GPUPeriodIndexCalendarFields
): number {
  const {year, month = 1, day = 1, hour = 0} = fields;
  switch (calendarUnit) {
    case 'year':
      return year;
    case 'quarter':
      return year * 4 + Math.trunc((month - 1) / 3);
    case 'month':
      return year * 12 + month - 1;
    case 'week':
    case 'day':
      return getGPUPeriodIndexDaysFromCivil(year, month, day);
    case 'hour':
      return getGPUPeriodIndexDaysFromCivil(year, month, day) * 24 + hour;
  }
}

/** Whole steps in one period: `interval`, times 7 for a week (counted in days). */
export function getGPUPeriodIndexStepsPerPeriod(step: GPUTemporalCalendarStep): number {
  return (step.interval ?? 1) * (step.calendarUnit === 'week' ? 7 : 1);
}

/**
 * Period of a local calendar time, or `-1` when it falls before `origin` or past the last of
 * `count` periods. The CPU twin of the {@link GPUPeriodIndex} calendar kernel, for tests and for
 * callers that bin on the CPU.
 */
export function getGPUPeriodIndexCalendarPeriod(
  step: GPUTemporalCalendarStep,
  origin: GPUPeriodIndexCalendarFields,
  count: number,
  fields: GPUPeriodIndexCalendarFields
): number {
  const difference =
    getGPUPeriodIndexStepIndex(step.calendarUnit, fields) -
    getGPUPeriodIndexStepIndex(step.calendarUnit, origin);
  const period = Math.trunc(difference / getGPUPeriodIndexStepsPerPeriod(step));
  return difference >= 0 && period < count ? period : -1;
}
