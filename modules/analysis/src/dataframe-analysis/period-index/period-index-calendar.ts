// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUTemporalCalendarStep,
  GPUTemporalCalendarUnit
} from '../../analysis-core/contracts/temporal-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

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

/** Validates a proleptic-Gregorian civil date, including astronomical year zero. @internal */
export function validateGPUPeriodIndexCivilDate(
  id: string,
  prop: string,
  year: number,
  month: number,
  day: number
): void {
  validateRequirement(
    id,
    Number.isInteger(year),
    `${prop}.year must be an integer; received ${year}`
  );
  validateCapacity(id, `${prop}.month`, month, 1, 12);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const maximumDay = month === 2 ? (leap ? 29 : 28) : [4, 6, 9, 11].includes(month) ? 30 : 31;
  const date = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
  validateRequirement(
    id,
    Number.isInteger(day) && day >= 1 && day <= maximumDay,
    `${prop}.day must be in 1..${maximumDay} for ${date}; received ${day}`
  );
}

/**
 * Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's `days_from_civil`), in integer
 * arithmetic with truncating division, the same formula the {@link GPUPeriodIndex} kernel evaluates.
 */
export function getGPUPeriodIndexDaysFromCivil(year: number, month: number, day: number): number {
  validateGPUPeriodIndexCivilDate('period-index-calendar', 'fields', year, month, day);
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

/** Checks the signed WGSL divisor, including the seven-day weekly multiplier. @internal */
export function validateGPUPeriodIndexStepsPerPeriod(
  id: string,
  step: GPUTemporalCalendarStep
): void {
  validateCapacity(
    id,
    'periods.step steps per period (MAXIMUM_CALENDAR_STEPS_PER_PERIOD)',
    getGPUPeriodIndexStepsPerPeriod(step),
    1,
    0x7fffffff
  );
}

/** Checks the signed whole-step origin used by calendar WGSL. @internal */
export function validateGPUPeriodIndexOriginStep(
  id: string,
  calendarUnit: GPUTemporalCalendarUnit,
  origin: GPUPeriodIndexCalendarFields
): void {
  if (['day', 'week', 'hour'].includes(calendarUnit)) {
    validateGPUPeriodIndexCivilDate(
      id,
      'periods.origin',
      origin.year,
      origin.month ?? 1,
      origin.day ?? 1
    );
  }
  validateCapacity(
    id,
    'periods.origin step index (MINIMUM_CALENDAR_STEP_INDEX..MAXIMUM_CALENDAR_STEP_INDEX)',
    getGPUPeriodIndexStepIndex(calendarUnit, origin),
    -0x80000000,
    0x7fffffff
  );
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
  validateGPUPeriodIndexOriginStep('period-index-calendar', step.calendarUnit, origin);
  // GPUCalendarBuckets marks every missing/undecodable row with this signed-year sentinel.
  // Match the GPU kernel's exclusion before inspecting the row's other sentinel fields.
  if (fields.year === -0x80000000) return -1;
  const difference =
    getGPUPeriodIndexStepIndex(step.calendarUnit, fields) -
    getGPUPeriodIndexStepIndex(step.calendarUnit, origin);
  const period = Math.trunc(difference / getGPUPeriodIndexStepsPerPeriod(step));
  return difference >= 0 && period < count ? period : -1;
}
