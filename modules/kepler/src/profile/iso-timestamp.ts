// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Returned by {@link parseKeplerIsoTimestamp} for text that has the shape of a timestamp but names an impossible time. */
export const KEPLER_IMPOSSIBLE_TIMESTAMP = Number.NEGATIVE_INFINITY;

function readDigits(text: string, start: number, count: number): number {
  let value = 0;
  for (let index = start; index < start + count; index++) {
    const code = text.charCodeAt(index);
    if (code < 48 || code > 57) return -1;
    value = value * 10 + (code - 48);
  }
  return value;
}

function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's algorithm); valid for year 0 and up. */
function getDaysFromCivil(year: number, month: number, day: number): number {
  const shiftedYear = month <= 2 ? year - 1 : year;
  const era = Math.floor(shiftedYear / 400);
  const yearOfEra = shiftedYear - era * 400;
  const dayOfYear = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1;
  const dayOfEra =
    yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100) + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}

/**
 * Milliseconds since the epoch of an ISO 8601 date or date-time string (`YYYY-MM-DD`, optionally
 * `THH:MM`, `:SS`, `.fraction` and a `Z` or `+HH:MM` / `+HHMM` zone). NaN when the text does not
 * have that shape. {@link KEPLER_IMPOSSIBLE_TIMESTAMP} when it has the shape but not a real time: a day
 * beyond the month's length (leap years honoured), month outside 1 to 12, hour above 23 (so
 * `24:00:00` is rejected), minute or second above 59, a zone offset beyond 23:59. The engine's
 * `Date.parse` rolls such text over to another date, so the check is done here. A date-time
 * without a zone designator is read as UTC, because `Date.parse` would read it in the zone of the
 * machine and the same data would then profile differently on two machines.
 */
export function parseKeplerIsoTimestamp(text: string): number {
  // Cheap rejection first: most text is not a timestamp, and this runs on every string.
  const length = text.length;
  if (length < 10 || text.charCodeAt(4) !== 45 || text.charCodeAt(7) !== 45) return Number.NaN;
  const year = readDigits(text, 0, 4);
  const month = readDigits(text, 5, 2);
  const day = readDigits(text, 8, 2);
  if (year < 0 || month < 0 || day < 0) return Number.NaN;
  let hour = 0;
  let minute = 0;
  let second = 0;
  let fractionMilliseconds = 0;
  let offsetMinutes = 0;
  let offsetHour = 0;
  let offsetMinute = 0;
  if (length > 10) {
    if (text.charCodeAt(10) !== 84 || text.charCodeAt(13) !== 58) return Number.NaN;
    hour = readDigits(text, 11, 2);
    minute = readDigits(text, 14, 2);
    if (hour < 0 || minute < 0) return Number.NaN;
    let position = 16;
    if (position < length && text.charCodeAt(position) === 58) {
      second = readDigits(text, position + 1, 2);
      if (second < 0) return Number.NaN;
      position += 3;
      if (position < length && text.charCodeAt(position) === 46) {
        position++;
        const fractionStart = position;
        let scale = 100;
        while (position < length) {
          const code = text.charCodeAt(position);
          if (code < 48 || code > 57) break;
          if (scale >= 1) fractionMilliseconds += (code - 48) * scale;
          scale /= 10;
          position++;
        }
        if (position === fractionStart) return Number.NaN;
        fractionMilliseconds = Math.floor(fractionMilliseconds);
      }
    }
    if (position < length) {
      const code = text.charCodeAt(position);
      if (code === 90) {
        if (position + 1 !== length) return Number.NaN;
      } else if (code === 43 || code === 45) {
        offsetHour = readDigits(text, position + 1, 2);
        const minuteStart = text.charCodeAt(position + 3) === 58 ? position + 4 : position + 3;
        offsetMinute = readDigits(text, minuteStart, 2);
        if (offsetHour < 0 || offsetMinute < 0 || minuteStart + 2 !== length) return Number.NaN;
        offsetMinutes = (offsetHour * 60 + offsetMinute) * (code === 45 ? -1 : 1);
      } else {
        return Number.NaN;
      }
    }
  }
  if (month < 1 || month > 12) return KEPLER_IMPOSSIBLE_TIMESTAMP;
  const daysInMonth = month === 2 && isLeapYear(year) ? 29 : DAYS_IN_MONTH[month - 1];
  if (day < 1 || day > daysInMonth) return KEPLER_IMPOSSIBLE_TIMESTAMP;
  if (hour > 23 || minute > 59 || second > 59 || offsetHour > 23 || offsetMinute > 59)
    return KEPLER_IMPOSSIBLE_TIMESTAMP;
  return (
    getDaysFromCivil(year, month, day) * 86_400_000 +
    ((hour * 60 + minute) * 60 + second) * 1000 +
    fractionMilliseconds -
    offsetMinutes * 60_000
  );
}
