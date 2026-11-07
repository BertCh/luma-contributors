// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Source encoding table for time columns. Every parser returns an exact bigint nanosecond count
 * (microseconds are derived and flagged when they would truncate) plus an ambiguity report.
 * Nothing here rounds silently: lossy or assumed interpretations are listed in `ambiguities`.
 *
 * Microsecond counts must fit Int64; nanosecond counts may exceed it (flagged).
 * Result kinds: 'instant' (UTC epoch), 'time-of-day' (time32/time64, nanoseconds since midnight)
 * and 'duration' (signed nanoseconds). ISO strings without an offset are interpreted with
 * `assumedOffsetSeconds` (default 0, UTC) and always report 'local-time-without-offset'.
 */

export type TimeEncodingName =
  | 'epoch-seconds'
  | 'epoch-milliseconds'
  | 'epoch-microseconds'
  | 'epoch-nanoseconds'
  | 'iso8601'
  | 'date32'
  | 'date64'
  | 'time32-second'
  | 'time32-millisecond'
  | 'time64-microsecond'
  | 'time64-nanosecond'
  | 'duration-second'
  | 'duration-millisecond'
  | 'duration-microsecond'
  | 'duration-nanosecond';

export type TimeValueKind = 'instant' | 'time-of-day' | 'duration';

export type TimeAmbiguity =
  | 'naive-timestamp-assumed-utc'
  | 'local-time-without-offset'
  | 'date-only-interpreted-as-midnight'
  | 'binary-float-source'
  | 'epoch-unit-magnitude-suspect'
  | 'fraction-truncated-to-nanoseconds'
  | 'date64-not-midnight'
  | 'sub-microsecond-precision';

export type TimeParseFailure =
  | 'null-value'
  | 'not-finite'
  | 'unsafe-integer'
  | 'out-of-range'
  | 'malformed'
  | 'invalid-calendar-date'
  | 'invalid-time-of-day'
  | 'leap-second'
  | 'wrong-value-type'
  | 'sub-nanosecond';

export type TimeParseOptions = {
  /** Offset applied to ISO strings that carry none. Default 0. */
  assumedOffsetSeconds?: number;
  /** For epoch encodings: false when the Arrow timestamp type declares no timezone. */
  timezoneDeclared?: boolean;
};

export type TimeParseSuccess = {
  ok: true;
  kind: TimeValueKind;
  nanoseconds: bigint;
  /** Floor microseconds (toward negative infinity for instants and durations). */
  microseconds: bigint;
  /** False when the exact nanosecond count exceeds Int64 (timestamp[ns] could not hold it). */
  fitsInt64Nanoseconds: boolean;
  ambiguities: TimeAmbiguity[];
};

export type TimeParseResult = TimeParseSuccess | {ok: false; reason: TimeParseFailure};

type EncodingDescriptor = {nanosecondsPerTick: bigint; kind: TimeValueKind};

export const TIME_ENCODING_TABLE: Record<TimeEncodingName, EncodingDescriptor> = {
  'epoch-seconds': {nanosecondsPerTick: 1_000_000_000n, kind: 'instant'},
  'epoch-milliseconds': {nanosecondsPerTick: 1_000_000n, kind: 'instant'},
  'epoch-microseconds': {nanosecondsPerTick: 1_000n, kind: 'instant'},
  'epoch-nanoseconds': {nanosecondsPerTick: 1n, kind: 'instant'},
  iso8601: {nanosecondsPerTick: 1n, kind: 'instant'},
  date32: {nanosecondsPerTick: 86_400_000_000_000n, kind: 'instant'},
  date64: {nanosecondsPerTick: 1_000_000n, kind: 'instant'},
  'time32-second': {nanosecondsPerTick: 1_000_000_000n, kind: 'time-of-day'},
  'time32-millisecond': {nanosecondsPerTick: 1_000_000n, kind: 'time-of-day'},
  'time64-microsecond': {nanosecondsPerTick: 1_000n, kind: 'time-of-day'},
  'time64-nanosecond': {nanosecondsPerTick: 1n, kind: 'time-of-day'},
  'duration-second': {nanosecondsPerTick: 1_000_000_000n, kind: 'duration'},
  'duration-millisecond': {nanosecondsPerTick: 1_000_000n, kind: 'duration'},
  'duration-microsecond': {nanosecondsPerTick: 1_000n, kind: 'duration'},
  'duration-nanosecond': {nanosecondsPerTick: 1n, kind: 'duration'}
};

const NANOSECONDS_PER_DAY = 86_400_000_000_000n;
const INT64_MINIMUM = -(1n << 63n);
const INT64_MAXIMUM = (1n << 63n) - 1n;
/** Epoch magnitude above which a declared unit is suspicious (year ~5138 in that unit). */
const SUSPECT_MAGNITUDE: Partial<Record<TimeEncodingName, bigint>> = {
  'epoch-seconds': 100_000_000_000n,
  'epoch-milliseconds': 100_000_000_000_000n,
  'epoch-microseconds': 100_000_000_000_000_000n
};

function floorDivide(numerator: bigint, denominator: bigint): bigint {
  const quotient = numerator / denominator;
  return numerator % denominator !== 0n && numerator < 0n !== denominator < 0n
    ? quotient - 1n
    : quotient;
}

function succeed(
  kind: TimeValueKind,
  nanoseconds: bigint,
  ambiguities: TimeAmbiguity[]
): TimeParseResult {
  // The microsecond count must fit Int64 (Arrow timestamp[us]); nanoseconds are exact bigint and
  // may exceed Int64 (years beyond 2262), which fitsInt64Nanoseconds reports.
  const microseconds = floorDivide(nanoseconds, 1000n);
  if (microseconds < INT64_MINIMUM || microseconds > INT64_MAXIMUM) {
    return {ok: false, reason: 'out-of-range'};
  }
  if (microseconds * 1000n !== nanoseconds) ambiguities.push('sub-microsecond-precision');
  return {
    ok: true,
    kind,
    nanoseconds,
    microseconds,
    fitsInt64Nanoseconds: nanoseconds >= INT64_MINIMUM && nanoseconds <= INT64_MAXIMUM,
    ambiguities
  };
}

/** Decimal string of a finite JS number, exactly as its shortest round-trip form, to bigint ticks. */
function convertNumberToNanoseconds(
  value: number,
  nanosecondsPerTick: bigint,
  ambiguities: TimeAmbiguity[]
): bigint | TimeParseFailure {
  if (!Number.isFinite(value)) return 'not-finite';
  if (Number.isInteger(value)) {
    if (!Number.isSafeInteger(value)) return 'unsafe-integer';
    return BigInt(value) * nanosecondsPerTick;
  }
  // Fractional ticks: the double is not the decimal the producer wrote; report it and use the
  // shortest round-trip decimal expansion.
  ambiguities.push('binary-float-source');
  const text = value.toString();
  const match = /^(-?)(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/.exec(text);
  if (!match) return 'malformed';
  const fraction = match[3] ?? '';
  const exponent = Number(match[4] ?? '0');
  const digits = BigInt(match[2] + fraction);
  const scale = exponent - fraction.length;
  let numerator = digits * nanosecondsPerTick;
  let denominator = 1n;
  if (scale >= 0) numerator *= 10n ** BigInt(scale);
  else denominator = 10n ** BigInt(-scale);
  if (numerator % denominator !== 0n) return 'sub-nanosecond';
  const magnitude = numerator / denominator;
  return match[1] ? -magnitude : magnitude;
}

export function daysFromCivil(year: bigint, month: number, day: number): bigint {
  // Howard Hinnant's days_from_civil.
  const shiftedYear = month <= 2 ? year - 1n : year;
  const era = floorDivide(shiftedYear, 400n);
  const yearOfEra = shiftedYear - era * 400n;
  const monthIndex = BigInt((month + 9) % 12);
  const dayOfYear = (153n * monthIndex + 2n) / 5n + BigInt(day) - 1n;
  const dayOfEra = yearOfEra * 365n + yearOfEra / 4n - yearOfEra / 100n + dayOfYear;
  return era * 146097n + dayOfEra - 719468n;
}

function getDaysInMonth(year: bigint, month: number): number {
  if (month === 2) {
    const isLeap = (year % 4n === 0n && year % 100n !== 0n) || year % 400n === 0n;
    return isLeap ? 29 : 28;
  }
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}

const ISO_PATTERN =
  /^([+-]\d{6}|\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?(Z|z|[+-]\d{2}(?::?\d{2})?)?)?$/;

function parseIso8601(text: string, options: TimeParseOptions): TimeParseResult {
  const match = ISO_PATTERN.exec(text.trim());
  if (!match) return {ok: false, reason: 'malformed'};
  const ambiguities: TimeAmbiguity[] = [];
  const year = BigInt(match[1]);
  // "-000000" is not a valid ISO 8601 extended year.
  if (match[1] === '-000000') return {ok: false, reason: 'invalid-calendar-date'};
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (month < 1 || month > 12 || day < 1 || day > getDaysInMonth(year, month)) {
    return {ok: false, reason: 'invalid-calendar-date'};
  }
  let secondsOfDay = 0n;
  let fractionNanoseconds = 0n;
  let offsetSeconds = 0;
  if (match[4] === undefined) {
    ambiguities.push('date-only-interpreted-as-midnight');
    ambiguities.push('local-time-without-offset');
    offsetSeconds = options.assumedOffsetSeconds ?? 0;
  } else {
    const hour = Number(match[4]);
    const minute = Number(match[5]);
    const second = Number(match[6] ?? '0');
    if (hour > 23 || minute > 59) return {ok: false, reason: 'invalid-time-of-day'};
    // Leap seconds cannot be represented on the POSIX timeline: refuse rather than clamp.
    if (second === 60) return {ok: false, reason: 'leap-second'};
    if (second > 59) return {ok: false, reason: 'invalid-time-of-day'};
    secondsOfDay = BigInt(hour * 3600 + minute * 60 + second);
    const fraction = match[7] ?? '';
    if (fraction.length > 9) {
      ambiguities.push('fraction-truncated-to-nanoseconds');
    }
    fractionNanoseconds = BigInt(fraction.slice(0, 9).padEnd(9, '0'));
    const zone = match[8];
    if (zone === undefined) {
      ambiguities.push('local-time-without-offset');
      offsetSeconds = options.assumedOffsetSeconds ?? 0;
    } else if (zone !== 'Z' && zone !== 'z') {
      const sign = zone[0] === '-' ? -1 : 1;
      const digits = zone.slice(1).replace(':', '');
      const offsetHours = Number(digits.slice(0, 2));
      const offsetMinutes = Number(digits.slice(2) || '0');
      if (offsetHours > 23 || offsetMinutes > 59) return {ok: false, reason: 'malformed'};
      offsetSeconds = sign * (offsetHours * 3600 + offsetMinutes * 60);
    }
  }
  const days = daysFromCivil(year, month, day);
  const seconds = days * 86_400n + secondsOfDay - BigInt(offsetSeconds);
  return succeed('instant', seconds * 1_000_000_000n + fractionNanoseconds, ambiguities);
}

/** Parses one value of a declared encoding. Null/undefined are reported, never coerced to epoch. */
export function parseTimeSourceValue(
  encoding: TimeEncodingName,
  value: number | bigint | string | null | undefined,
  options: TimeParseOptions = {}
): TimeParseResult {
  if (value === null || value === undefined) return {ok: false, reason: 'null-value'};
  const descriptor = TIME_ENCODING_TABLE[encoding];
  if (encoding === 'iso8601') {
    return typeof value === 'string'
      ? parseIso8601(value, options)
      : {ok: false, reason: 'wrong-value-type'};
  }
  if (typeof value === 'string') return {ok: false, reason: 'wrong-value-type'};
  const ambiguities: TimeAmbiguity[] = [];
  let nanoseconds: bigint;
  if (typeof value === 'bigint') {
    nanoseconds = value * descriptor.nanosecondsPerTick;
  } else {
    const converted = convertNumberToNanoseconds(value, descriptor.nanosecondsPerTick, ambiguities);
    if (typeof converted === 'string') return {ok: false, reason: converted};
    nanoseconds = converted;
  }
  const ticks = nanoseconds / descriptor.nanosecondsPerTick;
  if (encoding === 'date32' && (ticks < -(1n << 31n) || ticks >= 1n << 31n)) {
    return {ok: false, reason: 'out-of-range'};
  }
  if (encoding === 'date64' && ticks % 86_400_000n !== 0n) ambiguities.push('date64-not-midnight');
  if (
    descriptor.kind === 'time-of-day' &&
    (nanoseconds < 0n || nanoseconds >= NANOSECONDS_PER_DAY)
  ) {
    return {ok: false, reason: 'out-of-range'};
  }
  if (descriptor.kind === 'instant' && encoding.startsWith('epoch-')) {
    if (options.timezoneDeclared === false) ambiguities.push('naive-timestamp-assumed-utc');
    const suspect = SUSPECT_MAGNITUDE[encoding];
    if (suspect !== undefined && (ticks >= suspect || ticks <= -suspect)) {
      ambiguities.push('epoch-unit-magnitude-suspect');
    }
  }
  return succeed(descriptor.kind, nanoseconds, ambiguities);
}

/**
 * Guesses which epoch unit a set of numbers is in from the median-ish magnitude of the first
 * non-zero value. Returns every unit that places the value between years 1971 and 2100, so callers
 * can refuse to guess when more than one fits.
 */
export function detectEpochUnitCandidates(
  value: number | bigint
): Extract<TimeEncodingName, `epoch-${string}`>[] {
  const magnitude = BigInt(typeof value === 'bigint' ? value : Math.trunc(value));
  const lower = 31_536_000n; // 1971-01-01 in seconds
  const upper = 4_102_444_800n; // 2100-01-01 in seconds
  const candidates: Extract<TimeEncodingName, `epoch-${string}`>[] = [];
  const units: [Extract<TimeEncodingName, `epoch-${string}`>, bigint][] = [
    ['epoch-seconds', 1n],
    ['epoch-milliseconds', 1000n],
    ['epoch-microseconds', 1_000_000n],
    ['epoch-nanoseconds', 1_000_000_000n]
  ];
  for (const [name, scale] of units) {
    if (magnitude >= lower * scale && magnitude < upper * scale) candidates.push(name);
  }
  return candidates;
}

/** Converts exact nanoseconds to ticks of a target epoch unit; `exact` is false when it truncated. */
export function convertNanosecondsToEpochTicks(
  nanoseconds: bigint,
  encoding: Extract<TimeEncodingName, `epoch-${string}`>
): {ticks: bigint; exact: boolean} {
  const scale = TIME_ENCODING_TABLE[encoding].nanosecondsPerTick;
  const ticks = floorDivide(nanoseconds, scale);
  return {ticks, exact: ticks * scale === nanoseconds};
}
