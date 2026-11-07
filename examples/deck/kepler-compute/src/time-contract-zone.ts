// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Time-zone contract. Pure, exact helpers between UTC epoch milliseconds and a named IANA zone
 * or fixed offset, plus a GPU-friendly offset table.
 *
 * Kepler's `timezone` (moment-timezone) only changes label formatting; Kepler bins are UTC
 * thresholds. This contract adds exact local-day / hour / minute-of-day classification for kernels.
 *
 * Documented DST choices (Temporal "compatible"):
 * - Gap (nonexistent local time, spring forward): resolve with the offset BEFORE the transition, so
 *   the result lands after the gap (02:30 in a 02:00->03:00 gap becomes 03:30 local).
 * - Overlap (ambiguous local time, fall back): resolve to the EARLIER instant (first occurrence).
 * Offsets are whole seconds. Bucketing is by local wall-clock label, so a repeated local hour
 * (fall back) maps both occurrences to the same local-hour bucket key.
 */

export type TimeZoneSpecification =
  | {kind: 'utc'}
  | {kind: 'fixed'; offsetSeconds: number}
  | {kind: 'iana'; name: string};

export type LocalTimeResolutionPolicy = {
  gap: 'later' | 'earlier' | 'reject';
  overlap: 'earlier' | 'later' | 'reject';
};

export const DEFAULT_LOCAL_TIME_RESOLUTION: LocalTimeResolutionPolicy = {
  gap: 'later',
  overlap: 'earlier'
};

export type ZoneOffsetTable = {
  /** Sorted UTC transition instants, epoch milliseconds (a multiple of 1000). */
  transitionMilliseconds: bigint[];
  /** offsetSeconds[i] applies before transition i; the last entry applies after the last. */
  offsetSeconds: number[];
  /** Covered range [start, end] used to build the table; outside it the edge offsets extrapolate. */
  startMilliseconds: bigint;
  endMilliseconds: bigint;
};

export type ZoneOffsetTableWords = {
  /** Int64 (low, high) word pairs, little-endian, one pair per transition. */
  transitionWords: Uint32Array;
  /** Int32 offsets, length transitionCount + 1. */
  offsets: Int32Array;
};

export type LocalTimeFields = {
  offsetSeconds: number;
  localMilliseconds: bigint;
  /** Days since 1970-01-01 of the local calendar date. */
  localDayIndex: bigint;
  localHour: number;
  localMinuteOfDay: number;
};

const MILLISECONDS_PER_SECOND = 1000n;
const MILLISECONDS_PER_MINUTE = 60_000n;
const MILLISECONDS_PER_HOUR = 3_600_000n;
const MILLISECONDS_PER_DAY = 86_400_000n;
const DATE_RANGE_MILLISECONDS = 8_640_000_000_000_000n;

export function floorDivideBigInt(numerator: bigint, denominator: bigint): bigint {
  const quotient = numerator / denominator;
  return numerator % denominator !== 0n && numerator < 0n !== denominator < 0n
    ? quotient - 1n
    : quotient;
}

export function floorModuloBigInt(numerator: bigint, denominator: bigint): bigint {
  return numerator - floorDivideBigInt(numerator, denominator) * denominator;
}

/** Accepts "UTC", "Z", "+05:30", "-0800", "+05", or an IANA name. */
export function parseTimeZoneSpecification(zone: string): TimeZoneSpecification {
  if (zone === 'UTC' || zone === 'Z' || zone === 'GMT') return {kind: 'utc'};
  const match = /^([+-])(\d{2})(?::?(\d{2}))?$/.exec(zone);
  if (match) {
    const hours = Number(match[2]);
    const minutes = Number(match[3] ?? '0');
    // Offsets outside +-23:59 or minutes >= 60 are not valid fixed offsets.
    if (hours > 23 || minutes > 59) throw new Error('Invalid fixed UTC offset');
    const magnitude = hours * 3600 + minutes * 60;
    return {kind: 'fixed', offsetSeconds: match[1] === '-' ? -magnitude : magnitude};
  }
  // Throws RangeError for an unknown zone.
  new Intl.DateTimeFormat('en-US', {timeZone: zone});
  return {kind: 'iana', name: zone};
}

const formatterCache = new Map<string, Intl.DateTimeFormat>();

function getFormatter(name: string): Intl.DateTimeFormat {
  let formatter = formatterCache.get(name);
  if (!formatter) {
    formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: name,
      hourCycle: 'h23',
      calendar: 'iso8601',
      numberingSystem: 'latn',
      era: 'short',
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      second: 'numeric'
    });
    formatterCache.set(name, formatter);
  }
  return formatter;
}

/** Exact zone offset in seconds at a UTC instant (floored to the second). */
export function getZoneOffsetSeconds(
  zone: TimeZoneSpecification,
  utcMilliseconds: bigint | number
): number {
  if (zone.kind === 'utc') return 0;
  if (zone.kind === 'fixed') return zone.offsetSeconds;
  const instant = BigInt(utcMilliseconds);
  // Intl cannot format outside the Date range; extrapolate with the nearest representable edge.
  const clamped = Number(
    instant > DATE_RANGE_MILLISECONDS
      ? DATE_RANGE_MILLISECONDS
      : instant < -DATE_RANGE_MILLISECONDS
        ? -DATE_RANGE_MILLISECONDS
        : instant
  );
  const flooredSecondMilliseconds = Math.floor(clamped / 1000) * 1000;
  const parts = getFormatter(zone.name).formatToParts(new Date(flooredSecondMilliseconds));
  const fields = {year: 0, month: 0, day: 0, hour: 0, minute: 0, second: 0};
  let isBeforeCommonEra = false;
  for (const part of parts) {
    if (part.type === 'era') isBeforeCommonEra = part.value.startsWith('B');
    else if (part.type in fields) fields[part.type as keyof typeof fields] = Number(part.value);
  }
  const year = isBeforeCommonEra ? 1 - fields.year : fields.year;
  const localAsUtc = new Date(0);
  localAsUtc.setUTCFullYear(year, fields.month - 1, fields.day);
  localAsUtc.setUTCHours(fields.hour % 24, fields.minute, fields.second, 0);
  return (localAsUtc.getTime() - flooredSecondMilliseconds) / 1000;
}

export function convertUtcToLocalMilliseconds(
  zone: TimeZoneSpecification,
  utcMilliseconds: bigint
): bigint {
  return utcMilliseconds + BigInt(getZoneOffsetSeconds(zone, utcMilliseconds)) * 1000n;
}

export type LocalTimeResolution =
  | {kind: 'unique' | 'gap' | 'overlap'; utcMilliseconds: bigint}
  | {kind: 'rejected'; reason: 'gap' | 'overlap'};

/**
 * Local wall-clock milliseconds (the local time expressed as if it were UTC) -> UTC instant.
 * Assumes at most one offset transition within +-24 hours of the local time.
 */
export function convertLocalToUtcMilliseconds(
  zone: TimeZoneSpecification,
  localMilliseconds: bigint,
  policy: LocalTimeResolutionPolicy = DEFAULT_LOCAL_TIME_RESOLUTION
): LocalTimeResolution {
  const offsetBefore = getZoneOffsetSeconds(zone, localMilliseconds - MILLISECONDS_PER_DAY);
  const offsetAfter = getZoneOffsetSeconds(zone, localMilliseconds + MILLISECONDS_PER_DAY);
  const candidateBefore = localMilliseconds - BigInt(offsetBefore) * 1000n;
  const candidateAfter = localMilliseconds - BigInt(offsetAfter) * 1000n;
  if (offsetBefore === offsetAfter) return {kind: 'unique', utcMilliseconds: candidateBefore};
  const beforeValid = getZoneOffsetSeconds(zone, candidateBefore) === offsetBefore;
  const afterValid = getZoneOffsetSeconds(zone, candidateAfter) === offsetAfter;
  if (beforeValid && afterValid) {
    if (policy.overlap === 'reject') return {kind: 'rejected', reason: 'overlap'};
    const earlier = candidateBefore < candidateAfter ? candidateBefore : candidateAfter;
    const later = candidateBefore < candidateAfter ? candidateAfter : candidateBefore;
    return {kind: 'overlap', utcMilliseconds: policy.overlap === 'earlier' ? earlier : later};
  }
  if (beforeValid) return {kind: 'unique', utcMilliseconds: candidateBefore};
  if (afterValid) return {kind: 'unique', utcMilliseconds: candidateAfter};
  if (policy.gap === 'reject') return {kind: 'rejected', reason: 'gap'};
  // Gap: 'later' uses the pre-transition offset (lands after the gap), 'earlier' the post offset.
  return {
    kind: 'gap',
    utcMilliseconds: policy.gap === 'later' ? candidateBefore : candidateAfter
  };
}

/**
 * Builds the sorted transition table over [start, end] by 12-hour scanning plus second bisection.
 * Transitions closer together than the scan step (12h) can be missed; real zones do not do this.
 */
export function buildZoneOffsetTable(
  zone: TimeZoneSpecification,
  startMilliseconds: bigint,
  endMilliseconds: bigint
): ZoneOffsetTable {
  const transitionMilliseconds: bigint[] = [];
  const offsetSeconds = [getZoneOffsetSeconds(zone, startMilliseconds)];
  if (zone.kind === 'iana') {
    const step = 12n * MILLISECONDS_PER_HOUR;
    let previousInstant = startMilliseconds;
    let previousOffset = offsetSeconds[0];
    while (previousInstant < endMilliseconds) {
      const nextInstant =
        previousInstant + step < endMilliseconds ? previousInstant + step : endMilliseconds;
      const nextOffset = getZoneOffsetSeconds(zone, nextInstant);
      if (nextOffset !== previousOffset) {
        let low = previousInstant;
        let high = nextInstant;
        // Invariant: offset(low) === previousOffset, offset(high) !== previousOffset.
        while (high - low > MILLISECONDS_PER_SECOND) {
          const middle = low + floorDivideBigInt(high - low, 2n * 1000n) * 1000n;
          if (getZoneOffsetSeconds(zone, middle) === previousOffset) low = middle;
          else high = middle;
        }
        transitionMilliseconds.push(floorDivideBigInt(high, 1000n) * 1000n);
        offsetSeconds.push(nextOffset);
        previousOffset = nextOffset;
      }
      previousInstant = nextInstant;
    }
  }
  return {transitionMilliseconds, offsetSeconds, startMilliseconds, endMilliseconds};
}

/** Binary search: the offset index (0..transitionCount) for a UTC instant. */
export function findZoneOffsetIndex(table: ZoneOffsetTable, utcMilliseconds: bigint): number {
  let low = 0;
  let high = table.transitionMilliseconds.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (table.transitionMilliseconds[middle] <= utcMilliseconds) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** Pure classification oracle: exact local day/hour/minute-of-day from the offset table. */
export function classifyLocalTime(
  table: ZoneOffsetTable,
  utcMilliseconds: bigint
): LocalTimeFields {
  const offsetSeconds = table.offsetSeconds[findZoneOffsetIndex(table, utcMilliseconds)];
  const localMilliseconds = utcMilliseconds + BigInt(offsetSeconds) * 1000n;
  const localDayIndex = floorDivideBigInt(localMilliseconds, MILLISECONDS_PER_DAY);
  const millisecondsOfDay = localMilliseconds - localDayIndex * MILLISECONDS_PER_DAY;
  const localMinuteOfDay = Number(millisecondsOfDay / MILLISECONDS_PER_MINUTE);
  return {
    offsetSeconds,
    localMilliseconds,
    localDayIndex,
    localHour: Math.floor(localMinuteOfDay / 60),
    localMinuteOfDay
  };
}

export function packZoneOffsetTable(table: ZoneOffsetTable): ZoneOffsetTableWords {
  const transitionWords = new Uint32Array(table.transitionMilliseconds.length * 2);
  table.transitionMilliseconds.forEach((value, index) => {
    const unsigned = BigInt.asUintN(64, value);
    transitionWords[index * 2] = Number(unsigned & 0xffffffffn);
    transitionWords[index * 2 + 1] = Number(unsigned >> 32n);
  });
  return {transitionWords, offsets: Int32Array.from(table.offsetSeconds)};
}

/** Same classification as classifyLocalTime but reading only the packed word arrays (kernel emulation). */
export function classifyLocalMinuteOfDayFromWords(
  words: ZoneOffsetTableWords,
  utcLow: number,
  utcHigh: number
): {offsetSeconds: number; localMinuteOfDay: number; localDayIndex: bigint} {
  const count = words.transitionWords.length / 2;
  const compare = (index: number): number => {
    const lowWord = words.transitionWords[index * 2];
    const highWord = (words.transitionWords[index * 2 + 1] ^ 0x80000000) >>> 0;
    const instantHigh = (utcHigh ^ 0x80000000) >>> 0;
    if (highWord !== instantHigh) return highWord > instantHigh ? 1 : -1;
    return lowWord === utcLow ? 0 : lowWord > utcLow ? 1 : -1;
  };
  let low = 0;
  let high = count;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (compare(middle) <= 0) low = middle + 1;
    else high = middle;
  }
  const offsetSeconds = words.offsets[low];
  const instant = BigInt.asIntN(64, (BigInt(utcHigh) << 32n) | BigInt(utcLow));
  const local = instant + BigInt(offsetSeconds) * 1000n;
  const dayIndex = floorDivideBigInt(local, MILLISECONDS_PER_DAY);
  return {
    offsetSeconds,
    localDayIndex: dayIndex,
    localMinuteOfDay: Number((local - dayIndex * MILLISECONDS_PER_DAY) / MILLISECONDS_PER_MINUTE)
  };
}

export type LocalBucketUnit = 'day' | 'hour' | 'minute';

const BUCKET_MILLISECONDS: Record<LocalBucketUnit, bigint> = {
  day: MILLISECONDS_PER_DAY,
  hour: MILLISECONDS_PER_HOUR,
  minute: MILLISECONDS_PER_MINUTE
};

/** Local-label bucket key (floor of local wall-clock time to the unit). Repeated local hours share a key. */
export function getLocalBucketKey(
  table: ZoneOffsetTable,
  utcMilliseconds: bigint,
  unit: LocalBucketUnit
): bigint {
  const local = classifyLocalTime(table, utcMilliseconds).localMilliseconds;
  return floorDivideBigInt(local, BUCKET_MILLISECONDS[unit]);
}

/**
 * UTC start instants of every bucket in [start, end), per unit, in time order. A repeated local
 * bucket (fall back) emits one edge per occurrence; a skipped bucket (gap) emits none. Day edges
 * are local midnights (23h/25h days appear naturally); a skipped midnight emits the first instant
 * after the gap. Edges are exact for transitions at whole minutes.
 */
export function getLocalBucketEdges(
  table: ZoneOffsetTable,
  unit: LocalBucketUnit,
  startMilliseconds: bigint,
  endMilliseconds: bigint
): bigint[] {
  const width = BUCKET_MILLISECONDS[unit];
  const edges: bigint[] = [];
  let segmentIndex = findZoneOffsetIndex(table, startMilliseconds);
  let segmentStart = startMilliseconds;
  while (segmentStart < endMilliseconds) {
    const segmentEnd =
      segmentIndex < table.transitionMilliseconds.length &&
      table.transitionMilliseconds[segmentIndex] < endMilliseconds
        ? table.transitionMilliseconds[segmentIndex]
        : endMilliseconds;
    const offsetMilliseconds = BigInt(table.offsetSeconds[segmentIndex]) * 1000n;
    // First aligned instant at or after segmentStart within this segment.
    let edge = segmentStart;
    const remainder = floorModuloBigInt(edge + offsetMilliseconds, width);
    if (remainder !== 0n) edge += width - remainder;
    if (segmentStart > startMilliseconds) {
      const previousLocalKey = floorDivideBigInt(
        segmentStart - 1n + BigInt(table.offsetSeconds[segmentIndex - 1]) * 1000n,
        width
      );
      const currentLocalKey = floorDivideBigInt(segmentStart + offsetMilliseconds, width);
      // A bucket start inside the gap: the first instant after the gap begins that bucket.
      if (remainder !== 0n && currentLocalKey > previousLocalKey) edges.push(segmentStart);
    }
    for (; edge < segmentEnd; edge += width) edges.push(edge);
    segmentStart = segmentEnd;
    segmentIndex++;
  }
  return edges;
}
