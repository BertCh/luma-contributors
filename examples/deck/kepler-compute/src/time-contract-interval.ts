// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Interval overlap contract for rows that are time intervals tested against a time window.
 *
 * Row semantics: start is required (null start never matches); end null means open ended (+infinity);
 * start > end is refused (never matches, reported as 'invalid'). A row with start === end is a
 * point when its end is closed and EMPTY (never matches) when its end is open.
 * Defaults: rows are [start, end); windows are [start, end] (Kepler time filters are inclusive).
 * The existing GPUKeplerTimeSelection endTimestamps path is the closed/closed variant.
 *
 * The WGSL generator reuses the TIME_WORDS_WGSL helper names (timeWordsCompare) from
 * @luma.gl/graph/gpu-dataframe and expects those helpers to be emitted before the snippet.
 */

export type IntervalBoundaryClosure = {
  /** Row start included in the row interval. Default true. */
  rowStartClosed: boolean;
  /** Row end included in the row interval. Default false. */
  rowEndClosed: boolean;
  /** Window start included. Default true. */
  windowStartClosed: boolean;
  /** Window end included. Default true. */
  windowEndClosed: boolean;
};

export const DEFAULT_INTERVAL_BOUNDARY_CLOSURE: IntervalBoundaryClosure = {
  rowStartClosed: true,
  rowEndClosed: false,
  windowStartClosed: true,
  windowEndClosed: true
};

export type IntervalRow = {
  /** null: invalid row (never matches). */
  start: bigint | null;
  /** null: open ended. */
  end: bigint | null;
};

export type IntervalWindow = {start: bigint; end: bigint};

export type IntervalOverlapResult = 'overlap' | 'disjoint' | 'invalid';

/** Exact oracle. Throws for a window whose start is after its end. */
export function classifyIntervalOverlap(
  row: IntervalRow,
  window: IntervalWindow,
  closure: IntervalBoundaryClosure = DEFAULT_INTERVAL_BOUNDARY_CLOSURE
): IntervalOverlapResult {
  // A window with start after end is a caller error: no valid range exists.
  if (window.start > window.end) throw new Error('Interval window start is after its end');
  if (row.start === null) return 'invalid';
  if (row.end !== null && row.start > row.end) return 'invalid';
  const hasEnd = row.end !== null;
  const rowEnd = row.end as bigint;
  // Empty row interval: [s, s) (or (s, s]) holds no instant.
  if (hasEnd && row.start === rowEnd && !(closure.rowStartClosed && closure.rowEndClosed)) {
    return 'disjoint';
  }
  // Row begins before the window ends.
  const startsBeforeWindowEnd =
    row.start < window.end ||
    (row.start === window.end && closure.rowStartClosed && closure.windowEndClosed);
  // Window begins before the row ends.
  const endsAfterWindowStart =
    !hasEnd ||
    window.start < rowEnd ||
    (window.start === rowEnd && closure.windowStartClosed && closure.rowEndClosed);
  // A zero-width window [t, t] needs both closures; an open-ended zero-width window holds no instant.
  if (window.start === window.end && !(closure.windowStartClosed && closure.windowEndClosed)) {
    return 'disjoint';
  }
  return startsBeforeWindowEnd && endsAfterWindowStart ? 'overlap' : 'disjoint';
}

export function isIntervalOverlappingWindow(
  row: IntervalRow,
  window: IntervalWindow,
  closure: IntervalBoundaryClosure = DEFAULT_INTERVAL_BOUNDARY_CLOSURE
): boolean {
  return classifyIntervalOverlap(row, window, closure) === 'overlap';
}

/**
 * Generates a WGSL predicate over uint32x2 (low, high) Int64 words:
 *   fn <name>(start: vec2<u32>, end: vec2<u32>, hasEnd: bool,
 *             windowStart: vec2<u32>, windowEnd: vec2<u32>) -> bool
 * Rows with start > end return false (refused). Validity of the start word is the caller's job.
 * Closure flags are baked in as literal constants so the compiled kernel has no runtime branches on them.
 */
export function makeIntervalOverlapWgsl(
  functionName = 'isIntervalOverlappingWindow',
  closure: IntervalBoundaryClosure = DEFAULT_INTERVAL_BOUNDARY_CLOSURE
): string {
  const flag = (value: boolean): string => (value ? 'true' : 'false');
  return /* wgsl */ `
// Requires TIME_WORDS_WGSL (timeWordsCompare).
fn ${functionName}(start: vec2<u32>, end: vec2<u32>, hasEnd: bool, windowStart: vec2<u32>, windowEnd: vec2<u32>) -> bool {
  const rowStartClosed = ${flag(closure.rowStartClosed)};
  const rowEndClosed = ${flag(closure.rowEndClosed)};
  const windowStartClosed = ${flag(closure.windowStartClosed)};
  const windowEndClosed = ${flag(closure.windowEndClosed)};
  if (hasEnd && timeWordsCompare(start, end) > 0i) {
    return false;
  }
  if (hasEnd && timeWordsCompare(start, end) == 0i && !(rowStartClosed && rowEndClosed)) {
    return false;
  }
  if (timeWordsCompare(windowStart, windowEnd) == 0i && !(windowStartClosed && windowEndClosed)) {
    return false;
  }
  let startOrder = timeWordsCompare(start, windowEnd);
  let startsBeforeWindowEnd = startOrder < 0i || (startOrder == 0i && rowStartClosed && windowEndClosed);
  if (!startsBeforeWindowEnd) {
    return false;
  }
  if (!hasEnd) {
    return true;
  }
  let endOrder = timeWordsCompare(windowStart, end);
  return endOrder < 0i || (endOrder == 0i && windowStartClosed && rowEndClosed);
}
`;
}

/** TS twin of the WGSL timeWordsCompare, operating on (low, high) uint32 words. */
export function compareTimeWords(
  a: readonly [number, number],
  b: readonly [number, number]
): -1 | 0 | 1 {
  const aHigh = (a[1] ^ 0x80000000) >>> 0;
  const bHigh = (b[1] ^ 0x80000000) >>> 0;
  if (aHigh !== bHigh) return aHigh > bHigh ? 1 : -1;
  if (a[0] !== b[0]) return a[0] > b[0] ? 1 : -1;
  return 0;
}

export function splitInt64Words(value: bigint): [number, number] {
  const unsigned = BigInt.asUintN(64, value);
  return [Number(unsigned & 0xffffffffn), Number(unsigned >> 32n)];
}

/** Line-for-line TS emulation of the generated WGSL predicate on words. */
export function emulateIntervalOverlapWords(
  start: readonly [number, number],
  end: readonly [number, number],
  hasEnd: boolean,
  windowStart: readonly [number, number],
  windowEnd: readonly [number, number],
  closure: IntervalBoundaryClosure = DEFAULT_INTERVAL_BOUNDARY_CLOSURE
): boolean {
  const {rowStartClosed, rowEndClosed, windowStartClosed, windowEndClosed} = closure;
  if (hasEnd && compareTimeWords(start, end) > 0) return false;
  if (hasEnd && compareTimeWords(start, end) === 0 && !(rowStartClosed && rowEndClosed))
    return false;
  if (compareTimeWords(windowStart, windowEnd) === 0 && !(windowStartClosed && windowEndClosed)) {
    return false;
  }
  const startOrder = compareTimeWords(start, windowEnd);
  const startsBeforeWindowEnd =
    startOrder < 0 || (startOrder === 0 && rowStartClosed && windowEndClosed);
  if (!startsBeforeWindowEnd) return false;
  if (!hasEnd) return true;
  const endOrder = compareTimeWords(windowStart, end);
  return endOrder < 0 || (endOrder === 0 && windowStartClosed && rowEndClosed);
}
