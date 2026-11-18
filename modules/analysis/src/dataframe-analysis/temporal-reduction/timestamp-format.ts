// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Numeric/statistics presentation: truncate sub-millisecond ticks toward zero before
 * narrowing to Number. Source ticks and exact copy/export text remain unchanged.
 * Arrow units are 0=seconds, 1=milliseconds, 2=microseconds, 3=nanoseconds.
 */
export function getTimestampPresentationMilliseconds(value: bigint, unit: 0 | 1 | 2 | 3): number {
  return Number(unit === 0 ? value * 1000n : value / [1n, 1n, 1000n, 1_000_000n][unit]);
}

/**
 * Calendar display in the containing epoch millisecond, floored before narrowing to Number.
 * Negative fractional milliseconds stay on the same day as exact formatting and GPUCalendarBuckets.
 * Arrow units are 0=seconds, 1=milliseconds, 2=microseconds, 3=nanoseconds.
 * Throws RangeError for source instants outside Date's inclusive +/-8,640,000,000,000,000 ms range,
 * before narrowing can lose a millisecond. For wider calendar grouping, floor days in bigint first.
 */
export function getTimestampCalendarMilliseconds(value: bigint, unit: 0 | 1 | 2 | 3): number {
  // Date's inclusive range is below 2^53 milliseconds, preserving every containing millisecond.
  const maximumMilliseconds = 8_640_000_000_000_000n;
  const divisor = [1n, 1n, 1000n, 1_000_000n][unit];
  const scaledValue = unit === 0 ? value * 1000n : value;
  const maximumTicks = maximumMilliseconds * divisor;
  if (scaledValue < -maximumTicks || scaledValue > maximumTicks)
    throw new RangeError(
      'getTimestampCalendarMilliseconds value must be within the inclusive Date range of +/-8640000000000000 milliseconds'
    );
  const milliseconds = scaledValue / divisor;
  return Number(milliseconds - (scaledValue % divisor < 0n ? 1n : 0n));
}

/** Lossless ISO timestamp text from exact Arrow ticks (0=seconds, 1=ms, 2=us, 3=ns).
 * Values outside Date range retain exact ticks and their unit as epoch text. */
export function formatTimestampTicks(value: bigint, unit: 0 | 1 | 2 | 3): string {
  const scale = [1_000_000_000n, 1_000_000n, 1_000n, 1n][unit];
  const nanoseconds = value * scale;
  let seconds = nanoseconds / 1_000_000_000n;
  let fraction = nanoseconds % 1_000_000_000n;
  if (fraction < 0) {
    seconds--;
    fraction += 1_000_000_000n;
  }
  const milliseconds = Number(seconds * 1000n);
  const instant = new Date(milliseconds);
  if (!Number.isSafeInteger(milliseconds) || !Number.isFinite(instant.getTime()))
    return `${value} ${['s', 'ms', 'us', 'ns'][unit]} since Unix epoch`;
  return instant
    .toISOString()
    .replace(
      '.000Z',
      `${fraction ? `.${fraction.toString().padStart(9, '0').replace(/0+$/, '')}` : ''}Z`
    );
}
