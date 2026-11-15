// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Numeric calendar/statistics presentation: truncate sub-millisecond ticks toward zero before
 * narrowing to Number. Source ticks and exact copy/export text remain unchanged.
 * Arrow units are 0=seconds, 1=milliseconds, 2=microseconds, 3=nanoseconds.
 */
export function getTimestampPresentationMilliseconds(value: bigint, unit: 0 | 1 | 2 | 3): number {
  return Number(unit === 0 ? value * 1000n : value / [1n, 1n, 1000n, 1_000_000n][unit]);
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
