// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUKeplerTimeUnit} from '@luma-contributors/kepler';

/** BigInt oracle for the resident chart's unit normalization and Date-range sentinel. */
export function normalizeResidentCalendarTimeOnCPU(value: bigint, unit: GPUKeplerTimeUnit): bigint {
  if (unit === 'millisecond') return value;
  const nanosecondsPerUnit = {second: 1_000_000_000n, microsecond: 1000n, nanosecond: 1n};
  const nanoseconds = value * nanosecondsPerUnit[unit];
  const maximumNanoseconds = 8_640_000_000_000_000_000_000n;
  if (nanoseconds < -maximumNanoseconds || nanoseconds > maximumNanoseconds)
    return (1n << 63n) - 1n;
  // Independent BigInt floor division, including a negative fractional millisecond.
  return nanoseconds >= 0n ? nanoseconds / 1_000_000n : -((-nanoseconds + 999_999n) / 1_000_000n);
}
