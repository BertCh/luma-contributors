// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  formatTimestampTicks,
  getTimestampCalendarMilliseconds,
  getTimestampPresentationMilliseconds
} from '../../../src/dataframe-analysis/temporal-reduction/index';

describe('calendar timestamp presentation', () => {
  it('rejects Int64 ticks that would round across a pre-epoch midnight', () => {
    const midnight = -9_072_000_000_000_000_000n;
    // Exact day arithmetic distinguishes the microsecond immediately before this midnight.
    expect((midnight - 1n) / 86_400_000_000n - 1n).toBe(-105_000_001n);
    expect(midnight / 86_400_000_000n).toBe(-105_000_000n);
    expect(midnight - 1n).toBeGreaterThan(-(1n << 63n));
    // Number cannot distinguish the exact containing millisecond from the next midnight.
    expect(Number((midnight - 1n) / 1000n - 1n)).toBe(Number(midnight / 1000n));
    expect(() => getTimestampCalendarMilliseconds(midnight - 1n, 2)).toThrow(RangeError);
    expect(() => getTimestampCalendarMilliseconds(midnight, 2)).toThrow(RangeError);
  });

  it('supports Date endpoints and refuses source ticks outside them before narrowing', () => {
    const dateLimit = 8_640_000_000_000_000n;
    for (const unit of [0, 1, 2, 3] as const) {
      const ticksPerMillisecond = [0n, 1n, 1000n, 1_000_000n][unit];
      const maximumTicks = unit === 0 ? dateLimit / 1000n : dateLimit * ticksPerMillisecond;
      expect(getTimestampCalendarMilliseconds(maximumTicks, unit)).toBe(Number(dateLimit));
      expect(getTimestampCalendarMilliseconds(-maximumTicks, unit)).toBe(Number(-dateLimit));
      expect(getTimestampCalendarMilliseconds(maximumTicks - 1n, unit)).toBe(
        Number(dateLimit - (unit === 0 ? 1000n : 1n))
      );
      expect(getTimestampCalendarMilliseconds(-maximumTicks + 1n, unit)).toBe(
        Number(-dateLimit + (unit === 0 ? 1000n : unit === 1 ? 1n : 0n))
      );
      expect(() => getTimestampCalendarMilliseconds(maximumTicks + 1n, unit)).toThrow(/Date range/);
      expect(() => getTimestampCalendarMilliseconds(-maximumTicks - 1n, unit)).toThrow(
        /Date range/
      );
    }
  });

  for (const unit of [2, 3] as const) {
    it(`keeps negative and positive fractions on the exact calendar day for Arrow unit ${unit}`, () => {
      const scale = unit === 2 ? 1000n : 1_000_000n;
      const ticks = [-scale - 1n, -scale, -1n, 0n, 1n, scale, scale + 1n];
      const milliseconds = ticks.map(value => getTimestampCalendarMilliseconds(value, unit));
      expect(milliseconds).toEqual([-2, -1, -1, 0, 0, 1, 1]);
      for (let index = 0; index < ticks.length; index++) {
        const exact = formatTimestampTicks(ticks[index], unit);
        expect(new Date(milliseconds[index]).toISOString().slice(0, 10)).toBe(exact.slice(0, 10));
        expect(Math.floor(milliseconds[index] / 86_400_000)).toBe(ticks[index] < 0n ? -1 : 0);
      }
      // Retain the older numeric conversion for callers who explicitly need truncation.
      expect(ticks.map(value => getTimestampPresentationMilliseconds(value, unit))).toEqual([
        -1, -1, 0, 0, 0, 1, 1
      ]);
    });
  }

  it('retains whole seconds and milliseconds on either side of the epoch', () => {
    expect([-1n, 0n, 1n].map(value => getTimestampCalendarMilliseconds(value, 0))).toEqual([
      -1000, 0, 1000
    ]);
    expect([-1n, 0n, 1n].map(value => getTimestampCalendarMilliseconds(value, 1))).toEqual([
      -1, 0, 1
    ]);
  });
});
