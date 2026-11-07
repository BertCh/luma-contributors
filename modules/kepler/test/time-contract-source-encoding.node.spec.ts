// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  convertNanosecondsToEpochTicks,
  daysFromCivil,
  detectEpochUnitCandidates,
  parseTimeSourceValue,
  type TimeEncodingName
} from '../../../examples/deck/kepler-compute/src/time-contract-source-encoding';

const ok = (encoding: TimeEncodingName, value: number | bigint | string, options = {}) => {
  const result = parseTimeSourceValue(encoding, value, options);
  if (!result.ok) throw new Error(`unexpected failure ${result.reason}`);
  return result;
};
const fail = (encoding: TimeEncodingName, value: number | bigint | string | null) => {
  const result = parseTimeSourceValue(encoding, value);
  return result.ok ? null : result.reason;
};

describe('epoch encodings', () => {
  it('converts each unit to exact nanoseconds and microseconds', () => {
    expect(ok('epoch-seconds', 1_700_000_000).nanoseconds).toBe(1_700_000_000_000_000_000n);
    expect(ok('epoch-milliseconds', 1_700_000_000_123).microseconds).toBe(1_700_000_000_123_000n);
    expect(ok('epoch-microseconds', 1_700_000_000_123_456n).nanoseconds).toBe(
      1_700_000_000_123_456_000n
    );
    const nanoseconds = ok('epoch-nanoseconds', 1_700_000_000_123_456_789n);
    expect(nanoseconds.microseconds).toBe(1_700_000_000_123_456n);
    expect(nanoseconds.ambiguities).toContain('sub-microsecond-precision');
  });

  it('floors negative sub-microsecond values toward negative infinity', () => {
    const result = ok('epoch-nanoseconds', -1n);
    expect(result.microseconds).toBe(-1n);
    expect(result.nanoseconds).toBe(-1n);
  });

  it('reports float sources, unsafe integers, non-finite and Int64 overflow', () => {
    const fractional = ok('epoch-milliseconds', 1_700_000_000_123.456);
    expect(fractional.ambiguities).toContain('binary-float-source');
    expect(fractional.nanoseconds).toBe(1_700_000_000_123_456_000n);
    expect(fail('epoch-milliseconds', 2 ** 60)).toBe('unsafe-integer');
    expect(fail('epoch-milliseconds', Number.NaN)).toBe('not-finite');
    expect(fail('epoch-seconds', 10n ** 17n)).toBe('out-of-range');
    expect(ok('epoch-seconds', 10n ** 11n).fitsInt64Nanoseconds).toBe(false);
    expect(ok('epoch-seconds', 1_700_000_000).fitsInt64Nanoseconds).toBe(true);
    expect(fail('epoch-seconds', null)).toBe('null-value');
    expect(fail('epoch-seconds', '12')).toBe('wrong-value-type');
    expect(fail('epoch-seconds', 0.0000000001)).toBe('sub-nanosecond');
  });

  it('flags unit-magnitude suspicion and naive timestamps', () => {
    expect(ok('epoch-seconds', 1_700_000_000_000).ambiguities).toContain(
      'epoch-unit-magnitude-suspect'
    );
    expect(ok('epoch-seconds', 1_700_000_000).ambiguities).toEqual([]);
    expect(ok('epoch-milliseconds', 5, {timezoneDeclared: false}).ambiguities).toContain(
      'naive-timestamp-assumed-utc'
    );
  });

  it('detects candidate units; one value can be ambiguous across none or many', () => {
    expect(detectEpochUnitCandidates(1_700_000_000)).toEqual(['epoch-seconds']);
    expect(detectEpochUnitCandidates(1_700_000_000_000)).toEqual(['epoch-milliseconds']);
    expect(detectEpochUnitCandidates(1_700_000_000_000_000n)).toEqual(['epoch-microseconds']);
    expect(detectEpochUnitCandidates(1_700_000_000_000_000_000n)).toEqual(['epoch-nanoseconds']);
    expect(detectEpochUnitCandidates(5)).toEqual([]);
  });

  it('round trips nanoseconds to epoch ticks and reports truncation', () => {
    expect(convertNanosecondsToEpochTicks(1_500_000_000n, 'epoch-seconds')).toEqual({
      ticks: 1n,
      exact: false
    });
    expect(convertNanosecondsToEpochTicks(-1n, 'epoch-microseconds')).toEqual({
      ticks: -1n,
      exact: false
    });
    expect(convertNanosecondsToEpochTicks(2_000_000n, 'epoch-milliseconds')).toEqual({
      ticks: 2n,
      exact: true
    });
  });
});

describe('ISO 8601 strings', () => {
  it('applies offsets exactly and keeps nanosecond fractions', () => {
    expect(ok('iso8601', '1970-01-01T00:00:00Z').nanoseconds).toBe(0n);
    expect(ok('iso8601', '2024-03-10T01:30:00-05:00').nanoseconds).toBe(
      BigInt(Date.parse('2024-03-10T06:30:00Z')) * 1_000_000n
    );
    expect(ok('iso8601', '2024-03-10T01:30:00+0530').nanoseconds).toBe(
      BigInt(Date.parse('2024-03-09T20:00:00Z')) * 1_000_000n
    );
    expect(ok('iso8601', '2024-03-10 01:30:00.123456789Z').nanoseconds % 1_000_000_000n).toBe(
      123_456_789n
    );
    expect(ok('iso8601', '2024-03-10T01:30Z').microseconds).toBe(
      BigInt(Date.parse('2024-03-10T01:30:00Z')) * 1000n
    );
  });

  it('agrees with Date.parse across a calendar sweep including leap days and negative years', () => {
    for (let day = 0; day < 800; day++) {
      const date = new Date(Date.UTC(2023, 0, 1) + day * 86_400_000 + 12_345_678);
      const text = date.toISOString();
      expect(ok('iso8601', text).nanoseconds).toBe(BigInt(date.getTime()) * 1_000_000n);
    }
    expect(ok('iso8601', '-000001-01-01T00:00:00Z').nanoseconds).toBe(
      BigInt(Date.parse('-000001-01-01T00:00:00Z')) * 1_000_000n
    );
    expect(ok('iso8601', '+010000-01-01T00:00:00Z').nanoseconds).toBe(
      BigInt(Date.parse('+010000-01-01T00:00:00Z')) * 1_000_000n
    );
  });

  it('reports ambiguity for missing offsets, date-only values and long fractions', () => {
    const local = ok('iso8601', '2024-03-10T01:30:00', {assumedOffsetSeconds: -18000});
    expect(local.ambiguities).toEqual(['local-time-without-offset']);
    expect(local.nanoseconds).toBe(BigInt(Date.parse('2024-03-10T06:30:00Z')) * 1_000_000n);
    expect(ok('iso8601', '2024-03-10').ambiguities).toEqual([
      'date-only-interpreted-as-midnight',
      'local-time-without-offset'
    ]);
    expect(ok('iso8601', '2024-03-10T00:00:00.1234567891Z').ambiguities).toContain(
      'fraction-truncated-to-nanoseconds'
    );
  });

  it('refuses invalid dates, leap seconds and malformed strings', () => {
    expect(fail('iso8601', '2023-02-29T00:00:00Z')).toBe('invalid-calendar-date');
    expect(ok('iso8601', '2024-02-29T00:00:00Z').ok).toBe(true);
    expect(fail('iso8601', '2024-13-01')).toBe('invalid-calendar-date');
    expect(fail('iso8601', '2024-12-31T23:59:60Z')).toBe('leap-second');
    expect(fail('iso8601', '2024-01-01T24:00:00Z')).toBe('invalid-time-of-day');
    expect(fail('iso8601', 'yesterday')).toBe('malformed');
    expect(fail('iso8601', '2024-01-01T00:00:00+25:00')).toBe('malformed');
    expect(fail('iso8601', 5)).toBe('wrong-value-type');
    expect(fail('iso8601', '+292278994-01-01T00:00:00Z')).toBe('malformed');
  });

  it('counts civil days exactly', () => {
    expect(daysFromCivil(1970n, 1, 1)).toBe(0n);
    expect(daysFromCivil(1969n, 12, 31)).toBe(-1n);
    expect(daysFromCivil(2000n, 3, 1)).toBe(11017n);
  });
});

describe('arrow date, time and duration', () => {
  it('date32 is days, date64 is milliseconds with a midnight check', () => {
    expect(ok('date32', 19_723).nanoseconds).toBe(BigInt(Date.parse('2024-01-01Z')) * 1_000_000n);
    expect(fail('date32', 2 ** 31)).toBe('out-of-range');
    expect(ok('date64', 86_400_000).ambiguities).toEqual([]);
    expect(ok('date64', 86_400_001).ambiguities).toContain('date64-not-midnight');
  });

  it('time32/time64 are time-of-day only and bounded to a day', () => {
    const noon = ok('time32-second', 43_200);
    expect(noon.kind).toBe('time-of-day');
    expect(noon.nanoseconds).toBe(43_200_000_000_000n);
    expect(fail('time32-second', 86_400)).toBe('out-of-range');
    expect(fail('time32-millisecond', -1)).toBe('out-of-range');
    expect(ok('time64-microsecond', 86_399_999_999n).microseconds).toBe(86_399_999_999n);
    expect(fail('time64-nanosecond', 86_400_000_000_000n)).toBe('out-of-range');
  });

  it('duration is signed and keeps units', () => {
    const negative = ok('duration-millisecond', -1500);
    expect(negative.kind).toBe('duration');
    expect(negative.nanoseconds).toBe(-1_500_000_000n);
    expect(ok('duration-nanosecond', 7n).ambiguities).toContain('sub-microsecond-precision');
    expect(fail('duration-second', 10n ** 17n)).toBe('out-of-range');
  });
});
