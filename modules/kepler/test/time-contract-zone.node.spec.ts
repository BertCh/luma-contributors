// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  buildZoneOffsetTable,
  classifyLocalMinuteOfDayFromWords,
  classifyLocalTime,
  convertLocalToUtcMilliseconds,
  getLocalBucketEdges,
  getLocalBucketKey,
  getZoneOffsetSeconds,
  packZoneOffsetTable,
  parseTimeZoneSpecification
} from '../../../examples/deck/kepler-compute/src/time-contract-zone';

const newYork = parseTimeZoneSpecification('America/New_York');
const utc = (text: string): bigint => BigInt(Date.parse(text));
const localAsUtc = (text: string): bigint => BigInt(Date.parse(`${text}Z`));

describe('zone specification', () => {
  it('parses utc, fixed offsets and iana names and rejects bad input', () => {
    expect(parseTimeZoneSpecification('UTC')).toEqual({kind: 'utc'});
    expect(parseTimeZoneSpecification('+05:30')).toEqual({kind: 'fixed', offsetSeconds: 19800});
    expect(parseTimeZoneSpecification('-0800')).toEqual({kind: 'fixed', offsetSeconds: -28800});
    expect(parseTimeZoneSpecification('+05')).toEqual({kind: 'fixed', offsetSeconds: 18000});
    expect(newYork).toEqual({kind: 'iana', name: 'America/New_York'});
    expect(() => parseTimeZoneSpecification('+24:00')).toThrow();
    expect(() => parseTimeZoneSpecification('Mars/Olympus')).toThrow();
  });

  it('reads exact offsets around the 2024 New York transitions', () => {
    expect(getZoneOffsetSeconds(newYork, utc('2024-03-10T06:59:59Z'))).toBe(-18000);
    expect(getZoneOffsetSeconds(newYork, utc('2024-03-10T07:00:00Z'))).toBe(-14400);
    expect(getZoneOffsetSeconds(newYork, utc('2024-11-03T05:59:59Z'))).toBe(-14400);
    expect(getZoneOffsetSeconds(newYork, utc('2024-11-03T06:00:00Z'))).toBe(-18000);
    expect(getZoneOffsetSeconds(parseTimeZoneSpecification('Asia/Kolkata'), 0n)).toBe(19800);
    expect(
      getZoneOffsetSeconds(
        parseTimeZoneSpecification('Asia/Kathmandu'),
        utc('2024-01-01T00:00:00Z')
      )
    ).toBe(20700);
  });
});

describe('local to utc resolution', () => {
  it('uses the documented gap and overlap choices', () => {
    const gap = convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-03-10T02:30:00'));
    expect(gap).toEqual({kind: 'gap', utcMilliseconds: utc('2024-03-10T07:30:00Z')});
    const gapEarlier = convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-03-10T02:30:00'), {
      gap: 'earlier',
      overlap: 'earlier'
    });
    expect(gapEarlier).toEqual({kind: 'gap', utcMilliseconds: utc('2024-03-10T06:30:00Z')});
    const overlap = convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-11-03T01:30:00'));
    expect(overlap).toEqual({kind: 'overlap', utcMilliseconds: utc('2024-11-03T05:30:00Z')});
    const overlapLater = convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-11-03T01:30:00'), {
      gap: 'later',
      overlap: 'later'
    });
    expect(overlapLater).toEqual({kind: 'overlap', utcMilliseconds: utc('2024-11-03T06:30:00Z')});
    expect(
      convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-03-10T02:30:00'), {
        gap: 'reject',
        overlap: 'reject'
      })
    ).toEqual({kind: 'rejected', reason: 'gap'});
    expect(
      convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-11-03T01:30:00'), {
        gap: 'reject',
        overlap: 'reject'
      })
    ).toEqual({kind: 'rejected', reason: 'overlap'});
    expect(convertLocalToUtcMilliseconds(newYork, localAsUtc('2024-07-01T12:00:00'))).toEqual({
      kind: 'unique',
      utcMilliseconds: utc('2024-07-01T16:00:00Z')
    });
  });
});

describe('offset table', () => {
  const start = utc('2024-01-01T00:00:00Z');
  const end = utc('2025-01-01T00:00:00Z');
  const table = buildZoneOffsetTable(newYork, start, end);

  it('finds exactly the two 2024 transitions', () => {
    expect(table.transitionMilliseconds).toEqual([
      utc('2024-03-10T07:00:00Z'),
      utc('2024-11-03T06:00:00Z')
    ]);
    expect(table.offsetSeconds).toEqual([-18000, -14400, -18000]);
  });

  it('classification matches Intl for sampled instants including edges', () => {
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      hourCycle: 'h23',
      hour: 'numeric',
      minute: 'numeric'
    });
    const samples: bigint[] = [];
    for (let step = 0n; step < 366n * 24n * 4n; step += 7n) samples.push(start + step * 900_000n);
    for (const transition of table.transitionMilliseconds)
      for (const delta of [-1n, 0n, 1n, -3_600_000n, 3_600_000n]) samples.push(transition + delta);
    const words = packZoneOffsetTable(table);
    for (const instant of samples) {
      const fields = classifyLocalTime(table, instant);
      const parts = formatter.formatToParts(new Date(Number(instant)));
      const hour = Number(parts.find(part => part.type === 'hour')!.value) % 24;
      const minute = Number(parts.find(part => part.type === 'minute')!.value);
      expect(fields.localMinuteOfDay).toBe(hour * 60 + minute);
      const unsigned = BigInt.asUintN(64, instant);
      const emulated = classifyLocalMinuteOfDayFromWords(
        words,
        Number(unsigned & 0xffffffffn),
        Number(unsigned >> 32n)
      );
      expect(emulated.localMinuteOfDay).toBe(fields.localMinuteOfDay);
      expect(emulated.offsetSeconds).toBe(fields.offsetSeconds);
      expect(emulated.localDayIndex).toBe(fields.localDayIndex);
    }
  });

  it('packs Int64 words and offsets', () => {
    const words = packZoneOffsetTable(table);
    expect(Array.from(words.offsets)).toEqual([-18000, -14400, -18000]);
    expect(words.transitionWords.length).toBe(4);
    const rebuilt = (BigInt(words.transitionWords[1]) << 32n) | BigInt(words.transitionWords[0]);
    expect(rebuilt).toBe(table.transitionMilliseconds[0]);
  });

  it('day edges give 23 and 25 hour local days and hour edges repeat on fall back', () => {
    const dayEdges = getLocalBucketEdges(
      table,
      'day',
      utc('2024-03-09T00:00:00Z'),
      utc('2024-03-12T00:00:00Z')
    );
    expect(dayEdges).toEqual([
      utc('2024-03-09T05:00:00Z'),
      utc('2024-03-10T05:00:00Z'),
      utc('2024-03-11T04:00:00Z')
    ]);
    expect(dayEdges[2] - dayEdges[1]).toBe(23n * 3_600_000n);
    const fall = getLocalBucketEdges(
      table,
      'day',
      utc('2024-11-02T00:00:00Z'),
      utc('2024-11-05T00:00:00Z')
    );
    expect(fall[2] - fall[1]).toBe(25n * 3_600_000n);
    const hourEdges = getLocalBucketEdges(
      table,
      'hour',
      utc('2024-11-03T04:00:00Z'),
      utc('2024-11-03T09:00:00Z')
    );
    expect(hourEdges.length).toBe(5);
    expect(
      getLocalBucketKey(table, utc('2024-11-03T05:30:00Z'), 'hour') ===
        getLocalBucketKey(table, utc('2024-11-03T06:30:00Z'), 'hour')
    ).toBe(true);
    const springHours = getLocalBucketEdges(
      table,
      'hour',
      utc('2024-03-10T05:00:00Z'),
      utc('2024-03-10T09:00:00Z')
    );
    // Local 02:00 does not exist: 05:00Z(00:00) 06:00Z(01:00) 07:00Z(03:00) 08:00Z(04:00).
    expect(springHours).toEqual([
      utc('2024-03-10T05:00:00Z'),
      utc('2024-03-10T06:00:00Z'),
      utc('2024-03-10T07:00:00Z'),
      utc('2024-03-10T08:00:00Z')
    ]);
  });

  it('half-hour zones align minute buckets and hour edges to local time', () => {
    const kolkata = parseTimeZoneSpecification('Asia/Kolkata');
    const kolkataTable = buildZoneOffsetTable(kolkata, 0n, 86_400_000n * 10n);
    expect(kolkataTable.transitionMilliseconds).toEqual([]);
    const edges = getLocalBucketEdges(kolkataTable, 'hour', 0n, 3n * 3_600_000n);
    expect(edges).toEqual([1_800_000n, 5_400_000n, 9_000_000n]);
    expect(classifyLocalTime(kolkataTable, 0n).localMinuteOfDay).toBe(330);
  });

  it('negative epochs floor into the previous local day', () => {
    const fields = classifyLocalTime(table, -1n);
    expect(fields.localDayIndex).toBe(-1n);
    expect(fields.localMinuteOfDay).toBe(1439 - 300);
  });
});
