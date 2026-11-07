// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  DEFAULT_INTERVAL_BOUNDARY_CLOSURE,
  classifyIntervalOverlap,
  emulateIntervalOverlapWords,
  makeIntervalOverlapWgsl,
  splitInt64Words,
  type IntervalBoundaryClosure
} from '../../../examples/deck/kepler-compute/src/time-contract-interval';

const window = {start: 10n, end: 20n};

describe('interval overlap oracle', () => {
  it('default [start,end) rows against [start,end] windows at every boundary', () => {
    const check = (start: bigint | null, end: bigint | null) =>
      classifyIntervalOverlap({start, end}, window);
    expect(check(0n, 9n)).toBe('disjoint');
    expect(check(0n, 10n)).toBe('disjoint'); // row end open touches window start
    expect(check(0n, 11n)).toBe('overlap');
    expect(check(20n, 30n)).toBe('overlap'); // row start closed touches closed window end
    expect(check(21n, 30n)).toBe('disjoint');
    expect(check(12n, 12n)).toBe('disjoint'); // empty [12,12)
    expect(check(5n, null)).toBe('overlap');
    expect(check(21n, null)).toBe('disjoint');
    expect(check(20n, null)).toBe('overlap');
    expect(check(30n, 20n)).toBe('invalid');
    expect(check(null, 20n)).toBe('invalid');
  });

  it('closed rows, open windows and zero-width cases', () => {
    const closedRow: IntervalBoundaryClosure = {
      ...DEFAULT_INTERVAL_BOUNDARY_CLOSURE,
      rowEndClosed: true
    };
    expect(classifyIntervalOverlap({start: 0n, end: 10n}, window, closedRow)).toBe('overlap');
    expect(classifyIntervalOverlap({start: 12n, end: 12n}, window, closedRow)).toBe('overlap');
    const openWindow: IntervalBoundaryClosure = {
      ...closedRow,
      windowStartClosed: false,
      windowEndClosed: false
    };
    expect(classifyIntervalOverlap({start: 0n, end: 10n}, window, openWindow)).toBe('disjoint');
    expect(classifyIntervalOverlap({start: 20n, end: 30n}, window, openWindow)).toBe('disjoint');
    expect(classifyIntervalOverlap({start: 0n, end: 30n}, {start: 15n, end: 15n}, openWindow)).toBe(
      'disjoint'
    );
    expect(classifyIntervalOverlap({start: 0n, end: 30n}, {start: 15n, end: 15n})).toBe('overlap');
    expect(() => classifyIntervalOverlap({start: 0n, end: 1n}, {start: 5n, end: 4n})).toThrow();
  });

  it('matches a brute-force instant model on a small integer domain', () => {
    // Interpret instants on a half-integer lattice so open/closed boundaries are observable.
    const toLattice = (value: bigint) => Number(value) * 2;
    const rowContains = (
      start: number,
      end: number | null,
      point: number,
      closure: IntervalBoundaryClosure
    ) =>
      (point > start || (point === start && closure.rowStartClosed)) &&
      (end === null || point < end || (point === end && closure.rowEndClosed));
    const windowContains = (
      start: number,
      end: number,
      point: number,
      c: IntervalBoundaryClosure
    ) =>
      (point > start || (point === start && c.windowStartClosed)) &&
      (point < end || (point === end && c.windowEndClosed));
    for (let mask = 0; mask < 16; mask++) {
      const closure: IntervalBoundaryClosure = {
        rowStartClosed: Boolean(mask & 1),
        rowEndClosed: Boolean(mask & 2),
        windowStartClosed: Boolean(mask & 4),
        windowEndClosed: Boolean(mask & 8)
      };
      for (let rowStart = 0n; rowStart <= 5n; rowStart++) {
        for (const rowEnd of [null, ...[0n, 1n, 2n, 3n, 4n, 5n]]) {
          for (let windowStart = 0n; windowStart <= 5n; windowStart++) {
            for (let windowEnd = windowStart; windowEnd <= 5n; windowEnd++) {
              const result = classifyIntervalOverlap(
                {start: rowStart, end: rowEnd},
                {start: windowStart, end: windowEnd},
                closure
              );
              if (rowEnd !== null && rowStart > rowEnd) {
                expect(result).toBe('invalid');
                continue;
              }
              // Dense model: test integer and half-integer points only; open ends need no more.
              let expected = false;
              for (let point = -2; point <= 22; point++) {
                // Dense points include epsilon neighbours via the 2x lattice (even = integer).
                if (
                  rowContains(
                    toLattice(rowStart),
                    rowEnd === null ? null : toLattice(rowEnd),
                    point,
                    closure
                  ) &&
                  windowContains(toLattice(windowStart), toLattice(windowEnd), point, closure)
                ) {
                  expected = true;
                  break;
                }
              }
              expect(result === 'overlap').toBe(expected);
            }
          }
        }
      }
    }
  });
});

describe('WGSL predicate', () => {
  it('declares the expected signature and reuses timeWordsCompare', () => {
    const source = makeIntervalOverlapWgsl('rowOverlapsWindow', {
      rowStartClosed: true,
      rowEndClosed: true,
      windowStartClosed: false,
      windowEndClosed: true
    });
    expect(source).toContain('fn rowOverlapsWindow(start: vec2<u32>');
    expect(source).toContain('timeWordsCompare');
    expect(source).toContain('const rowEndClosed = true;');
    expect(source).toContain('const windowStartClosed = false;');
    expect(makeIntervalOverlapWgsl()).toContain('fn isIntervalOverlappingWindow(');
  });

  it('word emulation equals the oracle across sign-crossing and 2^32 boundaries', () => {
    const values = [
      -(1n << 63n),
      -(1n << 32n) - 1n,
      -(1n << 32n),
      -1n,
      0n,
      1n,
      (1n << 32n) - 1n,
      1n << 32n,
      (1n << 32n) + 1n,
      (1n << 63n) - 1n
    ];
    for (let mask = 0; mask < 16; mask++) {
      const closure: IntervalBoundaryClosure = {
        rowStartClosed: Boolean(mask & 1),
        rowEndClosed: Boolean(mask & 2),
        windowStartClosed: Boolean(mask & 4),
        windowEndClosed: Boolean(mask & 8)
      };
      for (const start of values) {
        for (const end of [null, ...values]) {
          for (const windowStart of values) {
            for (const windowEnd of values) {
              if (windowStart > windowEnd) continue;
              const expected =
                classifyIntervalOverlap(
                  {start, end},
                  {start: windowStart, end: windowEnd},
                  closure
                ) === 'overlap';
              const actual = emulateIntervalOverlapWords(
                splitInt64Words(start),
                splitInt64Words(end ?? 0n),
                end !== null,
                splitInt64Words(windowStart),
                splitInt64Words(windowEnd),
                closure
              );
              expect(actual).toBe(expected);
            }
          }
        }
      }
    }
  });
});
