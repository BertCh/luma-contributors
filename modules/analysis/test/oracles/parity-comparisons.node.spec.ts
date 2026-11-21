// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, it, vi} from 'vitest';
import {compareParity as compareWaveOne} from './parity-wp1-compare';
import {compareRasterParity} from './parity-wp4-compare';
import {compareParity as compareWaveSix} from './parity-wp6-compare';

const comparisons = [
  {
    name: 'WP1',
    soft: true,
    compare: (actual: number[], expected: number[], exact = false) =>
      compareWaveOne(
        'Example',
        {name: 'literal', reference: 'hand-worked', n: 2},
        'values',
        actual,
        expected,
        exact ? 'E' : 'S'
      )
  },
  {
    name: 'WP4',
    soft: false,
    compare: (actual: number[], expected: number[], exact = false) =>
      compareRasterParity(
        'Example',
        'literal',
        'hand-worked',
        'values',
        actual,
        expected,
        exact ? 'E' : 'S'
      )
  },
  {
    name: 'WP6',
    soft: true,
    compare: (actual: number[], expected: number[], exact = false) =>
      compareWaveSix('Example', 'literal', 'hand-worked', 2, 'values', actual, expected, exact)
  }
];

afterEach(() => vi.restoreAllMocks());

it.each(
  comparisons
)('$name compares finite values, nonfinite masks and exact row counts through its real comparator', ({
  compare,
  soft
}) => {
  // Turn soft assertions into catchable assertions for negative controls; spy verifies their use.
  const softAssertion = vi.spyOn(expect, 'soft').mockImplementation(expect);
  const log = vi.spyOn(console, 'log').mockImplementation(() => {});
  const cases = [
    {name: 'literal finite', actual: [1, 2], expected: [1, 2], mismatches: 0},
    {name: 'unexpected NaN', actual: [1, NaN], expected: [1, 2], mismatches: 1},
    {name: 'expected NaN', actual: [1, NaN], expected: [1, NaN], mismatches: 0},
    {name: 'opposite infinities', actual: [Infinity, 2], expected: [-Infinity, 2], mismatches: 1},
    {name: 'missing row', actual: [1], expected: [1, 2], mismatches: 1},
    {name: 'extra row', actual: [1, 2, 3], expected: [1, 2], mismatches: 1}
  ];
  for (const fixture of cases) {
    if (fixture.mismatches)
      expect(() => compare(fixture.actual, fixture.expected), fixture.name).toThrow();
    else expect(() => compare(fixture.actual, fixture.expected), fixture.name).not.toThrow();
    const message = JSON.parse(String(log.mock.lastCall?.[1])) as Record<string, unknown>;
    expect(message, fixture.name).toMatchObject({
      class: 'Example',
      case: 'literal',
      reference: 'hand-worked',
      metric: 'values',
      mismatches: fixture.mismatches
    });
    expect(Number.isFinite(message['maxAbs'])).toBe(true);
    expect(message).toHaveProperty('maxRel');
    expect(message).toHaveProperty('tolerance');
  }
  expect(softAssertion.mock.calls.length > 0).toBe(soft);
});

it.each(comparisons)('$name retains approximate tolerance and integer-exact mode', ({compare}) => {
  vi.spyOn(expect, 'soft').mockImplementation(expect);
  vi.spyOn(console, 'log').mockImplementation(() => {});
  // At reference 2 the S bound is 0.000002 + 0.0002; these literals straddle it.
  expect(() => compare([1, 2.0001], [1, 2])).not.toThrow();
  expect(() => compare([1, 2.001], [1, 2])).toThrow();
  expect(() => compare([1, 2.0001], [1, 2], true)).toThrow();
  expect(() => compare([1, 2], [1, 2], true)).not.toThrow();
});

it('retains global versus per-element scale, WP1 overrides and WP4 L residual diagnostics', () => {
  vi.spyOn(expect, 'soft').mockImplementation(expect);
  const log = vi.spyOn(console, 'log').mockImplementation(() => {});
  expect(() => comparisons[0].compare([1.5, 1e6], [1, 1e6])).not.toThrow();
  expect(() => comparisons[1].compare([1.5, 1e6], [1, 1e6])).not.toThrow();
  expect(() => comparisons[2].compare([1.5, 1e6], [1, 1e6])).toThrow();
  expect(() =>
    compareWaveOne(
      'Example',
      {name: 'literal', reference: 'hand-worked', n: 1},
      'values',
      [1.5],
      [1],
      'S',
      1,
      0
    )
  ).not.toThrow();
  compareRasterParity('Example', 'literal', 'hand-worked', 'values', [1, 2], [1, 2], 'L');
  expect(JSON.parse(String(log.mock.lastCall?.[1]))['tolerance']).toEqual({
    profile: 'L',
    absolute: 0.000002,
    relative: 0.0001,
    scaledResidual: 0.0001
  });
});

it('rejects an L residual that passes the S elementwise bound (S1)', () => {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  // S allows 0.001 + 0.1 = 0.101; L residual 0.1005 / 1000 = 0.0001005 exceeds 0.0001.
  expect(() =>
    compareRasterParity('Example', 'residual', 'hand-worked', 'values', [1000.1005], [1000], 'S')
  ).not.toThrow();
  expect(() =>
    compareRasterParity('Example', 'residual', 'hand-worked', 'values', [1000.1005], [1000], 'L')
  ).toThrow(/scaled residual/);
});
