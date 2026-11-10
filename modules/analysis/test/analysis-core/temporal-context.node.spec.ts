// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  assertCompatibleGPUTemporalContexts,
  getGPUTemporalContextCompatibility,
  GPU_IRREGULAR_TEMPORAL_AXIS,
  resolveGPUTemporalContext,
  validateGPUTemporalContext,
  type GPUTemporalContext
} from '../../src/analysis-core/contracts/index';

const WORDS: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};
const RELATIVE: GPUTemporalContext = {...WORDS, encoding: 'float32-relative', epoch: 1_000n};

describe('validateGPUTemporalContext', () => {
  const valid: [string, GPUTemporalContext][] = [
    ['Int64 words from the Unix epoch', WORDS],
    ['every unit', {...WORDS, unit: 'nanosecond'}],
    ['Int64 words from an origin', {...WORDS, epoch: -5n}],
    ['a declared time zone', {...WORDS, timeZone: '+05:30'}],
    ['float32 relative to an origin', RELATIVE],
    ['float32 relative to an explicit zero origin', {...RELATIVE, epoch: 0n}],
    ['double-single from the Unix epoch', {...WORDS, encoding: 'double-single'}],
    ['a fixed numeric step', {...WORDS, axis: {kind: 'regular', step: 3_600_000}}],
    ['a bigint step beyond 2^53', {...WORDS, axis: {kind: 'regular', step: 1n << 60n}}],
    ['a fractional step on float times', {...RELATIVE, axis: {kind: 'regular', step: 0.5}}],
    [
      'a calendar step',
      {...WORDS, axis: {kind: 'regular', step: {calendarUnit: 'month', interval: 3}}}
    ],
    [
      'a calendar step without an interval',
      {...WORDS, axis: {kind: 'regular', step: {calendarUnit: 'week'}}}
    ],
    ['the shared irregular axis', {...WORDS, axis: GPU_IRREGULAR_TEMPORAL_AXIS}]
  ];
  for (const [name, context] of valid) {
    it(`accepts ${name}`, () => {
      expect(() => validateGPUTemporalContext('test', context)).not.toThrow();
    });
  }

  const invalid: [string, GPUTemporalContext, RegExp][] = [
    ['an unknown unit', {...WORDS, unit: 'minute' as never}, /temporal unit minute/],
    ['an unknown encoding', {...WORDS, encoding: 'float64' as never}, /temporal encoding float64/],
    ['a non-bigint epoch', {...WORDS, epoch: 1000 as never}, /epoch must be/],
    ['an origin outside Int64', {...WORDS, epoch: 1n << 63n}, /epoch must be/],
    ['float32 relative to the Unix epoch', {...RELATIVE, epoch: 'unix'}, /bigint origin/],
    ['an empty time zone', {...WORDS, timeZone: ' '}, /timeZone/],
    ['a missing axis', {...WORDS, axis: undefined as never}, /regular or irregular/],
    ['an unknown axis kind', {...WORDS, axis: {kind: 'cyclic'} as never}, /regular or irregular/],
    ['a zero step', {...WORDS, axis: {kind: 'regular', step: 0}}, /positive/],
    ['a negative bigint step', {...WORDS, axis: {kind: 'regular', step: -1n}}, /positive/],
    ['a non-finite step', {...RELATIVE, axis: {kind: 'regular', step: Infinity}}, /positive/],
    ['a missing step', {...WORDS, axis: {kind: 'regular'} as never}, /positive/],
    [
      'a fractional step on Int64 words',
      {...WORDS, axis: {kind: 'regular', step: 0.5}},
      /safe integer/
    ],
    [
      'an unknown calendar unit',
      {...WORDS, axis: {kind: 'regular', step: {calendarUnit: 'fortnight' as never}}},
      /calendar step/
    ],
    [
      'a zero calendar interval',
      {...WORDS, axis: {kind: 'regular', step: {calendarUnit: 'day', interval: 0}}},
      /calendar step/
    ]
  ];
  for (const [name, context, message] of invalid) {
    it(`rejects ${name}`, () => {
      expect(() => validateGPUTemporalContext('test', context)).toThrow(message);
    });
  }
});

describe('resolveGPUTemporalContext', () => {
  it('returns the context it was given, unconverted', () => {
    expect(resolveGPUTemporalContext('test', WORDS)).toBe(WORDS);
  });

  it('validates before checking support', () => {
    expect(() =>
      resolveGPUTemporalContext(
        'test',
        {...RELATIVE, epoch: 'unix'},
        {encodings: ['float32-relative']}
      )
    ).toThrow(/bigint origin/);
  });

  it('checks each support declaration', () => {
    expect(() => resolveGPUTemporalContext('op', RELATIVE, {encodings: ['int64-words']})).toThrow(
      /op does not support float32-relative times; it reads int64-words/
    );
    expect(() =>
      resolveGPUTemporalContext('op', WORDS, {units: ['second', 'microsecond']})
    ).toThrow(/op does not support millisecond times; it reads second or microsecond/);
    expect(() => resolveGPUTemporalContext('op', WORDS, {axisKinds: ['regular']})).toThrow(
      /op does not support a irregular temporal axis/
    );
    expect(() => resolveGPUTemporalContext('op', {...WORDS, epoch: 7n}, {unixEpoch: true})).toThrow(
      /op needs times counted from the Unix epoch/
    );
  });

  it('accepts what the support declaration lists', () => {
    const support = {
      encodings: ['int64-words'],
      units: ['millisecond'],
      axisKinds: ['irregular'],
      unixEpoch: true
    } as const;
    expect(resolveGPUTemporalContext('op', WORDS, support)).toBe(WORDS);
    expect(resolveGPUTemporalContext('op', {...WORDS, epoch: 0n}, support).epoch).toBe(0n);
  });
});

describe('getGPUTemporalContextCompatibility', () => {
  it('accepts equal unit, epoch and encoding', () => {
    expect(getGPUTemporalContextCompatibility(WORDS, {...WORDS})).toEqual({compatible: true});
  });

  it('treats the Unix epoch and a zero origin as the same instant', () => {
    expect(getGPUTemporalContextCompatibility(WORDS, {...WORDS, epoch: 0n}).compatible).toBe(true);
  });

  it('ignores time zone and axis', () => {
    const other: GPUTemporalContext = {
      ...WORDS,
      timeZone: 'UTC',
      axis: {kind: 'regular', step: 1000}
    };
    expect(getGPUTemporalContextCompatibility(WORDS, other).compatible).toBe(true);
  });

  const mismatches: [string, GPUTemporalContext, RegExp][] = [
    ['unit', {...WORDS, unit: 'second'}, /units differ \(millisecond and second\)/],
    ['epoch', {...WORDS, epoch: 5n}, /epochs differ \(unix and 5\)/],
    [
      'encoding',
      {...WORDS, encoding: 'double-single'},
      /encodings differ \(int64-words and double-single\)/
    ]
  ];
  for (const [name, target, reason] of mismatches) {
    it(`reports a ${name} mismatch`, () => {
      const compatibility = getGPUTemporalContextCompatibility(WORDS, target);
      expect(compatibility.compatible).toBe(false);
      expect('reason' in compatibility && compatibility.reason).toMatch(reason);
      expect(() => assertCompatibleGPUTemporalContexts('op', WORDS, target)).toThrow(
        /op incompatible temporal contexts/
      );
    });
  }

  it('compares origins as exact Int64 values beyond 2^53', () => {
    const left = {...WORDS, epoch: (1n << 53n) + 1n};
    const right = {...WORDS, epoch: 1n << 53n};
    expect(getGPUTemporalContextCompatibility(left, right).compatible).toBe(false);
  });

  it('validates both sides', () => {
    expect(() =>
      getGPUTemporalContextCompatibility({...WORDS, unit: 'minute' as never}, WORDS)
    ).toThrow(/source temporal context temporal unit/);
    expect(() =>
      getGPUTemporalContextCompatibility(WORDS, {...WORDS, encoding: 'float64' as never})
    ).toThrow(/target temporal context temporal encoding/);
  });

  it('does not throw from assert when compatible', () => {
    expect(() => assertCompatibleGPUTemporalContexts('op', WORDS, WORDS)).not.toThrow();
  });
});
