// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import type {ArrowAnalyticsTimeUnit} from '@luma-contributors/analysis/analysis-core';
import {describe, expect, it} from 'vitest';
import {
  convertKeplerTimeInstant,
  type GPUKeplerTimeUnit,
  type GPUKeplerTimeWindow
} from '../src/selection/gpu-time-selection';

const SOURCE = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../src/selection/gpu-time-selection.ts'
);
const NANOSECONDS: Record<ArrowAnalyticsTimeUnit, bigint> = {
  second: 1_000_000_000n,
  millisecond: 1_000_000n,
  microsecond: 1_000n,
  nanosecond: 1n
};

// Compile-time checks: the shared unit and the deprecated alias are interchangeable.
const aliasUnit: GPUKeplerTimeUnit = 'second' satisfies ArrowAnalyticsTimeUnit;
const windowUnit: GPUKeplerTimeWindow['unit'] = aliasUnit satisfies ArrowAnalyticsTimeUnit;

describe('Kepler time selection uses the shared time unit', () => {
  it('no longer declares its own unit union', () => {
    const source = readFileSync(SOURCE, 'utf8');
    expect(source).toMatch(/GPUKeplerTimeUnit = ArrowAnalyticsTimeUnit;/);
    expect(source.match(/GPUKeplerTimeUnit/g)).toHaveLength(1);
    expect(source).not.toMatch(/= 'second' \| 'millisecond'/);
  });

  for (const unit of Object.keys(NANOSECONDS) as ArrowAnalyticsTimeUnit[]) {
    it(`converts ${unit} instants exactly`, () => {
      expect(convertKeplerTimeInstant(3n, unit, 'nanosecond')).toBe(3n * NANOSECONDS[unit]);
      expect(convertKeplerTimeInstant(3n * NANOSECONDS[unit], 'nanosecond', unit)).toBe(3n);
    });
  }

  it('still rejects units outside the shared type', () => {
    expect(() => convertKeplerTimeInstant(1n, windowUnit, 'fortnight' as never)).toThrow(
      /Unsupported time unit/
    );
  });
});
