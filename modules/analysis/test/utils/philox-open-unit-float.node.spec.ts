// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getPhiloxOpenUnitFloat,
  getPhiloxUnitFloat
} from '../../src/analysis-core/kernels/philox-random';

// The open-interval uniform of analysis-core/kernels/philox-random.ts (math review finding
// core-numerics-05): the top 23 bits of the word plus one half, over 2^23. Expected values are
// exact binary fractions written out by hand.

it('the open unit float never reaches 0 or 1 and is an exact float32', () => {
  // Largest word: (2^23 - 1 + 1/2) / 2^23 = 1 - 2^-24.
  expect(getPhiloxOpenUnitFloat(0xffffffff)).toBe(1 - 2 ** -24);
  expect(getPhiloxOpenUnitFloat(0xffffffff)).toBeLessThan(1);
  // Smallest word: (0 + 1/2) / 2^23 = 2^-24.
  expect(getPhiloxOpenUnitFloat(0)).toBe(2 ** -24);
  expect(getPhiloxOpenUnitFloat(0)).toBeGreaterThan(0);
  // The low nine bits do not enter: words 0 to 511 share a value, word 512 is the next one.
  expect(getPhiloxOpenUnitFloat(511)).toBe(2 ** -24);
  expect(getPhiloxOpenUnitFloat(512)).toBe(3 * 2 ** -24);
  // The middle of the range: word 2^31 has top 23 bits 2^22, so (2^22 + 1/2) / 2^23.
  expect(getPhiloxOpenUnitFloat(0x80000000)).toBe(0.5 + 2 ** -24);
  for (const word of [0, 1, 511, 512, 0x7fffffff, 0x80000000, 0xfffffdff, 0xfffffe00, 0xffffffff]) {
    const value = getPhiloxOpenUnitFloat(word);
    expect(Math.fround(value), `word ${word}`).toBe(value);
    // log() of the float32 value is finite and strictly negative at both ends.
    expect(Math.fround(Math.log(value))).toBeLessThan(0);
    expect(Number.isFinite(Math.log(value))).toBe(true);
  }
});

it('the 24-bit open-interval form it replaces returns exactly 1 in float32', () => {
  // (f32(word >> 8) + 0.5) / 2^24 for the largest word: 16777215.5 is not a float32 and rounds
  // to 16777216, so the quotient is 1.0 and log(u) is 0.
  expect(Math.fround(Math.fround(0xffffffff >>> 8) + 0.5) / 16777216).toBe(1);
  // The half-open helper keeps its documented range [0, 1 - 2^-24].
  expect(getPhiloxUnitFloat(0)).toBe(0);
  expect(getPhiloxUnitFloat(0xffffffff)).toBe(1 - 2 ** -24);
});
