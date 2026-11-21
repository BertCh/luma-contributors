// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeColumnStatistics} from './group-statistics-oracle';

it('B4 group reference wraps published words once and retains wide loss evidence', () => {
  const result = computeColumnStatistics(new Array(4).fill(2 ** 61), {
    variance: 'population',
    fractions: [],
    sumScale: 1
  });
  // 4*2^61=2^63: signed int64 publication is -2^63; the wide total proves numeric loss.
  expect(result.sum).toBe(-9223372036854775808n);
  expect(result.wideSum).toBe(9223372036854775808n);
  expect(result.sumOverflow).toBe(true);
});

it('B4 group reference computes float reductions from exact f32 units through cancellation', () => {
  const result = computeColumnStatistics([2 ** 56, 3, -(2 ** 56)], {
    variance: 'population',
    fractions: [],
    sumScale: 1
  });
  // Opposite exact powers cancel; 3 remains, and dividing by three gives 1.
  expect(result.sumValue).toBe(3);
  expect(result.mean).toBe(1);
  expect(result.sum).toBe(3n);
  expect(result.sumOverflow).toBe(false);
});
