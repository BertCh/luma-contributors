// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeZonalStatistics} from './zonal-statistics-oracle';

it('finite weighted inputs retain counts and denominator when their float32 product overflows (GEO-128)', () => {
  // Independently: 4 * 2^127 = 2^129 exceeds float32's maximum. Dropping this product
  // leaves numerator 2; finite-input count stays 2 and denominator 4 + 1 = 5.
  const result = computeZonalStatistics({
    featureRows: [0, 0],
    featureCount: 1,
    values: [2 ** 127, 2],
    weights: [4, 1]
  });
  expect(result.counts).toEqual([2]);
  expect(result.valueCounts).toEqual([2]);
  expect(result.sums).toEqual([2]);
  expect(result.weightSums).toEqual([5]);
  expect(result.means).toEqual([0.4]);
  expect(result.minima).toEqual([2]);
  expect(result.maxima).toEqual([2 ** 127]);
});
