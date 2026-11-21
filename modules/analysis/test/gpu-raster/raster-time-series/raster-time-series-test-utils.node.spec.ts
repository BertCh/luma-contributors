// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getMaximumAbsoluteDifference} from './raster-time-series-test-utils';

it('rejects an empty or shortened actual series instead of reporting zero error', () => {
  expect(() => getMaximumAbsoluteDifference([], [1, 2, 3])).toThrow(/lengths differ \(0 and 3\)/);
  expect(() => getMaximumAbsoluteDifference([1, 2], [1, 2, 3])).toThrow(/lengths differ/);
  expect(() => getMaximumAbsoluteDifference([1, 2, 3, 4], [1, 2, 3])).toThrow(/lengths differ/);
});

it('fails every tolerance when only one side is NaN', () => {
  expect(getMaximumAbsoluteDifference([1, Number.NaN, 3], [1, 2, 3])).toBe(Infinity);
  expect(getMaximumAbsoluteDifference([1, 2, 3], [1, 2, Number.NaN])).toBe(Infinity);
  // A one-sided NaN after a finite difference is still reported, not hidden by the maximum.
  expect(getMaximumAbsoluteDifference([1.5, Number.NaN], [1, 2])).toBe(Infinity);
});

it('keeps matching NaN and infinities equal and reports the largest finite difference', () => {
  expect(
    getMaximumAbsoluteDifference(
      [Number.NaN, Infinity, -Infinity],
      [Number.NaN, Infinity, -Infinity]
    )
  ).toBe(0);
  expect(getMaximumAbsoluteDifference([Infinity], [-Infinity])).toBe(Infinity);
  // |1.25 - 1| = 0.25 and |-2 - (-2.5)| = 0.5, so the maximum is 0.5.
  expect(getMaximumAbsoluteDifference(Float32Array.of(1.25, -2, 7), [1, -2.5, 7])).toBe(0.5);
});
