// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_CELL_DEFAULT_SUM_SCALE,
  getGPUCellSumScale,
  validateSumScale
} from '../../src/analysis-core/cell-key-wgsl';

// getGPUCellSumScale picks the fixed-point scale of cell-table, group and join sums from the data
// (math review finding core-numerics-03). Expected scales are worked out by hand from the rule:
// the largest power of two with maximumMagnitude * maximumRowCount * scale <= 2^61.

it('the default scale is 16 fractional bits', () => {
  expect(GPU_CELL_DEFAULT_SUM_SCALE).toBe(65536);
});

it('getGPUCellSumScale returns the largest power of two that leaves the 64-bit sum its headroom', () => {
  // 2^61 / (1 * 2^31) = 2^30.
  expect(getGPUCellSumScale(1)).toBe(2 ** 30);
  // 2^61 / (1 * 1) = 2^61.
  expect(getGPUCellSumScale(1, 1)).toBe(2 ** 61);
  // 2^61 / (1e-4 * 2^31) = 2^30 * 10^4 = 2^43.29: scale 2^43.
  expect(getGPUCellSumScale(1e-4)).toBe(2 ** 43);
  // 2^61 / (1000 * 2^20) = 2^41 / 1000 = 2^31.03: scale 2^31.
  expect(getGPUCellSumScale(1000, 2 ** 20)).toBe(2 ** 31);
  // An exact power of two on the limit is kept: 2^61 / (8 * 2^10) = 2^48.
  expect(getGPUCellSumScale(8, 1024)).toBe(2 ** 48);
  // Just above a power of two the scale drops by one bit.
  expect(getGPUCellSumScale(8.000001, 1024)).toBe(2 ** 47);
  // Scales stay normal float32 powers of two at either extreme.
  expect(getGPUCellSumScale(1e-35, 1)).toBe(2 ** 127);
  expect(getGPUCellSumScale(3e38, 2 ** 32)).toBe(2 ** -99);
});

it('the returned scale is accepted by validateSumScale and bounds the total below 2^61', () => {
  for (const magnitude of [1e-30, 3.3e-7, 1e-4, 0.5, 1, 7, 65535.5, 1e9, 1e30]) {
    for (const rowCount of [1, 2, 1000, 2 ** 24, 2 ** 31, 2 ** 32]) {
      const scale = getGPUCellSumScale(magnitude, rowCount);
      expect(() => validateSumScale('spec', scale)).not.toThrow();
      expect(Number.isInteger(Math.log2(scale))).toBe(true);
      expect(magnitude * rowCount * scale).toBeLessThanOrEqual(2 ** 61);
      if (scale < 2 ** 127) {
        // The next power of two would pass the headroom.
        expect(magnitude * rowCount * scale * 2).toBeGreaterThan(2 ** 61);
      }
    }
  }
});

it('getGPUCellSumScale names the argument it refuses', () => {
  expect(() => getGPUCellSumScale(0)).toThrow(/maximumMagnitude must be positive and finite/);
  expect(() => getGPUCellSumScale(Number.NaN)).toThrow(/maximumMagnitude/);
  expect(() => getGPUCellSumScale(Number.POSITIVE_INFINITY)).toThrow(/maximumMagnitude/);
  expect(() => getGPUCellSumScale(1, 0)).toThrow(
    /maximumRowCount must be an integer in \[1, 2\^32\]/
  );
  expect(() => getGPUCellSumScale(1, 1.5)).toThrow(/maximumRowCount/);
  expect(() => getGPUCellSumScale(1, 2 ** 33)).toThrow(/maximumRowCount/);
});
