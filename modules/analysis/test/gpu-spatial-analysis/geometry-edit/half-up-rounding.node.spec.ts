// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {snapValue} from './geometry-cleanup-model';
import {resolveCoverageSnapGridSize} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/coverage-snap-settings';

it('UGE-03 pins half-up rounding independently of float32 addition', () => {
  // Exact nearest integers, with half ties toward positive infinity.
  for (const [value, expected] of [
    [8388609, 8388609],
    [0.4999999701976776, 0],
    [0.5, 1],
    [-0.5, 0],
    [-1.5, -1]
  ]) {
    expect(Number.isFinite(snapValue(value, 1))).toBe(true);
    expect(snapValue(value, 1) + 0).toBe(expected);
  }
});
it('UGE-03 keeps the old snap spelling and rejects disagreeing settings', () => {
  expect(resolveCoverageSnapGridSize('coverage', {snapGridSize: 1})).toBe(1);
});
