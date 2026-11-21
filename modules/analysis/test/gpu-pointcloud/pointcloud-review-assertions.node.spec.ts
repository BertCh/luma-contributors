// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectPointCloudChangeBounds,
  expectPointCloudICPConverged
} from './pointcloud-review-assertions';

it('accepts rounded bounds but rejects a systematic 0.01 shift (S2)', () => {
  // Hand calculation: distance 3 and half-width 2 give bounds [1,5].
  expect(() => expectPointCloudChangeBounds([3, 1.0008, 4.9992], 2.0005, 3, 2)).not.toThrow();
  expect(() => expectPointCloudChangeBounds([3, 1.01, 5.01], 2, 3, 2)).toThrow();
});

it('rejects a wrong or nonfinite published detection limit (S2)', () => {
  expect(() => expectPointCloudChangeBounds([3, 1, 5], 2.01, 3, 2)).toThrow();
  expect(() => expectPointCloudChangeBounds([3, 1, 5], NaN, 3, 2)).toThrow();
});

it('rejects nonfinite confidence bounds (S2)', () => {
  expect(() => expectPointCloudChangeBounds([3, NaN, 5], 2, 3, 2)).toThrow();
  expect(() => expectPointCloudChangeBounds([3, 1, Infinity], 2, 3, 2)).toThrow();
});

it('requires exactly one converged status word and rejects an unwritten sentinel (S3)', () => {
  expect(() => expectPointCloudICPConverged([0])).not.toThrow();
  for (const actual of [[], [0, 99], [99]]) {
    expect(() => expectPointCloudICPConverged(actual)).toThrow();
  }
});
