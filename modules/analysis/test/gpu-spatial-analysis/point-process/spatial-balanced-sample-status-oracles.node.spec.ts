// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialBalancedSampleOnCPU} from '../../../src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation';

it('balanced sample CPU pins every stratum count and invalid-frame code', () => {
  for (const invalid of [true, false, true]) {
    const result = computeSpatialBalancedSampleOnCPU(
      {
        positions: Float32Array.of(invalid ? NaN : 0, 0, 1, 0, 2, 0, 3, 0),
        frameIds: Uint32Array.of(10, 11, 12, 13),
        strata: Uint32Array.of(0, 0, 1, 1)
      },
      {
        allocations: [
          {stratumId: 0, sampleCount: 2},
          {stratumId: 1, sampleCount: 1}
        ],
        bounds: [0, 0, 4, 1],
        seed: 7
      }
    );
    // Invalid row -> every allocation [0,1]; otherwise both requested sample sizes fit.
    expect(Array.from(result.status)).toEqual(invalid ? [0, 1, 0, 1] : [2, 0, 1, 0]);
  }
});

it('balanced sample CPU distinguishes insufficient first stratum from a usable second stratum', () => {
  const actualStatuses: number[][] = [];
  for (const strata of [
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1]
  ]) {
    const result = computeSpatialBalancedSampleOnCPU(
      {
        positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
        frameIds: Uint32Array.of(10, 11, 12, 13),
        strata: Uint32Array.from(strata)
      },
      {
        allocations: [
          {stratumId: 0, sampleCount: 2},
          {stratumId: 1, sampleCount: 1}
        ],
        bounds: [0, 0, 4, 1],
        seed: 7
      }
    );
    actualStatuses.push(Array.from(result.status));
  }
  // One first-stratum row cannot fill allocation 2; three second-stratum rows can fill 1.
  // Restoring the original membership must recover both selected counts and both codes.
  expect(actualStatuses).toEqual([
    [2, 0, 1, 0],
    [0, 1, 1, 0],
    [2, 0, 1, 0]
  ]);
});
