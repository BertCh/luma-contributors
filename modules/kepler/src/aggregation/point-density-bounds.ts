// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';

/** Inclusive threshold of GPUPointDensity's exact rectangular key path. */
export const POINT_DENSITY_WIDE_BOUND_MAGNITUDE = 2 ** 125;

/** Keeps Kepler's rectangular GPU counts and weighted sums on the published coordinate rule. */
export function validateKeplerPointDensityGpuBounds(
  id: string,
  bounds: readonly [number, number, number, number]
): void {
  // At this threshold, the workgroup sum keys switch to exact edges while GPUGridBinning
  // continues using float32 division. An extent check alone misses asymmetric bounds.
  const maximumMagnitude = Math.max(...bounds.map(value => Math.abs(Math.fround(value))));
  validateRequirement(
    id,
    maximumMagnitude < POINT_DENSITY_WIDE_BOUND_MAGNITUDE,
    `bounds must have all endpoint magnitudes below 2^125 in float32; received maximum magnitude ${maximumMagnitude}. Rebase or rescale positions and bounds together.`,
    {
      code: 'NUMBER_RANGE',
      path: 'bounds',
      received: bounds,
      expected: {float32MagnitudeLessThan: POINT_DENSITY_WIDE_BOUND_MAGNITUDE}
    }
  );
}
