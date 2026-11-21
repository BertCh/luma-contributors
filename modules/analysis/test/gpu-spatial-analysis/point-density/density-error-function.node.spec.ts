// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getComplementaryErrorFunction} from '../../../src/gpu-spatial-analysis/point-density/complementary-error-function';
import {getGPUKernelDensityWindowOverlap} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {getGPUSpaceTimeTemporalWindowOverlap} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';

it('STD-09 density-owned erfc agrees with independent ordinary and tail values', () => {
  // /usr/bin/python3: math.erfc(x), not another evaluation of the fit polynomial.
  for (const [argument, expected] of [
    [0, 1],
    [1, 0.15729920705028513],
    [4, 1.541725790028002e-8],
    [-1, 1.842700792949715]
  ]) {
    const actual = getComplementaryErrorFunction(argument);
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1.3e-7 * expected);
  }
});

it('STD-09 both density consumers retain ordinary and either-tail window mass', () => {
  // Independent SciPy special.ndtr: Phi(1)-Phi(-1), Phi(-3.5)-Phi(-4.5).
  // A square spatial window has the square of the one-dimensional normal mass.
  for (const [low, high, temporalExpected, spatialExpected] of [
    [-1, 1, 0.6826894921370859, 0.4660649426743922],
    [3.5, 4.5, 0.00022923140591079495, 5.254703745583964e-8],
    [-4.5, -3.5, 0.00022923140591079495, 5.254703745583964e-8],
    // Python math.erfc: .5*(erfc(9/sqrt(2))-erfc(10/sqrt(2))); spatial mass is its square.
    [9, 10, 1.1285122074236008e-19, 1.2735398023040882e-38],
    [-10, -9, 1.1285122074236008e-19, 1.2735398023040882e-38]
  ]) {
    const temporal = getGPUSpaceTimeTemporalWindowOverlap('gaussian', 1, 0, [low, high]);
    const spatial = getGPUKernelDensityWindowOverlap('gaussian', 1, [0, 0], [low, low, high, high]);
    expect(Number.isFinite(temporal)).toBe(true);
    expect(Number.isFinite(spatial)).toBe(true);
    expect(Math.abs(temporal - temporalExpected)).toBeLessThanOrEqual(3e-7 * temporalExpected);
    expect(Math.abs(spatial - spatialExpected)).toBeLessThanOrEqual(5e-7 * spatialExpected);
  }
});
