// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPURasterPatchMetricsParameterValues} from '../../../src/raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {computeRasterFractionalZonalStatisticsOnCPU} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {computeRasterFractionalZonalDistributionOnCPU} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';

const coverage = {
  zoneIds: Uint32Array.of(0),
  cellIds: Uint32Array.of(0),
  fractions: Float32Array.of(1),
  zoneValidity: Uint32Array.of(1),
  count: Uint32Array.of(1),
  requiredCount: Uint32Array.of(1),
  overflow: Uint32Array.of(0)
};
const input = {id: 'one-cell', format: 'float32' as const, values: Float32Array.of(1)};

it('review B1 patch packet retains the physical area in both CPU consumers without overwrites', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [2, 1, 0, 4, 2 + 5 * 2 ** -24, 0] as const,
    pixelInterpretation: 'area' as const
  };
  // Closed form: 2*(2+5*2^-24)-4 = 5*2^-23, not the packed-axis 2^-21.
  const expected = 5.960464477539062e-7;
  const parameters = getGPURasterPatchMetricsParameterValues(grid);
  expect(parameters[6]).toBe(expected);
  for (const frame of [undefined, parameters]) {
    const scalar = computeRasterFractionalZonalStatisticsOnCPU(
      {input, coverage, parameters: frame},
      {grid, zoneCount: 1, quantity: 'density'}
    );
    const distribution = computeRasterFractionalZonalDistributionOnCPU(
      {input, coverage, parameters: frame, probabilities: Float32Array.of(0.5)},
      {grid, zoneCount: 1}
    );
    expect(Array.from(scalar.coveredArea)).toEqual([expected]);
    expect(Array.from(scalar.totals)).toEqual([expected]);
    expect(Array.from(distribution.coveredArea)).toEqual([expected]);
  }
});

it('review B3 legacy patch repacking deactivates a reused version-2 extension', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, -1, 20] as const,
    pixelInterpretation: 'area' as const
  };
  const target = getGPURasterFractionalCoverageParameterValues(
    {...grid, affine: [1, 0, 0, 0, -1, 50]},
    {localOrigin: [0, 50]}
  );
  expect(target[8]).toBe(2);
  getGPURasterPatchMetricsParameterValues(grid, target);
  expect(Array.from(target.subarray(8, 10))).toEqual([0, 0]);
  const fresh = getGPURasterPatchMetricsParameterValues(grid);
  for (const parameters of [fresh, target]) {
    const scalar = computeRasterFractionalZonalStatisticsOnCPU(
      {input, coverage, parameters},
      {grid, zoneCount: 1, quantity: 'density', cellSizeMode: 'geographic'}
    );
    const distribution = computeRasterFractionalZonalDistributionOnCPU(
      {input, coverage, parameters, probabilities: Float32Array.of(0.5)},
      {grid, zoneCount: 1, cellSizeMode: 'geographic'}
    );
    // Independent WGS84 midpoint metric at 19.5 degrees (Python math formula in area-packet fixture).
    for (const result of [scalar, distribution])
      expect(Array.from(result.coveredArea)).toEqual([11620371456]);
  }
});

it('review B1 invalid physical area leaves every target word unchanged', () => {
  const scale = 2 ** -60;
  const grid = {
    width: 1,
    height: 1,
    affine: [1.1 * scale, scale, 0, 1.21 * scale, 1.1 * scale, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const target = Float32Array.from({length: 10}, (_, index) => index + 10);
  const before = target.slice();
  expect(() => getGPURasterPatchMetricsParameterValues(grid, target)).toThrow(
    /physical.*determinant.*finite.*nonzero.*float32/
  );
  expect(target).toEqual(before);
});

it('forwarded B1 coverage packet preserves physical area without overwrites', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [2, 1, 0, 4, 2 + 5 * 2 ** -24, 0] as const,
    pixelInterpretation: 'area' as const
  };
  // Determinant = 2*(2+5*2^-24)-4 = 5*2^-23.
  const parameters = getGPURasterFractionalCoverageParameterValues(grid);
  expect(parameters[6]).toBe(5.960464477539062e-7);
  for (const packet of [undefined, parameters]) {
    const scalar = computeRasterFractionalZonalStatisticsOnCPU(
      {input, coverage, parameters: packet},
      {grid, zoneCount: 1, quantity: 'density'}
    );
    const distribution = computeRasterFractionalZonalDistributionOnCPU(
      {input, coverage, parameters: packet, probabilities: Float32Array.of(0.5)},
      {grid, zoneCount: 1}
    );
    expect([...scalar.totals]).toEqual([5.960464477539062e-7]);
    for (const result of [scalar, distribution])
      expect([...result.coveredArea]).toEqual([5.960464477539062e-7]);
  }
});

it('forwarded B3 coverage legacy repacking clears extensions and failed writes preserve targets', () => {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, -1, 20] as const,
    pixelInterpretation: 'area' as const
  };
  const target = getGPURasterFractionalCoverageParameterValues(
    {...grid, affine: [1, 0, 0, 0, -1, 50]},
    {localOrigin: [0, 50]}
  );
  getGPURasterFractionalCoverageParameterValues(grid, target);
  expect([...target.subarray(8, 10)]).toEqual([0, 0]);
  for (const parameters of [getGPURasterFractionalCoverageParameterValues(grid), target]) {
    const scalar = computeRasterFractionalZonalStatisticsOnCPU(
      {input, coverage, parameters},
      {grid, zoneCount: 1, quantity: 'density', cellSizeMode: 'geographic'}
    );
    const distribution = computeRasterFractionalZonalDistributionOnCPU(
      {input, coverage, parameters, probabilities: Float32Array.of(0.5)},
      {grid, zoneCount: 1, cellSizeMode: 'geographic'}
    );
    for (const result of [scalar, distribution])
      expect([...result.coveredArea]).toEqual([11620371456]);
  }
  const scale = 2 ** -60;
  const before = target.slice();
  expect(() =>
    getGPURasterFractionalCoverageParameterValues(
      {...grid, affine: [1.1 * scale, scale, 0, 1.21 * scale, 1.1 * scale, 0]},
      target
    )
  ).toThrow(/physical.*determinant/);
  expect(target).toEqual(before);
});
