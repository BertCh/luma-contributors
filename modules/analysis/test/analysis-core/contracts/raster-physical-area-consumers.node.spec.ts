// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {getGPURasterPatchMetricsParameterValues} from '../../../src/raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {computeRasterFractionalZonalStatisticsOnCPU} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {computeRasterFractionalZonalDistributionOnCPU} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';
import type {RasterGridAffine} from '../../../src/analysis-core/contracts/raster-grid';

const grid: RasterGridAffine = {
  width: 1,
  height: 1,
  affine: [2, 2, 7, 3, 3 + 5 * 2 ** -25, -11],
  pixelInterpretation: 'area'
};
// Exact binary64 determinant: 2*(3+5*2^-25)-2*3 = 5*2^-24.
const PHYSICAL_AREA = 2.980232238769531e-7;
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

// Literal expectations were recorded before the public-packer repairs.
describe('review B1 and S2 public packets preserve physical area', () => {
  it('coverage packet word 6 retains physical area without a test overwrite', () => {
    expect(getGPURasterFractionalCoverageParameterValues(grid)[6]).toBe(PHYSICAL_AREA);
  });

  it('one fully covered cell keeps the same physical area with the unmodified public packet', () => {
    const input = {
      input: {id: 'one-cell', format: 'float32' as const, values: Float32Array.of(1)},
      coverage: {
        zoneIds: Uint32Array.of(0),
        cellIds: Uint32Array.of(0),
        fractions: Float32Array.of(1),
        zoneValidity: Uint32Array.of(1),
        count: Uint32Array.of(1),
        requiredCount: Uint32Array.of(1),
        overflow: Uint32Array.of(0)
      }
    };
    const options = {grid, zoneCount: 1, quantity: 'intensive' as const};
    const withoutParameters = computeRasterFractionalZonalStatisticsOnCPU(input, options);
    const withParameters = computeRasterFractionalZonalStatisticsOnCPU(
      {...input, parameters: getGPURasterFractionalCoverageParameterValues(grid)},
      options
    );
    expect(withoutParameters.coveredArea[0]).toBe(PHYSICAL_AREA);
    expect(withParameters.coveredArea[0]).toBe(PHYSICAL_AREA);
  });

  it('patch-metrics packet word 6 retains physical area', () => {
    expect(getGPURasterPatchMetricsParameterValues(grid)[6]).toBe(PHYSICAL_AREA);
  });
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
