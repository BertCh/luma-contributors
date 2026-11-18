// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeRasterFractionalZonalStatisticsOnCPU} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import type {CPURasterFractionalZonalStatisticsInput} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {STATISTICS_PROPS} from './raster-fractional-zonal-statistics-operation-fixture';
import {getGPURasterFractionalCoverageParameterValues} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {createScalarAreaReviewFixture} from './scalar-area-review-fixture';

const AREA_GRID = {
  width: 1,
  height: 2,
  affine: [1, 0, 10, 0, -1, 61] as [number, number, number, number, number, number],
  pixelInterpretation: 'area' as const
};

function createInput(): CPURasterFractionalZonalStatisticsInput {
  return {
    input: {id: 'values', format: 'float32', values: Float32Array.of(2, 4)},
    coverage: {
      zoneIds: Uint32Array.of(0, 0),
      cellIds: Uint32Array.of(0, 1),
      fractions: Float32Array.of(1, 0.5),
      zoneValidity: Uint32Array.of(1),
      count: Uint32Array.of(2),
      requiredCount: Uint32Array.of(2),
      overflow: Uint32Array.of(0)
    }
  };
}

it('R3 accepts exactly one offset area and rejects a padded area array for one cell', () => {
  const valid = createScalarAreaReviewFixture('offset');
  const result = computeRasterFractionalZonalStatisticsOnCPU(valid.input, valid.props);
  expect(result.coveredArea[0]).toBe(3);
  expect(result.totals[0]).toBe(6);
  const oversized = createScalarAreaReviewFixture('oversized');
  expect(() =>
    computeRasterFractionalZonalStatisticsOnCPU(oversized.input, oversized.props)
  ).toThrow(/cellAreas/);
});

it('R7 accepts explicit areas without parameters on a huge affine', () => {
  const {input, props} = createScalarAreaReviewFixture('large-affine');
  const result = computeRasterFractionalZonalStatisticsOnCPU(input, props);
  expect(result.coveredArea[0]).toBe(3);
  expect(result.totals[0]).toBe(6);
});

// Independent WGS84 ellipsoid area element N M cos(latitude), converted to square degrees.
function getExpectedArea(latitude: number): number {
  const radians = Math.PI / 180;
  const eccentricitySquared = (1 / 298.257223563) * (2 - 1 / 298.257223563);
  const curvature = 1 - eccentricitySquared * Math.sin(latitude * radians) ** 2;
  return (
    ((6378137 * radians) ** 2 * (1 - eccentricitySquared) * Math.cos(latitude * radians)) /
    curvature ** 2
  );
}

it('uses row ground areas for scalar density integrals across two latitudes', () => {
  const result = computeRasterFractionalZonalStatisticsOnCPU(createInput(), {
    ...STATISTICS_PROPS,
    grid: AREA_GRID,
    zoneCount: 1,
    cellSizeMode: 'geographic'
  });
  const north = getExpectedArea(60.5);
  const south = getExpectedArea(59.5);
  expect(result.coveredArea[0] / (north + south / 2)).toBeCloseTo(1, 6);
  expect(result.validArea).toEqual(result.coveredArea);
  expect(result.totals[0] / (2 * north + 2 * south)).toBeCloseTo(1, 6);
  // Means keep their existing coverage-fraction weighting, independently of area units.
  expect(result.means[0]).toBe(Math.fround(4 / 1.5));
});

it('R4 suppresses ground areas from a repacked non-axis-aligned affine', () => {
  const grid = {
    ...AREA_GRID,
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, -1, 61] as typeof AREA_GRID.affine
  };
  const input = createInput();
  input.parameters = getGPURasterFractionalCoverageParameterValues({
    ...grid,
    affine: [1, 0, 0, 10, -1, 61]
  });
  const result = computeRasterFractionalZonalStatisticsOnCPU(input, {
    ...STATISTICS_PROPS,
    grid,
    zoneCount: 1,
    cellSizeMode: 'geographic'
  });
  expect(result.validity[0]).toBe(0);
  for (const port of ['coveredArea', 'validArea', 'totals', 'means'] as const)
    expect(result[port][0]).toBeNaN();
});

it('R6 suppresses unrepresentable aggregates while keeping an ordinary zone usable', () => {
  const input = createInput();
  input.input.values = Float32Array.of(1, 1, 2);
  input.cellAreas = Float32Array.of(2e38, 2e38, 3);
  input.coverage = {
    ...input.coverage,
    zoneIds: Uint32Array.of(0, 0, 1),
    cellIds: Uint32Array.of(0, 1, 2),
    fractions: Float32Array.of(1, 1, 1),
    count: Uint32Array.of(3),
    requiredCount: Uint32Array.of(3),
    zoneValidity: Uint32Array.of(1, 1)
  };
  const grid = {...AREA_GRID, width: 3, height: 1};
  const result = computeRasterFractionalZonalStatisticsOnCPU(input, {
    ...STATISTICS_PROPS,
    grid,
    zoneCount: 2
  });
  expect(Array.from(result.validity)).toEqual([0, 1]);
  for (const port of ['coveredArea', 'validArea', 'totals', 'means'] as const)
    expect(result[port][0]).toBeNaN();
  expect(result.totals[1]).toBe(6);
  expect(result.coveredArea[1]).toBe(3);
});

it('uses explicit scalar areas and rejects mixed area models', () => {
  const input = {...createInput(), cellAreas: Float32Array.of(3, 7)};
  const options = {...STATISTICS_PROPS, grid: AREA_GRID, zoneCount: 1};
  const result = computeRasterFractionalZonalStatisticsOnCPU(input, options);
  expect(result.coveredArea[0]).toBe(6.5);
  expect(result.totals[0]).toBe(20);
  expect(() =>
    computeRasterFractionalZonalStatisticsOnCPU(input, {
      ...options,
      cellSizeMode: 'geographic'
    })
  ).toThrow(/cellSizeMode must be uniform when cellAreas is given/);
});

it('suppresses a zone with invalid explicit areas and rejects short support columns', () => {
  const options = {...STATISTICS_PROPS, grid: AREA_GRID, zoneCount: 1};
  for (const invalid of [NaN, Infinity, 0, -1]) {
    const result = computeRasterFractionalZonalStatisticsOnCPU(
      {
        ...createInput(),
        cellAreas: Float32Array.of(3, invalid)
      },
      options
    );
    expect(result.validity[0]).toBe(0);
    for (const port of ['totals', 'means', 'coveredArea', 'validArea'] as const)
      expect(result[port][0]).toBeNaN();
  }
  expect(() =>
    computeRasterFractionalZonalStatisticsOnCPU(
      {
        ...createInput(),
        cellAreas: Float32Array.of(3)
      },
      options
    )
  ).toThrow(/cellAreas/);
});
