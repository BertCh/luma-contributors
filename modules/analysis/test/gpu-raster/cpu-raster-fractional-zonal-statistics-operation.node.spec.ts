// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRasterFractionalZonalStatisticsOnCPU,
  CPURasterFractionalZonalStatisticsOperation
} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {GPURasterFractionalZonalStatisticsOperation} from '../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import {
  createStatisticsInput,
  createStatisticsGraphInputs,
  createSingletonStatisticsProps,
  createSingletonStatisticsInput,
  STATISTICS_PROPS
} from './raster-fractional-zonal-statistics-operation-fixture';

it('matches hand-integrated disjoint cell pieces with unequal coverage and calibrated values', () => {
  // Zone 0: valid pieces at 5 and 9, weights 1/2 and 1/4, plus missing weight 3/4.
  // Zone 1: valid pieces at 17 and 25, weights 1 and 1/4, plus sentinel weight 1/2.
  // Zone 2 is covered but unobserved. Zone 3 has no support.
  for (const quantity of ['density', 'intensive', 'extensive'] as const) {
    const result = computeRasterFractionalZonalStatisticsOnCPU(createStatisticsInput(), {
      ...STATISTICS_PROPS,
      quantity
    });
    expect(Array.from(result.totals)).toEqual(
      quantity === 'extensive' ? [4.75, 23.25, NaN, NaN] : [28.5, 139.5, NaN, NaN]
    );
    expect(Array.from(result.means)).toEqual([Math.fround(19 / 3), Math.fround(93 / 5), NaN, NaN]);
    expect(Array.from(result.coveredArea)).toEqual([9, 10.5, 6, 0]);
    expect(Array.from(result.validArea)).toEqual([4.5, 7.5, 0, 0]);
    expect(Array.from(result.validity)).toEqual([1, 1, 0, 0]);
  }
});

it('retains the contributor mean denominator floor for normal float32 coverage below 1e-30', () => {
  const input = createStatisticsInput();
  input.coverage.count[0] = 1;
  input.coverage.fractions[0] = 1e-32;
  const result = computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS);
  // Calibrated value 5, weight about 1e-32, denominator 1e-30: mean about 0.05, not 5.
  expect(Math.abs(result.means[0] - 0.05)).toBeLessThanOrEqual(1e-8);
  expect(result.validity[0]).toBe(1);
  expect(result.validArea[0]).toBeGreaterThan(0);
});

it('matches completeness, count clamping, ignored IDs and the compile-time area fallback', () => {
  const input = createStatisticsInput();
  input.parameters = undefined;
  input.coverage.requiredCount[0] = 500; // Deliberately ignored by this contributor.
  input.coverage.zoneValidity[1] = 0;
  const result = computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS);
  expect(Array.from(result.totals)).toEqual([4.75, NaN, NaN, NaN]);
  expect(Array.from(result.coveredArea)).toEqual([1.5, NaN, 1, 0]);
  expect(Array.from(result.validArea)).toEqual([0.75, NaN, 0, 0]);
  input.coverage.overflow[0] = 1;
  const incomplete = computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS);
  for (const port of ['totals', 'means', 'coveredArea', 'validArea'] as const)
    expect(incomplete[port].every(Number.isNaN)).toBe(true);
  expect(Array.from(incomplete.validity)).toEqual([0, 0, 0, 0]);
  input.coverage.overflow[0] = 0;
  input.coverage.zoneValidity.fill(1);
  input.coverage.count[0] = 999;
  // Clamping includes exactly one additional tuple, value 25 times coverage 1.
  expect(computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS).totals[0]).toBe(
    29.75
  );
});

it('rounds the nodata sentinel and honors integer band formats and exact lengths', () => {
  const input = createStatisticsInput();
  input.input = {
    id: 'band',
    format: 'float32',
    values: Float32Array.of(0.1, 2, 3, 4, 5, 6),
    noDataValue: 0.1
  };
  input.coverage.count[0] = 1;
  expect(computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS).validity[0]).toBe(0);
  for (const format of ['sint32', 'uint32'] as const) {
    input.input = {
      id: 'band',
      format,
      values:
        format === 'sint32' ? Int32Array.of(-2, 1, 2, 3, 4, 5) : Uint32Array.of(4, 1, 2, 3, 4, 5)
    };
    expect(computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS).totals[0]).toBe(
      format === 'sint32' ? -6 : 12
    );
  }
  input.input.values = new Float32Array(7);
  expect(() => computeRasterFractionalZonalStatisticsOnCPU(input, STATISTICS_PROPS)).toThrow(
    /input.values/
  );
  const malformed = createStatisticsInput();
  malformed.input.validity = new Uint32Array(7);
  expect(() => computeRasterFractionalZonalStatisticsOnCPU(malformed, STATISTICS_PROPS)).toThrow(
    /input.validity/
  );
  malformed.input.validity = new Uint32Array(6);
  malformed.coverage.cellIds = new Uint32Array(1);
  expect(() => computeRasterFractionalZonalStatisticsOnCPU(malformed, STATISTICS_PROPS)).toThrow(
    /coverage.cellIds/
  );
  const badParameters = createStatisticsInput();
  badParameters.parameters = new Float32Array(7);
  expect(() =>
    computeRasterFractionalZonalStatisticsOnCPU(badParameters, STATISTICS_PROPS)
  ).toThrow(/parameters/);
});

it('checks construction-time validation of both classes and CPU port mapping', async () => {
  const descriptors = Object.fromEntries(
    Object.entries(createStatisticsGraphInputs()).map(([port, input]) => [
      port,
      {format: input.format, length: input.values.length}
    ])
  );
  for (const Operation of [
    CPURasterFractionalZonalStatisticsOperation,
    GPURasterFractionalZonalStatisticsOperation
  ])
    expect(() => new Operation({...STATISTICS_PROPS, zoneCount: 0})).toThrow(/^fractional /);
  expect(
    () =>
      new CPURasterFractionalZonalStatisticsOperation({
        ...STATISTICS_PROPS,
        grid: {...STATISTICS_PROPS.grid, width: 0}
      })
  ).toThrow(/^fractional /);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFractionalZonalStatisticsOperation({
        ...STATISTICS_PROPS,
        grid: {...STATISTICS_PROPS.grid, width: 0}
      }),
      descriptors
    )
  ).toThrow(/^fractional /);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFractionalZonalStatisticsOperation(STATISTICS_PROPS),
      descriptors
    )
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFractionalZonalStatisticsOperation(STATISTICS_PROPS),
      {...descriptors, parameters: {format: 'float32', length: 7}}
    )
  ).toThrow(/^fractional .*parameters/);
  const rows = Object.fromEntries(
    Object.entries(createStatisticsGraphInputs()).map(([port, input]) => [
      port,
      {
        format: input.format,
        length: input.values.length,
        chunks: [{values: input.values, length: input.values.length}],
        pack: () => input.values
      }
    ])
  );
  const props = {
    ...STATISTICS_PROPS,
    output: {...STATISTICS_PROPS.output, validity: 'resultValidity', totals: 'resultTotals'}
  };
  const result = await new CPURasterFractionalZonalStatisticsOperation(props).execute({
    inputs: rows
  });
  expect(result['resultTotals']).toEqual(Float32Array.of(28.5, 139.5, NaN, NaN));
  expect(result['resultValidity']).toEqual(Uint32Array.of(1, 1, 0, 0));
  expect(Object.keys(result).sort()).toEqual(Object.values(props.output).sort());
});

it('ignores surplus coverage rows on a unit singleton and keeps the hand-computed zone result', () => {
  const props = createSingletonStatisticsProps();
  for (const surplusCoverageRows of [false, true]) {
    const input = createSingletonStatisticsInput(surplusCoverageRows);
    const result = computeRasterFractionalZonalStatisticsOnCPU(input, props);
    // One whole unit-area cell of calibrated value 5; the second validity word is not a zone.
    expect(result.totals).toEqual(Float32Array.of(5));
    expect(result.means).toEqual(Float32Array.of(5));
    expect(result.coveredArea).toEqual(Float32Array.of(1));
    expect(result.validArea).toEqual(Float32Array.of(1));
    expect(result.validity).toEqual(Uint32Array.of(1));
    const descriptors = Object.fromEntries(
      Object.entries(createStatisticsGraphInputs(input)).map(([port, value]) => [
        port,
        {format: value.format, length: value.values.length}
      ])
    );
    expect(() =>
      constructRasterOperationContributor(
        new GPURasterFractionalZonalStatisticsOperation(props),
        descriptors
      )
    ).not.toThrow();
  }
});

it('continues to reject undersized zone validity, coverage columns and status arrays by name', () => {
  const props = createSingletonStatisticsProps();
  for (const port of [
    'zoneValidity',
    'cellIds',
    'fractions',
    'count',
    'requiredCount',
    'overflow'
  ] as const) {
    const input = createSingletonStatisticsInput();
    if (port === 'fractions') input.coverage.fractions = new Float32Array(0);
    else input.coverage[port] = new Uint32Array(0);
    expect(() => computeRasterFractionalZonalStatisticsOnCPU(input, props)).toThrow(
      new RegExp(`coverage.${port}`)
    );
  }
});
