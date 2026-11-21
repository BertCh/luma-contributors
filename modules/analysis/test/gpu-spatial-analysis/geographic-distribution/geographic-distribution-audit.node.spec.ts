// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeGeographicDistribution} from './geographic-distribution-oracle';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/geographic-distribution-parameters';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';
import {
  GEOGRAPHIC_INPUT,
  GEOGRAPHIC_EXPECTED,
  AXIAL_DIRECTIONAL_EXPECTED,
  COINCIDENT_INPUT
} from './geographic-distribution-audit-fixtures';

it('checks every float64 geographic reference output against independent population and circular moments', () => {
  const actual = computeGeographicDistribution(GEOGRAPHIC_INPUT);
  for (const [port, expected] of Object.entries(GEOGRAPHIC_EXPECTED)) {
    expectOracleRows(actual[port as keyof typeof actual], expected, 1e-12);
  }
  expectOracleRows(
    computeGeographicDistribution({...GEOGRAPHIC_INPUT, orientationOnly: true}).directionalMeans,
    AXIAL_DIRECTIONAL_EXPECTED,
    1e-12
  );
});

it('checks a signed cross-covariance and rotated direction against the independent eigensolver oracle', () => {
  const rotate = (values: number[]) => {
    const transformed: number[] = [];
    for (let index = 0; index < values.length; index += 2)
      transformed.push(values[index] - values[index + 1], values[index] + values[index + 1]);
    return transformed;
  };
  const actual = computeGeographicDistribution({
    ...GEOGRAPHIC_INPUT,
    positions: rotate(GEOGRAPHIC_INPUT.positions),
    lineEnds: rotate(GEOGRAPHIC_INPUT.lineEnds)
  });
  expectOracleRows(
    actual.ellipses,
    [
      -0.7853981633974483,
      0.7071067811865476,
      1.4142135623730951,
      -0.7853981633974483,
      0,
      2.8284271247461903,
      NaN,
      NaN,
      NaN
    ],
    1e-12
  );
  expectOracleRows(actual.meanCenters, [0, 0, 12, 12, NaN, NaN], 1e-12);
  expectOracleRows(
    actual.directionalMeans,
    [
      -2.677945044588987,
      0.4729537233052701,
      4.949747468305833,
      NaN,
      1,
      1.4142135623730951,
      NaN,
      NaN,
      NaN
    ],
    1e-12
  );
});

it('retains an exact coincident weighted median after one corrected step', () => {
  const actual = computeGeographicDistribution(COINCIDENT_INPUT);
  expectOracleRows(actual.medianCenters, [0, 0], 0);
  expect(actual.medianConverged).toEqual([1]);
});

it('uses weights in the mean and binds the single geometric-median iteration', () => {
  // Mean=(10+3*14)/4=13; inverse-distance mass=1/3+3=10/3,
  // pull=-1+3=2, so one step is 13+2/(10/3)=13.6. Exact median is 14.
  const actual = computeGeographicDistribution({
    positions: [10, 0, 14, 0],
    weights: [1, 3],
    groupCount: 1,
    maximumMedianIterations: 1
  });
  expectOracleRows(actual.meanCenters, [13, 0], 0);
  expectOracleRows(actual.medianCenters, [13.6, 0], 1e-12);
  expect(actual.medianConverged).toEqual([0]);
});

it('refuses settings that become infinity or a zero multiplier in the published float32 layout', () => {
  for (const settings of [
    {origin: [1e300, 0] as const},
    {standardDeviations: 1e300},
    {medianTolerance: 1e300},
    {standardDeviations: 1e-300}
  ]) {
    expect(() => getGPUGeographicDistributionParameterValues(settings)).toThrow();
  }
});
