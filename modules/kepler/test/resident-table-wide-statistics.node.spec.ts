// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {
  makeRelativeFloat32Column,
  makeWideColumnStats,
  shiftGPUColumnStatistics
} from '../../../examples/deck/kepler-compute/src/resident-table-wide-statistics';
import {chooseColumnStatisticsRoute} from '../../../examples/deck/kepler-compute/src/resident-table-population-routes';

const TIMESTAMP_BASE = 1_700_000_000_000;

test('relative Float32 conversion keeps millisecond timestamps distinguishable', () => {
  const vector = arrow.vectorFromArray(
    [TIMESTAMP_BASE, TIMESTAMP_BASE + 1000, null, TIMESTAMP_BASE + 86_400_000],
    new arrow.TimestampMillisecond()
  );
  const converted = makeRelativeFloat32Column(vector, 4);
  expect(converted.base).toBe(TIMESTAMP_BASE);
  expect([...converted.validity]).toEqual([1, 1, 0, 1]);
  expect(converted.validCount).toBe(3);
  expect([...converted.values]).toEqual([0, 1000, 0, 86_400_000]);
});

test('Int64 and Float64 values narrow to numbers and NaN counts as invalid', () => {
  const int64 = makeRelativeFloat32Column(
    arrow.vectorFromArray([10n, 20n, null], new arrow.Int64()),
    3
  );
  expect([...int64.validity]).toEqual([1, 1, 0]);
  expect(int64.base).toBe(10);
  const float64 = makeRelativeFloat32Column(
    arrow.vectorFromArray([1.5, Number.NaN, Infinity, 3.5], new arrow.Float64()),
    4
  );
  expect([...float64.validity]).toEqual([1, 0, 0, 1]);
  expect(float64.values[3]).toBe(2);
});

test('shifted profiles return to column units and map to native stats shapes', () => {
  const profile = {
    populationRows: 10,
    count: 8,
    nullCount: 2,
    minimum: 0,
    maximum: 100,
    sum: 400,
    mean: 50,
    standardDeviation: 10,
    sampleVariance: 100,
    distinctEstimate: 7.6,
    histogram: new Uint32Array([4, 4])
  };
  const shifted = shiftGPUColumnStatistics(profile, 1000);
  expect(shifted).toMatchObject({minimum: 1000, maximum: 1100, mean: 1050, sum: 400 + 8000});
  expect(shifted.sampleVariance).toBe(100);
  const time = makeWideColumnStats('timestamp', shifted);
  expect(time).toMatchObject({type: 'time', min: 1000, max: 1100, percentNulls: 0.2});
  expect(makeWideColumnStats('integer', shifted)).toMatchObject({type: 'numeric', mean: 1050});
  expect(makeWideColumnStats('string', profile, 5)).toEqual({
    type: 'categorical',
    uniqueValues: 5,
    percentNulls: 0.2
  });
});

test('statistics routing sends large wide columns to the GPU profile and refuses the rest', () => {
  const base = {gpuSupported: false, populationRows: 1_000_000};
  expect(chooseColumnStatisticsRoute({...base, wideSupported: true}).route).toBe('gpu');
  expect(
    chooseColumnStatisticsRoute({...base, populationRows: 1000, wideSupported: true}).route
  ).toBe('cpu');
  const refused = chooseColumnStatisticsRoute({...base, wideReason: 'No retained values.'});
  expect(refused.route).toBe('refused');
  expect(refused.reason).toContain('No retained values.');
});
