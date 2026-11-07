// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  chooseColumnStatisticsRoute,
  chooseTableSortRoute
} from '../../../examples/deck/kepler-compute/src/resident-table-population-routes';
import {
  estimateHistogramQuantile,
  makeNumericColumnStatsFromGPU
} from '../../../examples/deck/kepler-compute/src/resident-gpu-column-statistics';

test('GPU scalar sorts have no row bound; other types refuse above 262,144 rows with a reason', () => {
  expect(
    chooseTableSortRoute({orderKind: 'float32', hasPreparedGPUColumn: true, populationRows: 5e7})
      .route
  ).toBe('gpu');
  expect(
    chooseTableSortRoute({orderKind: 'int32', hasPreparedGPUColumn: true, populationRows: 5e7})
      .route
  ).toBe('gpu');
  expect(
    chooseTableSortRoute({
      orderKind: 'other-ordered',
      hasPreparedGPUColumn: true,
      populationRows: 262_144
    }).route
  ).toBe('cpu');
  const refused = chooseTableSortRoute({
    orderKind: 'other-ordered',
    hasPreparedGPUColumn: false,
    populationRows: 262_145
  });
  expect(refused.route).toBe('refused');
  expect(refused.reason).toContain('262,144');
  expect(refused.reason).toContain('GPU');
  expect(
    chooseTableSortRoute({orderKind: 'unordered', hasPreparedGPUColumn: false, populationRows: 1})
      .route
  ).toBe('refused');
});

test('statistics prefer the GPU, fall back to bounded CPU with the GPU reason, then refuse', () => {
  expect(chooseColumnStatisticsRoute({gpuSupported: true, populationRows: 1e7}).route).toBe('gpu');
  const cpu = chooseColumnStatisticsRoute({
    gpuSupported: false,
    gpuReason: 'Not Float32',
    populationRows: 100
  });
  expect(cpu).toEqual({route: 'cpu', reason: 'Not Float32'});
  const refused = chooseColumnStatisticsRoute({
    gpuSupported: false,
    gpuReason: 'Not Float32',
    populationRows: 300_000
  });
  expect(refused.route).toBe('refused');
  expect(refused.reason).toContain('Not Float32');
});

test('histogram quantiles interpolate inside equal-width bins', () => {
  const histogram = new Uint32Array([10, 10, 10, 10]);
  expect(estimateHistogramQuantile(histogram, 0, 4, 0)).toBe(0);
  expect(estimateHistogramQuantile(histogram, 0, 4, 1)).toBe(4);
  expect(estimateHistogramQuantile(histogram, 0, 4, 0.5)).toBeCloseTo(2);
  expect(estimateHistogramQuantile(histogram, 0, 4, 0.25)).toBeCloseTo(1);
  expect(estimateHistogramQuantile(new Uint32Array([0, 4, 0, 0]), 0, 4, 0.5)).toBeCloseTo(1.5);
});

test('GPU profile maps to Kepler numeric column statistics with exact edges', () => {
  const stats = makeNumericColumnStatsFromGPU({
    populationRows: 100,
    count: 80,
    nullCount: 20,
    minimum: 0,
    maximum: 8,
    sum: 320,
    mean: 4,
    standardDeviation: 2,
    sampleVariance: 4.05,
    distinctEstimate: 30,
    histogram: new Uint32Array([20, 20, 20, 20])
  }) as unknown as {
    type: string;
    bins: {x0: number; x1: number; count: number}[];
    quantiles: {label: string; value: number | null}[];
    mean: number;
    std: number;
    percentNulls: number;
  };
  expect(stats.type).toBe('numeric');
  expect(stats.bins.map(bin => [bin.x0, bin.x1, bin.count])).toEqual([
    [0, 2, 20],
    [2, 4, 20],
    [4, 6, 20],
    [6, 8, 20]
  ]);
  expect(stats.quantiles.map(quantile => quantile.label)).toEqual([
    'Min',
    '25%',
    '50%',
    '75%',
    'Max'
  ]);
  expect(stats.quantiles[0].value).toBe(0);
  expect(stats.quantiles[4].value).toBe(8);
  expect(stats.quantiles[2].value).toBeCloseTo(4);
  expect(stats.mean).toBe(4);
  expect(stats.std).toBeCloseTo(Math.sqrt(4.05));
  expect(stats.percentNulls).toBeCloseTo(0.2);
});

test('an all-null profile yields null statistics rather than NaN', () => {
  const stats = makeNumericColumnStatsFromGPU({
    populationRows: 5,
    count: 0,
    nullCount: 5,
    minimum: NaN,
    maximum: NaN,
    sum: 0,
    mean: NaN,
    standardDeviation: NaN,
    sampleVariance: NaN,
    distinctEstimate: 0,
    histogram: new Uint32Array(4)
  }) as unknown as {bins: unknown[]; mean: null; std: null; quantiles: {value: null}[]};
  expect(stats.bins).toEqual([]);
  expect(stats.mean).toBeNull();
  expect(stats.std).toBeNull();
  expect(stats.quantiles.every(quantile => quantile.value === null)).toBe(true);
});
