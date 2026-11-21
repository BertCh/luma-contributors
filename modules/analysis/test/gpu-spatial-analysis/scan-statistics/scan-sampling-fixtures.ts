// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import bounds from './scan-sampling.json';
import type {ScanOracleInput, ScanOracleModel} from './scan-statistic-oracle';
import {PERMUTATION_RANDOM_WGSL} from '../../../src/analysis-core/kernels/permutation-random';
import {EXPECTATION_BASED_SCAN_POISSON_WGSL} from '../../../src/gpu-spatial-analysis/scan-statistics/expectation-based-scan-poisson-wgsl';

export {bounds as SCAN_SAMPLING_BOUNDS};
export const SAMPLING_DECLARATIONS = `${PERMUTATION_RANDOM_WGSL}\n${EXPECTATION_BASED_SCAN_POISSON_WGSL}`;
export const SAMPLING_BODY = `let meanRow = index / ${bounds.sampleCount}u;
  let replicate = index % ${bounds.sampleCount}u;
  var stream = createPhiloxStream(vec2u(17u, 0u), replicate, meanRow, 0x45424353u);
  output[outputOffset + index] = drawPhiloxPoisson(&stream, means[meansOffset + meanRow]);`;

export function expectPoissonLaw(draws: ArrayLike<number>) {
  bounds.cases.forEach((reference, meanRow) => {
    const values = Array.from(
      {length: bounds.sampleCount},
      (_, row) => draws[meanRow * bounds.sampleCount + row]
    );
    values.forEach(value => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    });
    const sum = values.reduce((total, value) => total + value, 0);
    const average = sum / bounds.sampleCount;
    const variance =
      values.reduce((total, value) => total + (value - average) ** 2, 0) / (bounds.sampleCount - 1);
    const below = values.filter(value => value <= reference.threshold).length;
    expect(sum, `mean ${reference.mean}: sum`).toBeGreaterThanOrEqual(reference.sumMinimum);
    expect(sum, `mean ${reference.mean}: sum`).toBeLessThanOrEqual(reference.sumMaximum);
    expect(variance, `mean ${reference.mean}: variance`).toBeGreaterThanOrEqual(
      reference.varianceMinimum
    );
    expect(variance, `mean ${reference.mean}: variance`).toBeLessThanOrEqual(
      reference.varianceMaximum
    );
    expect(below, `mean ${reference.mean}: CDF`).toBeGreaterThanOrEqual(reference.belowMinimum);
    expect(below, `mean ${reference.mean}: CDF`).toBeLessThanOrEqual(reference.belowMaximum);
  });
}

export type ConditionalSamplingModel = ScanOracleModel | 'bernoulli-complement';
export function createConditionalSamplingScene(variant: ConditionalSamplingModel): ScanOracleInput {
  const model = variant === 'bernoulli-complement' ? 'bernoulli' : variant;
  return {
    model,
    positions: Float32Array.from(Array.from({length: 8}, (_, row) => [row, 0]).flat()),
    cases:
      variant === 'bernoulli-complement'
        ? Uint32Array.of(3, 2, 1, 2, 1, 0, 0, 0)
        : model === 'space-time-permutation'
          ? Uint32Array.of(2, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2)
          : model === 'bernoulli'
            ? Uint32Array.of(3, 2, 0, 0, 0, 0, 0, 0)
            : Uint32Array.of(5, 0, 0, 0, 0, 0, 0, 0),
    ...(model === 'poisson' ? {baseline: new Float32Array(8).fill(1)} : {}),
    ...(model === 'bernoulli'
      ? {
          controls:
            variant === 'bernoulli-complement'
              ? Uint32Array.of(0, 0, 0, 0, 0, 2, 1, 2)
              : Uint32Array.of(0, 0, 1, 2, 1, 2, 1, 2)
        }
      : {}),
    timeBuckets: model === 'space-time-permutation' ? 2 : 1,
    maximumTimeBuckets: 1,
    maximumWindowZones: 1,
    maximumPopulationFraction: 0.5,
    windowShape: 'circle',
    maximumClusters: 1,
    seed: 17,
    permutationCount: bounds.conditionalSampleCount
  };
}

export function expectConditionalLaw(model: ConditionalSamplingModel, maxima: ArrayLike<number>) {
  const reference = bounds.conditional.find(row => row.model === model)!;
  const values = Array.from(maxima);
  expect(values.length).toBe(bounds.conditionalSampleCount);
  values.forEach(value => {
    expect(Number.isFinite(value)).toBe(true);
    expect(
      reference.thresholds.some(row => Math.abs(row.score - value) < 2e-5),
      `${model}: supported maximum ${value}`
    ).toBe(true);
  });
  reference.thresholds.forEach(row => {
    const count = values.filter(value => value <= row.score + 2e-5).length;
    expect(count, `${model} CDF at ${row.score}`).toBeGreaterThanOrEqual(row.minimum);
    expect(count, `${model} CDF at ${row.score}`).toBeLessThanOrEqual(row.maximum);
  });
}
