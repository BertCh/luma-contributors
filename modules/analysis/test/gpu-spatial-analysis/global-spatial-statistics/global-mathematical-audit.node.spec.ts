// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeGlobalSpatialStatisticsOracle,
  getTwoSidedPValue,
  getPValue,
  type GlobalSpatialStatisticsOracle
} from './global-spatial-statistics-oracle';
import {
  GLOBAL_AUDIT_VALUES,
  GLOBAL_AUDIT_SECOND,
  GLOBAL_AUDIT_MASK,
  GLOBAL_AUDIT_WEIGHTS,
  GLOBAL_AUDIT_REFERENCE
} from './global-mathematical-audit-fixture';

export function flattenGlobalAudit(result: GlobalSpatialStatisticsOracle): number[] {
  return [
    result.count,
    result.s0,
    result.s1,
    result.s2,
    result.mean,
    result.variance,
    result.blackCount,
    result.islandCount,
    ...Object.values(result.moran),
    ...Object.values(result.geary),
    ...Object.values(result.getisOrdG),
    ...Object.values(result.bivariateMoran),
    ...Object.entries(result.joinCount)
      .filter(([name]) => name !== 'ordered')
      .map(([, value]) => value as number)
  ];
}

it('every global CPU output matches exhaustive permutations and projected Gaussian-sphere moments', () => {
  const maximumDifferences = new Array(52).fill(0);
  for (const reference of GLOBAL_AUDIT_REFERENCE)
    for (const scale of [1, 2 ** 20]) {
      const result = computeGlobalSpatialStatisticsOracle({
        values: Float32Array.from(GLOBAL_AUDIT_VALUES, value => value * scale),
        secondValues: Float32Array.from(GLOBAL_AUDIT_SECOND, value => value * scale),
        mask: Uint32Array.from(GLOBAL_AUDIT_MASK),
        weights: {
          offsets: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.offsets),
          neighbors: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.neighbors),
          weights: Float32Array.from(GLOBAL_AUDIT_WEIGHTS.weights)
        },
        islands: reference.islands,
        standardized: reference.standardized
      });
      const expected = reference.summary.map((value, index) =>
        index === 4 ? value * scale : index === 5 ? value * scale * scale : value
      );
      const actual = flattenGlobalAudit(result);
      expect(actual.length).toBe(52);
      for (const [index, value] of expected.entries()) {
        if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
        else {
          expect(Number.isFinite(actual[index])).toBe(true);
          maximumDifferences[index] = Math.max(
            maximumDifferences[index],
            Math.abs(actual[index] - value)
          );
          expect(Math.abs(actual[index] - value), `field ${index}`).toBeLessThanOrEqual(
            1e-7 * Math.max(1, Math.abs(value))
          );
        }
      }
      expect(result.joinCount.ordered).toEqual(reference.joinCounts);
    }
  console.info('global audit maximum absolute differences by output field', maximumDifferences);
});

it('bivariate randomization variance is undefined when the second variable is constant', () => {
  const result = computeGlobalSpatialStatisticsOracle({
    values: Float32Array.from(GLOBAL_AUDIT_VALUES),
    secondValues: new Float32Array(10).fill(5),
    mask: Uint32Array.from(GLOBAL_AUDIT_MASK),
    weights: {
      offsets: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.offsets),
      neighbors: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.neighbors),
      weights: Float32Array.from(GLOBAL_AUDIT_WEIGHTS.weights)
    }
  });
  expect(result.bivariateMoran.statistic).toBeNaN();
  expect(result.bivariateMoran.varianceRandomization).toBeNaN();
});

it('global CPU normal tails use the same normal-float32 floor as production', () => {
  expect(getTwoSidedPValue(20)).toBe(2 ** -126);
  expect(getPValue(20, 'greater')).toBe(2 ** -126);
  expect(getPValue(-20, 'lesser')).toBe(2 ** -126);
});

it('nonfinite positive weights are inactive like zero slots, consistently with tiled inference', () => {
  const weights = Float32Array.from(GLOBAL_AUDIT_WEIGHTS.weights);
  weights[10] = Infinity;
  const result = computeGlobalSpatialStatisticsOracle({
    values: Float32Array.from(GLOBAL_AUDIT_VALUES),
    secondValues: Float32Array.from(GLOBAL_AUDIT_SECOND),
    mask: Uint32Array.from(GLOBAL_AUDIT_MASK),
    weights: {
      offsets: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.offsets),
      neighbors: Uint32Array.from(GLOBAL_AUDIT_WEIGHTS.neighbors),
      weights
    }
  });
  expect(result.s0).toBe(21);
});
