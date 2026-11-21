// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {resolveSpatialAutocorrelationMinimumNeighbors} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-kernels';
import {packGPUSpatialParameterValues} from '../../../src/analysis-core/contracts/parameter-schema';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA,
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  GPU_LOCAL_MORAN_QUADRANT,
  getGPUSpatialAutocorrelationParameterValues,
  getGPUHotSpotAnalysisParameterValues,
  getGPULocalSpatialHeteroscedasticityParameterValues,
  getGPUMultivariateLocalGearyPermutationTestParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  computeHotSpotOracle,
  computeLocalMoranOracle,
  getTwoSidedPValue
} from './spatial-autocorrelation-oracle';
import {
  AUDIT_VALUES,
  AUDIT_MASK,
  AUDIT_WEIGHTS,
  AUDIT_REFERENCE,
  AUDIT_NEIGHBOR_COUNTS,
  AUDIT_QUADRANTS,
  AUDIT_GI_BINS
} from './local-autocorrelation-audit-fixture';
import {FIXED_POPULATION_REFERENCE} from './local-autocorrelation-audit-fixture';

it('fixed external population moments agree with independent conditional-lag moments through uint32max', () => {
  let maximumDifference = 0;
  for (const reference of FIXED_POPULATION_REFERENCE) {
    const input = {
      values: Float32Array.from(AUDIT_VALUES),
      mask: Uint32Array.from(AUDIT_MASK),
      weights: {
        offsets: Uint32Array.from(AUDIT_WEIGHTS.offsets),
        neighbors: Uint32Array.from(AUDIT_WEIGHTS.neighbors),
        weights: Float32Array.from(AUDIT_WEIGHTS.weights)
      },
      islands: 'keep' as const,
      parameters: {fixedMoments: {count: reference.count, mean: 4, variance: 25}}
    };
    const moran = computeLocalMoranOracle(input);
    const gi = computeHotSpotOracle({...input, selfWeight: 0});
    const star = computeHotSpotOracle({...input, selfWeight: 1});
    for (const [actual, expected] of [
      [moran.localI, reference.localI],
      [moran.zScores, reference.moran],
      [gi.zScores, reference.gi],
      [star.zScores, reference.star]
    ] as const) {
      for (const [row, value] of expected.entries()) {
        if (Number.isNaN(value)) expect(actual[row]).toBeNaN();
        else {
          expect(Number.isFinite(actual[row])).toBe(true);
          maximumDifference = Math.max(maximumDifference, Math.abs(actual[row] - value));
          expect(Math.abs(actual[row] - value)).toBeLessThan(1e-14);
        }
      }
    }
    for (const result of [moran, gi, star])
      expect(result.moments).toEqual({
        count: reference.count,
        mean: 4,
        variance: 25,
        sumOfSquares: 25 * reference.count
      });
  }
  console.info('fixed-population maximum error', maximumDifference);
});

it('declarative significance uses the same open float32 interval as the public packer', () => {
  for (const significanceLevel of [1, 1 - 2 ** -26, Number.MIN_VALUE, 0, -1, Infinity, NaN]) {
    expect(() => getGPUSpatialAutocorrelationParameterValues({significanceLevel})).toThrow();
    expect(() =>
      packGPUSpatialParameterValues(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA, {
        significanceLevel
      })
    ).toThrow();
  }
  for (const significanceLevel of [2 ** -126, 0.05, 1 - 2 ** -24]) {
    expect(
      Array.from(
        packGPUSpatialParameterValues(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA, {
          significanceLevel
        })
      )
    ).toEqual(Array.from(getGPUSpatialAutocorrelationParameterValues({significanceLevel})));
  }
});

it('every parameter word, exact uint32 population limb and failure atomicity follows the declared layout', () => {
  expect(resolveSpatialAutocorrelationMinimumNeighbors('audit', undefined)).toBe(1);
  expect(resolveSpatialAutocorrelationMinimumNeighbors('audit', 0xffffffff)).toBe(0xffffffff);
  for (const minimumNeighbors of [0, -1, 1.5, NaN, Infinity, 2 ** 32])
    expect(() => resolveSpatialAutocorrelationMinimumNeighbors('audit', minimumNeighbors)).toThrow(
      /audit.*minimumNeighbors/
    );
  expect(
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA.fields.map(field => [
      field.name,
      field.wordOffset,
      field.defaultValue
    ])
  ).toEqual([
    ['significanceLevel', 0, 0.05],
    ['useFixedMoments', 1, 0],
    ['fixedCount', 2, 0],
    ['fixedMean', 3, 0],
    ['fixedVariance', 4, 0],
    ['fixedCountHigh', 5, 0],
    ['fixedCountLow', 6, 0],
    ['useExtendedFixedCount', 7, 0]
  ]);
  expect(
    Array.from(
      packGPUSpatialParameterValues(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA, {
        significanceLevel: 0.25,
        useFixedMoments: 1,
        fixedCount: 16777219,
        fixedMean: 4,
        fixedVariance: 25,
        fixedCountHigh: 256,
        fixedCountLow: 3,
        useExtendedFixedCount: 1
      })
    )
  ).toEqual([0.25, 1, 16777220, 4, 25, 256, 3, 1]);
  // IEEE754 single and base-65536 identities; 16777219=256*65536+3, uint32max=65535*65536+65535.
  for (const [count, encoded, high, low, extended] of [
    [2 ** 24, 2 ** 24, 0, 0, 0],
    [2 ** 24 + 3, 16777220, 256, 3, 1],
    [0xffffffff, 4294967296, 65535, 65535, 1]
  ]) {
    const target = new Float32Array(10).fill(71);
    expect(
      getGPUSpatialAutocorrelationParameterValues(
        {significanceLevel: 0.25, fixedMoments: {count, mean: 4, variance: 25}},
        target
      )
    ).toBe(target);
    expect(Array.from(target)).toEqual([0.25, 1, encoded, 4, 25, high, low, extended, 71, 71]);
    for (const invalid of [
      {count: 1, mean: 4, variance: 25},
      {count: 2 ** 32, mean: 4, variance: 25},
      {count: 2.5, mean: 4, variance: 25},
      {count: 8, mean: Infinity, variance: 25},
      {count: 8, mean: 4, variance: Number.MIN_VALUE}
    ]) {
      const before = Array.from(target);
      expect(() =>
        getGPUSpatialAutocorrelationParameterValues({fixedMoments: invalid}, target)
      ).toThrow();
      expect(Array.from(target)).toEqual(before);
    }
  }
  const target = Float32Array.of(71, 73);
  for (const selfWeight of [0, 0.5, 1, 2 ** -126]) {
    expect(Array.from(getGPUHotSpotAnalysisParameterValues({selfWeight}, target))).toEqual([
      selfWeight,
      73
    ]);
  }
  for (const selfWeight of [-1, NaN, Infinity, 2 ** 128, Number.MIN_VALUE, 2 ** -127]) {
    const before = Array.from(target);
    expect(() => getGPUHotSpotAnalysisParameterValues({selfWeight}, target)).toThrow();
    expect(Array.from(target)).toEqual(before);
  }
  for (const residualPower of [1, 2] as const)
    expect(
      Array.from(getGPULocalSpatialHeteroscedasticityParameterValues({residualPower}, target))
    ).toEqual([residualPower, 73]);
  for (const residualPower of [0, 1.5, 3, NaN]) {
    expect(() =>
      getGPULocalSpatialHeteroscedasticityParameterValues(
        {residualPower: residualPower as 1},
        target
      )
    ).toThrow();
    expect(Array.from(target)).toEqual([2, 73]);
  }
  const integers = Uint32Array.of(7, 11, 13, 17);
  expect(
    Array.from(
      getGPUMultivariateLocalGearyPermutationTestParameterValues(
        {seed: [0xffffffff, 0x80000001], significanceLevel: 0.5},
        integers
      )
    )
  ).toEqual([0xffffffff, 0x80000001, 0x3f000000, 17]);
  for (const seed of [[-1, 0], [0, 2 ** 32], [0, 0.5], [0], [0, 0, 0]]) {
    expect(() =>
      getGPUMultivariateLocalGearyPermutationTestParameterValues(
        {seed: seed as [number, number]},
        integers
      )
    ).toThrow();
    expect(Array.from(integers)).toEqual([0xffffffff, 0x80000001, 0x3f000000, 17]);
  }
  // SciPy scipy.stats.norm.isf(alpha/2) for the three stated two-sided levels.
  expect(GPU_HOT_SPOT_SIGNIFICANCE_LEVELS).toEqual([0.1, 0.05, 0.01]);
  for (const [index, expected] of [
    1.6448536269514729, 1.9599639845400545, 2.575829303548901
  ].entries()) {
    expect(Number.isFinite(GPU_HOT_SPOT_CRITICAL_Z_SCORES[index])).toBe(true);
    expect(Math.abs(GPU_HOT_SPOT_CRITICAL_Z_SCORES[index] - expected)).toBeLessThan(1e-14);
  }
  expect(GPU_LOCAL_MORAN_QUADRANT).toEqual({
    NOT_SIGNIFICANT: 0,
    HIGH_HIGH: 1,
    LOW_HIGH: 2,
    LOW_LOW: 3,
    HIGH_LOW: 4
  });
});

it('the analytic CPU twin preserves the shared float32 normal p-value floor', () => {
  // erfc(20/sqrt(2))=5.507248237212663e-89 (libm/SciPy), below the documented 2^-126 floor.
  // A zero or an unfloored double tail is not the GPU contract.
  expect(getTwoSidedPValue(20)).toBe(2 ** -126);
  expect(getTwoSidedPValue(-20)).toBe(2 ** -126);
});

it('local Moran and Gi/Gi* CPU twins match exhaustive null distributions on every output', () => {
  const maximumDifferences: Record<string, number> = {};
  const compare = (name: string, actual: readonly number[], expected: readonly number[]) => {
    expect(actual.length, name).toBe(expected.length);
    for (const [index, value] of expected.entries()) {
      if (Number.isNaN(value)) {
        expect(actual[index], `${name}[${index}]`).toBeNaN();
      } else {
        expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
        const difference = Math.abs(actual[index] - value);
        maximumDifferences[name] = Math.max(maximumDifferences[name] ?? 0, difference);
        // The double twin uses the shared erfc approximation; the independent oracle uses libm.
        expect(difference, `${name}[${index}]`).toBeLessThan(1e-7);
      }
    }
  };
  for (const reference of AUDIT_REFERENCE) {
    for (const scale of [1, 2]) {
      const input = {
        values: Float32Array.from(AUDIT_VALUES, value => scale * value + (scale === 2 ? 32 : 0)),
        mask: Uint32Array.from(AUDIT_MASK),
        weights: {
          offsets: Uint32Array.from(AUDIT_WEIGHTS.offsets),
          neighbors: Uint32Array.from(AUDIT_WEIGHTS.neighbors),
          weights: Float32Array.from(AUDIT_WEIGHTS.weights)
        },
        islands: reference.islands,
        rowStandardized: reference.rowStandardized,
        parameters: {significanceLevel: 0.2}
      };
      const moran = computeLocalMoranOracle(input);
      compare('localI', moran.localI, reference.localI);
      compare(
        'spatialLag',
        moran.spatialLag,
        reference.spatialLag.map(value => value * scale)
      );
      compare('moran.zScores', moran.zScores, reference.moranZ);
      compare('moran.pValues', moran.pValues, reference.moranP);
      compare('moran.adjustedPValues', moran.adjustedPValues, reference.moranP);
      compare('moran.quadrants', moran.quadrants, AUDIT_QUADRANTS);
      compare('moran.neighborCounts', moran.neighborCounts, AUDIT_NEIGHBOR_COUNTS);
      compare('moran.included', moran.included.map(Number), [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]);
      compare('moran.islands', moran.islands.map(Number), [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
      for (const selfWeight of [0, 1]) {
        const hotSpot = computeHotSpotOracle({...input, selfWeight});
        const expectedScores = selfWeight ? reference.starZ : reference.giZ;
        const expectedPValues = selfWeight ? reference.starP : reference.giP;
        const prefix = selfWeight ? 'star' : 'gi';
        compare(`${prefix}.zScores`, hotSpot.zScores, expectedScores);
        compare(`${prefix}.pValues`, hotSpot.pValues, expectedPValues);
        compare(`${prefix}.adjustedPValues`, hotSpot.adjustedPValues, expectedPValues);
        compare(`${prefix}.bins`, hotSpot.bins, selfWeight ? reference.starBins : AUDIT_GI_BINS);
        compare(`${prefix}.neighborCounts`, hotSpot.neighborCounts, AUDIT_NEIGHBOR_COUNTS);
        expect(hotSpot.included).toEqual(moran.included);
        expect(hotSpot.islands).toEqual(moran.islands);
        for (const result of [moran, hotSpot]) {
          compare(
            'moments',
            [
              result.moments.count,
              result.moments.mean,
              result.moments.variance,
              Math.sqrt(result.moments.variance),
              result.islandCount
            ],
            [
              reference.globalStatistics[0],
              reference.globalStatistics[1] * scale + (scale === 2 ? 32 : 0),
              reference.globalStatistics[2] * scale * scale,
              reference.globalStatistics[3] * scale,
              1
            ]
          );
          compare(
            'sumOfSquares',
            [result.moments.sumOfSquares],
            [reference.globalStatistics[0] * reference.globalStatistics[2] * scale * scale]
          );
        }
      }
    }
  }
  console.info('exhaustive local autocorrelation maximum absolute differences', maximumDifferences);
});
