// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getConformalPredictionRanks} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-ranks';

it('selects exact ranks at eight million calibration rows without allocating rows', () => {
  expect(getConformalPredictionRanks(8_388_607, 0.25)).toEqual({
    lowerRank: 2_097_152,
    upperRank: 6_291_456
  });
});

it('uses the packed alpha exactly without decimal boundary snapping', () => {
  expect(getConformalPredictionRanks(49, 0.02)).toEqual({lowerRank: 0, upperRank: 50});
  expect(getConformalPredictionRanks(9, 0.7)).toEqual({lowerRank: 6, upperRank: 4});
  expect(getConformalPredictionRanks(9, 0.9)).toEqual({lowerRank: 8, upperRank: 2});
  expect(getConformalPredictionRanks(9, 0.1)).toEqual({lowerRank: 1, upperRank: 9});
});

it('handles uint32 count boundaries and tiny alpha without float products', () => {
  expect(getConformalPredictionRanks(0xfffffffe, 0.5)).toEqual({
    lowerRank: 2_147_483_647,
    upperRank: 2_147_483_648
  });
  expect(getConformalPredictionRanks(0xfffffffe, 1 - 2 ** -24)).toEqual({
    lowerRank: 4_294_967_039,
    upperRank: 256
  });
  for (const miscoverage of [2 ** -149, 2 ** -100, 2 ** -40]) {
    expect(getConformalPredictionRanks(0xfffffffe, miscoverage)).toEqual({
      lowerRank: 0,
      upperRank: 0xffffffff
    });
  }
  expect(getConformalPredictionRanks(0, 0.25)).toEqual({lowerRank: 0, upperRank: 1});
});

it('matches exact dyadic fractions on either side of integer ranks', () => {
  for (const count of [6, 7, 8, 99, 8_388_606, 8_388_607, 8_388_608, 0xfffffffe]) {
    for (const numerator of [1, 3, 7, 13, 31]) {
      const denominator = 32;
      const lowerRank = Math.floor((numerator * (count + 1)) / denominator);
      expect(getConformalPredictionRanks(count, numerator / denominator)).toEqual({
        lowerRank,
        upperRank: count + 1 - lowerRank
      });
    }
  }
});

it('rejects invalid counts and alpha values after packing', () => {
  for (const count of [-1, 1.5, 0xffffffff, NaN]) {
    expect(() => getConformalPredictionRanks(count, 0.1)).toThrow('calibrationCount');
  }
  for (const miscoverage of [0, 1, 1 - 1e-10, NaN, Infinity, 1e-100]) {
    expect(() => getConformalPredictionRanks(99, miscoverage)).toThrow('miscoverage');
  }
});

it('preserves the existing tenth, twentieth and two-fifths boundary cases', () => {
  for (const [count, miscoverage, lowerRank, upperRank] of [
    [99, 0.1, 10, 90],
    [19, 0.05, 1, 19],
    [24, 0.4, 10, 15],
    [44, 0.4, 18, 27],
    [40, 0.1, 4, 37],
    [5, 0.1, 0, 6]
  ]) {
    expect(getConformalPredictionRanks(count, miscoverage)).toEqual({lowerRank, upperRank});
  }
});
