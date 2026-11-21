// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {generateDotCountsOnCPU} from './dot-density-cpu';
import {getGPUDotDensityParameterValues} from '../../../src/gpu-spatial-analysis/dot-density/index';
import {getPhilox4x32, getPhiloxUnitFloat} from '../../../src/analysis-core/kernels/philox-random';
import {
  matchesTriangleBarycentricOracle,
  TRIANGLE_BARYCENTRIC_WEIGHTS
} from './dot-density-output-oracle-fixtures';

it('rejects draw-dependent barycentric permutations that concentrate every sample near one vertex', () => {
  const biased = TRIANGLE_BARYCENTRIC_WEIGHTS.map(row =>
    [...row].sort((left, right) => left - right)
  );
  expect(matchesTriangleBarycentricOracle(TRIANGLE_BARYCENTRIC_WEIGHTS)).toBe(true);
  expect(
    matchesTriangleBarycentricOracle(
      TRIANGLE_BARYCENTRIC_WEIGHTS.map(row => [row[2], row[0], row[1]])
    )
  ).toBe(true);
  expect(matchesTriangleBarycentricOracle(biased)).toBe(false);
});

it('assigns integer requests two and three to literal category IDs', () => {
  const actual = generateDotCountsOnCPU({
    positions: new Float32Array([0, 0, 1, 0, 0, 1]),
    featureOffsets: new Uint32Array([0, 1]),
    polygonOffsets: new Uint32Array([0, 1]),
    ringOffsets: new Uint32Array([0, 3]),
    values: new Float32Array([2, 3]),
    categoryCount: 2,
    capacity: 5,
    parameters: getGPUDotDensityParameterValues({seed: 123})
  });
  expect(Array.from(actual.categories)).toEqual([0, 0, 1, 1, 1]);
  expect(actual.count).toBe(5);
});

it('matches independent Python Philox known-answer blocks and line fractions', () => {
  expect(getPhilox4x32([0, 0, 0, 0], [123, 7])).toEqual([
    513388108, 4178107452, 2792688318, 3752164380
  ]);
  expect(
    [0, 1, 2].map(rank => getPhiloxUnitFloat(getPhilox4x32([0, rank, 0, 0], [7, 3])[0]))
  ).toEqual([0.20142346620559692, 0.20847898721694946, 0.09203100204467773]);
});
