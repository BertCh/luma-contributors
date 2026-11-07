// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerAnalysisRecipe,
  parseKeplerPointWeightsSettings,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {getKeplerPointWeightsMaximumNeighbors} from '../src/analysis/point-weights';
import {
  assertKeplerStorageBindingFits,
  getKeplerMaximumBindingElements
} from '../src/analysis/analysis-scaling';

const weights = {
  kind: 'distance-band',
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  radiusMeters: 1000,
  neighborCapacity: 4096,
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};

function makeRecipe(operation: Record<string, unknown>): KeplerAnalysisRecipe {
  return {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'bivariate',
    source: {datasetId: 'dataset', revision: 1},
    population: 'all',
    backendPolicy: 'require-gpu',
    operation: {
      kind: 'local-moran-points',
      column: 'x',
      weights,
      minimumNeighbors: 1,
      islands: 'exclude',
      multipleTesting: 'none',
      significanceLevel: 0.05,
      inference: 'permutation-conditional-two-sided',
      permutationCount: 99,
      seed: 1,
      ...operation
    }
  } as unknown as KeplerAnalysisRecipe;
}

test('bivariate Local Moran recipes carry a second column and require permutations', () => {
  const parsed = parseKeplerAnalysisRecipe(makeRecipe({secondColumn: 'y'}));
  expect(parsed.operation).toMatchObject({kind: 'local-moran-points', secondColumn: 'y'});
  expect(() =>
    parseKeplerAnalysisRecipe(
      makeRecipe({
        secondColumn: 'y',
        inference: 'analytic-conditional-normal-two-sided',
        permutationCount: undefined,
        seed: undefined,
        minimumNeighbors: 4
      })
    )
  ).toThrow();
});

test('distance-band permutation degree defaults to 64 and accepts 64 to 512', () => {
  expect(getKeplerPointWeightsMaximumNeighbors(parseKeplerPointWeightsSettings(weights))).toBe(64);
  const settings = parseKeplerPointWeightsSettings({...weights, permutationDegree: 256});
  expect(getKeplerPointWeightsMaximumNeighbors(settings)).toBe(256);
  expect(() => parseKeplerPointWeightsSettings({...weights, permutationDegree: 513})).toThrow();
  expect(() => parseKeplerPointWeightsSettings({...weights, permutationDegree: 32})).toThrow();
});

test('storage binding guard compares against both device limits', () => {
  const limits = {maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 256 * 2 ** 20};
  expect(getKeplerMaximumBindingElements(limits, 4)).toBe(2 ** 25);
  expect(() => assertKeplerStorageBindingFits(limits, 2 ** 25, 4, 'slots')).not.toThrow();
  expect(() => assertKeplerStorageBindingFits(limits, 2 ** 25 + 1, 4, 'slots')).toThrow();
});
