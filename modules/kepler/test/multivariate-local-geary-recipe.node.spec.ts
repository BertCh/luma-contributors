// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {parseKeplerAnalysisRecipe, type KeplerAnalysisRecipe} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

function makeRecipe(operation: Record<string, unknown>): KeplerAnalysisRecipe {
  return {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'multivariate',
    source: {datasetId: 'dataset', revision: 1},
    population: 'all',
    backendPolicy: 'require-gpu',
    operation: {
      kind: 'multivariate-local-geary-points',
      columns: ['a', 'b'],
      weights: {
        kind: 'knn',
        k: 4,
        longitudeColumn: 'longitude',
        latitudeColumn: 'latitude',
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      },
      significanceLevel: 0.05,
      permutationCount: 99,
      seed: 1,
      correction: 'none',
      tail: 'equal-tail',
      ...operation
    }
  } as unknown as KeplerAnalysisRecipe;
}

test('multivariate Local Geary recipes accept 2 to 8 distinct columns', () => {
  const parsed = parseKeplerAnalysisRecipe(makeRecipe({}));
  expect(parsed.operation).toMatchObject({columns: ['a', 'b'], permutationCount: 99});
  const eight = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  expect(parseKeplerAnalysisRecipe(makeRecipe({columns: eight})).operation).toMatchObject({
    columns: eight
  });
  expect(() => parseKeplerAnalysisRecipe(makeRecipe({columns: ['a']}))).toThrow();
  expect(() => parseKeplerAnalysisRecipe(makeRecipe({columns: [...eight, 'i']}))).toThrow();
  expect(() => parseKeplerAnalysisRecipe(makeRecipe({columns: ['a', 'a']}))).toThrow();
});

test('multivariate Local Geary bounds permutations and correction', () => {
  expect(() => parseKeplerAnalysisRecipe(makeRecipe({permutationCount: 10000}))).toThrow();
  expect(parseKeplerAnalysisRecipe(makeRecipe({permutationCount: 9999}))).toBeDefined();
  expect(() => parseKeplerAnalysisRecipe(makeRecipe({correction: 'holm'}))).toThrow();
  expect(() => parseKeplerAnalysisRecipe(makeRecipe({tail: 'other'}))).toThrow();
});

test('router routes multivariate Local Geary above 256 rows to the tiled contributor', () => {
  const recipe = makeRecipe({});
  const route = (rowCount: number) =>
    selectKeplerExecutionRoute(
      {
        kind: 'analysis' as const,
        recipe,
        input: {
          rowCount,
          chunkCount: 1,
          formats: {a: 'float32', b: 'float32', longitude: 'float32', latitude: 'float32'}
        },
        estimatedBytes: 128
      },
      {webgpu: true, availableBytes: 1024}
    );
  expect(route(256)).toEqual({backend: 'luma', reason: 'supported'});
  expect(route(257)).toEqual({backend: 'luma', reason: 'supported'});
  expect(route(2 ** 24 - 1)).toEqual({backend: 'luma', reason: 'supported'});
  expect(route(2 ** 24 + 1)).toEqual({backend: 'unavailable', reason: 'capacity'});
});
