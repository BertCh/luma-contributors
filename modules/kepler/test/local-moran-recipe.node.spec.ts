// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerAnalysisRecipe,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'local-moran',
  source: {datasetId: 'observations', revision: 2},
  population: 'selected',
  selectionRevision: 3,
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'local-moran-points',
    column: 'value',
    weights: {
      kind: 'knn',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 8,
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    },
    minimumNeighbors: 4,
    islands: 'exclude',
    multipleTesting: 'benjamini-hochberg',
    significanceLevel: 0.05,
    inference: 'analytic-conditional-normal-two-sided'
  }
};

test('local Moran recipes retain explicit inference, point topology and selected population identity', () => {
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 2,
    resolveSelectionRevision: () => 3
  });
  expect(restored).toEqual(recipe);
  expect(restored.operation).not.toBe(recipe.operation);
  expect(Object.isFrozen(restored.operation)).toBe(true);
  expect(() =>
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 2,
      resolveSelectionRevision: () => 4
    })
  ).toThrow('selection');
});

test('local Moran rejects inference changes and significance levels lost at Float32 endpoints', () => {
  for (const significanceLevel of [
    0,
    1,
    -0.1,
    Infinity,
    NaN,
    Number.MIN_VALUE,
    1 - Number.EPSILON
  ]) {
    expect(() =>
      parseKeplerAnalysisRecipe({
        ...recipe,
        operation: {...recipe.operation, significanceLevel}
      })
    ).toThrow('Float32');
  }
  for (const changes of [
    {inference: 'permutation'},
    {quadrantGating: 'none'},
    {minimumNeighbors: 0},
    {islands: 'own-rate'},
    {multipleTesting: 'automatic'}
  ]) {
    expect(() =>
      parseKeplerAnalysisRecipe({...recipe, operation: {...recipe.operation, ...changes}})
    ).toThrow();
  }
});

test('local Moran routing checks both coordinate precision and contributor row capacity', () => {
  const capabilities = {webgpu: true, availableBytes: 1024};
  const request = {
    kind: 'analysis' as const,
    recipe,
    input: {
      rowCount: 10,
      chunkCount: 1,
      formats: {value: 'float32', longitude: 'float32', latitude: 'float32'}
    },
    estimatedBytes: 128
  };
  expect(selectKeplerExecutionRoute(request, capabilities)).toEqual({
    backend: 'luma',
    reason: 'supported'
  });
  for (const rowCount of [0, 2 ** 24]) {
    expect(
      selectKeplerExecutionRoute({...request, input: {...request.input, rowCount}}, capabilities)
    ).toEqual({backend: 'unavailable', reason: 'capacity'});
  }
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {
          ...request.input,
          formats: {...request.input.formats, latitude: 'float64'}
        }
      },
      capabilities
    )
  ).toEqual({backend: 'unavailable', reason: 'precision'});
});

test('conditional permutation Local Moran serializes explicit controls and rejects a mismatched testing family', () => {
  const permutationRecipe = parseKeplerAnalysisRecipe({
    ...recipe,
    operation: {
      ...recipe.operation,
      minimumNeighbors: 1,
      inference: 'permutation-conditional-two-sided',
      permutationCount: 9999,
      seed: 2 ** 32 - 1
    }
  });
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(permutationRecipe), {
      resolveDatasetRevision: () => 2,
      resolveSelectionRevision: () => 3
    })
  ).toEqual(permutationRecipe);
  for (const changes of [
    {permutationCount: undefined},
    {permutationCount: 0},
    {permutationCount: 10000},
    {permutationCount: 1.5},
    {seed: undefined},
    {seed: -1},
    {seed: 2 ** 32},
    {seed: 0.5},
    {minimumNeighbors: 2}
  ])
    expect(() =>
      parseKeplerAnalysisRecipe({
        ...permutationRecipe,
        operation: {
          ...permutationRecipe.operation,
          ...changes
        }
      })
    ).toThrow();
  expect(() =>
    parseKeplerAnalysisRecipe({...recipe, operation: {...recipe.operation, seed: 0}})
  ).toThrow('Analytic');
});

test('conditional permutation Local Moran has a bounded source while analytic routing stays compatible', () => {
  const capabilities = {webgpu: true, availableBytes: 1024};
  for (const inference of [
    'analytic-conditional-normal-two-sided',
    'permutation-conditional-two-sided'
  ] as const) {
    const boundedRecipe = parseKeplerAnalysisRecipe({
      ...recipe,
      operation: {
        ...recipe.operation,
        inference,
        minimumNeighbors: 1,
        ...(inference === 'permutation-conditional-two-sided' ? {permutationCount: 1, seed: 0} : {})
      }
    });
    for (const rowCount of [4096, 4097, 2 ** 24 - 1, 2 ** 24]) {
      const route = selectKeplerExecutionRoute(
        {
          kind: 'analysis',
          recipe: boundedRecipe,
          input: {
            rowCount,
            chunkCount: 1,
            formats: {value: 'float32', longitude: 'float32', latitude: 'float32'}
          },
          estimatedBytes: 128
        },
        capabilities
      );
      expect(route.reason).toBe(rowCount >= 2 ** 24 ? 'capacity' : 'supported');
    }
  }
});
