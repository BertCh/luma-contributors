// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
  parseKeplerAnalysisRecipe,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'quantile-lisa',
  source: {datasetId: 'observations', revision: 2},
  population: 'selected',
  selectionRevision: 3,
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'quantile-lisa-points',
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
    quantileCount: 4,
    selectedQuantile: 4,
    seed: 2 ** 40 + 5,
    permutationCount: 999,
    significanceLevel: 0.05,
    inference: 'legacy-directed-pseudo',
    multipleTesting: 'none',
    islands: 'exclude',
    quantileMethod: 'type-5'
  }
};

test('Quantile LISA recipes preserve exact seed, membership and limited inference semantics', () => {
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

test('Quantile LISA rejects unsupported inference, invalid classes and unrepresentable controls', () => {
  for (const changes of [
    {quantileCount: 1},
    {selectedQuantile: 0},
    {selectedQuantile: 5},
    {seed: -1},
    {seed: Number.MAX_SAFE_INTEGER + 1},
    {permutationCount: 0},
    {permutationCount: MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS + 1},
    {significanceLevel: Number.MIN_VALUE},
    {significanceLevel: 1 - Number.EPSILON},
    {inference: 'two-sided'},
    {multipleTesting: 'benjamini-hochberg'},
    {islands: 'keep'},
    {quantileMethod: 'type-7'},
    {minimumNeighbors: 4}
  ]) {
    expect(() =>
      parseKeplerAnalysisRecipe({...recipe, operation: {...recipe.operation, ...changes}})
    ).toThrow();
  }
});

test('Quantile LISA routing checks source quantile capacity and geographic precision', () => {
  const capabilities = {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false};
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
  for (const rowCount of [0, 2, 4, 2 ** 24]) {
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
          formats: {...request.input.formats, longitude: 'float64'}
        }
      },
      capabilities
    )
  ).toEqual({backend: 'unavailable', reason: 'precision'});
});
