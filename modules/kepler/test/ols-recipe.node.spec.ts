// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  MAXIMUM_KEPLER_OLS_SOURCE_ROWS,
  MAXIMUM_KEPLER_OLS_PREDICTORS,
  parseKeplerOLSSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerOLSSettings
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const settings: KeplerOLSSettings = {
  responseColumn: 'response',
  predictorColumns: ['second', 'first'],
  intercept: 'included',
  ridgeLambda: 0,
  variance: 'degrees-of-freedom',
  missing: 'complete-cases'
};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'model',
  source: {datasetId: 'observations', revision: 1},
  population: 'selected',
  selectionRevision: 2,
  backendPolicy: 'require-gpu',
  operation: {kind: 'ols', ...settings}
};

test('OLS replay preserves predictor order, complete-case population and fixed variance conventions', () => {
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 1,
    resolveSelectionRevision: () => 2
  });
  expect(restored).toEqual(recipe);
  if (restored.operation.kind !== 'ols') throw new Error('Expected OLS');
  expect(restored.operation.predictorColumns).toEqual(['second', 'first']);
  expect(restored.operation.predictorColumns).not.toBe(settings.predictorColumns);
  expect(Object.isFrozen(restored.operation.predictorColumns)).toBe(true);
  expect(() =>
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 1,
      resolveSelectionRevision: () => 3
    })
  ).toThrow('selection');
});

test('OLS admits all 15 core predictors and rejects invalid design or silently changed ridge settings', () => {
  const predictorColumns = Array.from(
    {length: MAXIMUM_KEPLER_OLS_PREDICTORS},
    (_, index) => `predictor${index}`
  );
  expect(parseKeplerOLSSettings({...settings, predictorColumns}).predictorColumns).toHaveLength(15);
  expect(parseKeplerOLSSettings({...settings, ridgeLambda: 0.1}).ridgeLambda).toBe(
    Math.fround(0.1)
  );
  for (const change of [
    {predictorColumns: []},
    {predictorColumns: [...predictorColumns, 'extra']},
    {predictorColumns: ['first', 'first']},
    {predictorColumns: ['response']},
    {predictorColumns: ['']},
    {ridgeLambda: -1},
    {ridgeLambda: Infinity},
    {ridgeLambda: Number.MIN_VALUE},
    {ridgeLambda: NaN},
    {intercept: 'excluded'},
    {variance: 'maximum-likelihood'},
    {missing: 'impute-zero'},
    {robust: true}
  ])
    expect(() => parseKeplerOLSSettings({...settings, ...change})).toThrow();
});

test('OLS routing validates all design fields and explicitly bounded source packing', () => {
  const capabilities = {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false};
  const request = {
    kind: 'analysis' as const,
    recipe,
    estimatedBytes: 128,
    input: {
      rowCount: 10,
      chunkCount: 1,
      formats: {response: 'float32', first: 'float32', second: 'float32'}
    }
  };
  expect(selectKeplerExecutionRoute(request, capabilities).reason).toBe('supported');
  expect(
    selectKeplerExecutionRoute(
      {...request, input: {...request.input, rowCount: MAXIMUM_KEPLER_OLS_SOURCE_ROWS}},
      capabilities
    ).reason
  ).toBe('supported');
  for (const rowCount of [0, MAXIMUM_KEPLER_OLS_SOURCE_ROWS + 1])
    expect(
      selectKeplerExecutionRoute({...request, input: {...request.input, rowCount}}, capabilities)
        .reason
    ).toBe('capacity');
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {...request.input, formats: {...request.input.formats, second: 'float64'}}
      },
      capabilities
    ).reason
  ).toBe('precision');
  expect(
    selectKeplerExecutionRoute({...request, input: {...request.input, chunkCount: 2}}, capabilities)
      .reason
  ).toBe('packing-required');
});
