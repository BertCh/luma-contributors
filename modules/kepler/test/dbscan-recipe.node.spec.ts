// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerDBSCANSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerDBSCANSettings
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const settings: KeplerDBSCANSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  epsilon: 500,
  minimumSamples: 4
};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'clusters',
  source: {datasetId: 'points', revision: 1},
  population: 'selected',
  selectionRevision: 2,
  backendPolicy: 'require-gpu',
  operation: {kind: 'dbscan-points', ...settings}
};

test('DBSCAN recipes retain geographic distances and selection without a value column', () => {
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 1,
      resolveSelectionRevision: () => 2
    })
  ).toEqual(recipe);
  expect(() =>
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 1,
      resolveSelectionRevision: () => 3
    })
  ).toThrow('selection');
});

test('DBSCAN rejects unsupported metrics, implicit units, zero radius and excessive core rank', () => {
  for (const change of [
    {metric: 'native'},
    {units: 'degrees'},
    {epsilon: Infinity},
    {epsilon: Number.MIN_VALUE},
    {epsilon: 0},
    {minimumSamples: 0},
    {minimumSamples: 257},
    {column: 'value'},
    {maximumIterations: 1}
  ])
    expect(() => parseKeplerDBSCANSettings({...settings, ...change})).toThrow();
});

test('DBSCAN routes geographic source rows within the bounded OPTICS envelope', () => {
  const capabilities = {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false};
  const request = {
    kind: 'analysis' as const,
    recipe,
    estimatedBytes: 128,
    input: {rowCount: 4, chunkCount: 1, formats: {longitude: 'float32', latitude: 'float32'}}
  };
  expect(selectKeplerExecutionRoute(request, capabilities).reason).toBe('supported');
  for (const rowCount of [0, 4097])
    expect(
      selectKeplerExecutionRoute({...request, input: {...request.input, rowCount}}, capabilities)
        .reason
    ).toBe('capacity');
  expect(
    selectKeplerExecutionRoute(
      {...request, input: {...request.input, formats: {longitude: 'float64', latitude: 'float32'}}},
      capabilities
    ).reason
  ).toBe('precision');
});
