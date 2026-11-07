// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerIDWSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerIDWSettings
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const settings: KeplerIDWSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: [-74, 40, -73, 41], width: 128, height: 128},
  searchRadius: 25000,
  power: 2,
  neighborCount: 8,
  minimumNeighborCount: 3,
  taper: 'none'
};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'interpolation',
  source: {datasetId: 'samples', revision: 1},
  population: 'selected',
  selectionRevision: 2,
  backendPolicy: 'require-gpu',
  operation: {kind: 'idw-points', column: 'value', ...settings}
};

test('IDW recipes retain explicit grid and selection without requiring a weights reference', () => {
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 1,
    resolveSelectionRevision: () => 2
  });
  expect(restored).toEqual(recipe);
  if (restored.operation.kind !== 'idw-points') throw new Error('Expected IDW');
  expect(restored.operation.grid.bounds).not.toBe(settings.grid.bounds);
  expect(Object.isFrozen(restored.operation.grid.bounds)).toBe(true);
});

test('IDW rejects unbounded work and geographic query boxes crossing unsupported seams', () => {
  for (const change of [
    {grid: {...settings.grid, width: 8193}},
    {grid: {...settings.grid, bounds: [179, 40, 179.99, 41]}},
    {grid: {...settings.grid, bounds: [-1, 89, 1, 89.8]}},
    {grid: {...settings.grid, bounds: [0, 0, 0, 1]}},
    {searchRadius: Infinity},
    {searchRadius: Number.MIN_VALUE},
    {power: 33},
    {power: -1},
    {neighborCount: 65},
    {minimumNeighborCount: 9},
    {units: 'degrees'},
    {variance: true}
  ])
    expect(() => parseKeplerIDWSettings({...settings, ...change})).toThrow();
});

test('IDW routing uses source capacity and coordinate precision independently of raster size', () => {
  const capabilities = {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false};
  const request = {
    kind: 'analysis' as const,
    recipe,
    estimatedBytes: 128,
    input: {
      rowCount: 5,
      chunkCount: 1,
      formats: {value: 'float32', longitude: 'float32', latitude: 'float32'}
    }
  };
  expect(selectKeplerExecutionRoute(request, capabilities).reason).toBe('supported');
  for (const rowCount of [0, 2 ** 24])
    expect(
      selectKeplerExecutionRoute({...request, input: {...request.input, rowCount}}, capabilities)
        .reason
    ).toBe('capacity');
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {...request.input, formats: {...request.input.formats, latitude: 'float64'}}
      },
      capabilities
    ).reason
  ).toBe('precision');
});
