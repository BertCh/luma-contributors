// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerAnalysisRecipe,
  parseKeplerKrigingSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerKrigingSettings
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const settings: KeplerKrigingSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: [-1, -1, 1, 1], width: 4, height: 4},
  searchRadiusMeters: 200000,
  neighborCount: 8,
  minimumNeighbors: 3,
  variogram: {model: 'spherical', nugget: 0.1, sill: 10, rangeMeters: 100000}
};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'ordinary-kriging',
  source: {datasetId: 'stations', revision: 2},
  population: 'selected',
  selectionRevision: 3,
  backendPolicy: 'require-gpu',
  operation: {kind: 'kriging-points', column: 'temperature', ...settings}
};

test('ordinary kriging recipes normalize manual variogram units and preserve source/selection replay', () => {
  const parsed = parseKeplerKrigingSettings(settings);
  expect(parsed.variogram.nugget).toBe(Math.fround(0.1));
  expect(parsed.grid.bounds).not.toBe(settings.grid.bounds);
  expect(parsed.variogram).not.toBe(settings.variogram);
  expect(Object.isFrozen(parsed.variogram)).toBe(true);
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 2,
      resolveSelectionRevision: () => 3
    })
  ).toEqual(parseKeplerAnalysisRecipe(recipe));
  for (const override of [
    {neighborCount: 17},
    {neighborCount: 0},
    {minimumNeighbors: 9},
    {minimumNeighbors: 0},
    {searchRadiusMeters: Infinity},
    {searchRadiusMeters: 0},
    {searchRadiusMeters: 1e-50},
    {metric: 'native'},
    {units: 'kilometers'},
    {searchRadius: 100},
    {grid: {...settings.grid, width: 8193}},
    {grid: {...settings.grid, width: 8192, height: 8192}},
    {grid: {...settings.grid, bounds: [179, -1, 179.9, 1]}},
    {grid: {...settings.grid, bounds: [-1, 89, 1, 89.5]}}
  ])
    expect(() => parseKeplerKrigingSettings({...settings, ...override})).toThrow();
  for (const override of [
    {model: 'linear'},
    {nugget: -1},
    {nugget: Infinity},
    {sill: Infinity},
    {nugget: 0, sill: 0},
    {nugget: 2e38, sill: 2e38},
    {rangeMeters: 0},
    {rangeMeters: 1e-50},
    {rangeMeters: Infinity},
    {range: 1000}
  ])
    expect(() =>
      parseKeplerKrigingSettings({...settings, variogram: {...settings.variogram, ...override}})
    ).toThrow();
});

test('kriging settings accept every analysis variogram family and reject unknown ones', () => {
  for (const model of ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2'] as const)
    expect(
      parseKeplerKrigingSettings({...settings, variogram: {...settings.variogram, model}}).variogram
        .model
    ).toBe(model);
  for (const model of ['matern-1/2', 'linear', 'Gaussian', undefined])
    expect(() =>
      parseKeplerKrigingSettings({...settings, variogram: {...settings.variogram, model}} as never)
    ).toThrow();
});

test('kriging routing admits bounded geographic grids and requires explicit packed Float32 sources', () => {
  const request = {
    kind: 'analysis' as const,
    recipe,
    estimatedBytes: 65536,
    input: {
      rowCount: 4096,
      chunkCount: 1,
      formats: {longitude: 'float32', latitude: 'float32', temperature: 'float32'}
    }
  };
  const capabilities = {
    webgpu: true,
    availableBytes: 100000,
    duckdb: false,
    analysisBackend: false
  };
  expect(selectKeplerExecutionRoute(request, capabilities)).toEqual({
    backend: 'luma',
    reason: 'supported'
  });
  for (const rowCount of [0, 2 ** 24])
    expect(
      selectKeplerExecutionRoute({...request, input: {...request.input, rowCount}}, capabilities)
        .reason
    ).toBe('capacity');
  expect(
    selectKeplerExecutionRoute({...request, input: {...request.input, chunkCount: 2}}, capabilities)
      .reason
  ).toBe('packing-required');
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {...request.input, formats: {...request.input.formats, longitude: 'float64'}}
      },
      capabilities
    ).reason
  ).toBe('precision');
});
