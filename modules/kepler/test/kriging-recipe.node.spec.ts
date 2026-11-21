// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUInverseDistanceWeighting,
  GPUKriging
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
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

test.each([
  {bounds: [179, -0.5, 179.9, 0.5]},
  {bounds: [-1, 89, 1, 89.5]}
])('wrapped interpolation recipes construct and replay seam/pole searches at $bounds', ({
  bounds
}) => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.assign(device.limits, {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16_384,
    maxComputeWorkgroupsPerDimension: 65_535
  });
  const graph = new GPUCommandGraph(device);
  const add = vi.spyOn(graph, 'add');
  const columns = Object.fromEntries(
    ['longitude', 'latitude', 'temperature'].map(name => [
      name,
      createTransientView(graph, name, 'float32', 3)
    ])
  );
  const findLastDescriptor = Object.getOwnPropertyDescriptor(Array.prototype, 'findLast');
  try {
    // ES2022 consumers have no Array.findLast, including in their TypeScript declarations.
    Object.defineProperty(Array.prototype, 'findLast', {value: undefined, configurable: true});
    for (const kind of ['idw-points', 'kriging-points'] as const) {
      const common = {
        longitudeColumn: settings.longitudeColumn,
        latitudeColumn: settings.latitudeColumn,
        coordinateSpace: settings.coordinateSpace,
        metric: settings.metric,
        units: settings.units,
        grid: {...settings.grid, bounds},
        neighborCount: 3
      };
      const operation =
        kind === 'idw-points'
          ? {
              ...common,
              kind,
              column: 'temperature',
              searchRadius: 200000,
              power: 2,
              minimumNeighborCount: 1,
              taper: 'none'
            }
          : {
              ...common,
              kind,
              column: 'temperature',
              searchRadiusMeters: 200000,
              minimumNeighbors: 1,
              variogram: settings.variogram
            };
      const input = {
        ...recipe,
        id: kind,
        population: 'all',
        selectionRevision: undefined,
        operation
      };
      // Both boxes reach the seam/pole after expansion; historical clipped searches still refuse.
      expect(() => parseKeplerAnalysisRecipe(input)).toThrow(/search domain/);
      expect(() =>
        parseKeplerAnalysisRecipe({...input, operation: {...operation, longitudeSearch: 'clipped'}})
      ).toThrow(/search domain/);
      expect(() =>
        parseKeplerAnalysisRecipe({...input, operation: {...operation, longitudeSearch: 'typo'}})
      ).toThrow(/clipped.*wrapped/);
      const parsed = parseKeplerAnalysisRecipe({
        ...input,
        operation: {...operation, longitudeSearch: 'wrapped'}
      });
      const serialized = serializeKeplerAnalysisRecipe(parsed);
      expect(parsed.operation).toHaveProperty('longitudeSearch', 'wrapped');
      expect((parsed.operation as KeplerKrigingSettings).grid.bounds).not.toBe(bounds);
      expect(Object.isFrozen((parsed.operation as KeplerKrigingSettings).grid.bounds)).toBe(true);
      const restored = restoreKeplerAnalysisRecipe(serialized, {resolveDatasetRevision: () => 2});
      expect(restored).toEqual(parsed);
      const start = add.mock.calls.length;
      const result = addKeplerAnalysisRecipe(graph, restored, {source: recipe.source, columns});
      const ExpectedContributor = kind === 'idw-points' ? GPUInverseDistanceWeighting : GPUKriging;
      const contributor = add.mock.calls
        .slice(start)
        .map(call => call[0])
        .find(candidate => candidate instanceof ExpectedContributor);
      expect(contributor).toBeInstanceOf(ExpectedContributor);
      expect((contributor as GPUKriging).props.longitudeSearch).toBe('wrapped');
      result.destroyParameters();
    }
  } finally {
    if (findLastDescriptor) Object.defineProperty(Array.prototype, 'findLast', findLastDescriptor);
    else Reflect.deleteProperty(Array.prototype, 'findLast');
    device.destroy();
  }
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
