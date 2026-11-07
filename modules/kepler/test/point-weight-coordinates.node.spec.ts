// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  parseKeplerPointWeightCoordinates,
  getKeplerPointWeightCoordinateColumns,
  getKeplerPointWeightSpatialContext,
  matchesKeplerPointWeightSpatialContext,
  areKeplerPointWeightCoordinatesEqual
} from '../src/analysis/point-weight-coordinates';
import {addKeplerPointWeights} from '../src/analysis/point-weights';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import {
  parseKeplerPointWeightsSettings,
  serializeKeplerAnalysisRecipe,
  restoreKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const coordinates = {
  coordinateSpace: 'planar' as const,
  metric: 'native' as const,
  units: 'meters' as const,
  xColumn: 'east',
  yColumn: 'north',
  crs: 'EPSG:32618',
  bounds: [0, 0, 1000, 1000] as const
};
const settings = {...coordinates, kind: 'knn' as const, k: 2};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'projected-rates',
  source: {datasetId: 'projected', revision: 1},
  population: 'all',
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'spatial-rate-points',
    events: 'events',
    populations: 'population',
    weights: settings,
    scale: 1,
    islandPolicy: 'undefined'
  }
};

test('projected declarations preserve explicit roles, detached bounds and replay identity', () => {
  const parsed = parseKeplerPointWeightCoordinates(coordinates);
  expect(parsed).toEqual(coordinates);
  if (parsed.coordinateSpace !== 'planar') throw new Error('Expected planar coordinates');
  expect(parsed.bounds).not.toBe(coordinates.bounds);
  expect(Object.isFrozen(parsed.bounds)).toBe(true);
  expect(getKeplerPointWeightCoordinateColumns(parsed)).toEqual(['east', 'north']);
  const context = getKeplerPointWeightSpatialContext(parsed);
  expect(context).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {kind: 'crs', id: 'EPSG:32618', crs: 'EPSG:32618'}
  });
  expect(matchesKeplerPointWeightSpatialContext(parsed, context)).toBe(true);
  expect(matchesKeplerPointWeightSpatialContext(parsed, {...context, frame: undefined})).toBe(
    false
  );
  expect(areKeplerPointWeightCoordinatesEqual(parsed, {...parsed, crs: 'EPSG:3857'})).toBe(false);
  expect(areKeplerPointWeightCoordinatesEqual(parsed, {...parsed, bounds: [0, 0, 999, 1000]})).toBe(
    false
  );
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 1
    })
  ).toEqual(recipe);
  const input = {
    rowCount: 3,
    chunkCount: 1,
    formats: {
      east: 'float32',
      north: 'float32',
      events: 'float32',
      population: 'float32'
    }
  };
  const capabilities = {webgpu: true, availableBytes: 1024};
  expect(
    selectKeplerExecutionRoute({kind: 'analysis', recipe, input, estimatedBytes: 128}, capabilities)
      .backend
  ).toBe('luma');
  expect(
    selectKeplerExecutionRoute(
      {
        kind: 'analysis',
        recipe,
        input: {...input, formats: {...input.formats, east: 'float64'}},
        estimatedBytes: 128
      },
      capabilities
    ).reason
  ).toBe('precision');
});

test('projected Float32 extents and radii reject unsafe arithmetic without inventing geographic roles', () => {
  expect(
    parseKeplerPointWeightCoordinates({...coordinates, bounds: [181, 0, 181, 0]})
  ).toMatchObject({bounds: [181, 0, 181, 0]});
  for (const override of [
    {longitudeColumn: 'east'},
    {metric: 'great-circle'},
    {units: 'feet'},
    {crs: ' '},
    {bounds: [0, 0, Infinity, 1]},
    {bounds: [1, 0, 0, 1]},
    {bounds: [-1e20, 0, 1e20, 1]}
  ])
    expect(() => parseKeplerPointWeightCoordinates({...coordinates, ...override})).toThrow();
  for (const radiusMeters of [0, -1, 1e20, 1e-30, Infinity])
    expect(() =>
      parseKeplerPointWeightsSettings({
        ...coordinates,
        kind: 'distance-band',
        radiusMeters,
        neighborCapacity: 6
      })
    ).toThrow();
  expect(
    parseKeplerPointWeightsSettings({
      ...coordinates,
      kind: 'distance-band',
      radiusMeters: 100.1,
      neighborCapacity: 6
    })
  ).toMatchObject({radiusMeters: Math.fround(100.1)});
});

test('projected builder and borrowed recipes retain CRS metadata and reject stale extent or frame', () => {
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
  const pointWeights = addKeplerPointWeights(graph, {
    source: recipe.source,
    settings,
    x: createTransientView(graph, 'east', 'float32', 3),
    y: createTransientView(graph, 'north', 'float32', 3)
  });
  const columns = {
    events: createTransientView(graph, 'events', 'float32', 3),
    population: createTransientView(graph, 'population', 'float32', 3)
  };
  try {
    expect(pointWeights.weights.spatialContext).toEqual(
      getKeplerPointWeightSpatialContext(coordinates)
    );
    expect(pointWeights.weights.neighbors.length).toBe(6);
    expect(pointWeights.parameterByteLength).toBe(48);
    for (const invalid of [
      {...pointWeights, settings: {...settings, bounds: [0, 0, 1001, 1000] as const}},
      {...pointWeights, settings: {...settings, xColumn: 'other'}},
      {
        ...pointWeights,
        weights: {
          ...pointWeights.weights,
          spatialContext: {
            ...getKeplerPointWeightSpatialContext(coordinates),
            frame: {kind: 'crs' as const, id: 'other', crs: 'other'}
          }
        }
      }
    ])
      expect(() =>
        addKeplerAnalysisRecipe(graph, recipe, {
          source: recipe.source,
          columns,
          pointWeights: invalid
        })
      ).toThrow();
    const result = addKeplerAnalysisRecipe(graph, recipe, {
      source: recipe.source,
      columns,
      pointWeights
    });
    if (!('pointWeights' in result)) throw new Error('Expected point weights');
    expect(result.pointWeights.parameterByteLength).toBe(0);
    expect(result.pointWeights.coordinateMask).toBe(pointWeights.coordinateMask);
    result.destroyParameters();
  } finally {
    pointWeights.destroyParameters();
    device.destroy();
  }
});
