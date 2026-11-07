// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogDefaultParameters,
  normalizeCatalogParameters,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';
import type {CatalogOperationBinding} from '../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM5_RECIPE_BINDINGS as BINDINGS} from '../src/analysis/catalog-platform5-recipes';

function points(
  rows: {coordinates: number[]; properties?: Record<string, unknown>}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map(row => ({
      geometry: {type: 'Point', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  };
}

function checkBinding(binding: CatalogOperationBinding): void {
  expect(binding.engine).toBe('gpu-or-cpu');
  expect(binding.gpuContributorNames.length).toBeGreaterThan(0);
  expect(binding.run).toBeTypeOf('function');
  expect(binding.runGpu).toBeTypeOf('function');
  // Defaults must satisfy their own declared ranges.
  expect(() =>
    normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))
  ).not.toThrow();
}

const SQUARE = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
  [0, 0]
];

test('recipe bindings are unique, declared and runnable on the CPU engine', () => {
  expect(BINDINGS.map(binding => binding.id).sort()).toEqual([
    'recipe-hot-spot-lattice',
    'recipe-points-in-polygons-choropleth',
    'recipe-straight-line-catchments',
    'recipe-zonal-distribution'
  ]);
  BINDINGS.forEach(checkBinding);
});

test('choropleth CPU oracle counts, sums and leaves unmatched points out', () => {
  const binding = BINDINGS.find(item => item.id === 'recipe-points-in-polygons-choropleth')!;
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [{geometry: {type: 'Polygon', coordinates: [SQUARE]}, properties: {}}]
  };
  const input = points([
    {coordinates: [1, 1], properties: {v: 2}},
    {coordinates: [5, 5], properties: {v: 4}},
    {coordinates: [5, 6]},
    {coordinates: [50, 50], properties: {v: 100}}
  ]);
  const result = runCatalogOperation(
    binding,
    {statistic: 'mean', valueProperty: 'v', coordinateSystem: 'planar'},
    {points: input, polygons}
  );
  expect(result.tables[0].columns['point_count']).toEqual([3]);
  expect(result.tables[0].columns['value']).toEqual([3]);
  expect(result.summary['unmatchedCount']).toBe(1);
  expect(() =>
    runCatalogOperation(
      binding,
      {statistic: 'sum', valueProperty: '', coordinateSystem: 'planar'},
      {points: input, polygons}
    )
  ).toThrow();
});

test('zonal distribution CPU oracle reports median, mode and quantiles', () => {
  const binding = BINDINGS.find(item => item.id === 'recipe-zonal-distribution')!;
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [{geometry: {type: 'Polygon', coordinates: [SQUARE]}, properties: {}}]
  };
  const input = points(
    [1, 2, 2, 3, 10].map((v, index) => ({coordinates: [1 + index, 5], properties: {v}}))
  );
  const result = runCatalogOperation(
    binding,
    {valueProperty: 'v', probabilities: [0, 0.5, 1], coordinateSystem: 'planar'},
    {points: input, polygons}
  );
  const columns = result.tables[0].columns;
  expect(columns['median']).toEqual([2]);
  expect(columns['mode']).toEqual([2]);
  expect(columns['unique_count']).toEqual([4]);
  expect([columns['quantile_0'][0], columns['quantile_1'][0], columns['quantile_2'][0]]).toEqual([
    1, 2, 10
  ]);
});

test('catchments CPU oracle allocates every cell to the nearest facility', () => {
  const binding = BINDINGS.find(item => item.id === 'recipe-straight-line-catchments')!;
  const facilities = points([{coordinates: [1, 5]}, {coordinates: [19, 5]}]);
  const observations = points([
    {coordinates: [2, 5], properties: {v: 1}},
    {coordinates: [18, 5], properties: {v: 3}},
    {coordinates: [19, 1], properties: {v: 5}}
  ]);
  const result = runCatalogOperation(
    binding,
    {valueProperty: 'v', cellSize: 2, coordinateSystem: 'planar'},
    {facilities, observations}
  );
  const cellCounts = result.tables[0].columns['cell_count'] as number[];
  expect(cellCounts[0] + cellCounts[1]).toBe(
    (result.summary['columns'] as number) * (result.summary['rows'] as number)
  );
  expect(result.tables[0].columns['mean'][0]).toBe(1);
  expect(result.tables[0].columns['mean'][1]).toBe(4);
});

test('lattice hot spot CPU oracle finds the hot cluster and refuses constant lattices', () => {
  const binding = BINDINGS.find(item => item.id === 'recipe-hot-spot-lattice')!;
  const cluster = points(
    Array.from({length: 80}, (_, index) => ({
      coordinates: [20 + (index % 8), 20 + Math.floor(index / 8)]
    }))
  );
  const spread = points(
    Array.from({length: 40}, (_, index) => ({coordinates: [(index * 7) % 50, (index * 13) % 50]}))
  );
  const result = runCatalogOperation(
    binding,
    {
      valueProperty: '',
      cellSize: 5,
      criterion: 'queen',
      radius: 1,
      minimumNeighbors: 3,
      multipleTesting: 'none',
      coordinateSystem: 'planar'
    },
    {points: {type: 'FeatureCollection', features: [...cluster.features, ...spread.features]}}
  );
  const bins = result.tables[0].columns['bin'] as number[];
  expect(Math.max(...bins)).toBeGreaterThan(0);
  expect(() =>
    runCatalogOperation(
      binding,
      {
        valueProperty: '',
        cellSize: 100,
        criterion: 'queen',
        radius: 1,
        minimumNeighbors: 1,
        multipleTesting: 'none',
        coordinateSystem: 'planar'
      },
      {points: points([{coordinates: [0, 0]}, {coordinates: [1, 1]}])}
    )
  ).toThrow();
});
