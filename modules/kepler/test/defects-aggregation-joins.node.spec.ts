// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {CatalogGpuContext} from '../src/analysis/catalog-binding-types';
import {
  getCatalogBinding,
  runCatalogOperation,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

function square(minimum: number, maximum: number): number[][] {
  return [
    [minimum, minimum],
    [maximum, minimum],
    [maximum, maximum],
    [minimum, maximum],
    [minimum, minimum]
  ];
}

const POINTS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {geometry: {type: 'Point', coordinates: [2, 2]}, properties: {value: 2}},
    {geometry: {type: 'Point', coordinates: [2.5, 2.5]}, properties: {value: 4}},
    {geometry: {type: 'Point', coordinates: [20, 20]}, properties: {value: 100}}
  ]
};
const POLYGONS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {geometry: {type: 'Polygon', coordinates: [square(0, 4)]}},
    {geometry: {type: 'Polygon', coordinates: [square(1, 3)]}}
  ]
};

for (const identifier of [
  'recipe-points-in-polygons-choropleth',
  'raster-join-points-in-polygons'
]) {
  const binding = getCatalogBinding(identifier)!;
  const countColumn = identifier.startsWith('recipe') ? 'point_count' : 'count';
  const parameters = {coordinateSystem: 'planar', valueProperty: 'value'};

  test(`${identifier} counts overlapping points in every containing feature`, () => {
    const result = runCatalogOperation(binding, parameters, {points: POINTS, polygons: POLYGONS});
    // The confirmed defect returned [2, 0]. There are two matches per input point here.
    expect(result.tables[0].columns[countColumn]).toEqual([2, 2]);
    expect(result.summary['multipleMatchCount']).toBe(2);
    expect(result.summary['pointCount']).toBe(3);
    expect(
      result.summary[identifier.startsWith('recipe') ? 'unmatchedCount' : 'outsideCount']
    ).toBe(1);
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({code: 'points-matched-multiple-polygons', count: 2})
    );
    const reversed = runCatalogOperation(binding, parameters, {
      points: POINTS,
      polygons: {...POLYGONS, features: [...POLYGONS.features].reverse()}
    });
    expect(reversed.tables[0].columns[countColumn]).toEqual([2, 2]);
  });

  test(`${identifier} counts one feature once when its multipart interiors overlap`, () => {
    const result = runCatalogOperation(binding, parameters, {
      points: POINTS,
      polygons: {
        type: 'FeatureCollection',
        features: [
          {geometry: {type: 'MultiPolygon', coordinates: [[square(0, 4)], [square(1, 3)]]}}
        ]
      }
    });
    expect(result.tables[0].columns[countColumn]).toEqual([2]);
    expect(result.summary['multipleMatchCount']).toBe(0);
  });

  test(`${identifier} visibly falls back before accessing a GPU for multiple matches`, async () => {
    const device = new Proxy({} as CatalogGpuContext['device'], {
      get() {
        throw new Error('A GPU must not be accessed for this input');
      }
    });
    const result = await runCatalogOperationAsync(
      binding,
      parameters,
      {points: POINTS, polygons: POLYGONS},
      {device}
    );
    expect(result.engine).toBe('cpu-reference');
    expect(result.fallbackReason).toMatch(new RegExp(`^${identifier} polygons.*multiple matches`));
    expect(result.diagnostics).toContainEqual(expect.objectContaining({code: 'gpu-fallback'}));
    expect(result.tables[0].columns[countColumn]).toEqual([2, 2]);
    expect(result.summary['multipleMatchCount']).toBe(2);
  });
}

test.each([
  ['count', 2],
  ['sum', 6],
  ['mean', 3],
  ['minimum', 2],
  ['maximum', 4]
])('choropleth %s includes every polygon match', (statistic, expected) => {
  const result = runCatalogOperation(
    getCatalogBinding('recipe-points-in-polygons-choropleth')!,
    {coordinateSystem: 'planar', valueProperty: 'value', statistic},
    {points: POINTS, polygons: POLYGONS}
  );
  expect(result.tables[0].columns['value']).toEqual([expected, expected]);
});

test('raster join sums and means include every polygon match and document GPU-only cellSize', () => {
  const binding = getCatalogBinding('raster-join-points-in-polygons')!;
  const execute = (cellSize: number) =>
    runCatalogOperation(
      binding,
      {coordinateSystem: 'planar', valueProperty: 'value', cellSize},
      {points: POINTS, polygons: POLYGONS}
    );
  expect(execute(1).tables[0].columns['sum']).toEqual([6, 6]);
  expect(execute(1).tables[0].columns['mean']).toEqual([3, 3]);
  expect(execute(1).tables).toEqual(execute(1000).tables);
  expect(binding.parameters.find(parameter => parameter.name === 'cellSize')!.label).toMatch(
    /GPU only/
  );
  expect(binding.declaredBehavior.join(' ')).toMatch(/CPU.*exact.*ignores cellSize/);
});

test('raster GPU also falls back when only the cell center has multiple polygon matches', async () => {
  const result = await runCatalogOperationAsync(
    getCatalogBinding('raster-join-points-in-polygons')!,
    {coordinateSystem: 'planar', cellSize: 2},
    {
      points: {
        type: 'FeatureCollection',
        features: [{geometry: {type: 'Point', coordinates: [0.5, 0.5]}}]
      },
      polygons: {
        type: 'FeatureCollection',
        features: [
          {geometry: {type: 'Polygon', coordinates: [square(0, 4)]}},
          {geometry: {type: 'Polygon', coordinates: [square(0.75, 3)]}}
        ]
      }
    },
    {device: {} as CatalogGpuContext['device']}
  );
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toMatch(
    /^raster-join-points-in-polygons polygons.*multiple matches/
  );
  expect(result.tables[0].columns['count']).toEqual([1, 0]);
  expect(result.summary['multipleMatchCount']).toBe(0);
});
