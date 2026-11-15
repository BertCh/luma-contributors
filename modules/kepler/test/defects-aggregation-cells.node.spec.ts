// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {runArealInterpolation} from '../src/analysis/catalog-platform3-b-zones';

const POINTS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [2, 6, null, 'abc'].map(value => ({
    geometry: {type: 'Point', coordinates: [0.25, 0.25]},
    properties: {value}
  }))
};

test.each(['sum', 'mean'])('cell-compare %s requires valueProperty', statistic => {
  for (const valueProperty of ['', '   '])
    expect(() =>
      runCatalogOperation(
        getCatalogBinding('cell-compare')!,
        {statistic, cellSize: 1, valueProperty},
        {before: POINTS, after: POINTS}
      )
    ).toThrow(/^cell-compare valueProperty.*sum or mean/);
});

test('cell-compare still accepts count without a column and value statistics with a column', () => {
  const binding = getCatalogBinding('cell-compare')!;
  const execute = (statistic: string, valueProperty = '') =>
    runCatalogOperation(
      binding,
      {statistic, cellSize: 1, valueProperty},
      {before: POINTS, after: POINTS}
    );
  expect(execute('count').tables[0].columns['before_value']).toEqual([4]);
  expect(execute('sum', 'value').tables[0].columns['before_value']).toEqual([8]);
  expect(execute('mean', 'value').tables[0].columns['before_value']).toEqual([4]);
});

test('cell-aggregation counts all four points while value statistics use the two valid values', () => {
  const result = runCatalogOperation(
    getCatalogBinding('cell-aggregation')!,
    {cellSize: 1, valueProperty: 'value'},
    {points: POINTS}
  );
  const columns = result.tables[0].columns;
  expect(columns['count']).toEqual([4]);
  expect(columns['sum']).toEqual([8]);
  expect(columns['mean']).toEqual([4]);
  expect(columns['min']).toEqual([2]);
  expect(columns['max']).toEqual([6]);
  expect(result.summary['sampleCount']).toBe(4);
  expect(result.summary['missingValueCount']).toBe(2);
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({code: 'missing-value', severity: 'warning', count: 2})
  );
});

test('cell-aggregation retains cells with no values and counts each MultiPoint position', () => {
  const result = runCatalogOperation(
    getCatalogBinding('cell-aggregation')!,
    {cellSize: 1, valueProperty: 'value'},
    {
      points: {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'MultiPoint',
              coordinates: [
                [1.2, 0.2],
                [1.4, 0.2]
              ]
            }
          },
          {geometry: null, properties: {value: 99}}
        ]
      }
    }
  );
  const columns = result.tables[0].columns;
  expect(columns['count']).toEqual([2]);
  expect(columns['sum']).toEqual([0]);
  for (const statistic of ['mean', 'min', 'max']) expect(columns[statistic]).toEqual([null]);
  expect(result.summary['missingValueCount']).toBe(2);
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({code: 'feature-dropped', count: 1})
  );
});

test('areal-interpolation runner requires an explicit extensive or intensive variableType', () => {
  const empty: CatalogFeatureCollection = {type: 'FeatureCollection', features: []};
  for (const variableType of [undefined, '', 'rate'])
    expect(() =>
      runArealInterpolation(empty, empty, {
        valueProperty: 'rate',
        variableType: variableType as 'extensive' | 'intensive',
        samplesPerSide: 4,
        coordinateSystem: 'planar'
      })
    ).toThrow(/^areal-interpolation variableType.*extensive.*intensive/);
});
