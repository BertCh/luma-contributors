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
import {PLATFORM5_REGRESSION_BINDINGS as BINDINGS} from '../src/analysis/catalog-platform5-regression';

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

test('regression bindings are unique, declared and runnable on the CPU engine', () => {
  expect(BINDINGS.map(binding => binding.id).sort()).toEqual([
    'regression-gwr',
    'regression-probit',
    'regression-sur'
  ]);
  BINDINGS.forEach(checkBinding);
});

function line(count: number, makeProperties: (x: number) => Record<string, number>) {
  return points(
    Array.from({length: count}, (_, index) => ({
      coordinates: [index, (index * 7) % 5],
      properties: makeProperties(index)
    }))
  );
}

test('GWR CPU oracle recovers a global line exactly with a wide kernel', () => {
  const binding = BINDINGS.find(item => item.id === 'regression-gwr')!;
  const data = line(30, x => ({a: Math.sin(x), y: 3 + 2 * Math.sin(x)}));
  const result = runCatalogOperation(
    binding,
    {
      responseProperty: 'y',
      predictorProperties: 'a',
      kernel: 'gaussian',
      bandwidth: 1000,
      coordinateSystem: 'planar'
    },
    {points: data}
  );
  for (const value of result.tables[0].columns['coef_intercept'] as number[])
    expect(value).toBeCloseTo(3, 6);
  for (const value of result.tables[0].columns['coef_a'] as number[])
    expect(value).toBeCloseTo(2, 6);
});

test('probit CPU oracle refuses non binary responses and separation', () => {
  const binding = BINDINGS.find(item => item.id === 'regression-probit')!;
  const parameters = {responseProperty: 'y', predictorProperties: 'a', coordinateSystem: 'planar'};
  expect(() =>
    runCatalogOperation(binding, parameters, {points: line(20, x => ({a: x, y: 2}))})
  ).toThrow();
  expect(() =>
    runCatalogOperation(binding, parameters, {points: line(20, x => ({a: x, y: x > 10 ? 1 : 0}))})
  ).toThrow();
  const overlapping = line(40, x => ({
    a: Math.sin(x * 2.3) * 3,
    y: Math.sin(x * 2.3) * 3 + Math.cos(x * 5.1) > 0 ? 1 : 0
  }));
  const result = runCatalogOperation(binding, parameters, {points: overlapping});
  expect((result.tables[0].columns['estimate'] as number[])[1]).toBeGreaterThan(0);
});

test('SUR CPU oracle equals equation OLS when the equations share one design', () => {
  const binding = BINDINGS.find(item => item.id === 'regression-sur')!;
  const data = line(30, x => ({
    a: Math.sin(x),
    y1: 1 + 2 * Math.sin(x) + Math.cos(3 * x) / 10,
    y2: -1 + Math.sin(x) - Math.cos(5 * x) / 10
  }));
  const result = runCatalogOperation(
    binding,
    {responseProperties: 'y1, y2', predictorProperties: 'a; a', coordinateSystem: 'planar'},
    {points: data}
  );
  const estimates = result.tables[0].columns['estimate'] as number[];
  expect(estimates[1]).toBeCloseTo(2, 1);
  expect(estimates[2]).toBeCloseTo(-1, 1);
  expect(estimates[3]).toBeCloseTo(1, 1);
  expect(() =>
    runCatalogOperation(
      binding,
      {responseProperties: 'y1', predictorProperties: 'a', coordinateSystem: 'planar'},
      {points: data}
    )
  ).toThrow();
});
