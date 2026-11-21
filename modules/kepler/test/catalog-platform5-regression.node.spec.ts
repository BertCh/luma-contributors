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

test('GWR CPU centers and equilibrates translated full-rank local systems', () => {
  const binding = BINDINGS.find(item => item.id === 'regression-gwr')!;
  // All weights are 1. Centered X'X = diag(16, 4), X'(y - 1e6) = [1/4, 4].
  // Thus slope=1, intercept=1/64. Both numbers survive the float32 input rounding.
  const data = points(
    Array.from({length: 16}, (_, index) => {
      const predictor = 1e6 + (index < 2 ? -1 : index < 4 ? 1 : 0);
      return {
        coordinates: [0, 0],
        properties: {a: predictor, y: predictor + (index >= 12 ? 1 / 16 : 0)}
      };
    })
  );
  const parameters = {
    responseProperty: 'y',
    predictorProperties: 'a',
    kernel: 'gaussian',
    bandwidth: 1,
    coordinateSystem: 'planar'
  };
  const result = runCatalogOperation(binding, parameters, {points: data});
  for (const column of ['coef_intercept', 'coef_a', 'se_a'])
    expect(result.tables[0].columns[column]).toHaveLength(16);
  for (const value of result.tables[0].columns['coef_intercept'] as number[]) {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(1 / 64, 9);
  }
  for (const value of result.tables[0].columns['coef_a'] as number[])
    expect(value).toBeCloseTo(1, 12);
  // RSS = 12*(-1/64)^2 + 4*(3/64)^2 = 3/256, tr(S)=2; slope variance = RSS/14/4.
  expect(result.summary['residualSumOfSquares']).toBeCloseTo(3 / 256, 12);
  expect(result.summary['effectiveParameters']).toBeCloseTo(2, 12);
  for (const value of result.tables[0].columns['se_a'] as number[]) {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(Math.sqrt(3 / 14336), 12);
  }
  const singular = points(
    Array.from({length: 16}, (_, index) => ({coordinates: [0, 0], properties: {a: 1e6, y: index}}))
  );
  expect(() => runCatalogOperation(binding, parameters, {points: singular})).toThrow(/singular/);
});

test('GWR covariance retains representable distant observation influences before squaring', () => {
  const binding = BINDINGS.find(item => item.id === 'regression-gwr')!;
  const data = points(
    Array.from({length: 8}, (_, index) => {
      const group = Math.floor(index / 4);
      return {
        coordinates: [28 * group, 0],
        properties: {a: group, y: 2 + 3 * group + (index % 2 ? 1 : -1)}
      };
    })
  );
  const result = runCatalogOperation(
    binding,
    {
      responseProperty: 'y',
      predictorProperties: 'a',
      kernel: 'gaussian',
      bandwidth: 1,
      coordinateSystem: 'planar'
    },
    {points: data}
  );
  // Each group mean uses four observations, irrespective of its positive local weight.
  // RSS=8, trace=2, sigma^2=8/6. Intercept variance=sigma^2/4; slope variance=sigma^2/2.
  for (const [column, expected] of [
    ['coef_intercept', 2],
    ['coef_a', 3],
    ['se_intercept', Math.sqrt(1 / 3)],
    ['se_a', Math.sqrt(2 / 3)]
  ] as const) {
    const values = result.tables[0].columns[column] as number[];
    expect(values).toHaveLength(8);
    for (const value of values) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(expected, 10);
    }
  }
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
