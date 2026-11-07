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
import {PLATFORM5_RASTER_BINDINGS as BINDINGS} from '../src/analysis/catalog-platform5-raster';

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

test('raster bindings are unique, declared and runnable on the CPU engine', () => {
  expect(BINDINGS.map(binding => binding.id).sort()).toEqual([
    'raster-cell-clusters',
    'raster-density-smoothing'
  ]);
  BINDINGS.forEach(checkBinding);
});

test('smoothing CPU oracle spreads a single point over a normalized stencil', () => {
  const binding = BINDINGS.find(item => item.id === 'raster-density-smoothing')!;
  const result = runCatalogOperation(
    binding,
    {valueProperty: '', cellSize: 1, kernel: 'box', radius: 1, coordinateSystem: 'planar'},
    {points: points([{coordinates: [0.5, 0.5]}, {coordinates: [5.5, 5.5]}])}
  );
  const smoothed = result.tables[0].columns['smoothed_value'] as number[];
  // Both points sit in grid corners, so each keeps four of its nine cells of one ninth.
  expect(smoothed.filter(value => value > 0).length).toBe(8);
  expect(smoothed.reduce((sum, value) => sum + value, 0)).toBeCloseTo(8 / 9, 5);
  expect(Math.max(...smoothed)).toBeCloseTo(1 / 9, 5);
});

test('cluster CPU oracle separates distant groups and merges diagonal ones with eight-connectivity', () => {
  const binding = BINDINGS.find(item => item.id === 'raster-cell-clusters')!;
  const input = points([
    {coordinates: [0.5, 0.5]},
    {coordinates: [1.5, 1.5]},
    {coordinates: [8.5, 8.5]}
  ]);
  const parameters = {valueProperty: '', cellSize: 1, minimumPoints: 1, coordinateSystem: 'planar'};
  expect(
    runCatalogOperation(binding, {...parameters, connectivity: '4'}, {points: input}).summary[
      'componentCount'
    ]
  ).toBe(3);
  expect(
    runCatalogOperation(binding, {...parameters, connectivity: '8'}, {points: input}).summary[
      'componentCount'
    ]
  ).toBe(2);
});
