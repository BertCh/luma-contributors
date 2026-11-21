// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogOperationBinding} from '../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM5_RECIPE_BINDINGS} from '../src/analysis/catalog-platform5-recipes';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

// Oracle policy: each recipe binding's CPU engine composes the same join and statistics on the CPU;
// the GPU engine runs the recipe wiring and must match it (float32 tolerance for sums).

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function getBinding(id: string): CatalogOperationBinding {
  return PLATFORM5_RECIPE_BINDINGS.find(binding => binding.id === id)!;
}

function makeGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function scatter(count: number, seed: number, extent = 100): CatalogFeatureCollection {
  const next = makeGenerator(seed);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, () => ({
      geometry: {type: 'Point', coordinates: [next() * extent, next() * extent]},
      // Quarter integers keep float32 sums exact; some points carry no value.
      properties: next() < 0.1 ? {} : {v: Math.floor(next() * 40) / 4}
    }))
  };
}

function squares(): CatalogFeatureCollection {
  const square = (x: number, y: number, size: number) => [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size],
    [x, y]
  ];
  return {
    type: 'FeatureCollection',
    features: [
      {geometry: {type: 'Polygon', coordinates: [square(0, 0, 40)]}, properties: {}},
      // A shell with a hole.
      {
        geometry: {
          type: 'Polygon',
          coordinates: [square(45, 5, 50), square(60, 20, 20).reverse()]
        },
        properties: {}
      },
      {geometry: {type: 'Polygon', coordinates: [square(2, 52, 40)]}, properties: {}},
      // Far from every point.
      {geometry: {type: 'Polygon', coordinates: [square(500, 500, 10)]}, properties: {}}
    ]
  };
}

async function runBoth(
  device: Device,
  binding: CatalogOperationBinding,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
) {
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {device, engine: 'cpu'});
  expect(gpu.engine, `${binding.id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(binding.gpuContributorNames);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function expectColumn(
  actual: readonly (string | number | boolean | null)[],
  expected: readonly (string | number | boolean | null)[],
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  expected.forEach((want, index) => {
    const got = actual[index];
    if (typeof want === 'number' && typeof got === 'number')
      expect(Math.abs(got - want), `${label}[${index}]`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(want))
      );
    else expect(got, `${label}[${index}]`).toEqual(want);
  });
}

test('recipe-points-in-polygons-choropleth matches the CPU join for every statistic', async () => {
  const device = await getDevice();
  const binding = getBinding('recipe-points-in-polygons-choropleth');
  const inputs = {points: scatter(400, 7), polygons: squares()};
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum']) {
    const {gpu, cpu} = await runBoth(
      device,
      binding,
      {statistic, valueProperty: 'v', coordinateSystem: 'planar'},
      inputs
    );
    const gpuTable = gpu.tables[0];
    const cpuTable = cpu.tables[0];
    expectColumn(gpuTable.columns['point_count'], cpuTable.columns['point_count'], 0, statistic);
    expectColumn(gpuTable.columns['value'], cpuTable.columns['value'], 1e-5, statistic);
    expect(gpu.summary['unmatchedCount']).toBe(cpu.summary['unmatchedCount']);
  }
  const {cpu} = await runBoth(
    device,
    binding,
    {statistic: 'count', valueProperty: '', coordinateSystem: 'planar'},
    inputs
  );
  const counts = cpu.tables[0].columns['point_count'] as number[];
  expect(counts[0]).toBeGreaterThan(0);
  expect(counts[3]).toBe(0);
});

test('recipe-zonal-distribution matches the CPU order statistics', async () => {
  const device = await getDevice();
  const binding = getBinding('recipe-zonal-distribution');
  const {gpu, cpu} = await runBoth(
    device,
    binding,
    {
      valueProperty: 'v',
      probabilities: [0.1, 0.5, 0.9],
      coordinateSystem: 'planar'
    },
    {points: scatter(600, 11), polygons: squares()}
  );
  for (const column of [
    'row_count',
    'valid_count',
    'unique_count',
    'median',
    'mode',
    'quantile_0',
    'quantile_1',
    'quantile_2'
  ])
    expectColumn(gpu.tables[0].columns[column], cpu.tables[0].columns[column], 1e-5, column);
  expect((cpu.tables[0].columns['row_count'] as number[])[0]).toBeGreaterThan(10);
  expect((cpu.tables[0].columns['median'] as unknown[])[3]).toBeNull();
});

test('SEAM-108: recipe-zonal-distribution refuses GPU statistics after assignment loss', async () => {
  const device = await getDevice();
  const binding = getBinding('recipe-zonal-distribution');
  // 100,000 points in 41 identical squares: 4,100,000 bounding-box pairs exceed the adapter's
  // 4,000,000 candidate cap, so the GPU zone assignment must report overflow.
  const pointCount = 100000;
  const next = makeGenerator(108);
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: pointCount}, (_, index) => ({
      geometry: {type: 'Point', coordinates: [1 + next() * 8, 1 + next() * 8]},
      properties: {v: index % 7}
    }))
  };
  const square = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
    [0, 0]
  ];
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 41}, () => ({
      geometry: {type: 'Polygon', coordinates: [square]},
      properties: {}
    }))
  };
  const result = await runCatalogOperationAsync(
    binding,
    {valueProperty: 'v', probabilities: [0.5], coordinateSystem: 'planar'},
    {points, polygons},
    {device}
  );
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toMatch(/zone-assignment\.(overflow|candidateOverflow)/);
  expect(result.diagnostics.some(item => item.code === 'gpu-fallback')).toBe(true);
  // Closed form: every point belongs to the first (smallest-row) square. Values index % 7 over
  // 100,000 rows give 14,286 each of 0..4 and 14,285 each of 5 and 6, so the linear median at
  // sorted positions 49,999 and 50,000 is 3 and the smallest most frequent value is 0.
  const table = result.tables[0];
  const rowCounts = table.columns['row_count'] as number[];
  expect(rowCounts[0]).toBe(pointCount);
  expect(rowCounts.slice(1).every(count => count === 0)).toBe(true);
  expect(table.columns['valid_count'][0]).toBe(pointCount);
  expect(table.columns['median'][0]).toBe(3);
  expect(table.columns['quantile_0'][0]).toBe(3);
  expect(table.columns['mode'][0]).toBe(0);
  expect(table.columns['unique_count'][0]).toBe(7);
});

test('recipe-straight-line-catchments matches the CPU allocation and statistics', async () => {
  const device = await getDevice();
  const binding = getBinding('recipe-straight-line-catchments');
  const facilities: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      [12.3, 20.7],
      [80.1, 15.2],
      [45.5, 83.9],
      [10.2, 70.4]
    ].map(coordinates => ({geometry: {type: 'Point', coordinates}, properties: {}}))
  };
  const {gpu, cpu} = await runBoth(
    device,
    binding,
    {valueProperty: 'v', cellSize: 2, coordinateSystem: 'planar'},
    {facilities, observations: scatter(500, 5)}
  );
  for (const column of ['cell_count', 'value_count', 'sum', 'mean', 'minimum', 'maximum'])
    expectColumn(gpu.tables[0].columns[column], cpu.tables[0].columns[column], 1e-4, column);
  const cellCounts = cpu.tables[0].columns['cell_count'] as number[];
  expect(cellCounts.reduce((sum, count) => sum + count, 0)).toBe(
    (cpu.summary['columns'] as number) * (cpu.summary['rows'] as number)
  );
});

function clustered(): CatalogFeatureCollection {
  const next = makeGenerator(21);
  const coordinates: number[][] = [];
  // A dense hot cluster, a sparse cold area and a uniform background.
  for (let index = 0; index < 500; index++) coordinates.push([20 + next() * 20, 20 + next() * 20]);
  for (let index = 0; index < 400; index++) coordinates.push([next() * 100, next() * 100]);
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {}
    }))
  };
}

test('recipe-hot-spot-lattice matches the CPU Gi* on the lattice', async () => {
  const device = await getDevice();
  const binding = getBinding('recipe-hot-spot-lattice');
  for (const [criterion, radius, multipleTesting] of [
    ['queen', 1, 'none'],
    ['rook', 2, 'none'],
    ['queen', 1, 'benjamini-hochberg']
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      binding,
      {
        valueProperty: '',
        cellSize: 5,
        criterion,
        radius,
        minimumNeighbors: 3,
        multipleTesting,
        coordinateSystem: 'planar'
      },
      {points: clustered()}
    );
    const label = `${criterion}-${radius}-${multipleTesting}`;
    for (const column of ['neighbor_count', 'z_score', 'p_value'])
      expectColumn(gpu.tables[0].columns[column], cpu.tables[0].columns[column], 1e-3, label);
    expectColumn(gpu.tables[0].columns['bin'], cpu.tables[0].columns['bin'], 0, label);
    const bins = cpu.tables[0].columns['bin'] as number[];
    expect(
      bins.some(bin => bin > 0),
      label
    ).toBe(true);
  }
});
