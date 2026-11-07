// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogOperationBinding} from '../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM5_RASTER_BINDINGS} from '../src/analysis/catalog-platform5-raster';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

// Oracle policy: the CPU engines are exact float64 stencil and flood-fill implementations over
// the same synthetic grid; the GPU engines run the raster contributors and must match them.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function getBinding(id: string): CatalogOperationBinding {
  return PLATFORM5_RASTER_BINDINGS.find(binding => binding.id === id)!;
}

function makeGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Two dense clumps, one thin bridge-free clump and sparse background noise. */
function clumps(): CatalogFeatureCollection {
  const next = makeGenerator(5);
  const coordinates: number[][] = [];
  for (const [cx, cy, spread, count] of [
    [25, 25, 8, 300],
    [70, 30, 6, 250],
    [50, 80, 5, 150]
  ])
    for (let index = 0; index < count; index++)
      coordinates.push([cx + (next() - 0.5) * 2 * spread, cy + (next() - 0.5) * 2 * spread]);
  for (let index = 0; index < 100; index++) coordinates.push([next() * 100, next() * 100]);
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {v: Math.floor(next() * 8) / 4}
    }))
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

function expectClose(
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

test('raster-density-smoothing matches the CPU stencil for both kernels', async () => {
  const device = await getDevice();
  const binding = getBinding('raster-density-smoothing');
  for (const [kernel, radius, valueProperty] of [
    ['gaussian', 3, ''],
    ['box', 2, 'v']
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      binding,
      {valueProperty, cellSize: 4, kernel, radius, coordinateSystem: 'planar'},
      {points: clumps()}
    );
    expectClose(
      gpu.tables[0].columns['smoothed_value'],
      cpu.tables[0].columns['smoothed_value'],
      1e-4,
      kernel
    );
    expectClose(gpu.tables[0].columns['point_count'], cpu.tables[0].columns['point_count'], 0, 'n');
    const smoothed = cpu.tables[0].columns['smoothed_value'] as number[];
    expect(Math.max(...smoothed)).toBeGreaterThan(0);
  }
});

test('raster-cell-clusters matches the CPU labeling for both connectivities', async () => {
  const device = await getDevice();
  const binding = getBinding('raster-cell-clusters');
  for (const connectivity of ['4', '8']) {
    const {gpu, cpu} = await runBoth(
      device,
      binding,
      {
        valueProperty: 'v',
        cellSize: 4,
        minimumPoints: 3,
        connectivity,
        coordinateSystem: 'planar'
      },
      {points: clumps()}
    );
    for (const column of ['component_id', 'cell_count', 'point_count', 'value_sum'])
      expectClose(gpu.tables[0].columns[column], cpu.tables[0].columns[column], 1e-5, column);
    expect(cpu.summary['componentCount'], connectivity).toBeGreaterThanOrEqual(3);
  }
});
