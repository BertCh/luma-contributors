// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  runCatalogOperationAsync,
  type CatalogRunResult
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

// Oracle policy: the CPU reference engines of the family B bindings (engine: 'cpu') are the oracle;
// each GPU-backed binding must run on its contributor and match the oracle to float32 tolerance.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function makeGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function points(rows: {coordinates: number[]; properties?: Record<string, unknown>}[]) {
  return {
    type: 'FeatureCollection',
    features: rows.map(row => ({
      geometry: {type: 'Point', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  } satisfies CatalogFeatureCollection;
}

function scatter(count: number, extent: number, seed: number) {
  const next = makeGenerator(seed);
  return points(
    Array.from({length: count}, () => {
      const x = next() * extent;
      const y = next() * extent;
      return {
        coordinates: [x, y],
        properties: {v: Math.sin(x / 200) * 10 + y / 100, w: 1 + next()}
      };
    })
  );
}

async function runBoth(
  device: Device,
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const binding = getCatalogBinding(id)!;
  const gpu = await runCatalogOperationAsync(
    binding,
    {coordinateSystem: 'planar', ...parameters},
    inputs,
    {device}
  );
  const cpu = await runCatalogOperationAsync(
    binding,
    {coordinateSystem: 'planar', ...parameters},
    inputs,
    {device, engine: 'cpu'}
  );
  expect(gpu.engine, `${id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(binding.gpuContributorNames);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function indexByCell(result: CatalogRunResult, column: string): Map<string, number> {
  const table = result.tables[0];
  const map = new Map<string, number>();
  (table.columns['cell_id'] as string[]).forEach((id, index) =>
    map.set(id, table.columns[column][index] as number)
  );
  return map;
}

function expectFieldsMatch(
  gpu: CatalogRunResult,
  cpu: CatalogRunResult,
  column: string,
  relativeTolerance: number
): void {
  const actual = indexByCell(gpu, column);
  const expected = indexByCell(cpu, column);
  const peak = Math.max(...[...expected.values()].map(Math.abs), 1e-12);
  for (const [cell, value] of expected) {
    expect(actual.has(cell), `cell ${cell}`).toBe(true);
    expect(Math.abs((actual.get(cell) as number) - value)).toBeLessThanOrEqual(
      relativeTolerance * peak
    );
  }
  expect(actual.size).toBe(expected.size);
}

test('idw-grid runs on GPUInverseDistanceWeighting and matches the CPU oracle', async () => {
  const device = await getDevice();
  const samples = scatter(60, 1000, 11);
  for (const neighborCount of [0, 8]) {
    const {gpu, cpu} = await runBoth(
      device,
      'idw-grid',
      {valueProperty: 'v', cellSize: 100, searchRadius: 300, neighborCount},
      {points: samples}
    );
    expect(cpu.tables[0].rowCount).toBeGreaterThan(10);
    expectFieldsMatch(gpu, cpu, 'value', 2e-3);
    expectFieldsMatch(gpu, cpu, 'sample_count', 1e-9);
  }
});

test('kernel-density-grid runs on GPUKernelDensity and matches the CPU oracle', async () => {
  const device = await getDevice();
  const samples = scatter(80, 1000, 23);
  for (const kernel of ['gaussian', 'quartic']) {
    const {gpu, cpu} = await runBoth(
      device,
      'kernel-density-grid',
      {cellSize: 50, bandwidth: 120, kernel, weightProperty: 'w', padding: 100},
      {points: samples}
    );
    expect(cpu.tables[0].rowCount).toBeGreaterThan(50);
    expectFieldsMatch(gpu, cpu, 'density', 2e-3);
  }
});

test('point-density-grid runs on GPUPointDensity and matches the CPU oracle', async () => {
  const device = await getDevice();
  const next = makeGenerator(5);
  // Points sit inside cells (never on a boundary) so float32 and float64 binning agree.
  const samples = points(
    Array.from({length: 120}, () => ({
      coordinates: [
        (Math.floor(next() * 8) + 0.2 + next() * 0.6) * 25,
        (Math.floor(next() * 8) + 0.2 + next() * 0.6) * 25
      ],
      properties: {w: 1 + Math.floor(next() * 5)}
    }))
  );
  for (const statistic of ['count', 'sum', 'mean']) {
    const {gpu, cpu} = await runBoth(
      device,
      'point-density-grid',
      {cellSize: 25, statistic, weightProperty: 'w'},
      {points: samples}
    );
    expectFieldsMatch(gpu, cpu, 'value', 1e-4);
    expectFieldsMatch(gpu, cpu, 'sample_count', 1e-9);
  }
});

test('density-clustering runs on GPUDBSCAN with identical canonical labels', async () => {
  const device = await getDevice();
  const next = makeGenerator(41);
  const blob = (centerX: number, centerY: number, count: number) =>
    Array.from({length: count}, () => ({
      coordinates: [centerX + (next() - 0.5) * 30, centerY + (next() - 0.5) * 30]
    }));
  const samples = points([
    ...blob(100, 100, 40),
    ...blob(600, 150, 40),
    ...blob(300, 700, 30),
    {coordinates: [900, 900]},
    {coordinates: [50, 900]}
  ]);
  const {gpu, cpu} = await runBoth(
    device,
    'density-clustering',
    {epsilon: 17.3, minimumSamples: 4},
    {points: samples}
  );
  expect(gpu.tables[0].columns['cluster']).toEqual(cpu.tables[0].columns['cluster']);
  expect(gpu.tables[0].columns['is_core']).toEqual(cpu.tables[0].columns['is_core']);
  expect(Number(cpu.summary['clusterCount'])).toBeGreaterThanOrEqual(3);
  expect(gpu.summary['noiseCount']).toBe(cpu.summary['noiseCount']);
});
