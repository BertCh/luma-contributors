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

// Oracle policy: the CPU reference engine is the oracle. Neighbor lists are built by the CPU and
// uploaded, so statistics are compared to float32 tolerance and permutation p-values loosely.

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

function makePoints(count: number, seed: number): CatalogFeatureCollection {
  const next = makeGenerator(seed);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, () => {
      const x = next() * 1000;
      const y = next() * 1000;
      return {
        geometry: {type: 'Point', coordinates: [x, y]},
        properties: {v: Math.sin(x / 150) * 5 + Math.cos(y / 220) * 4 + next(), w: next() * 10}
      };
    })
  };
}

function makeGrid(columns: number, rows: number): CatalogFeatureCollection {
  const features: CatalogFeatureCollection['features'] = [];
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++)
      features.push({
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [column * 10, row * 10],
              [column * 10 + 10, row * 10],
              [column * 10 + 10, row * 10 + 10],
              [column * 10, row * 10 + 10],
              [column * 10, row * 10]
            ]
          ]
        },
        properties: {v: Math.sin(column / 2) + Math.cos(row / 3) * 2, w: column + row * 0.5}
      });
  return {type: 'FeatureCollection', features};
}

async function runBoth(
  device: Device,
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const binding = getCatalogBinding(id)!;
  expect(binding.engine).toBe('gpu-or-cpu');
  const base = {coordinateSystem: 'planar'};
  const gpu = await runCatalogOperationAsync(binding, {...base, ...parameters}, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, {...base, ...parameters}, inputs, {
    device,
    engine: 'cpu'
  });
  expect(gpu.engine, `${id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  for (const name of gpu.dispatchedContributors)
    expect(binding.gpuContributorNames).toContain(name);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function expectColumnClose(
  gpu: CatalogRunResult,
  cpu: CatalogRunResult,
  column: string,
  relative: number
): void {
  const actual = gpu.tables[0].columns[column] as (number | null)[];
  const expected = cpu.tables[0].columns[column] as (number | null)[];
  expect(actual.length).toBe(expected.length);
  const scale = Math.max(1e-6, ...expected.map(value => Math.abs(value ?? 0)));
  expected.forEach((value, index) => {
    if (value === null) return expect(actual[index]).toBeNull();
    // Per-value tolerance with a column-scale floor, so near-zero entries do not dominate.
    const tolerance = Math.max(Math.abs(value) * relative, scale * relative * 0.01);
    expect(
      Math.abs((actual[index] as number) - value),
      `${column}[${index}] gpu=${actual[index]} cpu=${value}`
    ).toBeLessThan(tolerance);
  });
}

for (const neighborhood of ['knn', 'queen'] as const) {
  const makeInput = () => (neighborhood === 'queen' ? makeGrid(9, 8) : makePoints(150, 3));
  const settings = {neighborhood, neighborCount: 6, distanceBand: 150, valueProperty: 'v'};

  test(`global-spatial-statistics (${neighborhood}) matches the CPU Moran and Geary`, async () => {
    const device = await getDevice();
    const {gpu, cpu} = await runBoth(device, 'global-spatial-statistics', settings, {
      features: makeInput()
    });
    expect(gpu.dispatchedContributors).toEqual(['GPUGlobalSpatialStatistics']);
    for (const column of ['value', 'expected', 'variance', 'z_score'])
      expectColumnClose(gpu, cpu, column, 2e-3);
    expect(gpu.summary['linkCount']).toBe(cpu.summary['linkCount']);
  });

  test(`hotspot-gi-star (${neighborhood}) matches the CPU z-scores and classes`, async () => {
    const device = await getDevice();
    const {gpu, cpu} = await runBoth(device, 'hotspot-gi-star', settings, {features: makeInput()});
    expect(gpu.dispatchedContributors).toEqual(['GPUHotSpotAnalysis']);
    expectColumnClose(gpu, cpu, 'z_score', 2e-3);
    const gpuClasses = gpu.tables[0].columns['classification'] as string[];
    const cpuClasses = cpu.tables[0].columns['classification'] as string[];
    const z = cpu.tables[0].columns['z_score'] as number[];
    // A class may differ only when the z-score sits within float32 rounding of a threshold.
    gpuClasses.forEach((value, index) => {
      if (value === cpuClasses[index]) return;
      const gap = [1.645, 1.96, 2.576].map(threshold => Math.abs(Math.abs(z[index]) - threshold));
      expect(Math.min(...gap)).toBeLessThan(1e-2);
    });
  });

  test(`local-moran-statistics (${neighborhood}) matches the CPU statistic and quadrants`, async () => {
    const device = await getDevice();
    const {gpu, cpu} = await runBoth(
      device,
      'local-moran-statistics',
      {...settings, permutations: 199, seed: 5, significance: 0.05},
      {features: makeInput()}
    );
    expect(gpu.dispatchedContributors).toEqual(['GPULocalMoran', 'GPULocalPermutationTest']);
    expectColumnClose(gpu, cpu, 'local_i', 2e-3);
    expectColumnClose(gpu, cpu, 'spatial_lag', 2e-3);
    // Pseudo p-values use different generators and tails: the strongest rows must agree on significance.
    const gpuP = gpu.tables[0].columns['p_value'] as (number | null)[];
    const cpuP = cpu.tables[0].columns['p_value'] as (number | null)[];
    gpuP.forEach((value, index) => {
      expect(value === null).toBe(cpuP[index] === null);
      if (value !== null && cpuP[index]! <= 0.005) expect(value).toBeLessThanOrEqual(0.05);
    });
  });

  test(`local-geary-statistics (${neighborhood}) matches the CPU statistic`, async () => {
    const device = await getDevice();
    const {gpu, cpu} = await runBoth(
      device,
      'local-geary-statistics',
      {...settings, permutations: 199, seed: 5},
      {features: makeInput()}
    );
    expect(gpu.dispatchedContributors).toEqual(['GPULocalGeary', 'GPULocalPermutationTest']);
    expectColumnClose(gpu, cpu, 'local_c', 2e-3);
  });
}

test('skater-regions dispatches the minimum spanning tree and cuts, matching the CPU partition', async () => {
  const device = await getDevice();
  const features = makeGrid(10, 8);
  const {gpu, cpu} = await runBoth(
    device,
    'skater-regions',
    {
      properties: 'v,w',
      regionCount: 4,
      minimumRegionSize: 3,
      neighborhood: 'queen',
      neighborCount: 6,
      distanceBand: 150
    },
    {features}
  );
  expect(gpu.dispatchedContributors).toEqual([
    'GPUSpatialWeightsMinimumSpanningTree',
    'GPUSkaterRegions'
  ]);
  expect(gpu.summary['regionCount']).toBe(cpu.summary['regionCount']);
  // Region ids are arbitrary: compare the partitions through a label-to-label map.
  const gpuRegion = gpu.tables[0].columns['region'] as number[];
  const cpuRegion = cpu.tables[0].columns['region'] as number[];
  const mapping = new Map<number, number>();
  let disagreements = 0;
  gpuRegion.forEach((label, index) => {
    if (!mapping.has(label)) mapping.set(label, cpuRegion[index]);
    if (mapping.get(label) !== cpuRegion[index]) disagreements++;
  });
  expect(disagreements).toBe(0);
});
