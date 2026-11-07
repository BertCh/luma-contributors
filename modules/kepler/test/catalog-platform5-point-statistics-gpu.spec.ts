// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

// The CPU reference is the oracle: LOF scores must match to float32 tolerance, Knox counts exactly.

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

test('local-outlier-factor dispatches GPULocalOutlierFactor and matches the CPU scores', async () => {
  const device = await getDevice();
  const next = makeGenerator(21);
  const features: CatalogFeatureCollection['features'] = Array.from({length: 180}, () => ({
    geometry: {type: 'Point', coordinates: [next() * 100, next() * 100]},
    properties: {}
  }));
  // A few far outliers.
  for (const offset of [400, 520, 650])
    features.push({geometry: {type: 'Point', coordinates: [offset, offset * 0.7]}, properties: {}});
  const points: CatalogFeatureCollection = {type: 'FeatureCollection', features};
  const binding = getCatalogBinding('local-outlier-factor')!;
  const parameters = {neighborCount: 10, threshold: 1.5, coordinateSystem: 'planar'};
  const gpu = await runCatalogOperationAsync(binding, parameters, {points}, {device});
  const cpu = await runCatalogOperationAsync(
    binding,
    parameters,
    {points},
    {device, engine: 'cpu'}
  );
  expect(gpu.engine, gpu.fallbackReason).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(['GPULocalOutlierFactor']);
  const gpuLof = gpu.tables[0].columns['lof'] as number[];
  (cpu.tables[0].columns['lof'] as number[]).forEach((value, index) =>
    expect(Math.abs(gpuLof[index] - value), `lof ${index}`).toBeLessThan(
      Math.max(2e-3, value * 2e-3)
    )
  );
  expect(gpu.tables[0].columns['is_outlier']).toEqual(cpu.tables[0].columns['is_outlier']);
  expect(gpu.summary['outlierCount']).toBe(cpu.summary['outlierCount']);
  expect(cpu.summary['outlierCount'] as number).toBeGreaterThanOrEqual(3);
});

test('knox-test dispatches GPUKnoxTest and matches the CPU pair counts', async () => {
  const device = await getDevice();
  const next = makeGenerator(33);
  const features: CatalogFeatureCollection['features'] = [];
  // Space-time clusters: events near a center happen at nearly the same time.
  for (let cluster = 0; cluster < 6; cluster++) {
    const cx = next() * 1000;
    const cy = next() * 1000;
    const t0 = next() * 100;
    for (let event = 0; event < 12; event++)
      features.push({
        geometry: {type: 'Point', coordinates: [cx + next() * 20, cy + next() * 20]},
        properties: {t: t0 + next() * 0.5}
      });
  }
  const points: CatalogFeatureCollection = {type: 'FeatureCollection', features};
  const binding = getCatalogBinding('knox-test')!;
  const parameters = {
    timeProperty: 't',
    spatialThreshold: 40,
    temporalThreshold: 1,
    permutations: 499,
    seed: 3,
    coordinateSystem: 'planar'
  };
  const gpu = await runCatalogOperationAsync(binding, parameters, {points}, {device});
  const cpu = await runCatalogOperationAsync(
    binding,
    parameters,
    {points},
    {device, engine: 'cpu'}
  );
  expect(gpu.engine, gpu.fallbackReason).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(['GPUKnoxTest']);
  for (const column of [
    'event_count',
    'spatially_close_pairs',
    'temporally_close_pairs',
    'joint_close_pairs',
    'expected_joint_pairs',
    'ratio'
  ])
    expect(gpu.tables[0].columns[column], column).toEqual(cpu.tables[0].columns[column]);
  // Different permutation generators: both Monte Carlo p-values must call this strong interaction.
  expect(cpu.tables[0].columns['p_value'][0] as number).toBeLessThan(0.01);
  expect(gpu.tables[0].columns['p_value'][0] as number).toBeLessThan(0.01);
});
