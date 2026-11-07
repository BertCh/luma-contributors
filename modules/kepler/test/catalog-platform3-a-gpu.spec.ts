// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

// Oracle policy: the CPU reference engine (engine: 'cpu') is the oracle for every GPU-backed
// family A binding. Only spatial-hilbert-order has a GPU contributor today.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function makeScatter(count: number): CatalogFeatureCollection {
  let state = 12345;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, (_, index) => ({
      id: `p${index}`,
      geometry: {type: 'Point', coordinates: [-122.5 + next() * 0.5, 37.5 + next() * 0.5]},
      properties: {}
    }))
  };
}

test('spatial-hilbert-order runs on GPUHilbertKeys and matches the CPU oracle', async () => {
  const device = await getDevice();
  const binding = getCatalogBinding('spatial-hilbert-order');
  expect(binding?.engine).toBe('gpu-or-cpu');
  const features = makeScatter(500);
  const withEmpty: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [...features.features, {id: 'empty', geometry: null, properties: {}}]
  };
  for (const order of [4, 10, 16]) {
    const parameters = {order};
    const gpu = await runCatalogOperationAsync(
      binding!,
      parameters,
      {features: withEmpty},
      {device}
    );
    const cpu = await runCatalogOperationAsync(
      binding!,
      parameters,
      {features: withEmpty},
      {
        device,
        engine: 'cpu'
      }
    );
    expect(gpu.engine, `order ${order}: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
    expect(gpu.dispatchedContributors).toEqual(['GPUHilbertKeys']);
    expect(cpu.engine).toBe('cpu-reference');
    const gpuKeys = gpu.tables[0].columns['hilbert_key'] as number[];
    const cpuKeys = cpu.tables[0].columns['hilbert_key'] as number[];
    const gpuRows = gpu.tables[0].columns['source_index'] as number[];
    // Keys are ascending along the GPU rank, and every feature appears once.
    expect(gpuKeys.every((key, rank) => rank === 0 || key >= gpuKeys[rank - 1])).toBe(true);
    expect(new Set(gpuRows).size).toBe(501);
    // The GPU quantizes in float32 with its own rounding, so a feature within a rounding of a cell
    // border may land one cell away from the CPU emulation; the orders agree except for those.
    const cpuByRow = new Map<number, number>();
    (cpu.tables[0].columns['source_index'] as number[]).forEach((row, rank) =>
      cpuByRow.set(row, cpuKeys[rank])
    );
    const mismatches = gpuRows.filter((row, rank) => cpuByRow.get(row) !== gpuKeys[rank]).length;
    console.info(`hilbert order ${order}: ${mismatches} of 501 keys differ from the CPU oracle`);
    expect(mismatches).toBeLessThanOrEqual(order <= 10 ? 0 : 25);
    expect(gpu.tables[0].columns['source_index'][500]).toBe(500);
  }
});

test('spatial-hilbert-order falls back to the CPU engine when no feature has geometry', async () => {
  const device = await getDevice();
  const binding = getCatalogBinding('spatial-hilbert-order')!;
  const empty: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [{id: 'empty', geometry: null, properties: {}}]
  };
  const result = await runCatalogOperationAsync(binding, {}, {features: empty}, {device});
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toBeDefined();
});
