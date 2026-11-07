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

// Oracle policy: the CPU reference engine (engine: 'cpu') is the oracle for every PLATFORM4 GPU
// statistics binding; each must run on its contributor and match the oracle to float32 tolerance.

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

async function runBoth(
  device: Device,
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const binding = getCatalogBinding(id)!;
  expect(binding.engine).toBe('gpu-or-cpu');
  const base = binding.parameters.some(spec => spec.name === 'coordinateSystem')
    ? {coordinateSystem: 'planar'}
    : {};
  const gpu = await runCatalogOperationAsync(binding, {...base, ...parameters}, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, {...base, ...parameters}, inputs, {
    device,
    engine: 'cpu'
  });
  expect(gpu.engine, `${id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(binding.gpuContributorNames);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

test('line-density-grid dispatches GPULineDensity and matches the CPU lengths', async () => {
  const device = await getDevice();
  const next = makeGenerator(7);
  const lines: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 60}, (_, index) => ({
      id: `line${index}`,
      geometry:
        index % 5 === 0
          ? {
              type: 'MultiLineString',
              coordinates: [
                [
                  [next() * 1000, next() * 1000],
                  [next() * 1000, next() * 1000]
                ],
                [
                  [next() * 1000, next() * 1000],
                  [next() * 1000, next() * 1000],
                  [next() * 1000, next() * 1000]
                ]
              ]
            }
          : {
              type: 'LineString',
              coordinates: [
                [next() * 1000, next() * 1000],
                [next() * 1000, next() * 1000]
              ]
            },
      properties: {}
    }))
  };
  const {gpu, cpu} = await runBoth(
    device,
    'line-density-grid',
    {cellSize: 125, includeEmptyCells: true},
    {lines}
  );
  const gpuDensity = gpu.tables[0].columns['density'] as number[];
  const cpuDensity = cpu.tables[0].columns['density'] as number[];
  expect(gpuDensity.length).toBe(cpuDensity.length);
  const scale = Math.max(...cpuDensity);
  gpuDensity.forEach((value, index) =>
    expect(Math.abs(value - cpuDensity[index]), `cell ${index}`).toBeLessThan(scale * 1e-4)
  );
  expect(gpu.summary['outputCellCount']).toBe(cpu.summary['outputCellCount']);
});

test('kriging-grid dispatches GPUKriging and matches the CPU local kriging', async () => {
  const device = await getDevice();
  const next = makeGenerator(11);
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 150}, () => {
      const x = next() * 1000;
      const y = next() * 1000;
      return {
        geometry: {type: 'Point', coordinates: [x, y]},
        properties: {v: Math.sin(x / 200) * 10 + y / 100}
      };
    })
  };
  const {gpu, cpu} = await runBoth(
    device,
    'kriging-grid',
    {
      valueProperty: 'v',
      cellSize: 100,
      model: 'exponential',
      nugget: 0.05,
      sill: 20,
      range: 400,
      neighborCount: 8,
      searchRadius: 0,
      includeEmptyCells: true
    },
    {points}
  );
  const gpuValues = gpu.tables[0].columns['value'] as (number | null)[];
  const cpuValues = cpu.tables[0].columns['value'] as (number | null)[];
  expect(gpuValues.length).toBe(cpuValues.length);
  let compared = 0;
  cpuValues.forEach((expected, index) => {
    const actual = gpuValues[index];
    if (expected === null || actual === null) return;
    compared++;
    expect(Math.abs(actual - expected), `cell ${index}`).toBeLessThan(0.05);
  });
  expect(compared).toBeGreaterThan(cpuValues.length * 0.9);
});

test('multiple-testing-adjustment dispatches GPUMultipleTestingAdjustment and matches the CPU', async () => {
  const device = await getDevice();
  const next = makeGenerator(3);
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 200}, (_, index) => ({
      geometry: {type: 'Point', coordinates: [index, 0]},
      properties: {p: index % 17 === 0 ? 'not a number' : next() ** 3}
    }))
  };
  for (const method of ['bonferroni', 'holm', 'benjamini-hochberg']) {
    const {gpu, cpu} = await runBoth(
      device,
      'multiple-testing-adjustment',
      {pValueProperty: 'p', method, alpha: 0.2},
      {features}
    );
    const gpuAdjusted = gpu.tables[0].columns['adjusted_p_value'] as number[];
    const cpuAdjusted = cpu.tables[0].columns['adjusted_p_value'] as number[];
    expect(gpuAdjusted.length).toBe(cpuAdjusted.length);
    gpuAdjusted.forEach((value, row) =>
      expect(Math.abs(value - cpuAdjusted[row]), `${method} row ${row}`).toBeLessThan(1e-5)
    );
    expect(gpu.summary['testCount']).toBe(cpu.summary['testCount']);
  }
  // Sidak has no GPU method: the run falls back to the CPU engine and says why.
  const binding = getCatalogBinding('multiple-testing-adjustment')!;
  const sidak = await runCatalogOperationAsync(
    binding,
    {pValueProperty: 'p', method: 'sidak'},
    {features},
    {device}
  );
  expect(sidak.engine).toBe('cpu-reference');
  expect(sidak.fallbackReason).toContain('Sidak');
});

test('empirical-bayes-rates dispatches GPUEmpiricalBayesRates and matches the CPU', async () => {
  const device = await getDevice();
  const next = makeGenerator(21);
  const areas: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 120}, (_, index) => {
      const population = 50 + Math.floor(next() * 5000);
      return {
        geometry: {type: 'Point', coordinates: [index, 0]},
        properties: {events: Math.floor(population * (0.02 + next() * 0.04)), population}
      };
    })
  };
  const {gpu, cpu} = await runBoth(
    device,
    'empirical-bayes-rates',
    {numeratorProperty: 'events', denominatorProperty: 'population'},
    {features: areas}
  );
  for (const column of ['raw_rate', 'eb_rate', 'shrinkage']) {
    const actual = gpu.tables[0].columns[column] as number[];
    const expected = cpu.tables[0].columns[column] as number[];
    expect(actual.length).toBe(expected.length);
    actual.forEach((value, row) =>
      expect(Math.abs(value - expected[row]), `${column} row ${row}`).toBeLessThan(2e-4)
    );
  }
});
