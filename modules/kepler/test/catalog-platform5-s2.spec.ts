// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {runCatalogOperationAsync, type CatalogRunResult} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM5_S2_BINDINGS} from '../src/analysis/catalog-platform5-s2';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

// Oracle policy: the CPU reference engine of each binding is the oracle; the GPU path must run on
// the named contributor (engine 'gpu') and match it.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function getBinding(id: string) {
  const binding = PLATFORM5_S2_BINDINGS.find(candidate => candidate.id === id);
  if (!binding) throw new Error(`missing ${id}`);
  return binding;
}

function points(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {}
    }))
  };
}

function polygons(rings: number[][][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rings.map(ring => ({
      geometry: {type: 'Polygon', coordinates: [ring]},
      properties: {}
    }))
  };
}

async function runBoth(
  device: Device,
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const binding = getBinding(id);
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {device, engine: 'cpu'});
  expect(gpu.engine, `${id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(binding.gpuContributorNames);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function column(
  result: CatalogRunResult,
  name: string
): readonly (string | number | boolean | null)[] {
  return result.tables[0].columns[name];
}

function makeGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function scatter(count: number, seed: number): number[][] {
  const next = makeGenerator(seed);
  return Array.from({length: count}, () => [
    next() * 360 - 180,
    (Math.asin(next() * 2 - 1) * 180) / Math.PI
  ]);
}

// Points on every cube face, seams and near the poles.
const PROBES = [
  [0, 0],
  [90, 0],
  [0, 89],
  [180, 0],
  [-90, 0],
  [0, -89],
  [-179.9, 10],
  [45.01, 35.2],
  [44.99, 35.3]
];

test('s2-cell-hierarchy GPU equals the CPU reference for every relation and level', async () => {
  const device = await getDevice();
  const inputs = {points: points([...PROBES, ...scatter(300, 7)])};
  for (const level of [0, 1, 7, 18, 29, 30]) {
    for (const operation of ['children', 'parent', 'neighbors', 'range']) {
      const {gpu, cpu} = await runBoth(
        device,
        's2-cell-hierarchy',
        {level, operation, parentLevel: Math.floor(level / 2)},
        inputs
      );
      expect(column(gpu, 'cell_id'), `${operation} level ${level}`).toEqual(column(cpu, 'cell_id'));
      expect(column(gpu, 'relation')).toEqual(column(cpu, 'relation'));
      expect(gpu.summary['unavailableCount']).toBe(cpu.summary['unavailableCount']);
    }
  }
});

test('s2-cell-measures GPU matches the float64 CPU reference', async () => {
  const device = await getDevice();
  for (const level of [0, 3, 10, 16, 22]) {
    const {gpu, cpu} = await runBoth(
      device,
      's2-cell-measures',
      {level, radius: 6371008.8},
      {points: points([...PROBES, ...scatter(200, 11 + level)])}
    );
    expect(column(gpu, 'cell_id')).toEqual(column(cpu, 'cell_id'));
    for (const name of ['area', 'perimeter']) {
      const actual = column(gpu, name) as number[];
      const expected = column(cpu, name) as number[];
      actual.forEach((value, row) =>
        expect(
          Math.abs(value - expected[row]) / expected[row],
          `${name} level ${level} row ${row}`
        ).toBeLessThan(2e-3)
      );
    }
  }
});

test('s2-cap-covering GPU selects the same cells as the CPU reference', async () => {
  const device = await getDevice();
  const centers = points([
    [10, 20],
    [-100.3, 41.7],
    [0, 89],
    [179.9, -30],
    [60, -60]
  ]);
  for (const [level, radiusMeters, mode] of [
    [2, 1500000, 'exterior'],
    [3, 800000, 'exterior'],
    [3, 800000, 'interior'],
    [4, 123456, 'exterior'],
    [4, 2500000, 'interior'],
    [5, 400000, 'exterior']
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      's2-cap-covering',
      {level, radiusMeters, mode, maximumCells: 4096},
      {points: centers}
    );
    expect(column(gpu, 'cell_id'), `${mode} level ${level}`).toEqual(column(cpu, 'cell_id'));
    expect(column(gpu, 'point_feature_index')).toEqual(column(cpu, 'point_feature_index'));
  }
  const truncated = await runBoth(
    device,
    's2-cap-covering',
    {level: 3, radiusMeters: 3000000, maximumCells: 20},
    {points: centers}
  );
  expect(column(truncated.gpu, 'cell_id')).toEqual(column(truncated.cpu, 'cell_id'));
  expect(truncated.gpu.diagnostics.map(diagnostic => diagnostic.code)).toContain(
    's2-cover-truncated'
  );
});

test('s2-polygon-covering GPU equals the CPU traversal', async () => {
  const device = await getDevice();
  const shapes = polygons([
    [
      [-30, -20],
      [30, -20],
      [30, 25],
      [-30, 25],
      [-30, -20]
    ],
    [
      [100, 40],
      [130, 45],
      [120, 70],
      [100, 40]
    ],
    [
      [170, -10],
      [-170, -10],
      [-170, 10],
      [170, 10],
      [170, -10]
    ],
    [
      [-20, 70],
      [60, 72],
      [150, 75],
      [-120, 73],
      [-20, 70]
    ]
  ]);
  for (const [mode, minimumLevel, maximumLevel] of [
    ['exterior', 0, 4],
    ['interior', 0, 5],
    ['exterior', 2, 5]
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      's2-polygon-covering',
      {mode, minimumLevel, maximumLevel, maximumCells: 4096},
      {polygons: shapes}
    );
    expect(column(gpu, 'cell_id'), `${mode} ${minimumLevel}-${maximumLevel}`).toEqual(
      column(cpu, 'cell_id')
    );
    expect(column(gpu, 'polygon_feature_index')).toEqual(column(cpu, 'polygon_feature_index'));
    expect(gpu.tables[0].rowCount).toBeGreaterThan(0);
  }
  const bounded = await runBoth(
    device,
    's2-polygon-covering',
    {maximumLevel: 6, maximumCells: 7},
    {polygons: shapes}
  );
  expect(column(bounded.gpu, 'cell_id')).toEqual(column(bounded.cpu, 'cell_id'));
  expect(bounded.gpu.diagnostics.map(diagnostic => diagnostic.code)).toContain(
    's2-cover-truncated'
  );
});
