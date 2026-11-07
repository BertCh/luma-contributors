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
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

// Oracle policy: the CPU reference engines of the PLATFORM4 bindings are the oracle; each GPU path
// must run on its named contributors (engine 'gpu', dispatchedContributors accurate) and match.

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

/** Geographic scatter with quarter-integer values so fixed-point sums are exact. */
function scatter(count: number, seed: number, west = -20, south = 30, extent = 40) {
  const next = makeGenerator(seed);
  return points(
    Array.from({length: count}, () => ({
      coordinates: [west + next() * extent, south + next() * extent * 0.5],
      properties: {v: Math.floor(next() * 40) / 4}
    }))
  );
}

async function runBoth(
  device: Device,
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const binding = getCatalogBinding(id)!;
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {device, engine: 'cpu'});
  expect(gpu.engine, `${id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(binding.gpuContributorNames);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function indexRows(result: CatalogRunResult, key: string[]): Map<string, Record<string, unknown>> {
  const table = result.tables[0];
  const rows = new Map<string, Record<string, unknown>>();
  for (let row = 0; row < table.rowCount; row++) {
    const record: Record<string, unknown> = {};
    for (const name of Object.keys(table.columns)) record[name] = table.columns[name][row];
    rows.set(key.map(name => String(record[name])).join('|'), record);
  }
  return rows;
}

function expectColumnsMatch(
  gpu: CatalogRunResult,
  cpu: CatalogRunResult,
  key: string[],
  columns: string[],
  tolerance: number
): void {
  const actual = indexRows(gpu, key);
  const expected = indexRows(cpu, key);
  expect(actual.size).toBe(expected.size);
  for (const [rowKey, record] of expected) {
    const candidate = actual.get(rowKey);
    expect(candidate, `row ${rowKey}`).toBeDefined();
    for (const column of columns) {
      const want = record[column];
      const got = candidate![column];
      if (typeof want === 'number' && typeof got === 'number')
        expect(Math.abs(got - want), `${rowKey} ${column}`).toBeLessThanOrEqual(
          tolerance * Math.max(1, Math.abs(want))
        );
      else expect(got, `${rowKey} ${column}`).toEqual(want);
    }
  }
}

test('dggs-point-index keys match the CPU references for Quadbin, quadkey, geohash and S2', async () => {
  const device = await getDevice();
  const data = scatter(300, 3);
  for (const [family, resolution] of [
    ['quadbin', 9],
    ['quadkey', 9],
    ['geohash', 6],
    ['s2', 8]
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      'dggs-point-index',
      {family, resolution},
      {points: data}
    );
    const actual = gpu.tables[0].columns['cell_id'] as string[];
    const expected = cpu.tables[0].columns['cell_id'] as string[];
    expect(actual.length).toBe(300);
    expect(actual, family).toEqual(expected);
  }
});

test('dggs-point-index indexes H3 on the GPU only (valid H3 mode-1 ids, no CPU engine)', async () => {
  const device = await getDevice();
  const binding = getCatalogBinding('dggs-point-index')!;
  const gpu = await runCatalogOperationAsync(
    binding,
    {family: 'h3', resolution: 3},
    {points: scatter(50, 4)},
    {device}
  );
  expect(gpu.engine).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(['GPUPointToCell']);
  for (const id of gpu.tables[0].columns['cell_id'] as string[]) {
    expect(id).toMatch(/^08[0-9a-f]{14}$/); // mode 1, reserved 0, resolution 3 follows
    expect(id.slice(0, 3)).toBe('083');
  }
  const cpu = await runCatalogOperationAsync(
    binding,
    {family: 'h3', resolution: 3},
    {points: scatter(50, 4)},
    {device, engine: 'cpu'}
  ).catch(failure => failure as Error);
  expect((cpu as Error).message).toMatch(/H3/);
});

test('dggs-cell-aggregation matches the CPU oracle for Quadbin and runs H3 on the GPU', async () => {
  const device = await getDevice();
  const data = scatter(500, 5);
  const {gpu, cpu} = await runBoth(
    device,
    'dggs-cell-aggregation',
    {family: 'quadbin', resolution: 6, valueProperty: 'v'},
    {points: data}
  );
  expect(cpu.tables[0].rowCount).toBeGreaterThan(5);
  expectColumnsMatch(gpu, cpu, ['cell_id'], ['count', 'sum', 'mean', 'min', 'max'], 1e-6);
  // Boundary polygons agree to float32 precision.
  const gpuRows = indexRows(gpu, ['cell_id']);
  for (const [cell, record] of indexRows(cpu, ['cell_id'])) {
    const want = JSON.parse(String(record['geometry'])).coordinates[0] as number[][];
    const got = JSON.parse(String(gpuRows.get(cell)!['geometry'])).coordinates[0] as number[][];
    // The GPU ring starts at another corner and may wind the other way; compare the extents.
    expect(got).toHaveLength(want.length);
    const extent = (ring: number[][]) => [
      Math.min(...ring.map(vertex => vertex[0])),
      Math.min(...ring.map(vertex => vertex[1])),
      Math.max(...ring.map(vertex => vertex[0])),
      Math.max(...ring.map(vertex => vertex[1]))
    ];
    extent(got).forEach((value, index) =>
      expect(Math.abs(value - extent(want)[index])).toBeLessThan(1e-4)
    );
  }

  const h3 = await runCatalogOperationAsync(
    getCatalogBinding('dggs-cell-aggregation')!,
    {family: 'h3', resolution: 3, valueProperty: 'v'},
    {points: data},
    {device}
  );
  expect(h3.engine).toBe('gpu');
  expect(h3.dispatchedContributors).toEqual([
    'GPUPointToCell',
    'GPUCellAggregation',
    'GPUCellGeometry'
  ]);
  const counts = h3.tables[0].columns['count'] as number[];
  expect(counts.reduce((sum, count) => sum + count, 0)).toBe(500);
  expect(h3.tables[0].rowCount).toBeGreaterThan(3);
  const sums = h3.tables[0].columns['sum'] as number[];
  const expectedTotal = data.features.reduce(
    (sum, feature) => sum + (feature.properties!['v'] as number),
    0
  );
  expect(sums.reduce((sum, value) => sum + value, 0)).toBeCloseTo(expectedTotal, 3);
  for (const geometry of h3.tables[0].columns['geometry'] as string[]) {
    const ring = JSON.parse(geometry).coordinates[0] as number[][];
    expect(ring.length).toBeGreaterThanOrEqual(6);
  }
});

test('dggs-cell-rollup and dggs-cell-compare match the CPU oracle', async () => {
  const device = await getDevice();
  const rollup = await runBoth(
    device,
    'dggs-cell-rollup',
    {family: 'quadbin', resolution: 9, levels: 3, levelStep: 2, valueProperty: 'v'},
    {points: scatter(400, 6)}
  );
  expect(rollup.cpu.tables[0].rowCount).toBeGreaterThan(10);
  expectColumnsMatch(
    rollup.gpu,
    rollup.cpu,
    ['cell_id', 'level'],
    ['count', 'sum', 'min', 'max', 'parent_cell_id'],
    1e-6
  );
  for (const measure of ['count', 'sum'] as const) {
    const compare = await runBoth(
      device,
      'dggs-cell-compare',
      {family: 'quadbin', resolution: 5, valueProperty: 'v', measure},
      {before: scatter(300, 7), after: scatter(260, 8, -10)}
    );
    const statuses = new Set(compare.cpu.tables[0].columns['status'] as string[]);
    expect(statuses.has('appeared') && statuses.has('disappeared')).toBe(true);
    expectColumnsMatch(
      compare.gpu,
      compare.cpu,
      ['cell_id'],
      ['status', 'before', 'after', 'delta', 'ratio', 'percent_change', 'z_score'],
      2e-4
    );
  }
});

test('weights bindings run on their contributors and match the CPU oracle', async () => {
  const device = await getDevice();
  const next = makeGenerator(9);
  const data = points(
    Array.from({length: 120}, () => ({coordinates: [next() * 1000, next() * 1000]}))
  );
  const base = {coordinateSystem: 'planar'};
  for (const [operation, weightRule] of [
    ['union', 'sum'],
    ['union', 'binary'],
    ['intersection', 'max'],
    ['intersection', 'product'],
    ['difference', 'left'],
    ['symmetricDifference', 'binary']
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      'weights-algebra',
      {...base, neighborCount: 4, distanceBand: 150, operation, weightRule},
      {points: data}
    );
    expect(cpu.tables[0].rowCount, `${operation} ${weightRule}`).toBeGreaterThan(0);
    expectColumnsMatch(gpu, cpu, ['source_index', 'target_index'], ['weight'], 1e-6);
  }
  const transpose = await runBoth(
    device,
    'weights-transpose',
    {...base, neighborCount: 4, decayDistance: 200},
    {points: data}
  );
  expect(transpose.cpu.tables[0].rowCount).toBe(480);
  expectColumnsMatch(
    transpose.gpu,
    transpose.cpu,
    ['source_index', 'target_index'],
    ['weight'],
    1e-6
  );
  for (const operation of ['row', 'binary', 'double', 'variance'] as const) {
    const transform = await runBoth(
      device,
      'weights-transform',
      {...base, neighborCount: 4, decayDistance: 200, operation},
      {points: data}
    );
    expectColumnsMatch(
      transform.gpu,
      transform.cpu,
      ['source_index', 'target_index'],
      ['weight'],
      2e-5
    );
  }
});

test('regression-ols matches the float64 CPU fit to float32 tolerance', async () => {
  const device = await getDevice();
  const next = makeGenerator(10);
  const data = points(
    Array.from({length: 200}, (_, index) => {
      const a = next() * 10;
      const b = next() * 4 - 2;
      const c = next();
      return {
        coordinates: [index, 0],
        properties: {a, b, c, y: 1.5 + 2 * a - 3 * b + 0.5 * c + (next() - 0.5)}
      };
    })
  );
  const {gpu, cpu} = await runBoth(
    device,
    'regression-ols',
    {responseProperty: 'y', predictorProperties: 'a,b,c'},
    {points: data}
  );
  expectColumnsMatch(gpu, cpu, ['term'], ['estimate', 'standard_error', 't_statistic'], 2e-3);
  const actual = indexRows({...gpu, tables: [gpu.tables[1]]}, ['feature_index']);
  const expected = indexRows({...cpu, tables: [cpu.tables[1]]}, ['feature_index']);
  for (const [row, record] of expected)
    expect(
      Math.abs((actual.get(row)!['residual'] as number) - (record['residual'] as number))
    ).toBeLessThan(2e-3);
  expect(
    Math.abs((gpu.summary['rSquared'] as number) - (cpu.summary['rSquared'] as number))
  ).toBeLessThan(1e-4);
  expect(
    Math.abs(
      (gpu.summary['adjustedRSquared'] as number) - (cpu.summary['adjustedRSquared'] as number)
    )
  ).toBeLessThan(1e-4);
});

test('spherical-point-in-polygon matches the CPU oracle incl. antimeridian and invalid polygons', async () => {
  const device = await getDevice();
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [170, -10],
              [-170, -10],
              [-170, 10],
              [170, 10],
              [170, -10]
            ]
          ]
        },
        properties: {}
      },
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 60],
              [90, 60],
              [180, 60],
              [-90, 60],
              [0, 60]
            ],
            [
              [0, 80],
              [-90, 80],
              [180, 80],
              [90, 80],
              [0, 80]
            ]
          ]
        },
        properties: {}
      },
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [-10, -10],
              [10, 10],
              [10, -10],
              [-10, 10],
              [-10, -10]
            ]
          ]
        },
        properties: {}
      }
    ]
  };
  const next = makeGenerator(11);
  const data = points([
    {coordinates: [180, 0]},
    {coordinates: [-175, 5]},
    {coordinates: [0, 0]},
    {coordinates: [20, 70]},
    {coordinates: [20, 89]},
    ...Array.from({length: 60}, () => ({coordinates: [next() * 360 - 180, next() * 180 - 90]}))
  ]);
  for (const boundary of ['closed', 'open']) {
    const {gpu, cpu} = await runBoth(
      device,
      'spherical-point-in-polygon',
      {boundary},
      {
        points: data,
        polygons
      }
    );
    expect(cpu.tables[0].rowCount).toBeGreaterThan(1);
    expectColumnsMatch(
      gpu,
      cpu,
      ['point_feature_index', 'polygon_feature_index'],
      ['on_boundary'],
      0
    );
  }
});

test('raster-join-points-in-polygons stays within its published boundary error bound', async () => {
  const device = await getDevice();
  const ring = (x: number, y: number, size: number) => [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size],
    [x, y]
  ];
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {type: 'Polygon', coordinates: [ring(0, 0, 400), ring(150, 150, 100)]},
        properties: {}
      },
      {geometry: {type: 'Polygon', coordinates: [ring(500, 100, 300)]}, properties: {}},
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [100, 500],
              [450, 520],
              [300, 900],
              [100, 500]
            ]
          ]
        },
        properties: {}
      }
    ]
  };
  const next = makeGenerator(12);
  const data = points(
    Array.from({length: 800}, () => ({
      coordinates: [next() * 900, next() * 1000],
      properties: {v: Math.floor(next() * 8) / 4}
    }))
  );
  const binding = getCatalogBinding('raster-join-points-in-polygons')!;
  const parameters = {coordinateSystem: 'planar', cellSize: 4, valueProperty: 'v'};
  const gpu = await runCatalogOperationAsync(
    binding,
    parameters,
    {points: data, polygons},
    {device}
  );
  const cpu = await runCatalogOperationAsync(
    binding,
    parameters,
    {points: data, polygons},
    {
      device,
      engine: 'cpu'
    }
  );
  expect(gpu.engine, gpu.fallbackReason).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(['GPUPolygonRasterization', 'GPURasterJoin']);
  expect(gpu.summary['isApproximate']).toBe(1);
  const gpuCounts = gpu.tables[0].columns['count'] as number[];
  const cpuCounts = cpu.tables[0].columns['count'] as number[];
  const boundaryCounts = gpu.tables[0].columns['boundary_count'] as number[];
  const allBoundary =
    boundaryCounts.reduce((sum, count) => sum + count, 0) +
    (gpu.summary['unassignedBoundaryCount'] as number);
  expect(cpuCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(150);
  cpuCounts.forEach((exact, zone) => {
    // The documented bracket of the exact count.
    expect(exact).toBeGreaterThanOrEqual(gpuCounts[zone] - boundaryCounts[zone]);
    expect(exact).toBeLessThanOrEqual(gpuCounts[zone] - boundaryCounts[zone] + allBoundary);
  });
  // Away from boundaries the raster join is exact: a fine grid agrees on almost every point.
  expect(
    cpuCounts.reduce((sum, count, zone) => sum + Math.abs(count - gpuCounts[zone]), 0)
  ).toBeLessThanOrEqual(allBoundary);
});
