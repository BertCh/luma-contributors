// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {POINT_CLOUD_BINDINGS} from '../src/analysis/catalog-pointcloud-bindings';
import {
  makePointCloudFixture,
  POINT_CLOUD_PLANE,
  POINT_CLOUD_CANOPY
} from './catalog-pointcloud-fixtures';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

async function run(
  operation: string,
  inputs: Record<string, CatalogFeatureCollection>,
  parameters: Record<string, unknown> = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding(`pointcloud-${operation}`)!;
  const result = await runCatalogOperationAsync(
    binding,
    {coordinateSystem: 'planar', ...parameters},
    inputs,
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(result.dispatchedContributors).toEqual(binding.gpuContributorNames);
  for (const output of binding.outputs) {
    const table = result.tables.find(candidate => candidate.id === output.tableId)!;
    expect(table.rowCount).toBeLessThanOrEqual(output.maximumRows);
    for (const identity of output.identityColumns) expect(table.columns[identity]).toBeDefined();
  }
  return result;
}

test('neighborhoods: independent exact distances, ID tie breaks, counts and truncation', async () => {
  const result = await run(
    'neighborhoods',
    {points: POINT_CLOUD_PLANE},
    {radius: 1.5, capacity: 3}
  );
  const coordinates = POINT_CLOUD_PLANE.features.map(
    feature => feature.geometry!.coordinates as number[]
  );
  const columns = result.tables[0].columns;
  for (const [row, position] of coordinates.entries()) {
    const neighbors = coordinates
      .map((candidate, index) => ({
        index,
        squaredDistance: candidate.reduce(
          (total, value, axis) => total + (value - position[axis]) ** 2,
          0
        )
      }))
      .filter(candidate => candidate.index !== row && candidate.squaredDistance <= 1.5 ** 2)
      .sort(
        (first, second) =>
          first.squaredDistance - second.squaredDistance || first.index - second.index
      );
    expect(columns['neighbor_count'][row]).toBe(neighbors.length);
    for (let slot = 0; slot < 3; slot++) {
      expect(columns[`neighbor_${slot + 1}_row`][row]).toBe(neighbors[slot].index);
      expect(columns[`neighbor_${slot + 1}_distance`][row]).toBeCloseTo(
        Math.sqrt(neighbors[slot].squaredDistance),
        5
      );
    }
    expect(columns['neighborhood_status'][row]).toBe(neighbors.length > 3 ? 2 : 0);
  }
  expect(result.diagnostics.some(diagnostic => diagnostic.code === 'neighborhood_status')).toBe(
    true
  );
});

test('features: analytical plane normals, population covariance and shape descriptors', async () => {
  const result = await run('features', {points: POINT_CLOUD_PLANE}, {radius: 3, capacity: 64});
  const columns = result.tables[0].columns;
  expect(columns['source_id']).toEqual(POINT_CLOUD_PLANE.features.map(feature => feature.id));
  expect(columns['altitude']).toEqual(Array(9).fill(100));
  expect(columns['intensity']).toEqual(Array.from({length: 9}, (_, row) => row * 10));
  for (const value of columns['normal_z']) expect(value).toBeCloseTo(1, 5);
  expect(columns['eigenvalue_1'][4]).toBeCloseTo(0.75, 5);
  expect(columns['eigenvalue_2'][4]).toBeCloseTo(0.75, 5);
  expect(columns['eigenvalue_3'][4]).toBeCloseTo(0, 6);
  expect(columns['covariance_xx'][4]).toBeCloseTo(0.75, 5);
  expect(columns['planarity'][4]).toBeCloseTo(1, 5);
});

for (const operation of ['ground', 'smrf'])
  test(`${operation}: elevated duplicate is rejected and original rows survive`, async () => {
    const result = await run(operation, {points: POINT_CLOUD_CANOPY});
    const columns = result.tables[0].columns;
    expect(columns['ground_mask'].slice(0, 9)).toEqual(Array(9).fill(1));
    expect(columns['ground_mask'][9]).toBe(0);
    expect(columns['classification'].slice(0, 9)).toEqual(Array(9).fill(2));
    expect(columns['classification'][9]).toBe(1);
    expect(columns['source_row']).toEqual(Array.from({length: 10}, (_, row) => row));
  });

test('surfaces: cell-min DTM, cell-max DSM, canopy and south/top row identity', async () => {
  const result = await run('surfaces', {points: POINT_CLOUD_CANOPY});
  expect(result.tables[0].columns['terrain']).toEqual(Array(10).fill(100));
  expect(result.tables[0].columns['surface'][4]).toBe(103);
  expect(result.tables[0].columns['canopy'][9]).toBe(3);
  const grid = result.tables[1];
  expect(grid.rowCount).toBe(9);
  expect(grid.columns['terrain']).toEqual(Array(9).fill(100));
  expect(grid.columns['canopy'][4]).toBe(3);
  expect(Number(grid.columns['latitude'][0])).toBeGreaterThan(Number(grid.columns['latitude'][8]));
});

test('height: ground coverage bit, negative heights, null unsupported cells', async () => {
  const coordinates = [
    ...POINT_CLOUD_PLANE.features.map(feature => feature.geometry!.coordinates as number[]),
    [1000001, 4000001, 98],
    [1000004, 4000004, 104]
  ];
  const fixture = makePointCloudFixture(coordinates, [...Array<number>(9).fill(2), 1, 1]);
  const result = await run('height', {points: fixture});
  expect(result.tables[0].columns['height_above_ground'].slice(0, 9)).toEqual(Array(9).fill(0));
  expect(result.tables[0].columns['height_above_ground'][9]).toBe(-2);
  expect(result.tables[0].columns['height_above_ground'][10]).toBeNull();
  expect(result.tables[0].columns['height_status'][10]).toBe(2);
});

test('statistical filter: isolated point is an outlier; output does not compact rows', async () => {
  const fixture = makePointCloudFixture([
    ...POINT_CLOUD_PLANE.features.map(feature => feature.geometry!.coordinates as number[]),
    [1000100, 4000100, 100]
  ]);
  const result = await run(
    'statistical-filter',
    {points: fixture},
    {radius: 1000, standardDeviations: 1, capacity: 64}
  );
  expect(result.tables[0].columns['inlier']).toEqual([...Array(9).fill(1), 0]);
  expect(result.tables[0].rowCount).toBe(10);
});

test('normal scale: planar supports choose an eligible scale with upward normals', async () => {
  const result = await run(
    'normal-scale',
    {points: POINT_CLOUD_PLANE},
    {radii: [1.5, 3], capacity: 64}
  );
  expect(result.tables[0].columns['normal_status']).toEqual(Array(9).fill(0));
  for (const value of result.tables[0].columns['normal_z']) expect(value).toBeCloseTo(1, 5);
  for (const value of result.tables[0].columns['normal_scale']) expect([1.5, 3]).toContain(value);
});

test('ICP: known translation, transformed native coordinates and local transform history', async () => {
  const coordinates = [
    [0, 0, 0],
    [3, 0, 0],
    [0, 4, 0],
    [0, 0, 5],
    [2, 2, 2]
  ];
  const source = makePointCloudFixture(
    coordinates.map(position =>
      position.map((value, axis) => value + [1000000.2, 4000000.1, 100.3][axis])
    )
  );
  const target = makePointCloudFixture(
    coordinates.map(position =>
      position.map((value, axis) => value + [1000000, 4000000, 100][axis])
    )
  );
  const result = await run('icp', {source, target}, {maximumDistance: 2, maximumIterations: 10});
  expect([0, 1]).toContain(result.summary['registrationStatus']);
  const table = result.tables[0];
  for (let row = 0; row < 5; row++) {
    expect(table.columns['longitude'][row]).toBeCloseTo(coordinates[row][0] + 1000000, 4);
    expect(table.columns['latitude'][row]).toBeCloseTo(coordinates[row][1] + 4000000, 4);
    expect(table.columns['altitude'][row]).toBeCloseTo(coordinates[row][2] + 100, 4);
  }
  expect(result.tables[1].columns['transform_14'][0]).toBeCloseTo(-0.3, 4);
});

test('change: known normal displacement, epoch support and explicit alignment refusal', async () => {
  const comparison = makePointCloudFixture(
    POINT_CLOUD_PLANE.features.map(feature => {
      const position = feature.geometry!.coordinates as number[];
      return [position[0], position[1], position[2] + 0.2];
    })
  );
  await expect(run('change', {reference: POINT_CLOUD_PLANE, comparison})).rejects.toThrow(
    /alreadyAligned/
  );
  const result = await run(
    'change',
    {reference: POINT_CLOUD_PLANE, comparison},
    {alreadyAligned: true, radius: 3, capacity: 64}
  );
  for (const value of result.tables[0].columns['change']) expect(value).toBeCloseTo(0.2, 5);
  expect(result.tables[0].columns['change_status']).toEqual(Array(9).fill(0));
  expect(result.tables[0].columns['reference_count']).toEqual(Array(9).fill(9));
});

test('voxel merge: weighted centroids and representatives, empty support, certification and overflow', async () => {
  const left = makePointCloudFixture([
    [1000000, 4000000, 100],
    [1, 1, 1]
  ]);
  const right = makePointCloudFixture([
    [1000004, 4000000, 104],
    [2, 2, 2]
  ]);
  left.features.forEach((feature, row) => {
    feature.properties = {voxel_id: `cell-${row}`, count: row === 0 ? 3 : 0, representative: row};
  });
  right.features.forEach((feature, row) => {
    feature.properties = {voxel_id: `cell-${row}`, count: row === 0 ? 1 : 0, representative: row};
  });
  await expect(run('voxel-merge', {left, right})).rejects.toThrow(/disjointOwnedRows/);
  const result = await run(
    'voxel-merge',
    {left, right},
    {disjointOwnedRows: true, rightRowOffset: 100}
  );
  const columns = result.tables[0].columns;
  expect(columns['count']).toEqual([4, 0]);
  expect(columns['longitude'][0]).toBeCloseTo(1000001, 1);
  expect(columns['altitude'][0]).toBeCloseTo(101, 4);
  expect(columns['representative']).toEqual([0, null]);
  expect(columns['longitude'][1]).toBeNull();
  right.features[0].properties!['voxel_id'] = 'wrong';
  await expect(run('voxel-merge', {left, right}, {disjointOwnedRows: true})).rejects.toThrow(
    /voxel IDs/
  );
});

test('degenerate features and insufficient change support publish null measurements with statuses', async () => {
  const line = makePointCloudFixture([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0]
  ]);
  const features = await run('features', {points: line}, {radius: 4});
  expect(features.tables[0].columns['normal_z']).toEqual(Array(4).fill(null));
  const columns = features.tables[0].columns;
  expect(columns['feature_status']).toEqual(Array(4).fill(1));
  for (const name of ['normal_x', 'normal_y', 'linearity', 'planarity', 'scattering'])
    expect(columns[name]).toEqual(Array(4).fill(null));
  // Other-point supports are [1,2,3], [0,2,3], [0,1,3], [0,1,2]. Their population
  // variances remain meaningful even though each covariance has rank one.
  const expectedVariances = [2 / 3, 14 / 9, 14 / 9, 2 / 3];
  expectedVariances.forEach((expected, row) => {
    expect(columns['eigenvalue_1'][row]).toBeCloseTo(expected, 5);
    expect(columns['covariance_xx'][row]).toBeCloseTo(expected, 5);
  });
  for (const name of [
    'eigenvalue_2',
    'eigenvalue_3',
    'covariance_xy',
    'covariance_xz',
    'covariance_yy',
    'covariance_yz',
    'covariance_zz'
  ])
    expect(columns[name]).toEqual(Array(4).fill(0));
  const insufficient = await run('features', {points: line}, {radius: 4, capacity: 2});
  for (const name of [
    'eigenvalue_1',
    'eigenvalue_2',
    'eigenvalue_3',
    'covariance_xx',
    'covariance_xy',
    'covariance_xz',
    'covariance_yy',
    'covariance_yz',
    'covariance_zz'
  ])
    expect(insufficient.tables[0].columns[name]).toEqual(Array(4).fill(null));
  const change = await run(
    'change',
    {reference: POINT_CLOUD_PLANE, comparison: makePointCloudFixture([[1000000, 4000000, 100]])},
    {alreadyAligned: true, radius: 3, capacity: 64}
  );
  expect(change.tables[0].columns['change']).toEqual(Array(9).fill(null));
  expect(change.tables[0].columns['change_status']).toEqual(Array(9).fill(2));
});

test('all point-cloud GPU contributors are covered by the headless cases', () => {
  expect(POINT_CLOUD_BINDINGS).toHaveLength(11);
});

test('voxel count overflow preserves status and hides unusable geometry', async () => {
  const left = makePointCloudFixture([[0, 0, 0]]);
  const right = makePointCloudFixture([[1, 0, 0]]);
  left.features[0].properties = {voxel_id: 'one', count: 0xffffffff, representative: 0};
  right.features[0].properties = {voxel_id: 'one', count: 1, representative: 0};
  const result = await run('voxel-merge', {left, right}, {disjointOwnedRows: true});
  expect(Number(result.tables[0].columns['merge_status'][0]) & 1).toBe(1);
  expect(result.tables[0].columns['longitude'][0]).toBeNull();
  expect(result.tables[0].columns['representative'][0]).toBeNull();
  expect(result.diagnostics.some(diagnostic => diagnostic.code === 'merge_status')).toBe(true);
});

test('bilinear height exposes edge support failures without zero fill', async () => {
  const result = await run('height', {points: POINT_CLOUD_PLANE}, {interpolation: 'bilinear'});
  expect(result.tables[0].columns['height_status'][4]).toBe(0);
  expect(result.tables[0].columns['height_above_ground'][4]).toBe(0);
  expect(result.tables[0].columns['height_status'][0]).toBe(2);
  expect(result.tables[0].columns['height_above_ground'][0]).toBeNull();
});

test('ground prepares cell minima from all points independently of existing LAS classes', async () => {
  const fixture = makePointCloudFixture(
    POINT_CLOUD_CANOPY.features.map(feature => feature.geometry!.coordinates as number[]),
    [...Array<number>(9).fill(6), 2]
  );
  const result = await run('ground', {points: fixture});
  const columns = result.tables[0].columns;
  expect(result.dispatchedContributors).toEqual(['GPUPointCloudSurfaces', 'GPUPointCloudGround']);
  expect(columns['ground_mask']).toEqual([...Array(9).fill(1), 0]);
  expect(columns['classification']).toEqual([...Array(9).fill(2), 1]);
  expect(columns['source_classification']).toEqual([...Array(9).fill(6), 2]);
  // The windows extend beyond this 3x3 grid, but every point's cell has valid support.
  expect(columns['ground_status']).toEqual(Array(10).fill(2));
});

test('ground retains the documented radius-increment thresholds at both morphology stages', async () => {
  // Match the independent hand-worked oracle in gpu-pointcloud-ground-threshold.spec.ts:
  // 1 m cells and radii [1,2] give 0.3 m at both stages, not Zhang's 0.2/0.4 m.
  // A single-cell bump opens away at radius 1; a 3x3 block opens away at radius 2.
  const width = 16;
  const elevations = Array<number>(width * width).fill(0);
  const expected = Array<number>(width * width).fill(1);
  elevations[3 * width + 3] = 0.25;
  elevations[3 * width + 11] = 0.35;
  expected[3 * width + 11] = 0;
  for (let row = 9; row <= 11; row++) {
    for (let column = 2; column <= 4; column++) elevations[row * width + column] = 0.25;
    for (let column = 10; column <= 12; column++) {
      elevations[row * width + column] = 0.35;
      expected[row * width + column] = 0;
    }
  }
  const fixture = makePointCloudFixture(
    elevations.map((elevation, cell) => [
      1000000 + (cell % width),
      4000000 + Math.floor(cell / width),
      100 + elevation
    ]),
    Array<number>(width * width).fill(1)
  );
  const result = await run('ground', {points: fixture});
  expect(result.summary['gridWidth']).toBe(width);
  expect(result.summary['gridHeight']).toBe(width);
  expect(result.tables[0].columns['ground_mask']).toEqual(expected);
  expect(result.tables[0].columns['classification']).toEqual(
    expected.map(value => (value ? 2 : 1))
  );
});

test('voxel merge preserves projected millimeters when other voxel slots are empty', async () => {
  const left = makePointCloudFixture([
    [1000000.001, 4000000.001, 100.001],
    [0, 0, 0]
  ]);
  const right = makePointCloudFixture([
    [1000000.002, 4000000.002, 100.002],
    [0, 0, 0]
  ]);
  for (const input of [left, right])
    input.features.forEach((feature, row) => {
      feature.properties = {
        voxel_id: `cell-${row}`,
        count: row === 0 ? (input === left ? 3 : 1) : 0,
        representative: row
      };
    });
  const result = await run('voxel-merge', {left, right}, {disjointOwnedRows: true});
  const columns = result.tables[0].columns;
  expect(columns['longitude'][0]).toBeCloseTo(1000000.00125, 8);
  expect(columns['latitude'][0]).toBeCloseTo(4000000.00125, 8);
  expect(columns['altitude'][0]).toBeCloseTo(100.00125, 8);
  expect(columns['count']).toEqual([4, 0]);
  expect(columns['source_row']).toEqual([0, 1]);
  expect(columns['merge_status']).toEqual([0, 0]);
  for (const name of ['longitude', 'latitude', 'altitude', 'representative'])
    expect(columns[name][1]).toBeNull();
});
