// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

function points(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((coordinate, index) => ({
      id: `p${index}`,
      geometry: {type: 'Point', coordinates: coordinate},
      properties: {}
    }))
  } as CatalogFeatureCollection;
}

const GRID: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    [
      [0, 0],
      [100, 0],
      [200, 0]
    ],
    [
      [0, 100],
      [100, 100],
      [200, 100]
    ],
    [
      [0, 0],
      [0, 100]
    ],
    [
      [100, 0],
      [100, 100]
    ],
    [
      [200, 0],
      [200, 100]
    ]
  ].map((coordinates, index) => ({
    id: `line-${index}`,
    geometry: {type: 'LineString', coordinates},
    properties: {}
  }))
} as CatalogFeatureCollection;

test('shortest-path destinations snap on GPUNetworkSnapping and match the CPU snap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('network-shortest-path')!;
  const parameters = {
    coordinateSystem: 'planar',
    snapTolerance: 0,
    splitAtSharedVertices: true,
    costMode: 'length',
    maximumSnapDistance: 10
  };
  const inputs = {
    lines: GRID,
    origin: points([[0, 0]]),
    destinations: points([
      [100, 103],
      [203, 50],
      [160, 100],
      [500, 500]
    ])
  };
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {});
  expect(gpu.engine).toBe('gpu');
  expect(gpu.summary['destination_snap_engine']).toBe('gpu');
  expect(gpu.dispatchedContributors).toContain('GPUNetworkSnapping');
  const columns = (result: typeof gpu, name: string) => result.tables[0].columns[name];
  expect(columns(gpu, 'status')).toEqual(columns(cpu, 'status'));
  expect(columns(gpu, 'status')[3]).toBe('destination-not-snapped');
  columns(cpu, 'total_cost').forEach((expected, row) => {
    if (expected !== null)
      expect(columns(gpu, 'total_cost')[row] as number).toBeCloseTo(expected as number, 2);
    const snapped = columns(cpu, 'snap_distance_destination')[row];
    if (snapped !== null && snapped !== undefined)
      expect(columns(gpu, 'snap_distance_destination')[row] as number).toBeCloseTo(
        snapped as number,
        2
      );
  });
});
