// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import type {CatalogFeatureCollection, CatalogPosition} from '../src/analysis/catalog-geojson';
import {
  buildNetworkGraph,
  snapPointToNetwork,
  type CatalogNetworkParameters
} from '../src/analysis/catalog-network';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {snapCatalogPoints} from '../src/analysis/catalog-gpu-snapping';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

// Oracle: snapPointToNetwork (CPU, float64). The GPU snaps to polyline segments through
// GPUNetworkSnapping, so edge, fraction and position must agree to float32 precision.

const lines: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {
      id: 'bend',
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [0.01, 0],
          [0.01, 0.01],
          [0.02, 0.01]
        ]
      },
      properties: {}
    },
    {
      id: 'spur',
      geometry: {
        type: 'LineString',
        coordinates: [
          [0.02, 0.01],
          [0.03, 0.03]
        ]
      },
      properties: {}
    }
  ]
};

const points: CatalogPosition[] = [
  [0.005, 0.0004],
  [0.0108, 0.005],
  [0.015, 0.0099],
  [0.0295, 0.0295],
  [0.5, 0.5]
];

for (const coordinateSystem of ['EPSG:4326', 'planar'] as const) {
  test(`GPU snapping matches the CPU snap on ${coordinateSystem}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const parameters: CatalogNetworkParameters = {
      coordinateSystem,
      snapTolerance: 0,
      splitAtSharedVertices: true,
      costMode: 'length',
      forwardCostProperty: '',
      reverseCostProperty: '',
      onewayProperty: ''
    };
    const graph = buildNetworkGraph(lines, parameters, 1000);
    // Planar coordinates are in degrees here too, so scale the snap distance per system.
    const maximumDistance = coordinateSystem === 'planar' ? 0.003 : 400;
    const result = await snapCatalogPoints({device}, graph, points, maximumDistance);
    expect(result.engine, result.fallbackReason).toBe('gpu');
    points.forEach((point, index) => {
      const expected = snapPointToNetwork(graph, point, maximumDistance);
      const actual = result.snapped[index];
      if (!expected) {
        expect(actual, `point ${index}`).toBeNull();
        return;
      }
      expect(actual, `point ${index}`).not.toBeNull();
      expect(actual!.edge).toBe(expected.edge);
      expect(actual!.fraction).toBeCloseTo(expected.fraction, 4);
      expect(actual!.distance).toBeCloseTo(
        expected.distance,
        coordinateSystem === 'planar' ? 5 : 0
      );
      expect(actual!.position[0]).toBeCloseTo(expected.position[0], 5);
      expect(actual!.position[1]).toBeCloseTo(expected.position[1], 5);
    });
  });
}

test('snapping without a device falls back to the CPU snap with a reason', async () => {
  const graph = buildNetworkGraph(
    lines,
    {
      coordinateSystem: 'planar',
      snapTolerance: 0,
      splitAtSharedVertices: true,
      costMode: 'length',
      forwardCostProperty: '',
      reverseCostProperty: '',
      onewayProperty: ''
    },
    1000
  );
  const result = await snapCatalogPoints(undefined, graph, points.slice(0, 2), 0.003);
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toBe('no GPU device');
});

test('snap points operation dispatches GPUNetworkSnapping and agrees with the CPU engine', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('network-snap-points')!;
  const inputs = {
    lines,
    points: {
      type: 'FeatureCollection' as const,
      features: points.map(coordinates => ({
        geometry: {type: 'Point', coordinates},
        properties: {}
      }))
    }
  };
  const parameters = {coordinateSystem: 'EPSG:4326', maximumSnapDistance: 400};
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {engine: 'cpu'});
  expect(gpu.engine, gpu.fallbackReason).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(['GPUNetworkSnapping']);
  expect(gpu.tables[0].columns['status']).toEqual(cpu.tables[0].columns['status']);
  expect(gpu.tables[0].columns['edge_id']).toEqual(cpu.tables[0].columns['edge_id']);
});
