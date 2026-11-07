// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import type {CatalogFeatureCollection, CatalogPosition} from '../src/analysis/catalog-geojson';
import {buildNetworkGraph, snapPointToNetwork} from '../src/analysis/catalog-network';
import {snapCatalogPointsOnGpu} from '../src/analysis/catalog-gpu-snapping';

// Oracle: snapPointToNetwork (CPU, float64). The BVH path must pick the same edge as the exact
// GPU scan and the CPU snap; an overflowing candidate search must fall back to the scan.

const PARAMETERS = {
  coordinateSystem: 'planar' as const,
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'length' as const,
  forwardCostProperty: '',
  reverseCostProperty: '',
  onewayProperty: ''
};

function makeGridLines(size: number): CatalogFeatureCollection {
  const features: CatalogFeatureCollection['features'] = [];
  for (let index = 0; index <= size; index++) {
    features.push({
      id: `row${index}`,
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, index],
          [size, index + 0.25]
        ]
      },
      properties: {}
    });
    features.push({
      id: `column${index}`,
      geometry: {
        type: 'LineString',
        coordinates: [
          [index + 0.13, 0],
          [index, size]
        ]
      },
      properties: {}
    });
  }
  return {type: 'FeatureCollection', features};
}

function makePoints(count: number, extent: number): CatalogPosition[] {
  let state = 99;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return Array.from({length: count}, () => [next() * extent, next() * extent]);
}

test('BVH snapping agrees with the exact scan and the CPU snap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = buildNetworkGraph(makeGridLines(20), PARAMETERS, 10000);
  const points = makePoints(300, 20);
  const maximumDistance = 0.3;
  const searches: string[] = [];
  const indexed = await snapCatalogPointsOnGpu({device}, graph, points, maximumDistance, {
    spatialIndex: 'bvh',
    onSearch: search => searches.push(search)
  });
  const scanned = await snapCatalogPointsOnGpu({device}, graph, points, maximumDistance, {
    spatialIndex: 'scan',
    onSearch: search => searches.push(search)
  });
  expect(searches).toEqual(['bvh', 'scan']);
  let snappedCount = 0;
  points.forEach((point, index) => {
    const expected = snapPointToNetwork(graph, point, maximumDistance);
    expect(Boolean(indexed[index]), `point ${index} snapped`).toBe(Boolean(expected));
    expect(indexed[index]?.edge, `point ${index} edge`).toBe(scanned[index]?.edge);
    if (!expected) return;
    snappedCount++;
    expect(indexed[index]!.distance).toBeCloseTo(expected.distance, 4);
  });
  expect(snappedCount).toBeGreaterThan(20);
});

test('auto mode indexes a large point-by-segment product and small ones scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = buildNetworkGraph(makeGridLines(20), PARAMETERS, 10000);
  const searches: string[] = [];
  const record = (search: string) => searches.push(search);
  await snapCatalogPointsOnGpu({device}, graph, makePoints(4, 20), 0.3, {onSearch: record});
  await snapCatalogPointsOnGpu({device}, graph, makePoints(2000, 20), 0.3, {onSearch: record});
  expect(searches[0]).toBe('scan');
  expect(searches[1]).toBe('bvh');
});

test('an overflowing candidate search reruns the exact scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = buildNetworkGraph(makeGridLines(20), PARAMETERS, 10000);
  const points = makePoints(50, 20);
  const searches: string[] = [];
  // A radius covering the whole grid makes every segment a candidate for every point.
  const snapped = await snapCatalogPointsOnGpu({device}, graph, points, 40, {
    spatialIndex: 'bvh',
    onSearch: search => searches.push(search)
  });
  expect(searches).toEqual(['scan-after-overflow']);
  points.forEach((point, index) => {
    expect(snapped[index]?.edge).toBe(snapPointToNetwork(graph, point, 40)?.edge);
  });
});
