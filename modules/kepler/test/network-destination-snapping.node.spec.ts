// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {
  buildNetworkGraph,
  runNetworkShortestPaths,
  snapPointToNetwork
} from '../src/analysis/catalog-network';

function collection(
  type: 'LineString' | 'Point',
  coordinates: number[][]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features:
      type === 'Point'
        ? coordinates.map((coordinate, index) => ({
            id: `p${index}`,
            geometry: {type: 'Point', coordinates: coordinate},
            properties: {}
          }))
        : [{id: 'road', geometry: {type, coordinates}, properties: {}}]
  } as CatalogFeatureCollection;
}

const PARAMETERS = {
  coordinateSystem: 'planar',
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'length',
  forwardCostProperty: '',
  reverseCostProperty: '',
  onewayProperty: '',
  maximumSnapDistance: 10
} as never;
const LIMITS = {maximumEdges: 100, maximumDestinations: 10};

test('caller-supplied destination snaps replace the per-point CPU snap', () => {
  const lines = collection('LineString', [
    [0, 0],
    [100, 0]
  ]);
  const origins = collection('Point', [[0, 0]]);
  const destinations = collection('Point', [
    [60, 5],
    [60, 500]
  ]);
  const graph = buildNetworkGraph(lines, PARAMETERS, LIMITS.maximumEdges);
  const cpu = runNetworkShortestPaths(lines, origins, destinations, PARAMETERS, LIMITS, {graph});
  expect(cpu.tables[0].columns['status']).toEqual(['ok', 'destination-not-snapped']);
  // Snaps handed in from elsewhere (the GPU) are used verbatim: here destination 1 snaps to 80.
  const forced = snapPointToNetwork(graph, [80, 0], 10);
  const supplied = runNetworkShortestPaths(lines, origins, destinations, PARAMETERS, LIMITS, {
    graph,
    snappedDestinations: [forced, null]
  });
  expect(supplied.tables[0].columns['status']).toEqual(['ok', 'destination-not-snapped']);
  expect(supplied.tables[0].columns['total_cost'][0]).toBeCloseTo(80, 6);
  expect(cpu.tables[0].columns['total_cost'][0]).toBeCloseTo(60, 6);
  const originSupplied = runNetworkShortestPaths(lines, origins, destinations, PARAMETERS, LIMITS, {
    graph,
    snappedOrigin: null
  });
  expect(originSupplied.tables[0].columns['status'][0]).toBe('origin-not-snapped');
});
