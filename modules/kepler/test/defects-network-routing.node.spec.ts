// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import {MAXIMUM_NETWORK_EDGES} from '../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {
  buildRoutingGraph,
  MAXIMUM_ROUTING_EDGES
} from '../src/analysis/catalog-platform3-a-network';

const collection = (coordinates: number[][][]): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features: coordinates.map(positions => ({
    geometry: {type: 'LineString', coordinates: positions},
    properties: {cost: 1000}
  }))
});
const values = {
  coordinateSystem: 'planar',
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'length',
  forwardCostProperty: '',
  reverseCostProperty: '',
  onewayProperty: ''
};

test('the 210-line/21,840-edge evidence grid fits the declared 50,000-edge budget', () => {
  const horizontal = Array.from({length: 105}, (_, row) =>
    Array.from({length: 105}, (_, column) => [column, row])
  );
  const vertical = Array.from({length: 105}, (_, column) =>
    Array.from({length: 105}, (_, row) => [column, row])
  );
  const graph = buildRoutingGraph(
    collection([...horizontal, ...vertical]),
    values,
    'network-subgraph-filter'
  );
  expect(MAXIMUM_ROUTING_EDGES).toBe(MAXIMUM_NETWORK_EDGES);
  expect(graph.edges).toHaveLength(21840);
});

test('splitAtSharedVertices changes the binding graph instead of being forced on', () => {
  const lines = collection([
    [
      [0, 0],
      [10, 0],
      [20, 0]
    ],
    [
      [10, 0],
      [10, 10]
    ]
  ]);
  const binding = getCatalogBinding('network-subgraph-filter')!;
  expect(
    runCatalogOperation(binding, {...values, splitAtSharedVertices: true}, {lines}).tables[0]
      .rowCount
  ).toBe(3);
  expect(
    runCatalogOperation(binding, {...values, splitAtSharedVertices: false}, {lines}).tables[0]
      .rowCount
  ).toBe(2);
});

test('routing cap names the binding, edge limit and remedy', () => {
  const coordinates = Array.from({length: 25002}, (_, index) => [index, 0]);
  expect(() =>
    runCatalogOperation(getCatalogBinding('network-subgraph-filter')!, values, {
      lines: collection([coordinates, coordinates])
    })
  ).toThrow(/^network-subgraph-filter.*edgeCount.*reduce lines.*50000/);
});

test.each([
  1000, 10
])('Network K rejects property costs %s rather than comparing mismatched units', cost => {
  const lines = collection([
    [
      [0, 0],
      [1000, 0]
    ]
  ]);
  lines.features[0].properties = {cost};
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [0, 500, 1000].map(position => ({
      geometry: {type: 'Point', coordinates: [position, 0]},
      properties: {}
    }))
  };
  const binding = getCatalogBinding('network-k-function')!;
  expect(() =>
    runCatalogOperation(
      binding,
      {
        ...values,
        costMode: 'property',
        forwardCostProperty: 'cost',
        radii: [cost * 0.6]
      },
      {lines, points}
    )
  ).toThrow(/^network-k-function.*costMode.*length.*property-cost radii/);
  const result = runCatalogOperation(binding, {...values, radii: [600]}, {lines, points});
  expect(result.tables[0].columns['pair_count']).toEqual([4]);
  expect(result.tables[0].columns['k'][0]).toBeCloseTo(2000 / 3, 9);
  expect(result.tables[0].columns['expected_k']).toEqual([1200]);
});
