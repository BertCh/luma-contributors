// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {buildNetworkGraph, type CatalogNetworkParameters} from '../src/analysis/catalog-network';
import {
  prorateNodedProperties,
  type CatalogNodedNetwork
} from '../src/analysis/catalog-gpu-network';

const parameters: CatalogNetworkParameters = {
  coordinateSystem: 'planar',
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'property',
  forwardCostProperty: 'cost',
  reverseCostProperty: '',
  onewayProperty: ''
};

function lines(properties: Record<string, unknown>): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [10, 0]
          ]
        },
        properties
      }
    ]
  };
}

const points = (coordinates: number[]): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features: [{geometry: {type: 'Point', coordinates}, properties: {}}]
});

test.each([
  undefined,
  null,
  '',
  '  ',
  'abc',
  Infinity,
  false
])('missing/non-numeric property cost %s removes both arcs and reports their count', cost => {
  const graph = buildNetworkGraph(lines({cost}), parameters, 10);
  expect(graph.outgoing.flat()).toHaveLength(0);
  expect(graph.edges[0].forwardCost).toBeNull();
  expect(graph.edges[0].reverseCost).toBeNull();
  expect(graph.diagnostics).toContainEqual(
    expect.objectContaining({
      code: 'invalid-cost-arc-removed',
      severity: 'warning',
      count: 2
    })
  );
  const result = runCatalogOperation(getCatalogBinding('network-shortest-path')!, parameters, {
    lines: lines({cost}),
    origin: points([0, 0]),
    destinations: points([10, 0])
  });
  expect(result.tables[0].columns['total_cost']).toEqual([null]);
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({count: 2, code: 'invalid-cost-arc-removed'})
  );
});

test('a missing reverse cost removes only the reverse arc; explicit zero and numeric strings are valid', () => {
  const graph = buildNetworkGraph(
    lines({cost: '0', reverse: ''}),
    {...parameters, reverseCostProperty: 'reverse'},
    10
  );
  expect(graph.edges[0].forwardCost).toBe(0);
  expect(graph.edges[0].reverseCost).toBeNull();
  expect(graph.diagnostics[0].count).toBe(1);
  expect(buildNetworkGraph(lines({cost: '7'}), parameters, 10).edges[0].forwardCost).toBe(7);
});

test('negative costs fail ingestion with the binding, property and non-negative limit', () => {
  expect(() =>
    runCatalogOperation(getCatalogBinding('network-shortest-path')!, parameters, {
      lines: lines({cost: -4}),
      origin: points([0, 0]),
      destinations: points([10, 0])
    })
  ).toThrow(/^network-shortest-path.*cost.*at least 0/);
});

test.each([
  'forwardCostProperty',
  'reverseCostProperty'
])('length mode visibly warns when %s is named', name => {
  const result = runCatalogOperation(
    getCatalogBinding('network-build-graph')!,
    {
      coordinateSystem: 'planar',
      costMode: 'length',
      [name]: 'travelTime'
    },
    {lines: lines({travelTime: 123})}
  );
  expect(result.tables[0].columns['forward_cost']).toEqual([10]);
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({
      code: 'cost-properties-ignored',
      severity: 'warning',
      message: expect.stringContaining('costMode=property')
    })
  );
});

test.each([
  null,
  '',
  'abc',
  -4,
  8
])('GPU-free noded property proration preserves missing/negative cost %s for ingestion', cost => {
  const source = lines({cost});
  const pieces = lines({cost});
  pieces.features[0].geometry = {
    type: 'LineString',
    coordinates: [
      [0, 0],
      [5, 0]
    ]
  };
  const noded = {pieces, pieceSourceIndices: [0], piecePartIndices: [0]} as CatalogNodedNetwork;
  prorateNodedProperties(
    noded,
    source,
    {...parameters, reverseCostProperty: 'cost'},
    positions => positions[positions.length - 1][0] - positions[0][0]
  );
  // Identical forward/reverse column names must be prorated once.
  expect(pieces.features[0].properties?.['cost']).toBe(cost === 8 ? 4 : cost);
  if (cost === -4) expect(() => buildNetworkGraph(pieces, parameters, 10)).toThrow(/at least 0/);
  else
    expect(buildNetworkGraph(pieces, parameters, 10).edges[0].forwardCost).toBe(
      cost === 8 ? 4 : null
    );
});
