// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

const lines: CatalogFeatureCollection = {
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
      properties: {}
    }
  ]
};
const origins: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}]
};
const opportunities = (weights: unknown[]): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features: weights.map(weight => ({
    geometry: {type: 'Point', coordinates: [5, 0]},
    properties: {weight}
  }))
});
const parameters = {coordinateSystem: 'planar', weightProperty: 'weight', maximumCost: 10};
const binding = getCatalogBinding('network-accessibility')!;

test('missing/non-numeric weights do not count as reachable opportunities and are reported once', () => {
  const result = runCatalogOperation(binding, parameters, {
    lines,
    origins,
    opportunities: opportunities([5, undefined, 'abc', null, '', '  ', 0, '2'])
  });
  expect(result.tables[0].columns['reachable_count']).toEqual([3]);
  expect(result.tables[0].columns['accessibility']).toEqual([7]);
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({
      code: 'opportunity-weight-missing',
      severity: 'warning',
      count: 5
    })
  );
});

test('negative weight from the evidence is rejected instead of subtracting from accessibility', () => {
  expect(() =>
    runCatalogOperation(binding, parameters, {
      lines,
      origins,
      opportunities: opportunities([5, undefined, 'abc', -4])
    })
  ).toThrow(/^network-accessibility.*weightProperty.*weight.*at least 0/);
});

test('weights are validated before routing even for unsnapped opportunities', () => {
  const missing = opportunities([-4]);
  missing.features[0].geometry = {type: 'Point', coordinates: [900, 900]};
  expect(() =>
    runCatalogOperation(binding, parameters, {
      lines,
      origins,
      opportunities: missing
    })
  ).toThrow(/^network-accessibility.*at least 0/);
});

test('no weight column retains unweighted counting; missing weights do not set nearest_cost', () => {
  const unweighted = runCatalogOperation(
    binding,
    {...parameters, weightProperty: ''},
    {
      lines,
      origins,
      opportunities: opportunities([undefined, 'abc'])
    }
  );
  expect(unweighted.tables[0].columns['reachable_count']).toEqual([2]);
  const weighted = runCatalogOperation(binding, parameters, {
    lines,
    origins,
    opportunities: opportunities([null])
  });
  expect(weighted.tables[0].columns['reachable_count']).toEqual([0]);
  expect(weighted.tables[0].columns['nearest_cost']).toEqual([null]);
});
