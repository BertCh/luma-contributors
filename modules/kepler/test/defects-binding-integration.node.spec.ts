// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {expect, test, vi} from 'vitest';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperation,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import {getNetworkParameters} from '../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {
  deriveKeplerOperationDescriptor,
  getKeplerOperationDescriptor
} from '../src/descriptors/operation-descriptors';

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
      properties: {cost: -1}
    }
  ]
};
const points: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [{geometry: {type: 'Point', coordinates: [0, 0]}}]
};

// Saved workflows can retain removed controls; they must not change topology or fail validation.
test.each([
  'network-structure',
  'network-line-graph',
  'network-snap-points'
])('%s ignores obsolete cost and one-way parameters in saved workflows', async bindingId => {
  const binding = getCatalogBinding(bindingId)!;
  const legacy = {
    coordinateSystem: 'planar',
    costMode: 'property',
    forwardCostProperty: 'cost',
    reverseCostProperty: 'cost',
    onewayProperty: 'cost'
  };
  const parameters = normalizeCatalogParameters(binding, legacy);
  expect(Object.keys(parameters).sort()).toEqual(
    (bindingId === 'network-snap-points'
      ? [
          'coordinateSystem',
          'crossings',
          'maximumSnapDistance',
          'snapTolerance',
          'splitAtSharedVertices'
        ]
      : ['coordinateSystem', 'crossings', 'snapTolerance', 'splitAtSharedVertices']
    ).sort()
  );
  expect(getNetworkParameters(parameters)).toEqual({
    coordinateSystem: 'planar',
    snapTolerance: 0,
    splitAtSharedVertices: true,
    costMode: 'length',
    forwardCostProperty: '',
    reverseCostProperty: '',
    onewayProperty: ''
  });
  const inputs = bindingId === 'network-snap-points' ? {lines, points} : {lines};
  const baseline = runCatalogOperation(binding, {coordinateSystem: 'planar'}, inputs);
  expect(runCatalogOperation(binding, legacy, inputs)).toEqual(baseline);
  expect((await runCatalogOperationAsync(binding, legacy, inputs)).tables).toEqual(baseline.tables);
  expect(() => normalizeCatalogParameters(binding, {unknownControl: 1})).toThrow(
    `${bindingId}: unknownControl is an unknown parameter`
  );
});

test.each([
  'geometry-measures',
  'network-shortest-path'
])('%s validates coordinateSystem before either engine can run', async bindingId => {
  const projected: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'LineString',
          coordinates: [
            [390000, 5820000],
            [390010, 5820000]
          ]
        }
      }
    ]
  };
  const binding = getCatalogBinding(bindingId)!;
  const inputs =
    bindingId === 'geometry-measures'
      ? {features: projected}
      : {lines: projected, origin: points, destinations: points};
  const runGpu = vi.fn(binding.runGpu);
  expect(() => runCatalogOperation(binding, {}, inputs)).toThrow(`${bindingId}: coordinateSystem`);
  await expect(
    runCatalogOperationAsync({...binding, runGpu}, {}, inputs, {device: {} as Device})
  ).rejects.toThrow(`${bindingId}: coordinateSystem`);
  expect(runGpu).not.toHaveBeenCalled();
  expect(() => runCatalogOperation(binding, {coordinateSystem: 'planar'}, inputs)).not.toThrow();
});

test('segmentize preflights the exact 500,000-vertex cap before CPU or GPU work', async () => {
  const binding = getCatalogBinding('geometry-line-segmentize')!;
  const runGpu = vi.fn(binding.runGpu);
  const parameters = {coordinateSystem: 'planar', maximumSegmentLength: 0.00002};
  const message =
    'geometry-line-segmentize: maximumSegmentLength projects 500001 output vertices, exceeding the 500,000 limit';
  expect(() => runCatalogOperation(binding, parameters, {lines})).toThrow(message);
  await expect(
    runCatalogOperationAsync({...binding, runGpu}, parameters, {lines}, {device: {} as Device})
  ).rejects.toThrow(message);
  expect(runGpu).not.toHaveBeenCalled();
});

test('buffer selection declares mixed geometry and preserves point, line and polygon rows', () => {
  const binding = getCatalogBinding('buffer-selection')!;
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      points.features[0],
      lines.features[0],
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [-1, -1],
              [1, -1],
              [1, 1],
              [-1, 1],
              [-1, -1]
            ]
          ]
        }
      }
    ]
  };
  const result = runCatalogOperation(
    binding,
    {coordinateSystem: 'planar', distance: 0},
    {selectors: points, features}
  );
  expect(binding.outputs[0].geometry).toBe('mixed');
  expect(result.tables[0].geometryColumn).toBe('geometry');
  expect(result.tables[0].columns['geometry'].map(value => JSON.parse(value as string))).toEqual(
    features.features.map(feature => feature.geometry)
  );
  expect(result.tables[0].columns['distance']).toEqual([0, 0, 0]);
  expect(getKeplerOperationDescriptor(binding.id)!.outputs[0].kindFrom).toBe('features');
  expect(deriveKeplerOperationDescriptor(binding).outputs[0].kind).toBe('render-geometry');
});

test('snapping reports its own binding id when the built graph exceeds 50,000 edges', () => {
  const coordinates = Array.from({length: 25002}, (_, index) => [index, 0]);
  const oversized: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [0, 1].map(() => ({geometry: {type: 'LineString', coordinates}}))
  };
  expect(() =>
    runCatalogOperation(
      getCatalogBinding('network-snap-points')!,
      {coordinateSystem: 'planar'},
      {lines: oversized, points}
    )
  ).toThrow(/^network-snap-points.*edgeCount.*50000/);
});
