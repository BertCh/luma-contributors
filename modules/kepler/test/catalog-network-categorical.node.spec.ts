// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  listCatalogEntries,
  normalizeCatalogParameters,
  validateCatalogInputs,
  runCatalogOperationAsync,
  serializeCatalogInvocation,
  parseCatalogInvocation,
  matchCatalogInvocationInputs
} from '../src/analysis/catalog-bindings';
import {NETWORK_CATEGORICAL_BINDINGS} from '../src/analysis/catalog-network-categorical-bindings';
import {completeNetworkCategoricalCatalogEntries} from '../src/analysis/catalog-network-categorical-discovery';
import {
  readCategories,
  readNumbers,
  readPositions
} from '../src/analysis/catalog-network-categorical-common';
import {prepareCategoryWeights} from '../src/analysis/catalog-network-categorical-categories';
import {
  prepareFlowMatrix,
  prepareTraversedEdges
} from '../src/analysis/catalog-network-categorical-flows';
import {
  prepareDatumSamples,
  prepareDatumPositions
} from '../src/analysis/catalog-network-categorical-datum';
import {getNetworkCategoricalSampleInputs} from '../../../examples/deck/kepler-compute/src/analysis-catalog-network-categorical-samples';
import {
  parseCatalogFeatureCollection,
  type CatalogFeatureCollection
} from '../src/analysis/catalog-geojson';

const EXPECTED_CONTRIBUTORS = [
  'GPUKernelFlowSmoothing',
  'GPUMapMatchedEdgeVolumes',
  'GPURadiationFlowModel',
  'GPUNetworkPotentialPathArea',
  'GPUNetworkTravelTimePercentiles',
  'GPUColocationQuotient',
  'GPULocalCategoryDependence',
  'GPUSpatialEntropy',
  'GPUCategoryPairCounts',
  'GPUGeographicalDetector',
  'GPURedcapSingleLinkage',
  'GPUHealpixPointToCell',
  'GPUHealpixCellAggregation',
  'GPUDatumGridTransform',
  'GPUSinkhornDivergence'
];

function makePoints(properties: Record<string, unknown>[]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: properties.map((property, index) => ({
      id: `source-${index}`,
      geometry: {type: 'Point', coordinates: [index, 0]},
      properties: property
    }))
  };
}

for (const binding of NETWORK_CATEGORICAL_BINDINGS) {
  test(`${binding.id}: registered, discoverable, typed, serializable and has a valid UI sample`, async () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.runGpu).toBeTypeOf('function');
    const samples = getNetworkCategoricalSampleInputs(binding.id)!;
    expect(Object.keys(samples).sort()).toEqual(binding.inputs.map(input => input.name).sort());
    const inputs = Object.fromEntries(
      Object.entries(samples).map(([name, text]) => [name, parseCatalogFeatureCollection(text)])
    );
    validateCatalogInputs(binding, inputs);
    const invocationText = serializeCatalogInvocation(binding, {}, inputs);
    const restored = parseCatalogInvocation(invocationText);
    expect(restored.binding).toBe(binding);
    expect(matchCatalogInvocationInputs(restored.invocation, inputs)).toBe(true);
    expect(normalizeCatalogParameters(binding, restored.invocation.parameters)).toEqual(
      restored.invocation.parameters
    );
    expect(() => normalizeCatalogParameters(binding, {unknownParameter: 1})).toThrow();
    for (const parameter of binding.parameters) {
      if (parameter.kind === 'number' || parameter.kind === 'integer')
        expect(() => normalizeCatalogParameters(binding, {[parameter.name]: Infinity})).toThrow();
    }
    await expect(runCatalogOperationAsync(binding, {}, inputs)).rejects.toThrow(
      'GPU device required'
    );
  });
}

test('all requested contributors have executable discovery entries without changing the shared registry', () => {
  const entries = completeNetworkCategoricalCatalogEntries(listCatalogEntries());
  expect(completeNetworkCategoricalCatalogEntries(entries)).toEqual(entries);
  expect(new Set(entries.map(entry => entry.contributorName)).size).toBe(entries.length);
  for (const name of EXPECTED_CONTRIBUTORS) {
    const bindings = NETWORK_CATEGORICAL_BINDINGS.filter(binding =>
      binding.gpuContributorNames.includes(name)
    );
    expect(bindings.length, name).toBeGreaterThan(0);
    for (const binding of bindings) expect(getCatalogBinding(binding.id)).toBe(binding);
    const entry = entries.find(candidate => candidate.contributorName === name);
    expect(entry, name).toBeDefined();
    expect(entry!.runnable, name).toBe('catalog-binding');
    expect(entry!.engine).toBe('gpu');
    expect(entry!.bindingIds.length).toBeGreaterThan(0);
  }
});

test('category encoding preserves source row alignment, missing labels and scalar types', () => {
  const encoded = readCategories(
    makePoints([
      {category: 1},
      {category: '1'},
      {category: null},
      {},
      {category: false},
      {category: 1}
    ]),
    'category'
  );
  expect(Array.from(encoded.categories)).toEqual([0, 1, 0xffffffff, 0xffffffff, 2, 0]);
  expect(encoded.labels).toEqual(['number:1', 'string:1', 'boolean:false']);
  expect(() => readCategories(makePoints([{}]), 'category')).toThrow('Category count');
  expect(() =>
    readCategories(
      makePoints(Array.from({length: 33}, (_, index) => ({category: index}))),
      'category'
    )
  ).toThrow('32');
});

test('symmetric distance bands retain islands and source rows without creating self links', () => {
  const weights = prepareCategoryWeights(makePoints([{}, {}, {}, {}]), 'planar', 1.1);
  expect(Array.from(weights.offsets)).toEqual([0, 1, 3, 5, 6]);
  expect(Array.from(weights.neighbors)).toEqual([1, 0, 2, 1, 3, 2]);
  expect(weights.weights[0]).toBeCloseTo(1 / (1 + 1 / 1.1), 6);
  const islands = prepareCategoryWeights(makePoints([{}, {}]), 'planar', 0.1);
  expect(Array.from(islands.offsets)).toEqual([0, 0, 0]);
});

test('numeric inputs fail before uploading invalid Float32 values and never coerce null to zero', () => {
  for (const value of [null, undefined, '2', NaN, Infinity, 1e100, -1])
    expect(() => readNumbers(makePoints([{value}]), 'value', true)).toThrow();
  const points = makePoints([{}]);
  points.features[0].geometry = {type: 'Point', coordinates: [0, 91]};
  expect(() => readPositions(points, true)).toThrow();
  points.features[0].geometry = {
    type: 'MultiPoint',
    coordinates: [
      [0, 0],
      [1, 1]
    ]
  };
  expect(() => readPositions(points)).toThrow();
});

test('flow preparation resolves source ids, fills absent pairs with zero and refuses duplicates/unknown zones', () => {
  const zones = makePoints([{}, {}]);
  const flows = makePoints([{origin: 'source-0', destination: 'source-1', weight: 5}]);
  expect(Array.from(prepareFlowMatrix(zones, flows, 'origin', 'destination', 'weight'))).toEqual([
    0, 5, 0, 0
  ]);
  flows.features.push(flows.features[0]);
  expect(() => prepareFlowMatrix(zones, flows, 'origin', 'destination', 'weight')).toThrow(
    'unique'
  );
  flows.features = [
    makePoints([{origin: 'missing', destination: 'source-1', weight: 5}]).features[0]
  ];
  expect(() => prepareFlowMatrix(zones, flows, 'origin', 'destination', 'weight')).toThrow(
    'endpoint'
  );
});

test('matched traversal preparation preserves repeated edges and sub-track breaks; rejects truncation', () => {
  const tracks = makePoints([{edges: [0, 1, null, 0, 0xffffffff, 1]}]);
  const packed = prepareTraversedEdges(tracks, 'edges', 2, 8);
  expect(Array.from(packed.edges)).toEqual([
    0, 1, 0xffffffff, 0, 0xffffffff, 1, 0xffffffff, 0xffffffff
  ]);
  expect(Array.from(packed.counts)).toEqual([6]);
  expect(() => prepareTraversedEdges(tracks, 'edges', 2, 5)).toThrow('capacity');
  expect(() => prepareTraversedEdges(makePoints([{edges: [2]}]), 'edges', 2, 8)).toThrow('outside');
});

test('decoded datum grid has explicit shape and nodata, without inventing shifts', () => {
  const samples = makePoints(
    Array.from({length: 4}, (_, index) =>
      index === 2 ? {valid: false} : {longitudeShift: 1, latitudeShift: 2, heightShift: 3}
    )
  );
  const packed = prepareDatumSamples(samples, 2, 2);
  expect(Array.from(packed.mask)).toEqual([1, 1, 0, 1]);
  expect(Array.from(packed.shifts.slice(6, 9))).toEqual([0, 0, 0]);
  expect(() => prepareDatumSamples(samples, 3, 2)).toThrow('width * height');
  samples.features[0].properties = {};
  expect(() => prepareDatumSamples(samples, 2, 2)).toThrow('longitudeShift');
});

test('3D GeoJSON packs XY for spatial indexing and exactly XYZ high/residual words for datum transforms', () => {
  const points = makePoints([{}]);
  points.features[0].geometry = {type: 'Point', coordinates: [1.00000001, 2.00000002, 100.000001]};
  expect(Array.from(readPositions(points))).toEqual([1, 2]);
  const packed = prepareDatumPositions(points, 'height', true);
  expect(packed.high.length).toBe(3);
  expect(packed.low.length).toBe(3);
  expect(packed.high[0] + packed.low[0]).toBeCloseTo(1.00000001, 13);
  expect(packed.high[2] + packed.low[2]).toBeCloseTo(100.000001, 11);
  points.features[0].properties = {height: 10};
  expect(prepareDatumPositions(points, 'height', true).high[2]).toBe(10);
});
