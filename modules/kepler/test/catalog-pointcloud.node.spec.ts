// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';
import {POINT_CLOUD_BINDINGS} from '../src/analysis/catalog-pointcloud-bindings';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  normalizeCatalogParameters,
  serializeCatalogInvocation,
  parseCatalogInvocation,
  matchCatalogInvocationInputs,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import {
  makePointCloudCatalogGrid,
  preparePointCloudInputs,
  validatePointCloudRadii,
  makePointCloudTable
} from '../src/analysis/catalog-pointcloud-input';
import {
  findPointCloudCatalogFields,
  readPointCloudCatalogDataset,
  listPointCloudCatalogSources,
  getPointCloudCatalogSamples,
  getPointCloudCatalogCoordinates
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-pointcloud';
import {
  findNativeGeometryFields,
  readCatalogFeaturesFromNativeDataset,
  getCatalogSampleInputs
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {makeArrowTableFromCatalogTable} from '../../../examples/deck/kepler-compute/src/analysis-catalog-table';
import {readPointCloudFeatureValue} from '../src/analysis/catalog-pointcloud-feature-columns';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

function points(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, row) => ({
      id: `source-${row}`,
      geometry: {type: 'Point', coordinates: position},
      properties: {classification: 2, intensity: row * 10}
    }))
  };
}
const parameters = {coordinateSystem: 'planar'};

for (const binding of POINT_CLOUD_BINDINGS) {
  test(`${binding.id}: registry, ontology and serialization agree`, async () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(binding.gpuContributorNames.length).toBeGreaterThan(0);
    for (const contributor of binding.contributorNames) {
      const entry = listCatalogEntries().find(
        candidate => candidate.contributorName === contributor
      )!;
      expect(entry.runnable).toBe('catalog-binding');
      expect(entry.bindingIds).toContain(binding.id);
      expect(entry.engine).toBe('gpu');
    }
    const samples = getCatalogSampleInputs(binding.id);
    const inputs = Object.fromEntries(
      binding.inputs.map(input => [input.name, JSON.parse(samples[input.name])])
    );
    const invocation = serializeCatalogInvocation(binding, {}, inputs);
    const restored = parseCatalogInvocation(invocation);
    expect(restored.invocation.parameters).toEqual(getCatalogDefaultParameters(binding));
    expect(matchCatalogInvocationInputs(restored.invocation, inputs)).toBe(true);
    inputs[binding.inputs[0].name].features[0].geometry.coordinates[2] += 1;
    expect(matchCatalogInvocationInputs(restored.invocation, inputs)).toBe(false);
    await expect(runCatalogOperationAsync(binding, {}, inputs)).rejects.toThrow(
      'GPU device required'
    );
  });
}

test('all eleven contributors have executable bindings and no public helper is claimed as an operation', () => {
  const contributors = new Set(POINT_CLOUD_BINDINGS.flatMap(binding => binding.contributorNames));
  expect(contributors.size).toBe(11);
  expect([...contributors]).toContain('GPUPointCloudHeight');
  expect([...contributors]).toContain('GPUPointCloudVoxelMerge');
  expect(POINT_CLOUD_BINDINGS).toHaveLength(11);
});

test('shared local origin preserves projected millimeters and epoch separation', () => {
  const prepared = preparePointCloudInputs(
    {
      reference: points([
        [1000000, 4000000, 100],
        [1000000.001, 4000000, 100]
      ]),
      comparison: points([[1000000, 4000000, 100.25]])
    },
    parameters
  );
  expect(
    prepared.clouds['reference'].positions[3] - prepared.clouds['reference'].positions[0]
  ).toBeCloseTo(0.001, 7);
  expect(
    prepared.clouds['comparison'].positions[2] - prepared.clouds['reference'].positions[2]
  ).toBeCloseTo(0.25, 7);
});

test('geographic inputs share a metric frame and preserve physical altitude', () => {
  const prepared = preparePointCloudInputs(
    {
      points: points([
        [-87, 41, 100],
        [-86.99999, 41, 100.2]
      ])
    },
    {coordinateSystem: 'EPSG:4326'}
  );
  expect(
    prepared.clouds['points'].positions[3] - prepared.clouds['points'].positions[0]
  ).toBeGreaterThan(0.8);
  expect(
    prepared.clouds['points'].positions[3] - prepared.clouds['points'].positions[0]
  ).toBeLessThan(0.9);
  const table = makePointCloudTable(prepared.clouds['points'], 'features', {
    feature_status: [0, 0]
  });
  expect(table.columns['altitude']).toEqual([100, 100.2]);
  expect(table.columns['source_id']).toEqual(['source-0', 'source-1']);
  expect(table.columns['intensity']).toEqual([0, 10]);
  const output = makeArrowTableFromCatalogTable(table);
  expect(output.getChild('altitude')?.get(1)).toBe(100.2);
  expect(getPointCloudCatalogCoordinates(output, table.label)).toEqual({
    longitude: 'longitude',
    latitude: 'latitude',
    altitude: 'altitude'
  });
});

for (const coordinates of [[[0, 0]], [[0, 0, Number.NaN]], [[0, 0, Infinity]], [[181, 0, 0]]])
  test(`unusable XYZ refused: ${JSON.stringify(coordinates)}`, () => {
    expect(() =>
      preparePointCloudInputs({points: points(coordinates)}, {coordinateSystem: 'EPSG:4326'})
    ).toThrow(/catalog-pointcloud/);
  });

test('capacity, classifications and morphology parameters refuse before allocation', () => {
  expect(() =>
    preparePointCloudInputs(
      {points: points(Array.from({length: 2049}, () => [0, 0, 0]))},
      parameters
    )
  ).toThrow(/rows/);
  const invalid = points([[0, 0, 0]]);
  invalid.features[0].properties!['classification'] = -1;
  expect(() =>
    preparePointCloudInputs(
      {points: invalid},
      {...parameters, classificationProperty: 'classification'}
    )
  ).toThrow(/classification/);
  expect(() =>
    preparePointCloudInputs(
      {
        points: {
          type: 'FeatureCollection',
          features: [{geometry: {type: 'MultiPoint', coordinates: [[0, 0, 0]]}}]
        }
      },
      parameters
    )
  ).toThrow(/XYZ/);
  expect(() => validatePointCloudRadii([2, 1])).toThrow(/radii/);
  expect(() => validatePointCloudRadii([1, 1])).toThrow(/radii/);
  expect(() => validatePointCloudRadii([1.5], true)).toThrow(/integer/);
  expect(() =>
    normalizeCatalogParameters(
      POINT_CLOUD_BINDINGS.find(binding => binding.id === 'pointcloud-features')!,
      {capacity: 65}
    )
  ).toThrow();
  const cloud = preparePointCloudInputs(
    {
      points: points([
        [0, 0, 0],
        [100, 100, 0]
      ])
    },
    parameters
  ).clouds['points'];
  expect(() => makePointCloudCatalogGrid(cloud, 1)).toThrow(/cellSize/);
  expect(makePointCloudCatalogGrid(cloud, 2).rowDirection).toBe('south');
});

test('XYZ role discovery works for native CSV names and retained Arrow batches', () => {
  const table = arrow.tableFromArrays({
    X: [1, 2],
    Y: [3, 4],
    Z: [5, 6],
    classification: [2, 1],
    intensity: [100, 200]
  });
  const publication = {label: 'LAS-like data', dataset: {id: 'las', value: {source: table}}};
  const sources = listPointCloudCatalogSources([publication]);
  expect(sources).toHaveLength(1);
  const field = findNativeGeometryFields(sources[0])[0];
  expect(field).toContain('XYZ columns:');
  expect(readCatalogFeaturesFromNativeDataset(sources[0], field).features[1]).toEqual({
    id: 1,
    geometry: {type: 'Point', coordinates: [2, 4, 6]},
    properties: {X: 2, Y: 4, Z: 6, classification: 1, intensity: 200}
  });
  const geographic = {
    fields: [{name: 'lon'}, {name: 'lat'}, {name: 'alt'}],
    dataContainer: {
      numRows: () => 1,
      valueAt: (_row: number, column: number) => ['-87', '41', '0'][column]
    }
  };
  expect(
    readPointCloudCatalogDataset(geographic, findPointCloudCatalogFields(geographic)[0])!
      .features[0].geometry?.coordinates
  ).toEqual([-87, 41, 0]);
  expect(
    listPointCloudCatalogSources([
      {...publication, dataset: {...publication.dataset, population: {state: 'open'}}},
      {
        ...publication,
        dataset: {...publication.dataset, population: {state: 'complete', partial: true}}
      },
      {...publication, dataset: {...publication.dataset, value: {source: table, resident: true}}}
    ])
  ).toEqual([]);
});

test('missing and empty XYZ cells never coerce to zero; role mapping survives serialization', () => {
  const dataset = {
    fields: [{name: 'x'}, {name: 'y'}, {name: 'z'}],
    dataContainer: {
      numRows: () => 1,
      valueAt: (_row: number, column: number) => [0, 0, null][column]
    }
  };
  const field = findPointCloudCatalogFields(dataset)[0];
  expect(() => readPointCloudCatalogDataset(dataset, field)).toThrow(/finite/);
  expect(() => readPointCloudCatalogDataset(dataset, 'XYZ columns: ["x","y","missing"]')).toThrow(
    /exist/
  );
  expect(getPointCloudCatalogSamples('network-service-area')).toEqual({});
});

test('alignment and voxel ownership contracts refuse before any GPU allocation', async () => {
  const device = {
    createBuffer: () => {
      throw new Error('Unexpected GPU allocation');
    }
  } as unknown as Device;
  const cloud = points([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
  ]);
  await expect(
    runCatalogOperationAsync(
      getCatalogBinding('pointcloud-change')!,
      parameters,
      {reference: cloud, comparison: cloud},
      {device}
    )
  ).rejects.toThrow(/alreadyAligned/);
  await expect(
    runCatalogOperationAsync(
      getCatalogBinding('pointcloud-voxel-merge')!,
      parameters,
      {left: cloud, right: cloud},
      {device}
    )
  ).rejects.toThrow(/disjointOwnedRows/);
});

test('published aligned coordinates remain discoverable ahead of retained source aliases', () => {
  const input = points([[0, 0, 0]]);
  input.features[0].properties = {
    x: 0,
    y: 0,
    z: 0,
    lon: 0,
    lat: 0,
    Longitude: -1,
    Latitude: -1,
    Altitude: -1
  };
  const cloud = preparePointCloudInputs({points: input}, parameters).clouds['points'];
  const output = makePointCloudTable({...cloud, coordinates: [[10, 20, 30]]}, 'icp');
  const table = makeArrowTableFromCatalogTable(output);
  const [source] = listPointCloudCatalogSources([
    {label: output.label, dataset: {id: 'aligned', value: {source: table}}}
  ]);
  const fields = findNativeGeometryFields(source);
  expect(fields[0]).toBe('XYZ columns: ["longitude","latitude","altitude"]');
  expect(
    readCatalogFeaturesFromNativeDataset(source, fields[0]).features[0].geometry?.coordinates
  ).toEqual([10, 20, 30]);
  expect(fields).toContain('XYZ columns: ["x","y","z"]');
  expect(
    readCatalogFeaturesFromNativeDataset(source, 'XYZ columns: ["x","y","z"]').features[0].geometry
      ?.coordinates
  ).toEqual([0, 0, 0]);
  const reordered = {...source, fields: [...source.fields].reverse()};
  expect(findPointCloudCatalogFields(reordered)[0]).toBe(fields[0]);
});

test('voxel origin uses positive counts across both inputs and preserves empty slots', () => {
  const left = points([
    [1000000.001, 4000000.001, 100.001],
    [0, 0, 0],
    [0, 0, 0]
  ]);
  const right = points([
    [1000000.002, 4000000.002, 100.002],
    [1000000.003, 4000000.003, 100.003],
    [0, 0, 0]
  ]);
  left.features.forEach((feature, row) => {
    feature.properties!['members'] = row === 0 ? 3 : 0;
  });
  right.features.forEach((feature, row) => {
    feature.properties!['members'] = row < 2 ? 1 : 0;
  });
  const prepared = preparePointCloudInputs(
    {left, right},
    {...parameters, countProperty: 'members', disjointOwnedRows: true}
  );
  expect(prepared.origin[0]).toBeCloseTo(1000000.002, 8);
  expect(prepared.origin[1]).toBeCloseTo(4000000.002, 8);
  expect(prepared.origin[2]).toBeCloseTo(100.002, 8);
  expect(prepared.clouds['right'].positions[0] - prepared.clouds['left'].positions[0]).toBeCloseTo(
    0.001,
    8
  );
  expect(Array.from(prepared.clouds['left'].positions.slice(3))).toEqual(Array(6).fill(0));
  expect(Array.from(prepared.clouds['right'].positions.slice(6))).toEqual([0, 0, 0]);
  expect(Array.from(prepared.clouds['right'].voxelCounts!)).toEqual([1, 1, 0]);
  expect(prepared.clouds['left'].positions.length).toBe(9);
  expect(left.features[1].geometry?.coordinates).toEqual([0, 0, 0]);
});

test('all-empty voxels use a harmless zero frame and invalid counts refuse before packing', () => {
  const input = points([[1e100, -1e100, 1e100]]);
  input.features[0].properties!['count'] = 0;
  const settings = {...parameters, countProperty: 'count', disjointOwnedRows: true};
  const prepared = preparePointCloudInputs({left: input, right: input}, settings);
  expect(prepared.origin).toEqual([0, 0, 0]);
  expect(Array.from(prepared.clouds['left'].positions)).toEqual([0, 0, 0]);
  input.features[0].properties!['count'] = -1;
  expect(() => preparePointCloudInputs({left: input, right: input}, settings)).toThrow(
    /left.counts must be uint32/
  );
});

test('feature publication distinguishes rank, support, convergence and moment validity', () => {
  expect(readPointCloudFeatureValue(0.5, 'eigenvalue', 1, 3)).toBe(0.5);
  expect(readPointCloudFeatureValue(0.5, 'covariance', 1, 3)).toBe(0.5);
  expect(readPointCloudFeatureValue(0, 'normal', 1, 3)).toBeNull();
  expect(readPointCloudFeatureValue(0, 'shape', 1, 3)).toBeNull();
  expect(readPointCloudFeatureValue(0.5, 'covariance', 4, 3)).toBe(0.5);
  expect(readPointCloudFeatureValue(0.5, 'eigenvalue', 4, 3)).toBeNull();
  for (const kind of ['normal', 'shape', 'eigenvalue', 'covariance'] as const) {
    expect(readPointCloudFeatureValue(0, kind, 1, 2)).toBeNull();
    expect(readPointCloudFeatureValue(0.5, kind, 2, 3)).toBe(0.5);
    expect(readPointCloudFeatureValue(Number.NaN, kind, 0, 3)).toBeNull();
    expect(readPointCloudFeatureValue(Infinity, kind, 0, 3)).toBeNull();
  }
});
