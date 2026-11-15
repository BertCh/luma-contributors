// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  getCatalogResidentDatasetLabel,
  getCatalogResidentFields,
  getCatalogResidentGeometryFields,
  makeCatalogResidentDatasets,
  readCatalogFeaturesFromResidentDataset,
  type CatalogResidentDataset
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-resident-input';
import {readCatalogNativeInput} from '../../../examples/deck/kepler-compute/src/analysis-catalog-native-input';
import {
  makeCatalogTrajectoryNativeSource,
  TRAJECTORY_POINT_FIELD
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-trajectory';
import {makeCatalogPointColumns} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {withOriginalSourceTable} from '../../../examples/deck/kepler-compute/src/resident-table-provider';
import type {ResidentTableProvider} from '../../../examples/deck/kepler-compute/src/resident-data-table';
import {readSpacetimePoints} from '../src/analysis/catalog-spacetime-common';

function makeDataset(): CatalogResidentDataset {
  const columns: Record<string, unknown[]> = {
    longitude: [-87.65, -87.64],
    latitude: [41.88, 41.89],
    event_time: [1700000000000n, 1700000000001n],
    value: [4, 8],
    location: ['a', 'b']
  };
  return {
    id: 'resident-fixture',
    label: 'spacetime-observations.csv',
    rowCount: 2,
    provider: {
      fields: Object.keys(columns).map(name => ({name, sortable: true})),
      getRevision: () => 'revision-1',
      maximumPageSize: 1,
      readPage: vi.fn(async () => {
        throw new Error('Unexpected original-view read');
      }),
      readColumnValues: vi.fn(async ({column}) => ({
        values: columns[column],
        fieldType: column === 'location' ? 'string' : 'real'
      }))
    },
    isCurrent: () => true
  };
}

const MAPPING = makeCatalogPointColumns('longitude', 'latitude');

test('resident input uses bounded original column readers, preserving exact epoch numbers and row identities', async () => {
  const dataset = makeDataset();
  const points = await readCatalogFeaturesFromResidentDataset(dataset, MAPPING, 512);
  expect(dataset.provider.readColumnValues).toHaveBeenCalledTimes(5);
  expect(dataset.provider.readColumnValues).toHaveBeenCalledWith({
    column: 'event_time',
    population: 'all'
  });
  expect(points.features[1]).toEqual({
    id: 1,
    geometry: {type: 'Point', coordinates: [-87.64, 41.89]},
    properties: {
      longitude: -87.64,
      latitude: 41.89,
      event_time: 1700000000001,
      value: 8,
      location: 'b'
    }
  });
  const mapped = readSpacetimePoints(points, {
    coordinateSystem: 'EPSG:4326',
    timeProperty: 'event_time',
    timeUnit: 'milliseconds'
  });
  expect(mapped.timeOrigin).toBe(1700000000000);
  expect(mapped.times[1]).toBeCloseTo(0.001, 9);
});

test('resident input works through the real original-source provider and avoids prepared Float32 epoch rounding', async () => {
  const source = new arrow.Table({
    longitude: arrow.vectorFromArray([-87.65, -87.64], new arrow.Float64()),
    latitude: arrow.vectorFromArray([41.88, 41.89], new arrow.Float64()),
    event_time: arrow.vectorFromArray([1700000000000n, 1700000000001n], new arrow.Int64()),
    value: arrow.vectorFromArray([4, 8], new arrow.Float64())
  });
  const prepared: ResidentTableProvider = {
    datasetId: 'resident-fixture',
    sourceRevision: 1,
    fields: source.schema.fields.map(field => ({name: field.name, sortable: true})),
    maximumPageSize: 1,
    getRevision: () => 'revision-1',
    subscribe: () => () => {},
    readPage: vi.fn(async () => {
      throw new Error('Prepared scalar values must not be used');
    })
  };
  const ensureRows = vi.fn(async () => {});
  const provider = withOriginalSourceTable(prepared, source, () => true, {ensureRows});
  const points = await readCatalogFeaturesFromResidentDataset(
    {...makeDataset(), provider},
    MAPPING
  );
  expect(points.features.map(feature => feature.properties!['event_time'])).toEqual([
    1700000000000, 1700000000001
  ]);
  expect(ensureRows).toHaveBeenCalledWith([0]);
  expect(ensureRows).toHaveBeenCalledWith([1]);
  expect(prepared.readPage).not.toHaveBeenCalled();
});

test('resident capacity is checked before readback against both the catalog and operation bounds', async () => {
  const dataset = makeDataset();
  await expect(
    readCatalogFeaturesFromResidentDataset({...dataset, rowCount: 20001}, MAPPING)
  ).rejects.toThrow(/20000 rows/);
  await expect(
    readCatalogFeaturesFromResidentDataset({...dataset, rowCount: 513}, MAPPING, 512)
  ).rejects.toThrow(/512 rows/);
  expect(dataset.provider.readColumnValues).not.toHaveBeenCalled();
});

test('resident reads reject revisions or source ownership changing while columns are being read', async () => {
  const dataset = makeDataset();
  let revision = 'revision-1';
  dataset.provider.getRevision = () => revision;
  dataset.provider.readColumnValues = vi.fn(async () => {
    revision = 'revision-2';
    return {values: [1, 2], fieldType: 'real'};
  });
  await expect(readCatalogFeaturesFromResidentDataset(dataset, MAPPING)).rejects.toThrow(
    /changed during/
  );
  const removed = makeDataset();
  removed.isCurrent = () => false;
  await expect(readCatalogFeaturesFromResidentDataset(removed, MAPPING)).rejects.toThrow(
    /changed during/
  );
  expect(removed.provider.readColumnValues).not.toHaveBeenCalled();
});

test('incomplete, superseded and unavailable resident column readers fail without partial results', async () => {
  for (const result of [null, {values: [1], fieldType: 'real'}]) {
    const dataset = makeDataset();
    dataset.provider.readColumnValues = vi.fn(async () => result);
    await expect(readCatalogFeaturesFromResidentDataset(dataset, MAPPING)).rejects.toThrow(
      /incomplete or stale/
    );
  }
  const dataset = makeDataset();
  delete dataset.provider.readColumnValues;
  await expect(readCatalogFeaturesFromResidentDataset(dataset, MAPPING)).rejects.toThrow(
    /no original column readback/
  );
});

test('resident geometry metadata and labels are available without touching the native row facade', async () => {
  const dataset = makeDataset();
  expect(getCatalogResidentDatasetLabel({...dataset, rowCount: 48})).toBe(
    'spacetime-observations.csv (resident, 48 rows)'
  );
  expect(getCatalogResidentGeometryFields(dataset)).toEqual([makeCatalogPointColumns('', '')]);
  dataset.geometryColumn = 'geometry';
  dataset.provider.originalFields = [
    {name: 'geometry', sortable: false},
    {name: 'value', sortable: true}
  ];
  dataset.provider.readColumnValues = vi.fn(async ({column}) => ({
    values:
      column === 'geometry'
        ? ['{"type":"Point","coordinates":[1,2]}', '{"type":"Point","coordinates":[3,4]}']
        : [4, 8],
    fieldType: column === 'geometry' ? 'string' : 'real'
  }));
  expect(getCatalogResidentFields(dataset).map(field => field.name)).toEqual(['geometry', 'value']);
  expect(getCatalogResidentGeometryFields(dataset)).toEqual([
    'geometry',
    makeCatalogPointColumns('', '')
  ]);
  const points = await readCatalogFeaturesFromResidentDataset(dataset, 'geometry');
  expect(points.features[1].geometry).toEqual({type: 'Point', coordinates: [3, 4]});
});

test('resident integer values outside the exact JavaScript range are refused', async () => {
  const dataset = makeDataset();
  dataset.provider.readColumnValues = vi.fn(async () => ({
    values: [2n ** 63n, 0n],
    fieldType: 'integer'
  }));
  await expect(readCatalogFeaturesFromResidentDataset(dataset, MAPPING)).rejects.toThrow(
    /exact catalog numeric range/
  );
});

test('host lists the primary runtime and all publications with removal and runtime ownership guards', () => {
  const dataset = makeDataset();
  const published = {
    dataset: {id: 'resident-fixture', value: {source: {numRows: 2}}},
    label: dataset.label,
    consumer: {table: dataset.provider, explorer: {isCurrent: true}}
  };
  const runtime = {
    explorer: {dataset: {id: 'observations', value: {source: {numRows: 256}}}, isCurrent: true},
    nativeTable: dataset.provider,
    publications: new Map([['resident-fixture', published]])
  };
  const inputs = makeCatalogResidentDatasets(runtime as never, () => runtime as never);
  expect(inputs.map(input => input.id)).toEqual(['observations', 'resident-fixture']);
  expect(inputs[1].isCurrent()).toBe(true);
  runtime.publications.clear();
  expect(inputs[1].isCurrent()).toBe(false);
  const stale = makeCatalogResidentDatasets(runtime as never, () => null);
  expect(stale[0].isCurrent()).toBe(false);
});

function makeCollisionDataset(): CatalogResidentDataset {
  const source = new arrow.Table({
    longitude: arrow.vectorFromArray([1, 2], new arrow.Float64()),
    latitude: arrow.vectorFromArray([3, 4], new arrow.Float64()),
    event_time: arrow.vectorFromArray([1700000000000n, 1700000000001n], new arrow.Int64()),
    analysis_z_score: arrow.vectorFromArray([100, 200], new arrow.Float64())
  });
  const prepared: ResidentTableProvider = {
    datasetId: 'resident-fixture',
    sourceRevision: 1,
    fields: [
      ...source.schema.fields
        .filter(field => field.name !== 'analysis_z_score')
        .map(field => ({name: field.name, sortable: true})),
      {name: 'analysis_z_score', sortable: true, derived: true}
    ],
    maximumPageSize: 1,
    getRevision: () => 'revision-1',
    subscribe: () => () => {},
    readPage: vi.fn(async request => ({
      datasetId: 'resident-fixture',
      datasetRevision: 1,
      revision: 1,
      totalCount: 2,
      sourceRowIndices: Uint32Array.from([request.offset]),
      columns: {analysis_z_score: Float32Array.from([request.offset === 0 ? 7 : 9])},
      validity: {}
    }))
  };
  return {...makeDataset(), provider: withOriginalSourceTable(prepared, source, () => true)};
}

test('colliding derived aliases use original-view paging while source columns preserve exact values', async () => {
  const dataset = makeCollisionDataset();
  expect(getCatalogResidentFields(dataset).map(field => field.name)).toContain(
    'analysis_z_score_analysis'
  );
  const points = await readCatalogFeaturesFromResidentDataset(dataset, MAPPING);
  expect(points.features.map(feature => feature.properties!['analysis_z_score'])).toEqual([
    100, 200
  ]);
  expect(points.features.map(feature => feature.properties!['analysis_z_score_analysis'])).toEqual([
    7, 9
  ]);
  expect(points.features.map(feature => feature.properties!['event_time'])).toEqual([
    1700000000000, 1700000000001
  ]);
  expect(points.features[1].geometry).toEqual({type: 'Point', coordinates: [2, 4]});
});

test('published trajectory point mapping bypasses a matching resident geometry reader', async () => {
  const resident = makeDataset();
  const source = makeCatalogTrajectoryNativeSource({
    id: resident.id,
    label: resident.label,
    table: arrow.tableFromArrays({
      track_id: ['a', 'a'],
      timestamp: [0, 1000],
      longitude: [1, 2],
      latitude: [3, 4]
    })
  });
  const points = await readCatalogNativeInput({
    resident,
    source,
    field: TRAJECTORY_POINT_FIELD,
    isTrajectory: true,
    parameters: {longitudeProperty: 'longitude', latitudeProperty: 'latitude'}
  });
  expect(points.features.map(feature => feature.geometry?.coordinates)).toEqual([
    [1, 3],
    [2, 4]
  ]);
  expect(points.features.map(feature => feature.properties!['timestamp'])).toEqual([0, 1000]);
  expect(resident.provider.readColumnValues).not.toHaveBeenCalled();
  const ordinary = await readCatalogNativeInput({
    resident,
    source,
    field: MAPPING,
    isTrajectory: false,
    parameters: {}
  });
  expect(ordinary.features[0].geometry?.coordinates).toEqual([-87.65, 41.88]);
  expect(resident.provider.readColumnValues).toHaveBeenCalled();
});
