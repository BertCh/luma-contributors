// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';
import {TRAJECTORY_BINDINGS} from '../src/analysis/catalog-trajectory-bindings';
import {
  getCatalogSampleInputs,
  readCatalogFeaturesFromNativeDataset,
  readCatalogTrajectoryFeaturesFromNativeDataset,
  findNativeGeometryFields
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {
  makeCatalogTrajectoryNativeSource,
  readCatalogTrajectoryPoints,
  getCatalogTrajectoryTripOptions
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-trajectory';
import {previewArrowSource} from '../../../examples/deck/kepler-compute/src/source-preview';
import {prepareSandboxTrips} from '../../../examples/deck/kepler-compute/src/trip-source-options';
import {makeArrowTableFromCatalogTable} from '../../../examples/deck/kepler-compute/src/analysis-catalog-table';

import {
  prepareCatalogTrajectory,
  makeTrajectoryFixes,
  makeTrajectoryTracks
} from '../src/analysis/catalog-trajectory-common';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

const ORIGIN = Date.UTC(2026, 0, 1);

test('every trajectory UI operation has a valid shared sample, and CPU samples run', () => {
  for (const binding of TRAJECTORY_BINDINGS) {
    const sample = JSON.parse(getCatalogSampleInputs(binding.id)['tracks']);
    expect(sample.features).toHaveLength(2);
    if (binding.run) {
      const result = runCatalogOperation(binding, {}, {tracks: sample});
      expect(result.tables.length).toBeGreaterThan(0);
      expect(result.summary['trackCount']).toBe(2);
    }
  }
});

test('native point column mapping retains scalar track/time properties and handles chunked Arrow', () => {
  const table = arrow
    .tableFromArrays({
      animal: ['a', 'b'],
      time: [ORIGIN + 1000, ORIGIN],
      east: [1, 2],
      north: [0, 0]
    })
    .concat(
      arrow.tableFromArrays({
        animal: ['a', 'b'],
        time: [ORIGIN, ORIGIN + 1000],
        east: [0, 3],
        north: [0, 0]
      })
    );
  const source = makeCatalogTrajectoryNativeSource({id: 'native', label: 'Native fixes', table});
  const tracks = readCatalogTrajectoryPoints(source, 'east', 'north');
  const result = runCatalogOperation(
    getCatalogBinding('trajectory-thinning')!,
    {
      coordinateSystem: 'planar',
      trackProperty: 'animal',
      timeProperty: 'time',
      mode: 'minimum-time',
      tolerance: 0
    },
    {tracks}
  );
  expect(result.tables[1].columns['source_index']).toEqual([2, 0, 1, 3]);
  expect(result.tables[0].columns['track_id']).toEqual(['a', 'b']);
  expect(() => readCatalogTrajectoryPoints(source, 'missing', 'north')).toThrow(/existing/);
});

test('native timed paths produce native Arrow outputs with explicit Trips policy', () => {
  const sourceTable = arrow.tableFromArrays({
    track_id: ['null-trip', 'native-trip'],
    route: [
      null,
      JSON.stringify({
        type: 'LineString',
        coordinates: [
          [-74, 40, 0, ORIGIN],
          [-73.99, 40, 0, ORIGIN + 1000]
        ]
      })
    ]
  });
  const source = makeCatalogTrajectoryNativeSource({
    id: 'native',
    label: 'Native trip',
    table: sourceTable
  });
  const tracks = readCatalogFeaturesFromNativeDataset(source, 'route');
  const result = runCatalogOperation(
    getCatalogBinding('trajectory-thinning')!,
    {tolerance: 0},
    {tracks}
  );
  expect(result.diagnostics).toEqual([
    expect.objectContaining({code: 'trajectory-empty-geometry', count: 1})
  ]);
  expect(result.tables[1].columns['source_index']).toEqual([1, 1]);
  const paths = result.tables[0];
  const table = makeArrowTableFromCatalogTable(paths);
  expect(table.getChild('geometry')!.type).toBeInstanceOf(arrow.Utf8);
  expect(table.getChild('start_time')!.type).toBeInstanceOf(arrow.Float64);
  expect(JSON.parse(table.getChild('geometry')!.get(0)).coordinates[1][3]).toBe(ORIGIN + 1000);
  expect(getCatalogTrajectoryTripOptions(paths)).toEqual({
    timestampEncoding: 'unix-millisecond',
    originMilliseconds: ORIGIN,
    maximumTimeErrorMilliseconds: 0
  });
  expect(getCatalogTrajectoryTripOptions(result.tables[1])).toBeUndefined();
  // Sources requires an explicit Float64 policy for selected epoch metadata. Excluding the
  // redundant scalar columns leaves the exact original Arrow and the geometry clock intact.
  const rejected = previewArrowSource(table, {}, ['geometry']);
  expect(rejected.canUpload).toBe(false);
  expect(rejected.reason).toContain('start_time, end_time');
  const columns = table.schema.fields
    .map(field => field.name)
    .filter(name => !['geometry', 'start_time', 'end_time'].includes(name));
  expect(previewArrowSource(table, {columns}, ['geometry']).canUpload).toBe(true);
  expect(table.getChild('start_time')!.get(0)).toBe(ORIGIN);
  expect(table.getChild('end_time')!.get(0)).toBe(ORIGIN + 1000);
  const prepared = prepareSandboxTrips(table, 'geometry', getCatalogTrajectoryTripOptions(paths)!);
  expect(prepared.tripCount).toBe(1);
  expect(prepared.vertexCount).toBe(2);
  expect(prepared.invalidRows).toEqual([]);
  expect(prepared.actualTimeErrorMilliseconds).toBe(0);
  expect(prepared.animationDomainMilliseconds).toEqual([ORIGIN, ORIGIN + 1000]);
});

test('native point capacity is checked before reading cells', () => {
  const source = {
    id: 'oversized',
    label: '',
    fields: [{name: 'longitude'}, {name: 'latitude'}],
    dataContainer: {
      numRows: () => 2049,
      valueAt: () => {
        throw new Error('must not inspect');
      }
    }
  };
  expect(() => readCatalogTrajectoryPoints(source, 'longitude', 'latitude')).toThrow(
    /catalog-trajectory-native rowCount.*2048/
  );
});

test('discovery tolerates unrelated Int64 values and explicit ingestion retains them losslessly', () => {
  const unsafeInteger = 9007199254740993n;
  const table = new arrow.Table({
    unrelated: arrow.vectorFromArray([unsafeInteger], new arrow.Int64()),
    track_id: arrow.vectorFromArray(['track'], new arrow.Utf8()),
    route: arrow.vectorFromArray(
      [
        JSON.stringify({
          type: 'LineString',
          coordinates: [
            [1, 2, 120, 0],
            [2, 3, 150, 1000]
          ]
        })
      ],
      new arrow.Utf8()
    )
  });
  const source = makeCatalogTrajectoryNativeSource({id: 'unselected-source', label: '', table});
  expect(source.dataContainer.valueAt(0, 0)).toBe(unsafeInteger);
  expect(findNativeGeometryFields(source)).toEqual(['route']);
  const tracks = readCatalogTrajectoryFeaturesFromNativeDataset(source, 'route');
  expect(tracks.features[0].properties!['unrelated']).toBe('9007199254740993');
  const result = runCatalogOperation(
    getCatalogBinding('trajectory-thinning')!,
    {tolerance: 0},
    {tracks}
  );
  expect(JSON.parse(result.tables[1].columns['original_properties'][0] as string).unrelated).toBe(
    '9007199254740993'
  );
});

test('native numeric roles convert safe Int64 values and reject unsafe values only during explicit ingestion', () => {
  const table = new arrow.Table({
    animal: arrow.vectorFromArray([9007199254740993n, 9007199254740993n], new arrow.Int64()),
    time: arrow.vectorFromArray([BigInt(ORIGIN), BigInt(ORIGIN + 1000)], new arrow.Int64()),
    east: arrow.vectorFromArray([1n, 2n], new arrow.Int64()),
    north: arrow.vectorFromArray([2n, 3n], new arrow.Int64()),
    unsafe: arrow.vectorFromArray([9007199254740993n, 9007199254740995n], new arrow.Int64())
  });
  const source = makeCatalogTrajectoryNativeSource({id: 'point-source', label: '', table});
  expect(findNativeGeometryFields(source)).toEqual([]);
  const parameters = {trackProperty: 'animal', timeProperty: 'time', tolerance: 0};
  const tracks = readCatalogTrajectoryPoints(source, 'east', 'north', parameters);
  const result = runCatalogOperation(getCatalogBinding('trajectory-thinning')!, parameters, {
    tracks
  });
  expect(result.tables[0].columns['track_id']).toEqual(['9007199254740993']);
  expect(result.tables[1].columns['timestamp']).toEqual([ORIGIN, ORIGIN + 1000]);
  expect(() =>
    readCatalogTrajectoryPoints(source, 'east', 'north', {...parameters, timeProperty: 'unsafe'})
  ).toThrow(/property unsafe integer exceeds JavaScript exact range/);
  expect(() => readCatalogTrajectoryPoints(source, 'unsafe', 'north', parameters)).toThrow(
    /property unsafe integer exceeds JavaScript exact range/
  );
});

test('geometry discovery catches cell-read failures before rendering the catalog', () => {
  const source = {
    fields: [{name: 'unreadable'}],
    dataContainer: {
      numRows: () => 1,
      valueAt: () => {
        throw new Error('unselected cell cannot be read');
      }
    }
  };
  expect(findNativeGeometryFields(source)).toEqual([]);
});

test('no-op thinning preserves source altitude through fixes, timed paths and native Trips buffers', () => {
  const tracks: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        properties: {track_id: 'elevated'},
        geometry: {
          type: 'LineString',
          coordinates: [
            [1, 2, 120, 0],
            [2, 3, 150, 1000]
          ]
        }
      }
    ]
  };
  const result = runCatalogOperation(
    getCatalogBinding('trajectory-thinning')!,
    {tolerance: 0},
    {tracks}
  );
  const paths = result.tables[0];
  expect(JSON.parse(paths.columns['geometry'][0] as string).coordinates).toEqual([
    [1, 2, 120, 0],
    [2, 3, 150, 1000]
  ]);
  expect(
    result.tables[1].columns['geometry'].map(geometry => JSON.parse(geometry as string).coordinates)
  ).toEqual([
    [1, 2, 120],
    [2, 3, 150]
  ]);
  const prepared = prepareSandboxTrips(
    makeArrowTableFromCatalogTable(paths),
    'geometry',
    getCatalogTrajectoryTripOptions(paths)!
  );
  expect(prepared.invalidRows).toEqual([]);
  expect(Array.from(prepared.batches[0].positions)).toEqual([1, 2, 120, 2, 3, 150]);
  expect(Array.from(prepared.batches[0].epochMilliseconds)).toEqual([0, 1000]);
  const recursions = runCatalogOperation(getCatalogBinding('trajectory-recursions')!, {}, {tracks});
  expect(
    recursions.tables[0].columns['geometry'].map(
      geometry => JSON.parse(geometry as string).coordinates
    )
  ).toEqual([
    [1, 2, 120],
    [2, 3, 150]
  ]);
});

test('retained point altitude survives sorting and corrected XY output', () => {
  const tracks: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        properties: {track_id: 'a', timestamp: 1000},
        geometry: {type: 'Point', coordinates: [2, 3, 150]}
      },
      {
        properties: {track_id: 'a', timestamp: 0},
        geometry: {type: 'Point', coordinates: [1, 2, 120]}
      }
    ]
  };
  const binding = getCatalogBinding('trajectory-thinning')!;
  const trajectory = prepareCatalogTrajectory(tracks, getCatalogDefaultParameters(binding));
  const corrected = [
    [10, 20],
    [11, 21]
  ] as const;
  expect(
    makeTrajectoryFixes(trajectory, {}, undefined, undefined, corrected).columns['geometry'].map(
      geometry => JSON.parse(geometry as string).coordinates
    )
  ).toEqual([
    [10, 20, 120],
    [11, 21, 150]
  ]);
  expect(
    JSON.parse(
      makeTrajectoryTracks(trajectory, undefined, corrected).columns['geometry'][0] as string
    ).coordinates
  ).toEqual([
    [10, 20, 120, 0],
    [11, 21, 150, 1000]
  ]);
  tracks.features[0].geometry!.coordinates = [2, 3, NaN];
  expect(() => prepareCatalogTrajectory(tracks, getCatalogDefaultParameters(binding))).toThrow(
    /invalid altitude/
  );
});

for (const representation of ['JavaScript array', 'Arrow List', 'Arrow Int64 List']) {
  test(`native ${representation} timestamps match direct GeoJSON trajectory input`, () => {
    const geometry = {
      type: 'LineString',
      coordinates: [
        [2, 3, 150],
        [1, 2, 120]
      ]
    };
    const moments = [ORIGIN + 1000, ORIGIN];
    const direct: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [{id: 0, geometry, properties: {track_id: 'track', moments}}]
    };
    const parameters = {timeProperty: 'moments', tolerance: 0};
    const source =
      representation === 'JavaScript array'
        ? {
            fields: [{name: 'track_id'}, {name: 'moments'}, {name: 'route'}],
            dataContainer: {
              numRows: () => 1,
              valueAt: (_row: number, column: number) =>
                ['track', moments, JSON.stringify(geometry)][column]
            }
          }
        : makeCatalogTrajectoryNativeSource({
            id: 'array-times',
            label: '',
            table: new arrow.Table({
              track_id: arrow.vectorFromArray(['track'], new arrow.Utf8()),
              moments:
                representation === 'Arrow Int64 List'
                  ? arrow.vectorFromArray(
                      [moments.map(BigInt)],
                      new arrow.List(new arrow.Field('item', new arrow.Int64()))
                    )
                  : arrow.vectorFromArray(
                      [moments],
                      new arrow.List(new arrow.Field('item', new arrow.Float64()))
                    ),
              route: arrow.vectorFromArray([JSON.stringify(geometry)], new arrow.Utf8())
            })
          });
    expect(findNativeGeometryFields(source)).toEqual(['route']);
    const native = readCatalogTrajectoryFeaturesFromNativeDataset(source, 'route', parameters);
    expect(native.features[0].properties!['moments']).toEqual(moments);
    const binding = getCatalogBinding('trajectory-thinning')!;
    const expected = runCatalogOperation(binding, parameters, {tracks: direct});
    expect(runCatalogOperation(binding, parameters, {tracks: native}).tables).toEqual(
      expected.tables
    );
    expect(JSON.parse(expected.tables[0].columns['geometry'][0] as string).coordinates).toEqual([
      [1, 2, 120, ORIGIN],
      [2, 3, 150, ORIGIN + 1000]
    ]);
  });
}

test('typed non-feature support preserves the trajectory feature callback contract', () => {
  const binding = getCatalogBinding('trajectory-thinning')!;
  expect(binding.inputs[0].dataKind).toBeUndefined();
  expect(binding.runData).toBeUndefined();
  const tracks = JSON.parse(getCatalogSampleInputs(binding.id)['tracks']);
  const expected = binding.run!({tracks}, getCatalogDefaultParameters(binding));
  const actual = runCatalogOperation(binding, {}, {tracks});
  expect(actual.tables).toEqual(expected.tables);
  expect(actual.summary).toEqual(expected.summary);
});
