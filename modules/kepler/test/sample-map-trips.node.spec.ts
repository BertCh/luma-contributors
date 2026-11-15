// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {existsSync, readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {makeAutoPreparePlan} from '../../../examples/deck/kepler-compute/src/auto-prepare-plan';
import {getGeometryVariantRequest} from '../../../examples/deck/kepler-compute/src/geometry-column-variants';
import {
  preparePublishedGeometry,
  preparePublishedTrips
} from '../../../examples/deck/kepler-compute/src/source-publication';

test('the same four-coordinate GeoJSON produces static XYZ paths and animated timed paths', () => {
  // Synthetic flight-like input: the fourth ordinate is time, not a spatial dimension.
  const table = arrow.tableFromArrays({
    geometry: [
      JSON.stringify({
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: [
            [145.1, -38, 120, 1567557180],
            [145.2, -37.9, 210, 1567557200]
          ]
        }
      })
    ]
  });
  const geometry = preparePublishedGeometry(table, 'geometry', {timedLineStrings: true});
  const trips = preparePublishedTrips(table, 'geometry', {timestampEncoding: 'unix-second'});
  expect(geometry.metadata.invalidRows).toEqual([]);
  expect(geometry.metadata.featureTypes).toEqual({line: true, polygon: false, point: false});
  expect(Array.from(geometry.batches[0]!.paths.sourceRows)).toEqual([0]);
  expect(Array.from(geometry.batches[0]!.paths.valueOffsets)).toEqual([0, 2]);
  expect(Array.from(geometry.batches[0]!.paths.positions)).toEqual(
    Array.from(trips.batches[0]!.positions)
  );
  expect(trips.tripCount).toBe(1);
  expect(trips.invalidRows).toEqual([]);
  expect(Array.from(trips.batches[0]!.sourceRows)).toEqual([0]);
  expect(Array.from(trips.batches[0]!.epochMilliseconds)).toEqual([1567557180000, 1567557200000]);
});

test('timed LineString projection is explicit and does not accept extra or invalid spatial ordinates', () => {
  for (const coordinates of [
    [
      [1, 2, 3, 100, 5],
      [4, 5, 6, 200, 5]
    ],
    [
      [1, 2, 'invalid', 100],
      [4, 5, 6, 200]
    ]
  ]) {
    const table = arrow.tableFromArrays({
      geometry: [JSON.stringify({type: 'LineString', coordinates})]
    });
    const geometry = preparePublishedGeometry(table, 'geometry', {timedLineStrings: true});
    expect(geometry.metadata.invalidRows).toHaveLength(1);
    expect(geometry.batches[0]!.paths.sourceRows).toHaveLength(0);
  }
});

test('a published static path is reused even when its source also contains animated trips', () => {
  const dataset = {id: 'flights'} as Parameters<typeof getGeometryVariantRequest>[2];
  const layer = {
    id: 'jrj4wei',
    type: 'geojson',
    config: {columns: {geojson: {value: 'geometry'}}}
  } as Parameters<typeof getGeometryVariantRequest>[0];
  const primary = {geometryColumn: 'geometry', isTrip: true, hasGeometry: true, hasH3: false};
  expect(getGeometryVariantRequest(layer, primary, dataset)).toBeUndefined();
  expect(getGeometryVariantRequest(layer, {...primary, hasGeometry: false}, dataset)).toEqual({
    kind: 'geojson',
    column: 'geometry',
    dataset
  });
});

test.each([
  false,
  true
])('earnings-style native GeoArrow retains geometry and draw primitives (use cache: %s)', useCache => {
  const cached =
    'examples/deck/kepler-compute/scripts/kepler-demo-cache/samples/nyc_earnings/data.arrow';
  const coordinates = [
    [
      [
        [
          [-73, 41],
          [-72, 41],
          [-72, 42],
          [-73, 41]
        ]
      ]
    ]
  ];
  const coordinate = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
  const rings = new arrow.List(new arrow.Field('vertices', coordinate));
  const polygons = new arrow.List(new arrow.Field('rings', rings));
  const multipolygons = new arrow.List(new arrow.Field('polygons', polygons));
  const standIn = new arrow.Table({geometry: arrow.vectorFromArray(coordinates, multipolygons)});
  standIn.schema.fields[0]!.metadata.set('ARROW:extension:name', 'geoarrow.multipolygon');
  // Read-only optional fixture: never save or bundle sample payloads; a stand-in works offline.
  const table = useCache && existsSync(cached) ? arrow.tableFromIPC(readFileSync(cached)) : standIn;
  const plan = makeAutoPreparePlan(table, [
    {id: 'hkc2hq', type: 'geojson', isVisible: true, columns: {geojson: 'geometry'}}
  ]);
  expect(plan.geometryColumn).toBe('geometry');
  const prepared = preparePublishedGeometry(plan.table, 'geometry');
  expect(prepared.metadata.invalidRows).toEqual([]);
  expect(prepared.metadata.featureTypes.polygon).toBe(true);
  expect(
    plan.table.schema.fields
      .find(field => field.name === 'geometry')
      ?.metadata.get('ARROW:extension:name')
  ).toBe('geoarrow.multipolygon');
  expect(prepared.batches.some(batch => batch.polygonOutlines.positions.length > 0)).toBe(true);
}, 30000);

test('timed MultiLineString retains both static XYZ paths and animated trips', () => {
  const coordinates = [
    [
      [1, 2, 120, 0],
      [2, 3, 150, 1000]
    ],
    [
      [4, 5, 200, 2000],
      [5, 6, 250, 3000]
    ]
  ];
  const table = arrow.tableFromArrays({
    geometry: [JSON.stringify({type: 'MultiLineString', coordinates})]
  });
  const trips = preparePublishedTrips(table, 'geometry', {timestampEncoding: 'unix-millisecond'});
  expect(trips.invalidRows).toEqual([]);
  expect(trips.tripCount).toBe(2);
  const geometry = preparePublishedGeometry(table, 'geometry', {timedLineStrings: true});
  expect(geometry.metadata.invalidRows).toEqual([]);
  expect(Array.from(geometry.batches[0].paths.sourceRows)).toEqual([0, 0]);
  expect(Array.from(geometry.batches[0].paths.valueOffsets)).toEqual([0, 2, 4]);
  expect(Array.from(geometry.batches[0].paths.positions)).toEqual(
    Array.from(trips.batches[0].positions)
  );
  expect(preparePublishedGeometry(table, 'geometry').metadata.invalidRows).toHaveLength(1);
});
