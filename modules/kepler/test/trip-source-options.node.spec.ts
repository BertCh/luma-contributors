// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {
  readSandboxTripOptions,
  prepareSandboxTrips
} from '../../../examples/deck/kepler-compute/src/trip-source-options';

test('sandbox Trips options require explicit encoding and retain only portable time controls', () => {
  for (const timestampEncoding of ['unix-second', 'unix-millisecond', 'iso8601'])
    expect(
      readSandboxTripOptions({
        timestampEncoding,
        originMilliseconds: -1000,
        maximumTimeErrorMilliseconds: 0.25
      })
    ).toEqual({timestampEncoding, originMilliseconds: -1000, maximumTimeErrorMilliseconds: 0.25});
  expect(readSandboxTripOptions({timestampEncoding: 'unix-second'})).toEqual({
    timestampEncoding: 'unix-second',
    maximumTimeErrorMilliseconds: 0
  });
  for (const invalid of [
    null,
    [],
    {},
    {timestampEncoding: 'auto'},
    {timestampEncoding: 'unix-second', crs: 'EPSG:4326'},
    {timestampEncoding: 'unix-second', originMilliseconds: 0.5},
    {timestampEncoding: 'unix-second', maximumTimeErrorMilliseconds: Infinity},
    {timestampEncoding: 'unix-second', maximumTimeErrorMilliseconds: -1}
  ])
    expect(() => readSandboxTripOptions(invalid)).toThrow();
});

test('Trips review preserves invalid rows and rejects a source with no accepted paths', () => {
  const valid = JSON.stringify({
    type: 'LineString',
    coordinates: [
      [-73, 40, 0, 1000],
      [-74, 41, 0, 1100]
    ]
  });
  const invalid = JSON.stringify({
    type: 'LineString',
    coordinates: [
      [-73, 40, 0, 1100],
      [-74, 41, 0, 1000]
    ]
  });
  const source = new arrow.Table({
    geometry: arrow.vectorFromArray([invalid, valid], new arrow.Utf8())
  });
  const options = {timestampEncoding: 'unix-millisecond' as const};
  const prepared = prepareSandboxTrips(source, 'geometry', options);
  expect(prepared.rowCount).toBe(2);
  expect(prepared.tripCount).toBe(1);
  expect(prepared.invalidRows).toHaveLength(1);
  expect(Array.from(prepared.batches[0].sourceRows)).toEqual([1]);
  expect(source.numRows).toBe(2);
  expect(() => prepareSandboxTrips(source.slice(0, 1), 'geometry', options)).toThrow(
    /no valid Trips/
  );
  expect(() => prepareSandboxTrips(source, 'geometry', {timestampEncoding: 'iso8601'})).toThrow(
    /no valid Trips/
  );
});

test('Trips source review rejects unsupported retained CRS before GPU publication', () => {
  const source = new arrow.Table({
    geometry: arrow.vectorFromArray(
      [
        JSON.stringify({
          type: 'LineString',
          coordinates: [
            [0, 0, 0, 0],
            [1, 1, 0, 1]
          ]
        })
      ],
      new arrow.Utf8()
    )
  });
  source.schema.fields[0].metadata.set(
    'ARROW:extension:metadata',
    JSON.stringify({crs: 'EPSG:3857'})
  );
  expect(() =>
    prepareSandboxTrips(source, 'geometry', {timestampEncoding: 'unix-millisecond'})
  ).toThrow(/CRS/);
});

test('explicit part-column policy survives saved options and rejects malformed roles', () => {
  const options = {
    timestampEncoding: 'unix-millisecond',
    grouping: {
      timestampColumn: 'time',
      longitudeColumn: 'lon',
      latitudeColumn: 'lat',
      altitudeColumn: 'height',
      partColumn: 'part'
    }
  };
  expect(readSandboxTripOptions(JSON.parse(JSON.stringify(options)))).toMatchObject(options);
  expect(() =>
    readSandboxTripOptions({...options, grouping: {...options.grouping, partColumn: ''}})
  ).toThrow(/part column/);
  expect(() =>
    readSandboxTripOptions({...options, grouping: {...options.grouping, altitudeColumn: 4}})
  ).toThrow(/altitude column/);
});
