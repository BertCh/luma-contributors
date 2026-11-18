// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';
import {
  prepareKeplerGeoJSONTrips,
  prepareKeplerGeoJSONTripsFromArrow
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geojson-trips';
import {makeKeplerGeoJSONTripGPUVectors} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/trip-gpu-vectors';

function trip(times: readonly (number | string)[]) {
  return {
    type: 'Feature',
    properties: {name: 'route'},
    geometry: {
      type: 'LineString',
      coordinates: times.map((time, index) => [-74 + index / 100, 40, 10 + index, time])
    }
  };
}

test('numeric Unix seconds keep every millisecond vertex when integer milliseconds round-trip', () => {
  const start = Date.UTC(2004, 1, 1);
  const milliseconds = [start, start + 1, start + 1000];
  const seconds = milliseconds.map(value => value / 1000);
  expect(seconds[1] * 1000).not.toBe(milliseconds[1]);
  const prepared = prepareKeplerGeoJSONTrips([{id: 'seconds', rows: [trip(seconds)]}], {
    timestampEncoding: 'unix-second'
  });
  expect(prepared.tripCount).toBe(1);
  expect([...prepared.batches[0].epochMilliseconds]).toEqual(milliseconds);
  expect([...prepared.batches[0].timestamps]).toEqual([0, 1, 1000]);
  expect(prepared.batches[0].sourceTimestamps).toEqual(seconds);
  expect(prepared.invalidRows).toEqual([]);
  const fractional = prepareKeplerGeoJSONTrips([{id: 'fractional', rows: [trip([1.0001, 2])]}], {
    timestampEncoding: 'unix-second'
  });
  expect(fractional.tripCount).toBe(0);
  expect(fractional.invalidRows).toHaveLength(1);
});

test('direct GeoJSON trips preserve exact source time, altitude, source rows and physical batches', () => {
  const epoch = 1_700_000_000_000;
  const prepared = prepareKeplerGeoJSONTrips(
    [
      {id: 'first', rows: [null, trip([epoch, epoch + 1, epoch + 500])]},
      {id: 'empty', rows: []},
      {id: 'last', rows: [trip([epoch + 1000, epoch + 1001])]}
    ],
    {timestampEncoding: 'unix-millisecond', crs: 'EPSG:4326'}
  );
  expect(prepared.rowCount).toBe(3);
  expect(prepared.tripCount).toBe(2);
  expect(prepared.vertexCount).toBe(5);
  expect(prepared.originMilliseconds).toBe(epoch);
  expect(prepared.animationDomainMilliseconds).toEqual([epoch, epoch + 1001]);
  expect(prepared.actualTimeErrorMilliseconds).toBe(0);
  expect(prepared.batches.map(batch => batch.id)).toEqual(['first', 'empty', 'last']);
  expect(prepared.batches.map(batch => Array.from(batch.sourceRows))).toEqual([[1], [], [2]]);
  expect(prepared.batches.map(batch => Array.from(batch.valueOffsets))).toEqual([
    [0, 3],
    [0],
    [0, 2]
  ]);
  expect(Array.from(prepared.batches[0].epochMilliseconds)).toEqual([
    epoch,
    epoch + 1,
    epoch + 500
  ]);
  expect(prepared.batches[0].sourceTimestamps).toEqual([epoch, epoch + 1, epoch + 500]);
  expect(Array.from(prepared.batches[0].timestamps)).toEqual([0, 1, 500]);
  expect(prepared.batches[0].positions[2]).toBe(10);
  expect(prepared.batches[0].positions[5]).toBe(11);
  expect(prepared.crs).toBe('EPSG:4326');
});

test('Arrow JSON slices retain chunk identities and zero epoch values remain valid', () => {
  const first = arrow.vectorFromArray([null, JSON.stringify(trip([0, 1]))], new arrow.Utf8());
  const second = arrow.vectorFromArray([JSON.stringify(trip([2, 3])), null], new arrow.Utf8());
  const sliced = new arrow.Vector([...first.data, ...second.data]).slice(1, 3);
  const prepared = prepareKeplerGeoJSONTripsFromArrow(sliced, {
    timestampEncoding: 'unix-second',
    batchIds: ['left', 'right']
  });
  expect(prepared.batches.map(batch => batch.rowCount)).toEqual([1, 1]);
  expect(prepared.batches.map(batch => Array.from(batch.sourceRows))).toEqual([[0], [1]]);
  expect(prepared.originMilliseconds).toBe(0);
  expect(prepared.animationDomainMilliseconds).toEqual([0, 3000]);
  expect(prepared.batches[1].sourceTimestamps).toEqual([2, 3]);
  expect(Array.from(prepared.batches[1].timestamps)).toEqual([2000, 3000]);
});

test('trip time precision is explicit and original ISO strings remain unchanged', () => {
  const seconds = prepareKeplerGeoJSONTrips([{id: 'seconds', rows: [trip([1.25, 1.5])]}], {
    timestampEncoding: 'unix-second'
  });
  expect(seconds.batches[0].sourceTimestamps).toEqual([1.25, 1.5]);
  expect(Array.from(seconds.batches[0].epochMilliseconds)).toEqual([1250, 1500]);
  const precise = prepareKeplerGeoJSONTrips([{id: 'values', rows: [trip([0, 16_777_217])]}], {
    timestampEncoding: 'unix-millisecond'
  });
  expect(precise.tripCount).toBe(0);
  expect(precise.invalidRows[0].reason).toMatch(/error budget/);
  const approximate = prepareKeplerGeoJSONTrips([{id: 'values', rows: [trip([0, 16_777_217])]}], {
    timestampEncoding: 'unix-millisecond',
    originMilliseconds: 0,
    maximumTimeErrorMilliseconds: 1
  });
  expect(approximate.actualTimeErrorMilliseconds).toBe(1);
  expect(Array.from(approximate.batches[0].epochMilliseconds)).toEqual([0, 16_777_217]);
  expect(Array.from(approximate.batches[0].timestamps)).toEqual([0, 16_777_216]);
  const timestamps = ['2024-02-29T01:00:00.001+01:00', '2024-02-29T00:00:00.002Z'];
  const iso = prepareKeplerGeoJSONTrips([{id: 'iso', rows: [trip(timestamps)]}], {
    timestampEncoding: 'iso8601'
  });
  expect(iso.batches[0].sourceTimestamps).toEqual(timestamps);
  expect(Array.from(iso.batches[0].timestamps)).toEqual([0, 1]);
  expect(iso.originMilliseconds).toBe(Date.UTC(2024, 1, 29, 0, 0, 0, 1));
});

test('invalid topology, unordered vertices and inexact timestamps reject entire rows without renumbering', () => {
  const prepared = prepareKeplerGeoJSONTrips(
    [
      {
        id: 'invalid',
        rows: [
          trip([2, 1]),
          trip([0, 0.5]),
          {type: 'MultiLineString', coordinates: []},
          new Uint8Array([1, 2]),
          {
            type: 'LineString',
            coordinates: [
              [1, 2, 3],
              [2, 3, 4]
            ]
          },
          trip([5, 5])
        ]
      }
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  expect(prepared.invalidRows.map(row => row.sourceRowIndex)).toEqual([0, 1, 2, 3, 4]);
  expect(Array.from(prepared.batches[0].sourceRows)).toEqual([5]);
  expect(Array.from(prepared.batches[0].timestamps)).toEqual([0, 0]);
  const invalidIso = prepareKeplerGeoJSONTrips(
    [
      {
        id: 'iso',
        rows: [
          trip(['2023-02-29T00:00:00Z', '2023-03-01T00:00:00Z']),
          trip(['2024-01-01T00:00:00.0001Z', '2024-01-01T00:00:01Z'])
        ]
      }
    ],
    {timestampEncoding: 'iso8601'}
  );
  expect(invalidIso.tripCount).toBe(0);
  expect(invalidIso.invalidRows).toHaveLength(2);
});

test('trip upload preserves aligned variable-list chunks, primitive provenance and ownership', () => {
  const device = new NullDevice({});
  const source = prepareKeplerGeoJSONTrips(
    [
      {id: 'one', rows: [null, trip([0, 1])]},
      {id: 'two', rows: [trip([2, 3])]}
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  const uploaded = makeKeplerGeoJSONTripGPUVectors(device, source);
  expect(uploaded.getPath.format).toBe('vertex-list<float32x3>');
  expect(uploaded.getTimestamps.format).toBe('vertex-list<float32>');
  expect(uploaded.getPath.data).toHaveLength(2);
  expect(uploaded.getTimestamps.data.map(chunk => chunk.valueLength)).toEqual([2, 2]);
  expect(uploaded.getTimestamps.data[0].valueOffsets).toBe(source.batches[0].valueOffsets);
  expect(uploaded.resolveSourceRow(0)).toBe(1);
  expect(uploaded.resolveSourceRow(1)).toBe(2);
  expect(uploaded.resolveSourceRow(2)).toBe(-1);
  expect(uploaded.gpuByteLength).toBe(72);
  const buffers = [
    ...uploaded.getPath.data,
    ...uploaded.getTimestamps.data,
    ...uploaded.sourceRowIds.data
  ].map(chunk => chunk.buffer);
  uploaded.destroy();
  uploaded.destroy();
  expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
  const createBuffer = device.createBuffer.bind(device);
  const allocated = createBuffer({byteLength: 24});
  const spy = vi
    .spyOn(device, 'createBuffer')
    .mockReturnValueOnce(allocated)
    .mockImplementationOnce(() => {
      throw new Error('capacity');
    });
  expect(() => makeKeplerGeoJSONTripGPUVectors(device, source)).toThrow('capacity');
  expect(allocated.destroyed).toBe(true);
  spy.mockRestore();
  device.destroy();
});

test('trip embedded line styles remain source aligned through invalid/null rows and time rejection', () => {
  const styled = (times: readonly number[], properties: Record<string, unknown>) => ({
    ...trip(times),
    properties
  });
  const batches = [
    {
      id: 'first',
      rows: [
        null,
        styled([0, 1], {lineColor: [10, 20, 30], lineWidth: 3}),
        styled([2, 1], {lineColor: 'bad', lineWidth: 8})
      ]
    },
    {id: 'empty', rows: []},
    {
      id: 'last',
      rows: [
        styled([2, 3], {lineColor: false, lineWidth: 0}),
        styled([4, 5], {lineColor: [40, 50, 60, 0], lineWidth: -2}),
        styled([6, 7], {lineColor: 'bad', lineWidth: Infinity}),
        styled([0, 16_777_217], {lineWidth: 99})
      ]
    }
  ];
  const source = prepareKeplerGeoJSONTrips(batches, {timestampEncoding: 'unix-millisecond'});
  expect(source.rowCount).toBe(7);
  expect(source.tripCount).toBe(4);
  expect(source.sourceStyleProperties).toEqual(['lineColor', 'lineWidth']);
  expect(source.invalidRows.map(row => row.sourceRowIndex)).toEqual([2, 6]);
  expect(source.invalidStyleProperties.map(row => [row.sourceRowIndex, row.property])).toEqual([
    [5, 'lineColor'],
    [5, 'lineWidth']
  ]);
  expect(source.batches.map(batch => [...batch.sourceRows])).toEqual([[1], [], [3, 4, 5]]);
  expect(source.batches.map(batch => [...batch.properties!.lineColor!.validity])).toEqual([
    [0, 1, 0],
    [],
    [0, 1, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.properties!.lineWidth!.values])).toEqual([
    [0, 3, 0],
    [],
    [0, -2, 0, 0]
  ]);
  expect([...source.batches[0].properties!.lineColor!.values]).toEqual([
    0, 0, 0, 0, 10, 20, 30, 255, 0, 0, 0, 0
  ]);
  expect([...source.batches[2].properties!.lineColor!.values]).toEqual([
    0, 0, 0, 0, 40, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  expect(source.batches.map(batch => [...batch.timestamps])).toEqual([
    [0, 1],
    [],
    [2, 3, 4, 5, 6, 7]
  ]);
  expect(source.batches.map(batch => [...batch.epochMilliseconds])).toEqual([
    [0, 1],
    [],
    [2, 3, 4, 5, 6, 7]
  ]);
  const withoutProperties = prepareKeplerGeoJSONTrips(
    batches.map(batch => ({
      ...batch,
      rows: batch.rows.map(row => (row ? {...row, properties: {}} : row))
    })),
    {timestampEncoding: 'unix-millisecond'}
  );
  expect(source.typedStorageByteLength - withoutProperties.typedStorageByteLength).toBe(7 * 16);
});

test('trip property upload owns source-row channels independently of compacted primitive rows', () => {
  const device = new NullDevice({});
  const first = {...trip([0, 1]), properties: {lineColor: [10, 20, 30], lineWidth: 2}};
  const source = prepareKeplerGeoJSONTrips(
    [
      {id: 'first', rows: [null, first]},
      {id: 'empty', rows: []},
      {id: 'last', rows: [trip([2, 3])]}
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  const uploaded = makeKeplerGeoJSONTripGPUVectors(device, source);
  expect(uploaded.properties!.lineColor!.values.format).toBe('unorm8x4');
  expect(uploaded.properties!.lineWidth!.values.format).toBe('float32');
  expect(uploaded.properties!.lineWidth!.values.data.map(chunk => chunk.length)).toEqual([2, 0, 1]);
  expect(uploaded.sourceRowIds.data.map(chunk => chunk.length)).toEqual([1, 0, 1]);
  const propertyBuffers = Object.values(uploaded.properties!).flatMap(channel =>
    [...channel.values.data, ...channel.validity.data].map(chunk => chunk.buffer)
  );
  const allBuffers = [
    ...uploaded.getPath.data,
    ...uploaded.getTimestamps.data,
    ...uploaded.sourceRowIds.data
  ]
    .map(chunk => chunk.buffer)
    .concat(propertyBuffers);
  expect(uploaded.gpuByteLength).toBe(
    allBuffers.reduce((sum, buffer) => sum + buffer.byteLength, 0)
  );
  expect(new Set(allBuffers).size).toBe(allBuffers.length);
  uploaded.destroy();
  uploaded.destroy();
  expect(propertyBuffers.every(buffer => buffer.destroyed)).toBe(true);
  device.destroy();
});

test('rejects trips beyond the Float32 budget and reports the required budget', () => {
  const epoch = 1_700_000_000_000;
  const prepared = prepareKeplerGeoJSONTrips(
    [{id: 'long', rows: [trip([epoch, epoch + 30_000_001])]}],
    {
      timestampEncoding: 'unix-millisecond'
    }
  );
  expect(prepared.tripCount).toBe(0);
  expect(prepared.invalidRows).toHaveLength(1);
  expect(prepared.invalidRows[0].reason).toMatch(
    /^Relative trip times exceed the explicit Float32 error budget \(needs at least 1 ms, budget is 0 ms\)$/
  );
  const accepted = prepareKeplerGeoJSONTrips(
    [{id: 'long', rows: [trip([epoch, epoch + 30_000_001])]}],
    {
      timestampEncoding: 'unix-millisecond',
      maximumTimeErrorMilliseconds: 1
    }
  );
  expect(accepted.tripCount).toBe(1);
});

test('multipart timed paths preserve disconnected parts, duplicate source identity and row-atomic rejection', () => {
  const multipart = (parts: number[][]) => ({
    type: 'Feature',
    properties: {lineWidth: 7},
    geometry: {
      type: 'MultiLineString',
      coordinates: parts.map(times => trip(times).geometry.coordinates)
    }
  });
  const prepared = prepareKeplerGeoJSONTrips(
    [
      {
        id: 'first',
        rows: [
          null,
          multipart([
            [100, 200],
            [0, 50]
          ])
        ]
      },
      {id: 'empty', rows: []},
      {
        id: 'last',
        rows: [
          multipart([
            [0, 1],
            [3, 2]
          ]),
          multipart([
            [0, 1],
            [0, 16_777_217]
          ])
        ]
      }
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  expect(prepared.tripCount).toBe(2);
  expect(prepared.rowCount).toBe(4);
  expect(prepared.vertexCount).toBe(4);
  expect(prepared.animationDomainMilliseconds).toEqual([0, 200]);
  expect(prepared.batches.map(batch => [...batch.sourceRows])).toEqual([[1, 1], [], []]);
  expect([...prepared.batches[0].valueOffsets]).toEqual([0, 2, 4]);
  expect([...prepared.batches[0].timestamps]).toEqual([100, 200, 0, 50]);
  expect(prepared.invalidRows.map(row => row.sourceRowIndex).sort()).toEqual([2, 3]);
  expect([...prepared.batches[0].properties!.lineWidth!.values]).toEqual([0, 7]);
  const device = new NullDevice({});
  const uploaded = makeKeplerGeoJSONTripGPUVectors(device, prepared);
  expect(uploaded.resolveSourceRow(0)).toBe(1);
  expect(uploaded.resolveSourceRow(1)).toBe(1);
  expect(uploaded.getPath.data.map(chunk => chunk.length)).toEqual([2, 0, 0]);
  uploaded.destroy();
  device.destroy();
});
