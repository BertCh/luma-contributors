// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  prepareKeplerTabularTrips,
  prepareKeplerTabularTripsFromArrow
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/tabular-trips';
import {makeKeplerTabularTripGPUVectors} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/tabular-trip-gpu-vectors';

const epoch = 1_700_000_000_000;
const columns = {tripId: 'id', timestamp: 'time', longitude: 'lon', latitude: 'lat'};

function makeBatch(id: string, rows: [unknown, unknown, unknown, unknown][]) {
  return {
    id,
    tripIds: rows.map(row => row[0]),
    timestamps: rows.map(row => row[1]),
    longitudes: rows.map(row => row[2]),
    latitudes: rows.map(row => row[3])
  };
}

test('tabular trips group across batches, sort by time and keep explicit membership', () => {
  const prepared = prepareKeplerTabularTrips(
    [
      makeBatch('first', [
        ['a', epoch + 2000, -73.9, 40],
        [null, epoch, -74, 40],
        ['b', epoch + 500, -74, 41],
        ['b', 'bad', -74, 41]
      ]),
      makeBatch('second', [
        ['a', epoch, -74, 40],
        ['b', epoch + 1500, -73.9, 41],
        ['c', epoch, -74, 42],
        ['a', epoch + 1000, Infinity, 40]
      ])
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  expect(prepared.kind).toBe('tabular-trips');
  expect(prepared.tripCount).toBe(2);
  expect(prepared.originMilliseconds).toBe(epoch);
  expect(prepared.animationDomainMilliseconds).toEqual([epoch, epoch + 2000]);
  // Trips belong to the batch of their first valid source row; trip "c" has one vertex.
  expect(prepared.batches.map(batch => Array.from(batch.sourceRows))).toEqual([[0, 2], []]);
  expect(Array.from(prepared.batches[0].memberRows)).toEqual([4, 0, 2, 5]);
  expect(Array.from(prepared.batches[0].memberOffsets)).toEqual([0, 2, 4]);
  expect(Array.from(prepared.batches[0].timestamps)).toEqual([0, 2000, 500, 1500]);
  expect(prepared.invalidRows.map(row => row.sourceRowIndex).sort()).toEqual([3, 6, 7]);
});

test('tabular trips enforce the explicit Float32 error budget and read Arrow batches', () => {
  const table = new arrow.Table([
    new arrow.RecordBatch(
      new arrow.Schema(
        ['id', 'time', 'lon', 'lat'].map(
          name => new arrow.Field(name, name === 'id' ? new arrow.Utf8() : new arrow.Float64())
        )
      ),
      arrow.makeData({
        type: new arrow.Struct(
          ['id', 'time', 'lon', 'lat'].map(
            name => new arrow.Field(name, name === 'id' ? new arrow.Utf8() : new arrow.Float64())
          )
        ),
        length: 2,
        nullCount: 0,
        children: [
          arrow.vectorFromArray(['x', 'x'], new arrow.Utf8()).data[0],
          arrow.vectorFromArray([0, 20_000_000_001], new arrow.Float64()).data[0],
          arrow.vectorFromArray([1, 2], new arrow.Float64()).data[0],
          arrow.vectorFromArray([3, 4], new arrow.Float64()).data[0]
        ]
      })
    )
  ]);
  const options = {timestampEncoding: 'unix-millisecond' as const, originMilliseconds: 0};
  expect(prepareKeplerTabularTripsFromArrow(table, columns, options).tripCount).toBe(0);
  expect(
    prepareKeplerTabularTripsFromArrow(table, columns, {
      ...options,
      maximumTimeErrorMilliseconds: 4
    }).tripCount
  ).toBe(1);
});

test('tabular trip uploads account the grouping index and release buffers', () => {
  const device = new NullDevice({});
  const prepared = prepareKeplerTabularTrips(
    [
      makeBatch('only', [
        ['a', epoch, 0, 0],
        ['a', epoch + 1, 1, 1]
      ])
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  const vectors = makeKeplerTabularTripGPUVectors(device, prepared);
  expect(vectors.membership?.memberRows.length).toBe(2);
  expect(vectors.membership?.memberOffsets.length).toBe(2);
  expect(vectors.resolveSourceRow(0)).toBe(0);
  expect(vectors.gpuByteLength).toBeGreaterThan(prepared.vertexCount * 16);
  vectors.destroy();
});

test('explicit tabular parts group independently across batches without compound-key collisions', () => {
  const source = prepareKeplerTabularTrips(
    [
      {
        ...makeBatch('first', [
          ['a', 20, 0, 0],
          ['a', 5, 10, 10],
          ['a|b', 0, 20, 20]
        ]),
        partIds: ['b|c', 'c', 'c']
      },
      {
        ...makeBatch('second', [
          ['a', 10, 1, 1],
          ['a', 0, 11, 11],
          ['a|b', 1, 21, 21]
        ]),
        partIds: ['b|c', 'c', 'c']
      }
    ],
    {timestampEncoding: 'unix-millisecond'}
  );
  expect(source.tripCount).toBe(3);
  expect(source.batches.map(batch => [...batch.sourceRows])).toEqual([[0, 1, 2], []]);
  expect([...source.batches[0].valueOffsets]).toEqual([0, 2, 4, 6]);
  expect([...source.batches[0].memberRows]).toEqual([3, 0, 4, 1, 2, 5]);
  expect([...source.batches[0].timestamps]).toEqual([10, 20, 0, 5, 0, 1]);
  expect(() =>
    prepareKeplerTabularTrips([{...makeBatch('bad', [['a', 0, 0, 0]]), partIds: []}], {
      timestampEncoding: 'unix-millisecond'
    })
  ).toThrow(/matching lengths/);
});

test.each([
  new arrow.TimestampMicrosecond('UTC'),
  new arrow.TimestampNanosecond('UTC')
])('typed Arrow trips normalize exact ticks and refuse hidden submillisecond loss: %s', type => {
  const scale = type.unit === arrow.TimeUnit.MICROSECOND ? 1000n : 1_000_000n;
  const makeTable = (offset: bigint) =>
    new arrow.Table({
      id: arrow.vectorFromArray(['route', 'route'], new arrow.Utf8()),
      time: arrow.makeVector(
        arrow.makeData({
          type,
          data: new BigInt64Array([BigInt(epoch) * scale + offset, BigInt(epoch + 1) * scale]),
          length: 2
        })
      ),
      lon: arrow.makeVector(new Float32Array([0, 1])),
      lat: arrow.makeVector(new Float32Array([0, 1]))
    });
  const exact = prepareKeplerTabularTripsFromArrow(makeTable(0n), columns, {
    timestampEncoding: 'unix-millisecond'
  });
  expect(exact.animationDomainMilliseconds).toEqual([epoch, epoch + 1]);
  expect([...exact.batches[0].timestamps]).toEqual([0, 1]);
  expect(() =>
    prepareKeplerTabularTripsFromArrow(makeTable(1n), columns, {
      timestampEncoding: 'unix-millisecond'
    })
  ).toThrow(/precision/);
});
