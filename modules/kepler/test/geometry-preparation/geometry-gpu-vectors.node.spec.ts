// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makeKeplerGeometryGPUVectors,
  prepareKeplerGeometrySource
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {makeSourceRowResolver} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/gpu-upload-utils';
import {geometryBatches} from './geometry-fixture';

const SQUARE = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
  [0, 0]
];

it('prepares a very large MultiPoint without overflowing the call stack', () => {
  const coordinates = Array.from({length: 200_000}, (_, index) => [index % 100, index % 50]);
  const source = prepareKeplerGeometrySource([
    {id: 'large', rows: [{type: 'MultiPoint', coordinates}]}
  ]);
  expect(source.metadata.invalidRows).toEqual([]);
  expect(source.batches[0].points.sourceRows.length).toBe(200_000);
});

it('prepares large MultiLineString and MultiPolygon rows', () => {
  const lines = Array.from({length: 150_000}, () => [
    [0, 0],
    [1, 1]
  ]);
  const polygons = Array.from({length: 150_000}, () => [SQUARE]);
  const source = prepareKeplerGeometrySource([
    {
      id: 'large',
      rows: [
        {type: 'MultiLineString', coordinates: lines},
        {type: 'MultiPolygon', coordinates: polygons}
      ]
    }
  ]);
  expect(source.metadata.invalidRows).toEqual([]);
  expect(source.batches[0].paths.sourceRows.length).toBe(150_000);
});

it('prepares JSON string rows once with the same result as parsed rows', () => {
  const rows = [
    {
      type: 'Feature',
      properties: {lineWidth: 3},
      geometry: {type: 'LineString', coordinates: SQUARE}
    },
    {type: 'Point', coordinates: [5, 6]}
  ];
  const fromObjects = prepareKeplerGeometrySource([{id: 'a', rows}]);
  const fromStrings = prepareKeplerGeometrySource([
    {id: 'a', rows: rows.map(row => JSON.stringify(row))}
  ]);
  expect(fromStrings.metadata).toEqual(fromObjects.metadata);
  expect([...fromStrings.batches[0].representative!.positions]).toEqual([
    ...fromObjects.batches[0].representative!.positions
  ]);
  expect(fromStrings.metadata.sourceStyleProperties).toEqual(['lineWidth']);
  const malformed = prepareKeplerGeometrySource([{id: 'a', rows: ['{not json', rows[1]]}]);
  expect(malformed.metadata.invalidRows.map(row => row.sourceRowIndex)).toEqual([0]);
});

it('uploads batches with ownership, empty batches, and double destroy', () => {
  const device = new NullDevice({});
  const source = prepareKeplerGeometrySource(geometryBatches);
  const vectors = makeKeplerGeometryGPUVectors(device, source);
  expect(vectors.batches.map(batch => batch.id)).toEqual(['first', 'empty', 'last']);
  expect(vectors.points.positions.data.length).toBe(3);
  expect(vectors.points.positions.data.every(chunk => !chunk.ownsBuffer)).toBe(true);
  expect(vectors.paths.getPath.data.length).toBe(3);
  expect(vectors.polygons.positions.data.length).toBe(3);
  // The empty batch still has a valid (4 byte) buffer.
  expect(vectors.points.positions.data[1].buffer.byteLength).toBeGreaterThanOrEqual(4);
  expect(vectors.points.resolveSourceRow(0)).toBe(0);
  expect(vectors.points.resolveSourceRow(2)).toBe(4);
  expect(vectors.points.resolveSourceRow(3)).toBe(-1);
  expect(vectors.points.resolveSourceRow(-1)).toBe(-1);
  expect(vectors.buffers.length).toBeGreaterThan(0);
  expect(vectors.gpuByteLength).toBeGreaterThan(0);
  vectors.destroy();
  expect(vectors.buffers.every(buffer => buffer.destroyed)).toBe(true);
  expect(() => vectors.destroy()).not.toThrow();
  device.destroy();
});

it('uploads an empty source and preserves invalid rows', () => {
  const device = new NullDevice({});
  const empty = makeKeplerGeometryGPUVectors(device, prepareKeplerGeometrySource([]));
  expect(empty.points.positions.data.length).toBe(0);
  empty.destroy();
  const invalid = prepareKeplerGeometrySource([
    {id: 'bad', rows: [{type: 'Point', coordinates: ['x', 1]}]}
  ]);
  expect(invalid.metadata.invalidRows.length).toBe(1);
  const vectors = makeKeplerGeometryGPUVectors(device, invalid);
  expect(vectors.points.resolveSourceRow(0)).toBe(-1);
  vectors.destroy();
  device.destroy();
});

it('uploads fp64 low parts and multiple batches', () => {
  const device = new NullDevice({});
  const wide = SQUARE.map(([x, y]) => [x + 0.1234567891234, y + 0.1234567891234]);
  const source = prepareKeplerGeometrySource(
    [
      {id: 'one', rows: [{type: 'LineString', coordinates: wide}]},
      {id: 'two', rows: [{type: 'Polygon', coordinates: [wide]}]}
    ],
    {fp64: true}
  );
  const vectors = makeKeplerGeometryGPUVectors(device, source);
  expect(vectors.paths.getPathLow?.data.length).toBe(2);
  expect(vectors.polygons.positionsLow?.data.length).toBe(2);
  vectors.destroy();
  device.destroy();
});

it('omits fp64 low parts when the coordinates are Float32-representable', () => {
  const pointsOnly = prepareKeplerGeometrySource(
    [{id: 'points', rows: [{type: 'Point', coordinates: [1, 2]}]}],
    {fp64: true}
  );
  expect(pointsOnly.fp64).toBeUndefined();
  const polygonOnly = prepareKeplerGeometrySource(
    [{id: 'polygon', rows: [{type: 'Polygon', coordinates: [SQUARE]}]}],
    {fp64: true}
  );
  expect(polygonOnly.fp64).toBeUndefined();
  expect(polygonOnly.batches[0].paths.positionsLow).toBeUndefined();
  expect(polygonOnly.batches[0].polygonOutlines.positionsLow).toBeUndefined();
});

it('resolves source rows across batches with a binary search', () => {
  const resolve = makeSourceRowResolver([
    Uint32Array.of(10, 11),
    new Uint32Array(0),
    Uint32Array.of(20),
    Uint32Array.of(30, 31, 32)
  ]);
  expect([0, 1, 2, 3, 4, 5, 6].map(resolve)).toEqual([10, 11, 20, 30, 31, 32, -1]);
  expect(resolve(-1)).toBe(-1);
  expect(resolve(1.5)).toBe(-1);
  expect(makeSourceRowResolver([])(0)).toBe(-1);
});
