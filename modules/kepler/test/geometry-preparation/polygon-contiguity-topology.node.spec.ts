// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareKeplerGeometrySource} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {
  preparePolygonContiguityTopology,
  POLYGON_CONTIGUITY_MAXIMUM_ROWS,
  POLYGON_CONTIGUITY_MAXIMUM_VERTICES,
  type PreparedPolygonTopologySource
} from '@luma-contributors/analysis/gpu-spatial-analysis';

function polygon(coordinates: number[][][]) {
  return {type: 'Polygon', coordinates};
}
function square(left: number, bottom = 0) {
  return [
    [left, bottom],
    [left + 1, bottom],
    [left + 1, bottom + 1],
    [left, bottom + 1],
    [left, bottom]
  ];
}
function vertexIds(positions: Float32Array) {
  return Array.from({length: positions.length / 2}, (_, index) => positions[index * 2]);
}

test('exact source XY tokens preserve cross-batch adjacency without Float32 collapse', () => {
  const offset = 1 + 2 ** -30;
  expect(Math.fround(offset)).toBe(1);
  const source = prepareKeplerGeometrySource([
    {id: 'first', rows: [polygon([square(0)])]},
    {id: 'empty', rows: []},
    {id: 'second', rows: [polygon([square(1)]), polygon([square(offset)])]}
  ]);
  const topology = preparePolygonContiguityTopology(source);
  expect(topology.batches).toEqual([
    {id: 'first', rowCount: 1, sourceRowOffset: 0},
    {id: 'empty', rowCount: 0, sourceRowOffset: 1},
    {id: 'second', rowCount: 2, sourceRowOffset: 1}
  ]);
  expect([...topology.featureOffsets]).toEqual([0, 1, 2, 3]);
  expect([...topology.polygonOffsets]).toEqual([0, 1, 2, 3]);
  expect([...topology.ringOffsets]).toEqual([0, 4, 8, 12]);
  const identifiers = vertexIds(topology.positions);
  expect(identifiers[1]).toBe(identifiers[4]);
  expect(identifiers[2]).toBe(identifiers[7]);
  expect(identifiers[4]).not.toBe(identifiers[8]);
  expect(identifiers[7]).not.toBe(identifiers[11]);
  expect([...topology.geometryMask]).toEqual([1, 1, 1]);
  expect(topology.uniqueVertexCount).toBe(10);
  expect(topology).toMatchObject({
    equality: 'exact-source-xy',
    ignoresZ: true,
    topologyEncoding: 'source-xy-key-ids'
  });
  expect(topology.cpuByteLength).toBe(
    topology.positions.byteLength +
      topology.ringOffsets.byteLength +
      topology.polygonOffsets.byteLength +
      topology.featureOffsets.byteLength +
      topology.geometryMask.byteLength +
      topology.batches.reduce((total, batch) => total + 64 + batch.id.length * 2, 0)
  );
});

test('sliced and chunked polygon vectors retain null/empty rows and source offsets', () => {
  const source = prepareKeplerGeometrySource([
    {
      id: 'original',
      rows: [polygon([square(20)]), polygon([square(0)]), null, polygon([square(1)])]
    }
  ]);
  const polygons = source.batches[0].polygons;
  const sliced = polygons.slice(1, 3);
  const chunked = new arrow.Vector([...sliced.data, ...polygons.slice(3).data]);
  const subset: PreparedPolygonTopologySource = {
    metadata: {...source.metadata, rowCount: 3, invalidRows: []},
    batches: [{id: 'subset', rowCount: 3, sourceRowOffset: 0, polygons: chunked}]
  };
  const topology = preparePolygonContiguityTopology(subset);
  expect([...topology.geometryMask]).toEqual([1, 0, 1]);
  expect([...topology.featureOffsets]).toEqual([0, 1, 1, 2]);
  expect(topology.invalidRows).toEqual([]);
  expect(topology.vertexCount).toBe(8);
  const identifiers = vertexIds(topology.positions);
  expect(identifiers[1]).toBe(identifiers[4]);
  expect(identifiers[2]).toBe(identifiers[7]);
});

test('holes, multipart collections, repeated closure and ignored Z retain whole-edge topology', () => {
  const outer = square(0);
  outer.push([0, 0, 55], [-0, -0, 66]);
  const source = prepareKeplerGeometrySource([
    {
      id: 'parts',
      rows: [
        {
          type: 'GeometryCollection',
          geometries: [
            polygon([outer, square(0.2, 0.2)]),
            {type: 'MultiPolygon', coordinates: [[square(5)]]},
            {type: 'Point', coordinates: [0, 0]}
          ]
        }
      ]
    }
  ]);
  const topology = preparePolygonContiguityTopology(source);
  expect([...topology.featureOffsets]).toEqual([0, 2]);
  expect([...topology.polygonOffsets]).toEqual([0, 2, 3]);
  expect([...topology.ringOffsets]).toEqual([0, 4, 8, 12]);
  expect(topology.vertexCount).toBe(12);
  expect(topology.invalidRows).toEqual([]);
});

test('malformed nested geometry rejects the entire row without retaining its preceding valid ring', () => {
  const source = prepareKeplerGeometrySource([
    {id: 'invalid', rows: [polygon([square(0)]), polygon([square(1)]), polygon([square(2)])]}
  ]);
  const type = source.batches[0].polygons.type;
  const coordinate = (position: number[]) => [position[0], position[1], 0];
  const malformed = arrow.vectorFromArray(
    [
      [
        [
          square(0).map(coordinate),
          [
            [10, 10, 0],
            [11, 10, 0],
            [NaN, 11, 0],
            [10, 10, 0]
          ]
        ]
      ],
      [[square(1).map(coordinate)]],
      [[null]]
    ],
    type
  );
  const input = {...source, batches: [{...source.batches[0], polygons: malformed}]};
  const topology = preparePolygonContiguityTopology(input);
  expect([...topology.geometryMask]).toEqual([0, 1, 0]);
  expect([...topology.featureOffsets]).toEqual([0, 0, 1, 1]);
  expect(vertexIds(topology.positions)).toEqual([0, 1, 2, 3]);
  expect(topology.invalidRows).toEqual([
    {sourceRowIndex: 0, reason: 'invalid-topology'},
    {sourceRowIndex: 2, reason: 'invalid-topology'}
  ]);
});

test('source-invalid rows and malformed offsets cannot contribute partial topology', () => {
  const source = prepareKeplerGeometrySource([
    {id: 'offsets', rows: [polygon([square(0)]), polygon([square(1)])]}
  ]);
  source.metadata.invalidRows = [{sourceRowIndex: 0, reason: 'upstream rejected geometry'}];
  const data = source.batches[0].polygons.data[0];
  const offsets = data.valueOffsets.slice();
  offsets[2] = data.children[0].length + 1;
  const malformed = new arrow.Vector([
    data.clone(
      data.type,
      data.offset,
      data.length,
      data.nullCount,
      [offsets, data.values, data.nullBitmap, data.typeIds],
      data.children
    )
  ]);
  const topology = preparePolygonContiguityTopology({
    ...source,
    batches: [{...source.batches[0], polygons: malformed}]
  });
  expect([...topology.geometryMask]).toEqual([0, 0]);
  expect([...topology.featureOffsets]).toEqual([0, 0, 0]);
  expect(topology.positions).toHaveLength(0);
  expect([...topology.ringOffsets]).toEqual([0]);
  expect([...topology.polygonOffsets]).toEqual([0]);
  expect(topology.invalidRows.map(row => row.reason)).toEqual([
    'source-invalid',
    'invalid-topology'
  ]);
});

test('bounded preparation rejects excess rows or decoded vertices before token allocation', () => {
  const source = prepareKeplerGeometrySource([{id: 'limit', rows: [polygon([square(0)])]}]);
  expect(() =>
    preparePolygonContiguityTopology({
      ...source,
      metadata: {...source.metadata, rowCount: POLYGON_CONTIGUITY_MAXIMUM_ROWS + 1}
    })
  ).toThrow('row capacity');
  expect(() =>
    preparePolygonContiguityTopology({
      ...source,
      metadata: {
        ...source.metadata,
        invalidRows: [
          {sourceRowIndex: 0, reason: 'invalid'},
          {sourceRowIndex: 0, reason: 'duplicate'}
        ]
      }
    })
  ).toThrow('diagnostics exceed');
  const ring = Array.from({length: POLYGON_CONTIGUITY_MAXIMUM_VERTICES + 1}, (_, index) => [
    index,
    0,
    0
  ]);
  const polygons = arrow.vectorFromArray([[[ring]]], source.batches[0].polygons.type);
  expect(() =>
    preparePolygonContiguityTopology({...source, batches: [{...source.batches[0], polygons}]})
  ).toThrow('vertex capacity');
});
