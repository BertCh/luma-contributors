// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {computePolygonContiguityOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {makeExactPolygonTopology} from './polygon-contiguity-preparation-fixture';

test('CPU route shares an exact whole edge but refuses a binary64 gap hidden by Float32', () => {
  const topology = makeExactPolygonTopology();
  expect(topology.equality).toBe('exact-source-xy');
  expect(topology.topologyEncoding).toBe('source-xy-key-ids');
  expect([...topology.geometryMask]).toEqual([1, 1, 1]);
  for (const criterion of ['queen', 'rook'] as const) {
    const result = computePolygonContiguityOnCPU(topology, {criterion, capacity: 2});
    expect([...result.offsets]).toEqual([0, 1, 2, 2]);
    expect([...result.neighbors]).toEqual([1, 0]);
    expect([...result.weights]).toEqual([1, 1]);
    expect([...result.status.requiredCount]).toEqual([2]);
    expect([...result.status.overflow]).toEqual([0]);
    expect([...result.status.candidateOverflow]).toEqual([0]);
    const truncated = computePolygonContiguityOnCPU(topology, {criterion, capacity: 1});
    expect([...truncated.offsets]).toEqual([0, 1, 1, 1]);
    expect([...truncated.status.requiredCount]).toEqual([2]);
    expect([...truncated.status.overflow]).toEqual([1]);
  }
});

test('all empty source rows stay aligned without passing degenerate offset arrays to the twin', () => {
  const topology = makeExactPolygonTopology(true);
  const result = computePolygonContiguityOnCPU(topology, {criterion: 'rook', capacity: 0});
  expect([...topology.geometryMask]).toEqual([0, 0, 0]);
  expect([...result.offsets]).toEqual([0, 0, 0, 0]);
  expect([...result.status.requiredCount]).toEqual([0]);
  expect([...result.status.overflow]).toEqual([0]);
});

import * as arrow from 'apache-arrow';
import {preparePolygonContiguityTopology} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
test('ING-16 native Polygon keeps batch rows, holes and exact original coordinate equality', () => {
  const vertex = new arrow.FixedSizeList(2, new arrow.Field('coordinate', new arrow.Float64()));
  const ring = new arrow.List(new arrow.Field('vertex', vertex));
  const polygon = new arrow.List(new arrow.Field('ring', ring));
  const triangle = [
    [10, 20],
    [30, 20],
    [10, 35],
    [10, 20]
  ];
  const first = arrow.vectorFromArray([[triangle]], polygon);
  const second = arrow.vectorFromArray(
    [
      [triangle.map(([coordinateX, coordinateY]) => [coordinateX + 2 ** -30, coordinateY])],
      null,
      []
    ],
    polygon
  );
  const topology = preparePolygonContiguityTopology({
    metadata: {rowCount: 4, invalidRows: []},
    batches: [
      {id: 'first', rowCount: 1, sourceRowOffset: 0, polygons: first},
      {id: 'second', rowCount: 3, sourceRowOffset: 1, polygons: second}
    ]
  });
  expect([...topology.geometryMask]).toEqual([1, 1, 0, 0]);
  expect([...topology.featureOffsets]).toEqual([0, 1, 2, 2, 2]);
  expect([...topology.ringOffsets]).toEqual([0, 3, 6]);
  expect(topology.invalidRows).toEqual([]);
  // All three shifted x values round to the same float32 coordinates but remain distinct tokens.
  expect(topology.uniqueVertexCount).toBe(6);
  expect([
    ...computePolygonContiguityOnCPU(topology, {criterion: 'queen', capacity: 2}).offsets
  ]).toEqual([0, 0, 0, 0, 0]);
});

test('ING-16 retains multipart holes and original-coordinate equality outside metric float32 range', () => {
  const vertex = new arrow.FixedSizeList(2, new arrow.Field('coordinate', new arrow.Float64()));
  const ring = new arrow.List(new arrow.Field('vertex', vertex));
  const polygon = new arrow.List(new arrow.Field('ring', ring));
  const feature = new arrow.List(new arrow.Field('polygon', polygon));
  const outer = [
    [0, 0],
    [4, 0],
    [0, 4],
    [0, 0]
  ];
  const hole = [
    [1, 1],
    [2, 1],
    [1, 2],
    [1, 1]
  ];
  const island = [
    [10, 10],
    [11, 10],
    [10, 11],
    [10, 10]
  ];
  const polygons = arrow.vectorFromArray([[[outer, hole], [island]], [[hole]]], feature);
  const topology = preparePolygonContiguityTopology({
    metadata: {rowCount: 2, invalidRows: []},
    batches: [{id: 'parts', rowCount: 2, sourceRowOffset: 0, polygons}]
  });
  expect([...topology.featureOffsets]).toEqual([0, 2, 3]);
  expect([...topology.polygonOffsets]).toEqual([0, 2, 3, 4]);
  expect([...topology.ringOffsets]).toEqual([0, 3, 6, 9, 12]);
  expect([
    ...computePolygonContiguityOnCPU(topology, {criterion: 'rook', capacity: 2}).neighbors
  ]).toEqual([1, 0]);
  // One native Polygon ring with no finite float32 metric representation.
  const hugePolygon = arrow.vectorFromArray(
    [
      [
        [
          [1e100, 0],
          [2e100, 0],
          [1e100, 1],
          [1e100, 0]
        ]
      ]
    ],
    polygon
  );
  const exact = preparePolygonContiguityTopology({
    metadata: {rowCount: 1, invalidRows: []},
    batches: [{id: 'huge', rowCount: 1, sourceRowOffset: 0, polygons: hugePolygon}]
  });
  expect([...exact.geometryMask]).toEqual([1]);
  expect(exact.uniqueVertexCount).toBe(3);
});
