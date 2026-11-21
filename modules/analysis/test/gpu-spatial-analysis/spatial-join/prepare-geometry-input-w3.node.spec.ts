// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, it} from 'vitest';
import {prepareGeometryInput} from '../../../src/gpu-spatial-analysis/spatial-join/prepare-geometry-input';
import {computeSpatialPredicateJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {computeBufferSelectionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import {computePointInPolygonJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';

for (const separated of [false, true])
  for (const multi of [false, true]) {
    it(`ING-16 native ${multi ? 'MultiPolygon' : 'Polygon'} separated=${separated} retains a hole, two batches and slice identity`, () => {
      const coordinateType = separated
        ? new arrow.Struct([
            new arrow.Field('x', new arrow.Float64()),
            new arrow.Field('y', new arrow.Float64())
          ])
        : new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
      const ringType = new arrow.List(new arrow.Field('vertices', coordinateType));
      const polygonType = new arrow.List(new arrow.Field('rings', ringType));
      const type = multi ? new arrow.List(new arrow.Field('polygons', polygonType)) : polygonType;
      const coordinate = (x: number, y: number) => (separated ? {x, y} : [x, y]);
      const polygon = [
        [
          [10, 20],
          [30, 20],
          [30, 40],
          [10, 40],
          [10, 20]
        ].map(([x, y]) => coordinate(x, y)),
        [
          [15, 25],
          [15, 30],
          [20, 30],
          [20, 25],
          [15, 25]
        ].map(([x, y]) => coordinate(x, y))
      ];
      const feature = multi ? [polygon] : polygon;
      const first = arrow.vectorFromArray([null, feature], type);
      const second = arrow.vectorFromArray([feature, [], null], type);
      const vector = first.concat(second).slice(1, 5);
      const prepared = prepareGeometryInput(vector, {
        sourceRowOffset: 40,
        sourceIds: Uint32Array.of(99, 1, 7, 8),
        encoding: multi ? 'geoarrow.multipolygon' : 'geoarrow.polygon'
      });
      expect(Array.from(prepared.sourceRows)).toEqual([40, 41, 42, 43]);
      expect(Array.from(prepared.sourceIds)).toEqual([99, 1, 7, 8]);
      expect(Array.from(prepared.batchRowOffsets)).toEqual([0, 1, 4]);
      expect(Array.from(prepared.validity)).toEqual([1, 1, 1, 0]);
      expect(Array.from(prepared.empty)).toEqual([0, 0, 1, 0]);
      expect(Array.from(prepared.geometry.positions.slice(0, 4))).toEqual([10, 20, 30, 20]);
      expect(Array.from(prepared.geometry.ringOffsets!)).toEqual([0, 5, 10, 15, 20]);
      const result = computeSpatialPredicateJoinOnCPU(
        {kind: 'points', positions: Float32Array.of(12, 22, 17, 27)},
        prepared.geometry,
        {predicate: 'within', candidateCapacity: 8, pairCapacity: 8}
      );
      // The first point belongs to both polygon rows; the second is strictly in each hole.
      expect(result.status.count).toBe(2);
      expect(Array.from(result.leftRows.slice(0, 2))).toEqual([0, 0]);
      expect(Array.from(result.rightRows.slice(0, 2))).toEqual([0, 1]);
    });
  }

it('ING-16 rejects silent float64 collapse and reports an explicit lossy conversion', () => {
  const type = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
  const vector = arrow.vectorFromArray(
    [
      [2 ** 30, 1],
      [2 ** 30 + 1, 1]
    ],
    type
  );
  expect(() => prepareGeometryInput(vector)).toThrow(/precision.*allow-loss/);
  const prepared = prepareGeometryInput(vector, {precision: 'allow-loss'});
  expect(prepared.roundedCoordinateCount).toBe(1);
  expect(prepared.collapsedPositionCount).toBe(1);
  expect(Array.from(prepared.geometry.positions)).toEqual([2 ** 30, 1, 2 ** 30, 1]);
});

it('ING-16 rejects serialized and unsupported option schemas', () => {
  expect(() => prepareGeometryInput(arrow.vectorFromArray(['POINT(0 0)']))).toThrow();
  const vector = arrow.vectorFromArray(
    [[1, 2]],
    new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()))
  );
  expect(() => prepareGeometryInput(vector, {precision: 'unknown' as never})).toThrow(
    /reject-loss.*allow-loss/
  );
});

it('ING-16 distinguishes native empty Point from null and snapshots the supplied frame', () => {
  const vector = arrow.vectorFromArray(
    [[Number.NaN, Number.NaN], null, [1, 2]],
    new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()))
  );
  const frame = {
    id: 'local',
    kind: 'local-azimuthal-equidistant' as const,
    origin: [10, 20] as [number, number],
    unit: {name: 'meters', metersPerUnit: 1}
  };
  const prepared = prepareGeometryInput(vector, {frame});
  expect(Array.from(prepared.validity)).toEqual([1, 0, 1]);
  expect(Array.from(prepared.empty)).toEqual([1, 0, 0]);
  expect(Array.from(prepared.geometry.featureOffsets!)).toEqual([0, 0, 0, 1]);
  expect(Array.from(prepared.geometry.positions)).toEqual([1, 2]);
  frame.origin[0] = 99;
  expect(prepared.frame?.origin).toEqual([10, 20]);
});

it('B5 native Point preserves single-point queries and buffer reference ports', () => {
  const vector = arrow.vectorFromArray(
    [
      [3, 6],
      [7, 8]
    ],
    new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()))
  );
  const prepared = prepareGeometryInput(vector);
  expect(prepared.geometry.featureOffsets).toBeUndefined();
  const result = computeBufferSelectionOnCPU(
    {points: prepared.geometry, features: prepared.geometry, radius: 0},
    {candidateCapacity: 4, matchCapacity: 2}
  );
  expect(Array.from(result.nearestFeatureRows)).toEqual([0, 1]);
  expect(Array.from(result.outputMask)).toEqual([1, 1]);
  const join = computePointInPolygonJoinOnCPU(
    {
      points: prepared.geometry,
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 10, 0, 10, 10, 0, 10, 0, 0, 10, 0, 10, 10, 0, 10),
        featureOffsets: Uint32Array.of(0, 1, 2),
        polygonOffsets: Uint32Array.of(0, 1, 2),
        ringOffsets: Uint32Array.of(0, 4, 8)
      }
    },
    {candidateCapacity: 8, matchCapacity: 2}
  );
  expect(Array.from(join.polygonAssignmentCounts)).toEqual([2, 0]);
});
it('B6 explicit options cannot override authoritative GeoArrow field semantics', () => {
  const coordinateType = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
  const type = new arrow.List(
    new arrow.Field('rings', new arrow.List(new arrow.Field('vertices', coordinateType)))
  );
  const vector = arrow.vectorFromArray(
    [
      [
        [
          [0, 0],
          [1, 0],
          [0, 1],
          [0, 0]
        ]
      ]
    ],
    type
  );
  const field = new arrow.Field(
    'geometry',
    type,
    true,
    new Map([
      ['ARROW:extension:name', 'geoarrow.polygon'],
      ['ARROW:extension:metadata', JSON.stringify({edges: 'spherical'})]
    ])
  );
  const planarField = new arrow.Field(
    'geometry',
    type,
    true,
    new Map([['ARROW:extension:name', 'geoarrow.polygon']])
  );
  expect(() =>
    prepareGeometryInput(vector, {field: planarField, encoding: 'geoarrow.multilinestring'})
  ).toThrow(/encoding.*conflict/);
  expect(() => prepareGeometryInput(vector, {field, metadata: {edges: 'planar'}})).toThrow(
    /metadata.*edges.*conflict/
  );
});
it('B7 collapsedPositionCount counts distinct originals beyond one per rounded XY independent of row order', () => {
  const type = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
  const count = (xs: number[]) =>
    prepareGeometryInput(
      arrow.vectorFromArray(
        xs.map(x => [x, 0]),
        type
      ),
      {precision: 'allow-loss'}
    ).collapsedPositionCount;
  expect(count([16777216, 16777217, 16777217])).toBe(1);
  expect(count([16777217, 16777217, 16777216])).toBe(1);
  const batches = arrow.vectorFromArray([[16777216, 0]], type).concat(
    arrow.vectorFromArray(
      [
        [16777217, 0],
        [16777217, 0]
      ],
      type
    )
  );
  expect(prepareGeometryInput(batches, {precision: 'allow-loss'}).collapsedPositionCount).toBe(1);
});
