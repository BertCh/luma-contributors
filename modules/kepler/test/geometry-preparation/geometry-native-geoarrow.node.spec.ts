// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareNativeGeoArrowSource} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-native-geoarrow';
import {
  prepareKeplerGeometrySourceFromArrow,
  type PreparedKeplerGeometrySource
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {
  convertGeoArrowVectorToInterleaved,
  type GeoArrowNativeEncoding
} from '@luma.gl/graph/arrow';

const SQUARE = [
  [0, 0, 5],
  [4, 0, 6],
  [4, 4, 7],
  [0, 4, 8]
];
const HOLE = [
  [1, 1, 1],
  [2, 1, 1],
  [2, 2, 1],
  [1, 1, 1]
];

type Fixture = {
  encoding: GeoArrowNativeEncoding;
  depth: number;
  rows: unknown[];
};

const FIXTURES: Fixture[] = [
  {encoding: 'geoarrow.point', depth: 0, rows: [[1, 2, 3], null, [4.25, 5, 0], [7, 8, 9]]},
  {
    encoding: 'geoarrow.multipoint',
    depth: 1,
    rows: [
      [[1, 2, 3]],
      [],
      null,
      [
        [4, 5, 6],
        [7, 8, 9]
      ]
    ]
  },
  {
    encoding: 'geoarrow.linestring',
    depth: 1,
    rows: [
      [
        [0.1, 0.2, 0],
        [3, 4, 1]
      ],
      null,
      [],
      [[5, 6, 0]]
    ]
  },
  {
    encoding: 'geoarrow.multilinestring',
    depth: 2,
    rows: [
      [
        [
          [0.1, 0.2, 0],
          [3, 4, 1]
        ],
        []
      ],
      null,
      [[[5, 6, 7]]]
    ]
  },
  {
    encoding: 'geoarrow.polygon',
    depth: 2,
    rows: [[SQUARE, HOLE], null, [], [[], SQUARE.slice(0, 3)], [[[0, 0, 0]], []]]
  },
  {
    encoding: 'geoarrow.multipolygon',
    depth: 3,
    rows: [
      [[SQUARE, HOLE], [SQUARE.slice().reverse()]],
      null,
      [],
      [[[]], [SQUARE]],
      [[SQUARE.map(vertex => [vertex[0] * 1e-9 + 0.1, vertex[1], vertex[2]])]],
      [[[[0, 1, 2]]]]
    ]
  }
];

function makeInterleavedType(fixture: Fixture, size: 2 | 3): arrow.DataType {
  let type: arrow.DataType = new arrow.FixedSizeList(
    size,
    new arrow.Field(size === 2 ? 'xy' : 'xyz', new arrow.Float64(), false)
  );
  for (let depth = 0; depth < fixture.depth; depth++)
    type = new arrow.List(new arrow.Field('child', type, false));
  return type;
}

function dropZ(value: unknown, depth: number): unknown {
  if (value === null) return null;
  return depth === 0
    ? (value as number[]).slice(0, 2)
    : (value as unknown[]).map(child => dropZ(child, depth - 1));
}

function makeVector(fixture: Fixture, size: 2 | 3): arrow.Vector {
  const rows = size === 3 ? fixture.rows : fixture.rows.map(row => dropZ(row, fixture.depth));
  return arrow.vectorFromArray(rows, makeInterleavedType(fixture, size));
}

function toComparable(source: PreparedKeplerGeometrySource) {
  return {
    metadata: source.metadata,
    recipe: source.representativeRecipe,
    fp64: source.fp64,
    batches: source.batches.map(batch => ({
      id: batch.id,
      rowCount: batch.rowCount,
      sourceRowOffset: batch.sourceRowOffset,
      points: [...batch.points.positions, ...batch.points.sourceRows],
      paths: [
        [...batch.paths.positions],
        [...batch.paths.valueOffsets],
        [...batch.paths.sourceRows],
        batch.paths.positionsLow ? [...batch.paths.positionsLow] : null
      ],
      outlines: [
        [...batch.polygonOutlines.positions],
        [...batch.polygonOutlines.valueOffsets],
        [...batch.polygonOutlines.sourceRows],
        [...batch.polygonOutlines.sideOrientations!],
        batch.polygonOutlines.positionsLow ? [...batch.polygonOutlines.positionsLow] : null
      ],
      polygons: JSON.stringify(batch.polygons.toJSON()),
      representative: [...batch.representative!.positions, ...batch.representative!.validity]
    }))
  };
}

function expectFastEqualsGeneric(
  vector: arrow.Vector,
  encoding: GeoArrowNativeEncoding,
  dimension: 'xy' | 'xyz',
  fp64 = false
) {
  const generic = prepareKeplerGeometrySourceFromArrow(vector, {
    encoding,
    nativeFastPath: false,
    fp64
  });
  const fast = prepareKeplerGeometrySourceFromArrow(vector, {encoding, fp64});
  // The fast route must actually accept these buffers (no silent generic fallback).
  expect(prepareNativeGeoArrowSource(vector, {encoding, dimension, fp64})).not.toBeNull();
  expect(toComparable(fast)).toEqual(toComparable(generic));
  return fast;
}

function toSeparated(vector: arrow.Vector, size: 2 | 3): arrow.Vector {
  // Rebuild the leaf coordinate arrays as Struct<x, y, z> while keeping list structure.
  const rebuild = (data: arrow.Data): arrow.Data => {
    if (arrow.DataType.isFixedSizeList(data.type)) {
      const values = data.children[0].values as Float64Array;
      const names = ['x', 'y', 'z'].slice(0, size);
      const children = names.map((_, component) => {
        const column = new Float64Array(data.length);
        for (let index = 0; index < data.length; index++)
          column[index] = values[index * size + component];
        return arrow.makeData({type: new arrow.Float64(), length: data.length, data: column});
      });
      const type = new arrow.Struct(
        names.map(name => new arrow.Field(name, new arrow.Float64(), false))
      );
      return arrow.makeData({
        type,
        length: data.length,
        nullCount: data.nullCount,
        nullBitmap: data.nullBitmap,
        children
      });
    }
    const child = rebuild(data.children[0]);
    return arrow.makeData({
      type: new arrow.List(new arrow.Field('child', child.type, false)),
      length: data.length,
      nullCount: data.nullCount,
      nullBitmap: data.nullBitmap,
      valueOffsets: data.valueOffsets,
      child
    });
  };
  return new arrow.Vector(vector.data.map(rebuild));
}

describe('native GeoArrow fast path', () => {
  for (const fixture of FIXTURES)
    for (const size of [2, 3] as const) {
      it(`${fixture.encoding} xy${size === 3 ? 'z' : ''} matches the generic route`, () => {
        const vector = makeVector(fixture, size);
        const fast = expectFastEqualsGeneric(vector, fixture.encoding, size === 3 ? 'xyz' : 'xy');
        expect(fast.metadata.rowCount).toBe(fixture.rows.length);
        expectFastEqualsGeneric(vector, fixture.encoding, size === 3 ? 'xyz' : 'xy', true);
      });

      it(`${fixture.encoding} xy${size === 3 ? 'z' : ''} separated matches interleaved`, () => {
        const interleaved = makeVector(fixture, size);
        const separated = toSeparated(interleaved, size);
        const fast = prepareKeplerGeometrySourceFromArrow(separated, {encoding: fixture.encoding});
        const reference = prepareKeplerGeometrySourceFromArrow(interleaved, {
          encoding: fixture.encoding,
          nativeFastPath: false
        });
        expect(toComparable(fast)).toEqual(toComparable(reference));
        const converted = convertGeoArrowVectorToInterleaved(separated, {
          encoding: fixture.encoding
        });
        expect(
          toComparable(
            prepareKeplerGeometrySourceFromArrow(converted, {encoding: fixture.encoding})
          )
        ).toEqual(toComparable(reference));
      });
    }

  it('preserves chunk boundaries and slice offsets', () => {
    for (const fixture of FIXTURES) {
      const full = makeVector(fixture, 3);
      const sliced = full.slice(1, fixture.rows.length);
      const chunked = new arrow.Vector([
        ...makeVector(fixture, 3).slice(0, 2).data,
        ...arrow.vectorFromArray([], makeInterleavedType(fixture, 3)).data,
        ...makeVector(fixture, 3).slice(2).data
      ]);
      const separatedSlice = toSeparated(makeVector(fixture, 3), 3).slice(1);
      for (const vector of [sliced, chunked, separatedSlice]) {
        const fast = expectFastEqualsGeneric(vector, fixture.encoding, 'xyz');
        expect(fast.batches.length).toBe(vector.data.length);
      }
    }
  });

  it('rejects invalid rows like the generic route and keeps their source row space', () => {
    const type = makeInterleavedType(FIXTURES[1], 2);
    const vector = arrow.vectorFromArray([[[1, 2]], [[NaN, 1]], [[3e39, 0]], [[4, 5]]], type);
    const fast = expectFastEqualsGeneric(vector, 'geoarrow.multipoint', 'xy');
    expect(fast.metadata.invalidRows.map(row => row.sourceRowIndex)).toEqual([1, 2]);
    expect([...fast.batches[0].points.sourceRows]).toEqual([0, 3]);
  });

  it('reuses the Arrow offset buffers for polygon topology without per-row objects', () => {
    const fixture = FIXTURES[5];
    const vector = makeVector(fixture, 3);
    const fast = prepareKeplerGeometrySourceFromArrow(vector, {encoding: fixture.encoding});
    expect(fast.batches[0].polygons.length).toBe(fixture.rows.length);
    expect(fast.representativeRecipe).toBe('geoarrow-mean-center');
  });
});
