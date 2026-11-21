// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  makeGeoArrowColumnFromCatalogGeometries,
  readGeoArrowMultiPoint,
  readGeoArrowMultiLineString,
  readGeoArrowMultiPolygon
} from '../src/analysis/catalog-geoarrow';
it('ING-16 reads actual XY and feature order across multipoint chunks', () => {
  const first = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiPoint',
        coordinates: [
          [10, 20],
          [30, 40]
        ]
      }
    ],
    'geoarrow.multipoint'
  );
  const second = makeGeoArrowColumnFromCatalogGeometries(
    [{type: 'MultiPoint', coordinates: [[50, 60]]}],
    'geoarrow.multipoint'
  );
  const result = readGeoArrowMultiPoint({...first, chunks: [...first.chunks, ...second.chunks]});
  expect(Array.from(result.featureOffsets)).toEqual([0, 2, 3]);
  expect(Array.from(result.positions)).toEqual([10, 20, 30, 40, 50, 60]);
});
it('ING-16 rebases feature and line offsets across multiline chunks', () => {
  const first = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiLineString',
        coordinates: [
          [
            [10, 20],
            [30, 40]
          ],
          [
            [50, 60],
            [70, 80]
          ]
        ]
      }
    ],
    'geoarrow.multilinestring'
  );
  const second = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiLineString',
        coordinates: [
          [
            [90, 100],
            [110, 120]
          ]
        ]
      }
    ],
    'geoarrow.multilinestring'
  );
  const result = readGeoArrowMultiLineString({
    ...first,
    chunks: [...first.chunks, ...second.chunks]
  });
  expect(Array.from(result.featureOffsets)).toEqual([0, 2, 3]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 2, 4, 6]);
  expect(Array.from(result.positions)).toEqual([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
});
it('ING-16 rebases every nested polygon offset and preserves source XY order', () => {
  const first = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [10, 20],
              [30, 20],
              [10, 35],
              [10, 20]
            ]
          ]
        ]
      }
    ],
    'geoarrow.multipolygon'
  );
  const second = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [50, 60],
              [70, 60],
              [50, 75],
              [50, 60]
            ]
          ],
          [
            [
              [80, 90],
              [100, 90],
              [80, 105],
              [80, 90]
            ]
          ]
        ]
      }
    ],
    'geoarrow.multipolygon'
  );
  const result = readGeoArrowMultiPolygon({...first, chunks: [...first.chunks, ...second.chunks]});
  expect(Array.from(result.featureOffsets)).toEqual([0, 1, 3]);
  expect(Array.from(result.polygonOffsets)).toEqual([0, 1, 2, 3]);
  expect(Array.from(result.ringOffsets)).toEqual([0, 4, 8, 12]);
  expect(Array.from(result.positions)).toEqual([
    10, 20, 30, 20, 10, 35, 10, 20, 50, 60, 70, 60, 50, 75, 50, 60, 80, 90, 100, 90, 80, 105, 80, 90
  ]);
});

it('ING-16 retains empty source rows between populated chunks at every hierarchy', () => {
  const point = {type: 'MultiPoint' as const, coordinates: [[10, 20]]};
  const line = {
    type: 'MultiLineString' as const,
    coordinates: [
      [
        [10, 20],
        [30, 40]
      ]
    ]
  };
  const polygon = {
    type: 'MultiPolygon' as const,
    coordinates: [
      [
        [
          [10, 20],
          [30, 20],
          [10, 35],
          [10, 20]
        ]
      ]
    ]
  };
  for (const [encoding, geometry, reader] of [
    ['geoarrow.multipoint', point, readGeoArrowMultiPoint],
    ['geoarrow.multilinestring', line, readGeoArrowMultiLineString],
    ['geoarrow.multipolygon', polygon, readGeoArrowMultiPolygon]
  ] as const) {
    const first = makeGeoArrowColumnFromCatalogGeometries(
      [{type: geometry.type, coordinates: []}, geometry],
      encoding
    );
    const second = makeGeoArrowColumnFromCatalogGeometries(
      [{type: geometry.type, coordinates: []}, geometry],
      encoding
    );
    const packed = reader({...first, chunks: [...first.chunks, ...second.chunks]});
    expect(Array.from(packed.featureOffsets)).toEqual([0, 0, 1, 1, 2]);
    const expected =
      encoding === 'geoarrow.multipoint'
        ? [10, 20]
        : encoding === 'geoarrow.multilinestring'
          ? [10, 20, 30, 40]
          : [10, 20, 30, 20, 10, 35, 10, 20];
    expect(Array.from(packed.positions)).toEqual([...expected, ...expected]);
  }
});

it('ING-16 retains the terminal zero for empty columns', () => {
  for (const [encoding, reader] of [
    ['geoarrow.multipoint', readGeoArrowMultiPoint],
    ['geoarrow.multilinestring', readGeoArrowMultiLineString],
    ['geoarrow.multipolygon', readGeoArrowMultiPolygon]
  ] as const) {
    const packed = reader(makeGeoArrowColumnFromCatalogGeometries([], encoding));
    expect(Array.from(packed.featureOffsets)).toEqual([0]);
    expect(Array.from(packed.positions)).toEqual([]);
  }
});

it('B4 zero physical chunks have empty XY and terminal zero at every offset level', () => {
  const empty = makeGeoArrowColumnFromCatalogGeometries([], 'geoarrow.multipolygon');
  for (const [encoding, reader, offsetNames] of [
    ['geoarrow.multipoint', readGeoArrowMultiPoint, ['featureOffsets']],
    ['geoarrow.multilinestring', readGeoArrowMultiLineString, ['featureOffsets', 'lineOffsets']],
    [
      'geoarrow.multipolygon',
      readGeoArrowMultiPolygon,
      ['featureOffsets', 'polygonOffsets', 'ringOffsets']
    ]
  ] as const) {
    const packed = reader({...empty, encoding, chunks: []});
    expect(Array.from(packed.positions)).toEqual([]);
    for (const name of offsetNames) {
      expect(name in packed).toBe(true);
      expect(Array.from(Reflect.get(packed, name) as Uint32Array)).toEqual([0]);
    }
  }
});

it('B5 respects top-level and metadata spherical edges and rejects conflicting declarations', () => {
  const column = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiLineString',
        coordinates: [
          [
            [0, 0],
            [1, 0]
          ]
        ]
      }
    ],
    'geoarrow.multilinestring'
  );
  expect(() => readGeoArrowMultiLineString({...column, edges: 'spherical'})).toThrow(/spherical/);
  expect(() => readGeoArrowMultiLineString({...column, metadata: {edges: 'spherical'}})).toThrow(
    /spherical/
  );
  expect(() =>
    readGeoArrowMultiLineString({...column, edges: 'planar', metadata: {edges: 'spherical'}})
  ).toThrow(/edges.*conflict/);
  expect(Array.from(readGeoArrowMultiLineString({...column, edges: 'planar'}).positions)).toEqual([
    0, 0, 1, 0
  ]);
});

it('B6 each geometry reader refuses offsets belonging to another family', () => {
  const point = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiPoint',
        coordinates: [
          [0, 0],
          [1, 0]
        ]
      }
    ],
    'geoarrow.multipoint'
  );
  const line = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiLineString',
        coordinates: [
          [
            [0, 0],
            [1, 0]
          ],
          [
            [2, 0],
            [3, 0]
          ],
          [
            [4, 0],
            [5, 0],
            [6, 0]
          ]
        ]
      }
    ],
    'geoarrow.multilinestring'
  );
  const polygon = makeGeoArrowColumnFromCatalogGeometries(
    [
      {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [1, 0],
              [0, 1],
              [0, 0]
            ]
          ]
        ]
      }
    ],
    'geoarrow.multipolygon'
  );
  expect(() => readGeoArrowMultiPoint(line)).toThrow(/multipoint/i);
  expect(() => readGeoArrowMultiLineString(point)).toThrow(/multilinestring/i);
  expect(() => readGeoArrowMultiPolygon(line)).toThrow(/multipolygon/i);
  expect(() => readGeoArrowMultiPoint(polygon)).toThrow(/multipoint/i);
});

import {GeoArrowBuilder} from '@math.gl/arrow/geoarrow-utils';
it('S3 records float64 rounding, non-XY and installed int64-offset refusals', () => {
  const rounded = GeoArrowBuilder.build([{type: 'MultiPoint', coordinates: [[0.1, 0]]}], {
    encoding: 'geoarrow.multipoint',
    dimension: 'xy',
    coordinateType: 'float64',
    coordinateLayout: 'interleaved',
    offsetType: 'int32'
  });
  expect(() => readGeoArrowMultiPoint(rounded)).toThrow(/lose float64 precision/);
  const xyz = GeoArrowBuilder.build([{type: 'MultiPoint', coordinates: [[1, 2, 3]]}], {
    encoding: 'geoarrow.multipoint',
    dimension: 'xyz',
    coordinateType: 'float32',
    coordinateLayout: 'interleaved',
    offsetType: 'int32'
  });
  expect(() => readGeoArrowMultiPoint(xyz)).toThrow(/dimension must be xy/);
  const int64 = GeoArrowBuilder.build([{type: 'MultiPoint', coordinates: [[1, 2]]}], {
    encoding: 'geoarrow.multipoint',
    dimension: 'xy',
    coordinateType: 'float32',
    coordinateLayout: 'interleaved',
    offsetType: 'int64'
  });
  expect(() => readGeoArrowMultiPoint(int64)).toThrow(
    /LargeList requires a newer Apache Arrow runtime/
  );
});
