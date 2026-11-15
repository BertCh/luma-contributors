// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  prepareKeplerGeometrySource,
  prepareKeplerGeometrySourceFromArrow
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {geometryBatches} from './geometry-fixture';
import centerOracle from './json-center-oracle.json';
import arrowCenterOracle from './arrow-center-oracle.json';
import {convertGeoArrowVectorToDenseUnion, type GeoArrowNativeEncoding} from '@luma.gl/graph/arrow';

it('prepares mixed GeoJSON primitive spaces while retaining source batches, rows, holes and XYZ', () => {
  const source = prepareKeplerGeometrySource(geometryBatches);
  expect(source.metadata).toMatchObject({
    rowCount: 5,
    bounds: [0, 0, 21, 21],
    hasZ: true,
    hasPointAltitude: false,
    invalidRows: [],
    featureTypes: {point: true, line: true, polygon: true}
  });
  expect(source.batches.map(batch => [batch.id, batch.sourceRowOffset, batch.rowCount])).toEqual([
    ['first', 0, 2],
    ['empty', 2, 0],
    ['last', 2, 3]
  ]);
  expect(source.batches.map(batch => [...batch.points.sourceRows])).toEqual([[0, 0], [], [4]]);
  expect([...source.batches[0].paths.sourceRows]).toEqual([1, 1]);
  expect([...source.batches[0].paths.positions]).toEqual([0, 0, 7, 1, 1, 8, 2, 2, 0, 3, 3, 0]);
  expect([...source.batches[2].polygonOutlines.sourceRows]).toEqual([2, 2, 4]);
  expect(source.batches[2].polygons.get(0)?.get(0)?.length).toBe(2);
  expect(source.batches[2].polygons.length).toBe(3);
});

it('rejects an invalid complete row and records source properties and nonzero point altitude', () => {
  const source = prepareKeplerGeometrySource(
    [
      {
        id: 'batch',
        rows: [
          {
            type: 'GeometryCollection',
            geometries: [
              {type: 'Point', coordinates: [1, 2]},
              {type: 'Point', coordinates: [NaN, 0]}
            ]
          },
          JSON.stringify({
            type: 'Feature',
            properties: {radius: 4, lineColor: [1, 2, 3]},
            geometry: {type: 'Point', coordinates: [3, 4, 5]}
          })
        ]
      }
    ],
    {crs: 'EPSG:4326'}
  );
  expect(source.metadata.invalidRows).toHaveLength(1);
  expect(source.metadata.invalidRows[0].sourceRowIndex).toBe(0);
  expect(source.metadata.hasPointAltitude).toBe(true);
  expect(source.metadata.sourceStyleProperties).toEqual(['lineColor', 'radius']);
  expect(source.metadata.crs).toBe('EPSG:4326');
  expect([...source.batches[0].points.sourceRows]).toEqual([1]);
  expect(source.batches[0].polygons.length).toBe(2);
});

it('normalizes sliced native GeoArrow coordinates without losing source chunk boundaries', () => {
  const type = new arrow.FixedSizeList(3, new arrow.Field('xyz', new arrow.Float64(), false));
  const first = arrow
    .vectorFromArray(
      [
        [9, 9, 9],
        [1, 2, 0],
        [3, 4, 0]
      ],
      type
    )
    .slice(1);
  const second = arrow.vectorFromArray([[5, 6, 0]], type);
  const vector = new arrow.Vector([...first.data, ...second.data]);
  const source = prepareKeplerGeometrySourceFromArrow(vector, {encoding: 'geoarrow.point'});
  expect(source.batches.map(batch => batch.rowCount)).toEqual([2, 1]);
  expect(source.batches.map(batch => [...batch.points.positions])).toEqual([
    [1, 2, 0, 3, 4, 0],
    [5, 6, 0]
  ]);
  expect(source.batches.map(batch => [...batch.points.sourceRows])).toEqual([[0, 1], [2]]);
});

it('keeps measure and time coordinates out of the XYZ rendering boundary', () => {
  const source = prepareKeplerGeometrySource([
    {
      id: 'trip',
      rows: [
        {
          type: 'LineString',
          coordinates: [
            [1, 2, 3, 100],
            [4, 5, 6, 200]
          ]
        }
      ]
    }
  ]);
  expect(source.metadata.invalidRows).toHaveLength(1);
  expect(source.batches[0].paths.sourceRows.length).toBe(0);
  const vector = arrow.vectorFromArray(
    [[1, 2, 100]],
    new arrow.FixedSizeList(3, new arrow.Field('xym', new arrow.Float64(), false))
  );
  expect(() =>
    prepareKeplerGeometrySourceFromArrow(vector, {encoding: 'geoarrow.point', dimension: 'xym'})
  ).toThrow('XY or XYZ');
});

it('JSON representatives match pinned native normalization and Turf bounds-center fixtures', () => {
  const source = prepareKeplerGeometrySource([
    {id: 'first', rows: centerOracle.cases.slice(0, 5).map(fixture => fixture.source)},
    {id: 'empty', rows: []},
    {id: 'last', rows: centerOracle.cases.slice(5).map(fixture => fixture.source)}
  ]);
  expect(source.representativeRecipe).toBe('json-bounds-center');
  expect(source.batches.map(batch => batch.representative!.validity.length)).toEqual([
    5,
    0,
    centerOracle.cases.length - 5
  ]);
  const positions = source.batches.flatMap(batch => [...batch.representative!.positions]);
  const validity = source.batches.flatMap(batch => [...batch.representative!.validity]);
  for (const [index, fixture] of centerOracle.cases.entries()) {
    expect(validity[index], fixture.name).toBe(fixture.expected ? 1 : 0);
    expect(positions.slice(index * 2, index * 2 + 2), fixture.name).toEqual(
      fixture.expected ?? [0, 0]
    );
  }
  const type = new arrow.FixedSizeList(2, new arrow.Field('value', new arrow.Float64(), false));
  const native = prepareKeplerGeometrySourceFromArrow(arrow.vectorFromArray([[1, 2]], type), {
    encoding: 'geoarrow.point'
  });
  expect(native.representativeRecipe).toBe('geoarrow-mean-center');
  expect([...native.batches[0].representative!.positions]).toEqual([1, 2]);
});

it('native Arrow logical means match pinned loaders fixtures across all homogeneous geometry types', () => {
  for (const fixture of arrowCenterOracle.cases) {
    let type: arrow.DataType = new arrow.FixedSizeList(
      fixture.dimension,
      new arrow.Field(fixture.dimension === 3 ? 'xyz' : 'value', new arrow.Float64(), false)
    );
    for (let depth = 0; depth < fixture.depth; depth++)
      type = new arrow.List(new arrow.Field('child', type, false));
    const vector = arrow.vectorFromArray(fixture.rows, type);
    const source = prepareKeplerGeometrySourceFromArrow(vector, {
      encoding: fixture.encoding as GeoArrowNativeEncoding
    });
    expect(source.representativeRecipe, fixture.name).toBe('geoarrow-mean-center');
    expect(source.representativeUnsupportedReason).toBeUndefined();
    expect([...source.batches[0].representative!.validity], fixture.name).toEqual(
      fixture.rows.map(() => 1)
    );
    expect([...source.batches[0].representative!.positions], fixture.name).toEqual(
      fixture.expected.flat().map(Math.fround)
    );
  }
});

it('Arrow representatives preserve logical slices, null slots and empty chunks while ignoring backing capacity', () => {
  const type = new arrow.FixedSizeList(2, new arrow.Field('value', new arrow.Float64(), false));
  const coordinates = arrow.makeData({
    type: new arrow.Float64(),
    length: 8,
    data: new Float64Array([99, 99, 1, 2, 3, 4, 7, 8, 1000, 2000, 3000, 4000])
  });
  const data = arrow.makeData({
    type,
    length: 4,
    child: coordinates,
    nullCount: 1,
    nullBitmap: new Uint8Array([0b1011])
  });
  const sliced = new arrow.Vector([data]).slice(1);
  const empty = arrow.vectorFromArray([], type);
  const last = arrow.vectorFromArray([[9, 10]], type);
  const source = prepareKeplerGeometrySourceFromArrow(
    new arrow.Vector([...sliced.data, ...empty.data, ...last.data]),
    {encoding: 'geoarrow.point'}
  );
  expect(source.batches.map(batch => batch.rowCount)).toEqual([3, 0, 1]);
  expect(source.batches.map(batch => [...batch.representative!.validity])).toEqual([
    [1, 0, 1],
    [],
    [1]
  ]);
  expect(source.batches.map(batch => [...batch.representative!.positions])).toEqual([
    [1, 2, 0, 0, 7, 8],
    [],
    [9, 10]
  ]);
  expect(source.batches.map(batch => [...batch.points.sourceRows])).toEqual([[0, 2], [], [3]]);

  const lineType = new arrow.List(new arrow.Field('vertices', type, false));
  const lines = arrow
    .vectorFromArray(
      [
        [[99, 99]],
        [
          [0, 0],
          [8, 4]
        ],
        null,
        [[4, 8]]
      ],
      lineType
    )
    .slice(1);
  const lineSource = prepareKeplerGeometrySourceFromArrow(lines, {encoding: 'geoarrow.linestring'});
  expect([...lineSource.batches[0].representative!.validity]).toEqual([1, 0, 1]);
  expect([...lineSource.batches[0].representative!.positions]).toEqual([4, 2, 0, 0, 4, 8]);
});

it('dense union rendering remains available without claiming a nonexistent native Arrow mean recipe', () => {
  const source = arrow.vectorFromArray(['POINT (1 2)', 'LINESTRING (0 0, 8 4)'], new arrow.Utf8());
  const union = convertGeoArrowVectorToDenseUnion(source, {encoding: 'geoarrow.wkt'});
  const prepared = prepareKeplerGeometrySourceFromArrow(union, {encoding: 'geoarrow.geometry'});
  expect(prepared.metadata.featureTypes).toEqual({point: true, line: true, polygon: false});
  expect(prepared.representativeRecipe).toBeUndefined();
  expect(prepared.batches[0].representative).toBeUndefined();
  expect(prepared.representativeUnsupportedReason).toContain('dense union');
});

it('embedded GeoJSON channels retain source chunks and native truthy/default semantics without baking layer defaults', () => {
  const feature = (properties: Record<string, unknown>) => ({
    type: 'Feature',
    properties,
    geometry: {type: 'Point', coordinates: [1, 2]}
  });
  const source = prepareKeplerGeometrySource([
    {
      id: 'first',
      rows: [
        feature({fillColor: [10, 20, 30], lineColor: [40, 50, 60, 0], lineWidth: 2, radius: 3}),
        feature({fillColor: null, lineColor: false, lineWidth: 0, radius: -0})
      ]
    },
    {id: 'empty', rows: []},
    {
      id: 'last',
      rows: [
        feature({fillColor: [], lineColor: [NaN, 10, 20], lineWidth: -2, radius: NaN}),
        null,
        feature({fillColor: [300, -1, 1.9, 0.5], lineWidth: false, radius: ''})
      ]
    }
  ]);
  expect(source.metadata.fixedRadius).toBe(true);
  expect(source.metadata.invalidStyleProperties).toEqual([]);
  expect(source.metadata.sourceStyleProperties).toEqual([
    'fillColor',
    'lineColor',
    'lineWidth',
    'radius'
  ]);
  expect(source.batches.map(batch => batch.properties!.radius!.values.length)).toEqual([2, 0, 3]);
  expect(source.batches.map(batch => [...batch.properties!.radius!.validity])).toEqual([
    [1, 0],
    [],
    [0, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.properties!.radius!.values])).toEqual([
    [3, 0],
    [],
    [0, 0, 0]
  ]);
  expect([...source.batches[2].properties!.lineWidth!.values]).toEqual([-2, 0, 0]);
  expect([...source.batches[2].properties!.lineWidth!.validity]).toEqual([1, 0, 0]);
  expect([...source.batches[0].properties!.fillColor!.values]).toEqual([
    10, 20, 30, 255, 0, 0, 0, 0
  ]);
  expect([...source.batches[0].properties!.lineColor!.values]).toEqual([40, 50, 60, 0, 0, 0, 0, 0]);
  expect([...source.batches[2].properties!.fillColor!.values]).toEqual([
    0, 0, 0, 255, 0, 0, 0, 0, 44, 255, 1, 0
  ]);
  expect([...source.batches[2].properties!.fillColor!.validity]).toEqual([1, 0, 1]);
  expect([...source.batches[2].properties!.lineColor!.values]).toEqual([
    0, 10, 20, 255, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
});

it('embedded properties belong to the normalized root Feature and truthy malformed values are diagnosed separately', () => {
  const point = {type: 'Point', coordinates: [1, 2]};
  const first = {type: 'Feature', properties: {radius: 0}, geometry: point};
  const source = prepareKeplerGeometrySource([
    {
      id: 'first',
      rows: [
        {
          type: 'FeatureCollection',
          features: [first, {type: 'Feature', properties: {radius: 9}, geometry: point}]
        },
        {...point, properties: {radius: 9}},
        {
          type: 'Feature',
          properties: {radius: false},
          geometry: {type: 'GeometryCollection', geometries: [{...point, properties: {radius: 9}}]}
        },
        {type: 'Feature', properties: {radius: 9}, geometry: null}
      ]
    }
  ]);
  expect(source.metadata.fixedRadius).toBe(false);
  expect(source.metadata.sourceStyleProperties).toEqual(['radius']);
  expect([...source.batches[0].properties!.radius!.validity]).toEqual([0, 0, 0, 0]);

  const invalid = prepareKeplerGeometrySource([
    {
      id: 'bad',
      rows: [
        {
          type: 'Feature',
          properties: {fillColor: 'red', lineColor: 7, lineWidth: Infinity, radius: '2'},
          geometry: point
        }
      ]
    }
  ]);
  expect(invalid.metadata.invalidRows).toEqual([]);
  expect(
    invalid.metadata.invalidStyleProperties.map(diagnostic => [
      diagnostic.sourceRowIndex,
      diagnostic.property
    ])
  ).toEqual([
    [0, 'fillColor'],
    [0, 'lineColor'],
    [0, 'lineWidth'],
    [0, 'radius']
  ]);
  expect(invalid.metadata.fixedRadius).toBe(true);
  expect([...invalid.batches[0].properties!.radius!.validity]).toEqual([0]);
});

it('retains ring coordinates and source identities while deriving outward side normals for holes', () => {
  const shell = [
    [0, 0, 10],
    [8, 0, 20],
    [8, 8, 30],
    [0, 8, 40],
    [0, 0, 10]
  ];
  const hole = [
    [2, 2],
    [4, 2],
    [4, 4],
    [2, 4],
    [2, 2]
  ];
  const rows = [
    null,
    {type: 'Polygon', coordinates: [shell, hole]},
    {type: 'Polygon', coordinates: [[...shell].reverse(), [...hole].reverse()]}
  ];
  const source = prepareKeplerGeometrySource([
    {id: 'empty', rows: []},
    {id: 'rings', rows}
  ]);
  expect(source.batches[0].polygonOutlines.sideOrientations).toEqual(new Float32Array(0));
  const outlines = source.batches[1].polygonOutlines;
  expect([...outlines.sourceRows]).toEqual([1, 1, 2, 2]);
  expect([...outlines.sideOrientations!]).toEqual([1, -1, -1, 1]);
  expect([...outlines.positions.slice(0, shell.length * 3)]).toEqual(shell.flat());
  // Right normals multiplied by sideOrientations face out of shells and into holes.
  expect([0 * outlines.sideOrientations![0], -8 * outlines.sideOrientations![0]]).toEqual([0, -8]);
  expect(-2 * outlines.sideOrientations![1]).toBe(2);
});

it('retains vertical source channels with distinct native height and offset validity across chunks', () => {
  const feature = (properties: Record<string, unknown>) => ({
    type: 'Feature',
    properties,
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0, 10],
          [4, 0, 20],
          [0, 4, 30],
          [0, 0, 10]
        ]
      ]
    }
  });
  const source = prepareKeplerGeometrySource([
    {id: 'first', rows: [feature({elevation: 0, elevationOffset: 0, radius: 2}), null]},
    {id: 'empty', rows: []},
    {
      id: 'last',
      rows: [
        feature({elevation: -12, elevationOffset: -5}),
        feature({elevation: 21, elevationOffset: 8}),
        feature({elevation: NaN, elevationOffset: NaN}),
        feature({elevation: false, elevationOffset: false}),
        feature({})
      ]
    }
  ]);
  expect(source.metadata.sourceStyleProperties).toEqual(['radius', 'elevation', 'elevationOffset']);
  expect(source.metadata.invalidStyleProperties).toEqual([]);
  expect(source.batches.map(batch => batch.properties!.elevation!.values.length)).toEqual([
    2, 0, 5
  ]);
  expect(source.batches.map(batch => [...batch.properties!.elevation!.values])).toEqual([
    [0, 0],
    [],
    [-12, 21, 0, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.properties!.elevation!.validity])).toEqual([
    [0, 0],
    [],
    [1, 1, 0, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.properties!.elevationOffset!.values])).toEqual([
    [0, 0],
    [],
    [-5, 8, 0, 0, 0]
  ]);
  expect(source.batches.map(batch => [...batch.properties!.elevationOffset!.validity])).toEqual([
    [1, 0],
    [],
    [1, 1, 0, 0, 0]
  ]);
  expect([...source.batches[0].properties!.radius!.values]).toEqual([2, 0]);
  expect([...source.batches[2].polygonOutlines.sourceRows]).toEqual([2, 3, 4, 5, 6]);
  expect([...source.batches[0].polygonOutlines.positions]).toEqual([
    0, 0, 10, 4, 0, 20, 0, 4, 30, 0, 0, 10
  ]);
});

it('diagnoses unsupported truthy height and finite offset overflow without dropping geometry rows', () => {
  const values = ['20', Infinity, Number.MAX_VALUE, NaN, false, null, -0];
  const source = prepareKeplerGeometrySource([
    {
      id: 'vertical',
      rows: values.map(value => ({
        type: 'Feature',
        properties: {elevation: value, elevationOffset: value},
        geometry: {type: 'Point', coordinates: [1, 2, 3]}
      }))
    }
  ]);
  expect(source.metadata.invalidRows).toEqual([]);
  expect(
    source.metadata.invalidStyleProperties.map(diagnostic => [
      diagnostic.sourceRowIndex,
      diagnostic.property
    ])
  ).toEqual([
    [0, 'elevation'],
    [1, 'elevation'],
    [2, 'elevation'],
    [2, 'elevationOffset']
  ]);
  expect([...source.batches[0].properties!.elevation!.validity]).toEqual([0, 0, 0, 0, 0, 0, 0]);
  expect([...source.batches[0].properties!.elevationOffset!.validity]).toEqual([
    0, 0, 0, 0, 0, 0, 1
  ]);
  expect(Object.is(source.batches[0].properties!.elevationOffset!.values[6], -0)).toBe(true);
  expect([...source.batches[0].points.sourceRows]).toEqual([0, 1, 2, 3, 4, 5, 6]);
});

it('normalizes vertical properties from the root Feature without inventing native Arrow properties', () => {
  const point = {type: 'Point', coordinates: [1, 2]};
  const source = prepareKeplerGeometrySource([
    {
      id: 'features',
      rows: [
        {
          type: 'FeatureCollection',
          features: [
            {type: 'Feature', properties: {elevation: 0, elevationOffset: 0}, geometry: point},
            {type: 'Feature', properties: {elevation: 90, elevationOffset: 90}, geometry: point}
          ]
        },
        {...point, properties: {elevation: 80, elevationOffset: 80}},
        {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'GeometryCollection',
            geometries: [{...point, properties: {elevation: 70, elevationOffset: 70}}]
          }
        }
      ]
    }
  ]);
  expect([...source.batches[0].properties!.elevation!.validity]).toEqual([0, 0, 0]);
  expect([...source.batches[0].properties!.elevationOffset!.validity]).toEqual([1, 0, 0]);
  const coordinateType = new arrow.FixedSizeList(
    3,
    new arrow.Field('xyz', new arrow.Float64(), false)
  );
  const native = prepareKeplerGeometrySourceFromArrow(
    arrow.vectorFromArray([[1, 2, 3]], coordinateType),
    {encoding: 'geoarrow.point'}
  );
  expect(native.metadata.sourceStyleProperties).toEqual([]);
  expect(native.batches[0].properties).toBeUndefined();
});
