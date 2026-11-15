// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  countWorkflowVertices,
  getWorkflowColumnNames,
  getWorkflowDataKind,
  getWorkflowFeatures,
  hashWorkflowData,
  hashWorkflowText,
  hashWorkflowValue,
  makeWorkflowCSVText,
  makeWorkflowData,
  makeWorkflowDataFromFeatures,
  makeWorkflowGeoJSONText,
  summarizeWorkflowColumn
} from '../../src/workflows/workflow-data';
import {WorkflowError, type WorkflowData} from '../../src/workflows/workflow-types';
import {makeFixturePoints} from './workflow-fixtures';

function makeGeometryTable(values: Array<string | null>): CatalogOutputTable {
  return {
    id: 'geometry',
    label: 'Test geometries',
    rowCount: values.length,
    columns: {geometry: values},
    columnTypes: {geometry: 'string'},
    geometryColumn: 'geometry'
  };
}

function expectWorkflowError(action: () => unknown, code: string, message: string): void {
  expect(action).toThrow(WorkflowError);
  expect(action).toThrow(expect.objectContaining({code, message}));
}

describe('workflow data geometry', () => {
  test('retains unknown geometry kind and empty exports for an empty geometry table', () => {
    const table: CatalogOutputTable = {
      id: 'filtered-points',
      label: 'Filtered points',
      rowCount: 0,
      columns: {point_id: [], geometry: []},
      columnTypes: {point_id: 'string', geometry: 'string'},
      geometryColumn: 'geometry'
    };
    expect(getWorkflowDataKind(table)).toBe('features');
    const data = makeWorkflowData(table, {identityColumn: 'point_id'});
    expect(data).toEqual({table, kind: 'features', identityColumn: 'point_id'});
    expect(getWorkflowFeatures(data)).toEqual({type: 'FeatureCollection', features: []});
    expect(getWorkflowFeatures(data)).toBe(getWorkflowFeatures(data));
    expect(countWorkflowVertices(data)).toBe(0);
    expect(makeWorkflowGeoJSONText(data)).toBe('{"type":"FeatureCollection","features":[]}');
    expect(makeWorkflowCSVText(data)).toBe('point_id,geometry');
    const hash = hashWorkflowData(data);
    expect(hash).toMatch(/^[0-9a-f]{32}$/);
    expect(hash).toBe(hashWorkflowData(makeWorkflowData(table, {identityColumn: 'point_id'})));
    expect(hash).not.toBe(hashWorkflowData({...data, kind: 'table'}));
    expect(hash).not.toBe(
      hashWorkflowData({...data, table: {...table, geometryColumn: undefined}})
    );
    expect(getWorkflowDataKind({...table, geometryColumn: undefined})).toBe('table');
    expect(getWorkflowDataKind(makeGeometryTable([null]))).toBe('table');
    expect(makeWorkflowData(makeGeometryTable([null])).kind).toBe('table');
  });

  test('wraps an empty geometry feature collection as features', () => {
    const data = makeWorkflowDataFromFeatures(
      {type: 'FeatureCollection', features: []},
      {id: 'empty-features', label: 'Empty features'}
    );
    expect(data.kind).toBe('features');
    expect(getWorkflowFeatures(data)).toEqual({type: 'FeatureCollection', features: []});
    expect(makeWorkflowGeoJSONText(data)).toBe('{"type":"FeatureCollection","features":[]}');
    expect(makeWorkflowCSVText(data)).toBe('geometry');
  });

  test.each([
    ['Point', 'point'],
    ['MultiPoint', 'point'],
    ['LineString', 'line'],
    ['MultiLineString', 'line'],
    ['Polygon', 'polygon'],
    ['MultiPolygon', 'polygon'],
    ['GeometryCollection', 'features'],
    ['Unknown', 'features']
  ])('detects %s as %s', (type, kind) => {
    expect(getWorkflowDataKind(makeGeometryTable([JSON.stringify({type, coordinates: []})]))).toBe(
      kind
    );
  });

  test('handles null, missing, mixed, reordered, escaped and nested type keys', () => {
    expect(getWorkflowDataKind(makeGeometryTable([null, null]))).toBe('table');
    expect(getWorkflowDataKind({...makeGeometryTable([]), geometryColumn: undefined})).toBe(
      'table'
    );
    expect(
      getWorkflowDataKind(
        makeGeometryTable([
          '{"type":"Point","coordinates":[0,0]}',
          '{"type":"LineString","coordinates":[]}'
        ])
      )
    ).toBe('features');
    for (const text of [
      ' { "type" : "Point", "coordinates" : [ 1e-3, -2 ] } ',
      '{"coordinates":[0,0],"type":"Point"}',
      '{"extra":{"type":"Polygon"},"coordinates":[0,0],"type":"Point"}',
      '{"type":"Point","extra":{"type":"Polygon"},"coordinates":[0,0]}',
      '{"\\u0074ype":"Point","coordinates":[0,0]}'
    ])
      expect(getWorkflowDataKind(makeGeometryTable([null, text]))).toBe('point');
    expect(
      getWorkflowDataKind(makeGeometryTable(['{"type":"Point","coordinates":[],"type":"Polygon"}']))
    ).toBe('polygon');
    expect(getWorkflowDataKind(makeGeometryTable(['invalid']))).toBe('features');
  });

  test('round trips scalar properties, numeric and string ids, geometry and nulls', () => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          id: 42,
          properties: {name: 'first', value: 3, active: true},
          geometry: {type: 'Point', coordinates: [1, 2]}
        },
        {id: 'second', properties: {name: 'second', value: null, active: false}, geometry: null}
      ]
    };
    const data = makeWorkflowDataFromFeatures(collection, {id: 'round-trip', label: 'Round trip'});
    expect(data.kind).toBe('point');
    expect(data.identityColumn).toBe('feature_id');
    expect(data.table.columnTypes).toEqual({
      name: 'string',
      value: 'int32',
      active: 'boolean',
      geometry: 'string',
      feature_id: 'string'
    });
    expect(getWorkflowFeatures(data).features).toEqual(
      collection.features.map(feature => ({type: 'Feature', ...feature}))
    );
    expect(getWorkflowFeatures(data)).toBe(getWorkflowFeatures(data));
    expect(JSON.parse(makeWorkflowGeoJSONText(data))).toEqual(getWorkflowFeatures(data));
    expect(countWorkflowVertices(data)).toBe(1);
    expect(getWorkflowColumnNames(data)).toEqual(['name', 'value', 'active', 'feature_id']);
  });

  test('uses row indices without an identity and retains a missing identity column as a property', () => {
    const data = makeWorkflowData(
      makeGeometryTable([null, '{"type":"Point","coordinates":[0,0]}']),
      {identityColumn: 'missing'}
    );
    expect(getWorkflowFeatures(data).features.map(feature => feature.id)).toEqual([0, 1]);
    const original = makeFixturePoints(2);
    expect(
      getWorkflowFeatures({...original, identityColumn: 'missing'}).features[0].properties
    ).toEqual({point_id: 'p0', value: 0});
  });

  test('absent ids create no identity column and mixed present ids retain nulls', () => {
    const features = [{geometry: null}, {geometry: null, id: null}];
    const data = makeWorkflowDataFromFeatures(
      {type: 'FeatureCollection', features},
      {id: 'empty', label: 'Empty'}
    );
    expect(data.identityColumn).toBeUndefined();
    expect(data.table.columns).toEqual({geometry: [null, null]});
    const identified = makeWorkflowDataFromFeatures(
      {type: 'FeatureCollection', features: [...features, {geometry: null, id: 0}]},
      {id: 'ids', label: 'Ids'}
    );
    expect(getWorkflowFeatures(identified).features.map(feature => feature.id)).toEqual([
      null,
      null,
      0
    ]);
  });

  test('counts multipart and collection vertices', () => {
    const data = makeWorkflowData(
      makeGeometryTable([
        JSON.stringify({
          type: 'MultiPoint',
          coordinates: [
            [0, 0],
            [1, 1]
          ]
        }),
        JSON.stringify({
          type: 'LineString',
          coordinates: [
            [0, 0],
            [1, 1]
          ]
        }),
        JSON.stringify({
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [1, 0],
              [0, 0]
            ]
          ]
        }),
        JSON.stringify({
          type: 'GeometryCollection',
          geometries: [{type: 'Point', coordinates: [1, 2]}]
        })
      ])
    );
    expect(countWorkflowVertices(data)).toBe(8);
  });

  test('refuses non-geometry tables and malformed geometry rows with the exact issue', () => {
    const data = makeWorkflowData({...makeGeometryTable([]), geometryColumn: undefined});
    expectWorkflowError(
      () => getWorkflowFeatures(data),
      'geometry-required',
      'Test geometries: Geometry requires a geometry column.'
    );
    expectWorkflowError(
      () => makeWorkflowGeoJSONText(data),
      'geometry-required',
      'Test geometries: Geometry requires a geometry column.'
    );
    for (const value of ['broken', '12', '{}', '[]']) {
      expectWorkflowError(
        () => getWorkflowFeatures(makeWorkflowData(makeGeometryTable([null, value]))),
        'geometry-required',
        'Test geometries: Geometry column geometry at row 1 must contain valid GeoJSON geometry text.'
      );
    }
  });
});

describe('feature property columns', () => {
  test('treats absent properties as null even when their names exist on Object.prototype', () => {
    const data = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {geometry: null, properties: JSON.parse('{"toString":"custom","__proto__":"safe"}')},
          {geometry: null, properties: {}}
        ]
      },
      {id: 'property-names', label: 'Property names'}
    );
    expect(data.table.columns['toString']).toEqual(['custom', null]);
    expect(data.table.columns['__proto__']).toEqual(['safe', null]);
  });

  test('reads empty column names and treats inherited names as absent', () => {
    const table: CatalogOutputTable = {
      id: 'empty-names',
      label: 'Empty names',
      rowCount: 1,
      geometryColumn: '',
      columns: {'': ['{"type":"Point","coordinates":[0,0]}'], identity: ['named']},
      columnTypes: {'': 'string', identity: 'string'}
    };
    const data = makeWorkflowData(table, {identityColumn: 'toString'});
    expect(data.kind).toBe('point');
    expect(getWorkflowFeatures(data).features[0]).toEqual({
      type: 'Feature',
      id: 0,
      properties: {identity: 'named'},
      geometry: {type: 'Point', coordinates: [0, 0]}
    });
    expect(getWorkflowDataKind({...table, geometryColumn: 'toString'})).toBe('table');
    expect(countWorkflowVertices(makeWorkflowData({...table, geometryColumn: undefined}))).toBe(0);
    expectWorkflowError(
      () => summarizeWorkflowColumn(data, 'toString'),
      'unknown-parameter',
      'Empty names: Column toString must name an existing column.'
    );
  });

  test('infers all values, union order, boundaries and structured or mixed values', () => {
    const first = {
      integer: -2147483648,
      decimal: 1,
      overflow: 2147483648,
      flag: true,
      text: 'hello',
      mixed: 1,
      object: {a: 1},
      array: [1, 2],
      empty: null,
      nonFinite: NaN
    };
    const second = {
      integer: 2147483647,
      decimal: 1.5,
      overflow: 0,
      flag: false,
      text: 'there',
      mixed: false,
      object: {b: 2},
      array: [],
      empty: null,
      nonFinite: Infinity,
      late: 4
    };
    const data = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {geometry: null, properties: first},
          {geometry: null, properties: second}
        ]
      },
      {id: 'types', label: 'Types'}
    );
    expect(Object.keys(data.table.columns)).toEqual([...Object.keys(first), 'late', 'geometry']);
    expect(data.table.columnTypes).toEqual({
      integer: 'int32',
      decimal: 'float64',
      overflow: 'float64',
      flag: 'boolean',
      text: 'string',
      mixed: 'string',
      object: 'string',
      array: 'string',
      empty: 'string',
      nonFinite: 'float64',
      late: 'int32',
      geometry: 'string'
    });
    expect(data.table.columns['mixed']).toEqual(['1', 'false']);
    expect(data.table.columns['object']).toEqual(['{"a":1}', '{"b":2}']);
    expect(data.table.columns['array']).toEqual(['[1,2]', '[]']);
    expect(data.table.columns['nonFinite']).toEqual([NaN, Infinity]);
    expect(data.table.columns['late']).toEqual([null, 4]);
  });

  test('renames reserved properties without overwriting other properties', () => {
    const properties = JSON.parse(
      '{"geometry":"property geometry","geometry_property":"existing","feature_id":"property id","__proto__":"safe"}'
    ) as Record<string, unknown>;
    const data = makeWorkflowDataFromFeatures(
      {type: 'FeatureCollection', features: [{id: 'real-id', geometry: null, properties}]},
      {id: 'reserved', label: 'Reserved'}
    );
    expect(data.table.columns).toEqual({
      geometry_property_property: ['property geometry'],
      geometry_property: ['existing'],
      feature_id_property: ['property id'],
      ['__proto__']: ['safe'],
      geometry: [null],
      feature_id: ['real-id']
    });
  });
});

describe('column summaries', () => {
  const data = makeWorkflowData({
    id: 'summary',
    label: 'Summary',
    rowCount: 7,
    columns: {
      numeric: [1e16, 1, -1e16, NaN, Infinity, -Infinity, null],
      empty: [null, null, null, null, null, null, null],
      category: ['a', 'b', 'a', null, 'c', 'c', 'a'],
      flag: [true, false, true, null, true, false, true]
    },
    columnTypes: {numeric: 'float64', empty: 'float64', category: 'string', flag: 'boolean'}
  });

  test('uses finite values, compensated mean and all-null omissions', () => {
    expect(summarizeWorkflowColumn(data, 'numeric')).toEqual({
      name: 'numeric',
      type: 'float64',
      rowCount: 7,
      nullCount: 1,
      nonFiniteCount: 3,
      minimum: -1e16,
      maximum: 1e16,
      mean: 1 / 3
    });
    expect(summarizeWorkflowColumn(data, 'empty')).toEqual({
      name: 'empty',
      type: 'float64',
      rowCount: 7,
      nullCount: 7,
      nonFiniteCount: 0
    });
    const large = makeWorkflowData({
      id: 'large',
      label: 'Large',
      rowCount: 2,
      columns: {value: [Number.MAX_VALUE, Number.MAX_VALUE]},
      columnTypes: {value: 'float64'}
    });
    expect(summarizeWorkflowColumn(large, 'value').mean).toBe(Number.MAX_VALUE);
  });

  test('counts distinct values up to the limit, frequencies and booleans', () => {
    expect(summarizeWorkflowColumn(data, 'category')).toEqual({
      name: 'category',
      type: 'string',
      rowCount: 7,
      nullCount: 1,
      distinctCount: 3,
      distinctLimitReached: false,
      topValues: [
        {value: 'a', count: 3},
        {value: 'c', count: 2},
        {value: 'b', count: 1}
      ]
    });
    expect(summarizeWorkflowColumn(data, 'category', {distinctLimit: 2})).toMatchObject({
      distinctCount: 2,
      distinctLimitReached: true,
      topValues: [
        {value: 'a', count: 3},
        {value: 'c', count: 2},
        {value: 'b', count: 1}
      ]
    });
    expect(summarizeWorkflowColumn(data, 'flag').topValues).toEqual([
      {value: 'true', count: 4},
      {value: 'false', count: 2}
    ]);
    expect(summarizeWorkflowColumn(data, 'category', {distinctLimit: 0})).toMatchObject({
      distinctCount: 0,
      distinctLimitReached: true,
      topValues: [
        {value: 'a', count: 3},
        {value: 'c', count: 2},
        {value: 'b', count: 1}
      ]
    });
    expectWorkflowError(
      () => summarizeWorkflowColumn(data, 'missing'),
      'unknown-parameter',
      'Summary: Column missing must name an existing column.'
    );
  });

  test('defaults to 1000 distinct values and at most eight top values', () => {
    const many = makeWorkflowData({
      id: 'many',
      label: 'Many',
      rowCount: 1001,
      columns: {value: Array.from({length: 1001}, (_, index) => String(index))},
      columnTypes: {value: 'string'}
    });
    const summary = summarizeWorkflowColumn(many, 'value');
    expect(summary.distinctCount).toBe(1000);
    expect(summary.distinctLimitReached).toBe(true);
    expect(summary.topValues).toEqual(
      Array.from({length: 8}, (_, index) => ({value: String(index), count: 1}))
    );
  });
});

describe('workflow hashing', () => {
  test('refuses unsupported function values with a named issue', () => {
    expectWorkflowError(
      () => hashWorkflowValue(() => 1),
      'malformed',
      'Workflow value: function is not a supported JSON value.'
    );
  });

  test('pins UTF-16 and canonical content hashes', () => {
    expect(hashWorkflowText('')).toBe('92ca2f0e1388a0fbb0ef98030de5c6a9');
    expect(hashWorkflowText('workflow 🌍')).toBe('a896035f870fdf95577686f6576394d5');
    expect(hashWorkflowValue({b: [1, true, null], a: 'x'})).toBe(
      '620f955387549ac98eaf7ff5bfeef784'
    );
    expect(hashWorkflowData(makeFixturePoints(2))).toBe('efe3ba57f2d51d5122befb98757e9eab');
  });

  test('canonicalizes objects and typed arrays while separating types and boundaries', () => {
    expect(hashWorkflowValue({b: 2, a: 1, omitted: undefined})).toBe(
      hashWorkflowValue({a: 1, b: 2})
    );
    expect(hashWorkflowValue(new Int32Array([1, 2]))).toBe(hashWorkflowValue([1, 2]));
    expect(hashWorkflowValue([2, 1])).not.toBe(hashWorkflowValue([1, 2]));
    const values = [
      '1',
      1,
      [1, 2],
      '1,2',
      {a: 1},
      [['a', 1]],
      NaN,
      Infinity,
      -Infinity,
      -0,
      0,
      'NaN',
      'Infinity',
      '-Infinity',
      '-0',
      true,
      'true',
      null,
      'null',
      undefined,
      [],
      {}
    ];
    expect(new Set(values.map(hashWorkflowValue)).size).toBe(values.length);
    expect(hashWorkflowValue(['a', 'bc'])).not.toBe(hashWorkflowValue(['ab', 'c']));
  });

  test('covers data metadata, ordered names, declared types and every value, excludes display labels', () => {
    const original = makeFixturePoints(2);
    const baseline = hashWorkflowData(original);
    expect(
      hashWorkflowData({...original, table: {...original.table, id: 'another', label: 'Another'}})
    ).toBe(baseline);
    const changedValues: WorkflowData[] = [
      {...original, kind: 'features' as const},
      {...original, identityColumn: undefined},
      {...original, table: {...original.table, geometryColumn: undefined}},
      {
        ...original,
        table: {
          ...original.table,
          columnTypes: {...original.table.columnTypes, value: 'int32' as const}
        }
      },
      {
        ...original,
        table: {...original.table, columns: {...original.table.columns, value: [0, 99]}}
      },
      {
        ...original,
        table: {
          ...original.table,
          columns: {
            geometry: original.table.columns['geometry'],
            value: original.table.columns['value'],
            point_id: original.table.columns['point_id']
          }
        }
      }
    ];
    for (const changed of changedValues) expect(hashWorkflowData(changed)).not.toBe(baseline);
  });

  test('streams 100000 rows by eight columns within three seconds', () => {
    const columns = Object.fromEntries(
      Array.from({length: 8}, (_, index) => [
        `column${index}`,
        Array.from({length: 100000}, (_, row) => row * (index + 1))
      ])
    );
    const data = makeWorkflowData({
      id: 'performance',
      label: 'Performance',
      rowCount: 100000,
      columns,
      columnTypes: Object.fromEntries(Object.keys(columns).map(name => [name, 'float64' as const]))
    });
    const startedAt = performance.now();
    const hash = hashWorkflowData(data);
    expect(performance.now() - startedAt).toBeLessThan(3000);
    expect(hash).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe('CSV export', () => {
  test('never guards numeric identity values even in a string-declared column', () => {
    const data = makeWorkflowDataFromFeatures(
      {type: 'FeatureCollection', features: [{id: -1, geometry: null}]},
      {id: 'negative-id', label: 'Negative id'}
    );
    expect(makeWorkflowCSVText(data)).toBe('geometry,feature_id\n,-1');
  });
  test('quotes separators and line endings, escapes quotes, and guards only string-typed formulas', () => {
    const data = makeWorkflowData({
      id: 'csv',
      label: 'CSV',
      rowCount: 9,
      columns: {
        'text,header': ['a,b', 'a"b', 'a\nb', '=1', '+2', '-3', '@x', '\tx', '\rx'],
        numeric: [-1, 2, 3, 4, 5, 6, 7, 8, 9],
        boolean: [true, false, null, true, false, null, true, false, null]
      },
      columnTypes: {'text,header': 'string', numeric: 'float64', boolean: 'boolean'}
    });
    expect(makeWorkflowCSVText(data)).toBe(
      '"text,header",numeric,boolean\n"a,b",-1,true\n"a""b",2,false\n"a\nb",3,\n\'=1,4,true\n\'+2,5,false\n\'-3,6,\n\'@x,7,true\n\'\tx,8,false\n"\'\rx",9,'
    );
  });

  test('exports geometry text, nulls and a header for an empty table', () => {
    expect(
      makeWorkflowCSVText(
        makeWorkflowData(makeGeometryTable(['{"type":"Point","coordinates":[0,0]}', null]))
      )
    ).toBe('geometry\n"{""type"":""Point"",""coordinates"":[0,0]}"\n');
    expect(makeWorkflowCSVText(makeWorkflowData(makeGeometryTable([])))).toBe('geometry');
  });
});

describe('independent review regressions', () => {
  test('F1 guards hostile property names before quoting CSV headers', () => {
    const data = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {
            geometry: null,
            properties: {
              '=1+1': 'safe',
              '+SUM(1,2)': 'safe',
              '-1': 'safe',
              '@value': 'safe',
              '\tvalue': 'safe',
              '\rvalue': 'safe'
            }
          }
        ]
      },
      {id: 'headers', label: 'Headers'}
    );
    expect(makeWorkflowCSVText(data)).toBe(
      "'=1+1,\"'+SUM(1,2)\",'-1,'@value,'\tvalue,\"'\rvalue\",geometry\nsafe,safe,safe,safe,safe,safe,"
    );
  });

  test('F1 guards headers even in empty tables with numeric column declarations', () => {
    const data = makeWorkflowData({
      id: 'empty-headers',
      label: 'Empty headers',
      rowCount: 0,
      columns: {'=1+1': []},
      columnTypes: {'=1+1': 'float64'}
    });
    expect(makeWorkflowCSVText(data)).toBe("'=1+1");
  });

  test('F2 preserves tiny residuals when large values both overflow and cancel', () => {
    const values = [1e308, 1e308, 1e-20, -1e308, -1e308];
    const data = makeWorkflowData({
      id: 'mean',
      label: 'Mean',
      rowCount: values.length,
      columns: {value: values},
      columnTypes: {value: 'float64'}
    });
    const mean = summarizeWorkflowColumn(data, 'value').mean;
    expect(mean).toBeDefined();
    expect(Math.abs((mean! - 2e-21) / 2e-21)).toBeLessThan(1e-15);
  });

  test.each([
    {values: [1e308, 1e308, 1e-20, -1e308, -1e308], expected: 1e-20 / 5},
    {
      values: [Number.MAX_VALUE, Number.MAX_VALUE, 3, -Number.MAX_VALUE, -Number.MAX_VALUE],
      expected: 3 / 5
    },
    {values: [1e308, 1e308, -1e-300, -1e308, -1e308], expected: -1e-300 / 5},
    {values: [1e308, 1e308, Number.MIN_VALUE * 5, -1e308, -1e308], expected: Number.MIN_VALUE},
    {values: [Number.MAX_VALUE, Number.MAX_VALUE], expected: Number.MAX_VALUE},
    {values: [Number.MIN_VALUE, 0], expected: 0},
    {values: [-Number.MIN_VALUE, 0], expected: -0},
    {values: [Number.MIN_VALUE, Number.MIN_VALUE * 2], expected: Number.MIN_VALUE * 2}
  ])('F2 rounds overflow, cancellation and subnormal means independently of row order: $values', ({
    values,
    expected
  }) => {
    for (let index = 0; index < values.length; index++) {
      const rotated = [...values.slice(index), ...values.slice(0, index)];
      for (const ordered of [rotated, [...rotated].reverse()]) {
        const data = makeWorkflowData({
          id: 'rounding',
          label: 'Rounding',
          rowCount: ordered.length,
          columns: {value: ordered},
          columnTypes: {value: 'float64'}
        });
        expect(summarizeWorkflowColumn(data, 'value').mean).toBe(expected);
      }
    }
  });

  test('F3 preserves first-seen integer-looking names in column lists, CSV and hashes', () => {
    const data = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {geometry: null, properties: {name: 'first', '10': 10}},
          {geometry: null, properties: {'2': 2}}
        ]
      },
      {id: 'order', label: 'Order'}
    );
    expect(getWorkflowColumnNames(data)).toEqual(['10', 'name', '2']);
    expect(makeWorkflowCSVText(data)).toBe('10,name,2,geometry\n10,first,,\n,,2,');
    const reconstructed = {...data, table: {...data.table, columns: {...data.table.columns}}};
    expect(hashWorkflowData(data)).not.toBe(hashWorkflowData(reconstructed));
    expect(getWorkflowColumnNames({...data, table: {...data.table}})).toEqual(['10', 'name', '2']);
  });

  test('F4 reads valid flat and nested numeric geometries without parsing every row', () => {
    const geometries = [
      {type: 'Point', coordinates: [-0.25, 1e-100]},
      {
        type: 'MultiPoint',
        coordinates: [
          [1, 2],
          [3, 4]
        ]
      },
      {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [-1e20, 0.5]
        ]
      },
      {
        type: 'MultiLineString',
        coordinates: [
          [
            [0, 0],
            [1, 1]
          ]
        ]
      },
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [0, 0]
          ]
        ]
      },
      {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [1, 0],
              [0, 0]
            ]
          ]
        ]
      }
    ].map(geometry => JSON.stringify(geometry));
    const parse = vi.spyOn(JSON, 'parse');
    let kinds: string[] = [];
    let parseCount = 0;
    try {
      kinds = geometries.map(geometry => getWorkflowDataKind(makeGeometryTable([geometry])));
      parseCount = parse.mock.calls.length;
    } finally {
      parse.mockRestore();
    }
    expect(kinds).toEqual(['point', 'point', 'line', 'line', 'polygon', 'polygon']);
    expect(parseCount).toBe(0);
  });

  test.each([
    '[0,,0]',
    '[01,0]',
    '[+1,0]',
    '[1.,0]',
    '[.1,0]',
    '[1e,0]',
    '[0,0,]',
    '[[0,0],]',
    '[[0,0]',
    '[0,0]]',
    '[0\v,0]'
  ])('F4 refuses malformed numeric arrays %s regardless of key order', coordinates => {
    for (const geometry of [
      `{"type":"Point","coordinates":${coordinates}}`,
      `{"coordinates":${coordinates},"type":"Point"}`
    ]) {
      expect(getWorkflowDataKind(makeGeometryTable([geometry]))).toBe('features');
      expectWorkflowError(
        () => getWorkflowFeatures(makeWorkflowData(makeGeometryTable([geometry]))),
        'geometry-required',
        'Test geometries: Geometry column geometry at row 0 must contain valid GeoJSON geometry text.'
      );
    }
  });
});
