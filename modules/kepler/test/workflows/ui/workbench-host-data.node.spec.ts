// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, describe, expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {WorkflowError, type WorkflowDatasetRequest} from '../../../src/workflows/workflow-types';
import {makeFixturePoints} from '../workflow-fixtures';
import {
  chooseWorkflowDisplayChannel,
  makeWorkflowDataFromDrawings,
  makeWorkflowDataFromSqlTable,
  profileWorkflowDrawings,
  type WorkflowDrawingFeature,
  getNativeRevisionKey,
  getWorkflowDataBounds,
  makeWorkflowDataFromArrowTable,
  makeWorkflowPublication,
  profileNativeDataset,
  readNativeDatasetData,
  type NativeWorkflowDataset
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-host-data';

afterAll(() => {
  const version = (globalThis as typeof globalThis & {deck?: {VERSION?: unknown}}).deck?.VERSION;
  expect(
    String(version ?? ''),
    'Workbench specs must not load deck.gl 9 into the shared worker; mock Kepler and deck runtimes.'
  ).not.toMatch(/^9\./);
});

function makeDataset(
  fields: readonly {name: string; type: string}[],
  rows: readonly (readonly unknown[])[]
): NativeWorkflowDataset {
  return {
    id: 'native',
    label: 'Taxi trips',
    fields,
    dataContainer: {numRows: () => rows.length, valueAt: (row, column) => rows[row][column]}
  };
}

function makeRequest(
  geometry: WorkflowDatasetRequest['geometry'] = {mode: 'none'}
): WorkflowDatasetRequest {
  return {datasetId: 'native', geometry, population: 'all', maximumRows: 100};
}

function expectIssue(action: () => unknown, code: WorkflowError['code'], message: string): void {
  let failure: unknown;
  try {
    action();
  } catch (cause) {
    failure = cause;
  }
  expect(failure).toBeInstanceOf(WorkflowError);
  expect((failure as WorkflowError).code).toBe(code);
  expect((failure as Error).message).toBe(message);
}

const POINT = {type: 'Point', coordinates: [-87.5, 41.5]};

describe('workflow native dataset profiles', () => {
  test('detects bare and Feature-wrapped geometry, including after nulls', () => {
    const dataset = makeDataset(
      [
        {name: 'shape', type: 'string'},
        {name: 'wrapped', type: 'geojson'},
        {name: 'text', type: 'string'}
      ],
      [
        [null, null, '{"type":"Bogus"}'],
        [JSON.stringify(POINT), {type: 'Feature', geometry: POINT}, 'bad']
      ]
    );
    expect(profileNativeDataset(dataset, true)).toEqual({
      id: 'native',
      label: 'Taxi trips',
      rowCount: 2,
      resident: true,
      fields: dataset.fields,
      geometryFields: ['shape', 'wrapped']
    });
    const sparse = makeDataset(
      [{name: 'shape', type: 'string'}],
      [...Array.from({length: 20}, () => [null]), [POINT]]
    );
    expect(profileNativeDataset(sparse, false).geometryFields).toEqual(['shape']);
    const beyondSample = makeDataset(
      [{name: 'shape', type: 'string'}],
      [...Array.from({length: 16}, () => ['bad']), [POINT]]
    );
    expect(profileNativeDataset(beyondSample, false).geometryFields).toEqual([]);
  });

  test.each([
    ['lat', 'lng'],
    ['latitude', 'longitude'],
    ['pickup_latitude', 'pickup_longitude'],
    ['lat_pickup', 'lon_pickup'],
    ['pickupLatitude', 'pickupLongitude'],
    ['LAT_HOME', 'LONG_HOME']
  ])('pairs %s and %s', (latitude, longitude) => {
    const dataset = makeDataset(
      [
        {name: latitude, type: 'real'},
        {name: longitude, type: 'real'}
      ],
      [[1, 2]]
    );
    expect(profileNativeDataset(dataset, false).pointColumns).toEqual({longitude, latitude});
  });

  test('requires numeric fields and a shared prefix and suffix; avoids substring matches', () => {
    const dataset = makeDataset(
      ['plate', 'flat_rate', 'pickup_lat', 'dropoff_lon', 'latitude', 'longitude'].map(name => ({
        name,
        type: name === 'longitude' ? 'string' : 'real'
      })),
      [[1, 2, 3, 4, 5, '6']]
    );
    expect(profileNativeDataset(dataset, false).pointColumns).toBeUndefined();
    const pairs = makeDataset(
      ['pickup_lat', 'dropoff_lat', 'dropoff_lon', 'pickup_lon'].map(name => ({
        name,
        type: 'real'
      })),
      [[1, 2, 3, 4]]
    );
    expect(profileNativeDataset(pairs, false).pointColumns).toEqual({
      longitude: 'pickup_lon',
      latitude: 'pickup_lat'
    });
  });
});

describe('workflow native reads', () => {
  test('table mode declares scalar types, preserves timestamps and nulls', () => {
    const dataset = makeDataset(
      [
        {name: 'count', type: 'integer'},
        {name: 'wide', type: 'integer'},
        {name: 'value', type: 'real'},
        {name: 'active', type: 'boolean'},
        {name: 'time', type: 'timestamp'},
        {name: 'object', type: 'object'}
      ],
      [
        [1, 2147483648, 2.5, true, 123456, {a: 1}],
        [null, null, null, null, '2026-11-14', null]
      ]
    );
    const data = readNativeDatasetData(dataset, makeRequest());
    expect(data).toEqual({
      kind: 'table',
      identityColumn: 'row_index',
      invalidGeometryCount: 0,
      table: {
        id: 'native',
        label: 'Taxi trips',
        rowCount: 2,
        columns: {
          count: [1, null],
          wide: [2147483648, null],
          value: [2.5, null],
          active: [true, null],
          time: [123456, '2026-11-14'],
          object: ['{"a":1}', null],
          row_index: [0, 1]
        },
        columnTypes: {
          count: 'int32',
          wide: 'float64',
          value: 'float64',
          active: 'boolean',
          time: 'string',
          object: 'string',
          row_index: 'int32'
        }
      }
    });
  });

  test('unwraps Feature geometry and retains unparsable, invalid and missing cells as null', () => {
    const dataset = makeDataset(
      [
        {name: 'shape', type: 'geojson'},
        {name: 'value', type: 'integer'}
      ],
      [
        [{type: 'Feature', geometry: POINT}, 10],
        ['not JSON', 11],
        [null, 12],
        ['{"type":"Point","coordinates":[null,1]}', 13],
        [{type: 'FeatureCollection', features: []}, 14]
      ]
    );
    const data = readNativeDatasetData(
      dataset,
      makeRequest({mode: 'geojson-field', field: 'shape'})
    );
    expect(data.kind).toBe('point');
    expect(data.table.columns).toEqual({
      value: [10, 11, 12, 13, 14],
      geometry: [JSON.stringify(POINT), null, null, null, null],
      row_index: [0, 1, 2, 3, 4]
    });
    expect(data.table.columnTypes).toEqual({
      value: 'int32',
      geometry: 'string',
      row_index: 'int32'
    });
    expect(data.table.geometryColumn).toBe('geometry');
    expect(data.invalidGeometryCount).toBe(4);
  });

  test('constructs points only from finite numbers and preserves subset ordering and identity', () => {
    const dataset = makeDataset(
      [
        {name: 'lon', type: 'real'},
        {name: 'lat', type: 'real'}
      ],
      [
        [-87.5, 41.5],
        [null, 4],
        [1, Infinity],
        [0, 0],
        ['2', 3]
      ]
    );
    const request = makeRequest({mode: 'point-columns', longitude: 'lon', latitude: 'lat'});
    const data = readNativeDatasetData(dataset, request, [3, 0, 1, 2, 4]);
    expect(data.table.columns).toEqual({
      lon: [0, -87.5, null, 1, null],
      lat: [0, 41.5, 4, Infinity, 3],
      geometry: ['{"type":"Point","coordinates":[0,0]}', JSON.stringify(POINT), null, null, null],
      row_index: [3, 0, 1, 2, 4]
    });
    expect(data.table.columnTypes).toEqual({
      lon: 'float64',
      lat: 'float64',
      geometry: 'string',
      row_index: 'int32'
    });
    expect(data.invalidGeometryCount).toBe(3);
    expect(readNativeDatasetData(dataset, {...request, maximumRows: 1}, [3]).table.rowCount).toBe(
      1
    );
    expect(readNativeDatasetData(dataset, request, []).table.columns['row_index']).toEqual([]);
  });

  test('preserves scalar columns whose names collide with geometry and identity', () => {
    const dataset = makeDataset(
      ['geometry', 'geometry_source', 'row_index', 'lon', 'lat'].map(name => ({
        name,
        type: 'integer'
      })),
      [[5, 6, 99, 1, 2]]
    );
    const data = readNativeDatasetData(
      dataset,
      makeRequest({mode: 'point-columns', longitude: 'lon', latitude: 'lat'})
    );
    expect(data.table.columns).toEqual({
      geometry_source_2: [5],
      geometry_source: [6],
      row_index_source: [99],
      lon: [1],
      lat: [2],
      geometry: ['{"type":"Point","coordinates":[1,2]}'],
      row_index: [0]
    });
  });

  test('refuses over-cap populations before reading cells', () => {
    const dataset = makeDataset([{name: 'value', type: 'integer'}], [[1], [2]]);
    const readCell = vi.spyOn(dataset.dataContainer, 'valueAt');
    expectIssue(
      () => readNativeDatasetData(dataset, {...makeRequest(), maximumRows: 1}),
      'capacity-exceeded',
      'Taxi trips: 2 rows exceed Maximum rows (1); filter the dataset or raise Maximum rows.'
    );
    expect(readCell).not.toHaveBeenCalled();
    expect(readNativeDatasetData(dataset, {...makeRequest(), maximumRows: 2}).table.rowCount).toBe(
      2
    );
  });

  test.each([
    [{mode: 'geojson-field', field: 'missing'}, 'Geometry field'],
    [{mode: 'point-columns', longitude: 'missing', latitude: 'value'}, 'Longitude column'],
    [{mode: 'point-columns', longitude: 'value', latitude: 'missing'}, 'Latitude column']
  ] as const)('refuses missing fields in %j', (geometry, label) => {
    const dataset = makeDataset([{name: 'value', type: 'real'}], [[1]]);
    expectIssue(
      () => readNativeDatasetData(dataset, makeRequest(geometry)),
      'parameter-out-of-range',
      `Taxi trips: ${label} must name an existing field; got missing.`
    );
  });

  test('refuses invalid subset row indices', () => {
    const dataset = makeDataset([{name: 'value', type: 'integer'}], [[1], [2]]);
    for (const row of [-1, 2, 0.5])
      expectIssue(
        () => readNativeDatasetData(dataset, makeRequest(), [row]),
        'parameter-out-of-range',
        `Taxi trips: Row index must be between 0 and 1; got ${row}.`
      );
  });
});

describe('workflow Arrow reads and publications', () => {
  test('reads vectors through the public API with their declared types', () => {
    const table = new arrow.Table({
      count: arrow.vectorFromArray([1, null], new arrow.Int32()),
      value: arrow.vectorFromArray([1.25, null], new arrow.Float64()),
      active: arrow.vectorFromArray([true, null], new arrow.Bool()),
      shape: arrow.vectorFromArray([JSON.stringify(POINT), 'bad'], new arrow.Utf8())
    });
    const data = makeWorkflowDataFromArrowTable(
      table,
      makeRequest({mode: 'geojson-field', field: 'shape'}),
      'Arrow trips'
    );
    expect(data.table.columns).toEqual({
      count: [1, null],
      value: [1.25, null],
      active: [true, null],
      geometry: [JSON.stringify(POINT), null],
      row_index: [0, 1]
    });
    expect(data.table.columnTypes).toEqual({
      count: 'int32',
      value: 'float64',
      active: 'boolean',
      geometry: 'string',
      row_index: 'int32'
    });
    expect(data.invalidGeometryCount).toBe(1);
    expectIssue(
      () =>
        makeWorkflowDataFromArrowTable(table, {...makeRequest(), maximumRows: 1}, 'Arrow trips'),
      'capacity-exceeded',
      'Arrow trips: 2 rows exceed Maximum rows (1); filter the dataset or raise Maximum rows.'
    );
    const points = new arrow.Table({
      longitude: arrow.vectorFromArray([1, null], new arrow.Float64()),
      latitude: arrow.vectorFromArray([2, 3], new arrow.Float64())
    });
    expect(
      makeWorkflowDataFromArrowTable(
        points,
        makeRequest({mode: 'point-columns', longitude: 'longitude', latitude: 'latitude'}),
        'Points'
      ).table.columns['geometry']
    ).toEqual(['{"type":"Point","coordinates":[1,2]}', null]);
  });

  test('restores original row identity from resident extraction metadata', () => {
    const table = new arrow.Table({source_row: arrow.vectorFromArray([7, 12], new arrow.Uint32())});
    const metadata = new Map([
      ['luma:row-extraction', JSON.stringify({sourceRowColumn: 'source_row'})]
    ]);
    const extracted = new arrow.Table(
      new arrow.Schema(table.schema.fields, metadata),
      table.batches
    );
    expect(
      makeWorkflowDataFromArrowTable(extracted, makeRequest(), 'Extracted').table.columns[
        'row_index'
      ]
    ).toEqual([7, 12]);
  });

  test('publishes point coordinates with unique names, including MultiPoint and null rows', () => {
    const data = makeFixturePoints(3);
    data.table.columns['longitude'] = [4, 5, 6];
    data.table.columnTypes['longitude'] = 'int32';
    data.table.columns['geometry'] = [
      JSON.stringify(POINT),
      '{"type":"MultiPoint","coordinates":[[1,2],[3,4]]}',
      null
    ];
    const beforePublication = structuredClone(data);
    const publication = makeWorkflowPublication(data, 'Published');
    expect(publication.label).toBe('Published');
    expect(publication.coordinates).toEqual({longitude: 'longitude_2', latitude: 'latitude'});
    expect(publication.geometryColumn).toBeUndefined();
    expect(publication.table.schema.fields.map(field => [field.name, String(field.type)])).toEqual([
      ['point_id', 'Utf8'],
      ['value', 'Float64'],
      ['longitude', 'Int32'],
      ['longitude_2', 'Float64'],
      ['latitude', 'Float64']
    ]);
    expect(Array.from(publication.table.getChild('longitude_2')!)).toEqual([-87.5, 1, null]);
    expect(Array.from(publication.table.getChild('latitude')!)).toEqual([41.5, 2, null]);
    expect(data).toEqual(beforePublication);
  });

  test.each([
    [
      'line',
      {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [1, 1]
        ]
      }
    ],
    [
      'polygon',
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [0, 0]
          ]
        ]
      }
    ]
  ] as const)('keeps %s geometry as text', (kind, geometry) => {
    const dataset = makeDataset([{name: 'shape', type: 'geojson'}], [[geometry]]);
    const data = readNativeDatasetData(
      dataset,
      makeRequest({mode: 'geojson-field', field: 'shape'})
    );
    expect(data.kind).toBe(kind);
    const publication = makeWorkflowPublication(data, kind);
    expect(publication.geometryColumn).toBe('geometry');
    expect(publication.coordinates).toBeUndefined();
    expect(publication.table.getChild('geometry')!.get(0)).toBe(JSON.stringify(geometry));
  });

  test('publishes a table with no geometry role and preserves scalar Arrow types', () => {
    const dataset = makeDataset(
      [
        {name: 'value', type: 'real'},
        {name: 'active', type: 'boolean'}
      ],
      [
        [1.25, true],
        [null, false]
      ]
    );
    const publication = makeWorkflowPublication(
      readNativeDatasetData(dataset, makeRequest()),
      'Summary'
    );
    expect(publication.geometryColumn).toBeUndefined();
    expect(publication.coordinates).toBeUndefined();
    expect(publication.table.schema.fields.map(field => [field.name, String(field.type)])).toEqual([
      ['value', 'Float64'],
      ['active', 'Bool'],
      ['row_index', 'Int32']
    ]);
    expect(Array.from(publication.table.getChild('value')!)).toEqual([1.25, null]);
  });
});

describe('workflow native revisions', () => {
  test('keys selected population by relevant filter values and exact row identities', () => {
    const dataset = {
      ...makeDataset([{name: 'value', type: 'integer'}], [[1], [2]]),
      filteredIndex: [0]
    };
    const filter = {dataId: ['native'], value: [0, 10]};
    const changed = {...filter, value: [0, 11]};
    const selected = getNativeRevisionKey(dataset, [filter], 'selected');
    expect(getNativeRevisionKey(dataset, [changed], 'selected')).not.toBe(selected);
    expect(getNativeRevisionKey(dataset, [{value: [0, 10], dataId: ['native']}], 'selected')).toBe(
      selected
    );
    expect(
      getNativeRevisionKey(dataset, [filter, {dataId: ['other'], value: [9]}], 'selected')
    ).toBe(selected);
    expect(getNativeRevisionKey({...dataset, filteredIndex: [1]}, [filter], 'selected')).not.toBe(
      selected
    );
    expect(getNativeRevisionKey({...dataset, filteredIndex: []}, [filter], 'selected')).not.toBe(
      selected
    );
    expect(getNativeRevisionKey(dataset, [filter], 'all')).toBe(
      getNativeRevisionKey(dataset, [changed], 'all')
    );
  });

  test('keys both populations by dataset rows, schema and native revision', () => {
    const dataset = makeDataset([{name: 'value', type: 'integer'}], [[1]]);
    for (const population of ['all', 'selected'] as const) {
      const key = getNativeRevisionKey(dataset, [], population);
      expect(getNativeRevisionKey({...dataset, id: 'other'}, [], population)).not.toBe(key);
      expect(
        getNativeRevisionKey(
          {...dataset, fields: [{name: 'other', type: 'integer'}]},
          [],
          population
        )
      ).not.toBe(key);
      expect(getNativeRevisionKey({...dataset, dataRevision: 2}, [], population)).not.toBe(key);
      expect(
        getNativeRevisionKey(makeDataset(dataset.fields, [[1], [2]]), [], population)
      ).not.toBe(key);
    }
  });
});

function makeBoundsData(geometries: readonly unknown[]) {
  const data = makeFixturePoints(geometries.length);
  data.table.columns['geometry'] = geometries.map(geometry =>
    geometry === null ? null : (JSON.stringify(geometry) ?? null)
  );
  return data;
}

describe('workflow map bounds', () => {
  test('points contribute finite XY positions, including altitude-independent positions', () => {
    expect(
      getWorkflowDataBounds(
        makeBoundsData([
          {type: 'Point', coordinates: [-3, 4, null]},
          {type: 'Point', coordinates: [7, -2]}
        ])
      )
    ).toEqual([-3, -2, 7, 4]);
    expect(getWorkflowDataBounds(makeBoundsData([{type: 'Point', coordinates: [7, -2]}]))).toEqual([
      7, -2, 7, -2
    ]);
  });

  test('lines and polygon rings contribute every finite position', () => {
    expect(
      getWorkflowDataBounds(
        makeBoundsData([
          {
            type: 'LineString',
            coordinates: [
              [-8, 1],
              [4, 6],
              [9, -3]
            ]
          }
        ])
      )
    ).toEqual([-8, -3, 9, 6]);
    expect(
      getWorkflowDataBounds(
        makeBoundsData([
          {
            type: 'Polygon',
            coordinates: [
              [
                [-5, -4],
                [6, -4],
                [6, 7],
                [-5, -4]
              ],
              [
                [-1, 0],
                [1, 0],
                [0, 2],
                [-1, 0]
              ]
            ]
          }
        ])
      )
    ).toEqual([-5, -4, 6, 7]);
  });

  test.each([
    {
      type: 'MultiPoint',
      coordinates: [
        [-6, 2],
        [8, -4]
      ],
      expected: [-6, -4, 8, 2]
    },
    {
      type: 'MultiLineString',
      coordinates: [
        [
          [-6, 2],
          [0, 0]
        ],
        [
          [8, -4],
          [3, 7]
        ]
      ],
      expected: [-6, -4, 8, 7]
    },
    {
      type: 'MultiPolygon',
      coordinates: [
        [
          [
            [-6, 2],
            [-2, 2],
            [-6, 2]
          ]
        ],
        [
          [
            [8, -4],
            [3, 7],
            [8, -4]
          ]
        ]
      ],
      expected: [-6, -4, 8, 7]
    }
  ])('reads $type bounds', ({type, coordinates, expected}) => {
    expect(getWorkflowDataBounds(makeBoundsData([{type, coordinates}]))).toEqual(expected);
  });

  test('recurses through geometry collections and ignores null geometry rows', () => {
    expect(
      getWorkflowDataBounds(
        makeBoundsData([
          null,
          {
            type: 'GeometryCollection',
            geometries: [
              {type: 'Point', coordinates: [-10, 3]},
              {
                type: 'GeometryCollection',
                geometries: [
                  {
                    type: 'LineString',
                    coordinates: [
                      [2, -4],
                      [9, 8]
                    ]
                  }
                ]
              }
            ]
          },
          null
        ])
      )
    ).toEqual([-10, -4, 9, 8]);
  });

  test('ignores incomplete and non-finite positions without dropping usable positions in the row', () => {
    const data = makeBoundsData([
      {
        type: 'LineString',
        coordinates: [
          [null, 2],
          [4, 5],
          [3, null],
          [8, 7],
          [Infinity, 1],
          [1, -Infinity],
          ['9', 3],
          [4]
        ]
      }
    ]);
    data.table.columns['geometry'] = [
      ...data.table.columns['geometry'],
      'not JSON',
      '{"type":"constructor","coordinates":[100,100]}'
    ];
    data.table.rowCount = 3;
    expect(getWorkflowDataBounds(data)).toEqual([4, 5, 8, 7]);
  });

  test('returns null for tables, empty or missing geometry, and no finite coordinate pairs', () => {
    const table = readNativeDatasetData(
      makeDataset([{name: 'value', type: 'real'}], [[1]]),
      makeRequest()
    );
    expect(getWorkflowDataBounds(table)).toBeNull();
    expect(getWorkflowDataBounds(makeBoundsData([]))).toBeNull();
    expect(
      getWorkflowDataBounds(
        makeBoundsData([
          null,
          {type: 'Point', coordinates: [null, 2]},
          {type: 'Point', coordinates: [1, Infinity]}
        ])
      )
    ).toBeNull();
    expect(getWorkflowDataBounds({...makeBoundsData([POINT]), kind: 'table'})).toBeNull();
    expect(getWorkflowDataBounds({...makeBoundsData([POINT]), table: {...table.table}})).toBeNull();
  });
});

const DRAWING_FEATURES: WorkflowDrawingFeature[] = [
  {
    id: 'polygon',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [2, 0],
          [2, 2],
          [0, 0]
        ]
      ]
    }
  },
  {
    id: 'rectangle',
    properties: {shape: 'Rectangle'},
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [3, 3],
          [4, 3],
          [4, 4],
          [3, 3]
        ]
      ]
    }
  },
  {
    id: 7,
    geometry: {
      type: 'LineString',
      coordinates: [
        [0, 0],
        [1, 1]
      ]
    }
  },
  {
    id: 'named',
    properties: {name: 'Study area'},
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 0]
        ]
      ]
    }
  },
  {geometry: {type: 'Point', coordinates: [1, 2]}}
];

describe('workflow display, drawing and SQL decisions', () => {
  const schema = new arrow.Table({
    value: arrow.vectorFromArray([1.5], new arrow.Float64()),
    category: arrow.vectorFromArray(['A'], new arrow.Utf8()),
    active: arrow.vectorFromArray([true], new arrow.Bool())
  }).schema;
  test('chooses numeric scales and shipped sequential or diverging palettes', () => {
    expect(chooseWorkflowDisplayChannel(schema, {colorColumn: 'value'})).toEqual({
      field: {name: 'value', type: 'real'},
      scale: 'quantile',
      rangeName: 'Blues'
    });
    expect(
      chooseWorkflowDisplayChannel(schema, {
        colorColumn: 'value',
        colorScale: 'quantize',
        palette: 'diverging'
      })
    ).toEqual({field: {name: 'value', type: 'real'}, scale: 'quantize', rangeName: 'RdBu'});
    expect(
      chooseWorkflowDisplayChannel(schema, {colorColumn: 'value', colorScale: 'ordinal'})
    ).toEqual({field: {name: 'value', type: 'real'}, scale: 'ordinal', rangeName: 'Set2'});
  });
  test('categorical fields use ordinal even when a numeric scale is requested; missing fields are ignored', () => {
    expect(
      chooseWorkflowDisplayChannel(schema, {
        colorColumn: 'category',
        colorScale: 'quantile',
        palette: 'diverging'
      })
    ).toEqual({field: {name: 'category', type: 'string'}, scale: 'ordinal', rangeName: 'Set2'});
    expect(chooseWorkflowDisplayChannel(schema, {colorColumn: 'active'})).toEqual({
      field: {name: 'active', type: 'boolean'},
      scale: 'ordinal',
      rangeName: 'Set2'
    });
    expect(chooseWorkflowDisplayChannel(schema, {colorColumn: 'missing'})).toBeNull();
    expect(chooseWorkflowDisplayChannel(schema, undefined)).toBeNull();
  });
  test('drawings have editor-order labels, stable identities and canonical geometry', () => {
    expect(profileWorkflowDrawings(DRAWING_FEATURES)).toEqual([
      {id: 'polygon', label: 'Polygon 1', kind: 'polygon'},
      {id: 'rectangle', label: 'Rectangle 1', kind: 'polygon'},
      {id: '7', label: 'Line 1', kind: 'line'},
      {id: 'named', label: 'Study area', kind: 'polygon'},
      {id: 'drawing-5', label: 'Point 1', kind: 'point'}
    ]);
    const data = makeWorkflowDataFromDrawings(DRAWING_FEATURES);
    expect(data.kind).toBe('features');
    expect(data.identityColumn).toBe('drawing_id');
    expect(data.table.columnTypes).toEqual({
      drawing_id: 'string',
      label: 'string',
      geometry: 'string'
    });
    expect(data.table.columns['drawing_id']).toEqual([
      'polygon',
      'rectangle',
      '7',
      'named',
      'drawing-5'
    ]);
    expect(data.table.columns['geometry']).toEqual(
      DRAWING_FEATURES.map(feature => JSON.stringify(feature.geometry))
    );
  });
  test('drawing conversion numbers each shape kind and refuses ambiguous identities', () => {
    const features = [
      ...DRAWING_FEATURES,
      {...DRAWING_FEATURES[1], id: 'rectangle-2'},
      {...DRAWING_FEATURES[0], id: 'polygon-3'},
      {id: 'empty', geometry: null}
    ];
    expect(makeWorkflowDataFromDrawings(features).table.columns['label']).toEqual([
      'Polygon 1',
      'Rectangle 1',
      'Line 1',
      'Study area',
      'Point 1',
      'Rectangle 2',
      'Polygon 3'
    ]);
    expectIssue(
      () => makeWorkflowDataFromDrawings([DRAWING_FEATURES[0], DRAWING_FEATURES[0]]),
      'dataset-read-failed',
      'Drawn shapes: Shape id "polygon" must be unique.'
    );
  });
  test('SQL result conversion preserves selected input geometry names and scalar types', () => {
    const input = makeFixturePoints(1);
    input.table.geometryColumn = 'shape';
    const table = new arrow.Table({
      shape: arrow.vectorFromArray([JSON.stringify(POINT), null], new arrow.Utf8()),
      total: arrow.vectorFromArray([2, 3], new arrow.Int32())
    });
    const result = makeWorkflowDataFromSqlTable(table, {
      query: 'SELECT shape, total FROM input1',
      tables: {input1: input},
      maximumRows: 2
    });
    expect(result.kind).toBe('point');
    expect(result.table.geometryColumn).toBe('shape');
    expect(result.table.columns).toEqual({
      shape: [JSON.stringify(POINT), null],
      total: [2, 3],
      row_index: [0, 1]
    });
    expect(result.table.columnTypes).toEqual({shape: 'string', total: 'int32', row_index: 'int32'});
    const scalar = makeWorkflowDataFromSqlTable(
      new arrow.Table({geometry: arrow.vectorFromArray(['ordinary text'], new arrow.Utf8())}),
      {query: 'SELECT 1', tables: {}, maximumRows: 2}
    );
    expect(scalar.kind).toBe('table');
    expect(scalar.table.geometryColumn).toBeUndefined();
  });
});

describe('workflow SQL scalar conversion', () => {
  test('wave 2 F4 preserves SQL identity-like names and allocates an unused synthetic identity', () => {
    const result = makeWorkflowDataFromSqlTable(
      new arrow.Table({
        row_index: arrow.vectorFromArray([42], new arrow.Int32()),
        row_index_2: arrow.vectorFromArray([99], new arrow.Int32())
      }),
      {query: 'SELECT 42 AS row_index, 99 AS row_index_2', tables: {}, maximumRows: 1}
    );
    expect(result.table.columns).toEqual({row_index: [42], row_index_2: [99], row_index_3: [0]});
    expect(result.identityColumn).toBe('row_index_3');
    expect(result.table.columnTypes).toEqual({
      row_index: 'int32',
      row_index_2: 'int32',
      row_index_3: 'int32'
    });
  });

  test.each([
    {values: []},
    {values: [null, null]}
  ])('wave 2 F8 keeps known geometry metadata for $values', ({values}) => {
    const input = makeFixturePoints(1);
    input.table.geometryColumn = 'shape';
    const result = makeWorkflowDataFromSqlTable(
      new arrow.Table({shape: arrow.vectorFromArray(values, new arrow.Utf8())}),
      {query: 'SELECT shape FROM input1', tables: {input1: input}, maximumRows: 2}
    );
    expect(result.table.geometryColumn).toBe('shape');
    expect(result.table.columns['shape']).toEqual(values);
    expect(result.table.columnTypes['shape']).toBe('string');
    expect(result.table.rowCount).toBe(values.length);
  });

  test('converts safe Int64 aggregates to numbers and unsafe integer columns to exact strings', () => {
    const data = makeWorkflowDataFromSqlTable(
      new arrow.Table({
        count: arrow.vectorFromArray([25n, 9007199254740991n, null], new arrow.Int64()),
        identifier: arrow.vectorFromArray([1n, 9007199254740993n, null], new arrow.Int64()),
        unsigned: arrow.vectorFromArray([0n, 18446744073709551615n, null], new arrow.Uint64())
      }),
      {query: 'SELECT 1', tables: {}, maximumRows: 3}
    );
    expect(data.kind).toBe('table');
    expect(data.table.columns).toEqual({
      count: [25, 9007199254740991, null],
      identifier: ['1', '9007199254740993', null],
      unsigned: ['0', '18446744073709551615', null],
      row_index: [0, 1, 2]
    });
    expect(data.table.columnTypes).toEqual({
      count: 'float64',
      identifier: 'string',
      unsigned: 'string',
      row_index: 'int32'
    });
  });

  test('converts scaled decimals and ISO dates and timestamps with nulls', () => {
    const data = makeWorkflowDataFromSqlTable(
      new arrow.Table({
        amount: arrow.vectorFromArray(
          [
            new Uint32Array([12345, 0, 0, 0]),
            new Uint32Array([4294966062, 4294967295, 4294967295, 4294967295]),
            null
          ],
          new arrow.Decimal(2, 10)
        ),
        day: arrow.vectorFromArray(
          [new Date('2026-11-14'), new Date('1970-01-01'), null],
          new arrow.DateDay()
        ),
        time: arrow.vectorFromArray(
          [new Date('2026-11-14T12:34:56.789Z'), new Date('1969-12-31T23:59:59.000Z'), null],
          new arrow.TimestampMicrosecond()
        )
      }),
      {query: 'SELECT 1', tables: {}, maximumRows: 3}
    );
    expect(data.table.columns).toEqual({
      amount: [123.45, -12.34, null],
      day: ['2026-11-14T00:00:00.000Z', '1970-01-01T00:00:00.000Z', null],
      time: ['2026-11-14T12:34:56.789Z', '1969-12-31T23:59:59.000Z', null],
      row_index: [0, 1, 2]
    });
    expect(data.table.columnTypes).toEqual({
      amount: 'float64',
      day: 'string',
      time: 'string',
      row_index: 'int32'
    });
  });

  test('detects geometry only by an input geometry name with valid non-null values', () => {
    const input = makeFixturePoints(1);
    input.table.geometryColumn = 'shape';
    const request = {query: 'SELECT shape FROM input1', tables: {input1: input}, maximumRows: 2};
    const makeTable = (name: string, second: string | null) =>
      new arrow.Table({
        [name]: arrow.vectorFromArray([JSON.stringify(POINT), second], new arrow.Utf8())
      });
    const valid = makeWorkflowDataFromSqlTable(makeTable('shape', null), request);
    expect(valid.kind).toBe('point');
    expect(valid.table.geometryColumn).toBe('shape');
    expect(valid.table.columns['shape']).toEqual([JSON.stringify(POINT), null]);
    expect(makeWorkflowDataFromSqlTable(makeTable('other', null), request).kind).toBe('table');
    expect(makeWorkflowDataFromSqlTable(makeTable('shape', 'invalid'), request).kind).toBe('table');
  });
});

describe('review regressions at the data boundary', () => {
  test('F3 refuses Arrow integers that cannot be represented exactly as numbers', () => {
    for (const [type, value] of [
      [new arrow.Int64(), 9007199254740993n],
      [new arrow.Uint64(), 18446744073709551615n]
    ] as const) {
      const table = new arrow.Table({identifier: arrow.vectorFromArray([value], type)});
      expectIssue(
        () => makeWorkflowDataFromArrowTable(table, makeRequest(), 'Identifiers'),
        'dataset-read-failed',
        `Identifiers: Column identifier at row 0 cannot represent integer ${value} exactly as a JavaScript number.`
      );
    }
    const table = new arrow.Table({
      identifier: arrow.vectorFromArray([2147483647n, 9007199254740991n, null], new arrow.Int64())
    });
    expect(
      makeWorkflowDataFromArrowTable(table, makeRequest(), 'Identifiers').table.columns[
        'identifier'
      ]
    ).toEqual([2147483647, 9007199254740991, null]);
  });

  test('F4 refuses duplicate ordinary field names before reading any cells', () => {
    const dataset = makeDataset(
      [
        {name: 'value', type: 'string'},
        {name: 'value', type: 'string'}
      ],
      [['first', 'second']]
    );
    expect(dataset.fields.map(field => field.name)).toEqual(['value', 'value']);
    const readCell = vi.spyOn(dataset.dataContainer, 'valueAt');
    expectIssue(
      () => readNativeDatasetData(dataset, makeRequest()),
      'dataset-read-failed',
      'Taxi trips: Field name value is duplicated; rename the fields to unique names before reading the dataset.'
    );
    expect(readCell).not.toHaveBeenCalled();
  });

  test('F7 preserves prototype-sensitive column names when publishing tables and points', () => {
    for (const point of [false, true]) {
      const dataset = makeDataset(
        [
          {name: '__proto__', type: 'string'},
          {name: 'constructor', type: 'string'},
          {name: 'shape', type: 'geojson'}
        ],
        [['retained?', 'yes', POINT]]
      );
      const data = readNativeDatasetData(
        dataset,
        makeRequest(point ? {mode: 'geojson-field', field: 'shape'} : {mode: 'none'})
      );
      const beforePublication = structuredClone(data);
      const published = makeWorkflowPublication(data, 'Published');
      expect(published.table.getChild('__proto__')?.get(0)).toBe('retained?');
      expect(published.table.getChild('constructor')?.get(0)).toBe('yes');
      expect(data).toEqual(beforePublication);
    }
  });

  test('F9 enforces Int32 source identities in native reads and Arrow extraction metadata', () => {
    const dataset = makeDataset([{name: 'value', type: 'integer'}], []);
    dataset.dataContainer = {numRows: () => 2147483650, valueAt: () => 7};
    const valid = readNativeDatasetData(dataset, makeRequest(), [2147483647]);
    expect(makeWorkflowPublication(valid, 'Boundary').table.getChild('row_index')?.get(0)).toBe(
      2147483647
    );
    expectIssue(
      () => readNativeDatasetData(dataset, makeRequest(), [2147483648]),
      'capacity-exceeded',
      'Taxi trips: Row index 2147483648 exceeds the Int32 identity limit (2147483647).'
    );
    const boundaryTable = new arrow.Table({
      source_row: arrow.vectorFromArray([2147483647], new arrow.Uint32())
    });
    const boundaryExtracted = new arrow.Table(
      new arrow.Schema(
        boundaryTable.schema.fields,
        new Map([['luma:row-extraction', JSON.stringify({sourceRowColumn: 'source_row'})]])
      ),
      boundaryTable.batches
    );
    expect(
      makeWorkflowPublication(
        makeWorkflowDataFromArrowTable(boundaryExtracted, makeRequest(), 'Boundary'),
        'Boundary'
      )
        .table.getChild('row_index')
        ?.get(0)
    ).toBe(2147483647);
    const table = new arrow.Table({
      source_row: arrow.vectorFromArray([2147483648], new arrow.Uint32())
    });
    const extracted = new arrow.Table(
      new arrow.Schema(
        table.schema.fields,
        new Map([['luma:row-extraction', JSON.stringify({sourceRowColumn: 'source_row'})]])
      ),
      table.batches
    );
    expectIssue(
      () => makeWorkflowDataFromArrowTable(extracted, makeRequest(), 'Extracted'),
      'capacity-exceeded',
      'Extracted: Row index 2147483648 exceeds the Int32 identity limit (2147483647).'
    );
  });

  test.each([
    'constructor',
    '__proto__',
    'toString'
  ])('F10 rejects inherited geometry type %s', type => {
    const dataset = makeDataset(
      [{name: 'shape', type: 'geojson'}],
      [[JSON.stringify({type, coordinates: []})]]
    );
    expect(profileNativeDataset(dataset, false).geometryFields).toEqual([]);
    const data = readNativeDatasetData(
      dataset,
      makeRequest({mode: 'geojson-field', field: 'shape'})
    );
    expect(data.table.columns['geometry']).toEqual([null]);
    expect(data.invalidGeometryCount).toBe(1);
  });
});
