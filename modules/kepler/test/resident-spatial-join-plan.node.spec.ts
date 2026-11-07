// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  aggregateResidentSpatialJoinValues,
  createResidentSpatialJoinPlan,
  type ResidentSpatialJoinSourceMetadata
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-plan';
import {packSpatialJoinPolygons} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-polygons';
import {
  mergeSpatialJoinGeometries,
  readSpatialJoinPolygonGeometry
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-merge';
import {parseResidentJoinLineage} from '../../../examples/deck/kepler-compute/src/resident-join-lineage';

// Load the pinned native CPU implementation with only the real isInPolygon and aggregation helpers.
vi.mock('@kepler.gl/utils', async () => {
  const dist = new URL('../../../node_modules/@kepler.gl/utils/dist/esm/', import.meta.url)
    .pathname;
  const aggregation = await vi.importActual<Record<string, unknown>>(`${dist}aggregation.js`);
  const filters = await vi.importActual<Record<string, unknown>>(`${dist}filter-utils.js`);
  return {...aggregation, isInPolygon: filters['isInPolygon']};
});
const {spatialJoinDatasets, geometriesMatchPredicate} = await vi.importActual<{
  spatialJoinDatasets: (...args: unknown[]) => {
    data: {fields: unknown[]; rows: unknown[][]};
    metadata?: {derivedDataset?: unknown};
  };
  geometriesMatchPredicate: (left: unknown, right: unknown, predicate: string) => boolean;
}>(
  new URL(
    '../../../node_modules/@kepler.gl/table/dist/esm/dataset-ops/spatial-join.js',
    import.meta.url
  ).pathname
);
const {aggregateValues: aggregateNativeValues} = await vi.importActual<{
  aggregateValues: (values: unknown[], technique: string) => unknown;
}>(
  new URL(
    '../../../node_modules/@kepler.gl/table/dist/esm/dataset-ops/aggregations.js',
    import.meta.url
  ).pathname
);
const {RowDataContainer} = await import(
  // @ts-expect-error Direct implementation file has no declaration entry.
  new URL('../../../node_modules/@kepler.gl/utils/dist/esm/row-data-container.js', import.meta.url)
    .pathname
);

const square = {
  type: 'Feature',
  properties: {},
  geometry: {
    type: 'Polygon',
    coordinates: [
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0]
      ]
    ]
  }
};
const pointFields = [
  {name: 'lng', type: 'real', analyzerType: 'FLOAT', fieldIdx: 0, id: 'lng', format: ''},
  {name: 'lat', type: 'real', analyzerType: 'FLOAT', fieldIdx: 1, id: 'lat', format: ''},
  {name: 'size', type: 'integer', analyzerType: 'INT', fieldIdx: 2, id: 'size', format: ''}
];
const polygonFields = [
  {name: 'geometry', type: 'geojson', analyzerType: 'GEOMETRY', fieldIdx: 0, id: 'g', format: ''},
  {name: 'score', type: 'real', analyzerType: 'FLOAT', fieldIdx: 1, id: 's', format: ''}
];
const points = [
  [5, 5, 1],
  [0, 5, 2], // on the boundary
  [20, 20, 3],
  [5, 5, 4]
];
const leftMetadata: ResidentSpatialJoinSourceMetadata = {
  datasetId: 'points',
  label: 'Points',
  revision: 1,
  rowCount: points.length,
  fields: pointFields,
  columns: {lng: {format: 'float32'}, lat: {format: 'float32'}, size: {format: 'sint32'}},
  dictionaries: {},
  nullCounts: {},
  coordinates: {longitude: 'lng', latitude: 'lat'}
};
const rightMetadata: ResidentSpatialJoinSourceMetadata = {
  datasetId: 'polygons',
  label: 'Polygons',
  revision: 1,
  rowCount: 1,
  fields: polygonFields,
  columns: {score: {format: 'float32'}},
  dictionaries: {},
  nullCounts: {},
  geometryColumn: 'geometry'
};
const operation = {
  id: 'spatial',
  leftDataId: 'points',
  rightDataId: 'polygons',
  leftGeo: {kind: 'latlng', latField: 'lat', lngField: 'lng'},
  rightGeo: {kind: 'geojson', fieldName: 'geometry'},
  predicate: 'within',
  aggregations: {score: 'average'},
  resultId: 'result',
  resultLabel: 'result'
};
function table(id: string, fields: unknown[], rows: unknown[][]) {
  return {id, label: id, color: [1, 2, 3], fields, dataContainer: new RowDataContainer({rows})};
}

test('admitted schema and row semantics match pinned native Spatial Join', () => {
  const plan = createResidentSpatialJoinPlan(operation, leftMetadata, rightMetadata);
  const native = spatialJoinDatasets(
    table('points', pointFields, points),
    table('polygons', polygonFields, [[JSON.stringify(square), 7]]),
    {
      leftGeo: operation.leftGeo,
      rightGeo: operation.rightGeo,
      predicate: 'within',
      aggregations: operation.aggregations,
      resultId: 'result',
      label: 'result',
      operationId: 'spatial'
    }
  );
  expect(plan.provenance.fields).toEqual(native.data.fields);
  expect(plan.derivedDataset).toEqual(native.metadata?.derivedDataset);
  // Native: one row per left row; boundary excluded by within; count and average of matches.
  expect(native.data.rows.map(row => [row[3], row[4]])).toEqual([
    [1, 7],
    [0, null],
    [0, null],
    [1, 7]
  ]);
  expect(plan.provenance.boundaryPolicy).toBe('open');
  expect(
    createResidentSpatialJoinPlan(
      {...operation, predicate: 'intersects'},
      leftMetadata,
      rightMetadata
    ).provenance.boundaryPolicy
  ).toBe('closed');
});

test('predicate boundary behavior documented by the plan matches native', () => {
  const boundary = {kind: 'point', lngLat: [0, 5]};
  const polygon = {kind: 'polygon', feature: square};
  expect(geometriesMatchPredicate(boundary, polygon, 'within')).toBe(false);
  expect(geometriesMatchPredicate(boundary, polygon, 'intersects')).toBe(true);
  expect(geometriesMatchPredicate({kind: 'point', lngLat: [5, 5]}, polygon, 'within')).toBe(true);
});

test('aggregations match native null and count handling', () => {
  const values = [3, null, 5];
  expect(aggregateResidentSpatialJoinValues(values, 'count')).toBe(3);
  expect(aggregateResidentSpatialJoinValues(values, 'sum')).toBe(8);
  expect(aggregateResidentSpatialJoinValues(values, 'average')).toBe(4);
  expect(aggregateResidentSpatialJoinValues(values, 'maximum')).toBe(5);
  expect(aggregateResidentSpatialJoinValues(values, 'minimum')).toBe(3);
  expect(aggregateResidentSpatialJoinValues([null], 'sum')).toBeNull();
  expect(aggregateResidentSpatialJoinValues([], 'count')).toBe(0);
});

test('admission rejects unsupported configurations with SQL guidance', () => {
  for (const change of [
    {predicate: 'touches'},
    {aggregations: {score: 'median'}},
    {aggregations: {score: 'merge'}},
    {leftGeo: {kind: 'geojson', fieldName: 'geometry'}},
    {rightGeo: {kind: 'latlng', latField: 'lat', lngField: 'lng'}}
  ])
    expect(() =>
      createResidentSpatialJoinPlan({...operation, ...change}, leftMetadata, rightMetadata)
    ).toThrow(/SQL/);
});

test('plan has no row, polygon or pair caps', () => {
  const plan = createResidentSpatialJoinPlan(
    operation,
    {...leftMetadata, rowCount: 50_000_000},
    {...rightMetadata, rowCount: 100_000}
  );
  expect(plan.left.rowCount).toBe(50_000_000);
  expect(plan.right.rowCount).toBe(100_000);
  expect(plan).not.toHaveProperty('pairCapacity');
  for (const rowCount of [0, -1, 1.5])
    expect(() =>
      createResidentSpatialJoinPlan(operation, {...leftMetadata, rowCount}, rightMetadata)
    ).toThrow(/SQL/);
});

test('polygon packing rejects mixed geometry and lineage round-trips', () => {
  const coordinate = new arrow.FixedSizeList(2, new arrow.Field('c', new arrow.Float64()));
  const type = new arrow.List(
    new arrow.Field(
      'part',
      new arrow.List(new arrow.Field('ring', new arrow.List(new arrow.Field('v', coordinate))))
    )
  );
  const vector = arrow.vectorFromArray(
    [
      [
        [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0]
          ],
          [
            [2, 2],
            [4, 2],
            [4, 4]
          ]
        ]
      ],
      null
    ],
    type
  );
  const packed = packSpatialJoinPolygons({
    metadata: {
      rowCount: 2,
      featureTypes: {point: false, line: false, polygon: true},
      invalidRows: []
    },
    batches: [{id: '0', rowCount: 2, sourceRowOffset: 0, polygons: vector}]
  } as never);
  // Closing vertex dropped; a null row is an empty feature; hole ring follows its shell.
  expect([...packed.featureOffsets]).toEqual([0, 1, 1]);
  expect([...packed.polygonOffsets]).toEqual([0, 2]);
  expect([...packed.ringOffsets]).toEqual([0, 4, 7]);
  expect(packed.vertexCount).toBe(7);
  expect(() =>
    packSpatialJoinPolygons({
      metadata: {
        rowCount: 1,
        featureTypes: {point: true, line: false, polygon: true},
        invalidRows: []
      },
      batches: []
    } as never)
  ).toThrow(/polygon or multipolygon rows only.*SQL/);
  const plan = createResidentSpatialJoinPlan(operation, leftMetadata, rightMetadata);
  const parsed = parseResidentJoinLineage(JSON.parse(JSON.stringify(plan.provenance)));
  expect(parsed).toEqual(plan.provenance);
  expect(() => parseResidentJoinLineage({...plan.provenance, kind: 'other'})).toThrow();
});

test('merge of the polygon column matches pinned native and routes to the bounded CPU step', () => {
  const mergeOperation = {...operation, aggregations: {geometry: 'merge', score: 'average'}};
  const plan = createResidentSpatialJoinPlan(mergeOperation, leftMetadata, rightMetadata);
  const native = spatialJoinDatasets(
    table('points', pointFields, points),
    table('polygons', polygonFields, [[JSON.stringify(square), 7]]),
    {
      leftGeo: operation.leftGeo,
      rightGeo: operation.rightGeo,
      predicate: 'within',
      aggregations: mergeOperation.aggregations,
      resultId: 'result',
      label: 'result',
      operationId: 'spatial'
    }
  );
  expect(plan.provenance.fields).toEqual(native.data.fields);
  expect(plan.columns.find(column => column.outputColumn === 'geometry_merge')).toMatchObject({
    kind: 'aggregate',
    technique: 'merge',
    type: 'geojson'
  });
  expect(() =>
    createResidentSpatialJoinPlan(
      mergeOperation,
      {...leftMetadata, rowCount: 262145},
      rightMetadata
    )
  ).toThrow(/merge.*262144.*uncheck merge/);
});

test('merged geometry equals native mergeGeometries for polygon and multipolygon rows', () => {
  const coordinate = new arrow.FixedSizeList(2, new arrow.Field('c', new arrow.Float64()));
  const type = new arrow.List(
    new arrow.Field(
      'part',
      new arrow.List(new arrow.Field('ring', new arrow.List(new arrow.Field('v', coordinate))))
    )
  );
  const ring = (offset: number) => [
    [offset, 0],
    [offset + 1, 0],
    [offset + 1, 1],
    [offset, 0]
  ];
  // Row 0 and 1 are Polygons, row 2 is a two-part MultiPolygon.
  const vector = arrow.vectorFromArray([[[ring(0)]], [[ring(2)]], [[ring(4)], [ring(6)]]], type);
  const source = {
    batches: [{id: '0', rowCount: 3, sourceRowOffset: 0, polygons: vector}]
  } as never;
  const geometries = [0, 1, 2].map(row => readSpatialJoinPolygonGeometry(source, row));
  const asFeature = (geometry: unknown) => ({type: 'Feature', geometry, properties: {}});
  for (const rows of [[], [0], [2], [0, 1], [2, 2], [0, 2], [0, 1, 2]]) {
    const merged = mergeSpatialJoinGeometries(rows.map(row => geometries[row]));
    const expected = aggregateNativeValues(
      rows.map(row => asFeature(geometries[row])),
      'merge'
    );
    expect(merged === null ? null : JSON.parse(merged)).toEqual(expected);
  }
});

test('declared EPSG:4326 and EPSG:3857 sides plan a GPU point transform into the polygon CRS', () => {
  const crs = (leftCrs: string, rightCrs: string) =>
    createResidentSpatialJoinPlan({...operation, leftCrs, rightCrs}, leftMetadata, rightMetadata)
      .provenance.crs;
  expect(crs('EPSG:4326', 'EPSG:4326')?.pointTransform).toBe('none');
  expect(crs('EPSG:3857', 'EPSG:3857')?.pointTransform).toBe('none');
  expect(crs('EPSG:4326', 'EPSG:3857')?.pointTransform).toBe('longitude-latitude-to-mercator');
  expect(crs('900913', 'epsg:4326')).toEqual({
    points: 'EPSG:3857',
    polygons: 'EPSG:4326',
    pointTransform: 'mercator-to-longitude-latitude'
  });
  expect(() => crs('EPSG:4326', 'EPSG:2154')).toThrow(
    /Spatial Join runs on EPSG:4326, EPSG:3857, UTM/
  );
});
