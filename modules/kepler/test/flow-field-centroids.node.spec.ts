// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';

vi.mock('@kepler.gl/layers', () => ({parseGeoJsonRawFeature: () => null}));

const {FlowFieldCentroidCache, getFlowFieldGeometryCentroid, readFlowFieldCentroids} = await import(
  '../../../examples/deck/kepler-compute/src/flow-field-centroids'
);
const {getFlowFieldPositionColumns} = await import(
  '../../../examples/deck/kepler-compute/src/flow-field-bindings'
);

const parseFeature = (value: unknown) =>
  typeof value === 'string' && value !== 'bad' ? {geometry: JSON.parse(value)} : null;

function makeTable(values: string[]): arrow.Table {
  return arrow.tableFromArrays({geometry: values});
}

describe('flow field centroids', () => {
  test('averages all vertex positions like native getCentroidFromGeometry', () => {
    expect(getFlowFieldGeometryCentroid({type: 'Point', coordinates: [3, 4]})).toEqual([3, 4]);
    expect(
      getFlowFieldGeometryCentroid({
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [4, 0],
            [4, 2],
            [0, 2],
            [0, 0]
          ]
        ]
      })
    ).toEqual([1.6, 0.8]);
    expect(
      getFlowFieldGeometryCentroid({
        type: 'GeometryCollection',
        geometries: [
          {type: 'Point', coordinates: [0, 0]},
          {
            type: 'LineString',
            coordinates: [
              [2, 2],
              [4, 4]
            ]
          }
        ]
      })
    ).toEqual([2, 2]);
    expect(getFlowFieldGeometryCentroid({type: 'Polygon', coordinates: []})).toBeNull();
  });

  test('reads per-row centroids with NaN for failures and honors the row limit', () => {
    const table = makeTable([
      '{"type":"Point","coordinates":[1,2]}',
      'bad',
      '{"type":"Point","coordinates":[5,6]}'
    ]);
    const result = readFlowFieldCentroids(table, 'geometry', {parseFeature});
    expect(Array.from(result.longitudes.slice(0, 1))).toEqual([1]);
    expect(Number.isNaN(result.longitudes[1])).toBe(true);
    expect(result.latitudes[2]).toBe(6);
    expect(result.truncatedCount).toBe(0);
    const limited = readFlowFieldCentroids(table, 'geometry', {parseFeature, rowLimit: 2});
    expect(limited.truncatedCount).toBe(1);
    expect(Number.isNaN(limited.longitudes[2])).toBe(true);
  });

  test('passes the GeoArrow encoding from field metadata', () => {
    const table = makeTable(['x']);
    const field = table.schema.fields[0];
    field.metadata.set('ARROW:extension:name', 'geoarrow.point');
    const seen: unknown[] = [];
    readFlowFieldCentroids(table, 'geometry', {
      parseFeature: (_value, encoding) => {
        seen.push(encoding);
        return null;
      }
    });
    expect(seen).toEqual(['geoarrow.point']);
  });

  test('cache parses once per table and column', () => {
    const cache = new FlowFieldCentroidCache();
    const table = makeTable(['x']);
    const first = cache.get('a', table, 'geometry');
    expect(cache.get('a', table, 'geometry')).toBe(first);
    expect(cache.get('a', makeTable(['x']), 'geometry')).not.toBe(first);
  });

  test('lat/lng is preferred over geojson positions', () => {
    const value = (name: string) => ({value: name});
    expect(
      getFlowFieldPositionColumns({lat: value('a'), lng: value('b'), geojson: value('g')})
    ).toEqual({longitude: 'b', latitude: 'a'});
    expect(getFlowFieldPositionColumns({lat: value('a'), geojson: value('g')})).toEqual({
      geojson: 'g'
    });
    expect(getFlowFieldPositionColumns({})).toEqual({});
  });
});
