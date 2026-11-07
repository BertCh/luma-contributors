// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  buildWmsGetFeatureInfoUrl,
  buildWmsGetMapUrl,
  classifyWmsResponseError,
  formatWmsBbox,
  getWmsAxisOrder,
  getWmsPixelFromLngLat,
  isWmsTimeSupported,
  parseWmsFeatureInfoResponse,
  parseWmsTimeExtent,
  resolveWmsAttribution
} from '../../../examples/deck/kepler-compute/src/wms-identify';

const base = {
  serviceUrl: 'https://example.test/wms?map=a.map&REQUEST=bogus&vendor=1',
  layers: ['roads', 'rivers'],
  extent: [-10, 40, 10, 50] as [number, number, number, number],
  width: 400,
  height: 200
};

describe('wms request building', () => {
  it('uses lat,lon axis order for EPSG:4326 only in 1.3.0', () => {
    expect(getWmsAxisOrder('1.3.0', 'EPSG:4326')).toBe('latlon');
    expect(getWmsAxisOrder('1.3.0', 'CRS:84')).toBe('lonlat');
    expect(getWmsAxisOrder('1.1.1', 'EPSG:4326')).toBe('lonlat');
    expect(formatWmsBbox(base.extent, '1.3.0', 'EPSG:4326')).toBe('40,-10,50,10');
    expect(formatWmsBbox(base.extent, '1.1.1', 'EPSG:4326')).toBe('-10,40,10,50');
  });

  it('builds GetMap with CRS/SRS, TIME and keeps vendor parameters', () => {
    const url = new URL(
      buildWmsGetMapUrl({...base, version: '1.3.0', crs: 'EPSG:4326', time: '2020-01-01'})
    );
    expect(url.searchParams.get('CRS')).toBe('EPSG:4326');
    expect(url.searchParams.get('BBOX')).toBe('40,-10,50,10');
    expect(url.searchParams.get('TIME')).toBe('2020-01-01');
    expect(url.searchParams.get('REQUEST')).toBe('GetMap');
    expect(url.searchParams.get('map')).toBe('a.map');
    expect(url.searchParams.get('vendor')).toBe('1');
    const legacy = new URL(buildWmsGetMapUrl({...base, version: '1.1.1', crs: 'EPSG:4326'}));
    expect(legacy.searchParams.get('SRS')).toBe('EPSG:4326');
    expect(legacy.searchParams.has('CRS')).toBe(false);
  });

  it('builds GetFeatureInfo with I/J for 1.3.0 and X/Y for 1.1.1', () => {
    const v13 = new URL(
      buildWmsGetFeatureInfoUrl({
        ...base,
        version: '1.3.0',
        crs: 'EPSG:3857',
        pixelX: 10.4,
        pixelY: 20
      })
    );
    expect(v13.searchParams.get('I')).toBe('10');
    expect(v13.searchParams.get('J')).toBe('20');
    expect(v13.searchParams.get('QUERY_LAYERS')).toBe('roads,rivers');
    expect(v13.searchParams.get('BBOX')?.split(',')[0]).toBe('-1113194.9079327357');
    const v111 = new URL(
      buildWmsGetFeatureInfoUrl({
        ...base,
        version: '1.1.1',
        crs: 'EPSG:4326',
        pixelX: 1,
        pixelY: 2,
        infoFormat: 'text/plain'
      })
    );
    expect(v111.searchParams.get('X')).toBe('1');
    expect(v111.searchParams.get('INFO_FORMAT')).toBe('text/plain');
  });

  it('maps lng/lat to pixels in both CRS', () => {
    expect(getWmsPixelFromLngLat(0, 45, {...base, crs: 'EPSG:4326'})).toEqual([200, 100]);
    expect(getWmsPixelFromLngLat(-10, 50, {...base, crs: 'EPSG:3857'})).toEqual([0, 0]);
    expect(getWmsPixelFromLngLat(20, 45, {...base, crs: 'EPSG:4326'})).toBeUndefined();
  });
});

describe('wms time, errors and attribution', () => {
  it('parses and checks time extents', () => {
    const extent = parseWmsTimeExtent('2020-01-01,2021-01-01/2021-12-31/P1D', '2020-01-01');
    expect(extent.values).toEqual(['2020-01-01']);
    expect(extent.intervals).toEqual([{start: '2021-01-01', end: '2021-12-31', period: 'P1D'}]);
    expect(isWmsTimeSupported(extent, '2021-06-01')).toBe(true);
    expect(isWmsTimeSupported(extent, '2022-06-01')).toBe(false);
  });

  it('classifies failures including 200 responses carrying exceptions', () => {
    expect(classifyWmsResponseError({networkError: true}).kind).toBe('network');
    expect(classifyWmsResponseError({status: 404}).kind).toBe('http-error');
    const body =
      '<?xml version="1.0"?><ServiceExceptionReport><ServiceException code="LayerNotDefined">No layer &quot;x&quot;</ServiceException></ServiceExceptionReport>';
    expect(classifyWmsResponseError({status: 200, bodyText: body})).toMatchObject({
      kind: 'service-exception',
      exception: {code: 'LayerNotDefined', message: 'No layer "x"'}
    });
    expect(
      classifyWmsResponseError({status: 200, bodyText: '<html></html>', expectXml: true}).kind
    ).toBe('not-wms-xml');
    expect(
      classifyWmsResponseError({status: 200, bodyText: '<WMS_Capabilities/>', expectXml: true}).kind
    ).toBe('none');
  });

  it('resolves attribution', () => {
    expect(resolveWmsAttribution({label: 'Svc', tilesetDataUrl: 'https://s'})).toEqual({
      title: 'Svc',
      url: 'https://s'
    });
    expect(resolveWmsAttribution({})).toBeUndefined();
  });
});

describe('wms feature info parsing', () => {
  it('parses GeoJSON', () => {
    const result = parseWmsFeatureInfoResponse(
      JSON.stringify({features: [{id: 'a.1', properties: {pop: 5, nested: {x: 1}, none: null}}]})
    );
    expect(result).toEqual({
      format: 'json',
      features: [{name: 'a.1', attributes: [{name: 'pop', value: '5'}]}]
    });
  });

  it('parses GML with and without namespace prefixes, skipping geometry', () => {
    const gml =
      '<wfs:FeatureCollection xmlns:gml="g"><gml:featureMember><ns:roads><ns:the_geom><gml:Point/></ns:the_geom><ns:name>A &amp; B</ns:name><ns:empty></ns:empty></ns:roads></gml:featureMember></wfs:FeatureCollection>';
    expect(parseWmsFeatureInfoResponse(gml)).toEqual({
      format: 'gml',
      features: [{name: 'roads', attributes: [{name: 'name', value: 'A & B'}]}]
    });
    expect(
      parseWmsFeatureInfoResponse(
        '<FeatureCollection><featureMember><r><v>1</v></r></featureMember></FeatureCollection>'
      ).features
    ).toHaveLength(1);
  });

  it('parses text/plain and exceptions', () => {
    const text = "Results for FeatureType 'roads':\n-----\nname = Main\nlanes = 2\nnote = null\n";
    expect(parseWmsFeatureInfoResponse(text).features[0].attributes).toEqual([
      {name: 'name', value: 'Main'},
      {name: 'lanes', value: '2'}
    ]);
    expect(
      parseWmsFeatureInfoResponse(
        '<ServiceExceptionReport><ServiceException>bad</ServiceException></ServiceExceptionReport>'
      ).exception?.message
    ).toBe('bad');
    expect(parseWmsFeatureInfoResponse('').features).toEqual([]);
  });
});
