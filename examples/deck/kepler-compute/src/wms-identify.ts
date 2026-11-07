// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp, toRadians} from '@math.gl/core';
import {WGS84_SEMI_MAJOR_AXIS_METERS} from './earth-constants';

/**
 * WMS identify (GetFeatureInfo), CRS/axis-order, time and error helpers. Pure: builds URLs and
 * parses text, performs no network I/O, so every function is unit testable in node.
 *
 * Audit of kepler.gl WMSLayer (layers/wms-layer + deckgl-layers/wms) that motivates this file:
 * - Rendering: a deck.gl CompositeLayer fetches ONE image for the whole viewport (not tiles) and
 *   draws it with BitmapLayer; srs is EPSG:4326 in globe and EPSG:3857 otherwise. The image is a
 *   texture, so rendering is host/deck owned and works on WebGPU; there is no numeric data.
 * - Axis order: the request is built by @loaders.gl/wms. WMS 1.3.0 mandates lat,lon BBOX order
 *   for EPSG:4326 (lon,lat for CRS:84 and for 1.1.1 SRS=EPSG:4326). `getWmsAxisOrder` and
 *   `formatWmsBbox` encode that rule so the host can verify requests.
 * - Time: Kepler sends no TIME parameter and does not read the Dimension element; `buildWms*`
 *   here accept `time` and `parseWmsTimeExtent` parses Dimension extents.
 * - Capabilities errors: Kepler's onMetadataLoadError is console.error; servers return
 *   ServiceExceptionReport with HTTP 200. `classifyWmsResponseError` classifies these.
 * - Attribution: Kepler uses metadata.attribution/label as title and the SERVICE url as link.
 *   `resolveWmsAttribution` prefers the capabilities Attribution OnlineResource.
 * - Feature info: Kepler requests only application/vnd.ogc.gml and extracts `gml:featureMember`
 *   with DOMParser. `parseWmsFeatureInfoResponse` also handles GeoJSON, text/plain and
 *   namespace-free GML and reports exceptions.
 * NUMERIC ANALYSIS: a WMS image is rendered pixels. Analysis requires a real numeric source
 * (WCS/GeoTIFF, WFS or a table); GetFeatureInfo values are single-point identify results.
 */

export type WmsVersion = '1.1.1' | '1.3.0';
export type WmsCrs = 'EPSG:4326' | 'EPSG:3857' | 'CRS:84';

const WMS_PARAMETER_KEYS = new Set([
  'service',
  'request',
  'version',
  'layers',
  'query_layers',
  'styles',
  'format',
  'crs',
  'srs',
  'bbox',
  'width',
  'height',
  'transparent',
  'info_format',
  'feature_count',
  'time',
  'i',
  'j',
  'x',
  'y',
  'exceptions'
]);

export function getWmsAxisOrder(version: WmsVersion, crs: WmsCrs): 'lonlat' | 'latlon' {
  return version === '1.3.0' && crs === 'EPSG:4326' ? 'latlon' : 'lonlat';
}

/** BBOX in CRS units from a lng/lat extent: degrees for 4326/CRS:84, meters for 3857. */
export function formatWmsBbox(
  extent: readonly [number, number, number, number],
  version: WmsVersion,
  crs: WmsCrs
): string {
  const [west, south, east, north] = extent;
  if (crs === 'EPSG:3857') {
    const [minX, minY] = projectLngLatToWebMercator(west, south);
    const [maxX, maxY] = projectLngLatToWebMercator(east, north);
    return [minX, minY, maxX, maxY].join(',');
  }
  return getWmsAxisOrder(version, crs) === 'latlon'
    ? [south, west, north, east].join(',')
    : [west, south, east, north].join(',');
}

const EARTH_RADIUS = WGS84_SEMI_MAJOR_AXIS_METERS;

export function projectLngLatToWebMercator(longitude: number, latitude: number): [number, number] {
  const clampedLatitude = clamp(latitude, -85.0511287798, 85.0511287798);
  return [
    EARTH_RADIUS * toRadians(longitude),
    EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + toRadians(clampedLatitude) / 2))
  ];
}

export type WmsRequestOptions = Readonly<{
  serviceUrl: string;
  version: WmsVersion;
  layers: readonly string[];
  crs: WmsCrs;
  /** Viewport extent as west, south, east, north in degrees (always lng/lat; axis order is applied here). */
  extent: readonly [number, number, number, number];
  width: number;
  height: number;
  styles?: readonly string[];
  format?: string;
  transparent?: boolean;
  /** ISO 8601 instant or interval, forwarded as TIME. */
  time?: string;
}>;

export type WmsFeatureInfoOptions = WmsRequestOptions &
  Readonly<{
    pixelX: number;
    pixelY: number;
    queryLayers?: readonly string[];
    infoFormat?: string;
    featureCount?: number;
  }>;

function createWmsUrl(
  serviceUrl: string,
  request: string,
  options: WmsRequestOptions,
  extraParameters: Record<string, string>
): string {
  const url = new URL(serviceUrl);
  for (const key of [...url.searchParams.keys()])
    if (WMS_PARAMETER_KEYS.has(key.toLowerCase())) url.searchParams.delete(key);
  const crsKey = options.version === '1.3.0' ? 'CRS' : 'SRS';
  const parameters: Record<string, string> = {
    SERVICE: 'WMS',
    VERSION: options.version,
    REQUEST: request,
    LAYERS: options.layers.join(','),
    STYLES: (options.styles ?? options.layers.map(() => '')).join(','),
    [crsKey]: options.crs,
    BBOX: formatWmsBbox(options.extent, options.version, options.crs),
    WIDTH: String(options.width),
    HEIGHT: String(options.height),
    ...extraParameters
  };
  if (options.time) parameters['TIME'] = options.time;
  for (const [key, value] of Object.entries(parameters)) url.searchParams.set(key, value);
  return url.toString();
}

export function buildWmsGetMapUrl(options: WmsRequestOptions): string {
  return createWmsUrl(options.serviceUrl, 'GetMap', options, {
    FORMAT: options.format ?? 'image/png',
    TRANSPARENT: options.transparent === false ? 'FALSE' : 'TRUE'
  });
}

/** 1.3.0 uses I/J and info_format; 1.1.1 uses X/Y. Pixel origin is the image top-left. */
export function buildWmsGetFeatureInfoUrl(options: WmsFeatureInfoOptions): string {
  const queryLayers = options.queryLayers ?? options.layers;
  const pixelKeys = options.version === '1.3.0' ? ['I', 'J'] : ['X', 'Y'];
  return createWmsUrl(options.serviceUrl, 'GetFeatureInfo', options, {
    FORMAT: options.format ?? 'image/png',
    QUERY_LAYERS: queryLayers.join(','),
    INFO_FORMAT: options.infoFormat ?? 'application/json',
    FEATURE_COUNT: String(options.featureCount ?? 10),
    [pixelKeys[0]]: String(Math.round(options.pixelX)),
    [pixelKeys[1]]: String(Math.round(options.pixelY))
  });
}

/** Pixel of a lng/lat inside the requested extent, for GetFeatureInfo; undefined when outside. */
export function getWmsPixelFromLngLat(
  longitude: number,
  latitude: number,
  options: Pick<WmsRequestOptions, 'extent' | 'width' | 'height' | 'crs'>
): [number, number] | undefined {
  const [west, south, east, north] = options.extent;
  const project = (lng: number, lat: number): [number, number] =>
    options.crs === 'EPSG:3857' ? projectLngLatToWebMercator(lng, lat) : [lng, lat];
  const [minX, minY] = project(west, south);
  const [maxX, maxY] = project(east, north);
  const [x, y] = project(longitude, latitude);
  if (x < minX || x > maxX || y < minY || y > maxY) return undefined;
  return [
    Math.floor(((x - minX) / (maxX - minX)) * options.width),
    Math.floor(((maxY - y) / (maxY - minY)) * options.height)
  ];
}

export type WmsTimeExtent = Readonly<{
  /** Discrete instants, as listed. */
  values: readonly string[];
  /** start/end/period triplets, as listed. */
  intervals: readonly Readonly<{start: string; end: string; period?: string}>[];
  defaultValue?: string;
}>;

/** Parses a Dimension name="time" extent such as "2020-01-01,2020-02-01/2020-12-01/P1M". */
export function parseWmsTimeExtent(extentText: string, defaultValue?: string): WmsTimeExtent {
  const values: string[] = [];
  const intervals: {start: string; end: string; period?: string}[] = [];
  for (const part of extentText
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)) {
    const pieces = part.split('/');
    if (pieces.length >= 2) intervals.push({start: pieces[0], end: pieces[1], period: pieces[2]});
    else values.push(part);
  }
  return {values, intervals, defaultValue};
}

/** True when the requested instant is listed or lies inside a listed interval (period ignored). */
export function isWmsTimeSupported(extent: WmsTimeExtent, time: string): boolean {
  const instant = Date.parse(time);
  return (
    extent.values.some(value => Date.parse(value) === instant) ||
    extent.intervals.some(
      interval => instant >= Date.parse(interval.start) && instant <= Date.parse(interval.end)
    )
  );
}

export type WmsServiceException = Readonly<{code?: string; message: string}>;

export type WmsErrorKind = 'network' | 'http-error' | 'service-exception' | 'not-wms-xml' | 'none';

/** Classifies a response. Servers report WMS errors as HTTP 200 with ServiceExceptionReport. */
export function classifyWmsResponseError(response: {
  networkError?: boolean;
  status?: number;
  contentType?: string;
  bodyText?: string;
  expectXml?: boolean;
}): {kind: WmsErrorKind; exception?: WmsServiceException} {
  if (response.networkError) return {kind: 'network'};
  if (response.status !== undefined && (response.status < 200 || response.status >= 300))
    return {kind: 'http-error'};
  const exception = parseWmsServiceException(response.bodyText ?? '');
  if (exception) return {kind: 'service-exception', exception};
  if (
    response.expectXml &&
    !/^\s*(?:<\?xml[^>]*>\s*)?<(?:\w+:)?(?:WMS_Capabilities|WMT_MS_Capabilities)/.test(
      response.bodyText ?? ''
    )
  )
    return {kind: 'not-wms-xml'};
  return {kind: 'none'};
}

export function parseWmsServiceException(text: string): WmsServiceException | undefined {
  const match =
    /<(?:\w+:)?ServiceException\b([^>]*)>([\s\S]*?)<\/(?:\w+:)?ServiceException>/.exec(text) ??
    /<(?:\w+:)?Exception\b([^>]*)>\s*<(?:\w+:)?ExceptionText>([\s\S]*?)<\/(?:\w+:)?ExceptionText>/.exec(
      text
    );
  if (!match) return undefined;
  const code = /(?:code|exceptionCode)="([^"]*)"/.exec(match[1])?.[1];
  return {code, message: decodeXmlText(match[2]).trim()};
}

export type WmsAttribution = Readonly<{title: string; url: string}>;

export function resolveWmsAttribution(metadata: {
  attribution?: string;
  attributionUrl?: string;
  label?: string;
  tilesetDataUrl?: string;
}): WmsAttribution | undefined {
  const title = metadata.attribution ?? metadata.label;
  if (!title) return undefined;
  return {title, url: metadata.attributionUrl ?? metadata.tilesetDataUrl ?? ''};
}

export type WmsIdentifiedFeature = Readonly<{
  name: string;
  attributes: readonly Readonly<{name: string; value: string}>[];
}>;

export type WmsFeatureInfoResult = Readonly<{
  format: 'json' | 'gml' | 'text' | 'unknown';
  features: readonly WmsIdentifiedFeature[];
  exception?: WmsServiceException;
}>;

function decodeXmlText(text: string): string {
  return text
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&');
}

type XmlNode = {name: string; children: XmlNode[]; text: string};

function parseXmlTree(text: string): XmlNode {
  const root: XmlNode = {name: '#root', children: [], text: ''};
  const stack = [root];
  for (const token of text.match(/<[^>]+>|[^<]+/g) ?? []) {
    const current = stack[stack.length - 1];
    if (token.startsWith('<?') || token.startsWith('<!--') || token.startsWith('<!DOCTYPE'))
      continue;
    if (token.startsWith('<![CDATA[')) current.text += token;
    else if (token.startsWith('</')) stack.length > 1 && stack.pop();
    else if (token.startsWith('<')) {
      const name = /^<\s*([^\s/>]+)/.exec(token)?.[1] ?? '';
      const node: XmlNode = {name: name.replace(/^.*:/, ''), children: [], text: ''};
      current.children.push(node);
      if (!token.endsWith('/>')) stack.push(node);
    } else current.text += token;
  }
  return root;
}

const GEOMETRY_NAME = /geom|boundedby|shape|the_geom/i;

function collectGmlFeatures(node: XmlNode, features: WmsIdentifiedFeature[]): void {
  for (const child of node.children) {
    if (child.name === 'featureMember' || child.name === 'featureMembers') {
      for (const feature of child.children) {
        const attributes = feature.children
          .filter(
            attribute => attribute.children.length === 0 && !GEOMETRY_NAME.test(attribute.name)
          )
          .map(attribute => ({name: attribute.name, value: decodeXmlText(attribute.text).trim()}))
          .filter(attribute => attribute.value !== '');
        features.push({name: feature.name, attributes});
      }
    } else collectGmlFeatures(child, features);
  }
}

/** Parses an identify response by sniffing its body (the content type is advisory). */
export function parseWmsFeatureInfoResponse(bodyText: string): WmsFeatureInfoResult {
  const exception = parseWmsServiceException(bodyText);
  if (exception) return {format: 'unknown', features: [], exception};
  const trimmed = bodyText.trim();
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed) as {
        features?: {id?: unknown; properties?: Record<string, unknown>}[];
      };
      const features = (parsed.features ?? []).map((feature, index) => ({
        name: String(feature.id ?? `feature ${index}`),
        attributes: Object.entries(feature.properties ?? {})
          .filter(([, value]) => value !== null && value !== undefined && typeof value !== 'object')
          .map(([name, value]) => ({name, value: String(value)}))
      }));
      return {format: 'json', features};
    } catch {
      return {format: 'unknown', features: []};
    }
  }
  if (trimmed.startsWith('<')) {
    const features: WmsIdentifiedFeature[] = [];
    collectGmlFeatures(parseXmlTree(trimmed), features);
    return {format: 'gml', features};
  }
  // text/plain (GeoServer): "Results for FeatureType 'x':" then "name = value" lines.
  const attributes = trimmed
    .split(/\r?\n/)
    .map(line => /^\s*([^=:\s][^=]*?)\s*=\s*(.*?)\s*$/.exec(line))
    .filter((match): match is RegExpExecArray => Boolean(match) && match![2] !== 'null')
    .map(match => ({name: match[1], value: match[2]}));
  return {
    format: attributes.length ? 'text' : 'unknown',
    features: attributes.length ? [{name: 'feature', attributes}] : []
  };
}

/** What the host may tell analysis about a WMS layer. */
export function describeWmsAnalysisSupport(): Readonly<{
  numericAnalysisSupported: false;
  reason: string;
}> {
  return {
    numericAnalysisSupported: false,
    reason:
      'A WMS layer is a rendered image; use a numeric source (WCS, GeoTIFF, WFS or a table) for analysis. GetFeatureInfo is a single-point identify, not a sample population.'
  };
}
