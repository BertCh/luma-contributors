// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toRadians} from '@math.gl/core';
import {EARTH_RADIUS_METERS, getGreatCircleDistanceMeters} from './catalog-geodesy';

/** Coordinate systems a catalog binding can interpret explicitly; nothing is inferred. */
export type CatalogCoordinateSystem = 'EPSG:4326' | 'planar';

export type CatalogPosition = readonly [number, number];

export type CatalogGeometry = {type: string; coordinates?: unknown} | null;

export type CatalogFeature = {
  id?: string | number | null;
  geometry: CatalogGeometry;
  properties?: Record<string, unknown> | null;
};

export type CatalogFeatureCollection = {type: 'FeatureCollection'; features: CatalogFeature[]};

export type CatalogDiagnostic = {
  code: string;
  severity: 'info' | 'warning' | 'error';
  count: number;
  message: string;
};

export type CatalogColumnType = 'string' | 'float64' | 'int32' | 'boolean';

export type CatalogOutputTable = {
  id: string;
  label: string;
  rowCount: number;
  columns: Record<string, readonly (string | number | boolean | null)[]>;
  columnTypes: Record<string, CatalogColumnType>;
  /** Column holding GeoJSON geometry text; the native Sources workflow publishes it as geometry. */
  geometryColumn?: string;
};

export {EARTH_RADIUS_METERS};

/** Accepts a FeatureCollection, a single Feature or a bare geometry. */
export function parseCatalogFeatureCollection(text: string): CatalogFeatureCollection {
  const value = JSON.parse(text) as {type?: string; features?: CatalogFeature[]};
  if (value.type === 'FeatureCollection' && Array.isArray(value.features))
    return {type: 'FeatureCollection', features: value.features};
  if (value.type === 'Feature') return {type: 'FeatureCollection', features: [value as never]};
  // Invalid GeoJSON root; expected FeatureCollection, Feature or geometry.
  if (typeof value.type !== 'string') throw new Error('Invalid GeoJSON');
  return {
    type: 'FeatureCollection',
    features: [{geometry: value as CatalogGeometry, properties: {}}]
  };
}

export function getFeatureIdentity(feature: CatalogFeature): string | null {
  return feature.id === undefined || feature.id === null ? null : String(feature.id);
}

function isPosition(value: unknown): value is CatalogPosition {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === 'number' &&
    typeof value[1] === 'number'
  );
}

export function getLineParts(geometry: CatalogGeometry): CatalogPosition[][] {
  if (!geometry) return [];
  if (geometry.type === 'LineString')
    return [(geometry.coordinates as unknown[]).filter(isPosition)];
  if (geometry.type === 'MultiLineString')
    return (geometry.coordinates as unknown[][]).map(part => part.filter(isPosition));
  return [];
}

/** Polygon parts, each an array of rings (shell first). */
export function getPolygonParts(geometry: CatalogGeometry): CatalogPosition[][][] {
  if (!geometry) return [];
  if (geometry.type === 'Polygon')
    return [(geometry.coordinates as unknown[][]).map(ring => ring.filter(isPosition))];
  if (geometry.type === 'MultiPolygon')
    return (geometry.coordinates as unknown[][][]).map(polygon =>
      polygon.map(ring => ring.filter(isPosition))
    );
  return [];
}

export function getPointPositions(geometry: CatalogGeometry): CatalogPosition[] {
  if (!geometry) return [];
  if (geometry.type === 'Point')
    return isPosition(geometry.coordinates) ? [geometry.coordinates] : [];
  if (geometry.type === 'MultiPoint') return (geometry.coordinates as unknown[]).filter(isPosition);
  return [];
}

export function countFeatureVertices(feature: CatalogFeature): number {
  const geometry = feature.geometry;
  if (!geometry) return 0;
  const lines = getLineParts(geometry).reduce((count, part) => count + part.length, 0);
  const polygons = getPolygonParts(geometry).reduce(
    (count, polygon) => count + polygon.reduce((sum, ring) => sum + ring.length, 0),
    0
  );
  return lines + polygons + getPointPositions(geometry).length;
}

export function countCollectionVertices(collection: CatalogFeatureCollection): number {
  return collection.features.reduce((count, feature) => count + countFeatureVertices(feature), 0);
}

/** Haversine for EPSG:4326 (meters); Euclidean coordinate units for planar. */
export function measureDistance(
  first: CatalogPosition,
  second: CatalogPosition,
  coordinateSystem: CatalogCoordinateSystem
): number {
  if (coordinateSystem === 'planar') return Math.hypot(second[0] - first[0], second[1] - first[1]);
  return getGreatCircleDistanceMeters(first[0], first[1], second[0], second[1]);
}

export function measurePathLength(
  coordinates: readonly CatalogPosition[],
  coordinateSystem: CatalogCoordinateSystem
): number {
  let length = 0;
  for (let index = 1; index < coordinates.length; index++)
    length += measureDistance(coordinates[index - 1], coordinates[index], coordinateSystem);
  return length;
}

/** Signed area; planar shoelace or spherical (Chamberlain-Duquette) in square meters. */
export function measureRingSignedArea(
  ring: readonly CatalogPosition[],
  coordinateSystem: CatalogCoordinateSystem
): number {
  let sum = 0;
  for (let index = 0; index < ring.length - 1; index++) {
    const [x1, y1] = ring[index];
    const [x2, y2] = ring[index + 1];
    if (coordinateSystem === 'planar') sum += x1 * y2 - x2 * y1;
    else {
      sum += toRadians(x2 - x1) * (2 + Math.sin(toRadians(y1)) + Math.sin(toRadians(y2)));
    }
  }
  return coordinateSystem === 'planar' ? sum / 2 : (sum * EARTH_RADIUS_METERS ** 2) / 2;
}

export function makeCatalogGeometryText(type: string, coordinates: unknown): string {
  return JSON.stringify({type, coordinates});
}

/** Deterministic 32-bit FNV-1a content hash used to bind serialized invocations to inputs. */
export function hashCatalogText(text: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

export function addCatalogDiagnostic(
  diagnostics: CatalogDiagnostic[],
  code: string,
  severity: CatalogDiagnostic['severity'],
  message: string,
  count = 1
): void {
  const existing = diagnostics.find(diagnostic => diagnostic.code === code);
  if (existing) existing.count += count;
  else diagnostics.push({code, severity, count, message});
}

/** Builds an output table and infers nothing: callers declare every column type. */
export function makeCatalogTable(
  id: string,
  label: string,
  columnTypes: Record<string, CatalogColumnType>,
  geometryColumn?: string
): CatalogOutputTable & {columns: Record<string, (string | number | boolean | null)[]>} {
  const columns: Record<string, (string | number | boolean | null)[]> = {};
  for (const name of Object.keys(columnTypes)) columns[name] = [];
  return {id, label, rowCount: 0, columns, columnTypes, geometryColumn};
}

export function appendCatalogRow(
  table: CatalogOutputTable,
  row: Record<string, string | number | boolean | null>
): void {
  for (const name of Object.keys(table.columnTypes))
    (table.columns[name] as (string | number | boolean | null)[]).push(row[name] ?? null);
  table.rowCount++;
}
