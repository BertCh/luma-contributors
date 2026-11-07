// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewerFilterSpec} from './standalone-viewer-config';

/** Polygon filters (Kepler drawn features) evaluated on CPU positions. */

type Ring = readonly (readonly number[])[];

/** The fields of a GeoJSON Feature or geometry that the viewer reads. */
export type GeoJsonLike = {
  type?: string;
  geometry?: GeoJsonLike;
  geometries?: GeoJsonLike[];
  coordinates?: any;
  properties?: Record<string, unknown>;
};

/** Returns the geometry of a GeoJSON Feature, or the value itself for a bare geometry. */
export function readGeoJsonGeometry(value: unknown): GeoJsonLike | null {
  if (!value || typeof value !== 'object') return null;
  const geoJson: GeoJsonLike = value;
  return geoJson.type === 'Feature' ? (geoJson.geometry ?? null) : geoJson;
}

function isInRing(longitude: number, latitude: number, ring: Ring): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [x1, y1] = ring[index];
    const [x2, y2] = ring[previous];
    if (
      y1 > latitude !== y2 > latitude &&
      longitude < ((x2 - x1) * (latitude - y1)) / (y2 - y1) + x1
    )
      inside = !inside;
  }
  return inside;
}

function isInPolygonCoordinates(
  longitude: number,
  latitude: number,
  rings: readonly Ring[]
): boolean {
  if (!rings.length || !isInRing(longitude, latitude, rings[0])) return false;
  for (let hole = 1; hole < rings.length; hole++)
    if (isInRing(longitude, latitude, rings[hole])) return false;
  return true;
}

/** Point test against a GeoJSON Feature or geometry of type Polygon or MultiPolygon. */
export function isPointInFilterGeometry(
  longitude: number,
  latitude: number,
  value: unknown
): boolean {
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return false;
  const geometry = readGeoJsonGeometry(value);
  if (geometry?.type === 'Polygon')
    return isInPolygonCoordinates(longitude, latitude, geometry.coordinates);
  if (geometry?.type === 'MultiPolygon')
    return (geometry.coordinates as Ring[][]).some(polygon =>
      isInPolygonCoordinates(longitude, latitude, polygon)
    );
  return false;
}

export function getPolygonFiltersForLayer(
  filters: readonly ViewerFilterSpec[],
  layerId: string,
  datasetId: string
): ViewerFilterSpec[] {
  return filters.filter(
    filter =>
      filter.type === 'polygon' &&
      filter.enabled &&
      filter.layerIds.includes(layerId) &&
      (!filter.datasetIds.length || filter.datasetIds.includes(datasetId)) &&
      Boolean(filter.value)
  );
}

/**
 * Keeps rows whose representative points are all inside every applicable polygon filter. Native
 * uses one point for most layers, both ends for arc and line, and the centroid for cells/aggregations.
 */
export function applyPolygonFilters(
  rows: Uint32Array,
  polygonFilters: readonly ViewerFilterSpec[],
  getPoints: (row: number) => readonly (readonly [number, number])[]
): Uint32Array {
  if (!polygonFilters.length) return rows;
  const kept: number[] = [];
  for (const row of rows) {
    const points = getPoints(row);
    if (
      points.length &&
      polygonFilters.every(filter =>
        points.every(([longitude, latitude]) =>
          isPointInFilterGeometry(longitude, latitude, filter.value)
        )
      )
    )
      kept.push(row);
  }
  return Uint32Array.from(kept);
}
