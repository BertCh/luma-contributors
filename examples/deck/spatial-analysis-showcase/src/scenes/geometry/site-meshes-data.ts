// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

export const LIBRARY_CATEGORY = 1;
export const FIRE_STATION_CATEGORY = 3;
export const COURTYARD_BUILDING_ID = 354712;
export const SITE_MESH_ORIGIN: [number, number] = [-87.7, 41.86];

export type SiteMeshesData = {
  libraryPositions: Float32Array;
  libraryLngLat: [number, number][];
  libraryNames: string[];
  groupedPositions: Float32Array;
  groupIds: Uint32Array;
  groupedLngLat: [number, number][];
  clipBounds: Float32Array;
  clipPath: [number, number][];
  buildingPositions: Float32Array;
  buildingRingOffsets: Uint32Array;
  buildingPolygonOffsets: Uint32Array;
  buildingFeatureOffsets: Uint32Array;
  buildingPath: [number, number][][];
  buildingId: number;
  projection: ReturnType<LoadedDataset['getProjection']>;
};

/** Read all archived library points, all archived fire stations and the selected courtyard. */
export function buildSiteMeshesData(
  facilities: LoadedDataset,
  buildings: LoadedDataset,
  facilityNames: readonly Record<string, string>[]
): SiteMeshesData {
  const projection = facilities.getProjection(SITE_MESH_ORIGIN);
  const rawPositions = facilities.column<Float32Array>('position');
  const categories = facilities.column<Uint8Array>('category');
  const rowsByCategory = new Map<number, number[]>();
  for (let row = 0; row < facilities.count; row++) {
    const rows = rowsByCategory.get(categories[row]) ?? [];
    rows.push(row);
    rowsByCategory.set(categories[row], rows);
  }
  const libraryRows = rowsByCategory.get(LIBRARY_CATEGORY) ?? [];
  const fireStationRows = rowsByCategory.get(FIRE_STATION_CATEGORY) ?? [];
  const projectRows = (rows: readonly number[]) => {
    const positions = new Float32Array(rows.length * 2);
    const lngLat: [number, number][] = [];
    for (let index = 0; index < rows.length; index++) {
      const row = rows[index];
      const longitude = rawPositions[row * 2];
      const latitude = rawPositions[row * 2 + 1];
      const point = projection.project(longitude, latitude);
      positions[index * 2] = point[0];
      positions[index * 2 + 1] = point[1];
      lngLat.push([longitude, latitude]);
    }
    return {positions, lngLat};
  };
  const libraries = projectRows(libraryRows);
  const fireStations = projectRows(fireStationRows);
  const groupedPositions = new Float32Array(
    libraries.positions.length + fireStations.positions.length
  );
  groupedPositions.set(libraries.positions);
  groupedPositions.set(fireStations.positions, libraries.positions.length);
  const groupedLngLat = [...libraries.lngLat, ...fireStations.lngLat];
  const groupIds = new Uint32Array(groupedLngLat.length);
  groupIds.fill(0, 0, libraries.lngLat.length);
  groupIds.fill(1, libraries.lngLat.length);
  const paddingMeters = 800;
  const projectedExtents = libraries.positions;
  let minimumX = Number.POSITIVE_INFINITY;
  let minimumY = Number.POSITIVE_INFINITY;
  let maximumX = Number.NEGATIVE_INFINITY;
  let maximumY = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < projectedExtents.length; index += 2) {
    minimumX = Math.min(minimumX, projectedExtents[index]);
    minimumY = Math.min(minimumY, projectedExtents[index + 1]);
    maximumX = Math.max(maximumX, projectedExtents[index]);
    maximumY = Math.max(maximumY, projectedExtents[index + 1]);
  }
  minimumX -= paddingMeters;
  minimumY -= paddingMeters;
  maximumX += paddingMeters;
  maximumY += paddingMeters;
  const clipBounds = Float32Array.of(minimumX, minimumY, maximumX, maximumY);
  const clipPath: [number, number][] = [
    projection.unproject(minimumX, minimumY),
    projection.unproject(maximumX, minimumY),
    projection.unproject(maximumX, maximumY),
    projection.unproject(minimumX, maximumY),
    projection.unproject(minimumX, minimumY)
  ];
  const collection = buildings.geojson as {
    features?: Array<{
      geometry?: {type?: string; coordinates?: unknown};
      properties?: Record<string, unknown>;
    }>;
  } | null;
  const feature = (collection?.features ?? []).find(
    item => Number(item.properties?.bldg_id) === COURTYARD_BUILDING_ID
  );
  if (!feature?.geometry?.coordinates)
    throw new Error('Selected Chicago courtyard building is missing');
  const geometry = feature.geometry;
  const polygons =
    geometry.type === 'Polygon'
      ? [geometry.coordinates as number[][][]]
      : geometry.type === 'MultiPolygon'
        ? (geometry.coordinates as number[][][][])
        : [];
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const buildingPath: [number, number][][] = [];
  for (const polygon of polygons) {
    let hasShell = false;
    for (const ring of polygon) {
      if (ring.length < 4) continue;
      const path: [number, number][] = [];
      for (const [longitude, latitude] of ring) {
        const point = projection.project(longitude, latitude);
        positions.push(point[0], point[1]);
        path.push([longitude, latitude]);
      }
      buildingPath.push(path);
      ringOffsets.push(positions.length / 2);
      hasShell = true;
    }
    if (hasShell) polygonOffsets.push(ringOffsets.length - 1);
  }
  featureOffsets.push(polygonOffsets.length - 1);
  return {
    libraryPositions: libraries.positions,
    libraryLngLat: libraries.lngLat,
    libraryNames: libraryRows.map(row => facilityNames[row]?.name ?? `source row ${row}`),
    groupedPositions,
    groupIds,
    groupedLngLat,
    clipBounds,
    clipPath,
    buildingPositions: Float32Array.from(positions),
    buildingRingOffsets: Uint32Array.from(ringOffsets),
    buildingPolygonOffsets: Uint32Array.from(polygonOffsets),
    buildingFeatureOffsets: Uint32Array.from(featureOffsets),
    buildingPath,
    buildingId: COURTYARD_BUILDING_ID,
    projection
  };
}
