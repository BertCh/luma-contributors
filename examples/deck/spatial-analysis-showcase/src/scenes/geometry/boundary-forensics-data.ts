// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

export const BOUNDARY_FORENSICS_BUILDING_IDS = [
  814558, 357025, 814529, 355637, 814582, 357048, 894280, 355623, 824754, 814527, 356208, 357008
] as const;

export type BoundaryForensicsGeometry = {
  positions: Float32Array;
  reencodedPositions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  sourceIds: Uint32Array;
  sourcePaths: [number, number][][];
  reencodedPaths: [number, number][][];
  sourceMarkers: [number, number][];
  reencodedMarkers: [number, number][];
  featureCount: number;
  ringCount: number;
  vertexCount: number;
};

/** Project a deterministic handful of actual source building rings into one local metre frame. */
export function buildBoundaryForensicsGeometry(
  dataset: LoadedDataset,
  origin: readonly [number, number]
): BoundaryForensicsGeometry {
  const collection = dataset.geojson as {
    features?: Array<{
      geometry?: {type?: string; coordinates?: unknown};
      properties?: Record<string, unknown>;
    }>;
  } | null;
  const byId = new Map(
    (collection?.features ?? []).map(feature => [Number(feature.properties?.bldg_id), feature])
  );
  const projection = dataset.getProjection(origin);
  const positions: number[] = [];
  const reencodedPositions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const sourceIds: number[] = [];
  const sourcePaths: [number, number][][] = [];
  const reencodedPaths: [number, number][][] = [];
  const sourceMarkers: [number, number][] = [];
  const reencodedMarkers: [number, number][] = [];

  for (const buildingId of BOUNDARY_FORENSICS_BUILDING_IDS) {
    const feature = byId.get(buildingId);
    const geometry = feature?.geometry;
    if (!geometry?.coordinates) continue;
    const polygons =
      geometry.type === 'Polygon'
        ? [geometry.coordinates as number[][][]]
        : geometry.type === 'MultiPolygon'
          ? (geometry.coordinates as number[][][][])
          : [];
    let featurePolygonCount = 0;
    for (const polygon of polygons) {
      let hasShell = false;
      for (const ring of polygon) {
        if (ring.length < 4) continue;
        const projected = ring.map(
          ([longitude, latitude]) => projection.project(longitude, latitude) as [number, number]
        );
        const closed =
          projected.length > 1 &&
          projected[0][0] === projected[projected.length - 1][0] &&
          projected[0][1] === projected[projected.length - 1][1];
        const unique = closed ? projected.slice(0, -1) : projected;
        if (unique.length < 3) continue;
        const shift = unique.length > 2 ? 1 : 0;
        const reencoded = [...unique.slice(shift).reverse(), ...unique.slice(0, shift).reverse()];
        if (closed) reencoded.push(reencoded[0]);
        if (!hasShell) {
          hasShell = true;
          featurePolygonCount++;
        }
        const lngLatPath = ring.map(
          ([longitude, latitude]) => [longitude, latitude] as [number, number]
        );
        sourcePaths.push(lngLatPath);
        reencodedPaths.push(
          reencoded.map(point => projection.unproject(point[0], point[1]) as [number, number])
        );
        sourceMarkers.push(lngLatPath[0]);
        reencodedMarkers.push(reencodedPaths.at(-1)![0]);
        for (const point of projected) positions.push(point[0], point[1]);
        for (const point of reencoded) reencodedPositions.push(point[0], point[1]);
        ringOffsets.push(positions.length / 2);
      }
      if (hasShell) polygonOffsets.push(ringOffsets.length - 1);
    }
    if (featurePolygonCount) {
      sourceIds.push(buildingId);
      featureOffsets.push(featureOffsets.at(-1)! + featurePolygonCount);
    }
  }

  return {
    positions: Float32Array.from(positions),
    reencodedPositions: Float32Array.from(reencodedPositions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    featureOffsets: Uint32Array.from(featureOffsets),
    sourceIds: Uint32Array.from(sourceIds),
    sourcePaths,
    reencodedPaths,
    sourceMarkers,
    reencodedMarkers,
    featureCount: sourceIds.length,
    ringCount: ringOffsets.length - 1,
    vertexCount: positions.length / 2
  };
}
