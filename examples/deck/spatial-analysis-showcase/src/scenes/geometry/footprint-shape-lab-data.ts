// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {LocalMetricProjection} from '../../engine/projection';

export type FootprintKey = 'courtyard' | 'convex';
export type Footprint = {
  key: FootprintKey;
  label: string;
  sourceId: number;
  longitude: number;
  latitude: number;
  projection: LocalMetricProjection;
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  ringPaths: [number, number][][];
  vertexCount: number;
  uniqueVertexCount: number;
  ringCount: number;
  polygonCount: number;
};

/** Extract the two archived examples without changing stored closure or ring direction. */
export function loadFootprints(dataset: LoadedDataset): Record<FootprintKey, Footprint> {
  const collection = dataset.geojson as {
    features?: Array<{
      geometry?: {type?: string; coordinates?: unknown};
      properties?: Record<string, unknown>;
    }>;
  } | null;
  const features = collection?.features ?? [];
  const specs: [FootprintKey, number, string][] = [
    ['courtyard', 354712, 'Courtyard · 2 rings'],
    ['convex', 814558, 'Convex shell · 4 corners']
  ];
  const result = {} as Record<FootprintKey, Footprint>;
  for (const [key, sourceId, label] of specs) {
    const feature = features.find(item => Number(item.properties?.bldg_id) === sourceId);
    if (!feature?.geometry?.coordinates)
      throw new Error(`Archived building ${sourceId} is missing`);
    const geometry = feature.geometry;
    const polygons =
      geometry.type === 'Polygon'
        ? [geometry.coordinates as number[][][]]
        : geometry.type === 'MultiPolygon'
          ? (geometry.coordinates as number[][][][])
          : [];
    const shell = polygons[0]?.[0] ?? [];
    const origin = getRingCentroid(shell);
    const projection = new LocalMetricProjection(origin);
    const positions: number[] = [];
    const ringOffsets = [0];
    const polygonOffsets = [0];
    const featureOffsets = [0];
    const ringPaths: [number, number][][] = [];
    let uniqueVertexCount = 0;
    for (const polygon of polygons) {
      let polygonRingCount = 0;
      for (const ring of polygon) {
        if (ring.length < 4) continue;
        const path: [number, number][] = [];
        for (const coordinate of ring) {
          const point = projection.project(coordinate[0], coordinate[1]);
          positions.push(point[0], point[1]);
          path.push([coordinate[0], coordinate[1]]);
        }
        uniqueVertexCount += path.length - 1;
        ringPaths.push(path);
        ringOffsets.push(positions.length / 2);
        polygonRingCount++;
      }
      if (polygonRingCount) polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
    result[key] = {
      key,
      label,
      sourceId,
      longitude: origin[0],
      latitude: origin[1],
      projection,
      positions: Float32Array.from(positions),
      ringOffsets: Uint32Array.from(ringOffsets),
      polygonOffsets: Uint32Array.from(polygonOffsets),
      featureOffsets: Uint32Array.from(featureOffsets),
      ringPaths,
      vertexCount: positions.length / 2,
      uniqueVertexCount,
      ringCount: ringOffsets.length - 1,
      polygonCount: polygonOffsets.length - 1
    };
  }
  return result;
}

function getRingCentroid(ring: number[][]): [number, number] {
  const uniqueCoordinates = ring.length > 1 ? ring.slice(0, -1) : ring;
  let twiceArea = 0;
  let longitudeSum = 0;
  let latitudeSum = 0;
  for (let index = 0; index < uniqueCoordinates.length; index++) {
    const coordinate = uniqueCoordinates[index];
    const nextCoordinate = uniqueCoordinates[(index + 1) % uniqueCoordinates.length];
    const cross = coordinate[0] * nextCoordinate[1] - nextCoordinate[0] * coordinate[1];
    twiceArea += cross;
    longitudeSum += (coordinate[0] + nextCoordinate[0]) * cross;
    latitudeSum += (coordinate[1] + nextCoordinate[1]) * cross;
  }
  if (twiceArea === 0) {
    const coordinateCount = uniqueCoordinates.length;
    return [
      uniqueCoordinates.reduce((sum, coordinate) => sum + coordinate[0], 0) / coordinateCount,
      uniqueCoordinates.reduce((sum, coordinate) => sum + coordinate[1], 0) / coordinateCount
    ];
  }
  return [longitudeSum / (3 * twiceArea), latitudeSum / (3 * twiceArea)];
}
