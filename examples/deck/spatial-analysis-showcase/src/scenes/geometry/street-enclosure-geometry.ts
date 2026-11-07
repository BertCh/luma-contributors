// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

export type StreetEnclosureGeometry = {
  buildingSegments: Float32Array;
  analysisSegments: Float32Array;
  analysisBuildingIds: Uint32Array;
  analysisBuildingHeights: Float32Array;
  buildingCount: number;
  isInsideFootprint: (x: number, y: number) => boolean;
};

/** Project actual Chicago footprint rings to the scene's local metre frame. */
export function buildStreetEnclosureGeometry(
  dataset: LoadedDataset,
  origin: readonly [number, number],
  maximumAnalysisSegments = 20000
): StreetEnclosureGeometry {
  const collection = dataset.geojson as {
    features?: Array<{geometry?: {type?: string; coordinates?: unknown}}>;
  } | null;
  const project = dataset.getProjection(origin);
  const allSegments: number[] = [];
  const localPolygons: number[][][][] = [];
  const allOwners: number[] = [];
  const allBuildingHeights: number[] = [];
  let buildingCount = 0;

  for (const feature of collection?.features ?? []) {
    const geometry = feature.geometry;
    if (!geometry?.coordinates) continue;
    const polygons =
      geometry.type === 'Polygon'
        ? [geometry.coordinates as number[][][]]
        : geometry.type === 'MultiPolygon'
          ? (geometry.coordinates as number[][][][])
          : [];
    const featureSegments: number[] = [];
    const projectedPolygons: number[][][][] = [];
    for (const polygon of polygons) {
      const projectedPolygon: number[][][] = [];
      for (const ring of polygon) {
        const points = ring.map(([longitude, latitude]) => project.project(longitude, latitude));
        projectedPolygon.push(points);
        for (let index = 0; index + 1 < points.length; index++) {
          const first = points[index];
          const second = points[index + 1];
          featureSegments.push(first[0], first[1], second[0], second[1]);
        }
      }
      projectedPolygons.push(projectedPolygon);
    }
    if (!featureSegments.length) continue;
    buildingCount++;
    allSegments.push(...featureSegments);
    const buildingId = buildingCount - 1;
    allOwners.push(...Array.from({length: featureSegments.length / 4}, () => buildingId));
    // The source does not provide complete measured heights. Neutral placeholder values satisfy
    // the profile recipe's input contract; height outputs are intentionally not used.
    allBuildingHeights.push(0);
    localPolygons.push(...projectedPolygons);
  }

  const analysisSegments = allSegments.slice(0, maximumAnalysisSegments * 4);
  if (analysisSegments.length / 4 !== allOwners.length) {
    throw new Error('Chicago building sample exceeds the street-enclosure segment capacity');
  }
  return {
    buildingSegments: Float32Array.from(allSegments),
    analysisSegments: Float32Array.from(analysisSegments),
    analysisBuildingIds: Uint32Array.from(allOwners),
    analysisBuildingHeights: Float32Array.from(allBuildingHeights),
    buildingCount,
    isInsideFootprint(x: number, y: number) {
      const containsInRing = (ring: number[][]) => {
        let inside = false;
        for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
          const [firstX, firstY] = ring[index];
          const [secondX, secondY] = ring[previous];
          if (
            firstY > y !== secondY > y &&
            x < ((secondX - firstX) * (y - firstY)) / (secondY - firstY) + firstX
          )
            inside = !inside;
        }
        return inside;
      };
      const liesOnRing = (ring: number[][]) => {
        for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
          const first = ring[index];
          const second = ring[previous];
          const deltaX = second[0] - first[0];
          const deltaY = second[1] - first[1];
          const lengthSquared = deltaX * deltaX + deltaY * deltaY;
          const fraction =
            lengthSquared > 0
              ? Math.max(
                  0,
                  Math.min(1, ((x - first[0]) * deltaX + (y - first[1]) * deltaY) / lengthSquared)
                )
              : 0;
          const distanceX = x - (first[0] + fraction * deltaX);
          const distanceY = y - (first[1] + fraction * deltaY);
          if (distanceX * distanceX + distanceY * distanceY < 0.0001) return true;
        }
        return false;
      };
      return localPolygons.some(
        polygon =>
          polygon.some(liesOnRing) ||
          (containsInRing(polygon[0]) && !polygon.slice(1).some(containsInRing))
      );
    }
  };
}
