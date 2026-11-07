// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

export const BLOCK_RELATIONSHIP_BUILDINGS = [
  {id: 342227, row: 21, address: '16 W Erie St'},
  {id: 815550, row: 1118, address: '14 W Erie St'},
  {id: 815551, row: 1119, address: '22 W Erie St'}
] as const;
export const BLOCK_RELATIONSHIPS_ORIGIN: readonly [number, number] = [-87.62887, 41.89431];
export const BLOCK_RELATIONSHIPS_PROBES = 16;

export type BlockRelationshipsData = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  sourceIds: Uint32Array;
  lineBuildingIds: Uint32Array;
  sourcePaths: [number, number][][];
  probePositions: Float32Array;
  obstacleSegments: Float32Array;
  enclosure: Float32Array;
  extent: [number, number, number, number];
  projection: ReturnType<LoadedDataset['getProjection']>;
  addresses: string[];
  vertexCounts: number[];
};

/** Load the three published source features and make only explicitly synthetic probe geometry. */
export function buildBlockRelationshipsData(dataset: LoadedDataset): BlockRelationshipsData {
  const featureCollection = dataset.geojson as {
    features?: Array<{
      geometry?: {type?: string; coordinates?: unknown};
      properties?: Record<string, unknown>;
    }>;
  } | null;
  const featuresById = new Map(
    (featureCollection?.features ?? []).map(feature => [
      Number(feature.properties?.bldg_id),
      feature
    ])
  );
  const projection = dataset.getProjection(BLOCK_RELATIONSHIPS_ORIGIN);
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const sourceIds: number[] = [];
  const sourcePaths: [number, number][][] = [];
  const vertexCounts: number[] = [];
  const rings: {positions: [number, number][]; buildingIndex: number}[] = [];

  for (
    let buildingIndex = 0;
    buildingIndex < BLOCK_RELATIONSHIP_BUILDINGS.length;
    buildingIndex++
  ) {
    const building = BLOCK_RELATIONSHIP_BUILDINGS[buildingIndex];
    const geometry = featuresById.get(building.id)?.geometry;
    if (!geometry?.coordinates) throw new Error(`Missing archived building ${building.id}`);
    const polygons =
      geometry.type === 'Polygon'
        ? [geometry.coordinates as number[][][]]
        : geometry.type === 'MultiPolygon'
          ? (geometry.coordinates as number[][][][])
          : [];
    let polygonCount = 0;
    let featureVertexCount = 0;
    for (const polygon of polygons) {
      let hasValidRing = false;
      for (const ring of polygon) {
        if (ring.length < 4) continue;
        const path = ring.map(([longitude, latitude]) => projection.project(longitude, latitude));
        const first = path[0];
        const last = path.at(-1)!;
        if (first[0] !== last[0] || first[1] !== last[1]) path.push(first);
        if (path.length < 4) continue;
        hasValidRing = true;
        rings.push({positions: path, buildingIndex});
        sourcePaths.push(
          ring.map(([longitude, latitude]) => [longitude, latitude] as [number, number])
        );
        for (const [positionX, positionY] of path) positions.push(positionX, positionY);
        ringOffsets.push(positions.length / 2);
        featureVertexCount += path.length;
      }
      if (hasValidRing) polygonCount++;
      polygonOffsets.push(ringOffsets.length - 1);
    }
    if (polygonCount === 0)
      throw new Error(`Archived building ${building.id} has no polygon rings`);
    sourceIds.push(building.id);
    featureOffsets.push(featureOffsets.at(-1)! + polygonCount);
    vertexCounts.push(featureVertexCount);
  }

  const obstacleSegments: number[] = [];
  let minimumX = Number.POSITIVE_INFINITY;
  let minimumY = Number.POSITIVE_INFINITY;
  let maximumX = Number.NEGATIVE_INFINITY;
  let maximumY = Number.NEGATIVE_INFINITY;
  for (const ring of rings) {
    const uniqueVertices = ring.positions.length - 1;
    const signedArea = getSignedRingArea(ring.positions);
    const forward = signedArea > 0;
    for (let vertex = 0; vertex < uniqueVertices; vertex++) {
      const start = ring.positions[vertex];
      const end = ring.positions[vertex + 1];
      const obstacleStart = forward ? start : end;
      const obstacleEnd = forward ? end : start;
      obstacleSegments.push(obstacleStart[0], obstacleStart[1], obstacleEnd[0], obstacleEnd[1]);
      minimumX = Math.min(minimumX, start[0]);
      minimumY = Math.min(minimumY, start[1]);
      maximumX = Math.max(maximumX, start[0]);
      maximumY = Math.max(maximumY, start[1]);
    }
  }
  const margin = 18;
  const enclosure = Float32Array.of(
    minimumX - margin,
    minimumY - margin,
    maximumX + margin,
    minimumY - margin,
    maximumX + margin,
    maximumY + margin,
    minimumX - margin,
    maximumY + margin
  );
  const probePositions: number[] = [];
  const probeMargin = 4;
  const probeLeft = minimumX - probeMargin;
  const probeRight = maximumX + probeMargin;
  const probeBottom = minimumY - probeMargin;
  const probeTop = maximumY + probeMargin;
  const horizontalCount = 4;
  const verticalCount = 4;
  for (let index = 0; index < horizontalCount; index++) {
    const fraction = (index + 0.5) / horizontalCount;
    const coordinateX = probeLeft + fraction * (probeRight - probeLeft);
    probePositions.push(coordinateX, probeBottom, coordinateX, probeTop);
  }
  for (let index = 0; index < verticalCount; index++) {
    const fraction = (index + 0.5) / verticalCount;
    const coordinateY = probeBottom + fraction * (probeTop - probeBottom);
    probePositions.push(probeLeft, coordinateY, probeRight, coordinateY);
  }

  return {
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    featureOffsets: Uint32Array.from(featureOffsets),
    sourceIds: Uint32Array.from(sourceIds),
    lineBuildingIds: Uint32Array.from(rings.map(ring => ring.buildingIndex)),
    sourcePaths,
    probePositions: Float32Array.from(probePositions),
    obstacleSegments: Float32Array.from(obstacleSegments),
    enclosure,
    extent: [minimumX, minimumY, maximumX, maximumY],
    projection,
    addresses: BLOCK_RELATIONSHIP_BUILDINGS.map(building => building.address),
    vertexCounts
  };
}

function getSignedRingArea(path: [number, number][]): number {
  let twiceArea = 0;
  for (let vertex = 0; vertex + 1 < path.length; vertex++) {
    twiceArea += path[vertex][0] * path[vertex + 1][1] - path[vertex + 1][0] * path[vertex][1];
  }
  return twiceArea / 2;
}
