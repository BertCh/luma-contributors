// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded blob polygons for the raster zone runs: one wobbly polygon per grid slot of the world,
// in the four-level offset layout the analysis operations take (features, polygons, rings,
// vertices), plus a fan-triangulated mesh and outline segments for drawing.

import type {Random, World} from '../../../inputs';

export type BlobFeatures = {
  featureCount: number;
  /** Ring vertices (x, y); each ring is implicitly closed. */
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Fan-triangulated shells for drawing: ring vertices, then one center vertex per feature. */
  meshVertices: Float32Array;
  meshIndices: Uint32Array;
  /** Feature row per mesh triangle. */
  meshOwners: Uint32Array;
  meshTriangleCount: number;
  /** All ring edges as (x0, y0, x1, y1). */
  outline: Float32Array;
  outlineCount: number;
  /** Feature centers (x, y). */
  centers: Float32Array;
  /** Width and height of each feature's bounding box (w, h pairs). */
  extents: Float32Array;
  /** Rebuilds positions with another hole size (0 to 1 of the shell); needs `holes: true`. */
  getPositions(holeScale: number): Float32Array;
  /** Ring edges of any positions array from `getPositions`. */
  getOutline(positions: Float32Array): Float32Array;
};

export type BlobOptions = {
  /** Polygon slots per row; default from the aspect. */
  columns?: number;
  rows?: number;
  vertexCount?: number;
  /** Every other feature gets a hole ring. */
  holes?: boolean;
  /** Fraction of the slot a blob fills (radius over half the slot). */
  fill?: number;
};

export function createBlobFeatures(
  random: Random,
  world: World,
  options: BlobOptions = {}
): BlobFeatures {
  const rows = options.rows ?? 3;
  const columns = options.columns ?? Math.max(2, Math.round(rows * world.width));
  const vertexCount = options.vertexCount ?? 20;
  const fill = options.fill ?? 0.82;
  const slotWidth = world.width / columns;
  const slotHeight = world.height / rows;
  const featureCount = columns * rows;

  type Blob = {centerX: number; centerY: number; radius: number; phaseA: number; phaseB: number};
  const blobs: Blob[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      blobs.push({
        centerX: (column + 0.5 + (random() - 0.5) * 0.12) * slotWidth,
        centerY: (row + 0.5 + (random() - 0.5) * 0.12) * slotHeight,
        radius: 0.5 * Math.min(slotWidth, slotHeight) * fill * (0.8 + 0.2 * random()),
        phaseA: random() * Math.PI * 2,
        phaseB: random() * Math.PI * 2
      });
    }
  }
  const getRingPoint = (blob: Blob, index: number, scale: number, direction: number) => {
    const angle = (direction * index * Math.PI * 2) / vertexCount;
    const radius =
      blob.radius *
      scale *
      (1 + 0.16 * Math.sin(2 * angle + blob.phaseA) + 0.1 * Math.sin(3 * angle + blob.phaseB));
    return [blob.centerX + radius * Math.cos(angle), blob.centerY + radius * Math.sin(angle)];
  };

  const hasHole = (feature: number) => Boolean(options.holes) && feature % 2 === 0;
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  let vertexTotal = 0;
  for (let feature = 0; feature < featureCount; feature++) {
    const ringCount = hasHole(feature) ? 2 : 1;
    for (let ring = 0; ring < ringCount; ring++) {
      vertexTotal += vertexCount;
      ringOffsets.push(vertexTotal);
    }
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  }
  const getPositions = (holeScale: number) => {
    const positions = new Float32Array(vertexTotal * 2);
    let cursor = 0;
    blobs.forEach((blob, feature) => {
      for (let index = 0; index < vertexCount; index++) {
        positions.set(getRingPoint(blob, index, 1, 1), cursor++ * 2);
      }
      if (hasHole(feature)) {
        for (let index = 0; index < vertexCount; index++) {
          positions.set(getRingPoint(blob, index, Math.max(0.05, holeScale), -1), cursor++ * 2);
        }
      }
    });
    return positions;
  };
  const positions = getPositions(0.4);

  // Mesh and outline from the shells (and hole rings for the outline).
  const meshVertices = new Float32Array(featureCount * (vertexCount + 1) * 2);
  const meshIndices = new Uint32Array(featureCount * vertexCount * 3);
  const meshOwners = new Uint32Array(featureCount * vertexCount);
  blobs.forEach((blob, feature) => {
    const base = feature * (vertexCount + 1);
    for (let index = 0; index < vertexCount; index++) {
      meshVertices.set(getRingPoint(blob, index, 1, 1), (base + index) * 2);
    }
    meshVertices.set([blob.centerX, blob.centerY], (base + vertexCount) * 2);
    for (let index = 0; index < vertexCount; index++) {
      const triangle = feature * vertexCount + index;
      meshIndices.set(
        [base + index, base + ((index + 1) % vertexCount), base + vertexCount],
        triangle * 3
      );
      meshOwners[triangle] = feature;
    }
  });
  const getOutline = (ringPositions: Float32Array) => {
    const segments: number[] = [];
    for (let ring = 0; ring < ringOffsets.length - 1; ring++) {
      const start = ringOffsets[ring];
      const end = ringOffsets[ring + 1];
      for (let vertex = start; vertex < end; vertex++) {
        const next = vertex + 1 < end ? vertex + 1 : start;
        segments.push(
          ringPositions[vertex * 2],
          ringPositions[vertex * 2 + 1],
          ringPositions[next * 2],
          ringPositions[next * 2 + 1]
        );
      }
    }
    return Float32Array.from(segments);
  };
  const outline = getOutline(positions);
  const extents = new Float32Array(featureCount * 2);
  blobs.forEach((blob, feature) => {
    let minimumX = Infinity;
    let maximumX = -Infinity;
    let minimumY = Infinity;
    let maximumY = -Infinity;
    for (let index = 0; index < vertexCount; index++) {
      const [x, y] = getRingPoint(blob, index, 1, 1);
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
    }
    extents.set([maximumX - minimumX, maximumY - minimumY], feature * 2);
  });
  const centers = new Float32Array(featureCount * 2);
  blobs.forEach((blob, feature) => centers.set([blob.centerX, blob.centerY], feature * 2));

  return {
    featureCount,
    positions,
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    meshVertices,
    meshIndices,
    meshOwners,
    meshTriangleCount: featureCount * vertexCount,
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4,
    centers,
    extents,
    getPositions,
    getOutline
  };
}
