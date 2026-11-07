// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded polygon inputs in the GeoArrow four-level layout the join and cover operations take
// (positions, featureOffsets, polygonOffsets, ringOffsets), one shell ring per feature.

export type PackedPolygons = {
  featureCount: number;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Closed ring boundaries as (x0, y0, x1, y1) segments, for drawing the input. */
  outline: Float32Array;
  outlineCount: number;
  /** Ring centers, one (x, y) per feature. */
  centers: Float32Array;
};

/** A star-shaped ring around `center`; `roughness` 0 gives a regular polygon. */
export function createStarRing(
  random: () => number,
  center: readonly [number, number],
  radius: number,
  vertexCount: number,
  roughness = 0.5
): number[][] {
  const ring: number[][] = [];
  for (let index = 0; index < vertexCount; index++) {
    const angle = (2 * Math.PI * index) / vertexCount;
    const length = radius * (1 - roughness * random());
    ring.push([
      Math.fround(center[0] + length * Math.cos(angle)),
      Math.fround(center[1] + length * Math.sin(angle))
    ]);
  }
  return ring;
}

/** Packs one shell ring per feature. */
export function packPolygonRings(rings: readonly number[][][]): PackedPolygons {
  const positions: number[] = [];
  const outline: number[] = [];
  const centers: number[] = [];
  const featureOffsets = [0];
  const ringOffsets = [0];
  for (const ring of rings) {
    let sumX = 0;
    let sumY = 0;
    ring.forEach(([x, y], index) => {
      positions.push(x, y);
      const [nextX, nextY] = ring[(index + 1) % ring.length];
      outline.push(x, y, nextX, nextY);
      sumX += x;
      sumY += y;
    });
    centers.push(sumX / ring.length, sumY / ring.length);
    ringOffsets.push(positions.length / 2);
    featureOffsets.push(featureOffsets.length);
  }
  return {
    featureCount: rings.length,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(featureOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4,
    centers: Float32Array.from(centers)
  };
}

/** Unit-sphere (x, y, z) of a longitude/latitude in degrees. */
export function getUnitVector(longitude: number, latitude: number): [number, number, number] {
  const longitudeRadians = (longitude * Math.PI) / 180;
  const latitudeRadians = (latitude * Math.PI) / 180;
  return [
    Math.cos(latitudeRadians) * Math.cos(longitudeRadians),
    Math.cos(latitudeRadians) * Math.sin(longitudeRadians),
    Math.sin(latitudeRadians)
  ];
}

/** Closed circle outline of `segmentCount` segments (x0, y0, x1, y1) in degrees. */
export function createCircleOutline(
  center: readonly [number, number],
  radius: number,
  segmentCount = 64
): Float32Array {
  const segments = new Float32Array(segmentCount * 4);
  for (let index = 0; index < segmentCount; index++) {
    const first = (2 * Math.PI * index) / segmentCount;
    const second = (2 * Math.PI * (index + 1)) / segmentCount;
    segments.set(
      [
        center[0] + radius * Math.cos(first),
        center[1] + radius * Math.sin(first),
        center[0] + radius * Math.cos(second),
        center[1] + radius * Math.sin(second)
      ],
      index * 4
    );
  }
  return segments;
}

/** Flattens longitude/latitude rings (degrees) to unit-vector rows. */
export function packUnitVectors(rings: readonly number[][][]): Float32Array {
  return Float32Array.from(
    rings.flatMap(ring =>
      ring.flatMap(([longitude, latitude]) => getUnitVector(longitude, latitude))
    )
  );
}
