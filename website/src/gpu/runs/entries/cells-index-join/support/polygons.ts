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
  const radians = Math.PI / 180;
  const latitude = center[1] * radians;
  const angle = radius * radians;
  function destination(bearing: number): [number, number] {
    const north = Math.asin(
      Math.sin(latitude) * Math.cos(angle) +
        Math.cos(latitude) * Math.sin(angle) * Math.cos(bearing)
    );
    const east = Math.atan2(
      Math.sin(bearing) * Math.sin(angle) * Math.cos(latitude),
      Math.cos(angle) - Math.sin(latitude) * Math.sin(north)
    );
    return [center[0] + east / radians, north / radians];
  }
  const segments = new Float32Array(segmentCount * 4);
  let start = destination(0);
  for (let index = 0; index < segmentCount; index++) {
    const end = destination((2 * Math.PI * (index + 1)) / segmentCount);
    segments.set([...start, ...end], index * 4);
    start = end;
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
