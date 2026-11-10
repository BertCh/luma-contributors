// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import footprints from '../../../../data/amsterdam-buildings.json';

type Position = [number, number];
type World = {width: number; height: number};

export type CityLayout = {
  rings: Position[][];
  centers: Position[];
  heights: Float32Array;
  streetY: number;
};

function centerOf(ring: Position[]): Position {
  return [
    ring.reduce((sum, point) => sum + point[0], 0) / ring.length,
    ring.reduce((sum, point) => sum + point[1], 0) / ring.length
  ];
}

/** Fit whole Overtoom frontages in an oblique local frame, with the street running left to right. */
export function createCityLayout(world: World, count: number, separate = false): CityLayout {
  const norm = Math.hypot(1, 0.36);
  const buildings = footprints
    .map(building => {
      // Project RD metres onto the street axis and its normal. The transect lies at y = 0.
      const ring: Position[] = building.ring.map(([east, north]) => [
        (east + 0.36 * north) / norm,
        (north - 0.36 * east) / norm + 15
      ]);
      return {ring, center: centerOf(ring), height: building.height};
    })
    .filter(building => {
      const north = building.ring.map(point => point[1]);
      return Math.abs(building.center[1]) < 33 && Math.max(...north) - Math.min(...north) < 40;
    });
  // Include frontages on both sides of the street, starting near the midpoint of the tile.
  buildings.sort(
    (first, second) => Math.abs(first.center[0] - 12) - Math.abs(second.center[0] - 12)
  );
  const selected: typeof buildings = [];
  const occupiedVertices = new Set<string>();
  for (const building of buildings) {
    // The tessellation operator requires unique boundary samples across owners.
    if (separate && building.ring.some(point => occupiedVertices.has(point.join(',')))) continue;
    selected.push(building);
    if (separate) {
      for (const point of building.ring) occupiedVertices.add(point.join(','));
    }
    if (selected.length === count) break;
  }
  selected.sort((first, second) => first.center[0] - second.center[0]);
  const vertices = selected.flatMap(building => building.ring);
  const west = Math.min(...vertices.map(vertex => vertex[0]));
  const east = Math.max(...vertices.map(vertex => vertex[0]));
  const south = Math.min(...vertices.map(vertex => vertex[1]));
  const north = Math.max(...vertices.map(vertex => vertex[1]));
  const scale = 0.88 * Math.min(world.width / (east - west), world.height / (north - south));
  const originX = (world.width - (west + east) * scale) / 2;
  const streetY = (world.height - (south + north) * scale) / 2;
  const rings = selected.map(building =>
    building.ring.map(([x, y]): Position => [originX + x * scale, streetY + y * scale])
  );
  return {
    rings,
    centers: rings.map(centerOf),
    heights: Float32Array.from(selected, building => building.height * scale),
    streetY
  };
}

/** Alter setbacks for the interactive scenarios. A scale of one keeps the observed outlines. */
export function getBuildings(
  layout: CityLayout,
  fill = 1,
  widening = 0,
  /** One building moved by `offset` (world units) while the pointer carries it. */
  carried?: {building: number; offset: Position}
) {
  const segments: number[] = [];
  const owners: number[] = [];
  layout.rings.forEach((ring, building) => {
    const [centerX, centerY] = layout.centers[building];
    const shift = (Math.sign(centerY - layout.streetY) * widening) / 2;
    const [carriedX, carriedY] = carried && carried.building === building ? carried.offset : [0, 0];
    const scaled = ring.map(([x, y]) => [
      centerX + (x - centerX) * fill + carriedX,
      centerY + (y - centerY) * fill + shift + carriedY
    ]);
    scaled.forEach((point, corner) => {
      segments.push(...point, ...scaled[(corner + 1) % ring.length]);
      owners.push(building);
    });
  });
  return {
    count: layout.rings.length,
    segments: Float32Array.from(segments),
    segmentBuildingIds: Uint32Array.from(owners)
  };
}

function containsPoint(ring: Position[], x: number, y: number): boolean {
  let inside = false;
  for (let corner = 0, last = ring.length - 1; corner < ring.length; last = corner++) {
    const [firstX, firstY] = ring[corner];
    const [lastX, lastY] = ring[last];
    if (
      firstY > y !== lastY > y &&
      x < ((lastX - firstX) * (y - firstY)) / (lastY - firstY) + firstX
    ) {
      inside = !inside;
    }
  }
  return inside;
}

/** The building whose footprint contains (x, y), or -1 on open ground. */
export function findBuildingAt(layout: CityLayout, x: number, y: number): number {
  return layout.rings.findIndex(ring => containsPoint(ring, x, y));
}

/** Uniform sampling of open space, including sidewalks, the street and rear courtyards. */
export function sampleOpenSpace(layout: CityLayout, world: World): Float32Array {
  const points: Position[] = [];
  for (let row = 1; row < 17; row++) {
    for (let column = 1; column < 28; column++) {
      const x = (world.width * column) / 28;
      const y = (world.height * row) / 17;
      if (!layout.rings.some(ring => containsPoint(ring, x, y))) points.push([x, y]);
    }
  }
  const count = Math.min(96, points.length);
  return Float32Array.from(
    Array.from(
      {length: count},
      (_, index) => points[Math.floor((index * points.length) / count)]
    ).flat()
  );
}

/** Positions along the centerline, inset from the edges of the measured frontages. */
export function sampleStreet(layout: CityLayout, count: number): Float32Array {
  const horizontal = layout.centers.map(point => point[0]);
  const start = Math.min(...horizontal);
  const end = Math.max(...horizontal);
  return Float32Array.from(
    Array.from({length: count}, (_, index) => [
      start + ((index + 0.5) / count) * (end - start),
      layout.streetY
    ]).flat()
  );
}
