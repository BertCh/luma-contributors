// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Three rectangular zones as a four-level polygon layout, resizable per frame.

export const ZONE_COUNT = 3;

export type ZoneRectangles = {
  /** Four vertices per zone, one ring each. */
  positions: Float32Array;
  /** Outline segments (x0, y0, x1, y1), four per zone. */
  outline: Float32Array;
};

/** Zone offsets: feature, polygon and ring offsets are all one entry per zone. */
export const ZONE_FEATURE_OFFSETS = Uint32Array.from({length: ZONE_COUNT + 1}, (_, index) => index);
export const ZONE_RING_OFFSETS = Uint32Array.from(
  {length: ZONE_COUNT + 1},
  (_, index) => index * 4
);

/** Zones spread over the world; `scale` grows or shrinks each around its center. */
export function createZoneRectangles(
  world: {width: number; height: number},
  scale: number
): ZoneRectangles {
  const centers = [
    [0.2 * world.width, 0.66],
    [0.5 * world.width, 0.34],
    [0.8 * world.width, 0.62]
  ];
  const positions: number[] = [];
  const outline: number[] = [];
  for (const [centerX, centerY] of centers) {
    const halfWidth = 0.2 * scale * Math.min(world.width * 0.6, 1.4);
    const halfHeight = 0.2 * scale;
    const corners = [
      [centerX - halfWidth, centerY - halfHeight],
      [centerX + halfWidth, centerY - halfHeight],
      [centerX + halfWidth, centerY + halfHeight],
      [centerX - halfWidth, centerY + halfHeight]
    ];
    for (const [index, corner] of corners.entries()) {
      const next = corners[(index + 1) % 4];
      positions.push(corner[0], corner[1]);
      outline.push(corner[0], corner[1], next[0], next[1]);
    }
  }
  return {positions: Float32Array.from(positions), outline: Float32Array.from(outline)};
}

/** Two triangles per zone rectangle (vertex ids into `positions`) and the owner zone of each. */
export function createZoneTriangles(): {indices: Uint32Array; owners: Uint32Array} {
  const indices: number[] = [];
  const owners: number[] = [];
  for (let zone = 0; zone < ZONE_COUNT; zone++) {
    const base = zone * 4;
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    owners.push(zone, zone);
  }
  return {indices: Uint32Array.from(indices), owners: Uint32Array.from(owners)};
}

export const ZONE_TRIANGLE_COUNT = ZONE_COUNT * 2;
