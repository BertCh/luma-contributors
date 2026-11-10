// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Three rectangular zones as a four-level polygon layout, resizable and movable per frame.

import type {FleetInput} from './fleet';

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

export type ZoneCenter = readonly [number, number];

/** Zone centres snap to this world step, so a pointer at rest reproduces the default layout exactly. */
export const ZONE_SNAP = 0.02;

function snapToZoneGrid(value: number): number {
  return Math.round(value / ZONE_SNAP) * ZONE_SNAP;
}

/**
 * Three zone centres on the traffic: a deterministic three-means of the slow fixes of the fleet
 * (where vessels linger), so every zone receives tracks and stops.
 */
export function chooseZoneCenters(fleet: FleetInput): ZoneCenter[] {
  const slowFixes: [number, number][] = [];
  for (let row = 0; row < fleet.rowCount - 1; row++) {
    const isSameTrack = !fleet.trackOffsets.includes(row + 1);
    if (!isSameTrack) continue;
    const step = Math.hypot(
      fleet.positions[row * 2 + 2] - fleet.positions[row * 2],
      fleet.positions[row * 2 + 3] - fleet.positions[row * 2 + 1]
    );
    const duration = fleet.timestamps[row + 1] - fleet.timestamps[row];
    if (duration > 0 && step / duration < 0.001)
      slowFixes.push([fleet.positions[row * 2], fleet.positions[row * 2 + 1]]);
  }
  const fixes = slowFixes;
  let centers: [number, number][] = [];
  {
    // Farthest-point seeding keeps the three centres apart.
    centers = [fixes[0]];
    while (centers.length < ZONE_COUNT) {
      let best = fixes[0];
      let bestDistance = -1;
      for (const fix of fixes) {
        const distance = Math.min(
          ...centers.map(center => Math.hypot(fix[0] - center[0], fix[1] - center[1]))
        );
        if (distance > bestDistance) {
          bestDistance = distance;
          best = fix;
        }
      }
      centers.push(best);
    }
    for (let iteration = 0; iteration < 8; iteration++) {
      const sums = centers.map(() => [0, 0, 0]);
      for (const fix of fixes) {
        let nearest = 0;
        for (let index = 1; index < centers.length; index++) {
          if (
            Math.hypot(fix[0] - centers[index][0], fix[1] - centers[index][1]) <
            Math.hypot(fix[0] - centers[nearest][0], fix[1] - centers[nearest][1])
          ) {
            nearest = index;
          }
        }
        sums[nearest][0] += fix[0];
        sums[nearest][1] += fix[1];
        sums[nearest][2]++;
      }
      centers = centers.map((center, index) =>
        sums[index][2] ? [sums[index][0] / sums[index][2], sums[index][1] / sums[index][2]] : center
      );
    }
  }
  return centers
    .sort((first, second) => first[0] - second[0])
    .map(([x, y]) => [snapToZoneGrid(x), snapToZoneGrid(y)] as const);
}

/**
 * The pointer carries the zone whose home centre is nearest; the other zones stay at home, so the
 * layout is a pure function of the pointer.
 */
export function dragNearestZone(
  homes: readonly ZoneCenter[],
  pointer: {x: number; y: number}
): {centers: ZoneCenter[]; zone: number} {
  let zone = 0;
  for (let index = 1; index < homes.length; index++) {
    if (
      Math.hypot(pointer.x - homes[index][0], pointer.y - homes[index][1]) <
      Math.hypot(pointer.x - homes[zone][0], pointer.y - homes[zone][1])
    ) {
      zone = index;
    }
  }
  const centers = homes.map((home, index) =>
    index === zone ? ([snapToZoneGrid(pointer.x), snapToZoneGrid(pointer.y)] as const) : home
  );
  return {centers, zone};
}

/** Half extents of one zone rectangle at `scale`. */
export function getZoneHalfSize(
  world: {width: number; height: number},
  scale: number
): [number, number] {
  return [0.2 * scale * Math.min(world.width * 0.6, 1.4), 0.2 * scale];
}

/** Zone rectangles around `centers`; `scale` grows or shrinks each around its center. */
export function createZoneRectangles(
  world: {width: number; height: number},
  scale: number,
  centers: readonly ZoneCenter[]
): ZoneRectangles {
  const positions: number[] = [];
  const outline: number[] = [];
  for (const [centerX, centerY] of centers) {
    const [halfWidth, halfHeight] = getZoneHalfSize(world, scale);
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
