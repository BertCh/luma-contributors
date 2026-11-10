// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU lookups on zone and tract meshes: the zone under the pointer, its ring and its neighbors.

import type {TractsInput} from '../../../inputs';
import type {RunGlyph} from '../../../interaction';
import {findPolygonAt} from '../../../pick';

export type ZoneGeometry = {
  count: number;
  /** Packed (x, y) vertices, one ring per zone. */
  positions: ArrayLike<number>;
  ringOffsets: Uint32Array;
  triangleIndices: Uint32Array;
  triangleOwners: Uint32Array;
  triangleCount: number;
  centers: Float32Array;
};

/** A tract mesh as a zone geometry. */
export function getTractGeometry(tracts: TractsInput): ZoneGeometry {
  return {
    count: tracts.count,
    positions: tracts.vertices,
    ringOffsets: tracts.ringOffsets,
    triangleIndices: tracts.triangleIndices,
    triangleOwners: tracts.triangleOwners,
    triangleCount: tracts.triangleCount,
    centers: tracts.centers
  };
}

/** The zone containing (x, y), or the zone with the nearest center when the point is in none. */
export function findZoneAt(geometry: ZoneGeometry, x: number, y: number): number {
  const zone = findPolygonAt(
    geometry.positions,
    geometry.triangleIndices,
    geometry.triangleOwners,
    x,
    y,
    geometry.triangleCount
  );
  if (zone >= 0) return zone;
  let nearest = 0;
  let nearestDistance = Infinity;
  for (let candidate = 0; candidate < geometry.count; candidate++) {
    const distance = Math.hypot(
      geometry.centers[candidate * 2] - x,
      geometry.centers[candidate * 2 + 1] - y
    );
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = candidate;
    }
  }
  return nearest;
}

/** The ring vertices of one zone, for a closed `polyline` glyph. */
export function getZoneRing(geometry: ZoneGeometry, zone: number): Float32Array {
  const start = geometry.ringOffsets[zone] * 2;
  const end = geometry.ringOffsets[zone + 1] * 2;
  return Float32Array.from({length: end - start}, (_, index) => geometry.positions[start + index]);
}

export type ZoneNeighbors = {offsets: Uint32Array; neighbors: Uint32Array};

/** Rook neighbors: zones that share a boundary edge with exactly the same end coordinates. */
export function getZoneNeighbors(geometry: ZoneGeometry): ZoneNeighbors {
  const owners = new Map<string, number[]>();
  for (let zone = 0; zone < geometry.count; zone++) {
    const start = geometry.ringOffsets[zone];
    const count = geometry.ringOffsets[zone + 1] - start;
    for (let vertex = 0; vertex < count; vertex++) {
      const first = (start + vertex) * 2;
      const second = (start + ((vertex + 1) % count)) * 2;
      const a = `${geometry.positions[first]},${geometry.positions[first + 1]}`;
      const b = `${geometry.positions[second]},${geometry.positions[second + 1]}`;
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      owners.set(key, [...(owners.get(key) ?? []), zone]);
    }
  }
  const lists = Array.from({length: geometry.count}, () => new Set<number>());
  for (const zones of owners.values()) {
    for (const first of zones)
      for (const second of zones) if (first !== second) lists[first].add(second);
  }
  const offsets = new Uint32Array(geometry.count + 1);
  const neighbors: number[] = [];
  for (let zone = 0; zone < geometry.count; zone++) {
    neighbors.push(...lists[zone]);
    offsets[zone + 1] = neighbors.length;
  }
  return {offsets, neighbors: Uint32Array.from(neighbors)};
}

/** The neighbors of one zone. */
export function getNeighborsOf(neighbors: ZoneNeighbors, zone: number): Uint32Array {
  return neighbors.neighbors.subarray(neighbors.offsets[zone], neighbors.offsets[zone + 1]);
}

/** Closed ring glyphs: the zone solid, its neighbors dashed (at most `limit` of them). */
export function getNeighborhoodGlyphs(
  geometry: ZoneGeometry,
  zone: number,
  neighbors: ArrayLike<number>,
  limit = 8
): RunGlyph[] {
  const glyphs: RunGlyph[] = [];
  for (let slot = 0; slot < Math.min(neighbors.length, limit); slot++) {
    glyphs.push({
      shape: 'polyline',
      points: getZoneRing(geometry, neighbors[slot]),
      closed: true,
      dashed: true
    });
  }
  glyphs.push({shape: 'polyline', points: getZoneRing(geometry, zone), closed: true});
  return glyphs;
}

/** Boundary segments (x0, y0, x1, y1) of the listed zones, for a lines layer. */
export function getZoneOutlineSegments(
  geometry: ZoneGeometry,
  zones: Iterable<number>
): Float32Array {
  const segments: number[] = [];
  for (const zone of zones) {
    const start = geometry.ringOffsets[zone];
    const count = geometry.ringOffsets[zone + 1] - start;
    for (let vertex = 0; vertex < count; vertex++) {
      const first = (start + vertex) * 2;
      const second = (start + ((vertex + 1) % count)) * 2;
      segments.push(
        geometry.positions[first],
        geometry.positions[first + 1],
        geometry.positions[second],
        geometry.positions[second + 1]
      );
    }
  }
  return Float32Array.from(segments);
}
