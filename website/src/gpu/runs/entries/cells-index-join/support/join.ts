// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded zones and points shared by the spatial-join runs.

import {createGreatLakeRings} from '../../../../data/hydrography';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import {packPolygonRings, type PackedPolygons} from './polygons';

/** Simplified Great Lakes shores retain their relative positions and areas. */
export function createZones(world: {width: number; height: number}) {
  return packPolygonRings(createGreatLakeRings(world));
}

export type UploadedZones = {
  positions: RunBuffer<'float32x2'>;
  featureOffsets: RunBuffer<'uint32'>;
  polygonOffsets: RunBuffer<'uint32'>;
  ringOffsets: RunBuffer<'uint32'>;
  centers: RunBuffer<'float32x2'>;
  outline: RunBuffer<'float32x4'>;
  /** The join's `polygons` geometry port. */
  geometry: {
    kind: 'polygons';
    positions: RunBuffer<'float32x2'>['view'];
    featureOffsets: RunBuffer<'uint32'>['view'];
    polygonOffsets: RunBuffer<'uint32'>['view'];
    ringOffsets: RunBuffer<'uint32'>['view'];
  };
};

/** Uploads the zones and builds the polygon geometry port of the join operations. */
export function uploadZones(
  builder: RunBuilder,
  zones: PackedPolygons,
  prefix = 'zones'
): UploadedZones {
  const positions = builder.upload(`${prefix}-positions`, 'float32x2', zones.positions);
  const featureOffsets = builder.upload(`${prefix}-features`, 'uint32', zones.featureOffsets);
  const polygonOffsets = builder.upload(`${prefix}-polygons`, 'uint32', zones.polygonOffsets);
  const ringOffsets = builder.upload(`${prefix}-rings`, 'uint32', zones.ringOffsets);
  return {
    positions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    centers: builder.upload(`${prefix}-centers`, 'float32x2', zones.centers),
    outline: builder.upload(`${prefix}-outline`, 'float32x4', zones.outline),
    geometry: {
      kind: 'polygons',
      positions: positions.view,
      featureOffsets: featureOffsets.view,
      polygonOffsets: polygonOffsets.view,
      ringOffsets: ringOffsets.view
    }
  };
}

/** Uniform random points over the world. */
export function createRandomPoints(
  random: () => number,
  world: {width: number; height: number},
  count: number
): Float32Array {
  const positions = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    positions[index * 2] = random() * world.width;
    positions[index * 2 + 1] = random();
  }
  return positions;
}

/** Zone vertices and outline segments scaled about each zone center by `size`. */
export function scaleZones(
  zones: PackedPolygons,
  size: number
): {positions: Float32Array; outline: Float32Array} {
  const getCenter = (vertex: number, component: number) => {
    const ring = zones.ringOffsets.findIndex(
      (offset, row) => vertex >= offset && vertex < zones.ringOffsets[row + 1]
    );
    return zones.centers[ring * 2 + component];
  };
  return {
    positions: Float32Array.from(zones.positions, (value, index) => {
      const center = getCenter(Math.floor(index / 2), index % 2);
      return center + (value - center) * size;
    }),
    // Outline segment `i` starts at vertex `i`.
    outline: Float32Array.from(zones.outline, (value, index) => {
      const center = getCenter(Math.floor(index / 4), index % 2);
      return center + (value - center) * size;
    })
  };
}

export type Bounds = {minimumX: number; minimumY: number; maximumX: number; maximumY: number};

/** The bounding box of every zone ring, from packed (x, y) `positions` (scaled or not). */
export function getZoneBounds(zones: PackedPolygons, positions: ArrayLike<number>): Bounds[] {
  return Array.from({length: zones.featureCount}, (_, ring) => {
    const bounds = {
      minimumX: Infinity,
      minimumY: Infinity,
      maximumX: -Infinity,
      maximumY: -Infinity
    };
    for (let vertex = zones.ringOffsets[ring]; vertex < zones.ringOffsets[ring + 1]; vertex++) {
      bounds.minimumX = Math.min(bounds.minimumX, positions[vertex * 2]);
      bounds.maximumX = Math.max(bounds.maximumX, positions[vertex * 2]);
      bounds.minimumY = Math.min(bounds.minimumY, positions[vertex * 2 + 1]);
      bounds.maximumY = Math.max(bounds.maximumY, positions[vertex * 2 + 1]);
    }
    return bounds;
  });
}

/** Box outlines as (x0, y0, x1, y1) segments, each box grown by `distance` on every side. */
export function createBoundsSegments(bounds: readonly Bounds[], distance = 0): Float32Array {
  const segments = new Float32Array(bounds.length * 16);
  bounds.forEach((box, index) => {
    const x0 = box.minimumX - distance;
    const y0 = box.minimumY - distance;
    const x1 = box.maximumX + distance;
    const y1 = box.maximumY + distance;
    segments.set([x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0], index * 16);
  });
  return segments;
}

/** Distance from (x, y) to a box, and the closest point of the box. */
export function getBoundsDistance(
  box: Bounds,
  x: number,
  y: number
): {distance: number; point: [number, number]} {
  const pointX = Math.min(box.maximumX, Math.max(box.minimumX, x));
  const pointY = Math.min(box.maximumY, Math.max(box.minimumY, y));
  return {distance: Math.hypot(x - pointX, y - pointY), point: [pointX, pointY]};
}

/** The nearest box to (x, y). */
export function findNearestBounds(bounds: readonly Bounds[], x: number, y: number) {
  let best = {...getBoundsDistance(bounds[0], x, y), index: 0};
  bounds.forEach((box, index) => {
    const hit = getBoundsDistance(box, x, y);
    if (hit.distance < best.distance) best = {...hit, index};
  });
  return best;
}
