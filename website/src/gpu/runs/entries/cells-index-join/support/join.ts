// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded zones and points shared by the spatial-join runs.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import {createStarRing, packPolygonRings, type PackedPolygons} from './polygons';

/** Five star-shaped zones spread over the world. */
export function createZones(random: () => number, world: {width: number; height: number}) {
  const centers: [number, number][] = [
    [world.width * 0.16, 0.7],
    [world.width * 0.4, 0.34],
    [world.width * 0.5, 0.78],
    [world.width * 0.72, 0.36],
    [world.width * 0.86, 0.72]
  ];
  return packPolygonRings(centers.map(center => createStarRing(random, center, 0.22, 9, 0.45)));
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
