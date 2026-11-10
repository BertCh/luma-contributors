// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the runs of this group: packed polygon columns from tracts, and uploads of
// them as the views the polygon operations take.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import {createPolylines, type Random, type TractsInput, type World} from '../../../inputs';
import type {RunGlyph} from '../../../interaction';
import {findPolygonAt} from '../../../pick';

/** GeoArrow-style polygon columns: one polygon per feature, one shell ring per polygon. */
export type PackedPolygons = {
  featureCount: number;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

/** Each Census tract in this extract consists of a single exterior ring. */
export function getPackedPolygons(tracts: TractsInput): PackedPolygons {
  const count = tracts.count;
  const ascending = (step: number) => Uint32Array.from({length: count + 1}, (_, row) => row * step);
  return {
    featureCount: count,
    positions: tracts.vertices,
    featureOffsets: ascending(1),
    polygonOffsets: ascending(1),
    ringOffsets: tracts.ringOffsets
  };
}

/** Uploaded packed polygon columns, ready to spread into an operation's props. */
export type PolygonViews = {
  positions: RunBuffer<'float32x2'>['view'];
  featureOffsets: RunBuffer<'uint32'>['view'];
  polygonOffsets: RunBuffer<'uint32'>['view'];
  ringOffsets: RunBuffer<'uint32'>['view'];
};

export function uploadPackedPolygons(
  builder: RunBuilder,
  polygons: PackedPolygons,
  id = 'polygons'
): PolygonViews {
  return {
    positions: builder.upload(`${id}-positions`, 'float32x2', polygons.positions).view,
    featureOffsets: builder.upload(`${id}-feature-offsets`, 'uint32', polygons.featureOffsets).view,
    polygonOffsets: builder.upload(`${id}-polygon-offsets`, 'uint32', polygons.polygonOffsets).view,
    ringOffsets: builder.upload(`${id}-ring-offsets`, 'uint32', polygons.ringOffsets).view
  };
}

/** Flips y inside the unit-height world, for operations whose row 0 is the lowest y. */
export function flipVertical(positions: Float32Array): Float32Array {
  return Float32Array.from(positions, (value, index) => (index % 2 === 1 ? 1 - value : value));
}

/** The uploaded tract mesh a `polygons` layer draws, plus its outline. */
export function uploadTractMesh(builder: RunBuilder, tracts: TractsInput) {
  return {
    vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
    indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
    owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
    centers: builder.upload('centers', 'float32x2', tracts.centers),
    outline: builder.upload('outline', 'float32x4', tracts.outline)
  };
}

/** The tract under a world position, or the one with the nearest center when none is. */
export function findTractAt(tracts: TractsInput, x: number, y: number): number {
  const row = findPolygonAt(tracts.vertices, tracts.triangleIndices, tracts.triangleOwners, x, y);
  if (row >= 0) return row;
  let nearest = 0;
  let nearestDistance = Infinity;
  for (let tract = 0; tract < tracts.count; tract++) {
    const distance = Math.hypot(tracts.centers[tract * 2] - x, tracts.centers[tract * 2 + 1] - y);
    if (distance < nearestDistance) {
      nearest = tract;
      nearestDistance = distance;
    }
  }
  return nearest;
}

/** The closed outline of one tract as a glyph. */
export function createTractGlyph(tracts: TractsInput, row: number): RunGlyph {
  return {
    shape: 'polyline',
    closed: true,
    points: tracts.vertices.subarray(tracts.ringOffsets[row] * 2, tracts.ringOffsets[row + 1] * 2)
  };
}

/** The tract whose center is nearest to the middle of the world: a poster-friendly probe target. */
export function findMiddleTract(tracts: TractsInput, world: World): number {
  let best = 0;
  let bestDistance = Infinity;
  for (let row = 0; row < tracts.count; row++) {
    const distance = Math.hypot(
      tracts.centers[row * 2] - world.width / 2,
      tracts.centers[row * 2 + 1] - world.height / 2
    );
    if (distance < bestDistance) {
      best = row;
      bestDistance = distance;
    }
  }
  return best;
}

export type SpreadPolylines = ReturnType<typeof createPolylines>;

/**
 * Vessel tracks rescaled so that together they fill the frame (the raw extract bunches in the
 * middle of the world); the segment list is rebuilt from the moved vertices.
 */
export function createSpreadPolylines(
  random: Random,
  world: World,
  lineCount: number,
  vertexCount: number
): SpreadPolylines {
  const lines = createPolylines(random, world, lineCount, vertexCount);
  let [minimumX, minimumY, maximumX, maximumY] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < lines.vertices.length; index += 2) {
    minimumX = Math.min(minimumX, lines.vertices[index]);
    maximumX = Math.max(maximumX, lines.vertices[index]);
    minimumY = Math.min(minimumY, lines.vertices[index + 1]);
    maximumY = Math.max(maximumY, lines.vertices[index + 1]);
  }
  const margin = 0.06;
  const vertices = Float32Array.from(lines.vertices, (value, index) =>
    index % 2 === 0
      ? margin * world.height +
        ((value - minimumX) / (maximumX - minimumX || 1)) *
          (world.width - 2 * margin * world.height)
      : margin + ((value - minimumY) / (maximumY - minimumY || 1)) * (1 - 2 * margin)
  );
  const segments: number[] = [];
  for (let line = 0; line < lines.offsets.length - 1; line++) {
    for (let vertex = lines.offsets[line]; vertex < lines.offsets[line + 1] - 1; vertex++) {
      segments.push(
        vertices[vertex * 2],
        vertices[vertex * 2 + 1],
        vertices[vertex * 2 + 2],
        vertices[vertex * 2 + 3]
      );
    }
  }
  return {
    vertices,
    offsets: lines.offsets,
    segments: Float32Array.from(segments),
    segmentCount: segments.length / 4
  };
}
