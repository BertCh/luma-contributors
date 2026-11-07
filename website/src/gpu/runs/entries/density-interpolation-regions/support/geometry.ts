// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the runs of this group: packed polygon columns from tracts, and uploads of
// them as the views the polygon operations take.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {TractsInput} from '../../../inputs';

/** GeoArrow-style polygon columns: one polygon per feature, one shell ring per polygon. */
export type PackedPolygons = {
  featureCount: number;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

/** The hexagon-like tracts as packed polygons (each tract has six corners, counterclockwise). */
export function getPackedPolygons(tracts: TractsInput): PackedPolygons {
  const count = tracts.count;
  const ascending = (step: number) => Uint32Array.from({length: count + 1}, (_, row) => row * step);
  return {
    featureCount: count,
    positions: tracts.vertices.slice(0, count * 12),
    featureOffsets: ascending(1),
    polygonOffsets: ascending(1),
    ringOffsets: ascending(6)
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
