// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerSpatialSelection} from '@luma-contributors/kepler';

import {toRadians} from '@math.gl/core';

export const BRUSH_CIRCLE_SEGMENT_COUNT = 48;
const METERS_PER_DEGREE_LATITUDE = 111320;
const MAXIMUM_BRUSH_LATITUDE = 85;
/** Coordinates no Float32 point can reach; used so an inactive brush selects nothing. */
const INACTIVE_COORDINATE = 3e30;

export type BrushCircleResult =
  | {status: 'ready'; selection: KeplerSpatialSelection}
  | {status: 'inactive'; selection: KeplerSpatialSelection}
  | {status: 'unsupported'; selection: KeplerSpatialSelection; reason: string};

/** Same vertex count for every state, so moving the brush never changes selection topology. */
function makeInactiveSelection(): KeplerSpatialSelection {
  // The last vertex differs from the first: preparation drops a repeated closing vertex, which would
  // otherwise leave this ring one vertex shorter than a ready circle and change the topology key.
  const ring = Array.from(
    {length: BRUSH_CIRCLE_SEGMENT_COUNT},
    (_, index): readonly [number, number] => [
      INACTIVE_COORDINATE,
      index === BRUSH_CIRCLE_SEGMENT_COUNT - 1 ? 2 * INACTIVE_COORDINATE : INACTIVE_COORDINATE
    ]
  );
  return {kind: 'polygon', polygons: [[ring]]};
}

/**
 * Longitude/latitude circle approximation of a meter radius. Longitude spread is scaled by the
 * cosine of the center latitude; the brush is refused near the poles and across the antimeridian
 * because the GPU containment test is a linear degree-plane test.
 */
export function makeBrushCircleSelection(
  center: readonly [number, number] | null,
  radiusMeters: number
): BrushCircleResult {
  const inactive = makeInactiveSelection();
  if (!center || !(radiusMeters > 0)) return {status: 'inactive', selection: inactive};
  const [longitude, latitude] = center;
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude))
    return {status: 'inactive', selection: inactive};
  if (Math.abs(latitude) > MAXIMUM_BRUSH_LATITUDE)
    return {
      status: 'unsupported',
      selection: inactive,
      reason: 'The GPU brush is not available above 85 degrees latitude.'
    };
  const latitudeRadius = radiusMeters / METERS_PER_DEGREE_LATITUDE;
  const longitudeRadius = latitudeRadius / Math.cos(toRadians(latitude));
  if (
    Math.abs(longitude) + longitudeRadius > 180 ||
    Math.abs(latitude) + latitudeRadius > MAXIMUM_BRUSH_LATITUDE
  )
    return {
      status: 'unsupported',
      selection: inactive,
      reason: 'The GPU brush is not available across the antimeridian or near the poles.'
    };
  const ring = Array.from({length: BRUSH_CIRCLE_SEGMENT_COUNT}, (_, index): [number, number] => {
    const angle = (2 * Math.PI * index) / BRUSH_CIRCLE_SEGMENT_COUNT;
    return [
      longitude + longitudeRadius * Math.cos(angle),
      latitude + latitudeRadius * Math.sin(angle)
    ];
  });
  return {status: 'ready', selection: {kind: 'polygon', polygons: [[ring]]}};
}
