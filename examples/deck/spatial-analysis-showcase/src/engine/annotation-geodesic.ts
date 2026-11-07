// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toRadians} from '@math.gl/core';
import {
  EARTH_RADIUS_METERS,
  getGeodesicDestination,
  getGreatCircleDistanceMeters,
  normalizeLongitude
} from '../cartography/reference-geometry';
import type {LngLat} from '../cartography/types';
import {WEB_MERCATOR_MAXIMUM_LATITUDE} from './web-mercator';

export {
  EARTH_RADIUS_METERS,
  getGeodesicDestination,
  getGreatCircleDistanceMeters,
  normalizeLongitude
};

/** Meters per degree of arc on the mean-radius sphere used for every geodesic in the overlay. */
const METERS_PER_DEGREE = toRadians(EARTH_RADIUS_METERS);
/** Latitude limit of the Web Mercator plane; points are clamped here before projection. */
export const MAX_MERCATOR_LATITUDE = WEB_MERCATOR_MAXIMUM_LATITUDE;

/**
 * A geodesic circle: `vertexCount` great-circle destination points at `radiusMeters` around
 * `center`, longitudes normalized into `[-180, 180]`. The ring is open (the first vertex is not
 * repeated). Split it with {@link splitAtAntimeridian} before projecting.
 */
export function geodesicCircleRing(
  center: LngLat,
  radiusMeters: number,
  vertexCount: number
): LngLat[] {
  const ring: LngLat[] = [];
  for (let i = 0; i < vertexCount; i++) {
    ring.push(getGeodesicDestination(center, (i / vertexCount) * 360, radiusMeters));
  }
  return ring;
}

/**
 * A local circle on the equirectangular tangent plane (exact for city radii): offsets of
 * `radiusMeters` in degrees of latitude and longitude. The ring is open.
 */
export function localCircleRing(
  center: LngLat,
  radiusMeters: number,
  vertexCount: number
): LngLat[] {
  const [longitude, latitude] = center;
  const latitudeRadius = radiusMeters / METERS_PER_DEGREE;
  const longitudeRadius =
    radiusMeters / (METERS_PER_DEGREE * Math.max(Math.cos(toRadians(latitude)), 1e-6));
  const ring: LngLat[] = [];
  for (let i = 0; i < vertexCount; i++) {
    const angle = (i / vertexCount) * Math.PI * 2;
    ring.push([
      longitude + Math.cos(angle) * longitudeRadius,
      latitude + Math.sin(angle) * latitudeRadius
    ]);
  }
  return ring;
}

/** Result of {@link splitAtAntimeridian}. */
export type SplitPath = {
  /** Continuous pieces; each has monotone-in-world longitudes and never jumps across +-180. */
  pieces: LngLat[][];
  /** `true` when the input was a closed ring that needed no split, so the path may be closed. */
  closed: boolean;
};

/**
 * Cuts a path where it crosses the antimeridian (consecutive longitudes more than 180 degrees
 * apart), inserting the interpolated crossing point at +-180 on both sides. A closed ring that
 * crosses is returned as open pieces (its last and first pieces merged), because stroking along
 * the antimeridian would draw a false edge across the map.
 */
export function splitAtAntimeridian(points: readonly LngLat[], closed: boolean): SplitPath {
  if (points.length === 0) return {pieces: [], closed: false};
  const sequence = closed ? [...points, points[0]] : points;
  const pieces: LngLat[][] = [[sequence[0]]];
  for (let i = 1; i < sequence.length; i++) {
    const previous = sequence[i - 1];
    const current = sequence[i];
    if (Math.abs(current[0] - previous[0]) > 180) {
      const edge = previous[0] > 0 ? 180 : -180;
      const unwrapped = current[0] + (previous[0] > 0 ? 360 : -360);
      const t = (edge - previous[0]) / (unwrapped - previous[0]);
      const crossingLatitude = previous[1] + t * (current[1] - previous[1]);
      pieces[pieces.length - 1].push([edge, crossingLatitude]);
      pieces.push([[-edge, crossingLatitude]]);
    }
    pieces[pieces.length - 1].push(current);
  }
  if (pieces.length === 1) {
    if (closed) pieces[0].pop();
    return {pieces, closed};
  }
  if (closed) {
    // The ring starts and ends on the same vertex: join the last piece to the first.
    const last = pieces.pop() as LngLat[];
    pieces[0] = [...last, ...pieces[0].slice(1)];
  }
  return {pieces, closed: false};
}

/** "850 m", "2.4 km", "148 km", "1,250 km": metres under 1 km, one decimal under 100 km. */
export function formatDistance(meters: number): string {
  if (meters < 1000) return `${Math.round(meters)} m`;
  const kilometers = meters / 1000;
  if (kilometers < 100) return `${kilometers.toFixed(1)} km`;
  return `${Math.round(kilometers).toLocaleString('en-US')} km`;
}
