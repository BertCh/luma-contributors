// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_WGS84_SEMI_MAJOR_AXIS_METERS} from '@luma-contributors/analysis/analysis-core';
import {getDistanceScales, lngLatToWorld, MAX_LATITUDE, worldToLngLat} from '@math.gl/web-mercator';

/**
 * Shared spherical Web Mercator helpers for the showcase. The Mercator formula comes from
 * `@math.gl/web-mercator`; the wrappers here only convert between its 512-unit world and the
 * units the showcase scenes use (meters, radians, degrees, world fractions).
 */

/** Sphere radius of EPSG:3857, in meters. */
export const WEB_MERCATOR_RADIUS_METERS = GPU_WGS84_SEMI_MAJOR_AXIS_METERS;

/** Radius that expresses Mercator y in degrees (180 / PI). */
export const WEB_MERCATOR_DEGREES_RADIUS = 180 / Math.PI;

/** Width of the `@math.gl/web-mercator` world (`lngLatToWorld` returns [0, 512]). */
const WORLD_SIZE = 512;

/**
 * Latitude limit of the Web Mercator plane (about 85.0511 degrees, where the projected world is
 * square), re-exported from `@math.gl/web-mercator`. The single clamp used by the showcase.
 */
export const WEB_MERCATOR_MAXIMUM_LATITUDE = MAX_LATITUDE;

/** Clamps a latitude (degrees) to `[-limit, limit]`. */
export function clampLatitude(
  latitude: number,
  limit: number = WEB_MERCATOR_MAXIMUM_LATITUDE
): number {
  return Math.min(Math.max(latitude, -limit), limit);
}

/**
 * Mercator y of a latitude (degrees), scaled by `radius`. The default radius of 1 gives radians,
 * `WEB_MERCATOR_RADIUS_METERS` gives EPSG:3857 meters and `WEB_MERCATOR_DEGREES_RADIUS` degrees.
 * The latitude is not clamped.
 */
export function getMercatorY(latitude: number, radius: number = 1): number {
  const worldY = lngLatToWorld([0, latitude])[1];
  return radius * ((worldY / WORLD_SIZE) * 2 * Math.PI - Math.PI);
}

/** Inverse of {@link getMercatorY}: the latitude (degrees) of a Mercator y scaled by `radius`. */
export function getLatitudeFromMercatorY(mercatorY: number, radius: number = 1): number {
  const worldY = ((mercatorY / radius + Math.PI) / (2 * Math.PI)) * WORLD_SIZE;
  return worldToLngLat([0, worldY])[1];
}

/** Longitude and latitude (degrees) to EPSG:3857 meters. Latitude is not clamped. */
export function projectWebMercatorMeters(longitude: number, latitude: number): [number, number] {
  return [
    (WEB_MERCATOR_RADIUS_METERS * longitude * Math.PI) / 180,
    getMercatorY(latitude, WEB_MERCATOR_RADIUS_METERS)
  ];
}

/** EPSG:3857 meters to longitude and latitude (degrees). */
export function unprojectWebMercatorMeters(x: number, y: number): [number, number] {
  return [
    (x / WEB_MERCATOR_RADIUS_METERS) * (180 / Math.PI),
    getLatitudeFromMercatorY(y, WEB_MERCATOR_RADIUS_METERS)
  ];
}

/** Web Mercator y of a latitude as a fraction of the world height (north is 0, clamped to {@link WEB_MERCATOR_MAXIMUM_LATITUDE}). */
export function getNormalizedMercatorY(latitude: number): number {
  return 1 - lngLatToWorld([0, clampLatitude(latitude)])[1] / WORLD_SIZE;
}

/**
 * Ground metres per CSS pixel at `latitude` (degrees) and `zoom` for the 512 px world used by
 * deck.gl and MapLibre. The latitude scale comes from `getDistanceScales`; math.gl's own
 * circumference constant is the rounded 4e7 m, so it is normalized out and the true EPSG:3857
 * equatorial circumference (2 pi R, 78,271.5 m per pixel at zoom 0) is applied instead.
 */
export function getWebMercatorMetersPerPixel(latitude: number, zoom: number): number {
  const latitudeScale =
    getDistanceScales({longitude: 0, latitude}).metersPerUnit[0] /
    getDistanceScales({longitude: 0, latitude: 0}).metersPerUnit[0];
  return (2 * Math.PI * WEB_MERCATOR_RADIUS_METERS * latitudeScale) / (WORLD_SIZE * 2 ** zoom);
}
