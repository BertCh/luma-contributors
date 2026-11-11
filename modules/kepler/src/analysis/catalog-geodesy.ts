// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {safeMod, toDegrees, toRadians} from '@math.gl/core';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma-contributors/analysis/analysis-core';

/** Mean Earth radius of the spherical EPSG:4326 model used by every Kepler analysis (meters). */
export const EARTH_RADIUS_METERS = GPU_EARTH_MEAN_RADIUS_METERS;

/**
 * Haversine great-circle distance from radian differences and the two radian latitudes.
 * Hot loops pre-convert to radians and call this directly.
 */
export function getHaversineDistanceMeters(
  deltaLatitudeRadians: number,
  deltaLongitudeRadians: number,
  firstLatitudeRadians: number,
  secondLatitudeRadians: number
): number {
  const sinHalfLatitude = Math.sin(deltaLatitudeRadians / 2);
  const sinHalfLongitude = Math.sin(deltaLongitudeRadians / 2);
  const haversine =
    sinHalfLatitude * sinHalfLatitude +
    Math.cos(firstLatitudeRadians) *
      Math.cos(secondLatitudeRadians) *
      sinHalfLongitude *
      sinHalfLongitude;
  return 2 * EARTH_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(haversine)));
}

/** Great-circle distance (meters) between two longitude/latitude positions in degrees. */
export function getGreatCircleDistanceMeters(
  firstLongitude: number,
  firstLatitude: number,
  secondLongitude: number,
  secondLatitude: number
): number {
  return getHaversineDistanceMeters(
    toRadians(secondLatitude - firstLatitude),
    toRadians(secondLongitude - firstLongitude),
    toRadians(firstLatitude),
    toRadians(secondLatitude)
  );
}

/** Wraps a longitude in degrees to [-180, 180). */
export function normalizeLongitude(longitude: number): number {
  return safeMod(longitude + 180, 360) - 180;
}

/** Initial great-circle bearing in degrees clockwise from north, in [0, 360). */
export function getInitialBearingDegrees(
  fromLongitude: number,
  fromLatitude: number,
  toLongitude: number,
  toLatitude: number
): number {
  const fromLatitudeRadians = toRadians(fromLatitude);
  const toLatitudeRadians = toRadians(toLatitude);
  const deltaLongitude = toRadians(toLongitude - fromLongitude);
  const radians = Math.atan2(
    Math.sin(deltaLongitude) * Math.cos(toLatitudeRadians),
    Math.cos(fromLatitudeRadians) * Math.sin(toLatitudeRadians) -
      Math.sin(fromLatitudeRadians) * Math.cos(toLatitudeRadians) * Math.cos(deltaLongitude)
  );
  return (toDegrees(radians) + 360) % 360;
}

/** Destination `distanceMeters` along `bearingDegrees` from a longitude/latitude origin. */
export function getGeodesicDestination(
  originLongitude: number,
  originLatitude: number,
  bearingDegrees: number,
  distanceMeters: number
): [longitude: number, latitude: number] {
  const bearing = toRadians(bearingDegrees);
  const angular = distanceMeters / EARTH_RADIUS_METERS;
  const latitude = toRadians(originLatitude);
  const nextLatitude = Math.asin(
    Math.sin(latitude) * Math.cos(angular) +
      Math.cos(latitude) * Math.sin(angular) * Math.cos(bearing)
  );
  const nextLongitude =
    toRadians(originLongitude) +
    Math.atan2(
      Math.sin(bearing) * Math.sin(angular) * Math.cos(latitude),
      Math.cos(angular) - Math.sin(latitude) * Math.sin(nextLatitude)
    );
  return [normalizeLongitude(toDegrees(nextLongitude)), toDegrees(nextLatitude)];
}
