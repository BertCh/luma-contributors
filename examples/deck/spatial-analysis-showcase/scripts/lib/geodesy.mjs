// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared spherical geodesy for reference-check and data-prep scripts. Angle conversion comes from
// math.gl; the haversine formula is kept verbatim (math.gl's turf distance is float-equivalent but
// not bit-identical, and these scripts are numerical references for external tools).
import {toRadians, toDegrees} from '@math.gl/core';

export {toRadians, toDegrees};

/** IUGG mean Earth radius in meters. */
export const EARTH_RADIUS_METERS = 6371008.8;

/** Central angle in radians between two points given in radians (latitudes and longitude delta). */
export function measureHaversineAngleRadians(latitudeA, latitudeB, longitudeDelta) {
  const haversine =
    Math.sin((latitudeA - latitudeB) / 2) ** 2 +
    Math.cos(latitudeA) * Math.cos(latitudeB) * Math.sin(longitudeDelta / 2) ** 2;
  return 2 * Math.asin(Math.sqrt(Math.min(1, haversine)));
}

/** Great-circle distance between two lon/lat degree points, in the unit of `radius`. */
export function measureHaversineDistance(
  [longitudeA, latitudeA],
  [longitudeB, latitudeB],
  radius = EARTH_RADIUS_METERS
) {
  return (
    radius *
    measureHaversineAngleRadians(
      toRadians(latitudeA),
      toRadians(latitudeB),
      toRadians(longitudeB - longitudeA)
    )
  );
}
