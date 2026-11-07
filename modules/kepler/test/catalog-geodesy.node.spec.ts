// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EllipsoidGeodesic} from '@math.gl/geospatial';
import {expect, test} from 'vitest';
import {
  EARTH_RADIUS_METERS,
  getGeodesicDestination,
  getGreatCircleDistanceMeters,
  getInitialBearingDegrees,
  normalizeLongitude
} from '../src/analysis/catalog-geodesy';

// A sphere is an ellipsoid with equal axes; the iterative geodesic is an independent oracle.
const sphereGeodesic = new EllipsoidGeodesic({
  semiMajorAxis: EARTH_RADIUS_METERS,
  semiMinorAxis: EARTH_RADIUS_METERS
});

const PAIRS: [number, number, number, number][] = [
  [-122.4, 37.8, 2.35, 48.85],
  [0, 0, 10, 10],
  [170, -20, -175, 15],
  [13.4, 52.5, 13.5, 52.6]
];

test('great-circle distance matches the geodesic oracle on a sphere', () => {
  for (const [firstLongitude, firstLatitude, secondLongitude, secondLatitude] of PAIRS) {
    const oracle = sphereGeodesic.inverse(
      firstLatitude,
      firstLongitude,
      secondLatitude,
      secondLongitude
    );
    expect(
      getGreatCircleDistanceMeters(firstLongitude, firstLatitude, secondLongitude, secondLatitude)
    ).toBeCloseTo(oracle.s12, 6);
  }
});

test('initial bearing and destination match the geodesic oracle on a sphere', () => {
  for (const [firstLongitude, firstLatitude, secondLongitude, secondLatitude] of PAIRS) {
    const oracle = sphereGeodesic.inverse(
      firstLatitude,
      firstLongitude,
      secondLatitude,
      secondLongitude
    );
    const bearing = getInitialBearingDegrees(
      firstLongitude,
      firstLatitude,
      secondLongitude,
      secondLatitude
    );
    expect(bearing).toBeCloseTo((oracle.azi1 + 360) % 360, 6);
    const [longitude, latitude] = getGeodesicDestination(
      firstLongitude,
      firstLatitude,
      bearing,
      oracle.s12
    );
    expect(latitude).toBeCloseTo(secondLatitude, 6);
    expect(normalizeLongitude(longitude - secondLongitude)).toBeCloseTo(0, 6);
  }
});

test('normalizeLongitude wraps into [-180, 180)', () => {
  expect(normalizeLongitude(190)).toBeCloseTo(-170, 10);
  expect(normalizeLongitude(-190)).toBeCloseTo(170, 10);
  expect(normalizeLongitude(10)).toBeCloseTo(10, 10);
});
