// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  EARTH_RADIUS_METERS,
  measureRingSignedArea,
  type CatalogFeature,
  type CatalogPosition
} from '../src/analysis/catalog-geojson';
import {makeLocalFrame} from '../src/analysis/catalog-local-frame';
import {getFeatureAnchor} from '../src/analysis/catalog-platform3-b-common';

const frame = makeLocalFrame('planar', []);
const shell: CatalogPosition[] = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
  [0, 0]
];

test('centroid subtracts hole area, matching Shapely to 1e-9 independently of winding', () => {
  const hole: CatalogPosition[] = [
    [1, 1],
    [3, 1],
    [3, 9],
    [1, 9],
    [1, 1]
  ];
  for (const ring of [hole, [...hole].reverse()]) {
    const feature: CatalogFeature = {geometry: {type: 'Polygon', coordinates: [shell, ring]}};
    const anchor = getFeatureAnchor(feature, frame)!;
    // Shapely: (100 * 5 - 16 * 2) / (100 - 16) = 5.571428571428571.
    expect(anchor[0]).toBeCloseTo(5.571428571428571, 9);
    expect(anchor[1]).toBeCloseTo(5, 9);
  }
  const holeSix: CatalogPosition[] = [
    [1, 1],
    [7, 1],
    [7, 7],
    [1, 7],
    [1, 1]
  ];
  const anchor = getFeatureAnchor(
    {geometry: {type: 'Polygon', coordinates: [shell, holeSix]}},
    frame
  )!;
  expect(anchor[0]).toBeCloseTo(5.5625, 9);
});

test('multipolygon centroid weights each part by absolute area', () => {
  const second = shell.map(([x, y]) => [x + 20, y]).reverse();
  expect(
    getFeatureAnchor({geometry: {type: 'MultiPolygon', coordinates: [[shell], [second]]}}, frame)
  ).toEqual([15, 5]);
});

test('tiny degree polygons and large offsets retain the area centroid, not a vertex mean', () => {
  const triangle: CatalogPosition[] = [
    [0, 0],
    [3e-7, 0],
    [0, 3e-7],
    [0, 0]
  ];
  const anchor = getFeatureAnchor({geometry: {type: 'Polygon', coordinates: [triangle]}}, frame)!;
  expect(anchor[0]).toBeCloseTo(1e-7, 15);
  expect(anchor[1]).toBeCloseTo(1e-7, 15);
  const shifted = shell.map(([x, y]) => [x + 1e9, y + 1e9]);
  expect(getFeatureAnchor({geometry: {type: 'Polygon', coordinates: [shifted]}}, frame)).toEqual([
    1e9 + 5,
    1e9 + 5
  ]);
});

test('planar area 38 survives an offset of one billion, matching Shapely', () => {
  const rectangle: CatalogPosition[] = [
    [0, 0],
    [19, 0],
    [19, 2],
    [0, 2],
    [0, 0]
  ];
  const shifted = rectangle.map(([x, y]): CatalogPosition => [x + 1e9, y + 1e9]);
  expect(measureRingSignedArea(shifted, 'planar')).toBe(38);
  expect(measureRingSignedArea([...shifted].reverse(), 'planar')).toBe(-38);
});

test('antimeridian area equals the equivalent unwrapped spherical rectangle', () => {
  const wrapped: CatalogPosition[] = [
    [179.9, 10],
    [-179.9, 10],
    [-179.9, 11],
    [179.9, 11],
    [179.9, 10]
  ];
  const unwrapped: CatalogPosition[] = wrapped.map(([longitude, latitude]) => [
    longitude < 0 ? longitude + 360 : longitude,
    latitude
  ]);
  const expected =
    EARTH_RADIUS_METERS ** 2 *
    ((0.2 * Math.PI) / 180) *
    (Math.sin((11 * Math.PI) / 180) - Math.sin((10 * Math.PI) / 180));
  expect(Math.abs(measureRingSignedArea(wrapped, 'EPSG:4326')) / expected).toBeCloseTo(1, 10);
  expect(measureRingSignedArea(wrapped, 'EPSG:4326')).toBeCloseTo(
    measureRingSignedArea(unwrapped, 'EPSG:4326'),
    3
  );
});
