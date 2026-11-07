// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getResidentSpatialJoinPointTransform,
  normalizeResidentSpatialJoinCrs,
  transformResidentSpatialJoinPoint
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-reproject';
import {listSpatialJoinPairs} from '../../../examples/deck/kepler-compute/src/analysis-join-policy';

test('CRS labels normalize and unknown systems are not guessed', () => {
  expect(normalizeResidentSpatialJoinCrs('EPSG:4326')).toBe('EPSG:4326');
  expect(normalizeResidentSpatialJoinCrs('epsg:3857')).toBe('EPSG:3857');
  expect(normalizeResidentSpatialJoinCrs('CRS:84')).toBe('EPSG:4326');
  expect(normalizeResidentSpatialJoinCrs('EPSG:27700')).toBeUndefined();
});

test('point transforms round trip and reproject-then-join equals the lng/lat join', () => {
  const toMercator = getResidentSpatialJoinPointTransform('EPSG:4326', 'EPSG:3857');
  const toGeographic = getResidentSpatialJoinPointTransform('EPSG:3857', 'EPSG:4326');
  expect(getResidentSpatialJoinPointTransform('EPSG:3857', 'EPSG:3857')).toBe('none');
  const [x, y] = transformResidentSpatialJoinPoint(toMercator, -122.4, 37.8);
  const [longitude, latitude] = transformResidentSpatialJoinPoint(toGeographic, x, y);
  expect(longitude).toBeCloseTo(-122.4, 9);
  expect(latitude).toBeCloseTo(37.8, 9);
  // Polygon in Web Mercator meters, points given as longitude/latitude.
  const square = [-10, -10, 10, -10, 10, 10, -10, 10, -10, -10].reduce<[number, number][]>(
    (ring, value, index, values) =>
      index % 2
        ? ring
        : [...ring, transformResidentSpatialJoinPoint(toMercator, value, values[index + 1])],
    []
  );
  const geographicPoints: [number, number][] = [
    [0, 0],
    [9.9, 9.9],
    [10.1, 0],
    [-170, 5]
  ];
  const pairs = listSpatialJoinPairs(
    geographicPoints.map(([px, py]) => transformResidentSpatialJoinPoint(toMercator, px, py)),
    [[square]],
    'intersects'
  );
  expect(pairs.map(([pointIndex]) => pointIndex)).toEqual([0, 1]);
});
