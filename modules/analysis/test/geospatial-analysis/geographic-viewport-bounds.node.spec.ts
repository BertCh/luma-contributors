// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getGeographicViewportBounds} from '../../src/geospatial-analysis';

test('viewport bounds union every vertex across empty and disjoint batches', () => {
  const bounds = getGeographicViewportBounds([
    new Float32Array(),
    Float32Array.from([1, 2, 100, 3, 4, 200]),
    Float32Array.from([-10, 20, -100, 30, -40, -200])
  ]);
  expect(bounds).toEqual([-12, -43, 32, 23]);
});

test('Float32 fitting margin encloses original coordinates even on a degenerate axis', () => {
  const original = [-74.01, 40.78];
  const positions = Float32Array.from(original);
  expect(positions[1]).toBeLessThan(original[1]);
  const bounds = getGeographicViewportBounds([positions], 2)!;
  expect(bounds[0]).toBeLessThan(original[0]);
  expect(bounds[1]).toBeLessThan(original[1]);
  expect(bounds[2]).toBeGreaterThan(original[0]);
  expect(bounds[3]).toBeGreaterThan(original[1]);
  const zero = getGeographicViewportBounds([new Float32Array(2)], 2)!;
  expect(zero[0]).toBeLessThan(0);
  expect(zero[2]).toBeGreaterThan(0);
});

test('empty positions have no viewport extent', () => {
  expect(getGeographicViewportBounds([])).toBeNull();
  expect(getGeographicViewportBounds([new Float32Array()])).toBeNull();
});

test('malformed packed positions are rejected with a domain-named diagnostic', () => {
  expect(() => getGeographicViewportBounds([new Float32Array(2)])).toThrow(
    'GeographicViewportBounds positionBatches lengths'
  );
  expect(() => getGeographicViewportBounds([Float32Array.from([0, NaN])], 2)).toThrow(
    'GeographicViewportBounds positionBatches XY coordinates must be finite'
  );
});

test.each([
  [
    [-179, 10, 179, 11],
    [-180, 9.95, 180, 11.05]
  ],
  [
    [180, 10, 180, 11],
    [180 - 180 * 2 ** -23, 9.95, 180, 11.05]
  ],
  [
    [-180, -90, 180, 90],
    [-180, -90, 180, 90]
  ]
])('padded global and boundary extents remain admissible to native geographic fitting: %j', (positions, expected) => {
  const bounds = getGeographicViewportBounds([Float32Array.from(positions)], 2)!;
  expect(bounds).toEqual(expected);
  // Pinned Kepler validateBounds requires finite longitudes in [-180,180], latitudes in [-90,90].
  expect(bounds.every(Number.isFinite)).toBe(true);
  expect(bounds[0]).toBeGreaterThanOrEqual(-180);
  expect(bounds[2]).toBeLessThanOrEqual(180);
  expect(bounds[1]).toBeGreaterThanOrEqual(-90);
  expect(bounds[3]).toBeLessThanOrEqual(90);
});
