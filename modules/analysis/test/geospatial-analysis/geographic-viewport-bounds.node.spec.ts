// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getGeographicViewportBounds, getGeographicViewportFit} from '../../src/geospatial-analysis';

test('viewport bounds union every vertex across empty and disjoint batches', () => {
  const bounds = getGeographicViewportBounds(
    [
      new Float32Array(),
      Float32Array.from([1, 2, 100, 3, 4, 200]),
      Float32Array.from([-10, 20, -100, 30, -40, -200])
    ],
    3
  );
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
  expect(getGeographicViewportBounds([], 2)).toBeNull();
  expect(getGeographicViewportBounds([new Float32Array()], 3)).toBeNull();
});

test('malformed packed positions are rejected with a domain-named diagnostic', () => {
  expect(() => getGeographicViewportBounds([new Float32Array(2)], 3)).toThrow(
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

test('UNI-07 requires explicit strides for packed batches', () => {
  const xy = Float32Array.of(-30, 40, -29, 41, -28, 42);
  expect(() => getGeographicViewportBounds([xy], undefined as never)).toThrow(
    /positionSize.*explicitly/
  );
  expect(getGeographicViewportBounds([xy], 2)).toEqual([-30.1, 39.9, -27.9, 42.1]);
  expect(getGeographicViewportBounds([xy], 3)).toEqual([-33.55, -31.4, 44.55, 43.4]);
  function* unambiguous() {
    yield xy;
    yield Float32Array.of(100, 60, 3);
  }
  // XYZ union: x [-30,100], y [-28,60]; margins are 6.5 and 4.4 degrees.
  expect(getGeographicViewportBounds(unambiguous(), 3)).toEqual([-36.5, -32.4, 106.5, 64.4]);
});

test('UNI-18 rejects noncanonical bounded input and fits a compact longitude branch', () => {
  expect(() => getGeographicViewportBounds([Float32Array.of(181, 5)], 2)).toThrow(
    /longitude.*181.*-180.*180/
  );
  const fit = getGeographicViewportFit([Float32Array.of(179, 5, -179, 5)], 2)!;
  // Shortest arc is two degrees about the dateline, plus 5% on each side.
  expect(fit.longitudeCenterDegrees).toBe(-180);
  expect(fit.longitudeSpanDegrees).toBeCloseTo(2.2, 12);
  expect(fit.wrapsAntimeridian).toBe(true);
  const continuous = getGeographicViewportFit([Float32Array.of(181, 5)], 2)!;
  expect(continuous.longitudeCenterDegrees).toBe(-179);
  expect(continuous.longitudeSpanDegrees).toBeGreaterThan(0);
  expect(getGeographicViewportFit([], 2)).toBeNull();
});

test.each([
  -1e-7, -4e-9, 0, 4e-9, 1e-7
])('B2 fit contains a single longitude near Greenwich: %s', longitude => {
  const positions = Float32Array.of(longitude, 10);
  const fit = getGeographicViewportFit([positions], 2)!;
  expect(Number.isFinite(fit.longitudeSpanDegrees)).toBe(true);
  expect(fit.longitudeSpanDegrees).toBeGreaterThan(0);
  expect(fit.longitudeCenterDegrees).toBe(positions[0]);
  expect(fit.longitudeCenterDegrees - fit.longitudeSpanDegrees / 2).toBeLessThanOrEqual(
    positions[0]
  );
  expect(fit.longitudeCenterDegrees + fit.longitudeSpanDegrees / 2).toBeGreaterThanOrEqual(
    positions[0]
  );
});

test('B4 forwards the explicit positionSize', () => {
  const forwardBounds = (batches: Iterable<Float32Array>, size: 2 | 3) =>
    getGeographicViewportBounds(batches, size);
  // A single XYZ row: margins are one Float32 relative epsilon on each axis.
  expect(forwardBounds([Float32Array.of(1, 2, 3)], 3)).toEqual([
    1 - 2 ** -23,
    2 - 2 ** -22,
    1 + 2 ** -23,
    2 + 2 ** -22
  ]);
});
