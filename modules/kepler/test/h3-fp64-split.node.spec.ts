// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getFloat32CoordinateError,
  scaleSplitCoordinateAboutCenter,
  splitFloat64Coordinates,
  splitH3BoundaryPositions,
  subtractSplitCoordinate
} from '../../../examples/deck/kepler-compute/src/h3-fp64-split';
import {
  H3_EXTRUDED_HIGHLIGHT_COLOR,
  getH3HighlightProps
} from '../../../examples/deck/kepler-compute/src/h3-highlight';

test('high/low split reconstructs Float64 coordinates far beyond Float32', () => {
  const values = Float64Array.of(
    -122.41941550000001,
    37.77492950000001,
    179.99999123456,
    -0.0000123456789
  );
  const {high, low} = splitFloat64Coordinates(values);
  for (let index = 0; index < values.length; index++)
    expect(Math.abs(values[index] - (high[index] + low[index]))).toBeLessThan(1e-12);
  expect(getFloat32CoordinateError(values)).toBeGreaterThan(1e-7);
});

test('emulated double subtraction keeps sub-Float32 precision near the viewport center', () => {
  const center = -122.41941550000001;
  const vertex = center + 1.3e-6;
  const split = splitFloat64Coordinates(Float64Array.of(vertex, center));
  const emulated = subtractSplitCoordinate(
    split.high[0],
    split.low[0],
    split.high[1],
    split.low[1]
  );
  const naive = Math.fround(Math.fround(vertex) - Math.fround(center));
  expect(Math.abs(emulated - 1.3e-6)).toBeLessThan(Math.abs(naive - 1.3e-6) + 1e-12);
  expect(Math.abs(emulated - 1.3e-6)).toBeLessThan(1e-9);
});

test('boundary split unwraps dateline rings relative to their first vertex', () => {
  const positions = Float64Array.of(179.9, 10, -179.9, 10.1, -179.8, 10.2);
  const {high, low} = splitH3BoundaryPositions(positions, Uint32Array.of(0, 3));
  expect(high[2] + low[2]).toBeCloseTo(180.1, 6);
  expect(high[4] + low[4]).toBeCloseTo(180.2, 6);
});

test('extruded H3 hover uses the native translucent white highlight', () => {
  expect(getH3HighlightProps(true, false)).toEqual({
    autoHighlight: true,
    highlightColor: H3_EXTRUDED_HIGHLIGHT_COLOR
  });
  // Native flat cells are not auto-highlighted; they draw the hovered-outline layer instead.
  expect(getH3HighlightProps(false, false).autoHighlight).toBe(false);
  expect(getH3HighlightProps(true, true).autoHighlight).toBe(false);
});

test('coverage scaling keeps the split vertex far more precise than a Float32 mix', () => {
  const centerLongitude = -122.41941550000001;
  const vertexLongitude = -122.41941550000001 + 3.3e-6;
  const coverage = 0.8;
  const [center] = [splitFloat64Coordinates(Float64Array.of(centerLongitude))];
  const [vertex] = [splitFloat64Coordinates(Float64Array.of(vertexLongitude))];
  const exact = centerLongitude + (vertexLongitude - centerLongitude) * coverage;
  const scaled = scaleSplitCoordinateAboutCenter(
    vertex.high[0],
    vertex.low[0],
    center.high[0],
    center.low[0],
    coverage
  );
  const splitError = Math.abs(exact - (scaled.high + scaled.low));
  const float32Error = Math.abs(
    exact -
      Math.fround(
        Math.fround(center.high[0]) * (1 - coverage) + Math.fround(vertex.high[0]) * coverage
      )
  );
  expect(splitError).toBeLessThan(1e-9);
  expect(splitError).toBeLessThan(float32Error);
});
