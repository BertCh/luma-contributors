// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  MAXIMUM_KEPLER_IDW_SOURCE_ROWS,
  MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS,
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS,
  parseKeplerKrigingSettings,
  type KeplerKrigingSettings
} from '../src/recipes/analysis-recipe';
import {getKeplerIndexGridSize, getKeplerNeighborGridSize} from '../src/analysis/analysis-scaling';
import {
  estimateKeplerEmpiricalVariogram,
  fitKeplerVariogram,
  getKrigingConfidenceInterval,
  getStandardNormalQuantile,
  summarizeKrigingCrossValidation
} from '../src/analysis/kriging-variogram-fit';

const krigingSettings: KeplerKrigingSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: [-1, -1, 1, 1], width: 4, height: 4},
  searchRadiusMeters: 200000,
  neighborCount: 8,
  minimumNeighbors: 3,
  variogram: {model: 'spherical', nugget: 0.1, sill: 10, rangeMeters: 100000}
};

test('row, cell and slot caps are lifted to uint32 and device-memory limits', () => {
  expect(MAXIMUM_KEPLER_IDW_SOURCE_ROWS).toBe(2 ** 24 - 1);
  expect(MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS).toBe(2 ** 24 - 1);
  expect(MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS).toBe(2 ** 24 - 1);
  const large = parseKeplerKrigingSettings({
    ...krigingSettings,
    grid: {bounds: [-1, -1, 1, 1], width: 4096, height: 4096}
  });
  expect(large.grid.width * large.grid.height).toBe(2 ** 24);
  expect(() =>
    parseKeplerKrigingSettings({
      ...krigingSettings,
      grid: {bounds: [-1, -1, 1, 1], width: 4097, height: 4097}
    })
  ).toThrow();
});

test('spatial index shapes keep legacy sizes for small inputs and grow with source count', () => {
  expect(getKeplerIndexGridSize(4096)).toEqual([64, 32]);
  expect(getKeplerNeighborGridSize(16384)).toEqual([32, 16]);
  let previous = 0;
  for (const rowCount of [8192, 100_000, 1_000_000, 16_000_000]) {
    const [columns, rows] = getKeplerIndexGridSize(rowCount);
    expect(columns).toBe(rows * 2);
    expect(columns * rows).toBeGreaterThan(previous);
    expect(columns * rows).toBeLessThanOrEqual(2 ** 22 + 4 * rows);
    previous = columns * rows;
  }
  expect(getKeplerNeighborGridSize(1_000_000)[0]).toBeGreaterThan(32);
});

test('kriging recipes carry confidence level, cross-validation and fit provenance', () => {
  const parsed = parseKeplerKrigingSettings({
    ...krigingSettings,
    confidenceLevel: 0.9,
    crossValidation: true,
    variogramFit: {
      method: 'weighted-least-squares',
      weighting: 'cressie',
      binCount: 20,
      pairCount: 1000,
      maximumLagMeters: 50000,
      criterion: 1.5
    }
  });
  expect(parsed.confidenceLevel).toBe(0.9);
  expect(parsed.crossValidation).toBe(true);
  expect(Object.isFrozen(parsed.variogramFit)).toBe(true);
  expect(parseKeplerKrigingSettings(parsed)).toEqual(parsed);
  for (const override of [
    {confidenceLevel: 1},
    {confidenceLevel: 0},
    {crossValidation: 'yes'},
    {variogramFit: {method: 'ordinary', weighting: 'cressie'}}
  ])
    expect(() => parseKeplerKrigingSettings({...krigingSettings, ...override})).toThrow();
});

function makeField(count: number, order?: number[]) {
  const longitudes = new Float32Array(count);
  const latitudes = new Float32Array(count);
  const values = new Float32Array(count);
  const side = Math.ceil(Math.sqrt(count));
  for (let row = 0; row < count; row++) {
    const x = (row % side) / side;
    const y = Math.floor(row / side) / side;
    longitudes[row] = -0.5 + x;
    latitudes[row] = -0.5 + y;
    // A smooth field with a short-range wiggle: semivariance rises with distance.
    values[row] = 10 * Math.sin(3 * x) * Math.cos(2 * y) + Math.sin(40 * x + 7 * y);
  }
  if (!order) return {longitudes, latitudes, values};
  return {
    longitudes: Float32Array.from(order, row => longitudes[row]),
    latitudes: Float32Array.from(order, row => latitudes[row]),
    values: Float32Array.from(order, row => values[row])
  };
}

test('empirical variogram and automatic fit select a valid replayable model', () => {
  const field = makeField(900);
  const empirical = estimateKeplerEmpiricalVariogram(
    field.longitudes,
    field.latitudes,
    field.values,
    {binCount: 16}
  );
  expect(empirical.exhaustive).toBe(true);
  expect(empirical.validRowCount).toBe(900);
  expect(empirical.accumulatedPairCount).toBeGreaterThan(10000);
  const populated = Array.from(empirical.pairCounts).filter(count => count > 0).length;
  expect(populated).toBeGreaterThanOrEqual(8);
  const fit = fitKeplerVariogram(empirical);
  expect(['spherical', 'exponential', 'gaussian']).toContain(fit.model);
  expect(fit.nugget).toBeGreaterThanOrEqual(0);
  expect(fit.sill).toBeGreaterThanOrEqual(0);
  expect(fit.rangeMeters).toBeGreaterThan(0);
  expect(fit.candidates.length).toBeGreaterThanOrEqual(1);
  // Candidates are ranked and the winner has the smallest criterion.
  expect(fit.candidates[0].criterion).toBeLessThanOrEqual(fit.candidates.at(-1)!.criterion);
  // Float32 rounding makes the fit a valid kriging variogram that parses unchanged.
  const parsed = parseKeplerKrigingSettings({
    ...krigingSettings,
    variogram: {
      model: fit.model,
      nugget: fit.nugget,
      sill: fit.sill,
      rangeMeters: fit.rangeMeters
    }
  });
  expect(parsed.variogram.rangeMeters).toBe(fit.rangeMeters);
});

test('thinned pair sampling is deterministic and independent of row order', () => {
  const count = 900;
  const field = makeField(count);
  const reversed = makeField(
    count,
    Array.from({length: count}, (_, row) => count - 1 - row)
  );
  const options = {binCount: 12, maximumPairs: 5000, seed: 7};
  const first = estimateKeplerEmpiricalVariogram(
    field.longitudes,
    field.latitudes,
    field.values,
    options
  );
  const again = estimateKeplerEmpiricalVariogram(
    field.longitudes,
    field.latitudes,
    field.values,
    options
  );
  expect(first.exhaustive).toBe(false);
  expect(Array.from(again.pairCounts)).toEqual(Array.from(first.pairCounts));
  expect(first.accumulatedPairCount).toBeLessThan(first.candidatePairCount);
  // The hash depends on row indices, so a reordered table is a different sample of the same size class.
  const flipped = estimateKeplerEmpiricalVariogram(
    reversed.longitudes,
    reversed.latitudes,
    reversed.values,
    options
  );
  expect(flipped.accumulatedPairCount).toBeGreaterThan(first.accumulatedPairCount * 0.5);
  expect(flipped.accumulatedPairCount).toBeLessThan(first.accumulatedPairCount * 1.5);
});

test('confidence intervals and cross-validation diagnostics follow normal theory', () => {
  expect(getStandardNormalQuantile(0.975)).toBeCloseTo(1.959964, 5);
  expect(getStandardNormalQuantile(0.5)).toBeCloseTo(0, 9);
  expect(getStandardNormalQuantile(0.001)).toBeCloseTo(-3.090232, 5);
  const interval = getKrigingConfidenceInterval([10, 20, NaN], [4, 0, 1], 0.95);
  expect(interval.multiplier).toBeCloseTo(1.959964, 5);
  expect(interval.lower[0]).toBeCloseTo(10 - 2 * 1.959964, 4);
  expect(interval.upper[0]).toBeCloseTo(10 + 2 * 1.959964, 4);
  expect(interval.lower[1]).toBe(20);
  expect(Number.isNaN(interval.lower[2])).toBe(true);
  expect(() => getKrigingConfidenceInterval([1], [1], 1)).toThrow();
  const summary = summarizeKrigingCrossValidation(
    [1, 2, 3, 4, 5],
    [1.5, 2, 2, 4.5, NaN],
    [0.25, 1, 1, 0.25, 1],
    0.95
  );
  expect(summary.count).toBe(4);
  expect(summary.meanError).toBeCloseTo((0.5 + 0 - 1 + 0.5) / 4, 9);
  expect(summary.rootMeanSquareError).toBeCloseTo(Math.sqrt((0.25 + 0 + 1 + 0.25) / 4), 9);
  expect(summary.rootMeanSquareStandardizedError).toBeCloseTo(Math.sqrt((1 + 0 + 1 + 1) / 4), 9);
  expect(summary.coverage).toBe(1);
});

test('matches the neighbor lattice aspect to projected bounds within the cell budget', () => {
  const wide = getKeplerNeighborGridSize(1_000_000, [0, 0, 400000, 100000]);
  expect(wide[0] / wide[1]).toBeGreaterThan(3.5);
  expect(wide[0] / wide[1]).toBeLessThan(4.5);
  expect(wide[0] * wide[1]).toBeLessThanOrEqual(2 ** 22);
  const square = getKeplerNeighborGridSize(1_000_000, [0, 0, 1000, 1000]);
  expect(square[0]).toBe(square[1]);
  // Degenerate or extreme extents fall back to the 2:1 lattice; small inputs keep the legacy one.
  expect(getKeplerNeighborGridSize(1_000_000, [0, 0, 0, 10])).toEqual(
    getKeplerNeighborGridSize(1_000_000)
  );
  expect(getKeplerNeighborGridSize(1000, [0, 0, 400000, 100000])).toEqual([32, 16]);
});
