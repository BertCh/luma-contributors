// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogDiagnostic} from '../../src/analysis/catalog-geojson';
import {
  computeMoransI,
  getSignificanceAttainability,
  isConstantList,
  readRateRows,
  runMoranPermutationTest
} from '../../src/curated/bindings/curated-binding-utils';
import {MORAN_TINY} from './curated-bindings-oracle-data';
import {makeLattice, recordBindingMargin} from './curated-bindings-test-support';

// Oracle: /usr/bin/python3 .typecheck/b1-oracles.py (MORAN_TINY): seven areas on a path, Moran's I
// from its definition and the folded two-sided p-value over all 5040 orderings of the values.

function makeRowStandardized(neighbors: readonly (readonly number[])[]) {
  return neighbors.map(list => list.map(() => 1 / list.length));
}

describe('computeMoransI', () => {
  it('matches the textbook definition to 1e-12 on the seven-area path', () => {
    const weights = makeRowStandardized(MORAN_TINY.neighbors);
    const value = computeMoransI(MORAN_TINY.values, MORAN_TINY.neighbors, weights);
    recordBindingMargin('moran tiny I abs error', Math.abs(value - MORAN_TINY.moransI));
    expect(Math.abs(value - MORAN_TINY.moransI)).toBeLessThan(1e-12);
  });

  it('is invariant to adding a constant and to scaling the values', () => {
    const weights = makeRowStandardized(MORAN_TINY.neighbors);
    const base = computeMoransI(MORAN_TINY.values, MORAN_TINY.neighbors, weights);
    const moved = computeMoransI(
      MORAN_TINY.values.map(v => 3 * v + 100),
      MORAN_TINY.neighbors,
      weights
    );
    expect(Math.abs(moved - base)).toBeLessThan(1e-12);
  });

  it('is positive for a smooth ramp and negative for an alternating pattern', () => {
    const neighbors = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]];
    const weights = makeRowStandardized(neighbors);
    expect(computeMoransI([1, 2, 3, 4, 5, 6], neighbors, weights)).toBeGreaterThan(0.5);
    expect(computeMoransI([1, 9, 1, 9, 1, 9], neighbors, weights)).toBeLessThan(-0.5);
  });
});

describe('runMoranPermutationTest', () => {
  const weights = makeRowStandardized(MORAN_TINY.neighbors);

  it('is deterministic for a seed and changes with the seed', () => {
    const first = runMoranPermutationTest(MORAN_TINY.values, MORAN_TINY.neighbors, weights, 199, 4);
    const second = runMoranPermutationTest(
      MORAN_TINY.values,
      MORAN_TINY.neighbors,
      weights,
      199,
      4
    );
    expect(second).toEqual(first);
    const other = runMoranPermutationTest(MORAN_TINY.values, MORAN_TINY.neighbors, weights, 199, 5);
    expect(other.simulatedMean).not.toBe(first.simulatedMean);
    expect(other.statistic).toBe(first.statistic);
  });

  it('is within Monte Carlo error of the exhaustive folded p-value on seven areas', () => {
    const permutations = 9999;
    const result = runMoranPermutationTest(
      MORAN_TINY.values,
      MORAN_TINY.neighbors,
      weights,
      permutations,
      12
    );
    const standardError = Math.sqrt(
      (MORAN_TINY.exactFoldedP * (1 - MORAN_TINY.exactFoldedP)) / permutations
    );
    const distance = Math.abs(result.pseudoPValue - MORAN_TINY.exactFoldedP);
    recordBindingMargin(
      'moran tiny pseudo p distance in standard errors',
      distance / standardError
    );
    expect(distance).toBeLessThan(4 * standardError + 1 / (permutations + 1));
    // The mean of the shuffled statistics is the exact permutation mean, -1 / (n - 1).
    expect(Math.abs(result.simulatedMean - MORAN_TINY.permutationMean)).toBeLessThan(0.01);
  });

  it('never goes below 1 / (permutations + 1) and reaches it for an extreme ramp', () => {
    const neighbors = Array.from({length: 12}, (_, index) =>
      [index - 1, index + 1].filter(other => other >= 0 && other < 12)
    );
    const ramp = Array.from({length: 12}, (_, index) => index);
    const result = runMoranPermutationTest(ramp, neighbors, makeRowStandardized(neighbors), 99, 1);
    expect(result.exceedanceCount).toBe(0);
    expect(result.pseudoPValue).toBe(1 / 100);
    const random = runMoranPermutationTest(MORAN_TINY.values, MORAN_TINY.neighbors, weights, 99, 1);
    expect(random.pseudoPValue).toBeGreaterThanOrEqual(1 / 100);
    expect(random.pseudoPValue).toBeLessThanOrEqual(1);
  });
});

describe('isConstantList', () => {
  it('detects constant, varying, one-value and empty lists', () => {
    expect(isConstantList([2, 2, 2])).toBe(true);
    expect(isConstantList([2, 2, 3])).toBe(false);
    expect(isConstantList([5])).toBe(true);
    expect(isConstantList([])).toBe(true);
  });
});

describe('readRateRows', () => {
  it('keeps usable rows and counts every reason for dropping one, by status and diagnostic', () => {
    const properties = [
      {cases: 5, population: 100},
      {cases: 'x', population: 100},
      {cases: 5, population: Number.NaN},
      {cases: 5, population: 0},
      {cases: 5, population: -3},
      {cases: -1, population: 10},
      {cases: null, population: 10},
      {cases: '7', population: '70'}
    ];
    const lattice = makeLattice(4, 2, index => properties[index]);
    lattice.features.push({
      id: 'no-geometry',
      geometry: null,
      properties: {cases: 1, population: 10}
    });
    const diagnostics: CatalogDiagnostic[] = [];
    const result = readRateRows(lattice, 'cases', 'population', diagnostics);
    expect(result.rows.map(row => row.sourceIndex)).toEqual([0, 7]);
    expect(result.rows[1].cases).toBe(7);
    expect(result.droppedCount).toBe(7);
    expect(result.statuses).toEqual([
      'analysed',
      'missing-value',
      'missing-value',
      'non-positive-population',
      'non-positive-population',
      'negative-cases',
      'missing-value',
      'analysed',
      'missing-geometry'
    ]);
    const counts = Object.fromEntries(diagnostics.map(item => [item.code, item.count]));
    expect(counts).toEqual({
      'missing-value': 3,
      'non-positive-population': 2,
      'negative-cases': 1,
      'missing-geometry': 1
    });
  });

  it('returns nothing and no diagnostic for an empty collection', () => {
    const diagnostics: CatalogDiagnostic[] = [];
    const result = readRateRows({type: 'FeatureCollection', features: []}, 'a', 'b', diagnostics);
    expect(result).toEqual({rows: [], droppedCount: 0, statuses: []});
    expect(diagnostics).toEqual([]);
  });
});

describe('getSignificanceAttainability at Sidak precision and count boundaries', () => {
  it('retains small-alpha precision (Python Decimal, 70 digits)', () => {
    // /usr/bin/python3: Decimal precision=70; ceil(1/(1-(1-Decimal('1e-9'))**(Decimal(1)/5000))-1).
    expect(getSignificanceAttainability('sidak', 1e-9, 5000, 999).permutationsNeeded).toBe(
      4999999997500
    );
    expect(getSignificanceAttainability('sidak', 1e-9, 1, 999).permutationsNeeded).toBe(999999999);
  });

  it('allows 51 tests but blocks 52 with 999 permutations at 0.05', () => {
    expect(getSignificanceAttainability('sidak', 0.05, 51, 999)).toMatchObject({
      attainable: true,
      permutationsNeeded: 994
    });
    expect(getSignificanceAttainability('sidak', 0.05, 52, 999)).toMatchObject({
      attainable: false,
      permutationsNeeded: 1014
    });
    // Exact reciprocal boundaries must not invent an additional permutation due to roundoff.
    expect(getSignificanceAttainability('sidak', 0.01, 1, 99).permutationsNeeded).toBe(99);
    expect(getSignificanceAttainability('bonferroni', 0.05, 50, 999).permutationsNeeded).toBe(999);
  });
});
