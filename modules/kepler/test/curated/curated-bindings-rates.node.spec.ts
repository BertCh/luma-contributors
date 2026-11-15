// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {BENJAMINI_HOCHBERG, RATE_EXACT, RATE_LATTICE} from './curated-bindings-oracle-data';
import {
  getMaximumRelativeError,
  getNumbers,
  getResultTable,
  makeLattice,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Oracles: /usr/bin/python3 .typecheck/b1-oracles.py. RATE_LATTICE is a 5 by 5 lattice with
// Poisson cases (a block of four areas at 2.5 times the rate): Marshall's global empirical Bayes
// estimator with the prior variance clamped at zero, queen weights standardised by row, the local
// Moran statistic of the standardised smoothed rate, the expected counts under the overall rate,
// and the pooled spatial rate. RATE_EXACT is a 3 by 3 lattice whose conditional permutation
// p-values are computed by enumerating every k-subset of the other eight values.

const CLUSTER_ID = 'curated-rate-cluster-map';
const EXCESS_ID = 'curated-excess-risk';
const PLANAR = {coordinateSystem: 'planar'};
const FIELDS = {casesProperty: 'cases', populationProperty: 'population'};

function makeRateLattice(
  columns: number,
  rows: number,
  cases: readonly number[],
  population: readonly number[]
): CatalogFeatureCollection {
  return makeLattice(columns, rows, index => ({
    cases: cases[index],
    population: population[index]
  }));
}

function makeIsland(): CatalogFeature {
  return {
    id: 'island',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [100, 100],
          [101, 100],
          [101, 101],
          [100, 101],
          [100, 100]
        ]
      ]
    },
    properties: {cases: 40, population: 2000}
  };
}

/** Benjamini and Hochberg (1995): p times m over rank, then the running minimum from the largest p down. */
function adjustBenjaminiHochberg(pValues: readonly number[]): number[] {
  const count = pValues.length;
  const order = pValues.map((_, index) => index).sort((a, b) => pValues[a] - pValues[b]);
  const adjusted = new Array<number>(count);
  let running = 1;
  for (let rank = count - 1; rank >= 0; rank--) {
    running = Math.min(running, (pValues[order[rank]] * count) / (rank + 1));
    adjusted[order[rank]] = running;
  }
  return adjusted;
}

describe('test helper adjustBenjaminiHochberg', () => {
  it('agrees with scipy.stats.false_discovery_control on a vector with ties', () => {
    expect(
      getMaximumRelativeError(
        adjustBenjaminiHochberg(BENJAMINI_HOCHBERG.p),
        BENJAMINI_HOCHBERG.adjusted
      )
    ).toBeLessThan(1e-14);
  });
});

describe('curated-rate-cluster-map against NumPy', () => {
  const areas = makeRateLattice(5, 5, RATE_LATTICE.cases, RATE_LATTICE.population);
  const run = (parameters: Record<string, unknown> = {}) =>
    runCuratedBinding(CLUSTER_ID, {...FIELDS, ...PLANAR, permutations: 99, ...parameters}, {areas});

  it('reproduces the empirical Bayes rates to 1e-12, the local Moran statistic and lag to 1e-12', () => {
    const table = getResultTable(run(), 'rate-clusters');
    expect(table.rowCount).toBe(25);
    const checks: [string, readonly number[]][] = [
      ['raw_rate', RATE_LATTICE.rawRate],
      ['shrinkage', RATE_LATTICE.shrinkage],
      ['smoothed_rate', RATE_LATTICE.smoothedRate],
      ['local_i', RATE_LATTICE.localI],
      ['spatial_lag', RATE_LATTICE.spatialLag]
    ];
    for (const [column, expected] of checks) {
      const error = getMaximumRelativeError(getNumbers(table, column), expected, 1e-3);
      recordBindingMargin(`rate cluster map ${column} max relative error`, error);
      expect(error, column).toBeLessThan(1e-12);
    }
    expect(table.columns['neighbor_count']).toEqual(
      RATE_LATTICE.neighbors.map(list => list.length)
    );
    const summary = run().summary;
    expect(Math.abs((summary['globalRate'] as number) - RATE_LATTICE.globalRate)).toBeLessThan(
      1e-15
    );
    expect(
      Math.abs((summary['priorVariance'] as number) - RATE_LATTICE.priorVariance)
    ).toBeLessThan(1e-15);
  });

  it('gives pseudo p-values that are multiples of 1 / (permutations + 1), at least that, and repeat for a seed', () => {
    const first = getNumbers(getResultTable(run({seed: 7}), 'rate-clusters'), 'p_value');
    const again = getNumbers(getResultTable(run({seed: 7}), 'rate-clusters'), 'p_value');
    const other = getNumbers(getResultTable(run({seed: 8}), 'rate-clusters'), 'p_value');
    expect(again).toEqual(first);
    expect(other).not.toEqual(first);
    for (const p of first) {
      expect(p).toBeGreaterThanOrEqual(1 / 100 - 1e-15);
      expect(p).toBeLessThanOrEqual(1);
      expect(Math.abs(p * 100 - Math.round(p * 100))).toBeLessThan(1e-9);
    }
  });

  it('corrects the p-values with Benjamini-Hochberg over the tested areas by default', () => {
    const table = getResultTable(run({seed: 7}), 'rate-clusters');
    const pValues = getNumbers(table, 'p_value');
    const adjusted = getNumbers(table, 'adjusted_p_value');
    expect(getMaximumRelativeError(adjusted, adjustBenjaminiHochberg(pValues))).toBeLessThan(1e-12);
    adjusted.forEach((value, index) =>
      expect(value).toBeGreaterThanOrEqual(pValues[index] - 1e-15)
    );
    // 'none' leaves the pseudo p-values as they are, and never flags fewer areas than a correction.
    const uncorrected = getResultTable(run({seed: 7, multipleTesting: 'none'}), 'rate-clusters');
    expect(getNumbers(uncorrected, 'adjusted_p_value')).toEqual(pValues);
    const flagged = (t: typeof table) =>
      t.columns['cluster'].filter(c => ['HH', 'HL', 'LH', 'LL'].includes(String(c))).length;
    expect(flagged(table)).toBeLessThanOrEqual(flagged(uncorrected));
    for (const method of ['holm', 'bonferroni', 'sidak'])
      expect(
        getResultTable(run({seed: 7, multipleTesting: method}), 'rate-clusters').rowCount
      ).toBe(25);
  });
  // The gating of the cluster class by the corrected p-value is tested with data that has power in curated-bindings-fixes.node.spec.ts.
});

describe('curated-rate-cluster-map permutation inference', () => {
  it('is within Monte Carlo error of the exact conditional permutation p-value on a 3 by 3 lattice', () => {
    const areas = makeRateLattice(3, 3, RATE_EXACT.cases, RATE_EXACT.population);
    const permutations = 9999;
    const result = runCuratedBinding(
      CLUSTER_ID,
      {...FIELDS, ...PLANAR, permutations, seed: 3, significance: 0.5},
      {areas}
    );
    const table = getResultTable(result, 'rate-clusters');
    expect(
      getMaximumRelativeError(getNumbers(table, 'local_i'), RATE_EXACT.localI, 1e-3)
    ).toBeLessThan(1e-12);
    expect(
      getMaximumRelativeError(getNumbers(table, 'smoothed_rate'), RATE_EXACT.smoothedRate)
    ).toBeLessThan(1e-12);
    const pValues = getNumbers(table, 'p_value');
    let worstStandardErrors = 0;
    pValues.forEach((p, index) => {
      const exact = RATE_EXACT.exactPValue[index];
      const standardError = Math.sqrt((exact * (1 - exact)) / permutations);
      const distance = Math.abs(p - exact);
      worstStandardErrors = Math.max(worstStandardErrors, distance / Math.max(standardError, 1e-9));
      expect(distance, `row ${index}`).toBeLessThan(4 * standardError + 1 / (permutations + 1));
    });
    recordBindingMargin(
      'rate cluster exact p worst distance in standard errors',
      worstStandardErrors
    );
  });
});

describe('curated-rate-cluster-map declared behaviour', () => {
  const baseCases = [...RATE_LATTICE.cases];
  const basePopulation = [...RATE_LATTICE.population];
  const run = (areas: CatalogFeatureCollection, parameters: Record<string, unknown> = {}) =>
    runCuratedBinding(
      CLUSTER_ID,
      {...FIELDS, ...PLANAR, permutations: 99, seed: 7, ...parameters},
      {areas}
    );

  it('lists every input row, with a status and diagnostic for each reason a row is not analysed', () => {
    const areas = makeRateLattice(5, 5, baseCases, basePopulation);
    areas.features[0].properties = {cases: null, population: 100};
    areas.features[1].properties = {cases: Number.POSITIVE_INFINITY, population: 100};
    areas.features[2].properties = {cases: 4, population: 0};
    areas.features[3].properties = {cases: 4, population: -10};
    areas.features[4].properties = {cases: -2, population: 10};
    areas.features[5].properties = {cases: 'abc', population: 10};
    const result = run(areas);
    const table = getResultTable(result, 'rate-clusters');
    expect(table.rowCount).toBe(25);
    expect(table.columns['status'].slice(0, 6)).toEqual([
      'missing-value',
      'missing-value',
      'non-positive-population',
      'non-positive-population',
      'negative-cases',
      'missing-value'
    ]);
    for (let index = 0; index < 6; index++) {
      expect(table.columns['cluster'][index]).toBe('not-tested');
      expect(table.columns['p_value'][index]).toBeNull();
      expect(table.columns['smoothed_rate'][index]).toBeNull();
    }
    const counts = Object.fromEntries(result.diagnostics.map(item => [item.code, item.count]));
    expect(counts['missing-value']).toBe(3);
    expect(counts['non-positive-population']).toBe(2);
    expect(counts['negative-cases']).toBe(1);
    expect(result.summary['droppedCount']).toBe(6);
    expect(result.summary['areaCount']).toBe(19);
    // The dropped areas are not neighbours: the usable areas are tested among themselves.
    expect(Number(result.summary['testedCount'])).toBeGreaterThan(0);
  });

  it('leaves an island untested and out of the corrected family, and says so', () => {
    const areas = makeRateLattice(5, 5, baseCases, basePopulation);
    areas.features.push(makeIsland());
    const result = run(areas);
    const table = getResultTable(result, 'rate-clusters');
    expect(table.rowCount).toBe(26);
    expect(table.columns['neighbor_count'][25]).toBe(0);
    expect(table.columns['cluster'][25]).toBe('not-tested');
    expect(table.columns['p_value'][25]).toBeNull();
    expect(table.columns['adjusted_p_value'][25]).toBeNull();
    expect(result.diagnostics.find(item => item.code === 'island-features')?.count).toBe(1);
    expect(result.summary['testedCount']).toBe(25);
    const tested = getNumbers(table, 'p_value').slice(0, 25);
    expect(
      getMaximumRelativeError(
        getNumbers(table, 'adjusted_p_value').slice(0, 25),
        adjustBenjaminiHochberg(tested)
      )
    ).toBeLessThan(1e-12);
  });

  it('tests nothing, with a diagnostic, when the smoothed rates are all equal (no cases at all)', () => {
    const areas = makeRateLattice(4, 4, new Array(16).fill(0), new Array(16).fill(1000));
    const result = run(areas);
    expect(result.diagnostics.map(item => item.code)).toContain('constant-smoothed-rate');
    const table = getResultTable(result, 'rate-clusters');
    expect(new Set(table.columns['cluster'])).toEqual(new Set(['not-tested']));
    expect(result.summary['testedCount']).toBe(0);
  });

  it('shrinks every area to the overall rate, and tests nothing, when the rates vary less than chance', () => {
    // Identical rates: the prior variance estimate is negative, is clamped at zero, and the smoothed rates are equal.
    const areas = makeRateLattice(4, 4, new Array(16).fill(10), new Array(16).fill(1000));
    const result = run(areas);
    expect(result.summary['priorVariance']).toBe(0);
    expect(new Set(getNumbers(getResultTable(result, 'rate-clusters'), 'shrinkage'))).toEqual(
      new Set([0])
    );
    expect(result.diagnostics.map(item => item.code)).toContain('constant-smoothed-rate');
  });

  it('throws a named error for fewer than four usable areas, unnamed columns and too much permutation work', () => {
    const three = makeRateLattice(3, 1, [1, 2, 3], [100, 100, 100]);
    expect(() => run(three)).toThrow(/curated-rate-cluster-map: 3 usable areas; at least 4/);
    const areas = makeRateLattice(5, 5, baseCases, basePopulation);
    expect(() =>
      runCuratedBinding(CLUSTER_ID, {...PLANAR, populationProperty: 'population'}, {areas})
    ).toThrow(/curated-rate-cluster-map: parameter casesProperty must name a column/);
    expect(() => run(areas, {permutations: 98})).toThrow(/out of range/);
    const large = makeRateLattice(50, 42, new Array(2100).fill(5), new Array(2100).fill(100));
    expect(() => run(large, {permutations: 9999})).toThrow(
      /2100 areas times 9999 permutations exceeds/
    );
  });

  it('rejects point input before any work', () => {
    const points: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          id: 'p',
          geometry: {type: 'Point', coordinates: [0, 0]},
          properties: {cases: 1, population: 10}
        }
      ]
    };
    expect(() => run(points)).toThrow(/geometry kind not accepted/);
  });
});

describe('curated-excess-risk against NumPy', () => {
  const areas = makeRateLattice(5, 5, RATE_LATTICE.cases, RATE_LATTICE.population);
  const run = (parameters: Record<string, unknown> = {}, input = areas) =>
    runCuratedBinding(EXCESS_ID, {...FIELDS, ...PLANAR, ...parameters}, {areas: input});

  it('reproduces expected cases, excess risk, smoothed and spatial rates to 1e-12', () => {
    const result = run();
    const table = getResultTable(result, 'excess-risk');
    const scaled = (values: readonly number[]) => values.map(value => value * 1000);
    const checks: [string, readonly number[]][] = [
      ['expected_cases', RATE_LATTICE.expectedCases],
      ['excess_risk', RATE_LATTICE.excessRisk],
      ['raw_rate', scaled(RATE_LATTICE.rawRate)],
      ['smoothed_rate', scaled(RATE_LATTICE.smoothedRate)],
      ['spatial_rate', RATE_LATTICE.spatialRate]
    ];
    for (const [column, expected] of checks) {
      const error = getMaximumRelativeError(getNumbers(table, column), expected);
      recordBindingMargin(`excess risk ${column} max relative error`, error);
      expect(error, column).toBeLessThan(1e-12);
    }
    const expectedTotal = getNumbers(table, 'expected_cases').reduce(
      (sum, value) => sum + value,
      0
    );
    expect(Math.abs(expectedTotal - Number(result.summary['totalCases']))).toBeLessThan(1e-9);
  });

  it('scales the rates by rateScale and leaves the dimensionless excess risk alone', () => {
    const one = getResultTable(run({rateScale: 1}), 'excess-risk');
    const thousand = getResultTable(run({rateScale: 1000}), 'excess-risk');
    expect(
      getMaximumRelativeError(
        getNumbers(thousand, 'spatial_rate'),
        getNumbers(one, 'spatial_rate').map(v => v * 1000)
      )
    ).toBeLessThan(1e-14);
    expect(getNumbers(thousand, 'excess_risk')).toEqual(getNumbers(one, 'excess_risk'));
  });

  it('puts the planted block above one and the rest of the lattice mostly below it', () => {
    const excess = getNumbers(getResultTable(run(), 'excess-risk'), 'excess_risk');
    const block = [1 * 5 + 1, 1 * 5 + 2, 2 * 5 + 1, 2 * 5 + 2];
    const outside = excess.filter((_, index) => !block.includes(index));
    recordBindingMargin(
      'excess risk smallest value in the planted block',
      Math.min(...block.map(i => excess[i]))
    );
    for (const index of block) expect(excess[index]).toBeGreaterThan(1.5);
    expect(outside.filter(value => value < 1).length).toBeGreaterThanOrEqual(15);
  });

  it('gives an island no spatial rate by default, its own rate with own-rate, and counts it', () => {
    const withIsland = makeRateLattice(5, 5, RATE_LATTICE.cases, RATE_LATTICE.population);
    withIsland.features.push(makeIsland());
    const result = run({}, withIsland);
    const table = getResultTable(result, 'excess-risk');
    expect(table.columns['spatial_rate'][25]).toBeNull();
    expect(table.columns['neighbor_count'][25]).toBe(0);
    expect(result.diagnostics.find(item => item.code === 'island-features')?.count).toBe(1);
    const own = getResultTable(run({islandPolicy: 'own-rate'}, withIsland), 'excess-risk');
    expect(Math.abs(Number(own.columns['spatial_rate'][25]) - (1000 * 40) / 2000)).toBeLessThan(
      1e-12
    );
  });

  it('lists dropped rows with their status, and reports no excess risk when there are no cases', () => {
    const dropped = makeRateLattice(5, 5, RATE_LATTICE.cases, RATE_LATTICE.population);
    dropped.features[3].properties = {cases: 'x', population: 10};
    dropped.features[4].properties = {cases: 3, population: 0};
    const result = run({}, dropped);
    const table = getResultTable(result, 'excess-risk');
    expect(table.columns['status'].slice(3, 5)).toEqual([
      'missing-value',
      'non-positive-population'
    ]);
    expect(table.columns['excess_risk'][3]).toBeNull();
    expect(result.summary['droppedCount']).toBe(2);
    const empty = run({}, makeRateLattice(3, 3, new Array(9).fill(0), new Array(9).fill(100)));
    expect(empty.diagnostics.map(item => item.code)).toContain('no-cases');
    expect(new Set(getResultTable(empty, 'excess-risk').columns['excess_risk'])).toEqual(
      new Set([null])
    );
  });

  it('throws a named error for fewer than three usable areas', () => {
    expect(() => run({}, makeRateLattice(2, 1, [1, 2], [10, 10]))).toThrow(
      /curated-excess-risk: 2 usable areas; at least 3/
    );
  });
});

it('reports that the absolute and doubled-tail p-value definitions can cross in either direction', () => {
  // /usr/bin/python3, NumPy + itertools.combinations: enumerate each conditional subset of
  // the other eight standardized values [0,0,0,0,0,1,1,10,100] on a queen 3x3 lattice.
  // Area 1: (absolute, doubled-tail) = (1/28, 1/14); area 8: (55/56, 11/28).
  const cases = [0, 0, 0, 0, 0, 1, 1, 10, 100];
  const result = runCuratedBinding(
    CLUSTER_ID,
    {...PLANAR, ...FIELDS, permutations: 9999, seed: 43, multipleTesting: 'none'},
    {areas: makeRateLattice(3, 3, cases, new Array(9).fill(1000))}
  );
  const table = getResultTable(result, 'rate-clusters');
  expect(Math.abs(Number(table.columns['p_value'][1]) - 1 / 28)).toBeLessThan(0.025);
  expect(Number(table.columns['p_value'][1])).toBeLessThan(1 / 14);
  expect(Math.abs(Number(table.columns['p_value'][8]) - 55 / 56)).toBeLessThan(0.025);
  expect(Number(table.columns['p_value'][8])).toBeGreaterThan(11 / 28);
  expect(
    result.diagnostics.find(diagnostic => diagnostic.code === 'pseudo-p-values')?.message
  ).toContain('either may give the smaller p-value');
});
