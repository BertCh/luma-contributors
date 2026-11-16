// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {
  getCatalogDefaultParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import {
  CURATED_RATE_CLUSTER_MAP_BINDING,
  publishRateClusterMap
} from '../../src/curated/bindings/curated-rate-cluster-map-binding';
import {
  compareRateClusterClasses,
  createRateClusterOverflowDataset
} from './curated-bindings-rates-fixtures';
import {computeEmpiricalBayesOracle} from '../../../analysis/test/gpu-spatial-analysis/rate-smoothing/empirical-bayes-oracle';
import {
  prepareGpuRateClusterMap,
  readGpuRateClusterComputation,
  type RateClusterReadback
} from '../../src/curated/bindings/curated-rate-cluster-map-gpu';
import {GPU_EMPIRICAL_BAYES_SUMMARY} from '@luma-contributors/analysis/gpu-spatial-analysis';
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

describe('curated-rate-cluster-map GPU preparation and publication contract', () => {
  const parameters = {
    ...getCatalogDefaultParameters(CURATED_RATE_CLUSTER_MAP_BINDING),
    ...FIELDS,
    ...PLANAR,
    permutations: 99
  };
  const areas = makeRateLattice(2, 2, [4, 20, 7, 40], [100, 200, 300, 400]);

  it('registers the genuine recipe path while retaining an explicit CPU reference', () => {
    expect(CURATED_RATE_CLUSTER_MAP_BINDING.engine).toBe('gpu-or-cpu');
    expect(CURATED_RATE_CLUSTER_MAP_BINDING.runGpu).toBeTypeOf('function');
    expect(CURATED_RATE_CLUSTER_MAP_BINDING.gpuContributorNames).toEqual([
      'GPUEmpiricalBayesRates',
      'GPUContiguityWeights',
      'GPUSpatialWeightsTransform',
      'GPULocalMoran',
      'GPULocalPermutationTest',
      'GPUMultipleTestingAdjustment'
    ]);
    expect(CURATED_RATE_CLUSTER_MAP_BINDING.limitation).toContain('doubled-smaller-tail');
    const reference = runCuratedBinding(CLUSTER_ID, parameters, {areas});
    expect(reference.engine).toBe('cpu-reference');
    expect(reference.dispatchedContributors).toEqual([]);
  });

  it('maps the bundled sample and a second dataset with renamed numeric roles before GPU allocation', () => {
    const analysis = getKeplerCuratedAnalysis('rate-cluster-map')!;
    const sample = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const second = makeLattice(3, 3, index => ({
      reported: index * 3 + 1,
      exposed: index * 100 + 500
    }));
    for (const [collection, casesProperty, populationProperty, coordinateSystem] of [
      [sample, 'cases', 'population', 'EPSG:4326'],
      [second, 'reported', 'exposed', 'planar']
    ] as const) {
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'mapped',
        label: 'Mapped districts',
        coordinateSystem
      });
      let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
      for (const [role, column] of [
        ['cases', casesProperty],
        ['population', populationProperty]
      ])
        mapping = updateKeplerCuratedAnalysisMapping(
          mapping,
          {slot: 'areas', role, columns: [column]},
          analysis,
          [profile]
        );
      const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
        parameterOverrides: {coordinateSystem}
      });
      expect(request.inputDatasetIds).toEqual({areas: 'mapped'});
      expect(request.parameters['casesProperty']).toBe(casesProperty);
      expect(request.parameters['populationProperty']).toBe(populationProperty);
      const prepared = prepareGpuRateClusterMap({areas: collection}, request.parameters);
      expect(prepared.rows).toHaveLength(collection.features.length);
      expect(prepared.polygons.featureCount).toBe(collection.features.length);
      expect(prepared.events.every(Number.isFinite)).toBe(true);
    }
  });

  it('packs canonical rows, multipart features and original identities without running a CPU statistic', () => {
    const multipart = structuredClone(areas);
    const geometry = multipart.features[2].geometry!;
    if (geometry.type !== 'Polygon') throw new Error('Expected polygon fixture');
    multipart.features[2].geometry = {type: 'MultiPolygon', coordinates: [geometry.coordinates]};
    multipart.features.push({
      ...makeIsland(),
      id: 'excluded',
      properties: {cases: 10, population: 0}
    });
    const prepared = prepareGpuRateClusterMap({areas: multipart}, parameters);
    // Column first, then row: canonical anchors (0.5,0.5), (0.5,1.5), (1.5,0.5), (1.5,1.5).
    expect(prepared.rows.map(row => row.sourceIndex)).toEqual([0, 2, 1, 3]);
    expect(Array.from(prepared.events)).toEqual([4, 7, 20, 40]);
    expect(Array.from(prepared.populations)).toEqual([100, 300, 200, 400]);
    expect(Array.from(prepared.polygons.featureOffsets)).toEqual([0, 1, 2, 3, 4]);
    expect(Array.from(prepared.polygons.ringOffsets)).toEqual([0, 4, 8, 12, 16]);
    expect(Array.from(prepared.polygons.positions.slice(0, 8))).toEqual([0, 0, 1, 0, 1, 1, 0, 1]);
    expect(prepared.statuses).toEqual([
      'analysed',
      'analysed',
      'analysed',
      'analysed',
      'non-positive-population'
    ]);
    expect(prepared.droppedCount).toBe(1);
    expect(prepared.diagnostics.find(item => item.code === 'non-positive-population')?.count).toBe(
      1
    );
    const reversed = prepareGpuRateClusterMap(
      {areas: {type: 'FeatureCollection', features: [...multipart.features].reverse()}},
      parameters
    );
    expect(reversed.rows.map(row => row.feature.id)).toEqual(
      prepared.rows.map(row => row.feature.id)
    );
    expect(reversed.polygons.positions).toEqual(prepared.polygons.positions);
  });

  function createReadback(): RateClusterReadback {
    const summary = new Float32Array(GPU_EMPIRICAL_BAYES_SUMMARY.length);
    summary[GPU_EMPIRICAL_BAYES_SUMMARY.count] = 4;
    summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate] = 0.1;
    summary[GPU_EMPIRICAL_BAYES_SUMMARY.clampedPriorVariance] = 0.01;
    return {
      summary,
      rawRates: new Float32Array([0.04, 0.02, 0.1, 0.1]),
      smoothedRates: new Float32Array([0.05, 0.03, 0.1, 0.1]),
      localI: new Float32Array([2, NaN, -1, 0.5]),
      spatialLag: new Float32Array([1, NaN, -0.5, 0.2]),
      pValues: new Float32Array([0.02, NaN, 0.2, 0.01]),
      adjustedPValues: new Float32Array([0.03, NaN, 0.2, 0.03]),
      quadrants: new Uint32Array([1, 0, 0, 4]),
      offsets: new Uint32Array([0, 1, 1, 2, 3]),
      weightsOverflow: 0,
      weightsRequiredCount: 3,
      permutationOverflow: 0
    };
  }

  it('preserves null islands, recipe quadrant names and the rate-moment shrinkage formula', () => {
    const prepared = prepareGpuRateClusterMap({areas}, parameters);
    const result = readGpuRateClusterComputation(prepared, createReadback());
    expect(result.clusters).toEqual(['HH', 'not-tested', 'not-significant', 'HL']);
    expect(result.pValues[1]).toBeNull();
    expect(result.localI[1]).toBeNull();
    expect(result.neighborCounts).toEqual([1, 0, 1, 1]);
    // Closed form: a / (a + m / population), using the GPU's published float32 moments.
    expect(result.shrinkages[0]).toBeCloseTo(0.01 / (0.01 + 0.1 / 100), 7);
    const constant = createReadback();
    constant.summary[GPU_EMPIRICAL_BAYES_SUMMARY.clampedPriorVariance] = 0;
    const untested = readGpuRateClusterComputation(prepared, constant);
    expect(untested.clusters).toEqual(new Array(4).fill('not-tested'));
    expect(untested.pValues).toEqual(new Array(4).fill(null));
    expect(untested.shrinkages).toEqual(new Array(4).fill(0));
  });

  it('refuses incomplete GPU output and non-finite moments before any publication', () => {
    const prepared = prepareGpuRateClusterMap({areas}, parameters);
    for (const invalid of [
      {weightsOverflow: 1},
      {weightsRequiredCount: prepared.neighborCapacity + 1},
      {permutationOverflow: 1},
      {localI: new Float32Array(3)},
      {offsets: new Uint32Array(4)},
      {summary: new Float32Array(2)},
      {smoothedRates: new Float32Array([NaN, 1, 2, 3])}
    ])
      expect(() =>
        readGpuRateClusterComputation(prepared, {...createReadback(), ...invalid})
      ).toThrow(/curated-rate-cluster-map /);
  });

  it('checks usable rows, work and float32 input limits before GPU allocation', () => {
    const tooFew = structuredClone(areas);
    tooFew.features[0].properties = {cases: 1, population: 0};
    expect(() => prepareGpuRateClusterMap({areas: tooFew}, parameters)).toThrow(/at least 4/);
    const tooLarge = structuredClone(areas);
    tooLarge.features[0].properties = {cases: 1e30, population: 1e30};
    expect(() => prepareGpuRateClusterMap({areas: tooLarge}, parameters)).toThrow(
      /GPU rate input limit/
    );
    const large = makeRateLattice(50, 42, new Array(2100).fill(5), new Array(2100).fill(100));
    expect(() =>
      prepareGpuRateClusterMap({areas: large}, {...parameters, permutations: 9999})
    ).toThrow(/areas times permutations/);
  });

  it('rejects cases above population identically before CPU execution or GPU allocation (finding 1)', async () => {
    const invalid = makeRateLattice(2, 2, [1001, 100, 200, 300], new Array(4).fill(1000));
    const message =
      /curated-rate-cluster-map casesProperty "cases" at source row 0 must not exceed populationProperty "population"/;
    expect(() => prepareGpuRateClusterMap({areas: invalid}, parameters)).toThrow(message);
    expect(() => runCuratedBinding(CLUSTER_ID, parameters, {areas: invalid})).toThrow(message);
    await expect(
      runCatalogOperationAsync(
        CURATED_RATE_CLUSTER_MAP_BINDING,
        parameters,
        {areas: invalid},
        {device: {} as never}
      )
    ).rejects.toThrow(message);
    // Equality remains a valid rate for both engines.
    invalid.features[0].properties!['cases'] = 1000;
    expect(() => prepareGpuRateClusterMap({areas: invalid}, parameters)).not.toThrow();
    expect(() => runCuratedBinding(CLUSTER_ID, parameters, {areas: invalid})).not.toThrow();
  });

  it('can execute the exact headless overflow fixture on its CPU fallback (finding 2)', () => {
    const overflow = createRateClusterOverflowDataset();
    expect(
      Math.max(...overflow.features.map(feature => Number(feature.properties!['reported'])))
    ).toBe(4225);
    expect(overflow.features.every(feature => Number(feature.properties!['exposed']) >= 4225)).toBe(
      true
    );
    const result = runCuratedBinding(
      CLUSTER_ID,
      {...parameters, casesProperty: 'reported', populationProperty: 'exposed'},
      {areas: overflow}
    );
    expect(result.tables[0].rowCount).toBe(66);
    expect(result.engine).toBe('cpu-reference');
    expect(result.summary['areaCount']).toBe(66);
  });

  it('ignores island-only rate variation and leaves undefined local inference untested (finding 3)', () => {
    const collection = makeRateLattice(2, 2, [100, 100, 100, 100], [1000, 1000, 1000, 1000]);
    collection.features.push({...makeIsland(), properties: {cases: 900, population: 1000}});
    const prepared = prepareGpuRateClusterMap({areas: collection}, parameters);
    const oracle = computeEmpiricalBayesOracle({
      events: prepared.events,
      populations: prepared.populations,
      clampPriorVariance: true
    });
    const summary = new Float32Array(GPU_EMPIRICAL_BAYES_SUMMARY.length);
    summary[GPU_EMPIRICAL_BAYES_SUMMARY.count] = 5;
    summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate] = oracle.pooledRate;
    summary[GPU_EMPIRICAL_BAYES_SUMMARY.clampedPriorVariance] = Math.max(0, oracle.priorVariance);
    const readback: RateClusterReadback = {
      ...createReadback(),
      summary,
      rawRates: Float32Array.from(oracle.raw),
      smoothedRates: Float32Array.from(oracle.smoothed),
      localI: new Float32Array(5).fill(NaN),
      spatialLag: new Float32Array([0, 0, 0, 0, NaN]),
      pValues: new Float32Array([1, 1, 1, 1, NaN]),
      adjustedPValues: new Float32Array([1, 1, 1, 1, NaN]),
      quadrants: new Uint32Array(5),
      offsets: new Uint32Array([0, 3, 6, 9, 12, 12]),
      weightsRequiredCount: 12
    };
    expect(new Set(readback.smoothedRates).size).toBe(2);
    expect(new Set(readback.smoothedRates.slice(0, 4)).size).toBe(1);
    const computation = readGpuRateClusterComputation(prepared, readback);
    const result = publishRateClusterMap({areas: collection}, parameters, [], computation, true);
    expect(result.summary['testedCount']).toBe(0);
    for (const column of ['local_i', 'spatial_lag', 'p_value', 'adjusted_p_value'])
      expect(result.tables[0].columns[column], column).toEqual(new Array(5).fill(null));
    expect(result.tables[0].columns['cluster']).toEqual(new Array(5).fill('not-tested'));
    // Island-only spread is insufficient even if rounding leaves finite local statistics.
    const finiteStatistics = readGpuRateClusterComputation(prepared, {
      ...readback,
      localI: new Float32Array([0, 0, 0, 0, NaN])
    });
    expect(finiteStatistics.pValues).toEqual(new Array(5).fill(null));
    expect(finiteStatistics.clusters).toEqual(new Array(5).fill('not-tested'));
    // A nonfinite statistic is untested even when other connected rates vary.
    const undefinedStatistic = createReadback();
    undefinedStatistic.localI[0] = NaN;
    const varying = readGpuRateClusterComputation(
      prepareGpuRateClusterMap({areas}, parameters),
      undefinedStatistic
    );
    expect(varying.pValues[0]).toBeNull();
    expect(varying.clusters[0]).toBe('not-tested');
  });

  it('uses the recipe float32 threshold at and immediately above 0.05 throughout publication (finding 4)', () => {
    const settings = {...parameters, significance: 0.05, permutations: 999};
    const prepared = prepareGpuRateClusterMap({areas}, settings);
    const readback = createReadback();
    readback.pValues[0] = 50 / 1000;
    readback.adjustedPValues[0] = 50 / 1000;
    readback.adjustedPValues[2] = Math.fround(0.050000004);
    readback.quadrants[2] = 4;
    expect(readback.adjustedPValues[0]).toBeGreaterThan(0.05);
    expect(readback.adjustedPValues[2]).toBeGreaterThan(Math.fround(0.05));
    const computation = readGpuRateClusterComputation(prepared, readback);
    const result = publishRateClusterMap({areas}, settings, [], computation, true);
    expect(result.summary['requestedSignificance']).toBe(0.05);
    expect(result.summary['effectiveSignificance']).toBe(Math.fround(0.05));
    expect(result.tables[0].columns['cluster']).toEqual([
      'HH',
      'not-significant',
      'not-tested',
      'HL'
    ]);
    // Computation rows are canonical, while publication rows are in source order.
    computation.adjustedPValues.forEach((value, row) => {
      if (value !== null && ['HH', 'LH', 'LL', 'HL'].includes(computation.clusters[row]))
        expect(value).toBeLessThanOrEqual(Number(result.summary['effectiveSignificance']));
    });
    expect(result.diagnostics.find(item => item.code === 'pseudo-p-values')?.message).toContain(
      `at most ${Math.fround(0.05)}`
    );
  });

  it('rounds doubled-tail permutation requirements only after applying the multiplier (finding 5)', () => {
    for (const permutations of [65, 66, 67]) {
      const settings = {...parameters, significance: 0.03, permutations, multipleTesting: 'none'};
      const computation = readGpuRateClusterComputation(
        prepareGpuRateClusterMap({areas}, settings),
        createReadback()
      );
      const result = publishRateClusterMap({areas}, settings, [], computation, true);
      expect(result.summary['permutationsNeeded']).toBe(66);
      expect(result.summary['significanceNotAttainable']).toBe(permutations < 66 ? 1 : 0);
    }
  });

  it('reports the unsupported Sidak GPU correction as an explicit catalog fallback', async () => {
    // The unsupported method is rejected before accessing this inert device.
    const result = await runCatalogOperationAsync(
      CURATED_RATE_CLUSTER_MAP_BINDING,
      {...parameters, multipleTesting: 'sidak'},
      {areas},
      {device: {} as never}
    );
    expect(result.engine).toBe('cpu-reference');
    expect(result.fallbackReason).toContain('Sidak correction');
    expect(result.diagnostics.map(item => item.code)).toContain('gpu-fallback');
  });
});

describe('rate-cluster GPU class comparison boundary qualification', () => {
  const effectiveSignificance = Math.fround(0.05);
  const createRow = (
    sourceIndex: number,
    oracleAdjustedPValue: number,
    adjustedPValue = oracleAdjustedPValue
  ) => ({
    sourceIndex,
    oracleAdjustedPValue,
    adjustedPValue,
    actualClass: adjustedPValue <= effectiveSignificance ? 'HH' : 'not-significant',
    significantClass: 'HH'
  });

  it('accepts 38 clear classes and reports all three near-threshold rows instead of imposing a count cap', async ({
    annotate
  }) => {
    const rows = Array.from({length: 38}, (_, index) => createRow(index, index % 2 ? 0.2 : 0.01));
    rows.push(
      createRow(38, 0.049, 0.051),
      createRow(39, 0.051, 0.049),
      createRow(40, 0.055, 0.055)
    );
    const report = compareRateClusterClasses(rows, effectiveSignificance);
    expect(report.mismatches).toEqual([]);
    expect(report.testedCount).toBe(41);
    expect(report.comparedSourceIndices).toEqual(Array.from({length: 38}, (_, index) => index));
    expect(report.nearThresholdRows.map(row => row.sourceIndex)).toEqual([38, 39, 40]);
    expect(report.nearThresholdRows[0]).toMatchObject({
      oracleAdjustedPValue: 0.049,
      adjustedPValue: 0.051,
      oracleClass: 'HH',
      actualClass: 'not-significant'
    });
    await annotate(JSON.stringify(report), 'rate-cluster-class-parity');
  });

  it('still rejects a wrong oracle class clearly away from the threshold', () => {
    const report = compareRateClusterClasses([createRow(3, 0.01, 0.2)], effectiveSignificance);
    expect(report.mismatches).toEqual(['oracle class[3] not-significant != HH']);
  });

  it('still rejects an incorrect published gate even inside the boundary band', () => {
    const row = {...createRow(4, 0.049, 0.051), actualClass: 'HH'};
    const report = compareRateClusterClasses([row], effectiveSignificance);
    expect(report.nearThresholdRows).toHaveLength(1);
    expect(report.mismatches).toEqual(['class/gate[4] HH != not-significant']);
  });

  it('uses the effective float32 cutoff and checks the gate of untested rows', () => {
    const boundary = createRow(5, Math.fround(0.05));
    const untested = {...createRow(6, NaN), adjustedPValue: null, actualClass: 'not-tested'};
    expect(
      compareRateClusterClasses([boundary, untested], effectiveSignificance).mismatches
    ).toEqual([]);
    const invalid = compareRateClusterClasses(
      [{...untested, actualClass: 'not-significant'}],
      effectiveSignificance
    );
    expect(invalid.mismatches).toEqual(['class/gate[6] not-significant != not-tested']);
    expect(invalid.testedCount).toBe(0);
  });
});
