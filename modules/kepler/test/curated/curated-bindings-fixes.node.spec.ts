// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {getFeatureAnchor, makeSeededGenerator} from '../../src/analysis/catalog-platform3-b-common';
import {makeLocalFrame} from '../../src/analysis/catalog-local-frame';
import {buildAnalysisSet} from '../../src/analysis/catalog-platform3-b-statistics';
import {measureDistance} from '../../src/analysis/catalog-geojson';
import {fitOrdinaryLeastSquares} from '../../src/analysis/catalog-platform4-regression';
import {fitStandardizedLeastSquares} from '../../src/curated/bindings/curated-regression-with-diagnostics-binding';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {
  getCanonicalOrder,
  getSignificanceAttainability,
  runCanonicalMoranPermutationTest,
  runMoranPermutationTest,
  runConditionalLocalPermutationTest
} from '../../src/curated/bindings/curated-binding-utils';
import {KEPLER_CURATED_BINDINGS} from '../../src/curated/bindings/index';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {CURATED_RUNTIME_CAUTION_SUMMARY_KEYS} from '../../src/descriptors/descriptors-curated';
import {
  CORRELOGRAM_LATTICE,
  KRIGING_POINTS,
  RATE_LATTICE,
  REGRESSION_LATTICE,
  REGRESSION_SCALED
} from './curated-bindings-oracle-data';
import {
  getMaximumRelativeError,
  getNumbers,
  getResultTable,
  makeLattice,
  makePoints,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Tests of the fix round: scale-free regression, the information criteria convention, row-order
// independent permutation inference, attainability of corrections, model-fit warnings and the
// DBSCAN boundaries. Oracles: /usr/bin/python3 .typecheck/b1-oracles.py.

const PLANAR = {coordinateSystem: 'planar'};

/** A fixed shuffle of the features (Fisher-Yates with a seeded generator). */
function shuffleFeatures(
  collection: CatalogFeatureCollection,
  seed: number
): CatalogFeatureCollection {
  const random = makeSeededGenerator(seed);
  const features = [...collection.features];
  for (let index = features.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [features[index], features[swap]] = [features[swap], features[index]];
  }
  return {type: 'FeatureCollection', features};
}

/** Row of a table by feature id, for tables that carry `source_id`. */
function getRowsById(result: ReturnType<typeof runCuratedBinding>, tableId: string) {
  const table = getResultTable(result, tableId);
  const rows = new Map<string, Record<string, unknown>>();
  (table.columns['source_id'] as string[]).forEach((id, row) => {
    const entry: Record<string, unknown> = {};
    for (const name of Object.keys(table.columns)) entry[name] = table.columns[name][row];
    rows.set(id, entry);
  });
  return rows;
}

function expectRowsMatch(
  first: Map<string, Record<string, unknown>>,
  second: Map<string, Record<string, unknown>>,
  exactColumns: readonly string[],
  tolerance: number
) {
  expect(second.size).toBe(first.size);
  for (const [id, row] of first) {
    const other = second.get(id)!;
    for (const [name, value] of Object.entries(row)) {
      if (name === 'source_index') continue;
      if (typeof value === 'number' && !exactColumns.includes(name))
        expect(Math.abs((other[name] as number) - value), `${id} ${name}`).toBeLessThanOrEqual(
          tolerance * Math.max(1, Math.abs(value))
        );
      else expect(other[name], `${id} ${name}`).toEqual(value);
    }
  }
}

describe('regression: predictors of very different scales', () => {
  const run = (areas: CatalogFeatureCollection, names = 'income,share') =>
    runCuratedBinding(
      'curated-regression-with-diagnostics',
      {responseProperty: 'y', predictorProperties: names, permutations: 99, seed: 1, ...PLANAR},
      {areas}
    );

  it('fits income in dollars beside a share (correlation 0.1) as NumPy does, to 1e-9', () => {
    const areas = makeLattice(6, 6, index => ({
      y: REGRESSION_SCALED.y[index],
      income: REGRESSION_SCALED.income[index],
      share: REGRESSION_SCALED.share[index]
    }));
    const table = getResultTable(run(areas), 'coefficients');
    for (const [column, expected] of [
      ['estimate', REGRESSION_SCALED.coefficients],
      ['standard_error', REGRESSION_SCALED.standardErrors],
      ['t_statistic', REGRESSION_SCALED.tStatistics],
      ['p_value', REGRESSION_SCALED.pValues]
    ] as const) {
      const error = getMaximumRelativeError(getNumbers(table, column), expected);
      recordBindingMargin(`regression scaled ${column} max relative error`, error);
      expect(error, column).toBeLessThan(1e-9);
    }
  });

  it('gives the exact transform of the unit-scale fit for scale ratios 1 to 1e12 (every statistic)', () => {
    const base = getResultTable(
      runCuratedBinding(
        'curated-regression-with-diagnostics',
        {responseProperty: 'y', predictorProperties: 'x1,x2', permutations: 99, seed: 1, ...PLANAR},
        {
          areas: makeLattice(6, 6, i => ({
            y: REGRESSION_LATTICE.y[i],
            x1: REGRESSION_LATTICE.x1[i],
            x2: REGRESSION_LATTICE.x2[i]
          }))
        }
      ),
      'coefficients'
    );
    for (const ratio of [1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]) {
      const areas = makeLattice(6, 6, i => ({
        y: REGRESSION_LATTICE.y[i],
        x1: REGRESSION_LATTICE.x1[i],
        x2: REGRESSION_LATTICE.x2[i] * ratio
      }));
      const result = runCuratedBinding(
        'curated-regression-with-diagnostics',
        {responseProperty: 'y', predictorProperties: 'x1,x2', permutations: 99, seed: 1, ...PLANAR},
        {areas}
      );
      const table = getResultTable(result, 'coefficients');
      // Scaling a predictor by r divides its coefficient and standard error by r and changes nothing else.
      const scale = [1, 1, ratio];
      for (const [column, divide] of [
        ['estimate', true],
        ['standard_error', true],
        ['t_statistic', false],
        ['p_value', false]
      ] as const) {
        const expected = getNumbers(base, column).map((value, index) =>
          divide ? value / scale[index] : value
        );
        const error = getMaximumRelativeError(getNumbers(table, column), expected);
        recordBindingMargin(`regression scale ratio ${ratio} ${column} max relative error`, error);
        expect(error, `ratio ${ratio} ${column}`).toBeLessThan(1e-9);
      }
      expect(
        Math.abs(Number(result.summary['rSquared']) - REGRESSION_LATTICE.rSquared)
      ).toBeLessThan(1e-12);
    }
  });

  it('still refuses a constant predictor and an exactly collinear one, whatever the scale', () => {
    const constant = makeLattice(6, 6, i => ({
      y: REGRESSION_LATTICE.y[i],
      income: 5e4,
      share: REGRESSION_SCALED.share[i]
    }));
    expect(() => run(constant)).toThrow(/the predictor income is constant/);
    const collinear = makeLattice(6, 6, i => ({
      y: REGRESSION_LATTICE.y[i],
      income: REGRESSION_SCALED.income[i],
      share: REGRESSION_SCALED.income[i] * 1e-9 + 3
    }));
    expect(() => run(collinear)).toThrow(/predictors are collinear: share is an almost exact/);
  });

  it('counts the error variance in AIC and AICc, as R does, and says so', () => {
    const result = runCuratedBinding(
      'curated-regression-with-diagnostics',
      {responseProperty: 'y', predictorProperties: 'x1,x2', permutations: 99, ...PLANAR},
      {
        areas: makeLattice(6, 6, i => ({
          y: REGRESSION_LATTICE.y[i],
          x1: REGRESSION_LATTICE.x1[i],
          x2: REGRESSION_LATTICE.x2[i]
        }))
      }
    );
    const table = getResultTable(result, 'diagnostics');
    const names = table.columns['diagnostic'] as string[];
    // k = 3 coefficients + 1; the oracle is 2 k - 2 logL (+ 2 k (k + 1) / (n - k - 1)) with k = 4.
    expect(
      Math.abs((table.columns['value'][names.indexOf('aic')] as number) - REGRESSION_LATTICE.aic)
    ).toBeLessThan(1e-9);
    expect(
      Math.abs((table.columns['value'][names.indexOf('aicc')] as number) - REGRESSION_LATTICE.aicc)
    ).toBeLessThan(1e-9);
    expect(String(table.columns['note'][names.indexOf('aicc')])).toMatch(
      /error variance.*R AIC\(lm\).*exactly 2 lower/
    );
    expect(result.summary['aicc']).toBe(table.columns['value'][names.indexOf('aicc')]);
    expect(String(result.summary['informationCriterionConvention'])).toMatch(
      /statsmodels and spreg/
    );
  });
});

describe('permutation inference does not depend on the order of the rows', () => {
  const lattice = () =>
    makeLattice(5, 5, i => ({
      cases: RATE_LATTICE.cases[i],
      population: RATE_LATTICE.population[i]
    }));

  it('rate-cluster-map: every p-value, adjusted p-value and class is unchanged by a shuffle', () => {
    const parameters = {
      casesProperty: 'cases',
      populationProperty: 'population',
      permutations: 99,
      seed: 7,
      significance: 0.3,
      ...PLANAR
    };
    const first = runCuratedBinding('curated-rate-cluster-map', parameters, {areas: lattice()});
    for (const seed of [1, 2, 3]) {
      const second = runCuratedBinding('curated-rate-cluster-map', parameters, {
        areas: shuffleFeatures(lattice(), seed)
      });
      expectRowsMatch(
        getRowsById(first, 'rate-clusters'),
        getRowsById(second, 'rate-clusters'),
        ['p_value', 'adjusted_p_value', 'cluster', 'neighbor_count'],
        1e-12
      );
      expect(second.summary['significantCount']).toBe(first.summary['significantCount']);
    }
  });

  it('excess-risk, regression, correlogram, kriging and clusters give the same results after a shuffle', () => {
    // Excess risk.
    const excessParameters = {casesProperty: 'cases', populationProperty: 'population', ...PLANAR};
    const excess = runCuratedBinding('curated-excess-risk', excessParameters, {areas: lattice()});
    const excessShuffled = runCuratedBinding('curated-excess-risk', excessParameters, {
      areas: shuffleFeatures(lattice(), 4)
    });
    expectRowsMatch(
      getRowsById(excess, 'excess-risk'),
      getRowsById(excessShuffled, 'excess-risk'),
      ['status', 'neighbor_count'],
      1e-12
    );
    // Regression: residual of each area, the Moran statistic and its pseudo p-value.
    const regressionAreas = () =>
      makeLattice(6, 6, i => ({
        y: REGRESSION_LATTICE.y[i],
        x1: REGRESSION_LATTICE.x1[i],
        x2: REGRESSION_LATTICE.x2[i]
      }));
    const regressionParameters = {
      responseProperty: 'y',
      predictorProperties: 'x1,x2',
      permutations: 199,
      seed: 3,
      ...PLANAR
    };
    const regression = runCuratedBinding(
      'curated-regression-with-diagnostics',
      regressionParameters,
      {areas: regressionAreas()}
    );
    const regressionShuffled = runCuratedBinding(
      'curated-regression-with-diagnostics',
      regressionParameters,
      {areas: shuffleFeatures(regressionAreas(), 5)}
    );
    expectRowsMatch(
      getRowsById(regression, 'residuals'),
      getRowsById(regressionShuffled, 'residuals'),
      ['status', 'neighbor_count'],
      1e-9
    );
    const diagnostics = (r: typeof regression) => {
      const table = getResultTable(r, 'diagnostics');
      return new Map(
        (table.columns['diagnostic'] as string[]).map((name, row) => [
          name,
          [table.columns['value'][row], table.columns['p_value'][row]] as [number, number]
        ])
      );
    };
    const first = diagnostics(regression);
    const second = diagnostics(regressionShuffled);
    for (const [name, [value, p]] of first) {
      expect(Math.abs((second.get(name)![0] ?? 0) - (value ?? 0)), name).toBeLessThan(
        1e-9 * Math.max(1, Math.abs(value ?? 0))
      );
      if (name === 'residual-morans-i') expect(second.get(name)![1]).toBe(p);
    }
    // Correlogram: the table, order by order.
    const correlogramParameters = {
      valueProperty: 'v',
      maximumOrder: 4,
      permutations: 199,
      seed: 2,
      ...PLANAR
    };
    const correlogramAreas = () => makeLattice(8, 8, i => ({v: CORRELOGRAM_LATTICE.values[i]}));
    const correlogram = getResultTable(
      runCuratedBinding('curated-contiguity-correlogram', correlogramParameters, {
        areas: correlogramAreas()
      }),
      'correlogram'
    );
    const correlogramShuffled = getResultTable(
      runCuratedBinding('curated-contiguity-correlogram', correlogramParameters, {
        areas: shuffleFeatures(correlogramAreas(), 6)
      }),
      'correlogram'
    );
    expect(correlogramShuffled.columns['pseudo_p_value']).toEqual(
      correlogram.columns['pseudo_p_value']
    );
    expect(correlogramShuffled.columns['significant']).toEqual(correlogram.columns['significant']);
    expect(
      getMaximumRelativeError(
        getNumbers(correlogramShuffled, 'morans_i'),
        getNumbers(correlogram, 'morans_i')
      )
    ).toBeLessThan(1e-12);
    // Kriging: leave-one-out per point, and the summary.
    const krigingPoints = () =>
      makePoints(KRIGING_POINTS.x, KRIGING_POINTS.y, i => ({z: KRIGING_POINTS.z[i]}));
    const krigingParameters = {
      valueProperty: 'z',
      variogram: 'stated',
      model: 'spherical',
      nugget: 0.2,
      sill: 4,
      range: 600,
      cellSize: 200,
      neighborCount: 64,
      ...PLANAR
    };
    const kriging = runCuratedBinding('curated-kriging-with-diagnostics', krigingParameters, {
      points: krigingPoints()
    });
    const krigingShuffled = runCuratedBinding(
      'curated-kriging-with-diagnostics',
      krigingParameters,
      {points: shuffleFeatures(krigingPoints(), 8)}
    );
    const leaveFirst = getRowsById(kriging, 'cross-validation');
    const leaveSecond = getRowsById(krigingShuffled, 'cross-validation');
    for (const [id, row] of leaveFirst) {
      expect(
        Math.abs((leaveSecond.get(id)!['predicted'] as number) - (row['predicted'] as number))
      ).toBeLessThan(1e-8);
      expect(
        Math.abs((leaveSecond.get(id)!['error'] as number) - (row['error'] as number))
      ).toBeLessThan(1e-8);
    }
    expect(
      Math.abs(
        Number(krigingShuffled.summary['rootMeanSquareError']) -
          Number(kriging.summary['rootMeanSquareError'])
      )
    ).toBeLessThan(1e-9);
    // Clusters: the same partition and the same footprints.
    const random = makeSeededGenerator(9);
    const xs = Array.from({length: 120}, (_, i) => (i < 60 ? 0 : 50) + random() * 8);
    const ys = Array.from({length: 120}, () => random() * 8);
    const clusterParameters = {epsilon: 3, minimumPoints: 4, ...PLANAR};
    const clusters = runCuratedBinding('curated-clusters-and-footprints', clusterParameters, {
      points: makePoints(xs, ys)
    });
    const clustersShuffled = runCuratedBinding(
      'curated-clusters-and-footprints',
      clusterParameters,
      {points: shuffleFeatures(makePoints(xs, ys), 10)}
    );
    const partition = (r: typeof clusters) => {
      const rows = getRowsById(r, 'points');
      const groups = new Map<number, string[]>();
      for (const [id, row] of rows)
        groups.set(row['cluster'] as number, [...(groups.get(row['cluster'] as number) ?? []), id]);
      return [...groups.values()].map(ids => ids.sort().join(',')).sort();
    };
    expect(partition(clustersShuffled)).toEqual(partition(clusters));
    expect(getNumbers(getResultTable(clustersShuffled, 'footprints'), 'area').sort()).toEqual(
      getNumbers(getResultTable(clusters, 'footprints'), 'area').sort()
    );
  });

  it('counts rows with coincident anchors that can affect dataset-wide inference', () => {
    const anchors: [number, number][] = [
      [1, 1],
      [0, 5],
      [1, 1],
      [3, 0]
    ];
    const {order, tiedCount} = getCanonicalOrder(anchors);
    expect(order).toEqual([1, 0, 2, 3]);
    expect(tiedCount).toBe(2);
    const stacked = lattice();
    stacked.features.push({...stacked.features[0], id: 'twin'});
    const result = runCuratedBinding(
      'curated-rate-cluster-map',
      {casesProperty: 'cases', populationProperty: 'population', permutations: 99, ...PLANAR},
      {areas: stacked}
    );
    expect(result.summary['tiedAnchorCount']).toBe(2);
    expect(
      result.diagnostics.find(item => item.code === 'row-order-dependent-inference')?.count
    ).toBe(2);
    // Without ties the key is zero.
    expect(
      runCuratedBinding(
        'curated-rate-cluster-map',
        {casesProperty: 'cases', populationProperty: 'population', permutations: 99, ...PLANAR},
        {areas: lattice()}
      ).summary['tiedAnchorCount']
    ).toBe(0);
  });

  it('runConditionalLocalPermutationTest is deterministic and gives no p-value to an area without neighbours', () => {
    const deviations = [2, -1, 0.5, -1.5, 0];
    const neighbors = [[1, 2], [0, 3], [0], [1, 4], []];
    const anchors: [number, number][] = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0]
    ];
    const first = runConditionalLocalPermutationTest(deviations, neighbors, anchors, 99, 3);
    expect(runConditionalLocalPermutationTest(deviations, neighbors, anchors, 99, 3)).toEqual(
      first
    );
    expect(first.pValues[4]).toBeNull();
    for (const p of first.pValues.slice(0, 4)) expect(p).toBeGreaterThanOrEqual(1 / 100);
  });

  it('default permutation counts are 999 for every permutation-based binding', () => {
    for (const id of [
      'curated-rate-cluster-map',
      'curated-regression-with-diagnostics',
      'curated-contiguity-correlogram'
    ]) {
      const binding = runCuratedBindingDefaults(id);
      expect(binding).toBe(999);
    }
  });
});

function runCuratedBindingDefaults(id: string): number {
  // The default value of the permutations parameter, read from the registered binding.
  return KEPLER_CURATED_BINDINGS.find(binding => binding.id === id)!.parameters.find(
    parameter => parameter.name === 'permutations'
  )!.defaultValue as number;
}

describe('corrections that cannot reject', () => {
  it('getSignificanceAttainability follows the smallest pseudo p-value 1 / (permutations + 1)', () => {
    expect(getSignificanceAttainability('bonferroni', 0.05, 50, 999).attainable).toBe(true);
    expect(getSignificanceAttainability('bonferroni', 0.05, 51, 999).attainable).toBe(false);
    expect(getSignificanceAttainability('holm', 0.05, 144, 999).permutationsNeeded).toBe(2879);
    expect(getSignificanceAttainability('sidak', 0.05, 144, 999).attainable).toBe(false);
    const benjamini = getSignificanceAttainability('benjamini-hochberg', 0.05, 144, 999);
    expect(benjamini.attainable).toBe(true);
    expect(benjamini.requiredTiedMinimumCount).toBe(3);
    expect(getSignificanceAttainability('benjamini-hochberg', 0.001, 10, 99).attainable).toBe(
      false
    );
    expect(getSignificanceAttainability('none', 0.05, 10, 99).attainable).toBe(true);
  });

  it('rate-cluster-map says when no area could be flagged, and how many permutations would do', () => {
    const areas = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const parameters = {
      casesProperty: 'cases',
      populationProperty: 'population',
      permutations: 99,
      multipleTesting: 'bonferroni'
    };
    const result = runCuratedBinding('curated-rate-cluster-map', parameters, {areas});
    expect(result.summary['significanceNotAttainable']).toBe(1);
    expect(result.summary['permutationsNeeded']).toBe(2879);
    expect(
      result.diagnostics.find(item => item.code === 'significance-not-attainable')?.message
    ).toMatch(/about 2879 permutations are needed/);
    expect(result.summary['significantCount']).toBe(0);
    const benjamini = runCuratedBinding(
      'curated-rate-cluster-map',
      {...parameters, multipleTesting: 'benjamini-hochberg', permutations: 999},
      {areas}
    );
    expect(benjamini.summary['significanceNotAttainable']).toBe(0);
    expect(
      benjamini.diagnostics.find(item => item.code === 'significance-needs-ties')?.message
    ).toMatch(/at least 3 of the 144/);
  });

  it('classifies a cluster exactly when the corrected p-value is at most the significance, with data that has power', () => {
    const areas = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const base = {
      casesProperty: 'cases',
      populationProperty: 'population',
      permutations: 999,
      seed: 1
    };
    const corrected = getResultTable(
      runCuratedBinding('curated-rate-cluster-map', base, {areas}),
      'rate-clusters'
    );
    const uncorrected = getResultTable(
      runCuratedBinding('curated-rate-cluster-map', {...base, multipleTesting: 'none'}, {areas}),
      'rate-clusters'
    );
    const flagged = (table: typeof corrected) =>
      (table.columns['cluster'] as string[]).filter(label =>
        ['HH', 'HL', 'LH', 'LL'].includes(label)
      ).length;
    recordBindingMargin('rate map flagged, corrected', flagged(corrected));
    recordBindingMargin('rate map flagged, uncorrected', flagged(uncorrected));
    expect(flagged(corrected)).toBeGreaterThan(0);
    expect(flagged(corrected)).toBeLessThan(flagged(uncorrected));
    for (const table of [corrected, uncorrected]) {
      const adjusted = getNumbers(table, 'adjusted_p_value');
      (table.columns['cluster'] as string[]).forEach((label, index) =>
        expect(['HH', 'HL', 'LH', 'LL'].includes(label), `row ${index}`).toBe(
          adjusted[index] <= 0.05
        )
      );
    }
  });
});

describe('kriging: warnings, rows and capacity', () => {
  const random = makeSeededGenerator(12);
  const normal = () =>
    Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  const xs = Array.from({length: 120}, () => random() * 100);
  const ys = Array.from({length: 120}, () => random() * 100);
  const make = (getValue: (x: number, y: number) => number) =>
    makePoints(xs, ys, i => ({v: getValue(xs[i], ys[i])}));
  const run = (points: CatalogFeatureCollection) =>
    runCuratedBinding(
      'curated-kriging-with-diagnostics',
      {valueProperty: 'v', cellSize: 50, ...PLANAR},
      {points}
    );

  it('flags a variogram that is not identified: white noise, a trend, periodic data', () => {
    for (const [name, getValue] of [
      ['white noise', () => normal()],
      ['trend', (x: number, y: number) => 0.5 * x + 0.3 * y + 0.2 * normal()],
      ['periodic', (x: number) => Math.sin(x / 6) + 0.05 * normal()]
    ] as const) {
      const result = run(make(getValue));
      recordBindingMargin(`kriging ${name} poorModelFit`, Number(result.summary['poorModelFit']));
      expect(result.summary['poorModelFit'], name).toBe(1);
      expect(String(result.summary['poorModelFitReasons']).length).toBeGreaterThan(20);
      expect(
        result.diagnostics.map(item => item.code),
        name
      ).toContain('poor-model-fit');
    }
  });

  it('does not flag a field with a well-identified variogram (smoothed noise, correlation ending inside the lag)', () => {
    const size = 18;
    let state = 12345;
    const next = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296 - 0.5;
    };
    const noise = Array.from({length: size + 8}, () =>
      Array.from({length: size + 8}, () => next() * 6)
    );
    const gx: number[] = [];
    const gy: number[] = [];
    const gz: number[] = [];
    for (let row = 0; row < size; row++)
      for (let column = 0; column < size; column++) {
        let sum = 0;
        for (let dr = 0; dr < 5; dr++)
          for (let dc = 0; dc < 5; dc++) sum += noise[row + dr][column + dc];
        gx.push(column * 100);
        gy.push(row * 100);
        gz.push(sum + next() * 0.2);
      }
    const result = runCuratedBinding(
      'curated-kriging-with-diagnostics',
      {valueProperty: 'v', cellSize: 300, neighborCount: 12, binCount: 12, ...PLANAR},
      {points: makePoints(gx, gy, i => ({v: gz[i]}))}
    );
    recordBindingMargin(
      'kriging identified-field rmsse',
      Number(result.summary['rootMeanSquareStandardizedError'])
    );
    expect(result.summary['poorModelFit'], String(result.summary['poorModelFitReasons'])).toBe(0);
    expect(result.summary['poorModelFitReasons']).toBe('');
  });

  it('gives a row to every input feature, with a status', () => {
    const points = make(() => normal());
    points.features.push({id: 'twin', geometry: points.features[0].geometry, properties: {v: 1}});
    points.features.push({
      id: 'bad',
      geometry: {type: 'Point', coordinates: [1, 1]},
      properties: {v: 'x'}
    });
    points.features.push({id: 'none', geometry: null, properties: {v: 1}});
    const table = getResultTable(run(points), 'cross-validation');
    expect(table.rowCount).toBe(123);
    expect(table.columns['status'].slice(120)).toEqual([
      'merged-into-earlier-sample',
      'missing-value',
      'missing-geometry'
    ]);
  });

  it('names the binding, the limit and the remedy when the work is too large', () => {
    const count = 2000;
    const points = makePoints(
      Array.from({length: count}, (_, i) => ((i * 37) % 1000) + i * 1e-3),
      Array.from({length: count}, (_, i) => ((i * 91) % 1000) + i * 1e-3),
      i => ({v: Math.sin(i)})
    );
    expect(() =>
      runCuratedBinding(
        'curated-kriging-with-diagnostics',
        {
          valueProperty: 'v',
          cellSize: 5.1,
          variogram: 'stated',
          model: 'spherical',
          sill: 1,
          range: 300,
          ...PLANAR
        },
        {points}
      )
    ).toThrow(
      /curated-kriging-with-diagnostics: 38\d+ cells times 2000 samples .* limit of 60000000; raise cellSize to at least \d+ or use fewer samples/
    );
  });

  it('documents the summary keys of the run-time cautions, and the runs carry them', () => {
    const result = run(make(() => normal()));
    for (const {key} of Object.values(
      CURATED_RUNTIME_CAUTION_SUMMARY_KEYS['curated-kriging-with-diagnostics']
    ))
      expect(result.summary[key]).toBeDefined();
    const excess = runCuratedBinding(
      'curated-excess-risk',
      {casesProperty: 'cases', populationProperty: 'population', ...PLANAR},
      {
        areas: makeLattice(5, 5, i => ({
          cases: RATE_LATTICE.cases[i],
          population: RATE_LATTICE.population[i]
        }))
      }
    );
    for (const {key} of Object.values(CURATED_RUNTIME_CAUTION_SUMMARY_KEYS['curated-excess-risk']))
      expect(excess.summary[key]).toBeDefined();
  });
});

describe('excess risk: the smoothed ratio', () => {
  it('equals the empirical Bayes rate over the overall rate, to 1e-12', () => {
    const result = runCuratedBinding(
      'curated-excess-risk',
      {casesProperty: 'cases', populationProperty: 'population', ...PLANAR},
      {
        areas: makeLattice(5, 5, i => ({
          cases: RATE_LATTICE.cases[i],
          population: RATE_LATTICE.population[i]
        }))
      }
    );
    expect(
      getMaximumRelativeError(
        getNumbers(getResultTable(result, 'excess-risk'), 'smoothed_excess_risk'),
        RATE_LATTICE.smoothedExcessRisk
      )
    ).toBeLessThan(1e-12);
  });

  it('keeps a tiny area driven by one case from topping the smoothed map, and counts small denominators', () => {
    // One area of 20 residents with a single case beside areas of 3000 residents at the same rate.
    const areas = makeLattice(5, 5, i => ({
      cases: i === 12 ? 1 : 30 + (i % 4),
      population: i === 12 ? 20 : 3000
    }));
    const result = runCuratedBinding(
      'curated-excess-risk',
      {casesProperty: 'cases', populationProperty: 'population', ...PLANAR},
      {areas}
    );
    const table = getResultTable(result, 'excess-risk');
    const raw = getNumbers(table, 'excess_risk');
    const smoothed = getNumbers(table, 'smoothed_excess_risk');
    expect(raw.indexOf(Math.max(...raw))).toBe(12);
    expect(raw[12]).toBeGreaterThan(4);
    expect(smoothed[12]).toBeLessThan(2);
    expect(result.summary['smallDenominatorAreaCount']).toBeGreaterThanOrEqual(1);
    expect(result.summary['smallDenominatorThreshold']).toBe(5);
    expect(result.diagnostics.map(item => item.code)).toContain('small-denominators');
  });
});

describe('DBSCAN boundaries', () => {
  it.each([
    {
      name: 'exact pair without translation',
      horizontal: [0, 1],
      vertical: [0, 0],
      labels: [0, 0]
    },
    {
      name: 'pair just outside the requested radius without translation',
      horizontal: [0, 1.0000000005],
      vertical: [0, 0],
      labels: [-1, -1]
    },
    {
      name: 'exact pair after an inexact frame translation',
      horizontal: [3, 4, 0.1],
      vertical: [0, 0, 100],
      labels: [0, 0, -1]
    },
    {
      name: 'pair just outside the requested radius after translation',
      horizontal: [3, 4.0000000005, 0.1],
      vertical: [0, 0, 100],
      labels: [-1, -1, -1]
    }
  ])('AX-87: catalog and curated agree for $name', ({horizontal, vertical, labels}) => {
    const points = makePoints(horizontal, vertical);
    for (const bindingId of ['density-clustering', 'curated-clusters-and-footprints']) {
      const result = runCuratedBinding(
        bindingId,
        {epsilon: 1, minimumPoints: 2, ...PLANAR},
        {points}
      );
      const table = getResultTable(result, 'points');
      expect(getNumbers(table, 'cluster'), bindingId).toEqual(labels);
      expect(table.columns['is_core'], bindingId).toEqual(labels.map(label => label >= 0));
      expect(result.summary['noiseCount'], bindingId).toBe(
        labels.filter(label => label < 0).length
      );
    }
  });

  const run = (xs: number[], ys: number[], epsilon: number, minimumPoints: number) =>
    runCuratedBinding(
      'curated-clusters-and-footprints',
      {epsilon, minimumPoints, ...PLANAR},
      {points: makePoints(xs, ys)}
    );

  it('counts a pair exactly at the search distance as within it, even when the frame shift adds rounding', () => {
    // 4 - 0.1 and 3 - 0.1 differ by 1.0000000000000004 in the shifted frame.
    const inside = run([3, 4, 0.1], [0, 0, 100], 1, 2);
    expect(getNumbers(getResultTable(inside, 'points'), 'cluster')).toEqual([0, 0, -1]);
    const outside = run([3, 4.0001, 0.1], [0, 0, 100], 1, 2);
    expect(getNumbers(getResultTable(outside, 'points'), 'cluster')).toEqual([-1, -1, -1]);
    expect(String(inside.diagnostics.length)).toBeDefined();
  });

  it('gives a border point reachable from two clusters to the first in input order, and the other order flips it', () => {
    const left = [0, 0.3, 0.6, 0.9];
    const right = [2.8, 3.1, 3.4, 3.7];
    const border = 1.85;
    const first = run([...left, ...right, border], new Array(9).fill(0), 1, 4);
    const labels = getNumbers(getResultTable(first, 'points'), 'cluster');
    expect(labels.slice(0, 4)).toEqual([0, 0, 0, 0]);
    expect(labels.slice(4, 8)).toEqual([1, 1, 1, 1]);
    expect(labels[8]).toBe(0);
    expect(getResultTable(first, 'points').columns['is_core'][8]).toBe(false);
    const reversed = run([...right, ...left, border], new Array(9).fill(0), 1, 4);
    expect(getNumbers(getResultTable(reversed, 'points'), 'cluster')[8]).toBe(0);
    expect(getNumbers(getResultTable(reversed, 'points'), 'cluster').slice(0, 4)).toEqual([
      0, 0, 0, 0
    ]);
    // The border point joined the cluster of the right-hand points when they come first.
    expect(getNumbers(getResultTable(reversed, 'points'), 'cluster')[8]).toBe(
      getNumbers(getResultTable(reversed, 'points'), 'cluster')[0]
    );
  });
});

describe('unused helper types', () => {
  it('keeps CatalogFeature imported for the shuffle helpers', () => {
    const feature: CatalogFeature = {geometry: null, properties: {}};
    expect(feature.geometry).toBeNull();
  });
});

it('breaks unique-anchor distance ties canonically before residual weights and kriging selection', () => {
  const points = makePoints(
    Array.from({length: 25}, (_, index) => index % 5),
    Array.from({length: 25}, (_, index) => Math.floor(index / 5)),
    index => ({value: Math.sin((index + 1) * 2 * 1.7953), x: Math.cos(index * 0.7)})
  );
  const regressionParameters = {
    ...PLANAR,
    responseProperty: 'value',
    predictorProperties: 'x',
    neighborhood: 'knn',
    neighborCount: 3,
    permutations: 999,
    seed: 43
  };
  const krigingParameters = {
    ...PLANAR,
    valueProperty: 'value',
    variogram: 'stated',
    model: 'spherical',
    sill: 3,
    nugget: 0.1,
    range: 2,
    cellSize: 2,
    neighborCount: 4
  };
  const regression = runCuratedBinding(
    'curated-regression-with-diagnostics',
    regressionParameters,
    {areas: points}
  );
  const kriging = runCuratedBinding('curated-kriging-with-diagnostics', krigingParameters, {
    points
  });
  for (const reordered of [
    {...points, features: [...points.features].reverse()},
    shuffleFeatures(points, 117)
  ]) {
    const nextRegression = runCuratedBinding(
      'curated-regression-with-diagnostics',
      regressionParameters,
      {areas: reordered}
    );
    expect(getResultTable(nextRegression, 'diagnostics').columns['p_value']).toEqual(
      getResultTable(regression, 'diagnostics').columns['p_value']
    );
    expect(getResultTable(nextRegression, 'diagnostics').columns['value'].at(-1)).toBe(
      getResultTable(regression, 'diagnostics').columns['value'].at(-1)
    );
    expectRowsMatch(
      getRowsById(regression, 'residuals'),
      getRowsById(nextRegression, 'residuals'),
      ['fitted', 'residual', 'neighbor_count'],
      0
    );
    const nextKriging = runCuratedBinding('curated-kriging-with-diagnostics', krigingParameters, {
      points: reordered
    });
    expectRowsMatch(
      getRowsById(kriging, 'cross-validation'),
      getRowsById(nextKriging, 'cross-validation'),
      ['predicted', 'kriging_variance'],
      0
    );
    expect(getResultTable(nextKriging, 'kriging-cells')).toEqual(
      getResultTable(kriging, 'kriging-cells')
    );
    expect(getResultTable(nextKriging, 'cross-validation').columns['source_id']).toEqual(
      reordered.features.map(feature => feature.id)
    );
    expect(getResultTable(nextRegression, 'residuals').columns['source_id']).toEqual(
      reordered.features.map(feature => feature.id)
    );
  }
});

it('warns that coincident anchors affect inference for unique-anchor rows too', () => {
  const areas = makeLattice(5, 4, (index, horizontal, vertical) => ({
    cases: index === 0 ? 200 : (index * 7) % 13,
    population: index === 0 ? 400 : 40 + index * 23,
    value: horizontal * horizontal + vertical * 2 + Math.sin(index)
  }));
  areas.features.push({
    ...areas.features[0],
    id: 'duplicate',
    properties: {cases: 4, population: 400, value: -4}
  });
  const parameters = {
    ...PLANAR,
    casesProperty: 'cases',
    populationProperty: 'population',
    permutations: 999,
    seed: 43
  };
  const first = runCuratedBinding('curated-rate-cluster-map', parameters, {areas});
  const reversed = runCuratedBinding('curated-rate-cluster-map', parameters, {
    areas: {...areas, features: [...areas.features].reverse()}
  });
  expect(getRowsById(first, 'rate-clusters').get('a1')!['p_value']).not.toBe(
    getRowsById(reversed, 'rate-clusters').get('a1')!['p_value']
  );
  expect(
    first.diagnostics.find(diagnostic => diagnostic.code === 'row-order-dependent-inference')
      ?.message
  ).toMatch(/throughout the dataset, including areas with unique anchors/);
  for (const bindingId of [
    'curated-rate-cluster-map',
    'curated-regression-with-diagnostics',
    'curated-contiguity-correlogram'
  ]) {
    expect(
      KEPLER_CURATED_BINDINGS.find(binding => binding.id === bindingId)!.declaredBehavior.join(' ')
    ).toContain('inference throughout the dataset');
  }
});

// AX-87 probes compare the retained safeguards with the fixed catalog routines directly.
// Numerical measurements can be captured with CURATED_MARGINS_FILE; assertions pin the decisions.
describe('AX-87: safeguards with distinct semantics', () => {
  it('retains island presentation exclusion because the hot-spot scheme cannot encode not-tested', () => {
    const features = makePoints([0, 0, 1, 1, 1000], [0, 1, 0, 1, 1000], index => ({
      value: [1, 2, 3, 4, 60][index]
    }));
    const result = runCuratedBinding(
      'hotspot-gi-star',
      {...PLANAR, valueProperty: 'value', neighborhood: 'distance-band', distanceBand: 1.5},
      {features}
    );
    const table = result.tables[0];
    expect(table.columns['classification'][4]).toBe('not-tested');
    const output = getKeplerOperationDescriptor('hotspot-gi-star')!.outputs[0];
    const guarded = planKeplerResultPresentation(table, output)!;
    const columns = {...table.columns};
    delete columns['neighbor_count'];
    const unguarded = planKeplerResultPresentation({...table, columns}, output)!;
    expect(guarded.classScheme).toBe('hot-spot-confidence-bins');
    expect(guarded.excludedRows).toEqual([4]);
    expect(unguarded.scale).toBe('categorical');
    expect(unguarded.classScheme).toBeUndefined();
    recordBindingMargin(
      'AX-87 Gi-star islands correctly excluded by presentation',
      guarded.excludedRows!.length
    );
  });

  it('still needs canonical permutation streams after the neighbor tie fix', () => {
    const points = makePoints(
      Array.from({length: 25}, (_, index) => index % 5),
      Array.from({length: 25}, (_, index) => Math.floor(index / 5)),
      index => ({value: Math.sin((index + 1) * 2 * 1.7953)})
    );
    const evaluate = (collection: CatalogFeatureCollection) => {
      const analysis = buildAnalysisSet(collection, 'value', {
        neighborhood: 'knn',
        neighborCount: 3,
        distanceBand: 1,
        rowStandardize: true,
        coordinateSystem: 'planar'
      });
      const frame = makeLocalFrame('planar', []);
      const anchors = collection.features.map(feature => getFeatureAnchor(feature, frame)!);
      return {
        canonical: runCanonicalMoranPermutationTest(
          analysis.values,
          analysis.weights.neighbors,
          analysis.weights.weights,
          anchors,
          999,
          43
        ),
        inputOrder: runMoranPermutationTest(
          analysis.values,
          analysis.weights.neighbors,
          analysis.weights.weights,
          999,
          43
        ),
        neighborIds: new Map(
          collection.features.map((feature, row) => [
            feature.id,
            analysis.weights.neighbors[row].map(other => collection.features[other].id)
          ])
        )
      };
    };
    const first = evaluate(points);
    const reversed = evaluate({...points, features: [...points.features].reverse()});
    expect(reversed.neighborIds).toEqual(first.neighborIds);
    expect(reversed.canonical).toEqual(first.canonical);
    const delta = reversed.inputOrder.pseudoPValue - first.inputOrder.pseudoPValue;
    recordBindingMargin('AX-87 input-order permutation reversal p-value delta', delta);
    expect(Math.abs(delta)).toBeGreaterThan(0);
  });

  it('compares the two OLS fits, implied covariance and auxiliary diagnostics on the reviewer fixture', () => {
    const response = Float64Array.from(REGRESSION_SCALED.y);
    const predictors = Float64Array.from(
      REGRESSION_SCALED.income.flatMap((value, row) => [value, REGRESSION_SCALED.share[row]])
    );
    const sample = {
      response,
      predictors,
      predictorCount: 2,
      count: response.length,
      featureIndex: new Int32Array(response.length),
      coordinates: [] as [number, number][],
      droppedCount: 0,
      names: ['income', 'share']
    };
    const catalog = fitOrdinaryLeastSquares(sample);
    const curated = fitStandardizedLeastSquares(response, predictors, 2, response.length);
    for (const column of [
      'coefficients',
      'standardErrors',
      'tStatistics',
      'fitted',
      'residuals'
    ] as const) {
      const delta = Math.max(
        ...Array.from(curated[column], (value, row) => Math.abs(value - catalog[column][row]))
      );
      recordBindingMargin(`AX-87 OLS ${column} maximum absolute delta`, delta);
      expect(
        getMaximumRelativeError(Array.from(curated[column]), Array.from(catalog[column])),
        column
      ).toBeLessThan(1e-9);
    }
    // Standard-error squares pin the diagonal of the estimated covariance in original units.
    const varianceDelta = Math.max(
      ...curated.standardErrors.map((value, column) =>
        Math.abs(value ** 2 - catalog.standardErrors[column] ** 2)
      )
    );
    recordBindingMargin('AX-87 OLS covariance diagonal maximum absolute delta', varianceDelta);
    for (const column of [
      'rSquared',
      'adjustedRSquared',
      'residualSumOfSquares',
      'sigmaSquared'
    ] as const) {
      recordBindingMargin(
        `AX-87 OLS ${column} absolute delta`,
        Math.abs(curated[column] - catalog[column])
      );
      expect(curated[column]).toBeCloseTo(catalog[column], 9);
    }
    const catalogAuxiliary = fitOrdinaryLeastSquares({
      ...sample,
      response: Float64Array.from(catalog.residuals, value => value ** 2)
    });
    const curatedAuxiliary = fitStandardizedLeastSquares(
      Float64Array.from(curated.residuals, value => value ** 2),
      predictors,
      2,
      response.length
    );
    const delta = response.length * Math.abs(curatedAuxiliary.rSquared - catalogAuxiliary.rSquared);
    recordBindingMargin('AX-87 OLS Koenker-Bassett absolute delta', delta);
    expect(delta).toBeLessThan(1e-9);
  });

  it('retains the fused OLS scale range and stricter identification threshold', () => {
    const response = Float64Array.from([1, 4, 2, 7, 6, 9, 8, 12]);
    for (const scale of [1e-200, 1e200]) {
      const predictors = Float64Array.from(response, (_, row) => row * scale);
      const curated = fitStandardizedLeastSquares(response, predictors, 1, response.length);
      expect(Array.from(curated.fitted).every(Number.isFinite)).toBe(true);
      const evaluateCatalog = () =>
        fitOrdinaryLeastSquares({
          response,
          predictors,
          predictorCount: 1,
          count: response.length,
          featureIndex: new Int32Array(response.length),
          coordinates: [],
          droppedCount: 0,
          names: ['scaled']
        });
      expect(evaluateCatalog).toThrow(scale < 1 ? /nonzero spread/ : /collinear/);
    }
    const predictors = Float64Array.from(
      Array.from(response, (_, row) => [row, row + (row % 2) * 1e-5]).flat()
    );
    expect(() => fitStandardizedLeastSquares(response, predictors, 2, response.length)).toThrow(
      /collinear/
    );
    const catalog = fitOrdinaryLeastSquares({
      response,
      predictors,
      predictorCount: 2,
      count: response.length,
      featureIndex: new Int32Array(response.length),
      coordinates: [],
      droppedCount: 0,
      names: ['first', 'near-copy']
    });
    expect(catalog.coefficients.every(Number.isFinite)).toBe(true);
  });

  it('retains profile anchor range, degenerate-ring fallback and geographic line semantics', () => {
    const frame = makeLocalFrame('planar', []);
    const profileAnchor = (feature: CatalogFeature, coordinateSystem: 'planar' | 'EPSG:4326') =>
      profileKeplerFeatureCollection(
        {type: 'FeatureCollection', features: [feature]},
        {datasetId: 'ax87', label: 'AX-87 anchor probe', coordinateSystem}
      ).spacing!.center;
    const degenerate: CatalogFeature = {
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2, 0],
            [10, 0],
            [0, 0]
          ]
        ]
      },
      properties: {}
    };
    const degenerateCatalog = getFeatureAnchor(degenerate, frame)!;
    const degenerateProfile = profileAnchor(degenerate, 'planar');
    recordBindingMargin(
      'AX-87 degenerate polygon centroid x delta',
      degenerateProfile[0] - degenerateCatalog[0]
    );
    expect(degenerateProfile[0]).toBe(5);
    expect(degenerateCatalog[0]).toBe(3);
    const line: CatalogFeature = {
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [1, 0],
          [10, 0]
        ]
      },
      properties: {}
    };
    const lineCatalog = getFeatureAnchor(line, frame)!;
    const lineProfile = profileAnchor(line, 'EPSG:4326');
    recordBindingMargin(
      'AX-87 geographic line centroid longitude delta',
      lineProfile[0] - lineCatalog[0]
    );
    expect(lineProfile[0]).toBeCloseTo(5, 12);
    expect(lineCatalog[0]).toBe(11 / 3);
    for (const scale of [1e-200, 1e200]) {
      const triangle: CatalogFeature = {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [2 * scale, 0],
              [0, scale],
              [0, 0]
            ]
          ]
        },
        properties: {}
      };
      const profile = profileAnchor(triangle, 'planar');
      const catalog = getFeatureAnchor(triangle, frame)!;
      expect(profile[0] / scale).toBeCloseTo(2 / 3, 12);
      expect(profile[1] / scale).toBeCloseTo(1 / 3, 12);
      expect(catalog[0] / scale).not.toBeCloseTo(2 / 3, 12);
    }
  });

  it('distinguishes exact planar radius boundaries from a geographic metric mismatch', () => {
    const planar = makePoints([3, 4, 0.1], [0, 0, 100]);
    const exact = runCuratedBinding(
      'density-clustering',
      {...PLANAR, epsilon: 1, minimumPoints: 2},
      {points: planar}
    );
    expect(getNumbers(getResultTable(exact, 'points'), 'cluster')).toEqual([0, 0, -1]);
    const positions: [number, number][] = [
      [-1, 60],
      [1, 60],
      [-1, 61],
      [1, 61]
    ];
    const geographic = makePoints(
      positions.map(position => position[0]),
      positions.map(position => position[1])
    );
    const band = Math.max(
      ...positions.map((position, row) =>
        Math.min(
          ...positions
            .filter((_, other) => row !== other)
            .map(other => measureDistance(position, other, 'EPSG:4326'))
        )
      )
    );
    const getIslands = (distanceBand: number) =>
      buildAnalysisSet(geographic, '', {
        neighborhood: 'distance-band',
        neighborCount: 3,
        distanceBand,
        rowStandardize: true,
        coordinateSystem: 'EPSG:4326'
      }).weights.neighbors.filter(neighbors => neighbors.length === 0).length;
    recordBindingMargin('AX-87 geographic exact spherical band', band);
    recordBindingMargin('AX-87 geographic islands without margin', getIslands(band));
    recordBindingMargin('AX-87 geographic islands with margin', getIslands(band * 1.01));
    expect(getIslands(band)).toBeGreaterThan(0);
    expect(getIslands(band * 1.01)).toBe(0);
  });
});
