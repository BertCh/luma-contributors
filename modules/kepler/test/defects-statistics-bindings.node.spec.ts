// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {makeSeededGenerator} from '../src/analysis/catalog-platform3-b-common';
import {computeKrigingField} from '../src/analysis/catalog-platform3-b-grids';
import {
  runGlobalSpatialStatistics,
  runLocalMoran
} from '../src/analysis/catalog-platform3-b-statistics';
import {getConservativeCountPValue} from '../src/analysis/catalog-platform5-recipes-hot-spot';

function points(
  rows: {coordinates: number[]; properties?: Record<string, unknown>; id?: string}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map((row, index) => ({
      id: row.id ?? String(index),
      geometry: {type: 'Point', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  };
}
function run(
  id: string,
  parameters: Record<string, unknown>,
  collection: CatalogFeatureCollection,
  input = 'features'
) {
  return runCatalogOperation(
    getCatalogBinding(id)!,
    {coordinateSystem: 'planar', ...parameters},
    {[input]: collection}
  );
}
const weights = {
  neighborhood: 'knn' as const,
  neighborCount: 1,
  distanceBand: 1,
  rowStandardize: true,
  coordinateSystem: 'planar' as const
};

test('Gi star excludes an extreme island instead of classifying its own value', () => {
  const collection = points([
    ...Array.from({length: 146}, (_, index) => ({
      coordinates: [index % 10, Math.floor(index / 10)],
      properties: {value: index % 7}
    })),
    {coordinates: [10000, 10000], properties: {value: 60}, id: 'island'}
  ]);
  const result = run(
    'hotspot-gi-star',
    {valueProperty: 'value', neighborhood: 'distance-band', distanceBand: 1.5},
    collection
  );
  const table = result.tables[0];
  const island = table.columns['source_id'].indexOf('island');
  expect(table.columns['z_score'][island]).toBeNull();
  expect(table.columns['p_value'][island]).toBeNull();
  expect(table.columns['classification'][island]).toBe('not-tested');
  expect(result.diagnostics.find(diagnostic => diagnostic.code === 'island-features')?.count).toBe(
    1
  );
  expect(result.summary['islandCount']).toBe(1);
});

test('OLS fits income sd 15000 and share sd 0.01, correlation 0.10, against NumPy lstsq', () => {
  const collection = points(
    Array.from({length: 80}, (_, index) => {
      const angle = (index * 2 * Math.PI) / 80;
      const income = 60000 + 15000 * Math.SQRT2 * Math.cos(angle);
      const share =
        0.3 + 0.01 * Math.SQRT2 * (0.1 * Math.cos(angle) + Math.sqrt(0.99) * Math.sin(angle));
      return {
        coordinates: [index, 0],
        properties: {
          income,
          share,
          response: 7 + 0.002 * income + 120 * share + 0.3 * Math.cos(3 * angle)
        }
      };
    })
  );
  const binding = getCatalogBinding('regression-ols')!;
  const result = runCatalogOperation(
    binding,
    {responseProperty: 'response', predictorProperties: 'income,share'},
    {points: collection}
  );
  // /usr/bin/python3: numpy.linalg.lstsq(column_stack([ones(80), income, share]), response, rcond=None).
  const expected = [7.000000000000064, 0.0019999999999992996, 120.00000000000003];
  const actual = result.tables[0].columns['estimate'] as number[];
  expected.forEach((value, index) =>
    expect(Math.abs((actual[index] - value) / value)).toBeLessThan(1e-6)
  );
});

test('nearest-neighbor ties and seeded Moran inference are identical after reversing feature rows', () => {
  const collection = points(
    Array.from({length: 25}, (_, index) => ({
      coordinates: [index % 5, Math.floor(index / 5)],
      properties: {value: Math.sin(index * 7)}
    }))
  );
  const reversed = {...collection, features: [...collection.features].reverse()};
  const first = run('spatial-weights', {neighborCount: 3}, collection).tables[0];
  const second = run('spatial-weights', {neighborCount: 3}, reversed).tables[0];
  const links = (table: typeof first) =>
    table.columns['source_id']
      .map((source, index) => [
        source,
        table.columns['neighbor_id'][index],
        table.columns['weight'][index],
        table.columns['distance'][index]
      ])
      .sort((first, second) => JSON.stringify(first).localeCompare(JSON.stringify(second)));
  expect(links(first)).toEqual(links(second));
  const parameters = {valueProperty: 'value', neighborCount: 3, permutations: 199, seed: 42};
  const original = run('local-moran-statistics', parameters, collection).tables[0];
  const reordered = run('local-moran-statistics', parameters, reversed).tables[0];
  const resultsByIdentity = (table: typeof original) =>
    table.columns['source_id']
      .map((identity, index) => [
        identity,
        ...['local_i', 'p_value', 'adjusted_p_value', 'cluster'].map(
          name => table.columns[name][index]
        )
      ])
      .sort((first, second) => String(first[0]).localeCompare(String(second[0])));
  expect(resultsByIdentity(original)).toEqual(resultsByIdentity(reordered));
});

test('kriging ties choose canonical coordinates and feature keys after reversing rows', () => {
  const collection = points(
    Array.from({length: 25}, (_, index) => ({
      coordinates: [index % 5, Math.floor(index / 5)],
      properties: {value: Math.sin(index * 7)}
    }))
  );
  const parameters = {
    valueProperty: 'value',
    cellSize: 1,
    neighborCount: 3,
    nugget: 0.1,
    sill: 1,
    range: 3
  };
  const first = run('kriging-grid', parameters, collection, 'points').tables[0];
  const second = run(
    'kriging-grid',
    parameters,
    {...collection, features: [...collection.features].reverse()},
    'points'
  ).tables[0];
  expect(first.columns['value']).toEqual(second.columns['value']);
  expect(first.columns['variance']).toEqual(second.columns['variance']);
});

test('lattice defaults to BH and count p-values conservatively bound exact binomial tails', () => {
  expect(
    getCatalogDefaultParameters(getCatalogBinding('recipe-hot-spot-lattice')!)['multipleTesting']
  ).toBe('benjamini-hochberg');
  // Enumerate Binomial(20, 0.2) exactly, independently of the Chernoff formula.
  const probabilities = [0.8 ** 20];
  for (let count = 1; count <= 20; count++)
    probabilities.push((((probabilities[count - 1] * (21 - count)) / count) * 0.2) / 0.8);
  for (let observed = 0; observed <= 20; observed++) {
    const lower = probabilities.slice(0, observed + 1).reduce((sum, value) => sum + value, 0);
    const upper = probabilities.slice(observed).reduce((sum, value) => sum + value, 0);
    expect(getConservativeCountPValue(observed, 20, 0.2)).toBeGreaterThanOrEqual(
      Math.min(1, 2 * Math.min(lower, upper)) - 1e-12
    );
  }
});

test('seeded count-null lattice simulation measures false discoveries with BH', () => {
  let flaggedRuns = 0;
  let flaggedCells = 0;
  let cellCount = 0;
  for (let seed = 1; seed <= 220; seed++) {
    const random = makeSeededGenerator(seed);
    const collection = points(
      Array.from({length: 5000}, () => ({coordinates: [random() * 1000, random() * 1000]}))
    );
    const table = run(
      'recipe-hot-spot-lattice',
      {cellSize: Math.sqrt(1e6 / 5000)},
      collection,
      'points'
    ).tables[0];
    const flagged = (table.columns['bin'] as number[]).filter(value => value !== 0).length;
    flaggedRuns += Number(flagged > 0);
    flaggedCells += flagged;
    cellCount += table.rowCount;
  }
  process.stdout.write(
    `STATISTICS count-null BH (5,000 points/run): ${flaggedRuns}/220 runs (${((100 * flaggedRuns) / 220).toFixed(2)}%), ${flaggedCells}/${cellCount} cells at alpha=0.10\n`
  );
  expect(flaggedRuns).toBeLessThanOrEqual(33);
});

test('Local Moran applies BH and Bonferroni to tested rows and publishes raw and adjusted p', () => {
  const collection = points(
    [10, 11, 1, 2, 99].map((value, index) => ({coordinates: [index, 0], properties: {value}}))
  );
  const set = {
    sourceIndex: [0, 1, 2, 3, 4],
    values: [10, 11, 1, 2, 99],
    weights: {
      neighbors: [[1], [0], [3], [2], []],
      weights: [[1], [1], [1], [1], []],
      distances: [[1], [1], [1], [1], []]
    },
    droppedCount: 0
  };
  const precomputed = {
    set,
    statistic: [1, 1, 1, 1, 0],
    spatialLag: [1, 1, -1, -1, 0],
    pValue: [0.01, 0.04, 0.03, 0.005, 0.00001]
  };
  const parameters = {
    ...weights,
    valueProperty: 'value',
    permutations: 99,
    seed: 1,
    significance: 0.05
  };
  const result = runLocalMoran(collection, parameters, precomputed).tables[0];
  expect(result.columns['p_value']).toEqual([0.01, 0.04, 0.03, 0.005, null]);
  expect(result.columns['adjusted_p_value']).toEqual([0.02, 0.04, 0.04, 0.02, null]);
  const bonferroni = runLocalMoran(
    collection,
    {...parameters, multipleTesting: 'bonferroni'},
    precomputed
  ).tables[0];
  expect(bonferroni.columns['adjusted_p_value']).toEqual([0.04, 0.16, 0.12, 0.02, null]);
  expect(bonferroni.columns['cluster'][1]).toBe('not-significant');
  const defaults = getCatalogDefaultParameters(getCatalogBinding('local-moran-statistics')!);
  expect(defaults['multipleTesting']).toBe('benjamini-hochberg');
  expect(defaults['permutations']).toBe(999);
});

test('kriging white noise uncertainty follows the sample variance instead of a unit sill', () => {
  const random = makeSeededGenerator(44);
  const collection = points(
    Array.from({length: 100}, (_, index) => ({
      coordinates: [(index % 10) * 100, Math.floor(index / 10) * 100],
      properties: {
        value:
          10 *
          Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) *
          Math.cos(2 * Math.PI * random())
      }
    }))
  );
  const table = run('kriging-grid', {valueProperty: 'value', cellSize: 50}, collection, 'points')
    .tables[0];
  const variances = table.columns['variance'] as number[];
  expect(variances.reduce((sum, value) => sum + value, 0) / variances.length).toBeGreaterThan(50);
});

test('kriging CPU work cap names the binding, cell-point limit and remedy before allocating', () => {
  const count = 1501;
  const sample = {
    x: new Float64Array(count),
    y: new Float64Array(count),
    value: new Float64Array(count),
    sourceIndex: new Int32Array(count),
    partIndex: new Int32Array(count),
    count,
    droppedCount: 0
  };
  expect(() =>
    computeKrigingField(
      sample,
      {columns: 200, rows: 200, minX: 0, minY: 0, cellSize: 1},
      {model: 'spherical', nugget: 0, sill: 1, range: 1, neighborCount: 12, searchRadius: 0}
    )
  ).toThrow(/kriging-grid.*cell\/point work.*60,000,000.*increase cellSize/);
  const collection = points([
    {coordinates: [0, 0], properties: {value: 1}},
    {coordinates: [1000, 1000], properties: {value: 2}}
  ]);
  expect(() =>
    run('kriging-grid', {valueProperty: 'value', cellSize: 1}, collection, 'points')
  ).toThrow(/kriging-grid.*40,000 cell limit.*increase cellSize/);
});

test('GWR derives its bandwidth from a 20 km extent and rejects collinear fits above the 10 percent limit', () => {
  const collection = points(
    Array.from({length: 100}, (_, index) => ({
      coordinates: [(index % 10) * 2000, Math.floor(index / 10) * 2000],
      properties: {response: 3 + index * 2, predictor: index, duplicate: index * 2}
    }))
  );
  const result = run(
    'regression-gwr',
    {responseProperty: 'response', predictorProperties: 'predictor'},
    collection,
    'points'
  );
  expect(result.summary['bandwidth']).toBeGreaterThan(1000);
  expect(result.tables[0].columns['fitted'].every(value => value !== null)).toBe(true);
  expect(() =>
    run(
      'regression-gwr',
      {responseProperty: 'response', predictorProperties: 'predictor,duplicate'},
      collection,
      'points'
    )
  ).toThrow(/regression-gwr.*predictorProperties\/bandwidth.*10 percent.*remove collinear/);
});

test('Knox derived thresholds preserve the result when numeric times become epoch milliseconds', () => {
  const collection = points(
    Array.from({length: 12}, (_, index) => ({
      coordinates: [Math.floor(index / 4) * 10000 + (index % 4), 0],
      properties: {time: Math.floor(index / 4) * 100 + (index % 4)}
    }))
  );
  const epoch = {
    ...collection,
    features: collection.features.map(feature => ({
      ...feature,
      properties: {time: 1700000000000 + Number(feature.properties?.['time']) * 86400000}
    }))
  };
  const first = run('knox-test', {timeProperty: 'time', seed: 4}, collection, 'points');
  const second = run('knox-test', {timeProperty: 'time', seed: 4}, epoch, 'points');
  expect(second.tables[0].columns).toEqual(first.tables[0].columns);
  expect(second.summary['temporalThreshold']).toBe(
    Number(first.summary['temporalThreshold']) * 86400000
  );
  expect(first.tables[0].columns['p_value'][0]).toBeLessThan(0.1);
});

test('global Moran and Geary consistently use randomization variances', () => {
  const values = [0, 1, 2, 10];
  const collection = points(
    values.map((value, index) => ({coordinates: [index, 0], properties: {value}}))
  );
  // Enumerate all 24 label permutations for the directed 1-nearest graph, independent of moment formulas.
  const permutations = (remaining: number[]): number[][] =>
    remaining.length
      ? remaining.flatMap((value, index) =>
          permutations(remaining.filter((_, other) => other !== index)).map(tail => [
            value,
            ...tail
          ])
        )
      : [[]];
  const samples = permutations(values).map(order => {
    const mean = order.reduce((sum, value) => sum + value, 0) / 4;
    const denominator = order.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    const neighbors = [1, 0, 1, 2];
    return [
      order.reduce(
        (sum, value, index) => sum + (value - mean) * (order[neighbors[index]] - mean),
        0
      ) / denominator,
      (3 * order.reduce((sum, value, index) => sum + (value - order[neighbors[index]]) ** 2, 0)) /
        (8 * denominator)
    ];
  });
  const result = runGlobalSpatialStatistics(collection, {...weights, valueProperty: 'value'});
  for (let statistic = 0; statistic < 2; statistic++) {
    const mean = samples.reduce((sum, sample) => sum + sample[statistic], 0) / 24;
    const variance = samples.reduce((sum, sample) => sum + (sample[statistic] - mean) ** 2, 0) / 24;
    expect(result.tables[0].columns['variance'][statistic]).toBeCloseTo(variance, 12);
  }
  expect(
    result.diagnostics.find(diagnostic => diagnostic.code === 'analytic-inference')?.message
  ).toContain('both use the randomization variance');
});

test('similar locations requires an explicit referenceIndex rather than row zero', () => {
  const binding = getCatalogBinding('similar-locations')!;
  const collection = points([
    {coordinates: [0, 0], properties: {value: 1}},
    {coordinates: [1, 0], properties: {value: 2}}
  ]);
  expect(() => runCatalogOperation(binding, {properties: 'value'}, {features: collection})).toThrow(
    /similar-locations.*referenceIndex.*explicitly/
  );
  expect(
    runCatalogOperation(binding, {properties: 'value', referenceIndex: 0}, {features: collection})
      .tables[0].rowCount
  ).toBe(2);
});

test('empirical Bayes rejects cases above population and identifies the source row and columns', () => {
  const collection = points([
    {coordinates: [0, 0], properties: {cases: 50, population: 10}},
    {coordinates: [1, 0], properties: {cases: 1, population: 10}}
  ]);
  expect(() =>
    runCatalogOperation(
      getCatalogBinding('empirical-bayes-rates')!,
      {numeratorProperty: 'cases', denominatorProperty: 'population'},
      {features: collection}
    )
  ).toThrow(
    /empirical-bayes-rates.*numeratorProperty "cases" at row 0.*denominatorProperty "population".*50 > 10/
  );
});
