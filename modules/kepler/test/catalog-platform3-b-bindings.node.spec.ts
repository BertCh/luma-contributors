// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  normalizeCatalogParameters,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM3_B_BINDINGS} from '../src/analysis/catalog-platform3-b-bindings';

type Row = {
  coordinates: number[];
  properties?: Record<string, unknown>;
  id?: string;
};

function points(rows: Row[]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map(row => ({
      id: row.id,
      geometry: {type: 'Point', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  };
}

function polygons(
  rows: {ring: number[][]; properties?: Record<string, unknown>; id?: string}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map(row => ({
      id: row.id,
      geometry: {type: 'Polygon', coordinates: [row.ring]},
      properties: row.properties ?? {}
    }))
  };
}

function lines(coordinates: number[][][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map(line => ({
      geometry: {type: 'LineString', coordinates: line},
      properties: {}
    }))
  };
}

const box = (x: number, y: number, width: number, height = width) => [
  [x, y],
  [x + width, y],
  [x + width, y + height],
  [x, y + height],
  [x, y]
];

function run(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
) {
  const binding = getCatalogBinding(id);
  expect(binding, id).toBeDefined();
  const hasCoordinateSystem = binding!.parameters.some(spec => spec.name === 'coordinateSystem');
  return runCatalogOperation(
    binding!,
    hasCoordinateSystem ? {coordinateSystem: 'planar', ...parameters} : parameters,
    inputs
  );
}

function runWithoutCrs(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
) {
  return runCatalogOperation(getCatalogBinding(id)!, parameters, inputs);
}

function table(result: ReturnType<typeof run>, tableId: string) {
  const found = result.tables.find(candidate => candidate.id === tableId);
  expect(found, tableId).toBeDefined();
  return found!;
}

test('family B bindings are complete, runnable and visible in the catalog entries', () => {
  expect(PLATFORM3_B_BINDINGS.length).toBeGreaterThanOrEqual(20);
  const entries = listCatalogEntries({lensScenarios: []});
  const ids = new Set<string>();
  for (const binding of PLATFORM3_B_BINDINGS) {
    expect(ids.has(binding.id), binding.id).toBe(false);
    ids.add(binding.id);
    // PLATFORM4 may wrap a binding with a GPU runner; the registered binding keeps its identity.
    const registered = getCatalogBinding(binding.id)!;
    expect(registered.id).toBe(binding.id);
    expect(() =>
      normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))
    ).not.toThrow();
    expect(binding.declaredBehavior.length).toBeGreaterThan(0);
    expect(binding.limitation).not.toBe('');
    expect(binding.run, binding.id).toBeDefined();
    for (const output of binding.outputs) {
      expect(output.identityColumns.length).toBeGreaterThan(0);
      expect(output.maximumRows).toBeGreaterThan(0);
    }
    if (registered.engine === 'gpu-or-cpu') {
      expect(registered.gpuContributorNames.length).toBeGreaterThan(0);
      expect(registered.runGpu).toBeDefined();
    } else {
      expect(registered.engine).toBe('cpu-reference');
      expect(registered.gpuContributorNames).toEqual([]);
    }
    for (const name of binding.contributorNames) {
      const entry = entries.find(candidate => candidate.contributorName === name);
      expect(entry?.runnable, name).toBe('catalog-binding');
      expect(entry?.bindingIds).toContain(binding.id);
    }
  }
});

test('idw-grid interpolates by radius, reports counts and rejects bad input', () => {
  const samples = points([
    {coordinates: [0, 0], properties: {v: 10}},
    {coordinates: [20, 0], properties: {v: 30}},
    {coordinates: [5, 5], properties: {v: 'bad'}}
  ]);
  const result = run(
    'idw-grid',
    {valueProperty: 'v', cellSize: 10, searchRadius: 8},
    {points: samples}
  );
  const cells = table(result, 'idw-cells');
  expect(cells.columns['value']).toEqual([10, 30, 30]);
  expect(cells.columns['cell_id']).toEqual(['0:0', '0:1', '0:2']);
  expect(result.diagnostics.some(item => item.code === 'feature-dropped')).toBe(true);
  const wide = run(
    'idw-grid',
    {valueProperty: 'v', cellSize: 10, searchRadius: 1000},
    {points: samples}
  );
  const middle = table(wide, 'idw-cells').columns['value'][1] as number;
  expect(middle).toBeGreaterThan(10);
  expect(middle).toBeLessThan(30);
  const neighborLimit = run(
    'idw-grid',
    {valueProperty: 'v', cellSize: 10, searchRadius: 1000, neighborCount: 1},
    {points: samples}
  );
  expect(table(neighborLimit, 'idw-cells').columns['value']).toEqual([10, 30, 30]);
  expect(() => run('idw-grid', {cellSize: 10}, {points: samples})).toThrow('Property required');
  expect(() => run('idw-grid', {valueProperty: 'v', cellSize: 1e-6}, {points: samples})).toThrow(
    'Grid capacity exceeded'
  );
  expect(() => run('idw-grid', {valueProperty: 'missing'}, {points: samples})).toThrow(
    'No usable samples'
  );
  expect(() => run('idw-grid', {valueProperty: 'v', power: 99}, {points: samples})).toThrow(
    'Parameter out of range'
  );
});

test('kernel-density-grid conserves mass and honors weights and padding', () => {
  const samples = points([
    {coordinates: [0, 0], properties: {w: 1}},
    {coordinates: [20, 0], properties: {w: 3}}
  ]);
  const result = run(
    'kernel-density-grid',
    {cellSize: 5, bandwidth: 10, padding: 60, includeEmptyCells: true},
    {points: samples}
  );
  const cells = table(result, 'density-cells');
  const mass = (cells.columns['density'] as number[]).reduce((sum, value) => sum + value * 25, 0);
  expect(mass).toBeGreaterThan(1.9);
  expect(mass).toBeLessThan(2.05);
  const weighted = run(
    'kernel-density-grid',
    {cellSize: 5, bandwidth: 10, padding: 60, weightProperty: 'w', includeEmptyCells: true},
    {points: samples}
  );
  const weightedMass = (table(weighted, 'density-cells').columns['density'] as number[]).reduce(
    (sum, value) => sum + value * 25,
    0
  );
  expect(weightedMass).toBeGreaterThan(3.8);
  expect(weightedMass).toBeLessThan(4.1);
  const sparse = run('kernel-density-grid', {cellSize: 5, bandwidth: 0.5}, {points: samples});
  expect(table(sparse, 'density-cells').rowCount).toBeLessThan(
    (sparse.summary['columns'] as number) * (sparse.summary['rows'] as number)
  );
  expect(() => run('kernel-density-grid', {cellSize: 5, bandwidth: 0}, {points: samples})).toThrow(
    'Parameter out of range'
  );
});

test('point-density-grid counts, sums and averages per cell', () => {
  const samples = points([
    {coordinates: [1, 1], properties: {w: 2}},
    {coordinates: [2, 2], properties: {w: 4}},
    {coordinates: [12, 3], properties: {w: 10}}
  ]);
  const count = table(
    run('point-density-grid', {cellSize: 10}, {points: samples}),
    'density-cells'
  );
  expect(count.columns['value']).toEqual([2, 1]);
  expect(count.columns['sample_count']).toEqual([2, 1]);
  const sum = table(
    run(
      'point-density-grid',
      {cellSize: 10, statistic: 'sum', weightProperty: 'w'},
      {points: samples}
    ),
    'density-cells'
  );
  expect(sum.columns['value']).toEqual([6, 10]);
  const mean = table(
    run(
      'point-density-grid',
      {cellSize: 10, statistic: 'mean', weightProperty: 'w'},
      {points: samples}
    ),
    'density-cells'
  );
  expect(mean.columns['value']).toEqual([3, 10]);
  expect(() =>
    run('point-density-grid', {cellSize: 10, statistic: 'sum'}, {points: samples})
  ).toThrow('Property required');
});

test('line-density-grid splits segments at cell borders', () => {
  const result = run(
    'line-density-grid',
    {cellSize: 10},
    {
      lines: lines([
        [
          [0, 0],
          [20, 20]
        ]
      ])
    }
  );
  const cells = table(result, 'line-density-cells');
  expect(cells.rowCount).toBe(2);
  expect(cells.columns['cell_id']).toEqual(['0:0', '1:1']);
  for (const value of cells.columns['density'] as number[])
    expect(value).toBeCloseTo(Math.hypot(10, 10) / 100, 9);
});

test('line-length-per-polygon measures the inside parts of lines', () => {
  const result = run(
    'line-length-per-polygon',
    {},
    {
      polygons: polygons([
        {ring: box(0, 0, 10), id: 'a'},
        {ring: box(50, 50, 10), id: 'far'}
      ]),
      lines: lines([
        [
          [-5, 5],
          [5, 5]
        ]
      ])
    }
  );
  const rows = table(result, 'polygon-lengths');
  expect(rows.columns['polygon_id']).toEqual(['a', 'far']);
  expect(rows.columns['line_length']).toEqual([5, 0]);
  expect(rows.columns['line_count']).toEqual([1, 0]);
  expect(rows.columns['area']).toEqual([100, 100]);
});

test('kriging-grid returns finite values and non-negative variances', () => {
  const samples = points([
    {coordinates: [0, 0], properties: {v: 1}},
    {coordinates: [100, 0], properties: {v: 3}},
    {coordinates: [0, 100], properties: {v: 2}},
    {coordinates: [100, 100], properties: {v: 4}}
  ]);
  const result = run(
    'kriging-grid',
    {valueProperty: 'v', cellSize: 50, range: 300, sill: 2},
    {points: samples}
  );
  const cells = table(result, 'kriging-cells');
  expect(cells.rowCount).toBe(9);
  for (const value of cells.columns['value'] as number[]) expect(Number.isFinite(value)).toBe(true);
  const first = cells.columns['value'][0] as number;
  expect(first).toBeGreaterThan(0.5);
  expect(first).toBeLessThan(4.5);
  for (const value of cells.columns['variance'] as number[])
    expect(value).toBeGreaterThanOrEqual(0);
  const coincident = points([
    {coordinates: [0, 0], properties: {v: 1}},
    {coordinates: [0, 0], properties: {v: 2}}
  ]);
  const singular = run('kriging-grid', {valueProperty: 'v', cellSize: 10}, {points: coincident});
  expect(singular.diagnostics.some(item => item.code === 'singular-kriging-system')).toBe(true);
});

test('cell-aggregation, rollup and compare share lattice ids', () => {
  const samples = points([
    {coordinates: [0.2, 0.2], properties: {v: 1}},
    {coordinates: [0.4, 0.6], properties: {v: 3}},
    {coordinates: [1.5, 0.5], properties: {v: 10}},
    {coordinates: [3.5, 2.5], properties: {v: 100}}
  ]);
  const aggregated = table(
    runWithoutCrs('cell-aggregation', {cellSize: 1, valueProperty: 'v'}, {points: samples}),
    'cells'
  );
  expect(aggregated.columns['cell_id']).toEqual(['0:0', '1:0', '3:2']);
  expect(aggregated.columns['count']).toEqual([2, 1, 1]);
  expect(aggregated.columns['sum']).toEqual([4, 10, 100]);
  expect(aggregated.columns['mean']).toEqual([2, 10, 100]);
  expect(aggregated.columns['min']).toEqual([1, 10, 100]);
  expect(aggregated.columns['max']).toEqual([3, 10, 100]);
  const rollup = table(
    runWithoutCrs('cell-rollup', {cellSize: 1, levels: 2, factor: 2}, {points: samples}),
    'rollup'
  );
  const identifiers = new Set(rollup.columns['cell_id'] as string[]);
  (rollup.columns['parent_cell_id'] as (string | null)[]).forEach((parent, index) => {
    if (rollup.columns['level'][index] === 2) expect(parent).toBeNull();
    else expect(identifiers.has(parent as string)).toBe(true);
  });
  const topCount = (rollup.columns['count'] as number[]).filter(
    (_, index) => rollup.columns['level'][index] === 2
  );
  expect(topCount.reduce((sum, value) => sum + value, 0)).toBe(4);
  const after = points([
    {coordinates: [0.3, 0.3]},
    {coordinates: [0.4, 0.4]},
    {coordinates: [0.5, 0.5]},
    {coordinates: [5.5, 5.5]}
  ]);
  const comparison = table(
    runWithoutCrs('cell-compare', {cellSize: 1}, {before: samples, after}),
    'comparison'
  );
  const status = Object.fromEntries(
    (comparison.columns['cell_id'] as string[]).map((id, index) => [
      id,
      [comparison.columns['status'][index], comparison.columns['change'][index]]
    ])
  );
  expect(status['0:0']).toEqual(['both', 1]);
  expect(status['1:0']).toEqual(['disappeared', -1]);
  expect(status['5:5']).toEqual(['appeared', 1]);
});

test('zonal-point-statistics summarizes points per polygon', () => {
  const result = run(
    'zonal-point-statistics',
    {valueProperty: 'v'},
    {
      polygons: polygons([
        {ring: box(0, 0, 10), id: 'left'},
        {ring: box(10, 0, 10), id: 'right'},
        {ring: box(100, 100, 10), id: 'empty'}
      ]),
      points: points([
        {coordinates: [1, 1], properties: {v: 2}},
        {coordinates: [2, 2], properties: {v: 4}},
        {coordinates: [15, 5], properties: {v: 7}},
        {coordinates: [50, 50], properties: {v: 9}},
        {coordinates: [3, 3], properties: {v: 'x'}}
      ])
    }
  );
  const zones = table(result, 'zones');
  expect(zones.columns['polygon_id']).toEqual(['left', 'right', 'empty']);
  expect(zones.columns['count']).toEqual([2, 1, 0]);
  expect(zones.columns['sum']).toEqual([6, 7, 0]);
  expect(zones.columns['mean']).toEqual([3, 7, null]);
  expect(zones.columns['standard_deviation'][0]).toBeCloseTo(1, 9);
  expect(result.diagnostics.map(item => item.code)).toEqual(
    expect.arrayContaining(['feature-dropped', 'points-outside-zones'])
  );
});

test('areal-interpolation splits extensive and averages intensive values', () => {
  const sources = polygons([{ring: box(0, 0, 10), properties: {population: 100}}]);
  const targets = polygons([
    {ring: box(0, 0, 5, 10), id: 'west'},
    {ring: box(5, 0, 5, 10), id: 'east'},
    {ring: box(50, 50, 5), id: 'outside'}
  ]);
  const extensive = table(
    run(
      'areal-interpolation',
      {valueProperty: 'population', variableType: 'extensive'},
      {sources, targets}
    ),
    'targets'
  );
  expect(extensive.columns['target_id']).toEqual(['west', 'east', 'outside']);
  expect(extensive.columns['value'][0]).toBeCloseTo(50, 6);
  expect(extensive.columns['value'][1]).toBeCloseTo(50, 6);
  expect(extensive.columns['value'][2]).toBe(0);
  expect(extensive.columns['coverage'][0]).toBeCloseTo(1, 6);
  const intensive = table(
    run(
      'areal-interpolation',
      {valueProperty: 'population', variableType: 'intensive'},
      {sources, targets}
    ),
    'targets'
  );
  expect(intensive.columns['value'][0]).toBeCloseTo(100, 6);
  expect(intensive.columns['value'][2]).toBeNull();
});

const lineOfPoints = (values: number[]) =>
  points(values.map((value, index) => ({coordinates: [index * 10, 0], properties: {v: value}})));

test('spatial-weights builds knn, distance-band and contiguity links', () => {
  const knn = table(
    run('spatial-weights', {neighborCount: 2}, {features: lineOfPoints([1, 2, 3, 4, 5])}),
    'weights'
  );
  expect(knn.rowCount).toBe(10);
  expect(knn.columns['weight'][0]).toBe(0.5);
  const band = table(
    run(
      'spatial-weights',
      {neighborhood: 'distance-band', distanceBand: 10.5, rowStandardize: false},
      {features: lineOfPoints([1, 2, 3])}
    ),
    'weights'
  );
  expect(band.rowCount).toBe(4);
  const grid = polygons([
    {ring: box(0, 0, 10)},
    {ring: box(10, 0, 10)},
    {ring: box(0, 10, 10)},
    {ring: box(10, 10, 10)}
  ]);
  const queen = table(run('spatial-weights', {neighborhood: 'queen'}, {features: grid}), 'weights');
  expect(queen.rowCount).toBe(12);
  const rook = table(run('spatial-weights', {neighborhood: 'rook'}, {features: grid}), 'weights');
  expect(rook.rowCount).toBe(8);
  expect(() =>
    run('spatial-weights', {neighborhood: 'queen'}, {features: lineOfPoints([1, 2])})
  ).toThrow('Polygons required');
});

test('global, local and hot spot statistics detect a clustered surface', () => {
  const values = [10, 11, 12, 10, 11, 12, 1, 2, 1, 2, 1, 2];
  const features = lineOfPoints(values);
  const global = table(
    run('global-spatial-statistics', {valueProperty: 'v', neighborCount: 2}, {features}),
    'global'
  );
  expect(global.columns['statistic']).toEqual(['morans-i', 'gearys-c']);
  expect(global.columns['value'][0]).toBeGreaterThan(0.5);
  expect(global.columns['z_score'][0]).toBeGreaterThan(2);
  expect(global.columns['value'][1]).toBeLessThan(1);
  const moran = table(
    run(
      'local-moran-statistics',
      {valueProperty: 'v', neighborCount: 3, permutations: 99, seed: 7},
      {features}
    ),
    'local-moran'
  );
  expect(moran.rowCount).toBe(12);
  expect(moran.columns['local_i'][1]).toBeGreaterThan(0);
  expect(moran.columns['cluster'][1]).not.toBe('not-tested');
  const again = table(
    run(
      'local-moran-statistics',
      {valueProperty: 'v', neighborCount: 3, permutations: 99, seed: 7},
      {features}
    ),
    'local-moran'
  );
  expect(again.columns['p_value']).toEqual(moran.columns['p_value']);
  const untested = table(
    run(
      'local-moran-statistics',
      {valueProperty: 'v', neighborCount: 3, permutations: 0},
      {features}
    ),
    'local-moran'
  );
  expect(untested.columns['cluster'][0]).toBe('not-tested');
  const hot = table(
    run('hotspot-gi-star', {valueProperty: 'v', neighborCount: 3}, {features}),
    'hotspots'
  );
  expect(hot.columns['z_score'][1]).toBeGreaterThan(0);
  expect(hot.columns['z_score'][10]).toBeLessThan(0);
  const geary = table(
    run(
      'local-geary-statistics',
      {valueProperty: 'v', neighborCount: 2, permutations: 49, seed: 3},
      {features}
    ),
    'local-geary'
  );
  expect(geary.rowCount).toBe(12);
  expect(geary.columns['local_c'][1]).toBeLessThan(geary.columns['local_c'][5] as number);
  expect(() =>
    run(
      'global-spatial-statistics',
      {valueProperty: 'v'},
      {features: lineOfPoints([5, 5, 5, 5, 5])}
    )
  ).toThrow('Constant values');
  expect(() =>
    run('hotspot-gi-star', {valueProperty: 'v'}, {features: lineOfPoints([1, 2])})
  ).toThrow('Too few features');
});

test('empirical-bayes-rates shrinks small populations more', () => {
  const areas = points([
    {coordinates: [0, 0], properties: {events: 1, population: 10}, id: 'small'},
    {coordinates: [1, 0], properties: {events: 100, population: 1000}, id: 'large'},
    {coordinates: [2, 0], properties: {events: 50, population: 1000}, id: 'low'},
    {coordinates: [3, 0], properties: {events: 3, population: 0}, id: 'invalid'}
  ]);
  const rates = table(
    run(
      'empirical-bayes-rates',
      {numeratorProperty: 'events', denominatorProperty: 'population'},
      {features: areas}
    ),
    'rates'
  );
  expect(rates.columns['source_id']).toEqual(['small', 'large', 'low']);
  const shrinkage = rates.columns['shrinkage'] as number[];
  expect(shrinkage[0]).toBeLessThan(shrinkage[1]);
  const global = (1 + 100 + 50) / 2010;
  const raw = rates.columns['raw_rate'][0] as number;
  const smoothed = rates.columns['eb_rate'][0] as number;
  expect(Math.abs(smoothed - global)).toBeLessThan(Math.abs(raw - global));
});

test('multiple-testing-adjustment matches hand-computed adjustments', () => {
  const tests = points(
    [0.01, 0.04, 0.03, 0.005].map((p, index) => ({
      coordinates: [index, 0],
      properties: {p}
    }))
  );
  const adjust = (method: string) =>
    table(
      run('multiple-testing-adjustment', {pValueProperty: 'p', method}, {features: tests}),
      'adjusted'
    ).columns['adjusted_p_value'] as number[];
  const near = (actual: number[], expected: number[]) =>
    expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 9));
  near(adjust('bonferroni'), [0.04, 0.16, 0.12, 0.02]);
  near(adjust('holm'), [0.03, 0.06, 0.06, 0.02]);
  near(adjust('benjamini-hochberg'), [0.02, 0.04, 0.04, 0.02]);
  expect(adjust('sidak')[3]).toBeCloseTo(1 - 0.995 ** 4, 9);
  const rejected = table(
    run(
      'multiple-testing-adjustment',
      {pValueProperty: 'p', method: 'bonferroni', alpha: 0.05},
      {features: tests}
    ),
    'adjusted'
  ).columns['rejected'];
  expect(rejected).toEqual([true, false, false, true]);
  expect(() =>
    run('multiple-testing-adjustment', {pValueProperty: 'nothing'}, {features: tests})
  ).toThrow('No usable samples');
});

test('knox-test finds excess joint space-time pairs', () => {
  const events = points([
    {coordinates: [0, 0], properties: {t: 0}},
    {coordinates: [1, 0], properties: {t: 0.5}},
    {coordinates: [2, 0], properties: {t: 1}},
    {coordinates: [1000, 0], properties: {t: 100}},
    {coordinates: [1001, 0], properties: {t: 100.5}},
    {coordinates: [1002, 0], properties: {t: 101}}
  ]);
  const result = table(
    run(
      'knox-test',
      {timeProperty: 't', spatialThreshold: 5, temporalThreshold: 2, permutations: 199, seed: 5},
      {points: events}
    ),
    'knox'
  );
  expect(result.columns['joint_close_pairs'][0]).toBe(6);
  expect(result.columns['spatially_close_pairs'][0]).toBe(6);
  expect(result.columns['p_value'][0]).toBeLessThan(0.1);
  const many = points(
    Array.from({length: 2001}, (_, index) => ({coordinates: [index, 0], properties: {t: index}}))
  );
  expect(() => run('knox-test', {timeProperty: 't', spatialThreshold: 5}, {points: many})).toThrow(
    'Input feature capacity exceeded'
  );
});

const blobs = () =>
  points([
    ...[0, 1, 2, 3].map(index => ({coordinates: [index, index % 2]})),
    ...[0, 1, 2, 3].map(index => ({coordinates: [100 + index, index % 2]})),
    {coordinates: [50, 500]}
  ]);

test('density-clustering labels canonical DBSCAN clusters and noise', () => {
  const result = run('density-clustering', {epsilon: 3, minimumPoints: 3}, {points: blobs()});
  const clustered = table(result, 'points');
  expect(clustered.columns['cluster']).toEqual([0, 0, 0, 0, 1, 1, 1, 1, -1]);
  expect(clustered.columns['is_core'][8]).toBe(false);
  const clusters = table(result, 'clusters');
  expect(clusters.columns['cluster']).toEqual([0, 1]);
  expect(clusters.columns['size']).toEqual([4, 4]);
  expect(result.summary['noiseCount']).toBe(1);
  expect(() => run('density-clustering', {epsilon: 0}, {points: blobs()})).toThrow(
    'Parameter out of range'
  );
});

test('optics-ordering extracts the same two clusters', () => {
  const result = run(
    'optics-ordering',
    {maximumDistance: 10, minimumSamples: 3, extractionDistance: 3},
    {points: blobs()}
  );
  const ordered = table(result, 'points');
  const labels = ordered.columns['cluster'] as number[];
  expect(new Set(labels.slice(0, 4)).size).toBe(1);
  expect(new Set(labels.slice(4, 8)).size).toBe(1);
  expect(labels[0]).not.toBe(labels[4]);
  expect(labels[8]).toBe(-1);
  expect(new Set(ordered.columns['optics_order']).size).toBe(9);
});

test('local-outlier-factor scores an isolated point highest', () => {
  const samples = points([
    ...Array.from({length: 9}, (_, index) => ({coordinates: [index % 3, Math.floor(index / 3)]})),
    {coordinates: [30, 30]}
  ]);
  const result = table(
    run('local-outlier-factor', {neighborCount: 3, threshold: 1.5}, {points: samples}),
    'outliers'
  );
  const scores = result.columns['lof'] as number[];
  expect(Math.max(...scores)).toBe(scores[9]);
  expect(result.columns['is_outlier'][9]).toBe(true);
  expect(result.columns['is_outlier'][4]).toBe(false);
  expect(() => run('local-outlier-factor', {neighborCount: 20}, {points: samples})).toThrow(
    'Too few features'
  );
});

test('hierarchical-density-clustering separates blobs and drops small groups', () => {
  const result = table(
    run(
      'hierarchical-density-clustering',
      {minimumSamples: 2, minimumClusterSize: 3, cutDistance: 5},
      {points: blobs()}
    ),
    'points'
  );
  expect(result.columns['cluster']).toEqual([0, 0, 0, 0, 1, 1, 1, 1, -1]);
});

test('similar-locations ranks by standardized attribute distance', () => {
  const places = points([
    {coordinates: [0, 0], properties: {a: 1, b: 1}, id: 'reference'},
    {coordinates: [1, 0], properties: {a: 1.1, b: 1}, id: 'close'},
    {coordinates: [2, 0], properties: {a: 5, b: 9}, id: 'far'},
    {coordinates: [3, 0], properties: {a: 2}, id: 'incomplete'}
  ]);
  const result = table(
    run(
      'similar-locations',
      {properties: 'a,b', referenceIndex: 0, resultCount: 1},
      {features: places}
    ),
    'similarity'
  );
  expect(result.columns['source_id']).toEqual(['reference', 'close', 'far']);
  expect(result.columns['rank']).toEqual([0, 1, 2]);
  expect(result.columns['is_similar']).toEqual([false, true, false]);
  expect(() =>
    run('similar-locations', {properties: 'a,b', referenceIndex: 3}, {features: places})
  ).toThrow('Reference required');
});

test('location-allocation picks the medians of two demand clusters', () => {
  const demand = points([
    {coordinates: [0, 0]},
    {coordinates: [2, 0]},
    {coordinates: [1, 2]},
    {coordinates: [100, 0]},
    {coordinates: [102, 0]},
    {coordinates: [101, 2]}
  ]);
  const candidates = points([
    {coordinates: [1, 0.5], id: 'west'},
    {coordinates: [50, 0], id: 'middle'},
    {coordinates: [101, 0.5], id: 'east'}
  ]);
  const result = run('location-allocation', {facilityCount: 2}, {demand, candidates});
  expect(table(result, 'facilities').columns['source_id']).toEqual(['west', 'east']);
  const assignments = table(result, 'assignments');
  expect(assignments.columns['facility_id']).toEqual([
    'west',
    'west',
    'west',
    'east',
    'east',
    'east'
  ]);
  expect(() => run('location-allocation', {facilityCount: 5}, {demand, candidates})).toThrow(
    'Too few features'
  );
});

test('skater-regions cuts the tree where attributes change', () => {
  const strip = polygons([
    {ring: box(0, 0, 10), properties: {x: 1}},
    {ring: box(10, 0, 10), properties: {x: 1.2}},
    {ring: box(20, 0, 10), properties: {x: 10}},
    {ring: box(30, 0, 10), properties: {x: 10.3}}
  ]);
  const result = table(
    run('skater-regions', {properties: 'x', regionCount: 2}, {features: strip}),
    'regions'
  );
  const regions = result.columns['region'] as number[];
  expect(regions[0]).toBe(regions[1]);
  expect(regions[2]).toBe(regions[3]);
  expect(regions[0]).not.toBe(regions[2]);
  const tooMany = run(
    'skater-regions',
    {properties: 'x', regionCount: 4, minimumRegionSize: 2},
    {features: strip}
  );
  expect(tooMany.diagnostics.some(item => item.code === 'fewer-regions')).toBe(true);
  expect(() => run('skater-regions', {properties: 'x', regionCount: 9}, {features: strip})).toThrow(
    'Too few features'
  );
});

test('EPSG:4326 inputs use local meters', () => {
  const samples = points([
    {coordinates: [10, 50], properties: {v: 1}},
    {coordinates: [10.001, 50], properties: {v: 2}}
  ]);
  const result = runWithoutCrs(
    'point-density-grid',
    {cellSize: 50, coordinateSystem: 'EPSG:4326'},
    {points: samples}
  );
  // 0.001 degrees of longitude at 50 degrees latitude is about 71.5 meters.
  expect(result.summary['columns']).toBe(2);
  const geometry = JSON.parse(table(result, 'density-cells').columns['geometry'][0] as string);
  expect(geometry.coordinates[0][0][0]).toBeCloseTo(10, 9);
});
