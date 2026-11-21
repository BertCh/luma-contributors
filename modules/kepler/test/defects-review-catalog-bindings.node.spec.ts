// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test, vi} from 'vitest';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';
import * as gpuSession from '../src/analysis/catalog-gpu-session';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {makeLocalFrame} from '../src/analysis/catalog-local-frame';
import {getFeatureAnchor, readPointSamples} from '../src/analysis/catalog-platform3-b-common';
import {resolveKrigingSettings} from '../src/analysis/catalog-platform3-b-grids';

function points(coordinates: number[][], values: number[] = []): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      id: String(index),
      geometry: {type: 'Point', coordinates: position},
      properties: {value: values[index], time: index * 10}
    }))
  };
}

function run(
  id: string,
  parameters: Record<string, unknown>,
  collection: CatalogFeatureCollection,
  input = 'points'
) {
  return runCatalogOperation(
    getCatalogBinding(id)!,
    {coordinateSystem: 'planar', ...parameters},
    {[input]: collection}
  );
}

const gridCoordinates = Array.from({length: 9}, (_, index) => [index % 3, Math.floor(index / 3)]);
const gridValues = [1, 2, 4, 3, 6, 7, 5, 8, 9];

test.each([1e4, 1e-7])('review 1: default kriging preserves units scaled by %s', factor => {
  const parameters = {valueProperty: 'value', cellSize: 1};
  const original = run('kriging-grid', parameters, points(gridCoordinates, gridValues));
  const scaled = run(
    'kriging-grid',
    parameters,
    points(
      gridCoordinates,
      gridValues.map(value => value * factor)
    )
  );
  expect(original.tables[0].rowCount).toBe(9);
  expect(scaled.tables[0].rowCount).toBe(9);
  for (const [column, multiplier] of [
    ['value', factor],
    ['variance', factor ** 2]
  ] as const) {
    original.tables[0].columns[column].forEach((value, index) => {
      const expected = Number(value) * multiplier;
      const actual = Number(scaled.tables[0].columns[column][index]);
      expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * 1e-9);
    });
  }
});

test.each([
  'local-geary-statistics',
  'local-moran-statistics',
  'hotspot-gi-star'
])('review 2: %s preserves inference by ID after reversing the 25 review rows', id => {
  const collection = points(
    Array.from({length: 25}, (_, index) => [index % 5, Math.floor(index / 5)]),
    Array.from({length: 25}, (_, index) => Math.sin(7 * index))
  );
  const parameters = {
    valueProperty: 'value',
    neighborhood: 'knn',
    neighborCount: 3,
    ...(id === 'hotspot-gi-star' ? {} : {permutations: 199, seed: 42})
  };
  const original = run(id, parameters, collection, 'features').tables[0];
  const reversed = run(
    id,
    parameters,
    {...collection, features: [...collection.features].reverse()},
    'features'
  ).tables[0];
  const byIdentity = (table: typeof original) =>
    table.columns['source_id']
      .map((identity, index) => [identity, table.columns['p_value'][index]])
      .sort((first, second) => String(first[0]).localeCompare(String(second[0])));
  expect(byIdentity(reversed)).toEqual(byIdentity(original));
  expect(reversed.columns['source_id']).toEqual(
    collection.features.map(feature => feature.id).reverse()
  );
});

test('review 2: Gi star permutations map row-keyed GPU readback through canonical identities', async () => {
  const collection = points(
    Array.from({length: 25}, (_, index) => [index % 5, Math.floor(index / 5)]),
    Array.from({length: 25}, (_, index) => 2 + Math.sin(7 * index))
  );
  // Philox draws depend on the uploaded row index. Distinct synthetic readback values expose
  // any source-order dependence without constructing buffers or executing a GPU contributor.
  const session = vi.spyOn(gpuSession, 'withCatalogGpuSession').mockResolvedValue({
    observed: new Float32Array(25),
    pValues: Float32Array.from({length: 25}, (_, index) => (index + 1) / 200),
    // At alpha 0.05, the first ten literal p-values are significant (p <= alpha).
    significant: Uint32Array.from({length: 25}, (_, index) => Number(index < 10))
  });
  try {
    const binding = getCatalogBinding('local-lisa-variants')!;
    const parameters = normalizeCatalogParameters(binding, {
      variant: 'local-getis-ord-g-star',
      coordinateSystem: 'planar',
      valueProperty: 'value',
      neighborhood: 'knn',
      neighborCount: 3,
      permutations: 199,
      seed: 42
    });
    const context = {device: null!};
    const original = (await binding.runGpu!(context, {features: collection}, parameters)).tables[0];
    const reversed = (
      await binding.runGpu!(
        context,
        {features: {...collection, features: [...collection.features].reverse()}},
        parameters
      )
    ).tables[0];
    const byIdentity = (table: typeof original) =>
      table.columns['source_id']
        .map((identity, index) => [identity, table.columns['p_value'][index]])
        .sort((first, second) => String(first[0]).localeCompare(String(second[0])));
    expect(byIdentity(reversed)).toEqual(byIdentity(original));
    expect(reversed.columns['source_id']).toEqual(
      collection.features.map(feature => feature.id).reverse()
    );
  } finally {
    session.mockRestore();
  }
});

test('review 3: explicit zero Knox thresholds keep simultaneous/coincident events distinct from defaults', () => {
  const collection = points([
    [0, 0],
    [1, 0],
    [2, 0]
  ]);
  const result = run(
    'knox-test',
    {timeProperty: 'time', spatialThreshold: 3, temporalThreshold: 0},
    collection
  );
  expect(result.tables[0].columns['temporally_close_pairs']).toEqual([0]);
  expect(result.tables[0].columns['joint_close_pairs']).toEqual([0]);
  expect(result.summary['temporalThreshold']).toBe(0);
  const coincident = run(
    'knox-test',
    {timeProperty: 'time', spatialThreshold: 0, temporalThreshold: 30},
    collection
  );
  expect(coincident.tables[0].columns['spatially_close_pairs']).toEqual([0]);
  for (const parameters of [
    {timeProperty: 'time'},
    {timeProperty: 'time', temporalThreshold: undefined}
  ]) {
    expect(run('knox-test', parameters, collection).summary['temporalThreshold']).toBe(10);
  }
});

test('review 3: kriging honors explicit zero nugget and sill, and rejects an explicit zero range', () => {
  const sample = readPointSamples(
    points(gridCoordinates, gridValues),
    'value',
    makeLocalFrame('planar', [])
  );
  const settings = resolveKrigingSettings(sample, {
    model: 'spherical',
    nugget: 2,
    sill: 0,
    range: 3,
    neighborCount: 9,
    searchRadius: 0
  });
  expect(settings).toMatchObject({nugget: 2, sill: 0, range: 3});
  expect(resolveKrigingSettings(sample, {nugget: 0}).nugget).toBe(0);
  expect(resolveKrigingSettings(sample, {}).range).toBeGreaterThan(0);
  expect(() =>
    run(
      'kriging-grid',
      {valueProperty: 'value', cellSize: 1, range: 0},
      points(gridCoordinates, gridValues)
    )
  ).toThrow();
});

test('review 3: GWR derives only omitted bandwidths and preserves explicit positive bandwidths', () => {
  const collection = points(
    Array.from({length: 12}, (_, index) => [index * 10, 0]),
    Array.from({length: 12}, (_, index) => index ** 2)
  );
  const parameters = {responseProperty: 'value', predictorProperties: 'time'};
  expect(run('regression-gwr', parameters, collection).summary['bandwidth']).toBeGreaterThan(0);
  expect(
    run('regression-gwr', {...parameters, bandwidth: 1000}, collection).summary['bandwidth']
  ).toBe(1000);
  expect(() => run('regression-gwr', {...parameters, bandwidth: 0}, collection)).toThrow();
});

test.each([
  0, 1, 2, 3
])('review 4: S2 level %s clamps default and saved capacities with diagnostics', level => {
  const collection = points([[0, 0]]);
  const capacity = Math.min(6144, 6 * 4 ** level);
  const binding = getCatalogBinding('s2-cap-covering')!;
  const runCover = (parameters: Record<string, unknown>) =>
    runCatalogOperation(binding, parameters, {points: collection});
  const reference = runCover({level, maximumCells: capacity});
  for (const parameters of [{level}, {level, maximumCells: 512}, {level, maximumCells: 6144}]) {
    const result = runCover(parameters);
    expect(result.tables).toEqual(reference.tables);
    expect(result.tables[0].rowCount).toBeGreaterThan(0);
    if ((parameters.maximumCells ?? 384) > capacity) {
      const diagnostic = result.diagnostics.find(
        entry => entry.code === 's2-maximum-cells-clamped'
      );
      expect(diagnostic?.message).toContain('maximumCells');
      expect(diagnostic?.message).toContain(String(capacity));
    } else {
      expect(result.diagnostics.some(entry => entry.code === 's2-maximum-cells-clamped')).toBe(
        false
      );
    }
  }
});

test('review 4: S2 clamps explicit requests above the global contributor cap', () => {
  const result = runCatalogOperation(
    getCatalogBinding('s2-cap-covering')!,
    {level: 6, maximumCells: 10000},
    {points: points([[0, 0]])}
  );
  expect(result.tables[0].rowCount).toBeGreaterThan(0);
  expect(
    result.diagnostics.find(entry => entry.code === 's2-maximum-cells-clamped')?.message
  ).toContain('6144');
});

test('review 5: DBSCAN never counts a point twice when neighboring bucket keys collide', () => {
  const result = run(
    'density-clustering',
    {epsilon: 1e-9, minimumSamples: 2},
    points([
      [0, 0],
      [1e9, 0]
    ])
  ).tables[0];
  expect(result.columns['cluster']).toEqual([-1, -1]);
  expect(result.columns['is_core']).toEqual([false, false]);
});

test('review 6: distant multipart squares retain equal area and their weighted centroid', () => {
  const square = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
    [0, 0]
  ];
  const shifted = square.map(([x, y]) => [x + 1e9, y + 1e9]);
  for (const polygons of [
    [[square], [shifted]],
    [[shifted], [square]]
  ]) {
    const anchor = getFeatureAnchor(
      {geometry: {type: 'MultiPolygon', coordinates: polygons}},
      makeLocalFrame('planar', [])
    )!;
    for (const coordinate of anchor) {
      expect(Math.abs((coordinate - 500000005) / 500000005)).toBeLessThan(1e-6);
      expect(coordinate).toBe(500000005);
    }
  }
});

test('review 7: DBSCAN excludes represented out-of-radius distances without a frame shift', () => {
  for (const distance of [1, 1.000000000000001]) {
    const result = run(
      'density-clustering',
      {epsilon: 1, minimumSamples: 2},
      points([
        [0, 0],
        [distance, 0]
      ])
    ).tables[0];
    expect(result.columns['cluster']).toEqual(distance === 1 ? [0, 0] : [-1, -1]);
    expect(result.columns['is_core']).toEqual(distance === 1 ? [true, true] : [false, false]);
  }
});

test('review 7: DBSCAN still includes the original exactly-one boundary after local translation', () => {
  const result = run(
    'density-clustering',
    {epsilon: 1, minimumSamples: 2},
    points([
      [-0.1, 100],
      [0.1, 0],
      [1.1, 0],
      [5.1, 0],
      [6.1, 0]
    ])
  ).tables[0];
  expect(result.columns['cluster']).toEqual([-1, 0, 0, 1, 1]);
});

test('review 8: both descriptor overlap fixtures guard the measured counts exactly', () => {
  // Audit the assertions themselves: the measured fixture also passed with the weakened guards.
  const source = readFileSync(
    new URL('./descriptors/descriptors-network-aggregation-raster.node.spec.ts', import.meta.url),
    'utf8'
  );
  const scenario = source.slice(
    source.indexOf("it('retains the distribution coverage warning"),
    source.indexOf("it('raster-join-points-in-polygons: the action")
  );
  expect(scenario).toContain(
    "['recipe-points-in-polygons-choropleth', 'raster-join-points-in-polygons']"
  );
  expect(scenario).toMatch(/expect\(counts\)\.toEqual\(\[7,\s*7\]\)/);
  expect(scenario).toContain("expect(result.summary['multipleMatchCount']).toBe(7)");
});
