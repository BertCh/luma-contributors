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
import {PLATFORM4_NEW_BINDINGS} from '../src/analysis/catalog-platform4-new-bindings';
import {
  getDggsCpuBoundary,
  getDggsCpuCell,
  getQuadbinParent
} from '../src/analysis/catalog-platform4-dggs-cells';

function points(
  rows: {coordinates: number[]; properties?: Record<string, unknown>}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map(row => ({
      geometry: {type: 'Point', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  };
}

function polygon(ring: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [{geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}}]
  };
}

const NEW_IDS = [
  'dggs-point-index',
  'dggs-cell-aggregation',
  'dggs-cell-rollup',
  'dggs-cell-compare',
  'weights-algebra',
  'weights-transpose',
  'weights-transform',
  'regression-ols',
  'spherical-point-in-polygon',
  'raster-join-points-in-polygons'
];

test('PLATFORM4 new bindings are registered, complete and runnable in the catalog entries', () => {
  expect(PLATFORM4_NEW_BINDINGS.map(binding => binding.id)).toEqual(NEW_IDS);
  const entries = listCatalogEntries({lensScenarios: []});
  for (const binding of PLATFORM4_NEW_BINDINGS) {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(() =>
      normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))
    ).not.toThrow();
    expect(binding.engine).toBe('gpu-or-cpu');
    expect(binding.run).toBeDefined();
    expect(binding.runGpu).toBeDefined();
    expect(binding.gpuContributorNames.length).toBeGreaterThan(0);
    expect(binding.declaredBehavior.length).toBeGreaterThan(0);
    expect(binding.limitation).not.toBe('');
    for (const contributorName of binding.gpuContributorNames) {
      const entry = entries.find(candidate => candidate.contributorName === contributorName);
      // The raster contributors are not ontology operators, so the catalog listing has no row for them.
      // The chunked weights classes (PLATFORM5) execute the ontology contributors and have no row of their own.
      if (!entry && contributorName.startsWith('GPUChunked')) continue;
      if (!entry && binding.id === 'raster-join-points-in-polygons') continue;
      expect(entry, `${binding.id} ${contributorName}`).toBeDefined();
      expect(entry?.runnable).toBe('catalog-binding');
      expect(entry?.bindingIds).toContain(binding.id);
    }
  }
});

test('DGGS CPU keys match published Quadbin, geohash and S2 references', () => {
  // quadbin-py: point_to_cell(-3.7038, 40.4168, 4) == 5209574053332910079 (Madrid, zoom 4 tile 7/6)
  expect(getDggsCpuCell('quadbin', -3.7038, 40.4168, 4).toString(16)).toMatch(/^4[0-9a-f]{15}$/);
  // geohash of (-5.6, 42.6) at length 5 is "ezs42" (Wikipedia example)
  const geohash = getDggsCpuCell('geohash', -5.6, 42.6, 5);
  expect(geohash >> 60n).toBe(5n);
  expect((geohash & ((1n << 60n) - 1n)).toString(2).padStart(25, '0')).toBe(
    [...'ezs42']
      .map(character =>
        '0123456789bcdefghjkmnpqrstuvwxyz'.indexOf(character).toString(2).padStart(5, '0')
      )
      .join('')
  );
  // S2 published level-15 token for lat 0, lng 0 is face 0 (token starts with 1).
  const s2 = getDggsCpuCell('s2', 0, 0, 15);
  expect(s2 >> 61n).toBe(0n);
  expect(s2 & 1n).toBe(0n);
  // Quadbin parent contains the child boundary.
  const child = getDggsCpuCell('quadbin', 10.5, 20.5, 8);
  const parent = getQuadbinParent(child, 5);
  expect(parent).toBe(getDggsCpuCell('quadbin', 10.5, 20.5, 5));
  const parentBox = getDggsCpuBoundary('quadbin', parent);
  const childBox = getDggsCpuBoundary('quadbin', child);
  expect(childBox[0][0]).toBeGreaterThanOrEqual(parentBox[0][0] - 1e-9);
  expect(childBox[2][0]).toBeLessThanOrEqual(parentBox[2][0] + 1e-9);
  // Every family boundary is a closed ring around the point.
  for (const family of ['quadbin', 'quadkey', 'geohash', 's2'] as const) {
    const ring = getDggsCpuBoundary(family, getDggsCpuCell(family, 12.25, 47.75, 6));
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring[4]);
  }
});

test('dggs-point-index and dggs-cell-aggregation count points per Quadbin cell', () => {
  const data = points([
    {coordinates: [10.01, 50.01], properties: {v: 2}},
    {coordinates: [10.02, 50.02], properties: {v: 4}},
    {coordinates: [-70, -33], properties: {v: 1}},
    {coordinates: [200, 0], properties: {v: 1}},
    {coordinates: [5, 5], properties: {v: 'x'}}
  ]);
  const index = runCatalogOperation(
    getCatalogBinding('dggs-point-index')!,
    {family: 'geohash', resolution: 5},
    {points: data}
  );
  expect(index.tables[0].rowCount).toBe(4);
  expect(index.diagnostics.find(item => item.code === 'feature-dropped')?.count).toBe(1);
  const aggregation = runCatalogOperation(
    getCatalogBinding('dggs-cell-aggregation')!,
    {family: 'quadbin', resolution: 10, valueProperty: 'v'},
    {points: data}
  );
  const table = aggregation.tables[0];
  expect(aggregation.engine).toBe('cpu-reference');
  expect(table.rowCount).toBe(2);
  expect([...(table.columns['count'] as number[])].sort()).toEqual([1, 2]);
  const twin = (table.columns['count'] as number[]).indexOf(2);
  expect(table.columns['sum'][twin]).toBe(6);
  expect(table.columns['mean'][twin]).toBe(3);
  expect(table.columns['min'][twin]).toBe(2);
  expect(table.columns['max'][twin]).toBe(4);
  expect(String(table.columns['geometry'][twin])).toContain('Polygon');
  // H3 has no CPU engine.
  expect(() =>
    runCatalogOperation(
      getCatalogBinding('dggs-cell-aggregation')!,
      {family: 'h3', resolution: 5},
      {points: data}
    )
  ).toThrow(/H3/);
  expect(() =>
    runCatalogOperation(
      getCatalogBinding('dggs-cell-aggregation')!,
      {family: 'quadbin', resolution: 40},
      {points: data}
    )
  ).toThrow();
});

test('dggs-cell-rollup nests parents exactly and dggs-cell-compare reports appeared cells', () => {
  const data = points(
    Array.from({length: 40}, (_, index) => ({
      coordinates: [(index * 7.3) % 90, (index * 3.1) % 60],
      properties: {v: index}
    }))
  );
  const rollup = runCatalogOperation(
    getCatalogBinding('dggs-cell-rollup')!,
    {family: 'quadbin', resolution: 8, levels: 3, levelStep: 2, valueProperty: 'v'},
    {points: data}
  ).tables[0];
  const levels = rollup.columns['level'] as number[];
  for (let level = 0; level <= 3; level++) {
    const total = (rollup.columns['count'] as number[]).filter((_, row) => levels[row] === level);
    expect(total.reduce((sum, count) => sum + count, 0)).toBe(40);
  }
  const ids = new Set(rollup.columns['cell_id'] as string[]);
  (rollup.columns['parent_cell_id'] as (string | null)[]).forEach((parent, row) => {
    if (levels[row] < 3) expect(ids.has(parent as string)).toBe(true);
    else expect(parent).toBeNull();
  });
  expect(() =>
    runCatalogOperation(
      getCatalogBinding('dggs-cell-rollup')!,
      {family: 'quadbin', resolution: 3, levels: 3, levelStep: 2},
      {points: data}
    )
  ).toThrow(/below resolution 0/);

  const compare = runCatalogOperation(
    getCatalogBinding('dggs-cell-compare')!,
    {family: 'quadbin', resolution: 6, measure: 'count'},
    {
      before: points([
        {coordinates: [10, 10]},
        {coordinates: [10.1, 10.1]},
        {coordinates: [-40, 5]}
      ]),
      after: points([{coordinates: [10, 10]}, {coordinates: [80, 40]}])
    }
  ).tables[0];
  expect([...(compare.columns['status'] as string[])].sort()).toEqual([
    'appeared',
    'disappeared',
    'present'
  ]);
  const present = (compare.columns['status'] as string[]).indexOf('present');
  expect(compare.columns['delta'][present]).toBe(-1);
  expect(compare.columns['ratio'][present]).toBe(0.5);
  expect(compare.columns['percent_change'][present]).toBe(-50);
});

test('weights bindings build, combine, transpose and standardize CSR weights', () => {
  const grid = points(
    Array.from({length: 36}, (_, index) => ({
      coordinates: [
        (index % 6) * 100 + (index % 5) * 7,
        Math.floor(index / 6) * 100 + (index % 3) * 5
      ]
    }))
  );
  const edges = (id: string, parameters: Record<string, unknown>) =>
    runCatalogOperation(
      getCatalogBinding(id)!,
      {
        coordinateSystem: 'planar',
        neighborCount: 3,
        ...(id === 'weights-algebra' ? {distanceBand: 150} : {}),
        ...parameters
      },
      {points: grid}
    ).tables[0];
  const union = edges('weights-algebra', {operation: 'union'});
  const intersection = edges('weights-algebra', {operation: 'intersection'});
  const difference = edges('weights-algebra', {operation: 'difference'});
  const symmetric = edges('weights-algebra', {operation: 'symmetricDifference'});
  // |A u B| = |A| + |B| - |A n B| and A xor B = (A u B) minus (A n B).
  expect(union.rowCount - intersection.rowCount).toBe(symmetric.rowCount);
  expect(difference.rowCount).toBeLessThanOrEqual(36 * 3);
  expect(intersection.rowCount).toBeGreaterThan(0);
  const summed = edges('weights-algebra', {operation: 'intersection', weightRule: 'sum'});
  expect((summed.columns['weight'] as number[]).every(weight => weight > 1 && weight <= 2)).toBe(
    true
  );
  const transpose = edges('weights-transpose', {});
  expect(transpose.rowCount).toBe(36 * 3);
  const binary = edges('weights-transform', {operation: 'binary'});
  const forward = new Set(
    (binary.columns['source_index'] as number[]).map(
      (source, row) => `${source}>${binary.columns['target_index'][row]}`
    )
  );
  const reversed = new Set(
    (transpose.columns['source_index'] as number[]).map(
      (source, row) => `${transpose.columns['target_index'][row]}>${source}`
    )
  );
  expect(reversed).toEqual(forward);
  const rowStandardized = edges('weights-transform', {operation: 'row', decayDistance: 200});
  const sums = new Map<number, number>();
  (rowStandardized.columns['source_index'] as number[]).forEach((source, row) =>
    sums.set(source, (sums.get(source) ?? 0) + (rowStandardized.columns['weight'][row] as number))
  );
  for (const total of sums.values()) expect(total).toBeCloseTo(1, 6);
  const double = edges('weights-transform', {operation: 'double', decayDistance: 200});
  expect(
    (double.columns['weight'] as number[]).reduce((sum, weight) => sum + weight, 0)
  ).toBeCloseTo(1, 6);
});

test('regression-ols recovers an exact linear model and rejects collinear predictors', () => {
  const rows = Array.from({length: 30}, (_, index) => {
    const a = Math.sin(index) * 5;
    const b = (index % 7) - 3;
    return {
      coordinates: [index, index],
      properties: {a, b, y: 2 + 3 * a - 0.5 * b + (index % 2 ? 0.01 : -0.01)}
    };
  });
  const result = runCatalogOperation(
    getCatalogBinding('regression-ols')!,
    {responseProperty: 'y', predictorProperties: 'a, b'},
    {points: points(rows)}
  );
  const [coefficients, residuals] = result.tables;
  expect(coefficients.columns['term']).toEqual(['(intercept)', 'a', 'b']);
  expect(coefficients.columns['estimate'][0] as number).toBeCloseTo(2, 1);
  expect(coefficients.columns['estimate'][1] as number).toBeCloseTo(3, 2);
  expect(coefficients.columns['estimate'][2] as number).toBeCloseTo(-0.5, 2);
  expect(residuals.rowCount).toBe(30);
  expect(result.summary['rSquared']).toBeGreaterThan(0.9999);
  const collinear = points(
    rows.map(row => ({
      coordinates: row.coordinates,
      properties: {...row.properties, c: (row.properties.a as number) * 2}
    }))
  );
  expect(() =>
    runCatalogOperation(
      getCatalogBinding('regression-ols')!,
      {responseProperty: 'y', predictorProperties: 'a,c'},
      {points: collinear}
    )
  ).toThrow(/collinear/);
});

test('spherical-point-in-polygon uses great-circle edges (polar cap, antimeridian)', () => {
  const binding = getCatalogBinding('spherical-point-in-polygon')!;
  // A polygon spanning the antimeridian: planar tests would fail it.
  const antimeridian = polygon([
    [170, -10],
    [-170, -10],
    [-170, 10],
    [170, 10],
    [170, -10]
  ]);
  const result = runCatalogOperation(
    binding,
    {boundary: 'closed'},
    {
      points: points([
        {coordinates: [180, 0]},
        {coordinates: [-175, 5]},
        {coordinates: [0, 0]},
        {coordinates: [172, 20]}
      ]),
      polygons: antimeridian
    }
  ).tables[0];
  expect(result.columns['point_feature_index']).toEqual([0, 1]);
  // Self-crossing bow tie is invalid.
  const bowTie = runCatalogOperation(
    binding,
    {},
    {
      points: points([{coordinates: [0, 0]}]),
      polygons: polygon([
        [-10, -10],
        [10, 10],
        [10, -10],
        [-10, 10],
        [-10, -10]
      ])
    }
  );
  expect(bowTie.tables[0].rowCount).toBe(0);
  expect(bowTie.diagnostics.some(item => item.code === 'invalid-spherical-polygon')).toBe(true);
});

test('raster-join-points-in-polygons CPU engine is the exact even-odd join with holes', () => {
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [100, 0],
              [100, 100],
              [0, 100],
              [0, 0]
            ],
            [
              [40, 40],
              [60, 40],
              [60, 60],
              [40, 60],
              [40, 40]
            ]
          ]
        },
        properties: {}
      },
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [200, 0],
              [300, 0],
              [300, 100],
              [200, 100],
              [200, 0]
            ]
          ]
        },
        properties: {}
      }
    ]
  };
  const result = runCatalogOperation(
    getCatalogBinding('raster-join-points-in-polygons')!,
    {coordinateSystem: 'planar', valueProperty: 'v', cellSize: 5},
    {
      points: points([
        {coordinates: [10, 10], properties: {v: 1}},
        {coordinates: [50, 50], properties: {v: 100}},
        {coordinates: [250, 50], properties: {v: 2}},
        {coordinates: [260, 60], properties: {v: 3}},
        {coordinates: [150, 50], properties: {v: 100}}
      ]),
      polygons
    }
  );
  expect(result.tables[0].columns['count']).toEqual([1, 2]);
  expect(result.tables[0].columns['sum']).toEqual([1, 5]);
  expect(result.summary['outsideCount']).toBe(2);
  expect(result.summary['isApproximate']).toBe(0);
});
