// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {CatalogDiagnostic, CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {CatalogGpuUnsupportedError} from '../src/analysis/catalog-gpu-session';
import {
  runCoverageValidity,
  runSpatialPredicateJoin
} from '../src/analysis/catalog-platform3-a-join';
import {runLocationAllocation} from '../src/analysis/catalog-platform3-b-clusters';
import {addGpuKrigingStatusDiagnostics} from '../src/analysis/catalog-platform3-b-grids';
import {requireCompleteZonalDistributionStages} from '../src/analysis/catalog-platform5-recipes';

// Review 2026-11-19, consumer seams (SEAM-104, SEAM-106, SEAM-108, SEAM-115).

function points(
  coordinates: number[][],
  weights: (number | null | undefined)[] = []
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      id: String(index),
      geometry: {type: 'Point', coordinates: position},
      properties: {weight: weights[index]}
    }))
  };
}

function readColumn(output: ReturnType<typeof runLocationAllocation>, table: number, name: string) {
  return Array.from(output.tables[table].columns[name]);
}

test('SEAM-104: negative demand weight is skipped and counted, not inverted', () => {
  // Closed form: with the -2 row skipped, the only demand is weight 1 at (0, 0); candidate
  // `near` at (0, 0) costs 1 * 0 = 0 and `far` at (10, 0) costs 1 * 10 = 10, so `near` wins
  // with objective 0, demand count 1 and assigned demand weight 1.
  const output = runLocationAllocation(
    points(
      [
        [0, 0],
        [0, 0]
      ],
      [1, -2]
    ),
    points([
      [0, 0],
      [10, 0]
    ]),
    {facilityCount: 1, weightProperty: 'weight', coordinateSystem: 'planar'}
  );
  expect(readColumn(output, 0, 'candidate_index')).toEqual([0]);
  expect(readColumn(output, 0, 'demand_count')).toEqual([1]);
  expect(readColumn(output, 0, 'demand_weight')).toEqual([1]);
  expect(output.summary['demandCount']).toBe(1);
  expect(Number.isFinite(output.summary['objective'] as number)).toBe(true);
  expect(output.summary['objective']).toBe(0);
  expect(readColumn(output, 1, 'source_index')).toEqual([0]);
  const dropped = output.diagnostics.find(item => item.code === 'feature-dropped');
  expect(dropped?.count).toBe(1);
});

test('SEAM-104: only negative demand leaves too few features', () => {
  expect(() =>
    runLocationAllocation(points([[0, 0]], [-1]), points([[0, 0]]), {
      facilityCount: 1,
      weightProperty: 'weight',
      coordinateSystem: 'planar'
    })
  ).toThrow('Too few features');
});

test.each([
  0, 1e9
])('SEAM-106: a point strictly inside a hole is not within (offset %s)', offset => {
  // Shapely (/usr/bin/python3): Polygon(shell [o,o+10]^2, hole [o+2,o+8]^2).covers(Point(o+2.0005,
  // o+5)) is False for o = 0 and o = 1e9; boundary distance 0.0005 (0.00049996 after float64
  // rounding at 1e9).
  const square = (minimum: number, maximum: number) => [
    [offset + minimum, offset + minimum],
    [offset + maximum, offset + minimum],
    [offset + maximum, offset + maximum],
    [offset + minimum, offset + maximum],
    [offset + minimum, offset + minimum]
  ];
  const polygon: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        id: 'zone',
        geometry: {type: 'Polygon', coordinates: [square(0, 10), square(2, 8).reverse()]},
        properties: {}
      }
    ]
  };
  const inHole = points([[offset + 2.0005, offset + 5]]);
  const inShell = points([[offset + 1, offset + 5]]);
  expect(runSpatialPredicateJoin(inHole, polygon, 'within').tables[0].rowCount).toBe(0);
  expect(runSpatialPredicateJoin(inHole, polygon, 'intersects').tables[0].rowCount).toBe(0);
  expect(runSpatialPredicateJoin(inShell, polygon, 'within').tables[0].rowCount).toBe(1);
});

test('SEAM-108: zone-assignment loss refuses the GPU distribution', () => {
  const complete: Parameters<typeof requireCompleteZonalDistributionStages>[0] = [
    {stage: 'zone-assignment', fields: {overflow: 0, uncertainCount: 0}},
    {stage: 'zone-distributions', fields: {count: 2, requiredCount: 2, overflow: 0}}
  ];
  expect(() => requireCompleteZonalDistributionStages(complete)).not.toThrow();
  for (const [stage, field] of [
    ['zone-assignment', 'overflow'],
    ['zone-assignment', 'uncertainCount'],
    ['zone-distributions', 'overflow']
  ] as const) {
    const lossy = complete.map(item =>
      item.stage === stage ? {stage, fields: {...item.fields, [field]: 1}} : item
    );
    expect(() => requireCompleteZonalDistributionStages(lossy)).toThrow(CatalogGpuUnsupportedError);
    expect(() => requireCompleteZonalDistributionStages(lossy)).toThrow(`${stage}.${field}`);
  }
});

test('SEAM-115: GPU kriging reports invalid and uncertain cells separately', () => {
  const diagnostics: CatalogDiagnostic[] = [];
  addGpuKrigingStatusDiagnostics(diagnostics, {uncertainCount: 3, invalidCount: 5});
  const uncertain = diagnostics.find(item => item.code === 'singular-kriging-system');
  const invalid = diagnostics.find(item => item.code === 'kriging-cells-without-data');
  expect(uncertain?.count).toBe(3);
  expect(uncertain?.message).toMatch(/ill-conditioned/);
  expect(uncertain?.message).toMatch(/variance came out materially negative/);
  expect(invalid?.count).toBe(5);
  expect(invalid?.message).toMatch(/no sample/);
  const none: CatalogDiagnostic[] = [];
  addGpuKrigingStatusDiagnostics(none, {uncertainCount: 0, invalidCount: 0});
  expect(none).toEqual([]);
});

// Closed form for unit squares displaced horizontally: overlap area = max(0, 1 - shift).
// A 0.0005-wide overlap is positive; shift 1 touches an edge; shift 1.0005 separates.
test.each([
  0, 1e9
])('coverage local frame preserves thin overlaps and source identity (offset %s)', offset => {
  for (const [shift, expected] of [
    [0.9995, 1],
    [1, 0],
    [1.0005, 0]
  ]) {
    const polygons: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [0, shift].map((x, index) => ({
        id: `source-${index}`,
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [offset + x, offset],
              [offset + x + 1, offset],
              [offset + x + 1, offset + 1],
              [offset + x, offset + 1],
              [offset + x, offset]
            ]
          ]
        }
      }))
    };
    const originalGeometries = polygons.features.map(feature => JSON.stringify(feature.geometry));
    const output = runCoverageValidity(polygons);
    expect(output.tables[0].rowCount, `shift ${shift}`).toBe(expected);
    expect(readColumn(output, 1, 'source_index')).toEqual([0, 1]);
    expect(readColumn(output, 1, 'source_id')).toEqual(['source-0', 'source-1']);
    expect(readColumn(output, 1, 'geometry')).toEqual(originalGeometries);
    expect(polygons.features.map(feature => JSON.stringify(feature.geometry))).toEqual(
      originalGeometries
    );
    if (expected) {
      expect(readColumn(output, 0, 'first_index')).toEqual([0]);
      expect(readColumn(output, 0, 'second_index')).toEqual([1]);
      expect(readColumn(output, 0, 'first_id')).toEqual(['source-0']);
      expect(readColumn(output, 0, 'second_id')).toEqual(['source-1']);
    }
  }
});

function reviewRectangle(minimumX: number, minimumY: number, maximumX: number, maximumY: number) {
  return [
    [minimumX, minimumY],
    [maximumX, minimumY],
    [maximumX, maximumY],
    [minimumX, maximumY],
    [minimumX, minimumY]
  ];
}

function reviewPolygons(parts: number[][][][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: parts.map((coordinates, index) => ({
      id: `polygon-${index}`,
      properties: {},
      geometry: {type: 'Polygon', coordinates}
    }))
  };
}

test('B1: a distant disjoint feature cannot erase an unchanged coverage overlap', () => {
  // Intersection width 4 - 3.99975 = 0.00025, height 2: area 0.0005.
  const first = [reviewRectangle(0, 0, 4, 2)];
  const second = [reviewRectangle(3.99975, 0, 7, 2)];
  for (const parts of [
    [first, second],
    [first, second, [reviewRectangle(-1e9, -1e9, -1e9 + 1, -1e9 + 1)]]
  ]) {
    const output = runCoverageValidity(reviewPolygons(parts));
    expect(readColumn(output, 0, 'first_id')).toEqual(['polygon-0']);
    expect(readColumn(output, 0, 'second_id')).toEqual(['polygon-1']);
    expect(readColumn(output, 0, 'first_index')).toEqual([0]);
    expect(readColumn(output, 0, 'second_index')).toEqual([1]);
  }
});

test.each([
  0, 1e9
])('B2: duplicate holed polygons overlap; a hole-filling polygon only touches (offset %s)', offset => {
  // Shell area 12*8=96, hole area 6*4=24: duplicate intersection area 72.
  // The filler has intersection area zero with either donut, sharing just its boundary.
  const shell = reviewRectangle(offset, offset, offset + 12, offset + 8);
  const hole = reviewRectangle(offset + 3, offset + 2, offset + 9, offset + 6);
  const output = runCoverageValidity(reviewPolygons([[shell, hole], [shell, hole], [hole]]));
  expect(readColumn(output, 0, 'first_id')).toEqual(['polygon-0']);
  expect(readColumn(output, 0, 'second_id')).toEqual(['polygon-1']);
  expect(readColumn(output, 1, 'status')).toEqual(['overlaps', 'overlaps', 'valid']);
  expect(readColumn(output, 1, 'overlap_count')).toEqual([1, 1, 0]);
});

test('B4: a polygon filling a target hole is neither within it nor inversely contained', () => {
  const shell = reviewRectangle(0, 0, 12, 8);
  const hole = reviewRectangle(3, 2, 9, 6);
  const donut = reviewPolygons([[shell, hole]]);
  const filler = reviewPolygons([[hole]]);
  expect(runSpatialPredicateJoin(filler, donut, 'within').tables[0].rowCount).toBe(0);
  expect(runSpatialPredicateJoin(donut, filler, 'contains').tables[0].rowCount).toBe(0);
  expect(runSpatialPredicateJoin(filler, donut, 'intersects').tables[0].rowCount).toBe(1);
  expect(runSpatialPredicateJoin(donut, donut, 'within').tables[0].rowCount).toBe(1);
});

test('earlier-fix boundary note: bounds filtering admits the documented edge tolerance', () => {
  const polygon = reviewPolygons([[reviewRectangle(0, 0, 1, 1)]]);
  expect(
    runSpatialPredicateJoin(points([[-5e-13, 0.5]]), polygon, 'within').tables[0].rowCount
  ).toBe(1);
  expect(
    runSpatialPredicateJoin(points([[-2e-12, 0.5]]), polygon, 'within').tables[0].rowCount
  ).toBe(0);
});
