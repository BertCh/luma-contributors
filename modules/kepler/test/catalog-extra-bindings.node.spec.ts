// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  matchCatalogInvocationInputs,
  normalizeCatalogParameters,
  parseCatalogInvocation,
  runCatalogOperation,
  serializeCatalogInvocation
} from '../src/analysis/catalog-bindings';
import {EXTRA_CPU_BINDINGS} from '../src/analysis/catalog-extra-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

type Geometry = {type: string; coordinates: unknown};

function collect(
  geometries: (Geometry | {geometry: Geometry; id?: string; properties?: Record<string, unknown>})[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: geometries.map(item =>
      'geometry' in item
        ? {id: item.id, geometry: item.geometry, properties: item.properties ?? {}}
        : {geometry: item, properties: {}}
    )
  };
}

const line = (coordinates: number[][]): Geometry => ({type: 'LineString', coordinates});
const point = (x: number, y: number): Geometry => ({type: 'Point', coordinates: [x, y]});
const polygon = (...rings: number[][][]): Geometry => ({type: 'Polygon', coordinates: rings});
const square = (x: number, y: number, size: number) => [
  [x, y],
  [x + size, y],
  [x + size, y + size],
  [x, y + size],
  [x, y]
];

function run(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
) {
  const binding = getCatalogBinding(id);
  expect(binding, id).toBeDefined();
  return runCatalogOperation(binding!, parameters, inputs);
}

const geometryOf = (table: {columns: Record<string, readonly unknown[]>}, row: number) =>
  JSON.parse(table.columns['geometry'][row] as string);

test('every extra binding is runnable, defaults normalize and entries report the registered engine', () => {
  expect(EXTRA_CPU_BINDINGS.length).toBeGreaterThanOrEqual(20);
  const entries = listCatalogEntries({lensScenarios: []});
  for (const binding of EXTRA_CPU_BINDINGS) {
    expect(getCatalogBinding(binding.id)).toBeDefined();
    expect(() =>
      normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))
    ).not.toThrow();
    expect(binding.declaredBehavior.length).toBeGreaterThan(0);
    for (const name of binding.contributorNames) {
      const entry = entries.find(candidate => candidate.identity.exportName === name);
      expect(entry?.runnable, name).toBe('catalog-binding');
      expect(entry?.bindingIds).toContain(binding.id);
      // Fleet dwell and network disruption add GPU routes alongside the CPU reference bindings.
      const curatedBindingId =
        name === 'GPUPointInPolygonJoin'
          ? 'curated-fleet-dwell'
          : name === 'GPUNetworkCostMatrix'
            ? 'curated-network-disruption'
            : undefined;
      if (curatedBindingId) {
        expect(entry?.bindingIds).toContain(curatedBindingId);
        expect(entry?.engine, name).toBe('gpu-or-cpu');
      } else if (entry?.engine !== 'gpu-or-cpu') {
        // Other PLATFORM4 and PLATFORM5 dispatches also provide a CPU path.
        expect(entry?.engine, name).toBe(getCatalogBinding(binding.id)?.engine);
      }
    }
  }
});

test('line simplification drops vertices within tolerance and keeps endpoints', () => {
  const lines = collect([
    line([
      [0, 0],
      [1, 0.1],
      [2, -0.1],
      [3, 0]
    ])
  ]);
  const coarse = run('line-simplification', {tolerance: 0.5, coordinateSystem: 'planar'}, {lines});
  expect(geometryOf(coarse.tables[0], 0).coordinates).toEqual([
    [0, 0],
    [3, 0]
  ]);
  const fine = run('line-simplification', {tolerance: 0.05, coordinateSystem: 'planar'}, {lines});
  expect(fine.tables[0].columns['vertices_after'][0]).toBe(4);
});

test('line smoothing cuts corners at one quarter and three quarters', () => {
  const lines = collect([
    line([
      [0, 0],
      [4, 0],
      [4, 4]
    ])
  ]);
  const result = run('line-smooth', {iterations: 1}, {lines});
  expect(geometryOf(result.tables[0], 0).coordinates).toEqual([
    [0, 0],
    [1, 0],
    [3, 0],
    [4, 1],
    [4, 3],
    [4, 4]
  ]);
});

test('line precision reduction rounds and removes repeated vertices', () => {
  const lines = collect([
    line([
      [0.123456, 0.987654],
      [0.123449, 0.987651],
      [5, 5]
    ])
  ]);
  const result = run('line-precision-reduction', {decimals: 3}, {lines});
  expect(geometryOf(result.tables[0], 0).coordinates).toEqual([
    [0.123, 0.988],
    [5, 5]
  ]);
});

test('line merge joins two-way junctions and leaves T-junctions apart', () => {
  const chain = collect([
    line([
      [0, 0],
      [1, 0]
    ]),
    line([
      [1, 0],
      [2, 0]
    ]),
    line([
      [3, 0],
      [2, 0]
    ]),
    line([
      [10, 10],
      [11, 10]
    ])
  ]);
  const merged = run('line-merge', {coordinateSystem: 'planar'}, {lines: chain});
  expect(merged.tables[0].rowCount).toBe(2);
  expect(geometryOf(merged.tables[0], 0).coordinates).toEqual([
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0]
  ]);
  expect(merged.tables[0].columns['member_count'][0]).toBe(3);
  expect(merged.tables[0].columns['length'][0]).toBe(3);
  const tee = collect([
    line([
      [0, 0],
      [1, 0]
    ]),
    line([
      [1, 0],
      [2, 0]
    ]),
    line([
      [1, 0],
      [1, 1]
    ])
  ]);
  expect(run('line-merge', {coordinateSystem: 'planar'}, {lines: tee}).tables[0].rowCount).toBe(3);
});

test('line chunk cuts at multiples of the maximum length', () => {
  const lines = collect([
    line([
      [0, 0],
      [10, 0]
    ])
  ]);
  const result = run('line-chunk', {maximumChunkLength: 4, coordinateSystem: 'planar'}, {lines});
  const table = result.tables[0];
  expect(table.columns['length']).toEqual([4, 4, 2]);
  expect(table.columns['start_measure']).toEqual([0, 4, 8]);
  expect(geometryOf(table, 2).coordinates).toEqual([
    [8, 0],
    [10, 0]
  ]);
  expect(() =>
    run('line-chunk', {maximumChunkLength: 1e-9, coordinateSystem: 'planar'}, {lines})
  ).toThrow();
});

test('line locate reports distance along, offset, side and snapped point', () => {
  const lines = collect([
    line([
      [0, 0],
      [10, 0]
    ])
  ]);
  const points = collect([point(3, 2), point(3, -5), point(3, 0)]);
  const result = run(
    'line-locate',
    {maximumOffset: 3, coordinateSystem: 'planar'},
    {lines, points}
  );
  const table = result.tables[0];
  expect(table.columns['distance_along'][0]).toBeCloseTo(3, 12);
  expect(table.columns['fraction'][0]).toBeCloseTo(0.3, 12);
  expect(table.columns['offset'][0]).toBeCloseTo(2, 12);
  expect(table.columns['side'][0]).toBe('left');
  expect(geometryOf(table, 0).coordinates).toEqual([3, 0]);
  expect(table.columns['status'][1]).toBe('beyond-maximum-offset');
  expect(table.columns['side'][2]).toBe('on');
  expect(result.diagnostics.some(item => item.code === 'beyond-maximum-offset')).toBe(true);
});

test('linear referencing interpolates by fraction or distance and flags overruns', () => {
  const lines = collect([
    line([
      [0, 0],
      [10, 0],
      [10, 10]
    ])
  ]);
  const byFraction = run(
    'linear-referencing',
    {measures: [0.25, 0.5, 1.5], measureMode: 'fraction', coordinateSystem: 'planar'},
    {lines}
  ).tables[0];
  expect(geometryOf(byFraction, 0).coordinates).toEqual([5, 0]);
  expect(geometryOf(byFraction, 1).coordinates).toEqual([10, 0]);
  expect(byFraction.columns['status']).toEqual(['ok', 'ok', 'outside-line']);
  const byDistance = run(
    'linear-referencing',
    {measures: [15], measureMode: 'distance', coordinateSystem: 'planar'},
    {lines}
  ).tables[0];
  expect(geometryOf(byDistance, 0).coordinates).toEqual([10, 5]);
});

test('group convex hull builds one hull per group and omits degenerate groups', () => {
  const grouped = (group: string, x: number, y: number) => ({
    geometry: point(x, y),
    properties: {group}
  });
  const points = collect([
    grouped('a', 0, 0),
    grouped('a', 4, 0),
    grouped('a', 4, 4),
    grouped('a', 0, 4),
    grouped('a', 2, 2),
    grouped('b', 0, 0),
    grouped('b', 1, 1),
    grouped('b', 2, 2)
  ]);
  const result = run(
    'group-convex-hull',
    {groupProperty: 'group', coordinateSystem: 'planar'},
    {points}
  );
  const table = result.tables[0];
  expect(table.rowCount).toBe(1);
  expect(table.columns['group_value'][0]).toBe('a');
  expect(table.columns['point_count'][0]).toBe(5);
  expect(table.columns['hull_vertex_count'][0]).toBe(4);
  expect(table.columns['area'][0]).toBe(16);
  expect(result.diagnostics.find(item => item.code === 'degenerate-group')?.count).toBe(1);
});

test('nearest feature join finds the closest point, line or polygon', () => {
  const points = collect([point(0, 0), point(100, 100), point(1, 1)]);
  const targets = collect([
    {id: 'p', geometry: point(3, 4)},
    {
      id: 'l',
      geometry: line([
        [10, 0],
        [10, 10]
      ])
    },
    {id: 'poly', geometry: polygon(square(0.5, 0.5, 2))}
  ]);
  const table = run(
    'nearest-feature-join',
    {maximumDistance: 50, coordinateSystem: 'planar'},
    {points, targets}
  ).tables[0];
  // (0,0): point at 5, line at 10, polygon corner at hypot(0.5,0.5).
  expect(table.columns['target_source_id'][0]).toBe('poly');
  expect(table.columns['distance'][0]).toBeCloseTo(Math.hypot(0.5, 0.5), 12);
  expect(table.columns['status'][1]).toBe('no-target-within-distance');
  expect(table.columns['target_index'][1]).toBeNull();
  expect(table.columns['distance'][2]).toBe(0);
});

test('point in polygon join honors holes, boundaries and overlaps', () => {
  const polygons = collect([
    {id: 'big', geometry: polygon(square(0, 0, 10), square(4, 4, 2))},
    {id: 'small', geometry: polygon(square(0, 0, 2))}
  ]);
  const points = collect([point(1, 1), point(5, 5), point(10, 5), point(20, 20)]);
  const table = run('point-in-polygon-join', {}, {points, polygons}).tables[0];
  expect(table.columns['status']).toEqual(['inside', 'outside', 'inside', 'outside']);
  expect(table.columns['polygon_source_id']).toEqual(['big', null, 'big', null]);
  expect(table.columns['match_count']).toEqual([2, 0, 1, 0]);
});

test('buffer selection keeps features within the distance of any selector', () => {
  const selectors = collect([point(0, 0)]);
  const features = collect([
    {id: 'near', geometry: point(3, 4)},
    {id: 'far', geometry: point(30, 40)},
    {
      id: 'line',
      geometry: line([
        [0, 10],
        [10, 10]
      ])
    }
  ]);
  const table = run(
    'buffer-selection',
    {distance: 5, coordinateSystem: 'planar'},
    {selectors, features}
  ).tables[0];
  expect(table.columns['source_id']).toEqual(['near']);
  expect(table.columns['distance']).toEqual([5]);
});

test('geodesic destination moves along bearings in both coordinate systems', () => {
  const degreeMeters = (6371008.8 * Math.PI) / 180;
  const east = geometryOf(
    run(
      'geodesic-destination',
      {bearing: 90, distance: degreeMeters, coordinateSystem: 'EPSG:4326'},
      {points: collect([point(0, 0)])}
    ).tables[0],
    0
  ).coordinates;
  expect(east[0]).toBeCloseTo(1, 9);
  expect(east[1]).toBeCloseTo(0, 9);
  const north = geometryOf(
    run(
      'geodesic-destination',
      {bearing: 0, distance: degreeMeters, coordinateSystem: 'EPSG:4326'},
      {points: collect([point(0, 0)])}
    ).tables[0],
    0
  ).coordinates;
  expect(north[1]).toBeCloseTo(1, 9);
  const planar = geometryOf(
    run(
      'geodesic-destination',
      {bearing: 90, distance: 5, coordinateSystem: 'planar'},
      {points: collect([point(1, 1)])}
    ).tables[0],
    0
  ).coordinates;
  expect(planar[0]).toBeCloseTo(6, 12);
  expect(planar[1]).toBeCloseTo(1, 12);
});

test('geodesic pairs report distance and bearings and flag unpaired points', () => {
  const result = run(
    'geodesic-pairs',
    {coordinateSystem: 'planar'},
    {
      origins: collect([point(0, 0), point(0, 0), point(9, 9)]),
      destinations: collect([point(0, 10), point(10, 0)])
    }
  );
  const table = result.tables[0];
  expect(table.columns['distance']).toEqual([10, 10]);
  expect(table.columns['initial_bearing']).toEqual([0, 90]);
  expect(table.columns['final_bearing']).toEqual([0, 90]);
  expect(result.diagnostics.find(item => item.code === 'unpaired-points')?.count).toBe(1);
  const geographic = run(
    'geodesic-pairs',
    {coordinateSystem: 'EPSG:4326'},
    {origins: collect([point(0, 0)]), destinations: collect([point(0, 1)])}
  ).tables[0];
  expect(geographic.columns['distance'][0]).toBeCloseTo(111195, -1);
  expect(geographic.columns['initial_bearing'][0]).toBeCloseTo(0, 9);
});

test('geometry orientation rewinds shells and holes', () => {
  const clockwiseShell = [
    [0, 0],
    [0, 10],
    [10, 10],
    [10, 0],
    [0, 0]
  ];
  const counterclockwiseHole = [
    [4, 4],
    [6, 4],
    [6, 6],
    [4, 6],
    [4, 4]
  ];
  const polygons = collect([polygon(clockwiseShell, counterclockwiseHole)]);
  const table = run('geometry-orientation', {}, {polygons}).tables[0];
  expect(table.columns['rings_reversed'][0]).toBe(2);
  const [shell, hole] = geometryOf(table, 0).coordinates;
  expect(shell).toEqual([...clockwiseShell].reverse());
  expect(hole).toEqual([...counterclockwiseHole].reverse());
  const clockwise = run('geometry-orientation', {shellOrientation: 'clockwise'}, {polygons});
  expect(clockwise.tables[0].columns['rings_reversed'][0]).toBe(0);
});

test('affine transform scales, rotates then translates about the pivot', () => {
  const result = run(
    'affine-transform',
    {scaleX: 2, scaleY: 2, rotationDegrees: 90, translateX: 1, translateY: 1, pivot: 'origin'},
    {features: collect([point(2, 0)])}
  );
  const coordinates = geometryOf(result.tables[0], 0).coordinates;
  expect(coordinates[0]).toBeCloseTo(1, 12);
  expect(coordinates[1]).toBeCloseTo(5, 12);
  const about = run(
    'affine-transform',
    {rotationDegrees: 180, pivot: 'bounds-center'},
    {
      features: collect([
        line([
          [0, 0],
          [4, 2]
        ])
      ])
    }
  );
  const rotated = geometryOf(about.tables[0], 0).coordinates;
  expect(rotated[0][0]).toBeCloseTo(4, 12);
  expect(rotated[0][1]).toBeCloseTo(2, 12);
  expect(rotated[1][0]).toBeCloseTo(0, 12);
  expect(rotated[1][1]).toBeCloseTo(0, 12);
});

test('vertex snap merges near vertices within and across lines', () => {
  const lines = collect([
    line([
      [0, 0],
      [0.1, 0],
      [5, 0],
      [10, 0]
    ]),
    line([
      [10.4, 0.3],
      [20, 0]
    ])
  ]);
  const result = run('vertex-snap', {tolerance: 1, coordinateSystem: 'planar'}, {lines});
  const table = result.tables[0];
  expect(geometryOf(table, 0).coordinates).toEqual([
    [0, 0],
    [5, 0],
    [10, 0]
  ]);
  expect(table.columns['vertices_removed'][0]).toBe(1);
  expect(geometryOf(table, 1).coordinates[0]).toEqual([10, 0]);
});

test('grid generator covers the extent and refuses oversize grids', () => {
  const extent = collect([
    line([
      [0, 0],
      [10, 5]
    ])
  ]);
  const two = run('grid-generator', {cellWidth: 5, cellHeight: 5}, {extent}).tables[0];
  expect(two.rowCount).toBe(2);
  expect(geometryOf(two, 1).coordinates[0]).toEqual([
    [5, 0],
    [10, 0],
    [10, 5],
    [5, 5],
    [5, 0]
  ]);
  expect(run('grid-generator', {cellWidth: 3, cellHeight: 2.5}, {extent}).tables[0].rowCount).toBe(
    8
  );
  expect(() => run('grid-generator', {cellWidth: 0.001, cellHeight: 0.001}, {extent})).toThrow();
});

test('random points in polygons are seeded, inside and outside holes', () => {
  const polygons = collect([polygon(square(0, 0, 10), square(4, 4, 2))]);
  const first = run('random-points-in-polygon', {pointsPerPolygon: 50, seed: 7}, {polygons});
  const again = run('random-points-in-polygon', {pointsPerPolygon: 50, seed: 7}, {polygons});
  const other = run('random-points-in-polygon', {pointsPerPolygon: 50, seed: 8}, {polygons});
  expect(first.tables[0].rowCount).toBe(50);
  expect(first.tables[0].columns['geometry']).toEqual(again.tables[0].columns['geometry']);
  expect(first.tables[0].columns['geometry']).not.toEqual(other.tables[0].columns['geometry']);
  for (let row = 0; row < 50; row++) {
    const [x, y] = geometryOf(first.tables[0], row).coordinates;
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThanOrEqual(10);
    expect(y).toBeGreaterThanOrEqual(0);
    expect(y).toBeLessThanOrEqual(10);
    expect(x > 4 && x < 6 && y > 4 && y < 6).toBe(false);
  }
});

test('random points on lines lie on the line at the reported distance', () => {
  const lines = collect([
    line([
      [0, 0],
      [10, 0]
    ])
  ]);
  const first = run(
    'random-points-on-line',
    {pointsPerLine: 20, seed: 3, coordinateSystem: 'planar'},
    {lines}
  );
  const again = run(
    'random-points-on-line',
    {pointsPerLine: 20, seed: 3, coordinateSystem: 'planar'},
    {lines}
  );
  expect(first.tables[0].columns['geometry']).toEqual(again.tables[0].columns['geometry']);
  expect(first.tables[0].rowCount).toBe(20);
  for (let row = 0; row < 20; row++) {
    const [x, y] = geometryOf(first.tables[0], row).coordinates;
    expect(y).toBe(0);
    expect(x).toBeCloseTo(first.tables[0].columns['distance_along'][row] as number, 12);
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThanOrEqual(10);
  }
});

test('k-means separates two groups deterministically with hand-computed inertia', () => {
  const points = collect([
    point(0, 0),
    point(1, 0),
    point(0, 1),
    point(100, 100),
    point(101, 100),
    point(100, 101)
  ]);
  const result = run(
    'k-means',
    {clusterCount: 2, maximumIterations: 20, coordinateSystem: 'planar'},
    {points}
  );
  const [assignments, centroids] = result.tables;
  expect(assignments.columns['cluster']).toEqual([0, 0, 0, 1, 1, 1]);
  expect(centroids.columns['point_count']).toEqual([3, 3]);
  expect(geometryOf(centroids, 0).coordinates[0]).toBeCloseTo(1 / 3, 12);
  expect(geometryOf(centroids, 1).coordinates[0]).toBeCloseTo(100 + 1 / 3, 12);
  expect(result.summary['inertia']).toBeCloseTo(8 / 3, 12);
  const reduced = run(
    'k-means',
    {clusterCount: 9, coordinateSystem: 'planar'},
    {points: collect([point(0, 0), point(1, 1)])}
  );
  expect(reduced.tables[1].rowCount).toBe(2);
  expect(reduced.diagnostics.some(item => item.code === 'cluster-count-reduced')).toBe(true);
});

// A(0,0) B(10,0) C(20,0) D(10,10) with B as a three-way junction; island I1(100,100) I2(110,100).
const networkLines = collect([
  {
    id: 'ab',
    geometry: line([
      [0, 0],
      [10, 0]
    ])
  },
  {
    id: 'bc',
    geometry: line([
      [10, 0],
      [20, 0]
    ])
  },
  {
    id: 'bd',
    geometry: line([
      [10, 0],
      [10, 10]
    ])
  },
  {
    id: 'island',
    geometry: line([
      [100, 100],
      [110, 100]
    ])
  }
]);
const planar = {coordinateSystem: 'planar'};

test('network structure reports degrees, components and node kinds', () => {
  const result = run('network-structure', planar, {lines: networkLines});
  expect(result.summary['componentCount']).toBe(2);
  expect(result.summary['junctionCount']).toBe(1);
  expect(result.summary['deadEndCount']).toBe(5);
  const table = result.tables[0];
  expect([...table.columns['degree']].sort()).toEqual([1, 1, 1, 1, 1, 3]);
  expect(table.columns['node_kind'].filter(kind => kind === 'junction')).toHaveLength(1);
});

test('network centrality matches hand-computed degree and closeness', () => {
  const result = run('network-centrality', planar, {lines: networkLines});
  const table = result.tables[0];
  const junction = table.columns['degree'].indexOf(3);
  // n = 6; junction reaches 3 nodes at cost 10 each: (3/5) x (3/30).
  expect(table.columns['degree_centrality'][junction]).toBeCloseTo(3 / 5, 12);
  expect(table.columns['closeness_centrality'][junction]).toBeCloseTo(0.06, 12);
  // Dead end A reaches B(10), C(20), D(20): (3/5) x (3/50).
  const deadEnd = table.columns['degree'].indexOf(1);
  expect(table.columns['closeness_centrality'][deadEnd]).toBeCloseTo(0.036, 12);
  expect(table.columns['reachable_count'][deadEnd]).toBe(3);
});

test('network cost matrix computes origin by destination costs and statuses', () => {
  const result = run(
    'network-cost-matrix',
    {...planar, maximumSnapDistance: 100},
    {
      lines: networkLines,
      origins: collect([point(0, 0), point(20, 0)]),
      destinations: collect([point(10, 10), point(105, 100), point(500, 500)])
    }
  );
  const table = result.tables[0];
  expect(table.rowCount).toBe(6);
  expect(table.columns['total_cost'][0]).toBeCloseTo(20, 12);
  expect(table.columns['status'][1]).toBe('unreachable');
  expect(table.columns['status'][2]).toBe('destination-not-snapped');
  expect(table.columns['total_cost'][3]).toBeCloseTo(20, 12);
});

test('network neighborhood lists nodes within the cost budget', () => {
  const origin = collect([point(0, 0)]);
  const near = run(
    'network-neighborhood',
    {...planar, maximumCost: 15},
    {lines: networkLines, origin}
  );
  expect(near.tables[0].columns['cost']).toEqual([0, 10]);
  const wide = run(
    'network-neighborhood',
    {...planar, maximumCost: 25},
    {lines: networkLines, origin}
  );
  expect(wide.tables[0].rowCount).toBe(4);
});

test('network line graph links edges that share a node', () => {
  const result = run('network-line-graph', planar, {lines: networkLines});
  const table = result.tables[0];
  expect(table.rowCount).toBe(3);
  expect(new Set(table.columns['shared_node']).size).toBe(1);
});

test('extra parameters are range checked and invocations round trip', () => {
  const chunk = getCatalogBinding('line-chunk')!;
  expect(() => normalizeCatalogParameters(chunk, {maximumChunkLength: -1})).toThrow();
  expect(() => normalizeCatalogParameters(chunk, {unknown: 1})).toThrow();
  expect(normalizeCatalogParameters(chunk, {maximumChunkLength: '250'})['maximumChunkLength']).toBe(
    250
  );
  const kmeans = getCatalogBinding('k-means')!;
  expect(() => normalizeCatalogParameters(kmeans, {clusterCount: 2.5})).toThrow();
  const lines = collect([
    line([
      [0, 0],
      [10, 0]
    ])
  ]);
  const text = serializeCatalogInvocation(
    chunk,
    {maximumChunkLength: 4, coordinateSystem: 'planar'},
    {lines}
  );
  const restored = parseCatalogInvocation(text);
  expect(restored.binding.id).toBe('line-chunk');
  expect(restored.invocation.parameters['maximumChunkLength']).toBe(4);
  expect(matchCatalogInvocationInputs(restored.invocation, {lines})).toBe(true);
  expect(
    matchCatalogInvocationInputs(restored.invocation, {
      lines: collect([
        line([
          [0, 0],
          [9, 0]
        ])
      ])
    })
  ).toBe(false);
});

test('inputs of the wrong geometry kind are rejected before work starts', () => {
  expect(() => run('line-merge', {}, {lines: collect([point(0, 0)])})).toThrow();
  expect(() => run('point-in-polygon-join', {}, {points: collect([point(0, 0)])})).toThrow();
});
