// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  runCatalogOperation,
  type CatalogRunResult
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

function lineCollection(
  rows: {coordinates: number[][]; properties?: Record<string, unknown>}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map((row, index) => ({
      id: `line-${index}`,
      geometry: {type: 'LineString', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  };
}

function pointCollection(
  positions: number[][],
  properties: Record<string, unknown>[] = []
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: positions.map((coordinates, index) => ({
      id: `point-${index}`,
      geometry: {type: 'Point', coordinates},
      properties: properties[index] ?? {}
    }))
  };
}

function run(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): CatalogRunResult {
  const binding = getCatalogBinding(id);
  if (!binding) throw new Error(`Missing binding ${id}`);
  const result = runCatalogOperation(binding, {coordinateSystem: 'planar', ...parameters}, inputs);
  for (const table of result.tables) {
    const spec = binding.outputs.find(output => output.tableId === table.id);
    expect(spec, `${id} declares table ${table.id}`).toBeDefined();
    for (const column of spec!.identityColumns)
      expect(Object.keys(table.columns), `${id}.${table.id} has ${column}`).toContain(column);
  }
  return result;
}

const column = (result: CatalogRunResult, name: string, tableIndex = 0) =>
  result.tables[tableIndex].columns[name];

// Path 0-10-20 east, a branch north at x=10 and x=20: each edge is 10 long.
const network = lineCollection([
  {
    coordinates: [
      [0, 0],
      [10, 0]
    ],
    properties: {kind: 'main'}
  },
  {
    coordinates: [
      [10, 0],
      [20, 0]
    ],
    properties: {kind: 'main'}
  },
  {
    coordinates: [
      [20, 0],
      [20, 10]
    ],
    properties: {kind: 'main'}
  },
  {
    coordinates: [
      [10, 0],
      [10, 10]
    ],
    properties: {kind: 'side'}
  }
]);

test('accessibility counts opportunities within the cost and applies decay', () => {
  const inputs = {
    lines: network,
    origins: pointCollection([[0, 0]]),
    opportunities: pointCollection(
      [
        [10, 0],
        [20, 0],
        [20, 10],
        [500, 500]
      ],
      [{w: 1}, {w: 2}, {w: 4}, {w: 8}]
    )
  };
  const cumulative = run('network-accessibility', {maximumCost: 25}, inputs);
  expect(column(cumulative, 'reachable_count')[0]).toBe(2);
  expect(column(cumulative, 'accessibility')[0]).toBe(2);
  expect(column(cumulative, 'nearest_cost')[0]).toBeCloseTo(10, 9);
  expect(cumulative.diagnostics.map(item => item.code)).toContain('opportunity-not-snapped');
  const weighted = run(
    'network-accessibility',
    {maximumCost: 25, weightProperty: 'w', decay: 'linear'},
    inputs
  );
  expect(column(weighted, 'accessibility')[0]).toBeCloseTo(1 * 0.6 + 2 * 0.2, 9);
  const stranded = run(
    'network-accessibility',
    {maximumCost: 25},
    {
      ...inputs,
      origins: pointCollection([[900, 900]])
    }
  );
  expect(column(stranded, 'status')[0]).toBe('origin-not-snapped');
});

test('subgraph filter keeps matching edges and the largest component', () => {
  const byProperty = run(
    'network-subgraph-filter',
    {propertyName: 'kind', propertyValue: 'main'},
    {
      lines: network
    }
  );
  expect(byProperty.tables[0].rowCount).toBe(3);
  const byCost = run('network-subgraph-filter', {minimumCost: 11}, {lines: network});
  expect(byCost.tables[0].rowCount).toBe(0);
  const split = lineCollection([
    {
      coordinates: [
        [0, 0],
        [10, 0]
      ]
    },
    {
      coordinates: [
        [10, 0],
        [20, 0]
      ]
    },
    {
      coordinates: [
        [100, 100],
        [101, 100]
      ]
    }
  ]);
  const largest = run('network-subgraph-filter', {largestComponentOnly: true}, {lines: split});
  expect(column(largest, 'source_index')).toEqual([0, 1]);
});

test('coarsening merges degree-two chains with summed costs', () => {
  const result = run('network-coarsening', {}, {lines: network});
  expect(result.tables[0].rowCount).toBe(3);
  const lengths = (column(result, 'length') as number[]).sort((first, second) => first - second);
  expect(lengths).toEqual([10, 10, 20]);
  const merged = column(result, 'edge_count').indexOf(2);
  expect(column(result, 'forward_cost')[merged]).toBeCloseTo(20, 9);
  expect(column(result, 'source_indices')[merged]).toBe('1,2');
  const ring = run(
    'network-coarsening',
    {},
    {
      lines: lineCollection([
        {
          coordinates: [
            [0, 0],
            [10, 0]
          ]
        },
        {
          coordinates: [
            [10, 0],
            [10, 10]
          ]
        },
        {
          coordinates: [
            [10, 10],
            [0, 0]
          ]
        }
      ])
    }
  );
  expect(ring.tables[0].rowCount).toBe(1);
  expect(column(ring, 'edge_count')[0]).toBe(3);
});

test('flow aggregation sums weights per edge and lists unrouted pairs', () => {
  const result = run(
    'network-flow-aggregation',
    {weightProperty: 'w'},
    {
      lines: network,
      origins: pointCollection(
        [
          [0, 0],
          [0, 0],
          [0, 0]
        ],
        [{w: 1}, {w: 3}, {w: 5}]
      ),
      destinations: pointCollection([
        [20, 10],
        [10, 10],
        [900, 900]
      ])
    }
  );
  const byEdge = new Map<string, number>();
  column(result, 'source_index').forEach((index, row) =>
    byEdge.set(String(index), column(result, 'flow')[row] as number)
  );
  expect(byEdge.get('0')).toBe(4);
  expect(byEdge.get('1')).toBe(1);
  expect(byEdge.get('2')).toBe(1);
  expect(byEdge.get('3')).toBe(3);
  expect(column(result, 'status', 1)).toEqual(['ok', 'ok', 'destination-not-snapped']);
  expect(result.diagnostics.map(item => item.code)).toContain('pairs-not-routed');
});

test('alternative routes return distinct detours with unpenalized costs', () => {
  const loop = lineCollection([
    {
      coordinates: [
        [0, 0],
        [20, 0]
      ]
    },
    {
      coordinates: [
        [0, 0],
        [10, 8],
        [20, 0]
      ]
    }
  ]);
  const result = run(
    'network-alternative-routes',
    {routeCount: 2},
    {
      lines: loop,
      origin: pointCollection([[0, 0]]),
      destination: pointCollection([[20, 0]])
    }
  );
  expect(result.tables[0].rowCount).toBe(2);
  expect(column(result, 'cost')[0]).toBeCloseTo(20, 9);
  expect(column(result, 'cost')[1]).toBeCloseTo(2 * Math.hypot(10, 8), 9);
  expect(column(result, 'overlap_with_best')[1]).toBeCloseTo(0, 9);
  expect(column(result, 'detour_ratio')[1] as number).toBeGreaterThan(1);
  expect(() =>
    run(
      'network-alternative-routes',
      {},
      {
        lines: loop,
        origin: pointCollection([
          [0, 0],
          [1, 1]
        ]),
        destination: pointCollection([[20, 0]])
      }
    )
  ).toThrow();
});

test('map matching snaps a noisy trace and reports breaks', () => {
  const road = lineCollection([
    {
      coordinates: [
        [0, 0],
        [100, 0]
      ]
    }
  ]);
  const result = run(
    'network-map-matching',
    {searchRadius: 20},
    {
      lines: road,
      trace: pointCollection([
        [10, 3],
        [40, -2],
        [80, 4],
        [500, 500]
      ])
    }
  );
  expect(column(result, 'status')).toEqual(['matched', 'matched', 'matched', 'unmatched']);
  expect(column(result, 'fraction')[0]).toBeCloseTo(0.1, 9);
  expect(column(result, 'fraction')[2]).toBeCloseTo(0.8, 9);
  expect(result.tables[1].rowCount).toBe(1);
  const route = JSON.parse(String(column(result, 'geometry', 1)[0])).coordinates;
  expect(route[0][0]).toBeCloseTo(10, 9);
  expect(route[route.length - 1][0]).toBeCloseTo(80, 9);
  expect(result.diagnostics.map(item => item.code)).toContain('trace-point-unmatched');
});

test('network K function counts ordered pairs within each radius', () => {
  const result = run(
    'network-k-function',
    {radii: [21, 41, 101]},
    {
      lines: lineCollection([
        {
          coordinates: [
            [0, 0],
            [100, 0]
          ]
        }
      ]),
      points: pointCollection([
        [10, 0],
        [30, 0],
        [50, 0],
        [70, 0],
        [90, 0]
      ])
    }
  );
  expect(column(result, 'pair_count')).toEqual([8, 14, 20]);
  expect(column(result, 'k')[0]).toBeCloseTo((100 * 8) / 20, 9);
  expect(column(result, 'expected_k')[1]).toBe(82);
  const lonely = run(
    'network-k-function',
    {},
    {
      lines: lineCollection([
        {
          coordinates: [
            [0, 0],
            [100, 0]
          ]
        }
      ]),
      points: pointCollection([[10, 0]])
    }
  );
  expect(column(lonely, 'k')[0]).toBeNull();
});

test('edge bundling pulls compatible lines together and keeps identity', () => {
  const result = run(
    'network-edge-bundling',
    {cycles: 3, iterations: 30},
    {
      lines: lineCollection([
        {
          coordinates: [
            [0, 0],
            [100, 0]
          ]
        },
        {
          coordinates: [
            [0, 4],
            [100, 6]
          ]
        },
        {
          coordinates: [
            [0, 8],
            [100, 10]
          ]
        },
        {
          coordinates: [
            [0, 0],
            [0, 0]
          ]
        }
      ])
    }
  );
  expect(result.tables[0].rowCount).toBe(3);
  expect(column(result, 'source_id')).toEqual(['line-0', 'line-1', 'line-2']);
  expect(column(result, 'compatible_count')[0]).toBe(2);
  const middle = JSON.parse(String(column(result, 'geometry')[1])).coordinates;
  expect(middle).toHaveLength(9);
  expect(middle[0]).toEqual([0, 4]);
  expect(column(result, 'straightness')[1] as number).toBeLessThanOrEqual(1);
});

test('network operations reject missing inputs and out-of-range parameters', () => {
  const binding = getCatalogBinding('network-accessibility')!;
  expect(() =>
    runCatalogOperation(binding, {coordinateSystem: 'planar'}, {lines: network})
  ).toThrow();
  expect(() =>
    runCatalogOperation(
      binding,
      {coordinateSystem: 'planar', decay: 'sigmoid'},
      {lines: network, origins: pointCollection([[0, 0]]), opportunities: pointCollection([[1, 0]])}
    )
  ).toThrow();
});
