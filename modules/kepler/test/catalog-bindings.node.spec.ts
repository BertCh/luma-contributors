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
  serializeCatalogInvocation,
  type CatalogOperationBinding
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

function lineCollection(
  lines: {coordinates: number[][]; id?: string; properties?: Record<string, unknown>}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: lines.map(line => ({
      id: line.id,
      geometry: {type: 'LineString', coordinates: line.coordinates},
      properties: line.properties ?? {}
    }))
  };
}

function pointCollection(points: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: points.map(coordinates => ({geometry: {type: 'Point', coordinates}, properties: {}}))
  };
}

function polygonCollection(rings: number[][][][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rings.map(polygon => ({
      geometry: {type: 'Polygon', coordinates: polygon},
      properties: {}
    }))
  };
}

const square = (x: number, y: number, size: number) => [
  [x, y],
  [x + size, y],
  [x + size, y + size],
  [x, y + size],
  [x, y]
];

// A(0,0) B(10,0) C(30,0) D(30,10) E(0,5); G(-10,0); F(30,20); island at (100,100).
const networkLines = lineCollection([
  {
    id: 'ab',
    coordinates: [
      [0, 0],
      [10, 0]
    ]
  },
  {
    id: 'bc',
    coordinates: [
      [10, 0],
      [30, 0]
    ]
  },
  {
    id: 'cd',
    coordinates: [
      [30, 0],
      [30, 10]
    ],
    properties: {oneway: 'yes'}
  },
  {
    id: 'de',
    coordinates: [
      [30, 10],
      [0, 5]
    ],
    properties: {oneway: 'yes'}
  },
  {
    id: 'ea',
    coordinates: [
      [0, 5],
      [0, 0]
    ]
  },
  {
    id: 'island',
    coordinates: [
      [100, 100],
      [110, 100]
    ]
  },
  {
    id: 'spur',
    coordinates: [
      [30, 10],
      [30, 20]
    ],
    properties: {oneway: '-1'}
  },
  {
    id: 'ga',
    coordinates: [
      [-10, 0],
      [0, 0]
    ],
    properties: {oneway: 'yes'}
  }
]);
const buildParameters = {coordinateSystem: 'planar', onewayProperty: 'oneway'};
const networkParameters = {
  coordinateSystem: 'planar',
  onewayProperty: 'oneway',
  maximumSnapDistance: 1
};

function requireBinding(id: string): CatalogOperationBinding {
  const binding = getCatalogBinding(id);
  if (!binding) throw new Error(`missing ${id}`);
  return binding;
}

/** Independent Bellman-Ford over the published edge table. */
function computeReferenceCost(
  edges: Record<string, readonly (string | number | boolean | null)[]>,
  from: number,
  to: number
): number {
  const distance = new Map<number, number>([[from, 0]]);
  for (let pass = 0; pass < 20; pass++)
    for (let row = 0; row < edges['edge_id'].length; row++) {
      const a = edges['from_node'][row] as number;
      const b = edges['to_node'][row] as number;
      const forward = edges['forward_cost'][row] as number | null;
      const reverse = edges['reverse_cost'][row] as number | null;
      if (forward !== null && distance.has(a))
        distance.set(b, Math.min(distance.get(b) ?? Infinity, distance.get(a)! + forward));
      if (reverse !== null && distance.has(b))
        distance.set(a, Math.min(distance.get(a) ?? Infinity, distance.get(b)! + reverse));
    }
  return distance.get(to) ?? Infinity;
}

test('catalog entries report runnable bindings and a reason when not runnable', () => {
  const entries = listCatalogEntries();
  const serviceAreas = entries.find(entry => entry.contributorName === 'GPUNetworkServiceAreas');
  expect(serviceAreas?.runnable).toBe('catalog-binding');
  expect(serviceAreas?.bindingIds).toEqual(['network-service-area']);
  const unbound = entries.filter(entry => entry.runnable === 'no');
  expect(unbound.length).toBeGreaterThan(0);
  expect(unbound.every(entry => Boolean(entry.reason))).toBe(true);
});

test('parameters are range checked and unknown names rejected', () => {
  const binding = requireBinding('geometry-point-buffer');
  expect(normalizeCatalogParameters(binding, {radius: '50'})['radius']).toBe(50);
  expect(() => normalizeCatalogParameters(binding, {radius: -1})).toThrow();
  expect(() => normalizeCatalogParameters(binding, {segmentsPerQuarter: 2.5})).toThrow();
  expect(() => normalizeCatalogParameters(binding, {coordinateSystem: 'EPSG:3857'})).toThrow();
  expect(() => normalizeCatalogParameters(binding, {unknown: 1})).toThrow();
});

test('every binding declares inputs, outputs, cleanup, behavior and a serializable default', () => {
  // These bindings generate their raster/velocity fixtures rather than consume datasets.
  const generatedFixtureBindingIds = new Set([
    'raster-embedding-kmeans',
    'raster-embedding-nearest-prototype',
    'raster-embedding-nearest-pixels',
    'raster-embedding-knn',
    'raster-embedding-linear-probe',
    'raster-embedding-change',
    'raster-embedding-covariance',
    'raster-embedding-pca',
    'raster-embedding-projection',
    'raster-embedding-dequantize',
    'raster-embedding-quantize',
    'raster-embedding-pyramid',
    'raster-embedding-similarity',
    'raster-embedding-zone-mean',
    'raster-embedding-usable-mask',
    'raster-lagrangian-ftle'
  ]);
  for (const id of generatedFixtureBindingIds) expect(requireBinding(id).inputs, id).toEqual([]);
  for (const entry of listCatalogEntries().filter(candidate => candidate.bindingIds.length)) {
    for (const id of entry.bindingIds) {
      const binding = requireBinding(id);
      if (!generatedFixtureBindingIds.has(id)) expect(binding.inputs.length, id).toBeGreaterThan(0);
      expect(binding.outputs.length).toBeGreaterThan(0);
      expect(binding.cleanup).not.toBe('');
      expect(binding.declaredBehavior.length).toBeGreaterThan(0);
      expect(() =>
        normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))
      ).not.toThrow();
    }
  }
});

test('serialized invocations restore parameters and bind to input content', () => {
  const binding = requireBinding('network-service-area');
  const inputs = {lines: networkLines, origin: pointCollection([[0, 0]])};
  const text = serializeCatalogInvocation(
    binding,
    {...networkParameters, budgets: '15, 30'},
    inputs
  );
  const restored = parseCatalogInvocation(text);
  expect(restored.invocation.parameters['budgets']).toEqual([15, 30]);
  expect(matchCatalogInvocationInputs(restored.invocation, inputs)).toBe(true);
  expect(
    matchCatalogInvocationInputs(restored.invocation, {
      ...inputs,
      origin: pointCollection([[1, 0]])
    })
  ).toBe(false);
  expect(() => parseCatalogInvocation(text.replace('network-service-area', 'nope'))).toThrow();
});

test('inputs must satisfy declared geometry kinds and capacity', () => {
  const binding = requireBinding('geometry-validity');
  expect(() => runCatalogOperation(binding, {}, {polygons: pointCollection([[0, 0]])})).toThrow();
  expect(() => runCatalogOperation(binding, {}, {})).toThrow();
});

test('validity reports bowties, open rings, and holes outside the shell without repairing', () => {
  const result = runCatalogOperation(
    requireBinding('geometry-validity'),
    {},
    {
      polygons: polygonCollection([
        [square(0, 0, 10)],
        [
          [
            [0, 0],
            [10, 10],
            [10, 0],
            [0, 10],
            [0, 0]
          ]
        ],
        [square(0, 0, 10), square(20, 20, 2)],
        [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
          ]
        ]
      ])
    }
  );
  const table = result.tables[0];
  expect(table.columns['status']).toEqual(['valid', 'invalid', 'invalid', 'invalid']);
  expect(String(table.columns['issues'][1])).toContain('self-intersection');
  expect(String(table.columns['issues'][2])).toContain('hole-outside-shell');
  expect(String(table.columns['issues'][3])).toContain('ring-not-closed');
});

test('measures subtract holes and use declared units', () => {
  const planar = runCatalogOperation(
    requireBinding('geometry-measures'),
    {coordinateSystem: 'planar'},
    {features: polygonCollection([[square(0, 0, 10), square(2, 2, 2).reverse()]])}
  );
  expect(planar.tables[0].columns['area'][0]).toBeCloseTo(96, 9);
  expect(planar.tables[0].columns['length'][0]).toBeCloseTo(48, 9);
  const geographic = runCatalogOperation(
    requireBinding('geometry-measures'),
    {coordinateSystem: 'EPSG:4326'},
    {
      features: lineCollection([
        {
          coordinates: [
            [0, 0],
            [1, 0]
          ]
        }
      ])
    }
  );
  expect(geographic.tables[0].columns['length'][0]).toBeCloseTo(111195, -1);
});

test('point buffer keeps source identity and approximates the circle area', () => {
  const result = runCatalogOperation(
    requireBinding('geometry-point-buffer'),
    {radius: 1000, segmentsPerQuarter: 32, coordinateSystem: 'EPSG:4326'},
    {
      points: pointCollection([
        [0, 0],
        [10, 45]
      ])
    }
  );
  const table = result.tables[0];
  expect(table.columns['buffer_id']).toEqual(['0:0', '1:0']);
  expect(table.columns['area'][0] as number).toBeGreaterThan(Math.PI * 1e6 * 0.99);
  expect(table.columns['area'][0] as number).toBeLessThan(Math.PI * 1e6);
});

test('segmentize preserves identity and bounds segment length', () => {
  const result = runCatalogOperation(
    requireBinding('geometry-line-segmentize'),
    {maximumSegmentLength: 3, coordinateSystem: 'planar'},
    {
      lines: lineCollection([
        {
          id: 'a',
          coordinates: [
            [0, 0],
            [10, 0]
          ]
        }
      ])
    }
  );
  const geometry = JSON.parse(String(result.tables[0].columns['geometry'][0]));
  expect(geometry.coordinates).toHaveLength(5);
  expect(result.tables[0].columns['source_id'][0]).toBe('a');
  expect(() =>
    runCatalogOperation(
      requireBinding('geometry-line-segmentize'),
      {maximumSegmentLength: 1e-6, coordinateSystem: 'planar'},
      {
        lines: lineCollection([
          {
            coordinates: [
              [0, 0],
              [10, 0]
            ]
          }
        ])
      }
    )
  ).toThrow();
});

test('convex clip returns the intersection, new part identities and rejects concave clips', () => {
  const binding = requireBinding('geometry-convex-clip');
  const result = runCatalogOperation(
    binding,
    {},
    {
      features: {
        type: 'FeatureCollection',
        features: [
          {geometry: {type: 'Polygon', coordinates: [square(0, 0, 10)]}},
          {
            geometry: {
              type: 'LineString',
              coordinates: [
                [-5, 7],
                [15, 7]
              ]
            }
          },
          {geometry: {type: 'Polygon', coordinates: [square(50, 50, 1)]}}
        ]
      },
      clip: polygonCollection([[square(5, 5, 10)]])
    }
  );
  const table = result.tables[0];
  expect(table.rowCount).toBe(2);
  const clipped = JSON.parse(String(table.columns['geometry'][0]));
  expect(clipped.type).toBe('Polygon');
  const area = Math.abs(
    runCatalogOperation(
      requireBinding('geometry-measures'),
      {coordinateSystem: 'planar'},
      {
        features: {type: 'FeatureCollection', features: [{geometry: clipped}]}
      }
    ).tables[0].columns['area'][0] as number
  );
  expect(area).toBeCloseTo(25, 9);
  expect(JSON.parse(String(table.columns['geometry'][1])).coordinates).toEqual([
    [5, 7],
    [15, 7]
  ]);
  expect(result.diagnostics.some(item => item.code === 'feature-outside-clip')).toBe(true);
  const concave = polygonCollection([
    [
      [
        [0, 0],
        [10, 0],
        [5, 2],
        [10, 10],
        [0, 10],
        [0, 0]
      ]
    ]
  ]);
  expect(() => runCatalogOperation(binding, {}, {features: concave, clip: concave})).toThrow();
});

test('network build keeps source edge IDs, directions and components', () => {
  const result = runCatalogOperation(requireBinding('network-build-graph'), buildParameters, {
    lines: networkLines
  });
  const edges = result.tables[0].columns;
  expect(edges['source_id']).toEqual(['ab', 'bc', 'cd', 'de', 'ea', 'island', 'spur', 'ga']);
  expect(edges['direction']).toEqual([
    'both',
    'both',
    'forward',
    'forward',
    'both',
    'both',
    'reverse',
    'forward'
  ]);
  expect(new Set(edges['component']).size).toBe(2);
  expect(result.summary['componentCount']).toBe(2);
});

test('shortest path costs match a graph reference and keep edge IDs', () => {
  const graph = runCatalogOperation(requireBinding('network-build-graph'), buildParameters, {
    lines: networkLines
  });
  const edges = graph.tables[0].columns;
  const nodeA = edges['from_node'][0] as number;
  const nodeD = edges['to_node'][2] as number;
  const result = runCatalogOperation(requireBinding('network-shortest-path'), networkParameters, {
    lines: networkLines,
    origin: pointCollection([[0, 0]]),
    destinations: pointCollection([
      [30, 10],
      [100, 100],
      [30, 20],
      [500, 500]
    ])
  });
  const paths = result.tables[0].columns;
  expect(paths['status']).toEqual([
    'ok',
    'unreachable-disconnected',
    'unreachable-directed',
    'destination-not-snapped'
  ]);
  expect(paths['total_cost'][0] as number).toBeCloseTo(
    computeReferenceCost(edges, nodeA, nodeD),
    9
  );
  expect(paths['edge_ids'][0]).toBe('0:0:0,1:0:0,2:0:0');
  expect(paths['source_indices'][0]).toBe('0,1,2');
  // Reverse direction takes the one-way return leg.
  const reverse = runCatalogOperation(requireBinding('network-shortest-path'), networkParameters, {
    lines: networkLines,
    origin: pointCollection([[30, 10]]),
    destinations: pointCollection([[0, 0]])
  });
  expect(reverse.tables[0].columns['total_cost'][0] as number).toBeCloseTo(
    computeReferenceCost(edges, nodeD, nodeA),
    9
  );
  expect(reverse.tables[0].columns['edge_ids'][0]).toBe('3:0:0,4:0:0');
});

test('shortest path handles points in the middle of an edge, including the same edge', () => {
  const result = runCatalogOperation(requireBinding('network-shortest-path'), networkParameters, {
    lines: networkLines,
    origin: pointCollection([[2, 0]]),
    destinations: pointCollection([
      [8, 0],
      [1, 0],
      [20, 0]
    ])
  });
  const costs = result.tables[0].columns['total_cost'] as number[];
  expect(costs[0]).toBeCloseTo(6, 9);
  expect(costs[1]).toBeCloseTo(1, 9);
  expect(costs[2]).toBeCloseTo(18, 9);
});

test('service area separates reached, partial and the three unreachable kinds', () => {
  const result = runCatalogOperation(
    requireBinding('network-service-area'),
    {...networkParameters, budgets: [15]},
    {lines: networkLines, origin: pointCollection([[0, 0]])}
  );
  const table = result.tables[0].columns;
  const statusOf = (sourceId: string) => table['status'][table['source_id'].indexOf(sourceId)];
  expect(statusOf('ab')).toBe('reached-full');
  expect(statusOf('bc')).toBe('reached-partial');
  expect(table['reached_fraction'][table['source_id'].indexOf('bc')]).toBeCloseTo(0.25, 9);
  expect(statusOf('ea')).toBe('reached-full');
  expect(statusOf('de')).toBe('unreachable-beyond-budget');
  expect(statusOf('ga')).toBe('unreachable-directed');
  expect(statusOf('island')).toBe('unreachable-disconnected');
  expect(result.diagnostics.map(item => item.code)).toContain('unreachable-disconnected');
  const kept = runCatalogOperation(
    requireBinding('network-service-area'),
    {...networkParameters, budgets: [15], includeUnreachable: false},
    {lines: networkLines, origin: pointCollection([[0, 0]])}
  );
  expect(kept.tables[0].rowCount).toBeLessThan(table['status'].length);
});

test('unsnappable origins produce an error diagnostic instead of results', () => {
  const result = runCatalogOperation(
    requireBinding('network-service-area'),
    {...networkParameters, budgets: [15]},
    {lines: networkLines, origin: pointCollection([[500, 500]])}
  );
  expect(result.tables[0].rowCount).toBe(0);
  expect(result.diagnostics[0].severity).toBe('error');
});
