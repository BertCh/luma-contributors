// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as spatialAnalysis from '@luma-contributors/analysis/gpu-spatial-analysis';
import * as terrainContributors from '@luma-contributors/analysis/gpu-terrain';
import * as network from '@luma-contributors/analysis/gpu-network';
import * as geometryContributors from '@luma-contributors/analysis/geospatial-analysis';
import * as dataframeContributors from '@luma-contributors/analysis/dataframe-analysis';
import * as rasterContributors from '@luma-contributors/analysis/raster-analysis';
import * as pointCloudContributors from '@luma-contributors/analysis/pointcloud-analysis';
import * as publishedRasterContributors from '@luma.gl/graph/gpu-raster';
import {
  KEPLER_CATALOG_BINDINGS,
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  runCatalogOperationAsync,
  type CatalogOperationBinding
} from '../src/analysis/catalog-bindings';
import {measureDistance, type CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {
  buildNetworkGraph,
  makeOriginLegs,
  runNetworkServiceArea,
  runNetworkShortestPaths,
  searchNetwork,
  snapPointToNetwork
} from '../src/analysis/catalog-network';
import {
  buildCatalogDirectedCsr,
  getNetworkGraphArcs,
  makeNetworkSearchFromCosts,
  prorateNodedProperties,
  remapNodedSourceIndices,
  type CatalogNodedNetwork
} from '../src/analysis/catalog-gpu-network';
import {describeValidityMask, runGpuGeometryValidity} from '../src/analysis/catalog-gpu-geometry';
import * as validitySession from '../src/analysis/catalog-gpu-session';
import {
  CATALOG_IDENTITY_FRAME,
  createCatalogAzimuthalFrame,
  getCatalogPolygonArea,
  getCatalogRectangle,
  isCatalogPointInPolygon,
  packCatalogLines,
  packCatalogPolygons,
  readCatalogPolygons
} from '../src/analysis/catalog-gpu-pack';
import {listCatalogLensScenarios} from '../src/analysis/catalog-lens';
import {makeCatalogTable} from '../src/analysis/catalog-geojson';

function lines(
  rows: {id: string; coordinates: number[][]; properties?: Record<string, unknown>}[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map(row => ({
      id: row.id,
      geometry: {type: 'LineString', coordinates: row.coordinates},
      properties: row.properties ?? {}
    }))
  };
}

function points(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {}
    }))
  };
}

const NETWORK = lines([
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
  }
]);

const NETWORK_PARAMETERS = {
  coordinateSystem: 'planar' as const,
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'length' as const,
  forwardCostProperty: '',
  reverseCostProperty: '',
  onewayProperty: 'oneway',
  maximumSnapDistance: 5
};

function requireBinding(id: string): CatalogOperationBinding {
  const binding = getCatalogBinding(id);
  if (!binding) throw new Error(`Missing binding ${id}`);
  return binding;
}

test('azimuthal frame round-trips and keeps distances from the center exact', () => {
  const frame = createCatalogAzimuthalFrame(-87.65, 41.88);
  for (const position of [
    [-87.65, 41.88],
    [-87.6, 41.9],
    [-87.7, 41.85],
    [-86.9, 42.4]
  ] as const) {
    const [x, y] = frame.project(position);
    const [longitude, latitude] = frame.unproject(x, y);
    expect(longitude).toBeCloseTo(position[0], 9);
    expect(latitude).toBeCloseTo(position[1], 9);
  }
  const [x, y] = frame.project([-87.2, 42.1]);
  expect(Math.hypot(x, y)).toBeCloseTo(
    measureDistance([-87.65, 41.88], [-87.2, 42.1], 'EPSG:4326'),
    3
  );
  expect(CATALOG_IDENTITY_FRAME.project([3, 4])).toEqual([3, 4]);
});

test('polygon and line packing keeps the GeoArrow layout and source identity', () => {
  const features: CatalogFeatureCollection['features'] = [
    {geometry: {type: 'Point', coordinates: [0, 0]}},
    {
      geometry: {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [4, 0],
              [4, 4],
              [0, 4],
              [0, 0]
            ],
            [
              [1, 1],
              [1, 2],
              [2, 2],
              [1, 1]
            ]
          ],
          [
            [
              [10, 10],
              [12, 10],
              [12, 12],
              [10, 10]
            ]
          ]
        ]
      }
    },
    {
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0]
          ]
        ]
      }
    }
  ];
  const kept = packCatalogPolygons(features, {closing: 'keep'});
  expect(kept.sourceIndices).toEqual([1, 2]);
  expect(Array.from(kept.featureOffsets)).toEqual([0, 2, 3]);
  expect(Array.from(kept.polygonOffsets)).toEqual([0, 2, 3, 4]);
  expect(Array.from(kept.ringOffsets)).toEqual([0, 5, 9, 13, 17]);
  const dropped = packCatalogPolygons(features, {closing: 'drop'});
  expect(Array.from(dropped.ringOffsets)).toEqual([0, 4, 7, 10, 13]);
  const read = readCatalogPolygons(kept.positions, kept.ringOffsets, kept.polygonOffsets, [0, 3]);
  expect(read).toHaveLength(3);
  expect(read[0][0]).toHaveLength(5);
  expect(getCatalogPolygonArea(read[0])).toBeCloseTo(16 - 0.5, 6);
  const packedLines = packCatalogLines(
    lines([
      {
        id: 'a',
        coordinates: [
          [0, 0],
          [1, 1]
        ]
      },
      {
        id: 'b',
        coordinates: [
          [2, 2],
          [3, 3],
          [4, 4]
        ]
      }
    ]).features
  );
  expect(Array.from(packedLines.lineOffsets)).toEqual([0, 2, 5]);
  expect(packedLines.lineSourceIndices).toEqual([0, 1]);
});

test('rectangle detection accepts axis-aligned rectangles only', () => {
  expect(
    getCatalogRectangle([
      [0, 0],
      [4, 0],
      [4, 2],
      [0, 2],
      [0, 0]
    ])
  ).toEqual({
    minX: 0,
    minY: 0,
    maxX: 4,
    maxY: 2
  });
  expect(
    getCatalogRectangle([
      [0, 0],
      [4, 1],
      [4, 2],
      [0, 2],
      [0, 0]
    ])
  ).toBeNull();
  expect(
    getCatalogRectangle([
      [0, 0],
      [4, 0],
      [2, 3],
      [0, 0]
    ])
  ).toBeNull();
  expect(
    getCatalogRectangle([
      [0, 0],
      [4, 0],
      [4, 2],
      [0, 2],
      [1, 1],
      [0, 0]
    ])
  ).toBeNull();
});

test('point-in-polygon honors holes', () => {
  const rings = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [0, 0]
    ],
    [
      [4, 4],
      [6, 4],
      [6, 6],
      [4, 6],
      [4, 4]
    ]
  ] as [number, number][][];
  expect(isCatalogPointInPolygon([2, 2], rings)).toBe(true);
  expect(isCatalogPointInPolygon([5, 5], rings)).toBe(false);
  expect(isCatalogPointInPolygon([12, 5], rings)).toBe(false);
});

test('validity masks map to catalog issue names and separate errors from conventions', () => {
  const bit = spatialAnalysis.GPU_GEOMETRY_VALIDITY_BIT;
  const described = describeValidityMask(
    bit.selfIntersection | bit.repeatedVertex | bit.badOrientation | bit.holeOutsideShell
  );
  expect(described.errors).toEqual(['self-intersection', 'hole-outside-shell']);
  expect(described.warnings).toEqual([
    'duplicate-consecutive-vertex',
    'shell-not-counterclockwise'
  ]);
  expect(describeValidityMask(0)).toEqual({errors: [], warnings: []});
});

test('GPU validity does not claim a clean row after intersection retries remain incomplete', async () => {
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [1, 0],
              [0, 1],
              [0, 0]
            ]
          ]
        },
        properties: {}
      }
    ]
  };
  for (const overflow of [0, 1]) {
    const spy = vi
      .spyOn(validitySession, 'withCatalogGpuSession')
      .mockResolvedValue({mask: [0], overflow} as never);
    try {
      const result = await runGpuGeometryValidity({device: null} as never, polygons);
      expect(result.tables[0].columns['status']).toEqual([overflow ? 'incomplete' : 'valid']);
      expect(spy).toHaveBeenCalledTimes(overflow ? 2 : 1);
    } finally {
      spy.mockRestore();
    }
  }
});

test('directed CSR groups arcs by source node in arc order', () => {
  const csr = buildCatalogDirectedCsr(4, [
    {from: 2, to: 0, cost: 5},
    {from: 0, to: 1, cost: 1},
    {from: 0, to: 2, cost: 2},
    {from: 3, to: 3, cost: 7}
  ]);
  expect(Array.from(csr.offsets)).toEqual([0, 2, 2, 3, 4]);
  expect(Array.from(csr.neighbors)).toEqual([1, 2, 0, 3]);
  expect(Array.from(csr.weights)).toEqual([1, 2, 5, 7]);
});

test('a search rebuilt from reachability costs reproduces the CPU service area and shortest paths', () => {
  const graph = buildNetworkGraph(NETWORK, NETWORK_PARAMETERS, 1000);
  const snapped = snapPointToNetwork(graph, [0, 0], 5);
  if (!snapped) throw new Error('Origin must snap');
  const seeds = makeOriginLegs(graph, snapped).map(leg => ({node: leg.node, cost: leg.cost}));
  const reference = searchNetwork(graph, seeds);
  // Float32 rounding is what a GPU run returns; predecessors are the CPU parents here.
  const costs = Float32Array.from(reference.distance);
  const predecessors = Uint32Array.from(reference.parentNode, parent =>
    parent < 0 ? 0xffffffff : parent
  );
  const rebuilt = makeNetworkSearchFromCosts(graph, costs, predecessors, seeds);
  for (let node = 0; node < graph.nodes.length; node++) {
    expect(rebuilt.distance[node]).toBeCloseTo(reference.distance[node], 4);
    expect(rebuilt.parentNode[node]).toBe(reference.parentNode[node]);
    expect(rebuilt.parentEdge[node]).toBe(reference.parentEdge[node]);
    expect(rebuilt.seed[node]).toBe(reference.seed[node]);
  }
  const parameters = {...NETWORK_PARAMETERS, budgets: [15, 30], includeUnreachable: true};
  const limits = {maximumEdges: 1000};
  const cpu = runNetworkServiceArea(NETWORK, points([[0, 0]]), parameters, limits);
  const injected = runNetworkServiceArea(NETWORK, points([[0, 0]]), parameters, limits, {
    graph,
    search: () => rebuilt
  });
  expect(injected.tables[0].columns['status']).toEqual(cpu.tables[0].columns['status']);
  expect(injected.tables[0].columns['band']).toEqual(cpu.tables[0].columns['band']);
  const paths = runNetworkShortestPaths(
    NETWORK,
    points([[0, 0]]),
    points([
      [30, 10],
      [100, 100]
    ]),
    parameters,
    {maximumEdges: 1000, maximumDestinations: 10},
    {graph, search: () => rebuilt}
  );
  const cpuPaths = runNetworkShortestPaths(
    NETWORK,
    points([[0, 0]]),
    points([
      [30, 10],
      [100, 100]
    ]),
    parameters,
    {maximumEdges: 1000, maximumDestinations: 10}
  );
  expect(paths.tables[0].columns['status']).toEqual(cpuPaths.tables[0].columns['status']);
  expect(paths.tables[0].columns['edge_ids']).toEqual(cpuPaths.tables[0].columns['edge_ids']);
  expect(getNetworkGraphArcs(graph).length).toBe(
    graph.edges.reduce(
      (count, edge) =>
        count + (edge.forwardCost !== null ? 1 : 0) + (edge.reverseCost !== null ? 1 : 0),
      0
    )
  );
});

test('noded pieces prorate property costs and map back to input features', () => {
  const source = lines([
    {
      id: 'road',
      coordinates: [
        [0, 0],
        [10, 0]
      ],
      properties: {minutes: 10}
    }
  ]);
  const noded: CatalogNodedNetwork = {
    pieces: lines([
      {
        id: 'road',
        coordinates: [
          [0, 0],
          [4, 0]
        ],
        properties: {minutes: 10}
      },
      {
        id: 'road',
        coordinates: [
          [4, 0],
          [10, 0]
        ],
        properties: {minutes: 10}
      }
    ]),
    pieceSourceIndices: [0, 0],
    piecePartIndices: [0, 0],
    fromNodes: Uint32Array.of(0, 1),
    toNodes: Uint32Array.of(1, 2),
    lengths: Float32Array.of(4, 6),
    nodes: [],
    frame: CATALOG_IDENTITY_FRAME
  };
  prorateNodedProperties(
    noded,
    source,
    {costMode: 'property', forwardCostProperty: 'minutes', reverseCostProperty: ''},
    coordinates => Math.hypot(coordinates[coordinates.length - 1][0] - coordinates[0][0], 0)
  );
  const shares = noded.pieces.features.map(feature => Number(feature.properties?.['minutes']));
  expect(shares[0]).toBeCloseTo(4, 9);
  expect(shares[1]).toBeCloseTo(6, 9);
  const table = makeCatalogTable('t', 't', {source_index: 'int32', source_indices: 'string'});
  table.columns['source_index'].push(1, 0);
  table.columns['source_indices'].push('1,0', null);
  remapNodedSourceIndices([table], [7, 9]);
  expect(table.columns['source_index']).toEqual([9, 7]);
  expect(table.columns['source_indices']).toEqual(['9,7', null]);
});

test('every GPU contributor a binding names exists at its public package entry point', () => {
  const exports = {
    ...spatialAnalysis,
    ...network,
    ...terrainContributors,
    ...geometryContributors,
    ...publishedRasterContributors,
    ...rasterContributors,
    ...pointCloudContributors,
    ...dataframeContributors
  } as Record<string, unknown>;
  const missing: string[] = [];
  for (const binding of KEPLER_CATALOG_BINDINGS)
    for (const name of binding.gpuContributorNames)
      if (typeof exports[name] !== 'function') missing.push(`${binding.id}:${name}`);
  expect(missing).toEqual([]);
});

test('bindings declare their engine honestly and GPU-only operations carry no CPU engine', () => {
  const bindingIds = KEPLER_CATALOG_BINDINGS.map(binding => binding.id);
  expect(new Set(bindingIds).size).toBe(bindingIds.length);
  for (const binding of KEPLER_CATALOG_BINDINGS) {
    // Feature and typed-data bindings expose different runner methods for the same engine.
    const hasCpuEngine = typeof binding.run === 'function' || typeof binding.runData === 'function';
    const hasGpuEngine =
      typeof binding.runGpu === 'function' || typeof binding.runDataGpu === 'function';
    if (binding.engine === 'cpu-reference') {
      expect(hasCpuEngine, `${binding.id}: CPU reference engine required`).toBe(true);
      expect(hasGpuEngine, `${binding.id}: CPU-only binding must have no GPU engine`).toBe(false);
      expect(binding.gpuContributorNames).toEqual([]);
    } else if (binding.engine === 'gpu') {
      expect(hasCpuEngine, `${binding.id}: GPU-only binding must have no CPU engine`).toBe(false);
      expect(hasGpuEngine, `${binding.id}: GPU engine required`).toBe(true);
    } else {
      expect(hasCpuEngine, `${binding.id}: CPU fallback engine required`).toBe(true);
      expect(hasGpuEngine, `${binding.id}: GPU engine required`).toBe(true);
    }
    expect(() => getCatalogDefaultParameters(binding)).not.toThrow();
  }
  // All eight CPU-bound operations have a GPU path (the point buffer uses GPUPointBuffer).
  for (const id of [
    'geometry-point-buffer',
    'geometry-validity',
    'geometry-measures',
    'geometry-line-segmentize',
    'geometry-convex-clip',
    'network-build-graph',
    'network-shortest-path',
    'network-service-area'
  ])
    expect(requireBinding(id).engine).toBe('gpu-or-cpu');
  for (const id of [
    'geometry-line-buffer',
    'geometry-polygon-buffer',
    'geometry-overlay-intersection',
    'geometry-overlay-union',
    'geometry-overlay-difference',
    'geometry-topology-repair',
    'network-crossing-noding',
    'network-multi-origin-service-area',
    'network-isochrone-polygons'
  ])
    expect(requireBinding(id).engine).toBe('gpu');
});

test('catalog entries name the dispatched engine and keep honest reasons for the rest', () => {
  const entries = listCatalogEntries({lensScenarios: []});
  const byName = new Map(entries.map(entry => [entry.identity.exportName, entry]));
  expect(byName.get('GPUGeometryMeasures')?.engine).toBe('gpu-or-cpu');
  expect(byName.get('GPUPolygonOverlay')?.engine).toBe('gpu');
  expect(byName.get('GPUNetworkIsochrones')?.engine).toBe('gpu');
  expect(byName.get('GPUNetworkReachability')?.runnable).toBe('catalog-binding');
  // Drive-time bands dispatch service areas on the GPU. Network service areas also have a CPU route.
  expect(byName.get('GPUNetworkServiceAreas')?.engine).toBe('gpu-or-cpu');
  const unbound = entries.filter(entry => entry.runnable === 'no');
  expect(unbound.length).toBeGreaterThan(0);
  for (const entry of unbound) {
    expect(entry.reason).toContain('Computation-pack operator and formula entries describe');
    expect(entry.reason).toContain('lens scenario registry is empty');
  }
  const withLens = listCatalogEntries({
    lensScenarios: [
      {
        id: 'gpu-spatial-analysis.local-moran',
        entry: 'gpu-spatial-analysis',
        element: 'GPULocalMoran',
        title: 'Local Moran',
        audience: 'analyst'
      }
    ]
  });
  const moran = withLens.find(entry => entry.identity.exportName === 'GPULocalMoran');
  if (moran) {
    // A registered catalog binding wins over the lens scenario, which stays listed alongside it.
    expect(moran.runnable).toBe('catalog-binding');
    expect(moran.lensScenarioIds).toEqual(['gpu-spatial-analysis.local-moran']);
  }
  for (const entry of withLens.filter(candidate => candidate.runnable === 'no'))
    expect(entry.reason).not.toContain('registry is empty');
  // The registry is empty until a host imports the scenario modules.
  expect(Array.isArray(listCatalogLensScenarios())).toBe(true);
});

test('without a device operations fall back to the CPU engine and GPU-only operations throw', async () => {
  const measures = requireBinding('geometry-measures');
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [4, 0],
              [4, 3],
              [0, 3],
              [0, 0]
            ]
          ]
        },
        properties: {}
      }
    ]
  };
  const result = await runCatalogOperationAsync(measures, {coordinateSystem: 'planar'}, {features});
  expect(result.engine).toBe('cpu-reference');
  expect(result.dispatchedContributors).toEqual([]);
  expect(result.fallbackReason).toBe('No GPU device supplied');
  expect(result.tables[0].columns['area'][0]).toBe(12);
  const forced = await runCatalogOperationAsync(
    measures,
    {coordinateSystem: 'planar'},
    {features},
    {engine: 'cpu'}
  );
  expect(forced.fallbackReason).toBe('CPU engine requested');
  await expect(
    runCatalogOperationAsync(
      requireBinding('geometry-polygon-buffer'),
      {distance: 1, coordinateSystem: 'planar'},
      {polygons: features}
    )
  ).rejects.toThrow('GPU device required');
  const noded = await runCatalogOperationAsync(
    requireBinding('network-service-area'),
    {...NETWORK_PARAMETERS, crossings: 'noded', budgets: [15]},
    {lines: NETWORK, origin: points([[0, 0]])}
  );
  expect(noded.diagnostics.some(item => item.code === 'crossings-not-noded-cpu')).toBe(true);
});
