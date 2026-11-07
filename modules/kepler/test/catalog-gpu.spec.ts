// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  runCatalogOperationAsync,
  type CatalogOperationBinding,
  type CatalogRunResult
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection, CatalogPosition} from '../src/analysis/catalog-geojson';
import {getCatalogPolygonArea, isCatalogPointInPolygon} from '../src/analysis/catalog-gpu-pack';
import {listCatalogLensScenarios, runCatalogLensScenario} from '../src/analysis/catalog-lens';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

// Oracle policy: the CPU reference engines of the catalog (engine: 'cpu') are the oracle for the
// eight bound operations; the new GPU-only operations are checked against closed-form areas and
// point-membership grids because no CPU engine exists for them.

function requireBinding(id: string): CatalogOperationBinding {
  const binding = getCatalogBinding(id);
  if (!binding) throw new Error(`Missing binding ${id}`);
  return binding;
}

function polygonCollection(polygons: number[][][][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: polygons.map((coordinates, index) => ({
      id: `polygon-${index}`,
      geometry: {type: 'Polygon', coordinates},
      properties: {}
    }))
  };
}

function lineCollection(
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

function pointCollection(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {}
    }))
  };
}

const square = (x: number, y: number, size: number): number[][] => [
  [x, y],
  [x + size, y],
  [x + size, y + size],
  [x, y + size],
  [x, y]
];

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

async function runBoth(
  device: Device,
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const binding = requireBinding(id);
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {device, engine: 'cpu'});
  expect(gpu.engine, `${id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors.length).toBeGreaterThan(0);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function getColumn(result: CatalogRunResult, column: string, table = 0) {
  return result.tables[table].columns[column];
}

function parseGeometry(text: unknown): {type: string; coordinates: unknown} {
  return JSON.parse(String(text));
}

function getRingArea(ring: readonly (readonly number[])[]): number {
  let doubled = 0;
  for (let index = 0; index < ring.length; index++) {
    const next = ring[(index + 1) % ring.length];
    doubled += ring[index][0] * next[1] - next[0] * ring[index][1];
  }
  return Math.abs(doubled / 2);
}

function getGeometryArea(text: unknown): number {
  const geometry = parseGeometry(text);
  const polygons = (
    geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates
  ) as number[][][][];
  return polygons.reduce(
    (sum, rings) =>
      sum +
      rings.reduce((ringSum, ring, index) => ringSum + (index ? -1 : 1) * getRingArea(ring), 0),
    0
  );
}

test('catalog measures run on GPUGeometryMeasures and match the CPU oracle', async () => {
  const device = await getDevice();
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        id: 'with-hole',
        geometry: {type: 'Polygon', coordinates: [square(0, 0, 10), square(2, 2, 2).reverse()]},
        properties: {}
      },
      {
        id: 'line',
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [3, 4],
            [3, 10]
          ]
        },
        properties: {}
      },
      {id: 'point', geometry: {type: 'Point', coordinates: [5, 5]}, properties: {}},
      {
        id: 'multi',
        geometry: {
          type: 'MultiPolygon',
          coordinates: [[square(20, 0, 2)], [square(30, 0, 3)]]
        },
        properties: {}
      }
    ]
  };
  const {gpu, cpu} = await runBoth(
    device,
    'geometry-measures',
    {coordinateSystem: 'planar'},
    {
      features
    }
  );
  for (const column of ['area', 'length', 'vertex_count', 'minimum_x', 'maximum_y']) {
    getColumn(cpu, column).forEach((expected, row) => {
      const actual = getColumn(gpu, column)[row];
      if (expected === null) expect(actual, `${column} row ${row}`).toBeNull();
      else expect(actual as number, `${column} row ${row}`).toBeCloseTo(expected as number, 3);
    });
  }
  // GPU centroids are area-weighted: the 10x10 square minus a 2x2 hole centred at (3,3) has centroid (500 - 12) / 96.
  expect(getColumn(gpu, 'centroid_x')[0] as number).toBeCloseTo(488 / 96, 3);
  const spherical = await runBoth(
    device,
    'geometry-measures',
    {coordinateSystem: 'EPSG:4326'},
    {
      features: polygonCollection([[square(-87.7, 41.85, 0.05)]])
    }
  );
  const expected = getColumn(spherical.cpu, 'area')[0] as number;
  expect(Math.abs((getColumn(spherical.gpu, 'area')[0] as number) / expected - 1)).toBeLessThan(
    2e-3
  );
  expect(
    Math.abs(
      (getColumn(spherical.gpu, 'length')[0] as number) /
        (getColumn(spherical.cpu, 'length')[0] as number) -
        1
    )
  ).toBeLessThan(2e-3);
});

test('catalog validity runs on GPUGeometryValidity and agrees with the CPU oracle on status', async () => {
  const device = await getDevice();
  const polygons = polygonCollection([
    [square(0, 0, 10)],
    [
      [
        [20, 0],
        [30, 10],
        [30, 0],
        [20, 10],
        [20, 0]
      ]
    ],
    [square(40, 0, 10), square(60, 0, 2)],
    [square(70, 0, 10), square(72, 2, 2).reverse()]
  ]);
  const {gpu, cpu} = await runBoth(device, 'geometry-validity', {}, {polygons});
  expect(getColumn(gpu, 'status')).toEqual(getColumn(cpu, 'status'));
  expect(String(getColumn(gpu, 'issues')[1])).toContain('self-intersection');
  expect(String(getColumn(gpu, 'issues')[2])).toContain('hole-outside-shell');
});

test('catalog segmentize runs on GPULineSegmentize and matches the CPU oracle in planar space', async () => {
  const device = await getDevice();
  const lines = lineCollection([
    {
      id: 'diagonal',
      coordinates: [
        [0, 0],
        [10, 10]
      ]
    },
    {
      id: 'bent',
      coordinates: [
        [0, 0],
        [5, 0],
        [5, 7]
      ]
    }
  ]);
  const {gpu, cpu} = await runBoth(
    device,
    'geometry-line-segmentize',
    {coordinateSystem: 'planar', maximumSegmentLength: 2},
    {lines}
  );
  expect(getColumn(gpu, 'added_vertex_count')).toEqual(getColumn(cpu, 'added_vertex_count'));
  getColumn(cpu, 'geometry').forEach((expectedText, row) => {
    const expected = parseGeometry(expectedText).coordinates as number[][];
    const actual = parseGeometry(getColumn(gpu, 'geometry')[row]).coordinates as number[][];
    expect(actual).toHaveLength(expected.length);
    actual.forEach((position, index) => {
      expect(position[0]).toBeCloseTo(expected[index][0], 4);
      expect(position[1]).toBeCloseTo(expected[index][1], 4);
    });
  });
  const spherical = await runBoth(
    device,
    'geometry-line-segmentize',
    {coordinateSystem: 'EPSG:4326', maximumSegmentLength: 20000},
    {
      lines: lineCollection([
        {
          id: 'long',
          coordinates: [
            [-87.7, 41.8],
            [-87.3, 42.0]
          ]
        }
      ])
    }
  );
  // Same piece count: both measure segment length with haversine.
  expect(getColumn(spherical.gpu, 'added_vertex_count')).toEqual(
    getColumn(spherical.cpu, 'added_vertex_count')
  );
});

test('catalog clip runs on GPURectangleClip for rectangles and falls back for other clips', async () => {
  const device = await getDevice();
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {id: 'polygon', geometry: {type: 'Polygon', coordinates: [square(0, 0, 10)]}, properties: {}},
      {
        id: 'outside',
        geometry: {type: 'Polygon', coordinates: [square(50, 50, 1)]},
        properties: {}
      },
      {
        id: 'line',
        geometry: {
          type: 'LineString',
          coordinates: [
            [-5, 5],
            [15, 5]
          ]
        },
        properties: {}
      }
    ]
  };
  const rectangle = polygonCollection([[square(5, 5, 10)]]);
  const {gpu, cpu} = await runBoth(device, 'geometry-convex-clip', {}, {features, clip: rectangle});
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(getColumn(gpu, 'source_index')).toEqual(getColumn(cpu, 'source_index'));
  expect(getGeometryArea(getColumn(gpu, 'geometry')[0])).toBeCloseTo(
    getGeometryArea(getColumn(cpu, 'geometry')[0]),
    3
  );
  const lineLength = (text: unknown) => {
    const coordinates = parseGeometry(text).coordinates as number[][];
    return Math.hypot(
      coordinates[coordinates.length - 1][0] - coordinates[0][0],
      coordinates[coordinates.length - 1][1] - coordinates[0][1]
    );
  };
  expect(lineLength(getColumn(gpu, 'geometry')[1])).toBeCloseTo(
    lineLength(getColumn(cpu, 'geometry')[1]),
    3
  );
  const triangle = polygonCollection([
    [
      [
        [5, 5],
        [15, 5],
        [10, 15],
        [5, 5]
      ]
    ]
  ]);
  const fallback = await runCatalogOperationAsync(
    requireBinding('geometry-convex-clip'),
    {},
    {features, clip: triangle},
    {device}
  );
  expect(fallback.engine).toBe('cpu-reference');
  expect(fallback.fallbackReason).toContain('axis-aligned rectangle');
  expect(fallback.diagnostics.some(item => item.code === 'gpu-fallback')).toBe(true);
});

test('polygon and line buffers match closed-form areas', async () => {
  const device = await getDevice();
  const binding = requireBinding('geometry-polygon-buffer');
  const grown = await runCatalogOperationAsync(
    binding,
    {distance: 2, coordinateSystem: 'planar', quadSegments: 16},
    {polygons: polygonCollection([[square(0, 0, 10)]])},
    {device}
  );
  expect(grown.engine).toBe('gpu');
  expect(grown.dispatchedContributors).toContain('GPUBufferSurface');
  const grownArea = 100 + 4 * 10 * 2 + Math.PI * 4;
  expect(getColumn(grown, 'area')[0] as number).toBeGreaterThan(grownArea * 0.99);
  expect(getColumn(grown, 'area')[0] as number).toBeLessThan(grownArea * 1.01);
  const shrunk = await runCatalogOperationAsync(
    binding,
    {distance: -2, coordinateSystem: 'planar'},
    {polygons: polygonCollection([[square(0, 0, 10)]])},
    {device}
  );
  expect(getColumn(shrunk, 'area')[0] as number).toBeCloseTo(36, 0);
  const collapsedHole = await runCatalogOperationAsync(
    binding,
    {distance: 2, coordinateSystem: 'planar'},
    {polygons: polygonCollection([[square(0, 0, 20), square(8, 8, 2).reverse()]])},
    {device}
  );
  // A two-unit hole closes under a two-unit buffer: no hole remains.
  expect(getColumn(collapsedHole, 'hole_count')[0]).toBe(0);
  const separate = await runCatalogOperationAsync(
    binding,
    {distance: 1, coordinateSystem: 'planar', dissolve: false},
    {polygons: polygonCollection([[square(0, 0, 10)], [square(50, 0, 10)]])},
    {device}
  );
  expect(getColumn(separate, 'source_index')).toEqual([0, 1]);
  const lineBuffer = await runCatalogOperationAsync(
    requireBinding('geometry-line-buffer'),
    {distance: 1, coordinateSystem: 'planar', quadSegments: 16},
    {
      lines: lineCollection([
        {
          id: 'straight',
          coordinates: [
            [0, 0],
            [10, 0]
          ]
        }
      ])
    },
    {device}
  );
  const lineArea = 2 * 1 * 10 + Math.PI;
  expect(Math.abs((getColumn(lineBuffer, 'area')[0] as number) / lineArea - 1)).toBeLessThan(0.01);
  const geographic = await runCatalogOperationAsync(
    binding,
    {distance: 500, coordinateSystem: 'EPSG:4326'},
    {polygons: polygonCollection([[square(-87.65, 41.88, 0.01)]])},
    {device}
  );
  // 0.01 degrees is about 830 m by 1110 m; a 500 m buffer adds roughly its perimeter times 500 m.
  const area = getColumn(geographic, 'area')[0] as number;
  expect(area).toBeGreaterThan(830 * 1110 + 2 * (830 + 1110) * 500 * 0.95);
  expect(area).toBeLessThan(830 * 1110 + 2 * (830 + 1110) * 500 * 1.1 + Math.PI * 500 * 500 * 1.1);
});

test('overlay operations match point-membership oracles', async () => {
  const device = await getDevice();
  const left = polygonCollection([[square(0, 0, 10)]]);
  const right = polygonCollection([[square(5, 5, 10)]]);
  const expectations = {
    intersection: {area: 25, inside: (a: boolean, b: boolean) => a && b},
    union: {area: 175, inside: (a: boolean, b: boolean) => a || b},
    difference: {area: 75, inside: (a: boolean, b: boolean) => a && !b}
  } as const;
  for (const [operation, expected] of Object.entries(expectations)) {
    const result = await runCatalogOperationAsync(
      requireBinding(`geometry-overlay-${operation}`),
      {coordinateSystem: 'planar'},
      {left, right},
      {device}
    );
    expect(result.engine).toBe('gpu');
    const total = getColumn(result, 'area').reduce<number>(
      (sum, value) => sum + (value as number),
      0
    );
    expect(total, operation).toBeCloseTo(expected.area, 1);
    const polygons = getColumn(result, 'geometry').map(
      text => parseGeometry(text).coordinates as CatalogPosition[][]
    );
    let mismatches = 0;
    for (let x = -1.25; x < 16; x += 0.5)
      for (let y = -1.25; y < 16; y += 0.5) {
        const inLeft = x > 0 && x < 10 && y > 0 && y < 10;
        const inRight = x > 5 && x < 15 && y > 5 && y < 15;
        const inResult = polygons.some(rings => isCatalogPointInPolygon([x, y], rings));
        if (inResult !== expected.inside(inLeft, inRight)) mismatches++;
      }
    expect(mismatches, `${operation} membership mismatches`).toBe(0);
    expect(getCatalogPolygonArea(polygons[0])).toBeGreaterThan(0);
  }
});

test('topology repair splits a bow tie and leaves valid polygons unchanged', async () => {
  const device = await getDevice();
  const result = await runCatalogOperationAsync(
    requireBinding('geometry-topology-repair'),
    {coordinateSystem: 'planar'},
    {
      polygons: polygonCollection([
        [
          [
            [0, 0],
            [10, 10],
            [10, 0],
            [0, 10],
            [0, 0]
          ]
        ],
        [square(20, 0, 10)]
      ])
    },
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(getColumn(result, 'status')).toEqual(['repaired', 'unchanged']);
  expect(getColumn(result, 'polygon_count')).toEqual([2, 1]);
  expect(getColumn(result, 'repaired_area')[0] as number).toBeCloseTo(50, 2);
  expect(getColumn(result, 'repaired_area')[1] as number).toBeCloseTo(100, 2);
});

test('crossing noding cuts lines at the shared crossing and numbers the nodes', async () => {
  const device = await getDevice();
  const result = await runCatalogOperationAsync(
    requireBinding('network-crossing-noding'),
    {coordinateSystem: 'planar', snapTolerance: 0},
    {
      lines: lineCollection([
        {
          id: 'east-west',
          coordinates: [
            [0, 5],
            [10, 5]
          ]
        },
        {
          id: 'north-south',
          coordinates: [
            [5, 0],
            [5, 10]
          ]
        },
        {
          id: 'diagonal',
          coordinates: [
            [0, 0],
            [10, 10]
          ]
        }
      ])
    },
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(result.tables).toHaveLength(2);
  // Three lines through (5, 5): six pieces and seven nodes (six ends and the crossing).
  expect(result.tables[0].rowCount).toBe(6);
  expect(result.tables[1].rowCount).toBe(7);
  expect(Math.max(...(getColumn(result, 'degree', 1) as number[]))).toBe(6);
  expect([...new Set(getColumn(result, 'source_index', 0))]).toEqual([0, 1, 2]);
});

const GRID_NETWORK = lineCollection(
  [0, 50, 100].flatMap((offset, index) => [
    {
      id: `row-${index}`,
      coordinates: [
        [0, offset],
        [50, offset],
        [100, offset]
      ]
    },
    {
      id: `column-${index}`,
      coordinates: [
        [offset, 0],
        [offset, 50],
        [offset, 100]
      ]
    }
  ])
);

const ROUTING_PARAMETERS = {
  coordinateSystem: 'planar',
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'length',
  maximumSnapDistance: 10
};

test('service area and shortest paths run on GPUNetworkReachability and match the CPU oracle', async () => {
  const device = await getDevice();
  const area = await runBoth(
    device,
    'network-service-area',
    {...ROUTING_PARAMETERS, budgets: '60, 120, 250'},
    {lines: GRID_NETWORK, origin: pointCollection([[0, 0]])}
  );
  expect(getColumn(area.gpu, 'status')).toEqual(getColumn(area.cpu, 'status'));
  expect(getColumn(area.gpu, 'band')).toEqual(getColumn(area.cpu, 'band'));
  getColumn(area.cpu, 'first_reach_cost').forEach((expected, row) =>
    expect(getColumn(area.gpu, 'first_reach_cost')[row] as number).toBeCloseTo(
      expected as number,
      2
    )
  );
  expect(area.gpu.dispatchedContributors).toContain('GPUNetworkReachability');
  const paths = await runBoth(device, 'network-shortest-path', ROUTING_PARAMETERS, {
    lines: GRID_NETWORK,
    origin: pointCollection([[0, 0]]),
    destinations: pointCollection([
      [100, 100],
      [50, 100],
      [500, 500]
    ])
  });
  expect(getColumn(paths.gpu, 'status')).toEqual(getColumn(paths.cpu, 'status'));
  getColumn(paths.cpu, 'total_cost').forEach((expected, row) => {
    if (expected !== null)
      expect(getColumn(paths.gpu, 'total_cost')[row] as number).toBeCloseTo(expected as number, 2);
  });
  expect(getColumn(paths.gpu, 'total_cost')[0] as number).toBeCloseTo(200, 2);
});

test('crossing lines connect only when the GPU engine nodes them', async () => {
  const device = await getDevice();
  const crossing = lineCollection([
    {
      id: 'east-west',
      coordinates: [
        [0, 50],
        [100, 50]
      ]
    },
    {
      id: 'north-south',
      coordinates: [
        [50, 0],
        [50, 100]
      ]
    }
  ]);
  const parameters = {...ROUTING_PARAMETERS, budgets: '200', maximumSnapDistance: 5};
  const inputs = {lines: crossing, origin: pointCollection([[0, 50]])};
  const binding = requireBinding('network-service-area');
  const plain = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const noded = await runCatalogOperationAsync(
    binding,
    {...parameters, crossings: 'noded'},
    inputs,
    {device}
  );
  expect(noded.engine).toBe('gpu');
  expect(noded.dispatchedContributors).toContain('GPUNetworkNoding');
  const reached = (result: CatalogRunResult) =>
    getColumn(result, 'status').filter(status => String(status).startsWith('reached')).length;
  // Without noding the north-south road is a separate component; noded, its halves are reached.
  expect(reached(noded)).toBeGreaterThan(reached(plain));
  expect(getColumn(noded, 'source_index').every(value => value === 0 || value === 1)).toBe(true);
  const {maximumSnapDistance: _snapDistance, ...buildParameters} = ROUTING_PARAMETERS;
  const build = await runCatalogOperationAsync(
    requireBinding('network-build-graph'),
    {...buildParameters, crossings: 'noded'},
    {lines: crossing},
    {device}
  );
  expect(build.engine).toBe('gpu');
  expect(build.tables[0].rowCount).toBe(4);
  const unsupported = await runCatalogOperationAsync(
    requireBinding('network-build-graph'),
    buildParameters,
    {lines: crossing},
    {device}
  );
  expect(unsupported.engine).toBe('cpu-reference');
  expect(unsupported.fallbackReason).toContain('crossings');
});

test('multi-origin service areas match one CPU service area per origin', async () => {
  const device = await getDevice();
  const origins = pointCollection([
    [0, 0],
    [100, 100]
  ]);
  const result = await runCatalogOperationAsync(
    requireBinding('network-multi-origin-service-area'),
    {...ROUTING_PARAMETERS, budgets: '120, 400'},
    {lines: GRID_NETWORK, origins},
    {device}
  );
  expect(result.engine).toBe('gpu');
  const binding = requireBinding('network-service-area');
  for (const originIndex of [0, 1]) {
    const reference = await runCatalogOperationAsync(
      binding,
      {...ROUTING_PARAMETERS, budgets: '120, 400', includeUnreachable: false},
      {
        lines: GRID_NETWORK,
        origin: {type: 'FeatureCollection', features: [origins.features[originIndex]]}
      },
      {device, engine: 'cpu'}
    );
    const rows = getColumn(result, 'origin_index')
      .map((value, row) => (value === originIndex ? row : -1))
      .filter(row => row >= 0);
    expect(rows).toHaveLength(reference.tables[0].rowCount);
    rows.forEach((row, position) => {
      expect(getColumn(result, 'edge_id')[row]).toBe(getColumn(reference, 'edge_id')[position]);
      expect(getColumn(result, 'first_reach_cost')[row] as number).toBeCloseTo(
        getColumn(reference, 'first_reach_cost')[position] as number,
        2
      );
    });
  }
  // Every edge has at least one nearest origin.
  expect(getColumn(result, 'is_nearest_origin').some(Boolean)).toBe(true);
});

test('isochrone polygons contain nearer points in earlier bands', async () => {
  const device = await getDevice();
  const result = await runCatalogOperationAsync(
    requireBinding('network-isochrone-polygons'),
    {
      ...ROUTING_PARAMETERS,
      budgets: '30, 80, 160',
      bufferRadius: 6,
      rasterSize: 256
    },
    {lines: GRID_NETWORK, origins: pointCollection([[50, 50]])},
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(result.dispatchedContributors).toContain('GPUNetworkIsochrones');
  const bands = getColumn(result, 'band') as number[];
  const polygons = getColumn(result, 'geometry').map(
    text => parseGeometry(text).coordinates as CatalogPosition[][]
  );
  const bandOf = (point: CatalogPosition): number => {
    const index = polygons.findIndex(rings => isCatalogPointInPolygon(point, rings));
    return index < 0 ? -1 : bands[index];
  };
  // Street points at network cost 20, 20, 50 and 100 from the origin at the grid center.
  expect(bandOf([50, 70])).toBe(0);
  expect(bandOf([50, 100])).toBe(1);
  expect(bandOf([30, 50])).toBe(0);
  expect(bandOf([100, 100])).toBe(2);
  expect(bandOf([500, 500])).toBe(-1);
  expect(new Set(bands).size).toBeGreaterThanOrEqual(2);
});

test('a registered lens scenario runs through the catalog and reports its recorded nodes', async () => {
  const device = await getDevice();
  await import('@luma-contributors/lenses/computation-lenses/scenarios/index');
  const scenarios = listCatalogLensScenarios();
  expect(scenarios.length).toBeGreaterThan(0);
  const scenario = scenarios.find(candidate => candidate.element === 'GPUNetworkNoding');
  expect(scenario).toBeDefined();
  const output = await runCatalogLensScenario(device, scenario!.id);
  expect(output.dispatchedContributors).toEqual(['GPUNetworkNoding']);
  expect(output.tables[0].rowCount).toBeGreaterThan(0);
  expect(Number(output.summary['graphCount'])).toBeGreaterThan(0);
});

test('catalog point buffer runs on GPUPointBuffer and matches the CPU oracle', async () => {
  const device = await getDevice();
  const points = pointCollection([
    [-87.65, 41.85],
    [10.2, -33.4],
    [120.5, 0.01]
  ]);
  const geographic = await runBoth(
    device,
    'geometry-point-buffer',
    {radius: 250, segmentsPerQuarter: 6, coordinateSystem: 'EPSG:4326'},
    {points}
  );
  expect(geographic.gpu.dispatchedContributors).toEqual(['GPUPointBuffer']);
  expect(geographic.gpu.tables[0].rowCount).toBe(geographic.cpu.tables[0].rowCount);
  getColumn(geographic.cpu, 'geometry').forEach((expectedText, row) => {
    const expected = parseGeometry(expectedText).coordinates as number[][][];
    const actual = parseGeometry(getColumn(geographic.gpu, 'geometry')[row])
      .coordinates as number[][][];
    expect(actual[0].length).toBe(expected[0].length);
    expected[0].forEach((position, index) => {
      // 250 m is about 2.2e-3 degrees; float32 offsets keep ~1e-7 relative precision.
      expect(actual[0][index][0]).toBeCloseTo(position[0], 6);
      expect(actual[0][index][1]).toBeCloseTo(position[1], 6);
    });
  });
  const planar = await runBoth(
    device,
    'geometry-point-buffer',
    {radius: 5, segmentsPerQuarter: 4, coordinateSystem: 'planar'},
    {points: pointCollection([[100, 200]])}
  );
  expect(getColumn(planar.gpu, 'area')[0] as number).toBeCloseTo(
    getColumn(planar.cpu, 'area')[0] as number,
    3
  );
  const polar = await runBoth(
    device,
    'geometry-point-buffer',
    {radius: 5000, segmentsPerQuarter: 4, coordinateSystem: 'EPSG:4326'},
    {
      points: pointCollection([
        [0, 89.99],
        [0, 10]
      ])
    }
  );
  expect(polar.gpu.tables[0].rowCount).toBe(polar.cpu.tables[0].rowCount);
});
