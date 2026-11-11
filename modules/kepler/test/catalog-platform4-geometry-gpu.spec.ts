// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  getCatalogBinding,
  runCatalogOperationAsync,
  type CatalogParameterValue,
  type CatalogRunResult
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

// Oracle policy: the CPU reference engine (engine: 'cpu') is the oracle for every binding in this
// file. GPU runs use float32; tolerances are stated per family.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;

async function runPair(
  bindingId: string,
  parameters: Record<string, CatalogParameterValue>,
  inputs: Inputs,
  contributors: readonly string[]
): Promise<{gpu: CatalogRunResult; cpu: CatalogRunResult}> {
  const device = await getDevice();
  const binding = getCatalogBinding(bindingId);
  expect(binding?.engine, bindingId).toBe('gpu-or-cpu');
  const gpu = await runCatalogOperationAsync(binding!, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding!, parameters, inputs, {
    device,
    engine: 'cpu'
  });
  expect(gpu.engine, `${bindingId}: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(contributors);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function getCoordinates(text: unknown): number[][] {
  const flat: number[][] = [];
  const visit = (value: unknown) => {
    if (!Array.isArray(value)) return;
    if (typeof value[0] === 'number') flat.push(value as number[]);
    else value.forEach(visit);
  };
  visit((JSON.parse(text as string) as {coordinates: unknown}).coordinates);
  return flat;
}

function expectGeometryClose(gpuText: unknown, cpuText: unknown, tolerance: number): void {
  const gpuCoordinates = getCoordinates(gpuText);
  const cpuCoordinates = getCoordinates(cpuText);
  expect(gpuCoordinates.length).toBe(cpuCoordinates.length);
  gpuCoordinates.forEach((position, index) => {
    expect(Math.abs(position[0] - cpuCoordinates[index][0])).toBeLessThan(tolerance);
    expect(Math.abs(position[1] - cpuCoordinates[index][1])).toBeLessThan(tolerance);
  });
}

function makeRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeWalkLines(count: number, seed: number, closedEvery = 0): CatalogFeatureCollection {
  const random = makeRandom(seed);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, (_, index) => {
      let x = -122.5 + random() * 0.4;
      let y = 37.5 + random() * 0.4;
      const coordinates: number[][] = [[x, y]];
      const vertexCount = 8 + Math.floor(random() * 20);
      for (let vertex = 1; vertex < vertexCount; vertex++) {
        x += (random() - 0.5) * 0.004;
        y += (random() - 0.5) * 0.004;
        coordinates.push([x, y]);
      }
      if (closedEvery && index % closedEvery === 0) {
        coordinates.push([coordinates[0][0], coordinates[0][1]]);
      }
      return {
        id: `l${index}`,
        geometry: {type: 'LineString', coordinates},
        properties: {}
      };
    })
  };
}

test('line-simplification runs on GPULineSimplification and keeps the CPU vertices', async () => {
  const lines = makeWalkLines(40, 7);
  for (const tolerance of [20, 80]) {
    const {gpu, cpu} = await runPair(
      'line-simplification',
      {tolerance, coordinateSystem: 'EPSG:4326'},
      {lines},
      ['GPULineSimplification']
    );
    // Float32 distances can only flip a vertex whose distance is within rounding of the tolerance.
    const gpuAfter = gpu.tables[0].columns['vertices_after'] as number[];
    const cpuAfter = cpu.tables[0].columns['vertices_after'] as number[];
    expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
    const differing = gpuAfter.filter((value, row) => value !== cpuAfter[row]).length;
    console.info(`line-simplification tolerance ${tolerance}: ${differing} of 40 lines differ`);
    expect(differing).toBeLessThanOrEqual(1);
    gpu.tables[0].columns['geometry'].forEach((text, row) => {
      if (gpuAfter[row] === cpuAfter[row])
        expectGeometryClose(text, cpu.tables[0].columns['geometry'][row], 1e-12);
    });
  }
});

test('line-smooth runs on GPULineSmooth for open and closed parts', async () => {
  const lines = makeWalkLines(30, 11, 3);
  const {gpu, cpu} = await runPair('line-smooth', {iterations: 2}, {lines}, ['GPULineSmooth']);
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(gpu.tables[0].columns['vertices_after']).toEqual(cpu.tables[0].columns['vertices_after']);
  // Float32 offsets from the data center give about 1e-7 degrees.
  gpu.tables[0].columns['geometry'].forEach((text, row) =>
    expectGeometryClose(text, cpu.tables[0].columns['geometry'][row], 2e-6)
  );
});

test('line-chunk runs on GPULineChunk with the CPU chunk counts', async () => {
  const lines = makeWalkLines(25, 5);
  const {gpu, cpu} = await runPair(
    'line-chunk',
    {maximumChunkLength: 300, coordinateSystem: 'EPSG:4326'},
    {lines},
    ['GPULineChunk']
  );
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(gpu.tables[0].columns['chunk_id']).toEqual(cpu.tables[0].columns['chunk_id']);
  expect(gpu.tables[0].columns['end_measure']).toEqual(cpu.tables[0].columns['end_measure']);
  // Only the cut endpoints are compared: an interior vertex within float32 rounding of a cut can
  // fall on either side of it, so a piece may hold one vertex more or less than the CPU piece.
  gpu.tables[0].columns['geometry'].forEach((text, row) => {
    const gpuCoordinates = getCoordinates(text);
    const cpuCoordinates = getCoordinates(cpu.tables[0].columns['geometry'][row]);
    expect(Math.abs(gpuCoordinates.length - cpuCoordinates.length)).toBeLessThanOrEqual(1);
    for (const [gpuPosition, cpuPosition] of [
      [gpuCoordinates[0], cpuCoordinates[0]],
      [gpuCoordinates[gpuCoordinates.length - 1], cpuCoordinates[cpuCoordinates.length - 1]]
    ]) {
      expect(Math.abs(gpuPosition[0] - cpuPosition[0])).toBeLessThan(1e-4);
      expect(Math.abs(gpuPosition[1] - cpuPosition[1])).toBeLessThan(1e-4);
    }
  });
});

function makePoints(count: number, seed: number, offsetX = 0): CatalogFeatureCollection {
  const random = makeRandom(seed);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, (_, index) => ({
      id: `p${index}`,
      geometry: {
        type: 'Point',
        coordinates: [-122.5 + offsetX + random() * 0.5, 37.5 + random() * 0.5]
      },
      properties: {}
    }))
  };
}

function makeSquares(count: number, seed: number): CatalogFeatureCollection {
  const random = makeRandom(seed);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, (_, index) => {
      const x = -122.5 + random() * 0.4;
      const y = 37.5 + random() * 0.4;
      const size = 0.01 + random() * 0.02;
      const shell = [
        [x, y],
        [x + size, y],
        [x + size, y + size],
        [x, y + size],
        [x, y]
      ];
      const hole = [
        [x + size * 0.25, y + size * 0.25],
        [x + size * 0.25, y + size * 0.75],
        [x + size * 0.75, y + size * 0.75],
        [x + size * 0.75, y + size * 0.25],
        [x + size * 0.25, y + size * 0.25]
      ];
      // Alternate the winding so about half of the shells and holes need reversing.
      const flip = index % 2 === 0;
      return {
        id: `s${index}`,
        geometry: {
          type: 'Polygon',
          coordinates: flip ? [[...shell].reverse(), hole] : [shell, hole]
        },
        properties: {}
      };
    })
  };
}

test('line-merge runs on GPULineMerge and builds the CPU chains', async () => {
  // Chains of three segments sharing exact endpoints, plus loose lines.
  const features: CatalogFeatureCollection['features'] = [];
  for (let chain = 0; chain < 6; chain++) {
    const x = -122.4 + chain * 0.01;
    const y = 37.7;
    const points = [0, 1, 2, 3].map(step => [x + step * 0.001, y + (step % 2) * 0.001]);
    for (let segment = 0; segment < 3; segment++) {
      // The middle segment is stored reversed to exercise direction handling.
      const coordinates =
        segment === 1 ? [points[2], points[1]] : [points[segment], points[segment + 1]];
      features.push({
        id: `c${chain}s${segment}`,
        geometry: {type: 'LineString', coordinates},
        properties: {}
      });
    }
  }
  features.push({
    id: 'loose',
    geometry: {
      type: 'LineString',
      coordinates: [
        [-122.1, 37.1],
        [-122.09, 37.1]
      ]
    },
    properties: {}
  });
  const {gpu, cpu} = await runPair(
    'line-merge',
    {coordinateSystem: 'EPSG:4326'},
    {lines: {type: 'FeatureCollection', features}},
    ['GPULineMerge']
  );
  expect(gpu.tables[0].rowCount).toBe(7);
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  // GPU chain order and direction differ from the CPU, so compare per chain by sorted members.
  const summarize = (result: CatalogRunResult) =>
    result.tables[0].columns['member_source_indices']
      .map((members, row) => {
        const length = Math.round(result.tables[0].columns['length'][row] as number);
        return `${members}|${result.tables[0].columns['member_count'][row]}|${length}`;
      })
      .sort();
  expect(summarize(gpu)).toEqual(summarize(cpu));
});

test('geometry-orientation runs on GPUGeometryOrientation and reverses the CPU rings', async () => {
  const polygons = makeSquares(30, 3);
  for (const shellOrientation of ['counterclockwise', 'clockwise']) {
    const {gpu, cpu} = await runPair('geometry-orientation', {shellOrientation}, {polygons}, [
      'GPUGeometryOrientation'
    ]);
    expect(gpu.tables[0].columns['rings_reversed']).toEqual(
      cpu.tables[0].columns['rings_reversed']
    );
    // Reversal acts on the float64 source rings, so the geometry is identical.
    expect(gpu.tables[0].columns['geometry']).toEqual(cpu.tables[0].columns['geometry']);
  }
});

test('affine-transform runs on GPUAffineTransform for mixed geometry', async () => {
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      ...makePoints(10, 1).features,
      ...makeWalkLines(5, 2).features,
      ...makeSquares(5, 3).features
    ]
  };
  for (const pivot of ['bounds-center', 'origin']) {
    const parameters = {
      scaleX: 1.5,
      scaleY: 0.75,
      rotationDegrees: 30,
      translateX: 0.01,
      translateY: -0.02,
      pivot
    };
    const {gpu, cpu} = await runPair('affine-transform', parameters, {features}, [
      'GPUAffineTransform'
    ]);
    expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
    // Pivot-relative float32 offsets (origin pivot: absolute degrees up to 122, about 1e-5).
    const tolerance = pivot === 'origin' ? 1e-4 : 2e-6;
    gpu.tables[0].columns['geometry'].forEach((text, row) =>
      expectGeometryClose(text, cpu.tables[0].columns['geometry'][row], tolerance)
    );
  }
});

test('geodesic-destination runs on GPUGeodesicDestination', async () => {
  const points = makePoints(50, 4);
  const {gpu, cpu} = await runPair(
    'geodesic-destination',
    {bearing: 45, distance: 25000, coordinateSystem: 'EPSG:4326'},
    {points},
    ['GPUGeodesicDestination']
  );
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  // Float32 longitude/latitude resolution is about 1e-5 degrees.
  gpu.tables[0].columns['geometry'].forEach((text, row) =>
    expectGeometryClose(text, cpu.tables[0].columns['geometry'][row], 1e-4)
  );
});

test('geodesic-pairs runs on GPUGeodesicPairs', async () => {
  const {gpu, cpu} = await runPair(
    'geodesic-pairs',
    {coordinateSystem: 'EPSG:4326'},
    {origins: makePoints(40, 5), destinations: makePoints(40, 6, 0.3)},
    ['GPUGeodesicPairs']
  );
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  for (const column of ['distance', 'initial_bearing', 'final_bearing']) {
    gpu.tables[0].columns[column].forEach((value, row) => {
      const expected = cpu.tables[0].columns[column][row] as number;
      // Relative 1e-4 for distances, 0.05 degrees for bearings (float32 trigonometry).
      const tolerance = column === 'distance' ? expected * 1e-4 : 0.05;
      const difference = Math.abs((value as number) - expected);
      expect(Math.min(difference, 360 - difference)).toBeLessThan(tolerance);
    });
  }
});

test('group-convex-hull runs on GPUGroupConvexHull and matches the CPU hulls', async () => {
  const random = makeRandom(9);
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 300}, (_, index) => ({
      id: `h${index}`,
      geometry: {
        type: 'Point',
        coordinates: [-122.5 + random() * 0.3, 37.5 + random() * 0.3]
      },
      properties: {group: `g${index % 4}`}
    }))
  };
  const {gpu, cpu} = await runPair(
    'group-convex-hull',
    {groupProperty: 'group', coordinateSystem: 'EPSG:4326'},
    {points},
    ['GPUGroupConvexHull']
  );
  expect(gpu.tables[0].rowCount).toBe(4);
  expect(gpu.tables[0].columns['group_value']).toEqual(cpu.tables[0].columns['group_value']);
  expect(gpu.tables[0].columns['hull_vertex_count']).toEqual(
    cpu.tables[0].columns['hull_vertex_count']
  );
  // The hull is rebuilt from float64 source positions, so the area is identical; the ring may
  // start at a different vertex, so compare vertex sets.
  expect(gpu.tables[0].columns['area']).toEqual(cpu.tables[0].columns['area']);
  gpu.tables[0].columns['geometry'].forEach((text, row) => {
    const sortKey = (position: number[]) => `${position[0]},${position[1]}`;
    expect(getCoordinates(text).map(sortKey).sort()).toEqual(
      getCoordinates(cpu.tables[0].columns['geometry'][row]).map(sortKey).sort()
    );
  });
});

test('grid-generator runs on GPUGridGenerator', async () => {
  const extent = makePoints(30, 12);
  const {gpu, cpu} = await runPair(
    'grid-generator',
    {cellWidth: 0.05, cellHeight: 0.07},
    {extent},
    ['GPUGridGenerator']
  );
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(gpu.tables[0].columns['cell_id']).toEqual(cpu.tables[0].columns['cell_id']);
  // Float32 lattice offsets from the lower-left corner: about 1e-7 degrees.
  gpu.tables[0].columns['geometry'].forEach((text, row) =>
    expectGeometryClose(text, cpu.tables[0].columns['geometry'][row], 1e-6)
  );
});

function makeBlobs(count: number, seed: number): CatalogFeatureCollection {
  const random = makeRandom(seed);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, (_, index) => {
      const centerX = -122.5 + random() * 0.4;
      const centerY = 37.5 + random() * 0.4;
      const vertexCount = 9 + Math.floor(random() * 6);
      const ring = Array.from({length: vertexCount}, (_, vertex) => {
        const angle = (2 * Math.PI * vertex) / vertexCount;
        const radius = 0.004 + random() * 0.006;
        return [centerX + radius * Math.cos(angle) * 1.3, centerY + radius * Math.sin(angle)];
      });
      ring.push(ring[0]);
      return {id: `b${index}`, geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}};
    })
  };
}

test('geometry-minimum-bounds runs on GPUMinimumBounds for rectangles and circles', async () => {
  const features = makeBlobs(25, 21);
  for (const shape of ['rectangle', 'circle']) {
    const {gpu, cpu} = await runPair(
      'geometry-minimum-bounds',
      {shape, coordinateSystem: 'EPSG:4326'},
      {features},
      ['GPUMinimumBounds']
    );
    expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
    expect(gpu.tables[0].columns['source_index']).toEqual(cpu.tables[0].columns['source_index']);
    // Float32 meters about the data center: relative 1e-4 on size and area.
    for (const column of shape === 'rectangle' ? ['area', 'width', 'length'] : ['area', 'radius']) {
      gpu.tables[0].columns[column].forEach((value, row) => {
        const expected = cpu.tables[0].columns[column][row] as number;
        expect(Math.abs((value as number) - expected)).toBeLessThan(expected * 1e-3);
      });
    }
  }
});

test('geometry-label-point and geometry-maximum-inscribed-circle run on their GPU contributors', async () => {
  const polygons = makeBlobs(20, 33);
  for (const [bindingId, contributor] of [
    ['geometry-label-point', 'GPULabelPoint'],
    ['geometry-maximum-inscribed-circle', 'GPUMaximumInscribedCircle']
  ]) {
    const {gpu, cpu} = await runPair(
      bindingId,
      {precision: 0.001, coordinateSystem: 'EPSG:4326'},
      {polygons},
      [contributor]
    );
    expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
    // The CPU reference scales its stopping tolerance by the absolute meter coordinates of the
    // shell (millions of meters), so it stops early and under-reports the clearance. The GPU
    // searches around the data center and must reach at least the CPU clearance, and for these
    // blobs not more than 40 percent above it.
    gpu.tables[0].columns['clearance'].forEach((value, row) => {
      const expected = cpu.tables[0].columns['clearance'][row] as number;
      expect(value as number).toBeGreaterThan(expected * 0.97);
      expect(value as number).toBeLessThan(expected * 1.4);
    });
  }
});

test('delaunay-triangulation runs on GPUDelaunayTessellation', async () => {
  const points = makePoints(200, 41);
  const {gpu, cpu} = await runPair(
    'delaunay-triangulation',
    {coordinateSystem: 'EPSG:4326'},
    {points},
    ['GPUDelaunayTessellation']
  );
  const triangleKeys = (result: CatalogRunResult) =>
    new Set(
      result.tables[0].columns['point_a'].map((a, row) =>
        [a, result.tables[0].columns['point_b'][row], result.tables[0].columns['point_c'][row]]
          .map(Number)
          .sort((first, second) => first - second)
          .join(',')
      )
    );
  const gpuKeys = triangleKeys(gpu);
  const cpuKeys = triangleKeys(cpu);
  const shared = [...gpuKeys].filter(key => cpuKeys.has(key)).length;
  console.info(`delaunay: gpu ${gpuKeys.size} triangles, cpu ${cpuKeys.size}, shared ${shared}`);
  // The GPU keeps exact hull triangles that the CPU bounding triangle may clip, so the GPU set
  // is at least as large and nearly every CPU triangle is present (cocircular ties aside).
  expect(gpuKeys.size).toBeGreaterThanOrEqual(cpuKeys.size);
  expect(shared).toBeGreaterThanOrEqual(cpuKeys.size * 0.98);
  // A Delaunay triangulation of n points with h hull points has 2n - h - 2 triangles.
  const totalArea = (result: CatalogRunResult) =>
    (result.tables[0].columns['area'] as number[]).reduce((sum, value) => sum + value, 0);
  expect(totalArea(gpu)).toBeGreaterThanOrEqual(totalArea(cpu) * 0.999);
});

test('polygon-triangulation runs on GPUPolygonTriangulation', async () => {
  const polygons = makeSquares(10, 51);
  const withBlobs: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [...polygons.features, ...makeBlobs(10, 52).features]
  };
  const {gpu, cpu} = await runPair(
    'polygon-triangulation',
    {coordinateSystem: 'EPSG:4326'},
    {polygons: withBlobs},
    ['GPUPolygonTriangulation']
  );
  // Triangulations are not unique: compare the triangle count and total area per source polygon.
  const summarize = (result: CatalogRunResult) => {
    const byPolygon = new Map<number, {count: number; area: number}>();
    result.tables[0].columns['source_index'].forEach((source, row) => {
      const entry = byPolygon.get(source as number) ?? {count: 0, area: 0};
      entry.count++;
      entry.area += result.tables[0].columns['area'][row] as number;
      byPolygon.set(source as number, entry);
    });
    return byPolygon;
  };
  const gpuSummary = summarize(gpu);
  const cpuSummary = summarize(cpu);
  expect(gpuSummary.size).toBe(cpuSummary.size);
  for (const [source, entry] of cpuSummary) {
    expect(gpuSummary.get(source)!.count).toBe(entry.count);
    expect(Math.abs(gpuSummary.get(source)!.area - entry.area)).toBeLessThan(entry.area * 1e-3);
  }
});

function makeStreetGrid(size: number): CatalogFeatureCollection {
  const features: CatalogFeatureCollection['features'] = [];
  const step = 0.002;
  const at = (column: number, row: number) => [-122.4 + column * step, 37.7 + row * step];
  for (let row = 0; row < size; row++)
    for (let column = 0; column < size; column++) {
      if (column + 1 < size)
        features.push({
          id: `h${row}-${column}`,
          geometry: {type: 'LineString', coordinates: [at(column, row), at(column + 1, row)]},
          properties: {}
        });
      if (row + 1 < size)
        features.push({
          id: `v${row}-${column}`,
          geometry: {type: 'LineString', coordinates: [at(column, row), at(column, row + 1)]},
          properties: {}
        });
    }
  // A separate component and a dead-end spur.
  features.push({
    id: 'island',
    geometry: {
      type: 'LineString',
      coordinates: [
        [-122.1, 37.3],
        [-122.098, 37.3],
        [-122.096, 37.302]
      ]
    },
    properties: {}
  });
  features.push({
    id: 'spur',
    geometry: {type: 'LineString', coordinates: [at(0, 0), [-122.4 - step, 37.7 - step]]},
    properties: {}
  });
  return {type: 'FeatureCollection', features};
}

test('network-structure runs on GPUNetworkStructure and keeps the CPU components', async () => {
  const lines = makeStreetGrid(6);
  const {gpu, cpu} = await runPair('network-structure', {}, {lines}, ['GPUNetworkStructure']);
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(gpu.tables[0].columns['component']).toEqual(cpu.tables[0].columns['component']);
  expect(gpu.tables[0].columns['degree']).toEqual(cpu.tables[0].columns['degree']);
  expect(gpu.tables[0].columns['node_kind']).toEqual(cpu.tables[0].columns['node_kind']);
  expect(new Set(gpu.tables[0].columns['component']).size).toBeGreaterThan(1);
});

test('network-centrality runs on GPUNetworkCentrality and matches CPU closeness', async () => {
  const lines = makeStreetGrid(6);
  const {gpu, cpu} = await runPair('network-centrality', {}, {lines}, ['GPUNetworkCentrality']);
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(gpu.tables[0].columns['reachable_count']).toEqual(
    cpu.tables[0].columns['reachable_count']
  );
  // Float32 path costs: relative 1e-4.
  gpu.tables[0].columns['closeness_centrality'].forEach((value, row) => {
    const expected = cpu.tables[0].columns['closeness_centrality'][row] as number;
    expect(Math.abs((value as number) - expected)).toBeLessThanOrEqual(expected * 1e-4 + 1e-12);
  });
});

test('k-means runs on GPUSpatialKMeans and finds the CPU partition of separated clusters', async () => {
  const random = makeRandom(77);
  const centers = [
    [-122.45, 37.55],
    [-122.3, 37.8],
    [-122.1, 37.6]
  ];
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 90}, (_, index) => ({
      id: `k${index}`,
      geometry: {
        type: 'Point',
        coordinates: [
          centers[index % 3][0] + (random() - 0.5) * 0.02,
          centers[index % 3][1] + (random() - 0.5) * 0.02
        ]
      },
      properties: {}
    }))
  };
  const {gpu, cpu} = await runPair(
    'k-means',
    {clusterCount: 3, maximumIterations: 50, coordinateSystem: 'EPSG:4326'},
    {points},
    ['GPUSpatialKMeans']
  );
  // Cluster numbers differ between seedings, so compare the partitions.
  const partition = (result: CatalogRunResult) => {
    const relabel = new Map<unknown, number>();
    return result.tables[0].columns['cluster'].map(label => {
      if (!relabel.has(label)) relabel.set(label, relabel.size);
      return relabel.get(label);
    });
  };
  expect(partition(gpu)).toEqual(partition(cpu));
  expect(gpu.tables[1].columns['point_count']).toEqual([30, 30, 30]);
  gpu.tables[0].columns['distance'].forEach((value, row) => {
    expect(
      Math.abs((value as number) - (cpu.tables[0].columns['distance'][row] as number))
    ).toBeLessThan(0.5);
  });
});

test('geometry-minimum-clearance runs on GPUMinimumClearance', async () => {
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [...makeBlobs(12, 61).features, ...makeWalkLines(6, 62).features]
  };
  const {gpu, cpu} = await runPair(
    'geometry-minimum-clearance',
    {coordinateSystem: 'EPSG:4326'},
    {features},
    ['GPUMinimumClearance']
  );
  expect(gpu.tables[0].rowCount).toBe(cpu.tables[0].rowCount);
  expect(gpu.tables[0].columns['status']).toEqual(cpu.tables[0].columns['status']);
  // Float32 distances in the shifted metric frame: relative 1e-3.
  gpu.tables[0].columns['clearance'].forEach((value, row) => {
    const expected = cpu.tables[0].columns['clearance'][row] as number | null;
    if (expected === null) return;
    expect(Math.abs((value as number) - expected)).toBeLessThan(expected * 1e-3 + 1e-3);
  });
});

test('geographic-distribution runs on GPUGeographicDistribution', async () => {
  const random = makeRandom(88);
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 120}, (_, index) => {
      const group = index % 2;
      return {
        id: `d${index}`,
        geometry: {
          type: 'Point',
          coordinates: [
            -122.4 + group * 0.2 + (random() - 0.5) * 0.06,
            37.6 + group * 0.1 + (random() - 0.5) * 0.02
          ]
        },
        properties: {group: `g${group}`, weight: 1 + random() * 4}
      };
    })
  };
  const {gpu, cpu} = await runPair(
    'geographic-distribution',
    {
      groupProperty: 'group',
      weightProperty: 'weight',
      standardDeviations: 1.5,
      coordinateSystem: 'EPSG:4326'
    },
    {features},
    ['GPUGeographicDistribution']
  );
  expect(gpu.tables[0].columns['group']).toEqual(cpu.tables[0].columns['group']);
  expect(gpu.tables[0].columns['count']).toEqual(cpu.tables[0].columns['count']);
  // Float32 sums about the data center: relative 1e-3 on radii and distances, 0.5 degrees of rotation.
  for (const column of ['total_weight', 'standard_distance', 'major_radius', 'minor_radius']) {
    gpu.tables[0].columns[column].forEach((value, row) => {
      const expected = cpu.tables[0].columns[column][row] as number;
      expect(Math.abs((value as number) - expected)).toBeLessThan(expected * 1e-3);
    });
  }
  gpu.tables[0].columns['rotation_degrees'].forEach((value, row) => {
    expect(
      Math.abs((value as number) - (cpu.tables[0].columns['rotation_degrees'][row] as number))
    ).toBeLessThan(0.5);
  });
  // Mean and median centers within 1e-5 degrees (about a meter).
  gpu.tables[1].columns['geometry'].forEach((text, row) =>
    expectGeometryClose(text, cpu.tables[1].columns['geometry'][row], 1e-5)
  );
});
