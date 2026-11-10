// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// k above 32 and the space-time axis of GPUNeighborSearch, plus the contributors now consuming it.
// References are brute force on the same float32 inputs, written here independently of the WGSL.

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/index';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH,
  type GPUNeighborSearchStructure
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {
  getGPULocalOutlierFactorParameterValues,
  GPULocalOutlierFactor,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/outlier-detection/index';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {runHierarchicalDensityClustering} from '../spatial-clustering/hierarchical-density-clustering-harness';
import {createSeededRandom} from './neighbor-search-oracle';

const f32 = Math.fround;
const EXTENT = 1000;
const BOUNDS: [number, number, number, number] = [0, 0, EXTENT, EXTENT];
const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

type SearchOptions = {
  mode: 'knn' | 'radius';
  positions: Float32Array;
  queryPositions?: Float32Array;
  times?: Float32Array;
  queryTimes?: Float32Array;
  k?: number;
  radius?: number;
  timeRadius?: number;
  timeBounds?: [number, number];
  timeBucketCount?: number;
  bounds?: [number, number, number, number];
  gridSize?: [number, number];
  spatialContext?: GPUSpatialContext;
  searchStructure?: GPUNeighborSearchStructure;
  capacity: number;
};

type SearchResult = {
  offsets: number[];
  neighbors: number[];
  distances: number[];
  overflow: number;
  invalidCount: number;
};

async function runSearch(device: Device, options: SearchOptions): Promise<SearchResult> {
  const queryRows = (options.queryPositions ?? options.positions).length / 2;
  const {capacity} = options;
  const graph = new GPUCommandGraph(device, {id: 'point-neighbor-grid-unification'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2'>(
    name: string,
    data: Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      data.length / (format === 'float32' ? 1 : 2)
    );
  };
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: options.times
      ? GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
      : GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    values: getGPUNeighborSearchParameterValues({
      bounds: options.bounds ?? BOUNDS,
      radius: options.radius,
      timeRadius: options.timeRadius,
      timeBounds: options.timeBounds
    })
  });
  const offsets = output('offsets', 'uint32', queryRows + 1);
  const neighbors = output('neighbors', 'uint32', capacity);
  const weights = output('weights', 'float32', capacity);
  const distances = output('distances', 'float32', capacity);
  const overflow = output('overflow', 'uint32', 1);
  const invalidCount = output('invalid', 'uint32', 1);
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      new GPUNeighborSearch({
        mode: options.mode,
        k: options.k,
        spatialContext: options.spatialContext,
        gridSize: options.gridSize ?? [24, 24],
        searchStructure: options.searchStructure,
        positions: input('positions', options.positions, 'float32x2'),
        queryPositions: options.queryPositions
          ? input('query-positions', options.queryPositions, 'float32x2')
          : undefined,
        times: options.times ? input('times', options.times, 'float32') : undefined,
        queryTimes: options.queryTimes
          ? input('query-times', options.queryTimes, 'float32')
          : undefined,
        temporalContext: options.times ? TEMPORAL_CONTEXT : undefined,
        timeBucketCount: options.timeBucketCount,
        parameters: parameters.importToGraph(graph),
        output: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view,
          distances: distances.view,
          status: {overflow: overflow.view, invalidCount: invalidCount.view}
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const offsetValues = [...(await readUint32(offsets.buffer, queryRows + 1))];
    const used = offsetValues[queryRows];
    return {
      offsets: offsetValues,
      neighbors: used ? [...(await readUint32(neighbors.buffer, used))] : [],
      distances: used ? [...(await readFloat32(distances.buffer, used))] : [],
      overflow: (await readUint32(overflow.buffer, 1))[0],
      invalidCount: (await readUint32(invalidCount.buffer, 1))[0]
    };
  } finally {
    compiled?.destroy();
    parameters.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

/** Planar brute force in the float32 arithmetic the kernels use, rows ascending by ID. */
function computeReference(options: SearchOptions): {offsets: number[]; neighbors: number[]} {
  const {positions, times, mode, k = 0} = options;
  const queryPositions = options.queryPositions ?? positions;
  const queryTimes = options.queryTimes ?? times;
  const selfJoin = !options.queryPositions;
  const [minimumTime, maximumTime] = options.timeBounds ?? [-Infinity, Infinity];
  const timeRadius = f32(options.timeRadius ?? Infinity);
  const radius = f32(options.radius ?? Infinity);
  const radiusSquared = f32(radius * radius);
  const isTimeUsable = (time: number) =>
    !times || (Number.isFinite(time) && time >= f32(minimumTime) && time <= f32(maximumTime));
  const offsets = [0];
  const neighbors: number[] = [];
  for (let query = 0; query < queryPositions.length / 2; query++) {
    const row: {id: number; distanceSquared: number}[] = [];
    const queryTime = queryTimes ? queryTimes[query] : 0;
    if (isTimeUsable(queryTime)) {
      for (let target = 0; target < positions.length / 2; target++) {
        if (selfJoin && target === query) continue;
        if (times) {
          const targetTime = times[target];
          if (
            !isTimeUsable(targetTime) ||
            !(f32(Math.abs(f32(targetTime - queryTime))) <= timeRadius)
          ) {
            continue;
          }
        }
        const deltaX = f32(positions[target * 2] - queryPositions[query * 2]);
        const deltaY = f32(positions[target * 2 + 1] - queryPositions[query * 2 + 1]);
        const distanceSquared = f32(f32(deltaX * deltaX) + f32(deltaY * deltaY));
        if (Number.isFinite(radius) ? distanceSquared <= radiusSquared : true) {
          row.push({id: target, distanceSquared});
        }
      }
    }
    const selected =
      mode === 'knn'
        ? row.sort((a, b) => a.distanceSquared - b.distanceSquared || a.id - b.id).slice(0, k)
        : row;
    selected.sort((a, b) => a.id - b.id);
    neighbors.push(...selected.map(entry => entry.id));
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

function createPoints(seed: number, count: number): Float32Array {
  const random = createSeededRandom(seed);
  const points = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const dense = row % 4 !== 0;
    const x = dense ? 450 + random() * 120 : random() * EXTENT;
    const y = dense ? 300 + random() * 120 : random() * EXTENT;
    points[row * 2] = f32(x);
    points[row * 2 + 1] = f32(y);
  }
  return points;
}

function createTimes(seed: number, count: number): Float32Array {
  const random = createSeededRandom(seed);
  return Float32Array.from({length: count}, () => f32(random() * 100));
}

function expectSameRows(
  result: SearchResult,
  reference: {offsets: number[]; neighbors: number[]},
  label: string
): void {
  expect(result.overflow, `${label} overflow`).toBe(0);
  expect(result.offsets, `${label} offsets`).toEqual(reference.offsets);
  expect(result.neighbors, `${label} neighbors`).toEqual(reference.neighbors);
}

describe('GPUNeighborSearch k above 32', () => {
  it('matches brute force at k = 48 and k = 256 on both structures', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    expect(GPU_NEIGHBOR_SEARCH_MAXIMUM_K).toBe(256);
    const positions = createPoints(11, 700);
    for (const k of [48, GPU_NEIGHBOR_SEARCH_MAXIMUM_K]) {
      const options: SearchOptions = {
        mode: 'knn',
        positions,
        k,
        capacity: (positions.length / 2) * k
      };
      const reference = computeReference(options);
      for (const searchStructure of ['grid', 'bvh'] as const) {
        const result = await runSearch(device, {...options, searchStructure});
        expectSameRows(result, reference, `k ${k} ${searchStructure}`);
      }
    }
  }, 120_000);

  it('finds every row at k = 256 when the table is smaller than k', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = createPoints(12, 100);
    const options: SearchOptions = {
      mode: 'knn',
      positions,
      k: GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
      capacity: 100 * GPU_NEIGHBOR_SEARCH_MAXIMUM_K
    };
    const result = await runSearch(device, options);
    expect(result.offsets[100]).toBe(100 * 99);
    expectSameRows(result, computeReference(options), 'short table');
  }, 60_000);
});

/** Time spans and bucket counts for which the float32 bucket coordinate of the span is above 1. */
function findRoundingSpans(): {latest: number; timeBucketCount: number}[] {
  const cases: {latest: number; timeBucketCount: number}[] = [];
  for (const timeBucketCount of [7, 11, 13, 14, 28, 31, 365]) {
    let found = 0;
    for (let step = 1; step < 20000 && found < 2; step++) {
      const latest = Math.fround(1 + step * 0.0137);
      const width = Math.fround(latest / timeBucketCount);
      if (Math.fround(latest / Math.fround(width * timeBucketCount)) > 1) {
        cases.push({latest, timeBucketCount});
        found += 1;
      }
    }
  }
  return cases;
}

describe('GPUNeighborSearch time axis', () => {
  const ROWS = 1500;
  const TIME_BOUNDS: [number, number] = [0, 100];

  function createTimedScene(): {positions: Float32Array; times: Float32Array} {
    const positions = createPoints(21, ROWS);
    const times = createTimes(22, ROWS);
    return {positions, times};
  }

  for (const timeBucketCount of [undefined, 16]) {
    const label = `timeBucketCount ${timeBucketCount ?? 'default'}`;
    it(`radius cylinder matches brute force, ${label}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {positions, times} = createTimedScene();
      const options: SearchOptions = {
        mode: 'radius',
        positions,
        times,
        timeRadius: 6,
        timeBounds: TIME_BOUNDS,
        timeBucketCount,
        radius: 30,
        capacity: ROWS * ROWS
      };
      const reference = computeReference(options);
      // The cylinder must be strictly smaller than the spatial band.
      const spatialOnly = computeReference({...options, times: undefined, timeRadius: undefined});
      expect(reference.neighbors.length).toBeGreaterThan(0);
      expect(reference.neighbors.length).toBeLessThan(spatialOnly.neighbors.length);
      expectSameRows(await runSearch(device, options), reference, `radius ${label}`);
    }, 60_000);

    it(`kNN inside the window matches brute force, ${label}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {positions, times} = createTimedScene();
      const options: SearchOptions = {
        mode: 'knn',
        positions,
        times,
        timeRadius: 6,
        timeBounds: TIME_BOUNDS,
        timeBucketCount,
        k: 40,
        capacity: ROWS * 40
      };
      expectSameRows(await runSearch(device, options), computeReference(options), `knn ${label}`);
    }, 60_000);

    it(`cross join carries query times, ${label}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {positions, times} = createTimedScene();
      const queryPositions = createPoints(23, 400);
      const queryTimes = createTimes(24, 400);
      const base = {
        positions,
        times,
        queryPositions,
        queryTimes,
        timeRadius: 4,
        timeBounds: TIME_BOUNDS,
        timeBucketCount
      };
      const radius: SearchOptions = {...base, mode: 'radius', radius: 25, capacity: 400 * ROWS};
      expectSameRows(
        await runSearch(device, radius),
        computeReference(radius),
        `cross radius ${label}`
      );
      const knn: SearchOptions = {...base, mode: 'knn', k: 9, capacity: 400 * 9};
      expectSameRows(await runSearch(device, knn), computeReference(knn), `cross knn ${label}`);
    }, 60_000);
  }

  it('treats non-finite and out-of-bounds times as invalid rows', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = createPoints(31, 300);
    const times = createTimes(32, 300);
    times[5] = Number.NaN;
    times[6] = Number.POSITIVE_INFINITY;
    times[7] = 150;
    times[8] = -1;
    const options: SearchOptions = {
      mode: 'radius',
      positions,
      times,
      timeRadius: 10,
      timeBounds: TIME_BOUNDS,
      timeBucketCount: 8,
      radius: 200,
      capacity: 300 * 300
    };
    const result = await runSearch(device, options);
    expectSameRows(result, computeReference(options), 'invalid times');
    expect(result.invalidCount).toBe(4);
    for (const row of [5, 6, 7, 8]) {
      expect(result.offsets[row + 1] - result.offsets[row], `row ${row} empty`).toBe(0);
      expect(result.neighbors, `row ${row} never a neighbor`).not.toContain(row);
    }
  }, 60_000);

  it('keeps the latest row when its float32 bucket coordinate rounds above one', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Spans whose float32 `latest / (width * buckets)` is above 1 for the bucket count beside them.
    const cases = findRoundingSpans();
    expect(cases.length).toBeGreaterThan(8);
    for (const {latest, timeBucketCount} of cases) {
      const options: SearchOptions = {
        mode: 'radius',
        positions: Float32Array.from([10, 10, 20, 10, 30, 10, 40, 10]),
        times: Float32Array.from([0, latest / 3, latest / 2, latest]),
        timeRadius: latest * 2,
        timeBounds: [0, latest],
        timeBucketCount,
        radius: 200,
        capacity: 16
      };
      const result = await runSearch(device, options);
      const label = `span ${latest}, ${timeBucketCount} buckets`;
      expectSameRows(result, computeReference(options), label);
      expect(result.invalidCount, label).toBe(0);
      expect(result.neighbors, label).toContain(3);
    }
  }, 60_000);

  it('a window that excludes everything leaves empty rows', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = createPoints(33, 100);
    const times = Float32Array.from({length: 100}, (_, row) => f32(row * 10));
    const options: SearchOptions = {
      mode: 'radius',
      positions,
      times,
      timeRadius: 1,
      timeBounds: [0, 1000],
      timeBucketCount: 4,
      radius: 2000,
      capacity: 100 * 100
    };
    const result = await runSearch(device, options);
    expect(result.offsets[100]).toBe(0);
  }, 60_000);

  it('great-circle search applies the window on the geodesic loops', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(41);
    const rows = 200;
    const positions = new Float32Array(rows * 2);
    for (let row = 0; row < rows; row++) {
      // Straddles the antimeridian on the periodic lattice.
      const longitude = row % 2 ? 179 - random() * 2 : -179 + random() * 2;
      positions[row * 2] = f32(longitude);
      positions[row * 2 + 1] = f32(10 + random() * 2);
    }
    const times = createTimes(42, rows);
    const spatialContext: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'kilometers'
    };
    const common = {
      positions,
      spatialContext,
      bounds: [-180, -90, 180, 90] as [number, number, number, number]
    };
    // The reference is the spatial-only GPU result filtered by time in JavaScript, which shares no
    // code with the temporal predicate under test.
    const radius = 120;
    const timeRadius = 15;
    const spatial = await runSearch(device, {
      ...common,
      mode: 'radius',
      radius,
      capacity: rows * rows
    });
    const expectedOffsets = [0];
    const expectedNeighbors: number[] = [];
    for (let query = 0; query < rows; query++) {
      for (let slot = spatial.offsets[query]; slot < spatial.offsets[query + 1]; slot++) {
        const target = spatial.neighbors[slot];
        if (Math.abs(f32(times[target] - times[query])) <= timeRadius) {
          expectedNeighbors.push(target);
        }
      }
      expectedOffsets.push(expectedNeighbors.length);
    }
    expect(expectedNeighbors.length).toBeGreaterThan(0);
    expect(expectedNeighbors.length).toBeLessThan(spatial.neighbors.length);
    for (const timeBucketCount of [undefined, 12]) {
      const result = await runSearch(device, {
        ...common,
        mode: 'radius',
        radius,
        times,
        timeRadius,
        timeBounds: TIME_BOUNDS,
        timeBucketCount,
        capacity: rows * rows
      });
      expectSameRows(
        result,
        {offsets: expectedOffsets, neighbors: expectedNeighbors},
        'geodesic radius'
      );
    }
    // kNN from the same source: the full spatial ranking, cut to the k nearest inside the window.
    const k = 7;
    const ranking = await runSearch(device, {
      ...common,
      mode: 'knn',
      k: rows - 1,
      capacity: rows * rows
    });
    const knnOffsets = [0];
    const knnNeighbors: number[] = [];
    for (let query = 0; query < rows; query++) {
      const related: {id: number; distance: number}[] = [];
      for (let slot = ranking.offsets[query]; slot < ranking.offsets[query + 1]; slot++) {
        const target = ranking.neighbors[slot];
        if (Math.abs(f32(times[target] - times[query])) <= timeRadius) {
          related.push({id: target, distance: ranking.distances[slot]});
        }
      }
      related.sort((a, b) => a.distance - b.distance || a.id - b.id);
      knnNeighbors.push(
        ...related
          .slice(0, k)
          .map(entry => entry.id)
          .sort((a, b) => a - b)
      );
      knnOffsets.push(knnNeighbors.length);
    }
    for (const timeBucketCount of [undefined, 12]) {
      const result = await runSearch(device, {
        ...common,
        mode: 'knn',
        k,
        times,
        timeRadius,
        timeBounds: TIME_BOUNDS,
        timeBucketCount,
        capacity: rows * k
      });
      expectSameRows(result, {offsets: knnOffsets, neighbors: knnNeighbors}, 'geodesic knn');
    }
  }, 90_000);
});

/** Self-inclusive `samples`-th distance in float32, the core distance of HDBSCAN and OPTICS. */
function computeCoreDistance(positions: Float32Array, row: number, samples: number): number {
  const distances: number[] = [];
  for (let other = 0; other < positions.length / 2; other++) {
    const deltaX = f32(positions[other * 2] - positions[row * 2]);
    const deltaY = f32(positions[other * 2 + 1] - positions[row * 2 + 1]);
    distances.push(f32(Math.sqrt(f32(f32(deltaX * deltaX) + f32(deltaY * deltaY)))));
  }
  distances.sort((a, b) => a - b);
  return samples <= distances.length ? distances[samples - 1] : Infinity;
}

describe('contributors on GPUNeighborSearch', () => {
  it('LOF reads a k = 48 table and matches a float64 reference', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 400;
    const k = 48;
    const positions = createPoints(51, rows);
    const reference = computeReference({mode: 'knn', positions, k, capacity: rows * k});
    const distance = (first: number, second: number) =>
      Math.hypot(
        positions[first * 2] - positions[second * 2],
        positions[first * 2 + 1] - positions[second * 2 + 1]
      );
    const neighborsOf = (row: number) =>
      reference.neighbors.slice(reference.offsets[row], reference.offsets[row + 1]);
    const kDistance = (row: number) =>
      Math.max(...neighborsOf(row).map(neighbor => distance(row, neighbor)));
    const density = (row: number) => {
      const members = neighborsOf(row);
      const total = members.reduce(
        (sum, neighbor) => sum + Math.max(kDistance(neighbor), distance(row, neighbor)),
        0
      );
      return members.length / total;
    };
    const densities = Array.from({length: rows}, (_, row) => density(row));
    const expected = Array.from({length: rows}, (_, row) => {
      const members = neighborsOf(row);
      return (
        members.reduce((sum, neighbor) => sum + densities[neighbor], 0) /
        members.length /
        densities[row]
      );
    });

    const graph = new GPUCommandGraph(device, {id: 'lof-from-search'});
    const buffers: Buffer[] = [];
    const output = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const positionsBuffer = createInputBuffer(device, positions);
    buffers.push(positionsBuffer);
    const searchParameters = new GPUParameterBuffer(device, {
      id: 'search-parameters',
      format: 'float32',
      length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
      values: getGPUNeighborSearchParameterValues({bounds: BOUNDS})
    });
    const lofParameters = new GPUParameterBuffer(device, {
      id: 'lof-parameters',
      format: 'float32',
      length: GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH,
      values: getGPULocalOutlierFactorParameterValues({})
    });
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      const search = new GPUNeighborSearch({
        mode: 'knn',
        k,
        gridSize: [16, 16],
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
        parameters: searchParameters.importToGraph(graph),
        output: {
          offsets: output('offsets', 'uint32', rows + 1).view,
          neighbors: output('neighbors', 'uint32', rows * k).view,
          weights: output('weights', 'float32', rows * k).view,
          distances: output('distances', 'float32', rows * k).view,
          status: {overflow: output('overflow', 'uint32', 1).view}
        }
      });
      graph.add(search);
      const lof = output('lof', 'float32', rows);
      graph.add(
        new GPULocalOutlierFactor({
          neighbors: search.outputWeights,
          parameters: lofParameters.importToGraph(graph),
          output: {
            kDistance: output('k-distance', 'float32', rows).view,
            localReachabilityDensity: output('density', 'float32', rows).view,
            lof: lof.view
          }
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(lof.buffer, rows);
      for (let row = 0; row < rows; row++) {
        expect(Math.abs(actual[row] - expected[row]) / expected[row], `row ${row}`).toBeLessThan(
          1e-3
        );
      }
    } finally {
      compiled?.destroy();
      searchParameters.destroy();
      lofParameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }, 60_000);

  it('HDBSCAN core distances count the point itself, including minimumSamples 1', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = createPoints(61, 500);
    // A duplicate pair and a row the search must skip.
    positions[2] = positions[0];
    positions[3] = positions[1];
    positions[8] = Number.NaN;
    for (const minimumSamples of [1, 2, 7, 32]) {
      const result = await runHierarchicalDensityClustering(device, positions, {
        minimumClusterSize: 5,
        minimumSamples
      });
      const finite = Float32Array.from(positions).filter((_, word) => {
        const row = Math.floor(word / 2);
        return Number.isFinite(positions[row * 2]) && Number.isFinite(positions[row * 2 + 1]);
      });
      expect(result.coreDistances[4], 'non-finite row').toBe(Infinity);
      let finiteRow = 0;
      for (let row = 0; row < 500; row++) {
        if (row === 4) continue;
        const expected = computeCoreDistance(finite, finiteRow, minimumSamples);
        expect(
          Math.abs(result.coreDistances[row] - expected),
          `samples ${minimumSamples} row ${row}`
        ).toBeLessThanOrEqual(expected * 4e-7);
        finiteRow++;
      }
    }
  }, 120_000);

  it('OPTICS core distances honour maximumDistance beyond the old search limit', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 400;
    const minimumSamples = 200;
    const maximumDistance = 150;
    const positions = createPoints(71, rows);
    const graph = new GPUCommandGraph(device, {id: 'optics-from-search'});
    const buffers: Buffer[] = [];
    const output = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const positionsBuffer = createInputBuffer(device, positions);
    buffers.push(positionsBuffer);
    const core = output('core', 'float32', rows);
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      graph.add(
        new GPUOPTICS({
          positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
          minimumSamples,
          maximumDistance,
          extractionDistance: maximumDistance,
          output: {
            ordering: output('ordering', 'uint32', rows).view,
            coreDistances: core.view,
            reachability: output('reachability', 'float32', rows).view,
            predecessors: output('predecessors', 'uint32', rows).view,
            labels: output('labels', 'uint32', rows).view,
            status: output('status', 'uint32', 3).view
          }
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(core.buffer, rows);
      let finiteCount = 0;
      for (let row = 0; row < rows; row++) {
        const nearest = computeCoreDistance(positions, row, minimumSamples);
        const expected = nearest <= maximumDistance ? nearest : Infinity;
        if (Number.isFinite(expected)) finiteCount++;
        expect(
          expected === Infinity
            ? actual[row] === Infinity
            : Math.abs(actual[row] - expected) <= expected * 4e-7,
          `row ${row}: ${actual[row]} versus ${expected}`
        ).toBe(true);
      }
      // Both branches of the cap must be exercised.
      expect(finiteCount).toBeGreaterThan(0);
      expect(finiteCount).toBeLessThan(rows);
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }, 90_000);
});
