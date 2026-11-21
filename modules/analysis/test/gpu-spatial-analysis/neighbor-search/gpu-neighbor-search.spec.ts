// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {packGPUSpatialParameterValues} from '../../../src/analysis-core/contracts/parameter-schema';
import {
  GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA,
  type GPUNeighborSearchSettings
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {createNeighborSearchHarness, type NeighborSearchReadback} from './neighbor-search-harness';
import {
  computeNeighborSearchOracle,
  createSeededRandom,
  type NeighborSearchOracleInput
} from './neighbor-search-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const BOUNDS = [0, 0, 100, 100] as const;

function createRandomPoints(count: number, seed: number, extent = 100): Float32Array {
  const random = createSeededRandom(seed);
  const positions = new Float32Array(count * 2);
  for (let index = 0; index < positions.length; index++) {
    positions[index] = random() * extent;
  }
  return positions;
}

function isClose(actual: number, expected: number, absolute: number, relative: number): boolean {
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}

function expectMatchesOracle(
  result: NeighborSearchReadback,
  input: NeighborSearchOracleInput,
  label: string,
  capacity = Infinity
): void {
  const oracle = computeNeighborSearchOracle(input);
  const rows = oracle.counts.length;
  const total = oracle.offsets[rows];
  expect(result.neighborCounts, `${label} counts`).toEqual(oracle.counts);
  expect(result.totalNeighbors, `${label} total`).toBe(total);
  expect(result.overflow, `${label} overflow`).toBe(total > capacity ? 1 : 0);
  expect(result.offsets, `${label} offsets`).toEqual(
    oracle.offsets.map(o => Math.min(o, capacity))
  );
  if (total <= capacity) {
    expect(result.neighbors, `${label} neighbors`).toEqual(oracle.neighbors);
    for (let slot = 0; slot < total; slot++) {
      if (!isClose(result.distances[slot], oracle.distances[slot], 1e-6, 1e-6)) {
        throw new Error(
          `${label}: slot ${slot} distance ${result.distances[slot]} != ${oracle.distances[slot]}`
        );
      }
      if (!isClose(result.weights[slot], oracle.weights[slot], 1e-6, 2e-5)) {
        throw new Error(
          `${label}: slot ${slot} weight ${result.weights[slot]} != ${oracle.weights[slot]}`
        );
      }
    }
  }
}

it('GPUNeighborSearch kNN self join matches the brute-force oracle for k = 1, 4, 8, 32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createRandomPoints(1500, 7);
  for (const neighborCount of [1, 4, 8, 32]) {
    const parameters: GPUNeighborSearchSettings = {bounds: BOUNDS};
    const harness = createNeighborSearchHarness(device, {
      mode: 'knn',
      neighborCount,
      positions,
      capacity: 1500 * neighborCount,
      parameters,
      gridSize: [24, 24]
    });
    const result = await harness.run();
    expectMatchesOracle(
      result,
      {mode: 'knn', neighborCount, positions, parameters},
      `k=${neighborCount}`
    );
    harness.destroy();
  }
});

it('GPUNeighborSearch kNN resolves exact ties to the lowest ID and handles tiny and clustered sets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Integer lattice: many exactly equal distances.
  const lattice: number[] = [];
  for (let y = 0; y < 12; y++) {
    for (let x = 0; x < 12; x++) {
      lattice.push(x * 4, y * 4);
    }
  }
  // Coincident duplicates appended at the end.
  lattice.push(8, 8, 8, 8, 20, 20);
  const positions = new Float32Array(lattice);
  const rows = positions.length / 2;
  for (const gridSize of [
    [1, 1],
    [5, 3],
    [40, 40]
  ] as const) {
    for (const neighborCount of [3, 5, 9]) {
      const parameters: GPUNeighborSearchSettings = {bounds: [0, 0, 44, 44]};
      const harness = createNeighborSearchHarness(device, {
        mode: 'knn',
        neighborCount,
        positions,
        capacity: rows * neighborCount,
        parameters,
        gridSize
      });
      expectMatchesOracle(
        await harness.run(),
        {mode: 'knn', neighborCount, positions, parameters},
        `grid ${gridSize} k=${neighborCount}`
      );
      harness.destroy();
    }
  }
  // Fewer valid targets than k, and a bounded kNN radius.
  const few = new Float32Array([1, 1, 2, 2, 3, 3]);
  const parameters: GPUNeighborSearchSettings = {bounds: [0, 0, 4, 4]};
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    neighborCount: 8,
    positions: few,
    capacity: 24,
    parameters
  });
  expectMatchesOracle(
    await harness.run(),
    {mode: 'knn', neighborCount: 8, positions: few, parameters},
    'few'
  );
  const bounded = {bounds: [0, 0, 4, 4], maximumDistance: 1.5} as const;
  expectMatchesOracle(
    await harness.run(bounded),
    {mode: 'knn', neighborCount: 8, positions: few, parameters: bounded},
    'bounded'
  );
  harness.destroy();
});

it('GPUNeighborSearch kNN and radius cross joins with masks match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createRandomPoints(900, 11);
  const queryPositions = createRandomPoints(400, 12);
  const random = createSeededRandom(13);
  const mask = Uint32Array.from({length: 900}, () => (random() < 0.7 ? 1 : 0));
  const queryMask = Uint32Array.from({length: 400}, () => (random() < 0.8 ? 1 : 0));
  // A target outside the bounds is excluded; a cross-join query outside them is still searched.
  positions[0] = -5;
  queryPositions[3] = 150;
  const knnParameters: GPUNeighborSearchSettings = {
    bounds: BOUNDS,
    weightType: 'inverse-distance',
    power: 2,
    distanceFloor: 0.01
  };
  const knn = createNeighborSearchHarness(device, {
    mode: 'knn',
    neighborCount: 6,
    positions,
    queryPositions,
    mask,
    queryMask,
    capacity: 2400,
    parameters: knnParameters
  });
  expectMatchesOracle(
    await knn.run(),
    {
      mode: 'knn',
      neighborCount: 6,
      positions,
      queryPositions,
      mask,
      queryMask,
      parameters: knnParameters
    },
    'knn cross'
  );
  knn.destroy();

  const radiusParameters: GPUNeighborSearchSettings = {bounds: BOUNDS, searchRadius: 6};
  const radius = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    queryPositions,
    mask,
    queryMask,
    capacity: 20000,
    parameters: radiusParameters
  });
  expectMatchesOracle(
    await radius.run(),
    {mode: 'radius', positions, queryPositions, mask, queryMask, parameters: radiusParameters},
    'radius cross'
  );
  radius.destroy();
});

it('GPUNeighborSearch radius self join follows per-frame radius and weights without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createRandomPoints(2500, 21);
  const random = createSeededRandom(22);
  const mask = Uint32Array.from({length: 2500}, () => (random() < 0.9 ? 1 : 0));
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    mask,
    capacity: 200000,
    parameters: {bounds: BOUNDS, searchRadius: 3},
    gridSize: [64, 64]
  });
  const frames: GPUNeighborSearchSettings[] = [
    {bounds: BOUNDS, searchRadius: 3},
    {bounds: BOUNDS, searchRadius: 0.5},
    {bounds: BOUNDS, searchRadius: 8, rowStandardize: true},
    {
      bounds: BOUNDS,
      searchRadius: 5,
      weightType: 'inverse-distance',
      power: 1,
      distanceFloor: 0.001
    },
    {bounds: BOUNDS, searchRadius: 5, weightType: 'kernel', kernel: 'gaussian'},
    {
      bounds: BOUNDS,
      searchRadius: 5,
      weightType: 'kernel',
      kernel: 'triangular',
      rowStandardize: true
    },
    {bounds: BOUNDS, searchRadius: 5, weightType: 'kernel', kernel: 'epanechnikov'},
    {bounds: BOUNDS, searchRadius: 5, weightType: 'kernel', kernel: 'bisquare'},
    {bounds: BOUNDS, searchRadius: 5, weightType: 'kernel', kernel: 'uniform'},
    {bounds: [0, 0, 50, 50], searchRadius: 4},
    {bounds: BOUNDS, searchRadius: 250}
  ];
  for (const [frame, parameters] of frames.entries()) {
    expectMatchesOracle(
      await harness.run(parameters),
      {mode: 'radius', positions, mask, parameters},
      `frame ${frame}`,
      200000
    );
  }
  // A GPU-resident zero radius bypasses the host packer's positive-distance validation.
  const zeroRadius = packGPUSpatialParameterValues(GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA, {
    minimumX: BOUNDS[0],
    minimumY: BOUNDS[1],
    maximumX: BOUNDS[2],
    maximumY: BOUNDS[3],
    maximumDistance: 0
  });
  const empty = await harness.run(zeroRadius);
  expect(empty.neighborCounts).toEqual(Array(positions.length / 2).fill(0));
  expect(empty.offsets).toEqual(Array(positions.length / 2 + 1).fill(0));
  // New positions and mask between submissions.
  const moved = createRandomPoints(2500, 23);
  harness.writePositions(moved);
  harness.writeMask(new Uint32Array(2500).fill(1));
  const parameters = {bounds: BOUNDS, searchRadius: 4} as const;
  expectMatchesOracle(
    await harness.run(parameters),
    {mode: 'radius', positions: moved, parameters},
    'moved'
  );
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});

it('GPUNeighborSearch radius cell pruning preserves exact cell and circle boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The two-by-two lattice has 100-unit cells while the radius is one. These points exercise
  // targets exactly one unit away across horizontal and vertical cell boundaries, a diagonal cell
  // whose near corner is outside the circle, and far targets in every surrounding cell.
  const positions = new Float32Array([
    0, -1, -1, 0, 0, 0, -2, -1, -1, -2, 75, -1, -75, -1, -1, 75, -1, -75
  ]);
  const queryPositions = new Float32Array([-1, -1]);
  const parameters: GPUNeighborSearchSettings = {
    bounds: [-100, -100, 100, 100],
    searchRadius: 1
  };
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    queryPositions,
    capacity: positions.length / 2,
    parameters,
    gridSize: [2, 2]
  });
  expectMatchesOracle(
    await harness.run(),
    {mode: 'radius', positions, queryPositions, parameters},
    'coarse-grid boundaries'
  );
  harness.destroy();
});

it('GPUNeighborSearch kNN adaptive kernel weights and row standardization match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createRandomPoints(800, 31);
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    neighborCount: 10,
    positions,
    capacity: 8000,
    parameters: {bounds: BOUNDS}
  });
  for (const kernel of ['gaussian', 'triangular', 'epanechnikov', 'bisquare', 'uniform'] as const) {
    for (const rowStandardize of [false, true]) {
      const parameters: GPUNeighborSearchSettings = {
        bounds: BOUNDS,
        weightType: 'kernel',
        kernel,
        rowStandardize
      };
      expectMatchesOracle(
        await harness.run(parameters),
        {mode: 'knn', neighborCount: 10, positions, parameters},
        `${kernel} ${rowStandardize}`
      );
    }
  }
  // Coincident points: inverse distance without a floor writes 0, with a floor a finite weight.
  const coincident = new Float32Array([5, 5, 5, 5, 6, 5]);
  const small = createNeighborSearchHarness(device, {
    mode: 'knn',
    neighborCount: 2,
    positions: coincident,
    capacity: 6,
    parameters: {bounds: BOUNDS}
  });
  for (const distanceFloor of [0, 0.5]) {
    const parameters: GPUNeighborSearchSettings = {
      bounds: BOUNDS,
      weightType: 'inverse-distance',
      distanceFloor
    };
    const result = await small.run(parameters);
    expectMatchesOracle(
      result,
      {mode: 'knn', neighborCount: 2, positions: coincident, parameters},
      `floor ${distanceFloor}`
    );
    expect(result.weights[0]).toBe(distanceFloor === 0 ? 0 : 2);
  }
  // Without a distanceFloor the library default 1e-6 applies (math review cells-joins-04, the
  // same default as GPUNearestFeatureWeights): coincident points weigh (1e-6)^-power.
  for (const power of [1, 2]) {
    const parameters: GPUNeighborSearchSettings = {
      bounds: BOUNDS,
      weightType: 'inverse-distance',
      power
    };
    const result = await small.run(parameters);
    const expected = 1e6 ** power;
    expect(result.neighbors.slice(0, 2), `default floor power ${power}`).toEqual([1, 2]);
    expect(Math.abs(result.weights[0] - expected), `default floor power ${power}`).toBeLessThan(
      1e-5 * expected
    );
    expect(result.weights[1]).toBeCloseTo(1, 5);
  }
  harness.destroy();
  small.destroy();
});

it('GPUNeighborSearch clamps offsets and flags overflow when the capacity is exceeded', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createRandomPoints(600, 41);
  const parameters: GPUNeighborSearchSettings = {bounds: BOUNDS, searchRadius: 10};
  const oracle = computeNeighborSearchOracle({mode: 'radius', positions, parameters});
  const capacity = Math.floor(oracle.offsets[600] / 3);
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    capacity,
    parameters
  });
  const result = await harness.run();
  expectMatchesOracle(result, {mode: 'radius', positions, parameters}, 'radius overflow', capacity);
  // Complete rows before the capacity are exact.
  for (let row = 0; row < 600 && oracle.offsets[row + 1] <= capacity; row++) {
    const begin = oracle.offsets[row];
    const end = oracle.offsets[row + 1];
    expect(result.neighbors.slice(begin, end)).toEqual(oracle.neighbors.slice(begin, end));
  }
  // Every truncated row is still strictly ascending.
  for (let row = 0; row < 600; row++) {
    for (let slot = result.offsets[row] + 1; slot < result.offsets[row + 1]; slot++) {
      expect(result.neighbors[slot]).toBeGreaterThan(result.neighbors[slot - 1]);
    }
  }
  harness.destroy();

  const knn = createNeighborSearchHarness(device, {
    mode: 'knn',
    neighborCount: 4,
    positions,
    capacity: 1000,
    parameters: {bounds: BOUNDS}
  });
  expectMatchesOracle(
    await knn.run(),
    {mode: 'knn', neighborCount: 4, positions, parameters: {bounds: BOUNDS}},
    'knn overflow',
    1000
  );
  knn.destroy();
});

it('GPUNeighborSearch is bitwise reproducible across submissions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createRandomPoints(3000, 51);
  const parameters: GPUNeighborSearchSettings = {
    bounds: BOUNDS,
    searchRadius: 4,
    weightType: 'kernel',
    kernel: 'gaussian',
    rowStandardize: true
  };
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    capacity: 200000,
    parameters,
    gridSize: [8, 8]
  });
  const first = await harness.run();
  const second = await harness.run();
  expect(second.neighbors).toEqual(first.neighbors);
  expect(second.weightBits).toEqual(first.weightBits);
  harness.destroy();
});

it('GPUNeighborSearch kNN cell pruning matches the oracle on mixed density with border points', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A dense cluster next to a sparse background: queries hold k candidates after a ring or two
  // and then skip far cells (rows, row-clipped column ranges, end cells), while background
  // queries must keep expanding. Points on exact multiples of the cell width probe the borders.
  const random = createSeededRandom(41);
  const values: number[] = [];
  for (let row = 0; row < 500; row++) {
    values.push(40 + random() * 5, 40 + random() * 5);
  }
  for (let row = 0; row < 700; row++) {
    values.push(random() * 100, random() * 100);
  }
  for (let row = 0; row < 120; row++) {
    values.push(Math.floor(random() * 21) * 5, Math.floor(random() * 21) * 5);
  }
  const positions = new Float32Array(values);
  const rows = positions.length / 2;
  for (const gridSize of [
    [20, 20],
    [64, 7],
    [100, 100]
  ] as const) {
    for (const neighborCount of [1, 16, 32]) {
      const parameters: GPUNeighborSearchSettings = {bounds: BOUNDS};
      const harness = createNeighborSearchHarness(device, {
        mode: 'knn',
        neighborCount,
        positions,
        capacity: rows * neighborCount,
        parameters,
        gridSize
      });
      expectMatchesOracle(
        await harness.run(),
        {mode: 'knn', neighborCount, positions, parameters},
        `mixed grid ${gridSize} k=${neighborCount}`
      );
      harness.destroy();
    }
  }
}, 120000);
