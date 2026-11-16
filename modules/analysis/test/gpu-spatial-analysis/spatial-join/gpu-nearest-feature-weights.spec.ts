// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {assertValidCSR} from '../spatial-weights/spatial-weights-oracle';
import {assertValidCrossWeights} from './nearest-feature-weights-oracle';
import {createRandom} from './spatial-join-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Point = [number, number];

/** CPU libpysal-style KNN: k nearest by (distance, row), self excluded, rows sorted by ID. */
function computeKnnOracle(
  queries: Point[],
  features: Point[],
  k: number,
  options: {selfJoin: boolean; maxDistance?: number}
) {
  const rows: {id: number; distance: number}[][] = queries.map((query, row) => {
    const candidates = features
      .map((feature, id) => ({
        id,
        distance: Math.hypot(query[0] - feature[0], query[1] - feature[1])
      }))
      .filter(
        entry =>
          !(options.selfJoin && entry.id === row) &&
          (options.maxDistance === undefined || entry.distance <= options.maxDistance)
      );
    candidates.sort((a, b) => a.distance - b.distance || a.id - b.id);
    return candidates.slice(0, k).sort((a, b) => a.id - b.id);
  });
  return rows;
}

async function runAdapter(
  queries: Point[],
  features: Point[],
  options: {
    k: number;
    selfJoin: boolean;
    weightType?: 'binary' | 'inverse-distance';
    maxDistance?: number;
  }
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const joinK = options.selfJoin ? options.k + 1 : options.k;
  const slots = queries.length * joinK;
  const ids = rig.output('uint32', slots);
  const distances = rig.output('float32', slots);
  const counts = rig.output('uint32', queries.length);
  const joinOverflow = rig.output('uint32', 1);
  const output = rig.weightsOutput(queries.length, queries.length * options.k, true);
  const overflow = rig.output('uint32', 1);
  const maxDistance =
    options.maxDistance === undefined
      ? undefined
      : rig.input(Float32Array.of(options.maxDistance), 'float32', 1);
  const points = (values: Point[]) =>
    rig.input(Float32Array.from(values.flat()), 'float32x2', values.length);
  rig.run(
    new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions: points(queries)},
      features: {kind: 'points', positions: points(features)},
      k: joinK,
      radius: maxDistance,
      output: {
        neighborIds: ids.view,
        neighborCounts: counts.view,
        neighborDistances: distances.view,
        status: {overflow: joinOverflow.view}
      }
    }),
    new GPUNearestFeatureWeights({
      neighborIds: ids.view,
      neighborCounts: counts.view,
      neighborDistances: distances.view,
      slotCapacity: joinK,
      k: options.k,
      excludeSelf: options.selfJoin,
      weightType: options.weightType,
      output: {...output.spatialWeights, status: {overflow: overflow.view}}
    })
  );
  const csr = await output.read();
  const flag = (await readUint32(overflow.buffer, 1))[0];
  rig.destroy();
  return {csr, overflow: flag};
}

function randomPoints(count: number, seed: number): Point[] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => [random() * 100, random() * 100] as Point);
}

function expectMatchesOracle(
  csr: NonNullable<Awaited<ReturnType<typeof runAdapter>>>['csr'],
  oracle: ReturnType<typeof computeKnnOracle>,
  inverse: boolean
): void {
  let slot = 0;
  for (const [row, expected] of oracle.entries()) {
    expect(csr.offsets[row + 1] - csr.offsets[row], `row ${row} length`).toBe(expected.length);
    for (const entry of expected) {
      expect(csr.neighbors[slot], `row ${row} neighbor`).toBe(entry.id);
      expect(csr.distances[slot]).toBeCloseTo(entry.distance, 3);
      const weight = inverse ? 1 / Math.max(entry.distance, 1e-6) : 1;
      expect(Math.abs(csr.weights[slot] - weight)).toBeLessThanOrEqual(1e-4 * weight + 1e-6);
      slot++;
    }
  }
  expect(csr.neighbors.length).toBe(slot);
}

it('GPUNearestFeatureWeights builds KNN weights for a self-join (binary and inverse distance)', async () => {
  const points = randomPoints(60, 11);
  for (const weightType of ['binary', 'inverse-distance'] as const) {
    const result = await runAdapter(points, points, {k: 4, selfJoin: true, weightType});
    if (!result) return;
    assertValidCSR(result.csr, points.length, 'knn');
    const oracle = computeKnnOracle(points, points, 4, {selfJoin: true});
    expect(result.csr.neighbors.length).toBe(points.length * 4);
    expect(result.overflow).toBe(0);
    expectMatchesOracle(result.csr, oracle, weightType === 'inverse-distance');
  }
});

it('GPUNearestFeatureWeights gives coincident points the default-floor weight 1e6', async () => {
  // Math review cells-joins-04: one inverse-distance rule for the library. With the default
  // distanceFloor of 1e-6 two coincident points weigh (1e-6)^-1 = 1e6; GPUNeighborSearch pins
  // the same number in its own spec.
  const points: Point[] = [
    [5, 5],
    [5, 5],
    [6, 5]
  ];
  const result = await runAdapter(points, points, {
    k: 2,
    selfJoin: true,
    weightType: 'inverse-distance'
  });
  if (!result) return;
  expect(Array.from(result.csr.neighbors)).toEqual([1, 2, 0, 2, 0, 1]);
  const weights = Array.from(result.csr.weights);
  for (const [slot, expected] of [1e6, 1, 1e6, 1, 1, 1].entries()) {
    expect(Math.abs(weights[slot] - expected), `slot ${slot}`).toBeLessThanOrEqual(1e-5 * expected);
  }
});

it('GPUNearestFeatureWeights shortens rows when fewer than k neighbors exist', async () => {
  const points = randomPoints(40, 5);
  const options = {k: 3, selfJoin: true, maxDistance: 18};
  const result = await runAdapter(points, points, options);
  if (!result) return;
  const oracle = computeKnnOracle(points, points, 3, options);
  // Non-trivial: both full and short rows occur.
  const lengths = oracle.map(row => row.length);
  expect(Math.min(...lengths)).toBeLessThan(3);
  expect(Math.max(...lengths)).toBe(3);
  assertValidCSR(result.csr, points.length, 'knn-short');
  expectMatchesOracle(result.csr, oracle, false);
});

it('GPUNearestFeatureWeights supports cross weights (queries to other features)', async () => {
  const queries = randomPoints(25, 21);
  const features = randomPoints(50, 22);
  const result = await runAdapter(queries, features, {k: 5, selfJoin: false});
  if (!result) return;
  const oracle = computeKnnOracle(queries, features, 5, {selfJoin: false});
  expectMatchesOracle(result.csr, oracle, false);
  assertValidCrossWeights(result.csr, queries.length, features.length, 'cross-knn');
});

it('cross weights retain matching numeric query and feature IDs', async () => {
  const queries: Point[] = [
    [0, 0],
    [2, 0]
  ];
  const features: Point[] = [
    [0, 0],
    [2, 0],
    [10, 0]
  ];
  const result = await runAdapter(queries, features, {k: 1, selfJoin: false});
  expect(result.csr.offsets).toEqual([0, 1, 2]);
  expect(result.csr.neighbors).toEqual([0, 1]);
  expect(result.csr.distances).toEqual([0, 0]);
  expect(result.csr.weights).toEqual([1, 1]);
  expect(result.overflow).toBe(0);
  assertValidCrossWeights(result.csr, queries.length, features.length, 'cross-numeric-ids');
});

/** libpysal 4.15.0 `KNN.from_array` and `Graph.build_knn` (k = 4): 24 random points, identical neighbor sets. */
const LIBPYSAL_POINTS: Point[] = [
  [62.51, 89.721],
  [77.569, 22.521],
  [30.017, 87.355],
  [0.527, 82.123],
  [79.707, 46.793],
  [30.303, 27.843],
  [25.487, 44.508],
  [50.455, 55.35],
  [99.55, 79.266],
  [62.218, 98.896],
  [21.531, 16.021],
  [61.254, 4.394],
  [3.568, 51.489],
  [46.621, 91.717],
  [62.923, 51.412],
  [49.687, 24.751],
  [1.179, 19.24],
  [69.203, 20.061],
  [36.954, 0.373],
  [83.005, 15.446],
  [26.76, 88.033],
  [50.979, 84.715],
  [63.972, 74.177],
  [9.15, 54.114]
];
const LIBPYSAL_NEIGHBORS: number[][] = [
  [9, 13, 21, 22],
  [4, 11, 17, 19],
  [3, 13, 20, 21],
  [2, 12, 20, 23],
  [1, 7, 14, 17],
  [6, 10, 15, 18],
  [5, 7, 12, 23],
  [6, 14, 21, 22],
  [0, 4, 9, 22],
  [0, 13, 21, 22],
  [5, 6, 16, 18],
  [1, 15, 17, 19],
  [3, 6, 16, 23],
  [0, 2, 9, 21],
  [4, 7, 15, 22],
  [5, 11, 17, 18],
  [5, 6, 10, 12],
  [1, 11, 15, 19],
  [5, 10, 11, 15],
  [1, 4, 11, 17],
  [2, 3, 13, 21],
  [0, 9, 13, 22],
  [0, 7, 14, 21],
  [3, 5, 6, 12]
];

it('GPUNearestFeatureWeights matches libpysal KNN and Graph.build_knn neighbor sets', async () => {
  const result = await runAdapter(LIBPYSAL_POINTS, LIBPYSAL_POINTS, {k: 4, selfJoin: true});
  if (!result) return;
  LIBPYSAL_NEIGHBORS.forEach((expected, row) => {
    const begin = result.csr.offsets[row];
    const end = result.csr.offsets[row + 1];
    expect(result.csr.neighbors.slice(begin, end), `row ${row}`).toEqual(expected);
  });
});

/**
 * libpysal 4.15.0 breaks distance ties arbitrarily (KDTree order, and differently between `KNN` and
 * `Graph.build_knn`); this adapter keeps the lowest feature rows. Rows without a tie at the k-th
 * distance must agree with libpysal, whose k=3 sets on a 4x3 grid are pinned here.
 */
it('GPUNearestFeatureWeights agrees with libpysal on untied rows of a grid and keeps lowest rows on ties', async () => {
  const grid: Point[] = [];
  for (let x = 0; x < 4; x++) for (let y = 0; y < 3; y++) grid.push([x, y]);
  const result = await runAdapter(grid, grid, {k: 3, selfJoin: true});
  if (!result) return;
  const oracle = computeKnnOracle(grid, grid, 3, {selfJoin: true});
  expectMatchesOracle(result.csr, oracle, false);
  // libpysal KNN k=3: rows with a unique third neighbor (0, 2, 9, 11 have the diagonal at distance sqrt 2).
  const libpysalUntied: Record<number, number[]> = {
    0: [1, 3, 4],
    2: [1, 4, 5],
    9: [6, 7, 10],
    11: [7, 8, 10]
  };
  for (const [row, expected] of Object.entries(libpysalUntied)) {
    const begin = result.csr.offsets[Number(row)];
    expect(result.csr.neighbors.slice(begin, begin + 3)).toEqual(expected);
  }
});

it('GPUNearestFeatureWeights validates its props', () => {
  const view = {length: 4, format: 'uint32'} as never;
  expect(
    () =>
      new GPUNearestFeatureWeights({
        neighborIds: view,
        neighborCounts: view,
        slotCapacity: 0,
        output: {} as never
      })
  ).toThrow(/slotCapacity/);
});

it.each([
  64, 65, 256, 1024
])('nearest weights storage sorting matches a CPU oracle at %i slots', async k => {
  // Reversed feature rows force the adapter to reorder the join's distance order into ID order.
  const features: Point[] = Array.from({length: 1100}, (_, row) => [1100 - row, row % 3]);
  const queries: Point[] = [
    [0, 0],
    [17, 0]
  ];
  for (const weightType of ['binary', 'inverse-distance'] as const) {
    const result = await runAdapter(queries, features, {k, selfJoin: false, weightType});
    const expected = computeKnnOracle(queries, features, k, {selfJoin: false});
    expectMatchesOracle(result.csr, expected, weightType === 'inverse-distance');
    expect(result.overflow).toBe(0);
    assertValidCrossWeights(result.csr, queries.length, features.length, 'storage-knn');
  }
});

it('weights batches exclude self and padding, keep k below slotCapacity, and sort before clipping a row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const slotCapacity = 80;
  const ids = new Uint32Array(3 * slotCapacity).fill(0xffffffff);
  ids.set([0, ...Array.from({length: 79}, (_, row) => 79 - row)]);
  ids.set([1, 7, 3, 5], slotCapacity);
  ids.set([2, 70, 9, 8, 7, 6], 2 * slotCapacity);
  const distances = Float32Array.from({length: ids.length}, (_, slot) => slot % slotCapacity);
  const output = rig.weightsOutput(3, 71, true);
  const overflow = rig.output('uint32', 1);
  try {
    rig.run(
      new GPUNearestFeatureWeights({
        slotCapacity,
        k: 65,
        excludeSelf: true,
        scratchChunkRowCount: 1,
        weightType: 'inverse-distance',
        neighborIds: rig.input(ids, 'uint32', ids.length),
        neighborCounts: rig.input(Uint32Array.of(80, 5, 6), 'uint32', 3),
        neighborDistances: rig.input(distances, 'float32', distances.length),
        output: {...output.spatialWeights, status: {overflow: overflow.view}}
      })
    );
    const actual = await output.read();
    expect(actual.offsets).toEqual([0, 65, 68, 71]);
    const expectedIds = [...Array.from({length: 65}, (_, row) => row + 15), 3, 5, 7, 6, 7, 8];
    const expectedDistances = [...Array.from({length: 65}, (_, row) => 65 - row), 2, 3, 1, 5, 4, 3];
    expect(actual.neighbors).toEqual(expectedIds);
    expect(actual.distances).toEqual(expectedDistances);
    actual.weights.forEach((weight, slot) =>
      expect(weight).toBeCloseTo(1 / expectedDistances[slot], 6)
    );
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(1);
  } finally {
    rig.destroy();
  }
});
