// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A planar cross-join query outside the derived bounds must get the neighbors a brute-force scan
// gives; it used to be silently dropped (an empty row counted by `invalidCount`).

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {
  GPUNeighborSearchSettings,
  GPUNeighborSearchStructure
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {createNeighborSearchHarness} from './neighbor-search-harness';
import {computeNeighborSearchOracle, createSeededRandom} from './neighbor-search-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const BOUNDS = [0, 0, 100, 100] as const;

function createScene(): {positions: Float32Array; queryPositions: Float32Array} {
  const random = createSeededRandom(5);
  const positions = new Float32Array(1200);
  for (let index = 0; index < positions.length; index++) {
    positions[index] = Math.fround(random() * 100);
  }
  // Queries: inside, just outside each edge and corner, far outside, and exactly on the edge.
  const queryPositions = Float32Array.from([
    50, 50, -0.5, 50, 100.5, 50, 50, -0.5, 50, 100.5, -3, -3, 103, -3, -3, 103, 103, 103, 0, 0, 100,
    100, -1000, 40, 40, 5000, 1e6, 1e6, -1e6, 20, 30, -20, 60, 100.0001
  ]);
  // Ties: a query outside the bounds at equal distance from two targets.
  positions[0] = 10;
  positions[1] = 10;
  positions[2] = 10;
  positions[3] = 20;
  queryPositions[0] = -5;
  queryPositions[1] = 15;
  return {positions, queryPositions};
}

const STRUCTURES: GPUNeighborSearchStructure[] = ['grid', 'bvh'];

for (const searchStructure of STRUCTURES) {
  it(`GPUNeighborSearch cross-join queries outside the bounds match brute force, ${searchStructure}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, queryPositions} = createScene();
    const queryRows = queryPositions.length / 2;
    const cases: {
      label: string;
      mode: 'knn' | 'radius';
      k?: number;
      parameters: GPUNeighborSearchSettings;
    }[] = [
      {label: 'knn 1', mode: 'knn', k: 1, parameters: {bounds: BOUNDS}},
      {label: 'knn 7', mode: 'knn', k: 7, parameters: {bounds: BOUNDS}},
      {label: 'knn 7 within radius', mode: 'knn', k: 7, parameters: {bounds: BOUNDS, radius: 9}},
      {label: 'radius 6', mode: 'radius', parameters: {bounds: BOUNDS, radius: 6}},
      {label: 'radius 40', mode: 'radius', parameters: {bounds: BOUNDS, radius: 40}}
    ];
    for (const {label, mode, k, parameters} of cases) {
      const input = {mode, k, positions, queryPositions, parameters} as const;
      const oracle = computeNeighborSearchOracle(input);
      const harness = createNeighborSearchHarness(device, {
        mode,
        k,
        positions,
        queryPositions,
        capacity: Math.max(oracle.neighbors.length, 1),
        parameters,
        gridSize: [8, 8],
        searchStructure
      });
      const result = await harness.run();
      const name = `${searchStructure} ${label}`;
      // The scene must exercise the defect: some outside queries have neighbors in the reference.
      const outsideWithNeighbors = [1, 2, 3, 4, 5, 6, 7, 8].filter(
        query => oracle.counts[query] > 0
      ).length;
      expect(outsideWithNeighbors, `${name} reference has outside rows`).toBeGreaterThan(
        mode === 'knn' && parameters.radius === undefined ? 7 : 3
      );
      expect(result.overflow, `${name} overflow`).toBe(0);
      expect(result.offsets, `${name} offsets`).toEqual(oracle.offsets);
      expect(result.neighbors, `${name} neighbors`).toEqual(oracle.neighbors);
      expect(result.neighborCounts, `${name} counts`).toEqual(oracle.counts);
      for (let slot = 0; slot < oracle.distances.length; slot++) {
        expect(
          Math.abs(result.distances[slot] - oracle.distances[slot]),
          `${name} distance ${slot}`
        ).toBeLessThanOrEqual(1e-6 + 1e-6 * Math.abs(oracle.distances[slot]));
      }
      expect(result.invalidCount, `${name} invalid`).toBe(0);
      expect(result.offsets.length).toBe(queryRows + 1);
      harness.destroy();
    }
  }, 120_000);
}

it('GPUNeighborSearch still counts non-finite cross-join queries and out-of-bounds targets as invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 1,
    positions: new Float32Array([1, 1, 20, 20, 5, 5]),
    queryPositions: new Float32Array([2, 2, NaN, NaN, -30, -30, Infinity, 1]),
    capacity: 8,
    parameters: {bounds: [0, 0, 10, 10]}
  });
  const result = await harness.run();
  // Target (20, 20) is outside the bounds; the NaN and infinite queries are not finite.
  expect(result.invalidCount).toBe(3);
  expect(result.neighborCounts).toEqual([1, 0, 1, 0]);
  // The query at (-30, -30) is nearest to the target at (1, 1).
  expect(result.neighbors).toEqual([0, 0]);
  harness.destroy();
});

// A great-circle cross-join query outside `bounds` stays empty and counted: the longitude wrap of
// the great-circle lattice follows the bounds, so only the planar search was extended to
// outside queries. Documented on `GPUNeighborSearchProps.output` and in the class definition.
it('GPUNeighborSearch great-circle cross-join queries outside the bounds are empty and counted as invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 2,
    positions: new Float32Array([0, 0, 1, 0, 0, 1, 2, 2]),
    // Query 0 lies inside the bounds, query 1 at longitude 50 outside them, query 2 inside again.
    queryPositions: new Float32Array([0.2, 0.1, 50, 0, 1.5, 1.5]),
    capacity: 8,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    parameters: {bounds: [-5, -5, 5, 5]}
  });
  const result = await harness.run();
  expect(result.neighborCounts).toEqual([2, 0, 2]);
  expect(result.invalidCount).toBe(1);
  expect(result.offsets).toEqual([0, 2, 2, 4]);
  harness.destroy();
});

// Queries so far out that every squared distance overflows float32 are all tied at infinity, and
// the tie order is still the lowest target ID, as in a brute-force scan: the grid path used to
// narrow its column range with `inf - inf = NaN` and could skip the lower IDs in other columns.
for (const searchStructure of STRUCTURES) {
  it(`GPUNeighborSearch queries whose squared distances overflow tie to the lowest IDs, ${searchStructure}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(7);
    const positions = new Float32Array(400);
    for (let index = 0; index < positions.length; index++) {
      positions[index] = Math.floor(random() * 51);
    }
    const queryPositions = Float32Array.from([
      1e30, 1e30, -1e30, 25, 25, 3e38, -3e38, -3e38, 1e19, 1e19, 3.4e38, 0
    ]);
    const parameters: GPUNeighborSearchSettings = {bounds: [0, 0, 50, 50]};
    for (const k of [1, 3]) {
      const input = {mode: 'knn', k, positions, queryPositions, parameters} as const;
      const oracle = computeNeighborSearchOracle(input);
      const harness = createNeighborSearchHarness(device, {
        mode: 'knn',
        k,
        positions,
        queryPositions,
        capacity: (queryPositions.length / 2) * k,
        parameters,
        gridSize: [8, 8],
        searchStructure
      });
      const result = await harness.run();
      expect(result.neighbors, `${searchStructure} k ${k}`).toEqual(oracle.neighbors);
      expect(result.offsets).toEqual(oracle.offsets);
      expect(result.invalidCount).toBe(0);
      if (k === 1) {
        // Query 4 at (1e19, 1e19) still has finite squared distances (about 2e38); the others overflow.
        result.distances.forEach((distance, query) => {
          expect(Number.isFinite(distance), `${searchStructure} query ${query}`).toBe(query === 4);
        });
      }
      harness.destroy();
    }
  }, 120_000);
}
