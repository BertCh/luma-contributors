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

// Independent binary64 central angles on a unit sphere; none of the radius thresholds hits a
// pair distance. Targets still obey the bounds, while queries may lie anywhere on the sphere.
function getCentralAngle(first: readonly number[], second: readonly number[]): number {
  const radians = Math.PI / 180;
  const latitudeDifference = (second[1] - first[1]) * radians;
  const longitudeDifference = (second[0] - first[0]) * radians;
  const haversine =
    Math.sin(latitudeDifference / 2) ** 2 +
    Math.cos(first[1] * radians) *
      Math.cos(second[1] * radians) *
      Math.sin(longitudeDifference / 2) ** 2;
  return 2 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, haversine))));
}

for (const scene of [
  {
    name: 'regional bounds',
    bounds: [-5, -5, 5, 5],
    positions: [0, 0, 1, 0, -1, 0, 4, 3, -4, -3, 5, 5, 20, 0],
    queries: [0, 1, 6, 0, -6, 0, 0, 6, 0, -6, 50, 0, 179, 0, -179, 0, 720, 1, 0, 90, 0, -90]
  },
  {
    name: 'regional bounds across the antimeridian',
    bounds: [170, -80, 190, 80],
    positions: [170, 0, 179, 0, 181, 0, 190, 20, 180, 79, 180, -79, 0, 0],
    queries: [-179, 0, 169, 0, 191, 0, 0, 0, 540, 0, -540, 0, -180, 89, 180, -89]
  },
  {
    name: 'periodic world bounds',
    bounds: [-180, -90, 180, 90],
    positions: [-179, 0, 179, 0, 0, 0, 80, 40, 0, 89, 0, -89, 181, 0],
    queries: [181, 0, -181, 0, 720, 0, -720, 0, 540, 85, -540, -85, 0, 90, 0, -90]
  }
] as const) {
  for (const mode of ['knn', 'radius'] as const) {
    it(`GPUNeighborSearch great-circle outside queries match central angles: ${scene.name}, ${mode}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const positions = Float32Array.from(scene.positions);
      // Non-finite and impossible latitudes stay invalid. A masked impossible query is not counted.
      const queries = Float32Array.from([...scene.queries, NaN, 0, 0, 91, 0, -91, Infinity, 0]);
      const queryRows = queries.length / 2;
      const queryMask = new Uint32Array(queryRows).fill(1);
      queryMask[queryRows - 2] = 0;
      for (const radius of mode === 'knn' ? [undefined, 0.35] : [0.35, 3.2]) {
        const expectedNeighbors: number[] = [];
        const expectedDistances: number[] = [];
        const expectedOffsets = [0];
        for (let query = 0; query < queryRows; query++) {
          const point = [queries[2 * query], queries[2 * query + 1]];
          const candidates: {row: number; distance: number}[] = [];
          if (queryMask[query] && point.every(Number.isFinite) && Math.abs(point[1]) <= 90) {
            for (let row = 0; row < positions.length / 2; row++) {
              const target = [positions[2 * row], positions[2 * row + 1]];
              if (
                target[0] < scene.bounds[0] ||
                target[0] > scene.bounds[2] ||
                target[1] < scene.bounds[1] ||
                target[1] > scene.bounds[3]
              )
                continue;
              const distance = getCentralAngle(point, target);
              if (radius === undefined || distance <= radius) candidates.push({row, distance});
            }
            if (mode === 'knn') {
              // Symmetric ties are exact for these integer-degree inputs. Resolve by row ID.
              candidates.sort((first, second) =>
                Math.abs(first.distance - second.distance) < 1e-12
                  ? first.row - second.row
                  : first.distance - second.distance
              );
              candidates.length = Math.min(candidates.length, 3);
            }
            candidates.sort((first, second) => first.row - second.row);
          }
          expectedNeighbors.push(...candidates.map(candidate => candidate.row));
          expectedDistances.push(...candidates.map(candidate => candidate.distance));
          expectedOffsets.push(expectedNeighbors.length);
        }
        const harness = createNeighborSearchHarness(device, {
          mode,
          k: mode === 'knn' ? 3 : undefined,
          positions,
          queryPositions: queries,
          queryMask,
          capacity: Math.max(1, expectedNeighbors.length),
          gridSize: [16, 12],
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            sphereRadius: 1
          },
          parameters: {bounds: scene.bounds, radius}
        });
        try {
          const result = await harness.run();
          expect(result.invalidCount).toBe(4); // One excluded target, three invalid queries.
          expect(result.overflow).toBe(0);
          expect(result.offsets).toEqual(expectedOffsets);
          expect(result.neighbors).toEqual(expectedNeighbors);
          expectedDistances.forEach((distance, slot) => {
            // The existing spherical search budget: f32 trigonometry, scaled to a unit sphere.
            expect(Math.abs(result.distances[slot] - distance)).toBeLessThanOrEqual(
              2e-7 + 2e-5 * distance
            );
          });
        } finally {
          harness.destroy();
        }
      }
    }, 120_000);
  }
}

// A full turn leaves the position unchanged. The asymmetric, narrow bounds make their midpoint
// nonzero, so subtracting it before wrapping a large longitude shifts the candidate cell range.
for (const mode of ['radius', 'knn'] as const) {
  it(`great-circle ${mode} retains coincident targets after many longitude turns`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const longitude of [0, 36000, -36000, 3600000, -3600000]) {
      const harness = createNeighborSearchHarness(device, {
        mode,
        k: mode === 'knn' ? 1 : undefined,
        positions: Float32Array.from([0, 0]),
        queryPositions: Float32Array.from([longitude, 0]),
        gridSize: [64, 64],
        capacity: 1,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: 1
        },
        parameters: {bounds: [-0.01, -0.01, 0.012, 0.01], radius: 5e-6}
      });
      try {
        const result = await harness.run();
        const label = `${mode} longitude ${longitude}`;
        expect(result.neighbors, label).toEqual([0]);
        expect(result.distances, label).toEqual([0]);
        expect(result.offsets, label).toEqual([0, 1]);
        expect(result.neighborCounts, label).toEqual([1]);
        expect(result.totalNeighbors, label).toBe(1);
        expect(result.invalidCount, label).toBe(0);
        expect(result.overflow, label).toBe(0);
      } finally {
        harness.destroy();
      }
    }
  }, 120_000);
}

// At latitude zero on a unit sphere, the shortest distance is the absolute longitude separation
// in radians. A nonzero target exposes cancellation that a target at exactly zero conceals.
for (const mode of ['radius', 'knn'] as const) {
  it(`great-circle ${mode} preserves noncoincident distances after many longitude turns`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = Float32Array.from([0.001, 0]);
    const expectedDistance = (positions[0] * Math.PI) / 180;
    for (const longitude of [0, 36000, -36000, 3600000, -3600000]) {
      const harness = createNeighborSearchHarness(device, {
        mode,
        k: mode === 'knn' ? 1 : undefined,
        positions,
        queryPositions: Float32Array.from([longitude, 0]),
        gridSize: [64, 64],
        capacity: 1,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: 1
        },
        parameters: {bounds: [-0.01, -0.01, 0.012, 0.01], radius: 2e-5}
      });
      try {
        const result = await harness.run();
        const label = `${mode} longitude ${longitude}`;
        expect(result.neighbors, label).toEqual([0]);
        expect(result.offsets, label).toEqual([0, 1]);
        expect(result.neighborCounts, label).toEqual([1]);
        expect(result.totalNeighbors, label).toBe(1);
        expect(result.invalidCount, label).toBe(0);
        expect(result.overflow, label).toBe(0);
        // 1e-10 radians covers float32 rounding here, far below the erroneous 1.66e-5 excess.
        expect(Math.abs(result.distances[0] - expectedDistance), label).toBeLessThan(1e-10);
      } finally {
        harness.destroy();
      }
    }
  }, 120_000);
}

it('great-circle unbounded kNN preserves the nearer target after many longitude turns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The farther target has the lower ID. Quantizing both distances to the same value would
  // incorrectly select it by the ID tie rule, so exact neighbor IDs settle the ranking defect.
  const positions = Float32Array.from([0.0011, 0, 0.001, 0]);
  const queryLongitudes = [0, 36000, -36000, 3600000, -3600000];
  const expectedDistance = (positions[2] * Math.PI) / 180;
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 1,
    positions,
    queryPositions: Float32Array.from(queryLongitudes.flatMap(longitude => [longitude, 0])),
    gridSize: [64, 64],
    capacity: queryLongitudes.length,
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1
    },
    parameters: {bounds: [-0.01, -0.01, 0.012, 0.01]}
  });
  try {
    const result = await harness.run();
    expect(result.neighbors).toEqual(queryLongitudes.map(() => 1));
    expect(result.offsets).toEqual([0, 1, 2, 3, 4, 5]);
    expect(result.neighborCounts).toEqual(queryLongitudes.map(() => 1));
    expect(result.totalNeighbors).toBe(queryLongitudes.length);
    expect(result.invalidCount).toBe(0);
    expect(result.overflow).toBe(0);
    result.distances.forEach((distance, query) => {
      expect(
        Math.abs(distance - expectedDistance),
        `longitude ${queryLongitudes[query]}`
      ).toBeLessThan(1e-10);
    });
  } finally {
    harness.destroy();
  }
});

it('great-circle outside queries retain clamped CSR offsets and report insufficient capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions: Float32Array.from([0, 0, 1, 0, -1, 0]),
    queryPositions: Float32Array.from([6, 0, -6, 0]),
    capacity: 2,
    gridSize: [4, 4],
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1
    },
    parameters: {bounds: [-5, -5, 5, 5], radius: 0.3}
  });
  try {
    const result = await harness.run();
    expect(result.invalidCount).toBe(0);
    expect(result.totalNeighbors).toBe(6);
    expect(result.overflow).toBe(1);
    expect(result.offsets).toEqual([0, 2, 2]);
    // A truncated radius row retains whichever candidates the cell scatter visited first.
    expect(result.neighbors).toEqual([...result.neighbors].sort((first, second) => first - second));
    expect(new Set(result.neighbors).size).toBe(2);
    result.neighbors.forEach((neighbor, slot) => {
      expect([0, 1, 2]).toContain(neighbor);
      const distance = getCentralAngle(
        [6, 0],
        [
          [0, 0],
          [1, 0],
          [-1, 0]
        ][neighbor]
      );
      expect(Math.abs(result.distances[slot] - distance)).toBeLessThan(2e-6);
    });
    expect(result.neighborCounts).toEqual([3, 3]);
  } finally {
    harness.destroy();
  }
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

for (const scene of [
  {longitude: Math.fround(360.001), bounds: [-5, -5, 5, 5] as const},
  {longitude: Math.fround(-360.001), bounds: [-5, -5, 5, 5] as const},
  {longitude: Math.fround(1e20), bounds: [-5, -5, 5, 5] as const},
  {longitude: Math.fround(-1e20), bounds: [-5, -5, 5, 5] as const},
  {longitude: Math.fround(3.4028232635611926e38), bounds: [-5, -5, 5, 5] as const},
  {longitude: Math.fround(-1e-6), bounds: [0, -1, 180, 1] as const},
  {longitude: Math.fround(1e-6), bounds: [-180, -1, 0, 1] as const},
  {longitude: Math.fround(-1e-6), bounds: [0, -1, 360, 1] as const, targetLongitude: 360}
]) {
  it(`great-circle search preserves stored longitude ${scene.longitude} with bounds ${scene.bounds}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Large f32 values are exact integers. BigInt modulo is independent of floating quotient /
    // product rounding; 1e20 stored as f32 has remainder 272 degrees, equivalent to -88 degrees.
    const remainder =
      Math.abs(scene.longitude) >= 2 ** 24
        ? Number(BigInt(scene.longitude) % 360n)
        : scene.longitude % 360;
    const longitude =
      remainder > 180 ? remainder - 360 : remainder < -180 ? remainder + 360 : remainder;
    const expectedDistance = (Math.abs(longitude) * Math.PI) / 180;
    for (const mode of ['knn', 'radius'] as const) {
      for (const radius of mode === 'knn'
        ? [undefined, expectedDistance * 0.5, expectedDistance * 1.5]
        : [expectedDistance * 0.5, expectedDistance * 1.5]) {
        const included = radius === undefined || radius > expectedDistance;
        const harness = createNeighborSearchHarness(device, {
          mode,
          k: mode === 'knn' ? 1 : undefined,
          positions: Float32Array.from([scene.targetLongitude ?? 0, 0]),
          queryPositions: Float32Array.from([scene.longitude, 0]),
          gridSize: [64, 64],
          capacity: 1,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            sphereRadius: 1
          },
          parameters: {bounds: scene.bounds, radius}
        });
        try {
          const result = await harness.run();
          expect(result.neighbors).toEqual(included ? [0] : []);
          expect(result.neighborCounts).toEqual([included ? 1 : 0]);
          expect(result.offsets).toEqual([0, included ? 1 : 0]);
          expect(result.invalidCount).toBe(0);
          expect(result.overflow).toBe(0);
          if (included) {
            expect(Math.abs(result.distances[0] - expectedDistance)).toBeLessThan(
              expectedDistance * 2e-6
            );
          }
        } finally {
          harness.destroy();
        }
      }
    }
  }, 120_000);
}
