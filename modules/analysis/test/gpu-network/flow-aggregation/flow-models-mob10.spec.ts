// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  getGPUKernelFlowSmoothingParameterValues,
  GPUKernelFlowSmoothing,
  GPURadiationFlowModel,
  GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES,
  type GPURadiationFlowModelNormalization
} from '../../../src/gpu-network/flow-aggregation/index';
import {createNullWebGPUDevice, readFloat32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-spec-utils';

/** Deterministic pseudo-random numbers in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function createZones(zoneCount: number, seed: number) {
  const random = createRandom(seed);
  const positions = Float32Array.from(Array.from({length: zoneCount * 2}, () => random() * 100));
  const populations = Float32Array.from(
    Array.from({length: zoneCount}, () => 50 + Math.floor(random() * 950))
  );
  const outflows = Float32Array.from(
    Array.from({length: zoneCount}, () => 10 + Math.floor(random() * 90))
  );
  return {positions, populations, outflows};
}

/**
 * Binary64 radiation reference from the paper's definition, by a different route than the kernel:
 * per origin, sort the other zones by distance and take a prefix sum of population, counting only
 * zones strictly nearer than the destination.
 */
function radiationReference(
  positions: Float32Array,
  populations: Float32Array,
  outflows: Float32Array,
  normalization: GPURadiationFlowModelNormalization
): number[] {
  const zoneCount = populations.length;
  const total = populations.reduce((sum, value) => sum + value, 0);
  const flows = new Array<number>(zoneCount * zoneCount).fill(0);
  for (let origin = 0; origin < zoneCount; origin++) {
    const others = Array.from({length: zoneCount}, (_, zone) => zone)
      .filter(zone => zone !== origin)
      .map(zone => ({
        zone,
        distance: Math.hypot(
          positions[2 * zone] - positions[2 * origin],
          positions[2 * zone + 1] - positions[2 * origin + 1]
        )
      }))
      .sort((left, right) => left.distance - right.distance);
    let cumulative = 0;
    others.forEach(({zone}, rank) => {
      // Distinct distances in these fixtures, so the cumulative population before `rank` is s_ij.
      const between = cumulative;
      cumulative += populations[zone];
      const m = populations[origin];
      const n = populations[zone];
      let flow = (outflows[origin] * m * n) / ((m + between) * (m + n + between));
      if (normalization === 'finite-size') {
        flow /= 1 - m / total;
      }
      flows[origin * zoneCount + zone] = flow;
      expect(rank).toBeGreaterThanOrEqual(0);
    });
  }
  return flows;
}

async function runRadiation(
  zones: ReturnType<typeof createZones>,
  normalization: GPURadiationFlowModelNormalization
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'mob10-radiation');
  const zoneCount = zones.populations.length;
  const positions = harness.input('positions', zones.positions, 'float32x2', zoneCount);
  const populations = harness.input('populations', zones.populations, 'float32', zoneCount);
  const outflows = harness.input('outflows', zones.outflows, 'float32', zoneCount);
  const flows = harness.output('flows', 'float32', zoneCount * zoneCount);
  harness.graph.add(
    new GPURadiationFlowModel({
      positions,
      populations,
      outflows,
      normalization,
      output: {flows: flows.view}
    })
  );
  return harness.run(() => readFloat32(flows.buffer, zoneCount * zoneCount));
}

describe('GPURadiationFlowModel (MOB-10)', () => {
  it('matches the hand-computed three-zone example', async () => {
    // Zones on a line at x = 0, 1, 3 with populations 100, 50, 200 and outflows 10, 20, 30.
    // Origin 0: s_01 = 0, s_02 = 50 (zone 1 is nearer than zone 2).
    // Origin 1: s_10 = 0, s_12 = 100. Origin 2: s_21 = 0, s_20 = 50.
    const actual = await runRadiation(
      {
        positions: Float32Array.from([0, 0, 1, 0, 3, 0]),
        populations: Float32Array.from([100, 50, 200]),
        outflows: Float32Array.from([10, 20, 30])
      },
      'none'
    );
    const expected = [
      0,
      (10 * 100 * 50) / (100 * 150),
      (10 * 100 * 200) / (150 * 350),
      (20 * 50 * 100) / (50 * 150),
      0,
      (20 * 50 * 200) / (150 * 350),
      (30 * 200 * 100) / (250 * 350),
      (30 * 200 * 50) / (200 * 250),
      0
    ];
    expected.forEach((value, index) => {
      expect(actual[index]).toBeCloseTo(value, 4);
    });
    // The closed-form row sum of the unnormalized model for origin 0 is T_0 (1 - m_0 / M).
    expect(actual[1] + actual[2]).toBeCloseTo(10 * (1 - 100 / 350), 4);
  });

  it('matches a binary64 sorted-prefix-sum reference on 40 random zones, both normalizations', async () => {
    const zones = createZones(40, 7);
    for (const normalization of ['none', 'finite-size'] as const) {
      const actual = await runRadiation(zones, normalization);
      const expected = radiationReference(
        zones.positions,
        zones.populations,
        zones.outflows,
        normalization
      );
      expected.forEach((value, index) => {
        expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(1e-4 * Math.max(1, value));
      });
    }
  });

  it('finite-size normalization makes every row sum to its outflow', async () => {
    const zones = createZones(40, 11);
    const actual = await runRadiation(zones, 'finite-size');
    for (let origin = 0; origin < 40; origin++) {
      let sum = 0;
      for (let destination = 0; destination < 40; destination++) {
        sum += actual[origin * 40 + destination];
      }
      expect(sum / zones.outflows[origin]).toBeCloseTo(1, 4);
    }
  });

  it('rejects more zones than the reference-tier maximum', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'mob10-limit'});
    const zoneCount = GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES + 1;
    expect(
      () =>
        new GPURadiationFlowModel({
          id: 'radiation',
          positions: createTransientView(graph, 'positions', 'float32x2', zoneCount),
          populations: createTransientView(graph, 'populations', 'float32', zoneCount),
          outflows: createTransientView(graph, 'outflows', 'float32', zoneCount),
          output: {flows: createTransientView(graph, 'flows', 'float32', zoneCount * zoneCount)}
        })
    ).toThrow('radiation zoneCount must be at most 1024');
  });
});

/** Binary64 smoothing reference by the direct double sum over neighbouring origins and destinations. */
function smoothingReference(
  positions: Float32Array,
  locationWeights: Float32Array,
  matrix: Float32Array,
  neighborhoodSize: number,
  bandwidthRatio: number
) {
  const zoneCount = locationWeights.length;
  const distance = (a: number, b: number) =>
    Math.hypot(positions[2 * a] - positions[2 * b], positions[2 * a + 1] - positions[2 * b + 1]);
  const radii = Array.from({length: zoneCount}, (_, zone) => {
    const sorted = Array.from({length: zoneCount}, (__, other) => distance(zone, other)).sort(
      (left, right) => left - right
    );
    let radius = sorted[sorted.length - 1];
    for (const candidate of sorted) {
      let weight = 0;
      for (let other = 0; other < zoneCount; other++) {
        if (distance(zone, other) <= candidate) weight += locationWeights[other];
      }
      if (weight >= neighborhoodSize) {
        radius = candidate;
        break;
      }
    }
    return radius;
  });
  const kernel = (zone: number, other: number) => {
    const d = distance(zone, other);
    if (d > radii[zone]) return 0;
    const sigma = radii[zone] * bandwidthRatio;
    return sigma > 0 ? Math.exp((-d * d) / (2 * sigma * sigma)) : 1;
  };
  const smoothed = new Array<number>(zoneCount * zoneCount).fill(0);
  for (let origin = 0; origin < zoneCount; origin++) {
    for (let destination = 0; destination < zoneCount; destination++) {
      let sum = 0;
      let weightSum = 0;
      for (let a = 0; a < zoneCount; a++) {
        for (let b = 0; b < zoneCount; b++) {
          const weight = kernel(origin, a) * kernel(destination, b);
          sum += weight * matrix[a * zoneCount + b];
          weightSum += weight;
        }
      }
      smoothed[origin * zoneCount + destination] = sum / weightSum;
    }
  }
  return {smoothed, radii};
}

async function runSmoothing(options: {
  positions: Float32Array;
  locationWeights: Float32Array;
  neighborhoodSize: number;
  bandwidthRatio: number;
  matrix?: Float32Array;
  pairs?: {origins: number[]; destinations: number[]; weights: number[]; count: number};
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'mob10-smoothing');
  const zoneCount = options.locationWeights.length;
  const positions = harness.input('positions', options.positions, 'float32x2', zoneCount);
  const locationWeights = harness.input(
    'location-weights',
    options.locationWeights,
    'float32',
    zoneCount
  );
  const parameters = harness.input(
    'parameters',
    getGPUKernelFlowSmoothingParameterValues({
      neighborhoodSize: options.neighborhoodSize,
      bandwidthRatio: options.bandwidthRatio
    }),
    'float32',
    4
  );
  const smoothed = harness.output('smoothed', 'float32', zoneCount * zoneCount);
  const radii = harness.output('radii', 'float32', zoneCount);
  const pairs = options.pairs;
  harness.graph.add(
    new GPUKernelFlowSmoothing({
      positions,
      locationWeights,
      parameters,
      flows: pairs
        ? {
            kind: 'pairs',
            originZoneIds: harness.input(
              'origins',
              Uint32Array.from(pairs.origins),
              'uint32',
              pairs.origins.length
            ),
            destinationZoneIds: harness.input(
              'destinations',
              Uint32Array.from(pairs.destinations),
              'uint32',
              pairs.destinations.length
            ),
            weights: harness.input(
              'weights',
              Float32Array.from(pairs.weights),
              'float32',
              pairs.weights.length
            ),
            validPairCount: harness.input('count', Uint32Array.from([pairs.count]), 'uint32', 1)
          }
        : {
            kind: 'matrix',
            matrix: harness.input('matrix', options.matrix!, 'float32', zoneCount * zoneCount)
          },
      output: {smoothed: smoothed.view, radii: radii.view}
    })
  );
  return harness.run(async () => ({
    smoothed: await readFloat32(smoothed.buffer, zoneCount * zoneCount),
    radii: await readFloat32(radii.buffer, zoneCount)
  }));
}

describe('GPUKernelFlowSmoothing (MOB-10)', () => {
  const zoneCount = 12;
  const zones = createZones(zoneCount, 3);
  const random = createRandom(5);
  const matrix = Float32Array.from(
    Array.from({length: zoneCount * zoneCount}, () =>
      random() < 0.4 ? Math.floor(random() * 20) : 0
    )
  );
  const locationWeights = Float32Array.from({length: zoneCount}, () => 1);

  it('matches a binary64 reference that sums the neighbouring origin and destination pairs directly', async () => {
    for (const [size, ratio] of [
      [3, 0.5],
      [5.5, 0.35]
    ]) {
      const actual = await runSmoothing({
        positions: zones.positions,
        locationWeights,
        matrix,
        neighborhoodSize: size,
        bandwidthRatio: ratio
      });
      const expected = smoothingReference(zones.positions, locationWeights, matrix, size, ratio);
      expected.radii.forEach((value, index) => {
        expect(actual.radii[index]).toBeCloseTo(value, 3);
      });
      expected.smoothed.forEach((value, index) => {
        expect(Math.abs(actual.smoothed[index] - value)).toBeLessThanOrEqual(
          2e-4 * Math.max(1, value)
        );
      });
      // Smoothing must actually change the matrix.
      expect(actual.smoothed.some((value, index) => Math.abs(value - matrix[index]) > 0.1)).toBe(
        true
      );
    }
  });

  it('keeps the input when the neighbourhood is the zone itself', async () => {
    const actual = await runSmoothing({
      positions: zones.positions,
      locationWeights,
      matrix,
      neighborhoodSize: 0,
      bandwidthRatio: 0.5
    });
    matrix.forEach((value, index) => {
      expect(actual.smoothed[index]).toBeCloseTo(value, 5);
    });
  });

  it('tends to the global mean when the kernel is very wide', async () => {
    const actual = await runSmoothing({
      positions: zones.positions,
      locationWeights,
      matrix,
      neighborhoodSize: 1000,
      bandwidthRatio: 1e5
    });
    const mean = matrix.reduce((sum, value) => sum + value, 0) / matrix.length;
    for (const value of actual.smoothed) {
      expect(value).toBeCloseTo(mean, 3);
    }
  });

  it('is a normalised kernel mean: three zones by hand, and the total flow is not conserved', async () => {
    // Zones at x = 0, 1 and 10 with unit weights and a neighbourhood size of 2: zones 0 and 1 are
    // each other's neighbour (h = 1), and zone 2 reaches zone 1 only (h = 9). With
    // bandwidthRatio 0.5 every neighbour weight is exp(-h^2 / (2 (h / 2)^2)) = exp(-2) = k, so
    // the row-normalised weights are W_0 = [1, k, 0], W_1 = [k, 1, 0], W_2 = [0, k, 1], each
    // divided by 1 + k, and S = W M W^T.
    const k = Math.exp(-2);
    const run = (matrix: number[]) =>
      runSmoothing({
        positions: Float32Array.from([0, 0, 1, 0, 10, 0]),
        locationWeights: Float32Array.from([1, 1, 1]),
        matrix: Float32Array.from(matrix),
        neighborhoodSize: 2,
        bandwidthRatio: 0.5
      });
    // One flow of 100 inside zone 1: S_ij = 100 w_i w_j with w = [k, 1, k] / (1 + k). Zone 1 is a
    // neighbour of all three zones, so the total grows to 100 ((1 + 2k) / (1 + k))^2 = 125.26.
    const central = await run([0, 0, 0, 0, 100, 0, 0, 0, 0]);
    const centralWeights = [k, 1, k].map(value => value / (1 + k));
    for (let origin = 0; origin < 3; origin++) {
      for (let destination = 0; destination < 3; destination++) {
        expect(central.smoothed[origin * 3 + destination]).toBeCloseTo(
          100 * centralWeights[origin] * centralWeights[destination],
          3
        );
      }
    }
    expect(central.smoothed.reduce((sum, value) => sum + value, 0)).toBeCloseTo(125.26, 1);
    // One flow of 100 inside zone 2: no other zone has zone 2 as a neighbour, so only S_22 is
    // non-zero and the total shrinks to 100 / (1 + k)^2 = 77.58.
    const remote = await run([0, 0, 0, 0, 0, 0, 0, 0, 100]);
    remote.smoothed.forEach((value, index) => {
      expect(value).toBeCloseTo(index === 8 ? 100 / (1 + k) ** 2 : 0, 3);
    });
    expect(remote.smoothed.reduce((sum, value) => sum + value, 0)).toBeCloseTo(77.58, 1);
  });

  it('reads the OD pairs of the aggregation like the dense matrix, ignoring pairs past the count and bad weights', async () => {
    const origins: number[] = [];
    const destinations: number[] = [];
    const weights: number[] = [];
    const dense = new Float32Array(zoneCount * zoneCount);
    matrix.forEach((value, index) => {
      if (value !== 0) {
        origins.push(Math.floor(index / zoneCount));
        destinations.push(index % zoneCount);
        weights.push(value);
        dense[index] = value;
      }
    });
    const count = origins.length;
    // Rows past the count carry a sentinel zone, a stray zone and a NaN weight that must not land.
    origins.push(0xffffffff, 1, 2);
    destinations.push(0xffffffff, 2, 3);
    weights.push(5, 7, NaN);
    const fromPairs = await runSmoothing({
      positions: zones.positions,
      locationWeights,
      neighborhoodSize: 3,
      bandwidthRatio: 0.5,
      pairs: {origins, destinations, weights, count}
    });
    const fromMatrix = await runSmoothing({
      positions: zones.positions,
      locationWeights,
      matrix: dense,
      neighborhoodSize: 3,
      bandwidthRatio: 0.5
    });
    expect(fromPairs.smoothed).toEqual(fromMatrix.smoothed);
  });
});
