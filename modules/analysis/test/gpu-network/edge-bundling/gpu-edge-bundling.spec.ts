// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUEdgeBundlingSettings} from '../../../src/gpu-network/edge-bundling/index';
import {isSoftwareDevice, readUint32} from '../../utils/gpu-contributor-test-utils';
import {bundleEdgesOracle, type EdgeBundlingOracleInput} from './edge-bundling-oracle';
import {
  createClusteredScene,
  createFixture,
  type Fixture,
  type Scene
} from './edge-bundling-fixture';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  BUNDLING_INDEPENDENT_PATHS,
  BUNDLING_INDEPENDENT_SETTINGS
} from './bundling-independent-fixture';

it('pins endpoints, every coordinate and maximum point error over an independent full schedule (NET-121)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(
    device,
    {
      positions: Float32Array.of(100, 450, 900, 450, 100, 550, 900, 550),
      sources: Uint32Array.of(0, 2),
      targets: Uint32Array.of(1, 3)
    },
    {
      pointsPerEdge: 5,
      iterations: 8,
      densityResolution: 64,
      parameters: BUNDLING_INDEPENDENT_SETTINGS
    }
  );
  try {
    const paths = await fixture.run();
    expect(paths.length).toBe(20);
    expect(Array.from(paths).every(Number.isFinite)).toBe(true);
    for (const endpoint of [0, 4, 5, 9]) {
      expect(Array.from(paths.slice(2 * endpoint, 2 * endpoint + 2))).toEqual(
        BUNDLING_INDEPENDENT_PATHS.slice(2 * endpoint, 2 * endpoint + 2)
      );
    }
    const pointErrors = Array.from({length: 10}, (_, point) =>
      Math.hypot(
        paths[2 * point] - BUNDLING_INDEPENDENT_PATHS[2 * point],
        paths[2 * point + 1] - BUNDLING_INDEPENDENT_PATHS[2 * point + 1]
      )
    );
    expect(Math.max(...pointErrors)).toBeLessThan(0.25);
  } finally {
    fixture.destroy();
  }
});

// The 10k-edge timing measurement is in `gpu-edge-bundling-bench.spec.ts` (TOOL-110).

/**
 * Tight parity tolerance, as a fraction of the work box side, for a single iteration. The schedule
 * is chaotic (see the oracle), so longer runs are compared by quantiles instead.
 */
const PARITY_TOLERANCE = 1e-5;

function oracleFor(
  scene: Scene,
  fixture: Fixture,
  parameters: GPUEdgeBundlingSettings = {}
): Omit<EdgeBundlingOracleInput, never> {
  return {
    positions: scene.positions,
    sources: scene.sources,
    targets: scene.targets,
    mask: scene.mask,
    pointsPerEdge: fixture.pointsPerEdge,
    iterations: fixture.iterations,
    densityResolution: fixture.densityResolution,
    activeIterations: parameters.activeIterations,
    kernelRadius: parameters.kernelRadius,
    lambda: parameters.lambda,
    smoothing: parameters.smoothing,
    stepScale: parameters.stepScale
  };
}

/** p50, p90 and maximum coordinate difference as fractions of the box side. */
function getDivergence(
  actual: Float32Array,
  expected: Float32Array,
  side: number
): {p50: number; p90: number; max: number} {
  const sorted = Array.from(actual, (value, i) => Math.abs(value - expected[i]) / side).sort(
    (a, b) => a - b
  );
  return {
    p50: sorted[sorted.length >> 1],
    p90: sorted[Math.floor(sorted.length * 0.9)],
    max: sorted[sorted.length - 1]
  };
}

/** Maximum coordinate difference between two path arrays, as a fraction of the box side. */
function maxDivergence(actual: Float32Array, expected: Float32Array, side: number): number {
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]) / side);
  }
  return maximum;
}

const WORK_WIDTH = 1000;

/** `count` horizontal edges at evenly spaced heights, one vertex pair per edge. */
function createParallelScene(count: number, spread: number): Scene {
  const positions = new Float32Array(count * 4);
  const sources = new Uint32Array(count);
  const targets = new Uint32Array(count);
  for (let e = 0; e < count; e++) {
    const y = WORK_WIDTH / 2 + (e - (count - 1) / 2) * spread;
    positions.set([WORK_WIDTH * 0.1, y, WORK_WIDTH * 0.9, y], e * 4);
    sources[e] = e * 2;
    targets[e] = e * 2 + 1;
  }
  return {positions, sources, targets};
}

function getMidpointSpread(paths: Float32Array, count: number, pointsPerEdge: number): number {
  const ys = Array.from(
    {length: count},
    (_, e) => paths[(e * pointsPerEdge + (pointsPerEdge >> 1)) * 2 + 1]
  );
  return Math.max(...ys) - Math.min(...ys);
}

it('GPUEdgeBundling matches the CPU oracle after one iteration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createClusteredScene(48, 64, 7);
  const fixture = createFixture(device, scene, {
    pointsPerEdge: 16,
    iterations: 1,
    densityResolution: 128
  });
  const actual = await fixture.run();
  const expected = bundleEdgesOracle(oracleFor(scene, fixture));
  const divergence = maxDivergence(actual, expected.paths, expected.box[2]);
  console.log(
    `edge-bundling 1-iteration parity: max divergence ${divergence.toExponential(2)} of box side`
  );
  expect(divergence).toBeLessThan(isSoftwareDevice(device) ? 1e-4 : PARITY_TOLERANCE);
  // The iteration must have moved interior points away from the straight line.
  const straight = bundleEdgesOracle({...oracleFor(scene, fixture), activeIterations: 0});
  expect(maxDivergence(actual, straight.paths, expected.box[2])).toBeGreaterThan(0.005);
  fixture.destroy();
});

it('GPUEdgeBundling tracks the CPU oracle statistically over a full schedule', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createClusteredScene(48, 64, 7);
  const fixture = createFixture(device, scene, {
    pointsPerEdge: 16,
    iterations: 8,
    densityResolution: 128
  });
  const actual = await fixture.run();
  const expected = bundleEdgesOracle(oracleFor(scene, fixture));
  const {p50, p90, max} = getDivergence(actual, expected.paths, expected.box[2]);
  console.log(
    `edge-bundling 8-iteration divergence of box side: p50 ${p50.toExponential(2)} p90 ${p90.toExponential(2)} max ${max.toExponential(2)}; ${fixture.nodeCount} nodes`
  );
  expect(p50).toBeLessThan(2e-4);
  expect(p90).toBeLessThan(3e-3);
  expect(Array.from(actual).every(Number.isFinite)).toBe(true);
  const straight = bundleEdgesOracle({...oracleFor(scene, fixture), activeIterations: 0});
  expect(maxDivergence(actual, straight.paths, expected.box[2])).toBeGreaterThan(0.01);
  fixture.destroy();
});

it('GPUEdgeBundling attracts nearby parallel edges and pins endpoints exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pointsPerEdge = 33;
  const count = 6;
  const scene = createParallelScene(count, 12);
  const fixture = createFixture(device, scene, {
    pointsPerEdge,
    iterations: 12
  });
  const paths = await fixture.run();
  const straight = bundleEdgesOracle({
    ...oracleFor(scene, fixture),
    activeIterations: 0
  }).paths;
  expect(getMidpointSpread(paths, count, pointsPerEdge)).toBeLessThan(
    getMidpointSpread(straight, count, pointsPerEdge) * 0.9
  );
  for (let e = 0; e < count; e++) {
    for (const point of [0, pointsPerEdge - 1]) {
      const vertex = point === 0 ? scene.sources[e] : scene.targets[e];
      expect(paths[(e * pointsPerEdge + point) * 2]).toBe(scene.positions[vertex * 2]);
      expect(paths[(e * pointsPerEdge + point) * 2 + 1]).toBe(scene.positions[vertex * 2 + 1]);
    }
  }
  fixture.destroy();
});

it('GPUEdgeBundling leaves a lone straight edge essentially in place', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pointsPerEdge = 25;
  const lone = createFixture(device, createParallelScene(1, 0), {
    pointsPerEdge,
    iterations: 10
  });
  const lonePaths = await lone.run();
  // Box side is 0.8 * 1.1 of the width; the bilinear gradient keeps drift at the quantization floor.
  const side = WORK_WIDTH * 0.8 * 1.1;
  let loneDrift = 0;
  for (let i = 0; i < pointsPerEdge; i++) {
    loneDrift = Math.max(loneDrift, Math.abs(lonePaths[i * 2 + 1] - WORK_WIDTH / 2));
  }
  expect(loneDrift).toBeLessThan(side * 0.01);
  lone.destroy();

  const scene = createParallelScene(6, 12);
  const pulledFixture = createFixture(device, scene, {
    pointsPerEdge,
    iterations: 10
  });
  const pulled = await pulledFixture.run();
  const outerPull = Math.abs(pulled[(pointsPerEdge >> 1) * 2 + 1] - scene.positions[1]);
  console.log(
    `edge-bundling lone drift ${loneDrift.toFixed(3)}, outer pull ${outerPull.toFixed(3)}`
  );
  expect(outerPull).toBeGreaterThan(loneDrift * 5);
  pulledFixture.destroy();
});

it('GPUEdgeBundling honors per-frame parameters without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createClusteredScene(24, 32, 11);
  for (const parameterFormat of ['uint32', 'float32'] as const) {
    const fixture = createFixture(device, scene, {
      pointsPerEdge: 12,
      iterations: 10,
      densityResolution: 96,
      parameterFormat,
      parameters: {activeIterations: 1}
    });
    const side = bundleEdgesOracle(oracleFor(scene, fixture)).box[2];
    const compare = async (parameters: GPUEdgeBundlingSettings) =>
      maxDivergence(
        await fixture.run(),
        bundleEdgesOracle(oracleFor(scene, fixture, parameters)).paths,
        side
      );
    // One iteration compares tightly, and every parameter (except lambda) shapes it.
    expect(await compare({activeIterations: 1})).toBeLessThan(PARITY_TOLERANCE);
    const first = await fixture.run();
    const changed = {activeIterations: 1, kernelRadius: 0.05, smoothing: 0.3, stepScale: 0.8};
    fixture.setParameters(changed);
    expect(await compare(changed)).toBeLessThan(PARITY_TOLERANCE);
    const second = await fixture.run();
    expect(maxDivergence(second, first, side)).toBeGreaterThan(0.001);

    // Lambda only matters from the second iteration on, where parity is statistical.
    const decayed = {activeIterations: 3, lambda: 0.5};
    fixture.setParameters(decayed);
    const actualDecayed = await fixture.run();
    const oracleDecayed = bundleEdgesOracle(oracleFor(scene, fixture, decayed)).paths;
    expect(getDivergence(actualDecayed, oracleDecayed, side).p50).toBeLessThan(2e-4);
    const defaultLambda = bundleEdgesOracle(oracleFor(scene, fixture, {activeIterations: 3})).paths;
    expect(getDivergence(oracleDecayed, defaultLambda, side).p90).toBeGreaterThan(1e-3);

    fixture.setParameters({activeIterations: 0});
    const straight = await fixture.run();
    expect(
      maxDivergence(
        straight,
        bundleEdgesOracle({...oracleFor(scene, fixture), activeIterations: 0}).paths,
        side
      )
    ).toBeLessThan(1e-5);
    // activeIterations above the compiled maximum clamps instead of overrunning.
    fixture.setParameters({activeIterations: 1000});
    const clamped = await fixture.run();
    fixture.setParameters({activeIterations: 10});
    expect(maxDivergence(clamped, await fixture.run(), side)).toBe(0);
    fixture.destroy();
  }
});

it('GPUEdgeBundling activeIterations 0 produces exact straight subdivisions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createParallelScene(3, 20);
  const pointsPerEdge = 9;
  const fixture = createFixture(device, scene, {
    pointsPerEdge,
    iterations: 5,
    parameters: {activeIterations: 0}
  });
  const paths = await fixture.run();
  for (let e = 0; e < 3; e++) {
    for (let i = 0; i < pointsPerEdge; i++) {
      const t = i / (pointsPerEdge - 1);
      expect(paths[(e * pointsPerEdge + i) * 2]).toBeCloseTo(100 + 800 * t, 2);
      expect(paths[(e * pointsPerEdge + i) * 2 + 1]).toBeCloseTo(scene.positions[e * 4 + 1], 2);
    }
  }
  fixture.destroy();
});

it('GPUEdgeBundling excludes masked and invalid edges and collapses their paths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = createClusteredScene(32, 24, 3);
  const mask = new Uint32Array(24).fill(1);
  mask[2] = 0;
  mask[9] = 0;
  const targets = Uint32Array.from(base.targets);
  targets[5] = 9999; // out of range, dead
  const scene: Scene = {...base, targets, mask};
  const pointsPerEdge = 10;
  const fixture = createFixture(device, scene, {
    pointsPerEdge,
    iterations: 6,
    densityResolution: 96
  });
  const paths = await fixture.run();
  const expected = bundleEdgesOracle(oracleFor(scene, fixture));
  expect(getDivergence(paths, expected.paths, expected.box[2]).p50).toBeLessThan(2e-4);
  const oneIteration = createFixture(device, scene, {
    pointsPerEdge,
    iterations: 1,
    densityResolution: 96
  });
  expect(
    maxDivergence(
      await oneIteration.run(),
      bundleEdgesOracle(oracleFor(scene, oneIteration)).paths,
      expected.box[2]
    )
  ).toBeLessThan(PARITY_TOLERANCE);
  oneIteration.destroy();
  for (const dead of [2, 5, 9]) {
    const source = scene.sources[dead];
    for (let i = 0; i < pointsPerEdge; i++) {
      expect(paths[(dead * pointsPerEdge + i) * 2]).toBe(scene.positions[source * 2]);
      expect(paths[(dead * pointsPerEdge + i) * 2 + 1]).toBe(scene.positions[source * 2 + 1]);
    }
  }
  // Dead edges must not influence live ones: dropping them from the scene gives the same density.
  const liveIds = [...Array(24).keys()].filter(e => ![2, 5, 9].includes(e));
  const reduced: Scene = {
    positions: scene.positions,
    sources: Uint32Array.from(liveIds.map(e => scene.sources[e])),
    targets: Uint32Array.from(liveIds.map(e => scene.targets[e]))
  };
  const reducedFixture = createFixture(device, reduced, {
    pointsPerEdge,
    iterations: 6,
    densityResolution: 96
  });
  const reducedPaths = await reducedFixture.run();
  liveIds.forEach((edge, reducedEdge) => {
    for (let k = 0; k < pointsPerEdge * 2; k++) {
      expect(
        Math.abs(
          paths[edge * pointsPerEdge * 2 + k] - reducedPaths[reducedEdge * pointsPerEdge * 2 + k]
        )
      ).toBeLessThan(expected.box[2] * 1e-6);
    }
  });
  reducedFixture.destroy();
  fixture.destroy();
});

it('GPUEdgeBundling writes startIndices and the drawIndirect record', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createClusteredScene(16, 7, 5);
  const fixture = createFixture(device, scene, {
    pointsPerEdge: 6,
    iterations: 2,
    withIndices: true
  });
  await fixture.run();
  expect(await readUint32(fixture.startIndicesBuffer!, 8)).toEqual([0, 6, 12, 18, 24, 30, 36, 42]);
  expect(await readUint32(fixture.drawRecordBuffer!, 4)).toEqual([6, 7, 0, 0]);
  fixture.destroy();
});

it('GPUEdgeBundling is scale and translation invariant', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createClusteredScene(24, 30, 21);
  const scaled: Scene = {
    ...scene,
    positions: Float32Array.from(scene.positions, (value, i) => value * 4096 + (i % 2 ? 1e4 : -3e4))
  };
  const options = {pointsPerEdge: 12, iterations: 1, densityResolution: 96};
  const a = createFixture(device, scene, options);
  const b = createFixture(device, scaled, options);
  const pathsA = await a.run();
  const pathsB = await b.run();
  const sideA = bundleEdgesOracle(oracleFor(scene, a)).box[2];
  const sideB = bundleEdgesOracle(oracleFor(scaled, b)).box[2];
  let maximum = 0;
  for (let i = 0; i < pathsA.length; i++) {
    const mapped = (pathsA[i] * 4096 + (i % 2 ? 1e4 : -3e4)) / sideB;
    maximum = Math.max(maximum, Math.abs(mapped - pathsB[i] / sideB));
  }
  expect(sideB / sideA).toBeCloseTo(4096, 0);
  expect(maximum).toBeLessThan(1e-4);
  a.destroy();
  b.destroy();
});

it('GPUEdgeBundling geographic input matches planar bundling of cos-latitude scaled longitude', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const planar = createClusteredScene(24, 30, 33);
  // Map the 0..100 scene to lon/lat degrees: 40 degrees of longitude by 10 of latitude near 60 N.
  const geographic: Scene = {
    ...planar,
    positions: Float32Array.from(planar.positions, (value, i) =>
      i % 2 ? 55 + value * 0.1 : -20 + value * 0.4
    )
  };
  // The GPU takes the mid-latitude from the live edge endpoints only.
  let minLat = Infinity;
  let maxLat = -Infinity;
  for (const vertex of [...geographic.sources, ...geographic.targets]) {
    const latitude = geographic.positions[vertex * 2 + 1];
    minLat = Math.min(minLat, latitude);
    maxLat = Math.max(maxLat, latitude);
  }
  const xScale = Math.cos((0.5 * (minLat + maxLat) * Math.PI) / 180);
  const scaled: Scene = {
    ...geographic,
    positions: Float32Array.from(geographic.positions, (value, i) =>
      i % 2 ? value : value * xScale
    )
  };
  const options = {pointsPerEdge: 12, iterations: 1, densityResolution: 96};
  const a = createFixture(device, geographic, {...options, geographic: true});
  const b = createFixture(device, scaled, options);
  const flat = createFixture(device, geographic, options);
  const pathsA = await a.run();
  const pathsB = await b.run();
  const pathsFlat = await flat.run();
  const side = bundleEdgesOracle(oracleFor(scaled, b)).box[2];
  let maximum = 0;
  let flatMaximum = 0;
  for (let i = 0; i < pathsA.length; i++) {
    const mapped = i % 2 ? pathsB[i] : pathsB[i] / xScale;
    const factor = i % 2 ? 1 : xScale;
    maximum = Math.max(maximum, Math.abs(pathsA[i] - mapped) * factor);
    flatMaximum = Math.max(flatMaximum, Math.abs(pathsA[i] - pathsFlat[i]) * factor);
  }
  expect(maximum / side).toBeLessThan(1e-4);
  // Without the correction the density is stretched east-west and the paths differ.
  expect(flatMaximum / side).toBeGreaterThan(1e-3);
  a.destroy();
  b.destroy();
  flat.destroy();
});
