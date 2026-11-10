// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpaceTimeKFunctionParameterValues,
  GPUSpaceTimeKFunction,
  type GPUSpaceTimeKEdgeCorrection,
  type GPUSpaceTimePairCorrelationKernel
} from '../../../src/dataframe-analysis/pair-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createSpaceTimeKScene,
  getSpaceTimeKReference,
  type SpaceTimeKScene
} from './space-time-k-function-reference';

const BOUNDS = [0, 0, 10, 8] as const;
const TIME_WINDOW = [0, 20] as const;
const MAXIMUM_DISTANCE = 4.5;
const MAXIMUM_TIME_LAG = 9;
const DISTANCE_CLASS_COUNT = 4;
const TIME_CLASS_COUNT = 3;
const SPATIAL_BANDWIDTH = 0.9;
const TEMPORAL_BANDWIDTH = 1.7;
const BINS = DISTANCE_CLASS_COUNT * TIME_CLASS_COUNT;

type Options = {
  edgeCorrection: GPUSpaceTimeKEdgeCorrection;
  pairCorrelationKernel?: GPUSpaceTimePairCorrelationKernel;
  pairCorrelation?: boolean;
  local?: boolean;
  marginals?: boolean;
};

async function runKFunction(
  scene: SpaceTimeKScene,
  options: Options,
  shape = {maximumDistance: MAXIMUM_DISTANCE, maximumTimeLag: MAXIMUM_TIME_LAG}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = scene.times.length;
  const graph = new GPUCommandGraph(device, {id: 'space-time-k-graph'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    data: ArrayLike<number>
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const out = <Format extends 'float32' | 'uint32'>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const k = out('k', BINS, 'float32');
  const spatialK = options.marginals
    ? out('spatial-k', DISTANCE_CLASS_COUNT, 'float32')
    : undefined;
  const temporalK = options.marginals ? out('temporal-k', TIME_CLASS_COUNT, 'float32') : undefined;
  const d = options.marginals ? out('d', BINS, 'float32') : undefined;
  const pairCorrelation = options.pairCorrelation ? out('pcf', BINS, 'float32') : undefined;
  const pairCounts = out('pair-counts', BINS, 'uint32');
  const local = options.local ? out('local', rows * BINS, 'float32') : undefined;
  const status = out('status', 1, 'uint32');
  graph.add(
    new GPUSpaceTimeKFunction({
      positions: input('positions', 'float32x2', scene.positions),
      times: input('times', 'float32', scene.times),
      intensities: scene.intensities
        ? input('intensities', 'float32', scene.intensities)
        : undefined,
      mask: scene.mask ? input('mask', 'uint32', scene.mask) : undefined,
      parameters: input(
        'parameters',
        'float32',
        getGPUSpaceTimeKFunctionParameterValues({
          bounds: BOUNDS,
          maximumDistance: shape.maximumDistance,
          maximumTimeLag: shape.maximumTimeLag,
          timeWindow: TIME_WINDOW,
          spatialBandwidth: SPATIAL_BANDWIDTH,
          temporalBandwidth: TEMPORAL_BANDWIDTH
        })
      ),
      gridSize: [4, 4],
      distanceClassCount: DISTANCE_CLASS_COUNT,
      timeClassCount: TIME_CLASS_COUNT,
      edgeCorrection: options.edgeCorrection,
      pairCorrelationKernel: options.pairCorrelationKernel,
      output: {
        k: k.view,
        spatialK: spatialK?.view,
        temporalK: temporalK?.view,
        d: d?.view,
        pairCorrelation: pairCorrelation?.view,
        pairCounts: pairCounts.view,
        local: local?.view,
        status: status.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      k: Array.from(await readFloat32(k.buffer, BINS)),
      spatialK: spatialK && Array.from(await readFloat32(spatialK.buffer, DISTANCE_CLASS_COUNT)),
      temporalK: temporalK && Array.from(await readFloat32(temporalK.buffer, TIME_CLASS_COUNT)),
      d: d && Array.from(await readFloat32(d.buffer, BINS)),
      pairCorrelation:
        pairCorrelation && Array.from(await readFloat32(pairCorrelation.buffer, BINS)),
      pairCounts: Array.from(await readUint32(pairCounts.buffer, BINS)),
      local: local && Array.from(await readFloat32(local.buffer, rows * BINS)),
      status: (await readUint32(status.buffer, 1))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function getReference(
  scene: SpaceTimeKScene,
  options: Options,
  shape = {maximumDistance: MAXIMUM_DISTANCE, maximumTimeLag: MAXIMUM_TIME_LAG}
) {
  return getSpaceTimeKReference({
    scene,
    bounds: BOUNDS,
    timeWindow: TIME_WINDOW,
    maximumDistance: shape.maximumDistance,
    maximumTimeLag: shape.maximumTimeLag,
    distanceClassCount: DISTANCE_CLASS_COUNT,
    timeClassCount: TIME_CLASS_COUNT,
    edgeCorrection: options.edgeCorrection,
    pairCorrelationKernel: options.pairCorrelationKernel,
    spatialBandwidth: options.pairCorrelation ? SPATIAL_BANDWIDTH : undefined,
    temporalBandwidth: options.pairCorrelation ? TEMPORAL_BANDWIDTH : undefined
  });
}

/**
 * GPU versus the binary64 reference. Counts are exact. Weighted sums are fixed-point with a
 * rounding of 2^-17 per pair term, so a relative tolerance of 1e-4 of the largest value covers them
 * and the float32 evaluation of the arc weights.
 */
function expectClose(
  actual: ArrayLike<number>,
  expected: readonly number[],
  label: string,
  tolerance = 1e-4
) {
  const finite = expected.filter(Number.isFinite).map(Math.abs);
  const scale = Math.max(...finite, 1e-12);
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(actual[index], `${label} ${index} should be NaN`).toBeNaN();
    } else {
      expect(Number.isFinite(actual[index]), `${label} ${index} finite`).toBe(true);
      worst = Math.max(worst, Math.abs(actual[index] - value));
    }
  }
  expect(worst, label).toBeLessThanOrEqual(tolerance * scale);
}

// float32 distances and lags carry a relative error of about 6e-8, so a margin of 2e-5 classes
// cannot flip a class between the float32 kernels and the float64 reference.
it('has a scene away from every class boundary', () => {
  for (const seed of [4, 9, 17]) {
    const reference = getReference(createSpaceTimeKScene(seed), {edgeCorrection: 'isotropic'});
    expect(reference.minimumBoundaryMargin, `seed ${seed}`).toBeGreaterThan(2e-5);
  }
});

it.each([
  {
    name: 'isotropic homogeneous with marginals and D',
    seed: 4,
    options: {edgeCorrection: 'isotropic', marginals: true} as Options
  },
  {
    name: 'no edge correction homogeneous with marginals and D',
    seed: 9,
    options: {edgeCorrection: 'none', marginals: true} as Options
  }
])('GPUSpaceTimeKFunction matches the binary64 reference: $name', async ({seed, options}) => {
  const scene = createSpaceTimeKScene(seed, {mask: true});
  const reference = getReference(scene, options);
  const actual = await runKFunction(scene, options);
  expect(actual.status).toBe(0);
  expect(actual.pairCounts).toEqual(reference.pairCounts);
  expect(reference.pairCounts.every(count => count > 0)).toBe(true);
  expectClose(actual.k, reference.k, 'K');
  expectClose(actual.spatialK!, reference.spatialK, 'K_s');
  expectClose(actual.temporalK!, reference.temporalK, 'K_t');
  expectClose(actual.d!, reference.d, 'D');
  // The space-time interaction of the clusters is visible: D is not identically zero.
  expect(Math.max(...reference.d.map(Math.abs))).toBeGreaterThan(0.1);
});

it('GPUSpaceTimeKFunction matches the reference with first-order intensities', async () => {
  const scene = createSpaceTimeKScene(17, {intensities: true});
  const options: Options = {edgeCorrection: 'isotropic'};
  const reference = getReference(scene, options);
  const actual = await runKFunction(scene, options);
  expect(actual.status).toBe(0);
  expect(actual.pairCounts).toEqual(reference.pairCounts);
  expectClose(actual.k, reference.k, 'inhomogeneous K');
  // The homogeneous estimator on the same events differs, so the intensities were used.
  const homogeneous = getReference({...scene, intensities: undefined}, options);
  expect(
    Math.max(...reference.k.map((value, bin) => Math.abs(value - homogeneous.k[bin])))
  ).toBeGreaterThan(0.05);
});

it.each([
  'box',
  'epanechnikov'
] as const)('GPUSpaceTimeKFunction pair correlation matches the reference (%s)', async kernel => {
  const scene = createSpaceTimeKScene(4);
  const options: Options = {
    edgeCorrection: 'isotropic',
    pairCorrelation: true,
    pairCorrelationKernel: kernel
  };
  const reference = getReference(scene, options);
  const actual = await runKFunction(scene, options);
  expect(Math.max(...reference.pairCorrelation)).toBeGreaterThan(0);
  expectClose(actual.pairCorrelation!, reference.pairCorrelation, `pair correlation ${kernel}`);
  // K is unchanged by the wider pair search that the kernel needs.
  expectClose(actual.k, reference.k, 'K with pair correlation');
  expect(actual.pairCounts).toEqual(reference.pairCounts);
});

it('GPUSpaceTimeKFunction local statistics match the reference and average to K', async () => {
  const scene = createSpaceTimeKScene(9, {mask: true});
  const options: Options = {edgeCorrection: 'isotropic', local: true};
  const reference = getReference(scene, options);
  const actual = await runKFunction(scene, options);
  expectClose(actual.local!, reference.local, 'LISTA');
  // Excluded rows stay NaN, included rows average to the global K.
  const included = new Set(reference.included);
  const rows = scene.times.length;
  for (let row = 0; row < rows; row++) {
    if (!included.has(row)) {
      expect(actual.local![row * BINS], `excluded row ${row}`).toBeNaN();
    }
  }
  for (let bin = 0; bin < BINS; bin++) {
    let mean = 0;
    for (const row of reference.included) {
      mean += actual.local![row * BINS + bin] / reference.included.length;
    }
    expect(Math.abs(mean - actual.k[bin]), `bin ${bin}`).toBeLessThanOrEqual(
      2e-4 * Math.max(...actual.k)
    );
  }
});

it('GPUSpaceTimeKFunction ignores row order and is bitwise reproducible', async () => {
  const scene = createSpaceTimeKScene(9, {mask: true});
  const rows = scene.times.length;
  const order = Array.from({length: rows}, (_, index) => (index * 13 + 5) % rows);
  expect(new Set(order).size).toBe(rows);
  const shuffled: SpaceTimeKScene = {
    positions: order.flatMap(row => [scene.positions[2 * row], scene.positions[2 * row + 1]]),
    times: order.map(row => scene.times[row]),
    mask: order.map(row => scene.mask![row])
  };
  const options: Options = {edgeCorrection: 'isotropic', marginals: true};
  const first = await runKFunction(scene, options);
  const second = await runKFunction(scene, options);
  const permuted = await runKFunction(shuffled, options);
  expect(second.k).toEqual(first.k);
  expect(second.d).toEqual(first.d);
  expect(permuted.pairCounts).toEqual(first.pairCounts);
  expectClose(permuted.k, first.k, 'permuted K', 1e-6);
  expectClose(permuted.temporalK!, first.temporalK!, 'permuted K_t', 1e-6);
});

it('GPUSpaceTimeKFunction reports an undefined estimate when fewer than two events are included', async () => {
  const scene: SpaceTimeKScene = {positions: [5, 4, 30, 30], times: [3, 4]};
  const actual = await runKFunction(scene, {edgeCorrection: 'isotropic', marginals: true});
  expect(actual.status).toBe(1);
  expect(actual.k.every(Number.isNaN)).toBe(true);
  expect(actual.d!.every(Number.isNaN)).toBe(true);
  expect(actual.pairCounts.every(count => count === 0)).toBe(true);
});

it('GPUSpaceTimeKFunction puts a pair exactly on a class edge in the lower class (right-closed)', async () => {
  // Distances 0, 1, 2, 3, 4 and lags 0, 2, 4, 6, 8, 9 fall exactly on the edges of 4 distance
  // classes over (0, 4] and 3 lag classes over (0, 6], so floor and ceil classes differ.
  const scene: SpaceTimeKScene = {
    positions: [2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 2, 2],
    times: [4, 6, 8, 10, 12, 4]
  };
  const shape = {maximumDistance: 4, maximumTimeLag: 6};
  const options: Options = {edgeCorrection: 'none', marginals: true};
  const reference = getReference(scene, options, shape);
  const actual = await runKFunction(scene, options, shape);
  expect(reference.pairCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(10);
  expect(actual.pairCounts).toEqual(reference.pairCounts);
  expectClose(actual.k, reference.k, 'K on class edges');
  expectClose(actual.temporalK!, reference.temporalK, 'K_t on class edges');
  expectClose(actual.spatialK!, reference.spatialK, 'K_s on class edges');
});
