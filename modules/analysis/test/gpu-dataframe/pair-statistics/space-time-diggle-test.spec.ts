// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {fillShufflePermutation} from '../../../src/analysis-core/kernels/permutation-shuffle';
import {
  getGPUSpaceTimeDiggleRandomParameterValues,
  getGPUSpaceTimeKFunctionParameterValues,
  GPU_SPACE_TIME_DIGGLE_SUMMARY,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
  GPUSpaceTimeDiggleTest,
  type GPUSpaceTimeKEdgeCorrection
} from '../../../src/dataframe-analysis/pair-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
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

type Shape = {
  distanceClassCount: number;
  timeClassCount: number;
  permutationCount: number;
  edgeCorrection: GPUSpaceTimeKEdgeCorrection;
  seed: number;
};

async function runDiggleTest(scene: SpaceTimeKScene, shape: Shape) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'space-time-diggle-graph'});
  const buffers: Buffer[] = [];
  const bins = shape.distanceClassCount * shape.timeClassCount;
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
  const out = <Format extends 'float32' | 'uint32' = 'float32'>(
    name: string,
    length: number,
    format: Format = 'float32' as Format
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const d = out('d', bins);
  const statistics = out('statistics', shape.permutationCount + 1);
  const summary = out('summary', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH);
  const permutationMean = out('mean', bins);
  const permutationStandardDeviation = out('sd', bins);
  const status = out('status', 1, 'uint32');
  const test = new GPUSpaceTimeDiggleTest({
    positions: input('positions', 'float32x2', scene.positions),
    times: input('times', 'float32', scene.times),
    mask: scene.mask ? input('mask', 'uint32', scene.mask) : undefined,
    parameters: input(
      'parameters',
      'float32',
      getGPUSpaceTimeKFunctionParameterValues({
        bounds: BOUNDS,
        maximumDistance: MAXIMUM_DISTANCE,
        maximumTimeLag: MAXIMUM_TIME_LAG,
        timeWindow: TIME_WINDOW
      })
    ),
    randomParameters: input(
      'random-parameters',
      'uint32',
      getGPUSpaceTimeDiggleRandomParameterValues(shape.seed)
    ),
    gridSize: [4, 4],
    distanceClassCount: shape.distanceClassCount,
    timeClassCount: shape.timeClassCount,
    edgeCorrection: shape.edgeCorrection,
    permutationCount: shape.permutationCount,
    output: {
      d: d.view,
      statistics: statistics.view,
      summary: summary.view,
      permutationMean: permutationMean.view,
      permutationStandardDeviation: permutationStandardDeviation.view,
      status: status.view
    }
  });
  graph.add(test);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      d: Array.from(await readFloat32(d.buffer, bins)),
      statistics: Array.from(await readFloat32(statistics.buffer, shape.permutationCount + 1)),
      summary: Array.from(await readFloat32(summary.buffer, GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)),
      permutationMean: Array.from(await readFloat32(permutationMean.buffer, bins)),
      permutationStandardDeviation: Array.from(
        await readFloat32(permutationStandardDeviation.buffer, bins)
      ),
      estimatedCommandNodeCount: test.estimatedCommandNodeCount,
      batchSize: test.batchSize
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/**
 * Binary64 reference of the documented definition: the included events in row order, the swap-or-not
 * shuffle (its TypeScript twin, key `(seedLow, seedHigh)`, permutation number `p + 1`, the Diggle
 * Philox purpose) applied to their times, `K` of every permuted pattern from the K reference, and
 * the sum of `D` over the classes, the statistic of splancs `stmctest`.
 * `space-time-diggle-splancs.node.spec.ts` ties this K reference to a port of the splancs source,
 * and `space-time-diggle-splancs.spec.ts` compares the kernel with that port directly. The mean
 * and standard deviation per class are the optional diagnostic outputs.
 */
function getDiggleReference(scene: SpaceTimeKScene, shape: Shape) {
  const options = {
    bounds: BOUNDS,
    timeWindow: TIME_WINDOW,
    maximumDistance: MAXIMUM_DISTANCE,
    maximumTimeLag: MAXIMUM_TIME_LAG,
    distanceClassCount: shape.distanceClassCount,
    timeClassCount: shape.timeClassCount,
    edgeCorrection: shape.edgeCorrection,
    // The Diggle test follows splancs at exact ties (the scenes here have none).
    tieRule: 'splancs' as const
  };
  const observed = getSpaceTimeKReference({scene, ...options});
  const bins = shape.distanceClassCount * shape.timeClassCount;
  const included = observed.included;
  const seed = getGPUSpaceTimeDiggleRandomParameterValues(shape.seed);
  const permutedD: number[][] = [];
  for (let permutation = 0; permutation < shape.permutationCount; permutation++) {
    const order = new Uint32Array(included.length);
    fillShufflePermutation(
      order,
      included.length,
      [seed[0], seed[1]],
      permutation + 1,
      PHILOX_PURPOSES.spaceTimeDiggleShuffle.id
    );
    const times = [...scene.times];
    included.forEach((row, rank) => {
      times[row] = scene.times[included[order[rank]]];
    });
    const permuted = getSpaceTimeKReference({scene: {...scene, times}, ...options});
    // Locations and the multiset of times are fixed, so the marginals do not move.
    permuted.spatialK.forEach((value, index) => {
      expect(value).toBeCloseTo(observed.spatialK[index], 10);
    });
    permuted.temporalK.forEach((value, index) => {
      expect(value).toBeCloseTo(observed.temporalK[index], 10);
    });
    permutedD.push(permuted.d);
  }
  const mean = new Array<number>(bins).fill(0);
  const deviation = new Array<number>(bins).fill(0);
  for (let bin = 0; bin < bins; bin++) {
    mean[bin] = permutedD.reduce((sum, d) => sum + d[bin], 0) / shape.permutationCount;
    deviation[bin] = Math.sqrt(
      permutedD.reduce((sum, d) => sum + (d[bin] - mean[bin]) ** 2, 0) /
        (shape.permutationCount - 1)
    );
  }
  const getStatistic = (d: number[]) => d.reduce((total, value) => total + value, 0);
  const statistics = [observed.d, ...permutedD].map(getStatistic);
  const permuted = statistics.slice(1);
  return {
    observed,
    mean,
    deviation,
    statistics,
    greater: permuted.filter(value => value >= statistics[0]).length,
    lesser: permuted.filter(value => value <= statistics[0]).length,
    // The statistic sums every class.
    usedClasses: bins
  };
}

function expectDiggleMatches(
  actual: Awaited<ReturnType<typeof runDiggleTest>>,
  reference: ReturnType<typeof getDiggleReference>,
  shape: Shape
) {
  const scale = Math.max(...reference.observed.k.map(Math.abs));
  actual.d.forEach((value, bin) => {
    expect(Math.abs(value - reference.observed.d[bin]), `D ${bin}`).toBeLessThanOrEqual(
      1e-4 * scale
    );
  });
  // Each statistic is a sum of `bins` float32 differences `K - K_s K_t`.
  reference.statistics.forEach((value, index) => {
    expect(Math.abs(actual.statistics[index] - value), `statistic ${index}`).toBeLessThanOrEqual(
      1e-4 * scale * actual.d.length
    );
  });
  // The diagnostic outputs: mean and sample standard deviation of D over the permutations.
  reference.mean.forEach((value, bin) => {
    expect(Math.abs(actual.permutationMean[bin] - value), `mean ${bin}`).toBeLessThanOrEqual(
      1e-4 * scale
    );
    expect(
      Math.abs(actual.permutationStandardDeviation[bin] - reference.deviation[bin]),
      `standard deviation ${bin}`
    ).toBeLessThanOrEqual(1e-4 * scale);
  });
  const summary = actual.summary;
  const permutations = shape.permutationCount;
  expect(summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.greaterCount]).toBe(reference.greater);
  expect(summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.lesserCount]).toBe(reference.lesser);
  expect(summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]).toBeCloseTo(
    (reference.greater + 1) / (permutations + 1),
    6
  );
  expect(summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.usedClasses]).toBe(reference.usedClasses);
  expect(summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.includedCount]).toBe(
    reference.observed.included.length
  );
}

const SHAPES: Shape[] = [
  {
    distanceClassCount: 4,
    timeClassCount: 3,
    permutationCount: 23,
    edgeCorrection: 'isotropic',
    seed: 7
  },
  {
    distanceClassCount: 3,
    timeClassCount: 4,
    permutationCount: 19,
    edgeCorrection: 'none',
    seed: 2 ** 40 + 5
  },
  // Eight permutations per batch: three batches.
  {
    distanceClassCount: 8,
    timeClassCount: 16,
    permutationCount: 19,
    edgeCorrection: 'isotropic',
    seed: 11
  }
];

it.each(SHAPES)('GPUSpaceTimeDiggleTest matches the binary64 reference: %o', async shape => {
  const scene = createSpaceTimeKScene(4, {mask: true});
  const reference = getDiggleReference(scene, shape);
  const actual = await runDiggleTest(scene, shape);
  expectDiggleMatches(actual, reference, shape);
  expect(actual.batchSize).toBe(
    Math.min(
      shape.permutationCount,
      Math.floor(1024 / (shape.distanceClassCount * shape.timeClassCount))
    )
  );
  expect(actual.estimatedCommandNodeCount).toBeGreaterThan(0);
});

it('GPUSpaceTimeDiggleTest rejects the null: a pattern with persistent clusters has an extreme statistic', async () => {
  const shape = SHAPES[0];
  const scene = createSpaceTimeKScene(4, {mask: true});
  const reference = getDiggleReference(scene, shape);
  // The observed statistic is the largest of all: clusters persist in space and time.
  expect(reference.statistics[0]).toBeGreaterThan(Math.max(...reference.statistics.slice(1)));
  const actual = await runDiggleTest(scene, shape);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]).toBeCloseTo(
    1 / (shape.permutationCount + 1),
    6
  );
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.zScore]).toBeGreaterThan(2);
});

it('GPUSpaceTimeDiggleTest is reproducible for a seed, changes with it and ignores row order', async () => {
  const shape = SHAPES[0];
  const scene = createSpaceTimeKScene(9, {mask: true});
  const first = await runDiggleTest(scene, shape);
  const second = await runDiggleTest(scene, shape);
  expect(second.statistics).toEqual(first.statistics);
  const other = await runDiggleTest(scene, {...shape, seed: shape.seed + 1});
  expect(other.statistics.slice(1)).not.toEqual(first.statistics.slice(1));
  // The observed statistic is the sum of the observed D: it does not depend on the seed.
  expect(other.statistics[0]).toBe(first.statistics[0]);
  // Row order changes which event is which rank, so permutations differ, but the observed D does not.
  const rows = scene.times.length;
  const order = Array.from({length: rows}, (_, index) => (index * 13 + 5) % rows);
  const shuffled: SpaceTimeKScene = {
    positions: order.flatMap(row => [scene.positions[2 * row], scene.positions[2 * row + 1]]),
    times: order.map(row => scene.times[row]),
    mask: order.map(row => scene.mask![row])
  };
  const permuted = await runDiggleTest(shuffled, shape);
  permuted.d.forEach((value, bin) => {
    expect(Math.abs(value - first.d[bin]), `D ${bin}`).toBeLessThanOrEqual(
      1e-5 * Math.max(...first.d.map(Math.abs))
    );
  });
});

it('GPUSpaceTimeDiggleTest shuffles only the rows of the pair index when rows fall outside bounds', async () => {
  const shape = SHAPES[0];
  const scene = createSpaceTimeKScene(4, {mask: true});
  // Every seventh row leaves the study bounds but stays inside the time window and the mask.
  const positions = scene.positions.map((value, index) =>
    index % 14 === 0 && index > 0 ? BOUNDS[2] + 5 : value
  );
  const outside: SpaceTimeKScene = {...scene, positions};
  const reference = getDiggleReference(outside, shape);
  expect(reference.observed.included.length).toBeLessThan(
    getDiggleReference(scene, shape).observed.included.length
  );
  expectDiggleMatches(await runDiggleTest(outside, shape), reference, shape);
});

it('GPUSpaceTimeDiggleTest reports NaN when fewer than two events are included', async () => {
  const scene: SpaceTimeKScene = {positions: [5, 4, 30, 30], times: [3, 4]};
  const actual = await runDiggleTest(scene, SHAPES[0]);
  expect(actual.statistics.every(Number.isNaN)).toBe(true);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]).toBeNaN();
});
