// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * `GPUSpaceTimeDiggleTest` against the port of the splancs source (`stkhat`, `stmctest`;
 * `test/oracles/space-time-diggle-splancs-port.py`, a port and not a run of R). Math review
 * 2026-11-10, dataframe-10: the statistic is the plain sum of `D` over the classes, as splancs
 * `stmctest` and surveillance `stKtest` compute it. Review of the closing pass, frc-fields-01:
 * the class follows the splancs rules at exact ties, which whole-day times hit all the time, while
 * `GPUSpaceTimeKFunction` keeps the rules of its own reference, stpp.
 */

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
  GPUSpaceTimeKFunction
} from '../../../src/dataframe-analysis/pair-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getSplancsIntegerTimeReferenceOptions,
  SPLANCS_FIXTURE,
  type SplancsAnalysis
} from './space-time-diggle-splancs-fixture';
import {getSpaceTimeKReference} from './space-time-k-function-reference';

type Scene = {
  positions: number[];
  times: number[];
  timeLimits: readonly [number, number];
  maximumDistance: number;
  maximumTimeLag: number;
  distanceClassCount: number;
  timeClassCount: number;
  permutationCount: number;
  seed: number;
};

const FIXTURE_SCENE = {
  positions: SPLANCS_FIXTURE.positions,
  timeLimits: SPLANCS_FIXTURE.timeLimits,
  maximumDistance: SPLANCS_FIXTURE.maximumDistance,
  maximumTimeLag: SPLANCS_FIXTURE.maximumTimeLag,
  distanceClassCount: SPLANCS_FIXTURE.distanceClassCount,
  timeClassCount: SPLANCS_FIXTURE.timeClassCount
};

async function runDiggleTest(scene: Scene) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'space-time-diggle-splancs-graph'});
  const buffers: Buffer[] = [];
  const bins = scene.distanceClassCount * scene.timeClassCount;
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
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32', length)};
  };
  const d = output('d', bins);
  const k = output('k', bins);
  const spatialK = output('spatial-k', scene.distanceClassCount);
  const temporalK = output('temporal-k', scene.timeClassCount);
  const statistics = output('statistics', scene.permutationCount + 1);
  const summary = output('summary', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH);
  graph.add(
    new GPUSpaceTimeDiggleTest({
      positions: input('positions', 'float32x2', scene.positions),
      times: input('times', 'float32', scene.times),
      parameters: input(
        'parameters',
        'float32',
        getGPUSpaceTimeKFunctionParameterValues({
          bounds: SPLANCS_FIXTURE.bounds,
          maximumDistance: scene.maximumDistance,
          maximumTimeLag: scene.maximumTimeLag,
          timeWindow: scene.timeLimits
        })
      ),
      randomParameters: input(
        'random-parameters',
        'uint32',
        getGPUSpaceTimeDiggleRandomParameterValues(scene.seed)
      ),
      gridSize: [4, 4],
      distanceClassCount: scene.distanceClassCount,
      timeClassCount: scene.timeClassCount,
      edgeCorrection: 'isotropic',
      permutationCount: scene.permutationCount,
      output: {
        d: d.view,
        k: k.view,
        spatialK: spatialK.view,
        temporalK: temporalK.view,
        statistics: statistics.view,
        summary: summary.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      d: await readFloat32(d.buffer, bins),
      k: await readFloat32(k.buffer, bins),
      spatialK: await readFloat32(spatialK.buffer, scene.distanceClassCount),
      temporalK: await readFloat32(temporalK.buffer, scene.timeClassCount),
      statistics: await readFloat32(statistics.buffer, scene.permutationCount + 1),
      summary: await readFloat32(summary.buffer, GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/** Float32 kernels with 2^-16 weight units against the binary64 port. */
function expectMatchesPort(
  actual: Awaited<ReturnType<typeof runDiggleTest>>,
  port: SplancsAnalysis,
  label: string
): void {
  const scale = Math.max(...port.k.map(Math.abs));
  const expectClose = (values: number[], expected: number[], name: string, tolerance: number) => {
    expect(values.length).toBe(expected.length);
    values.forEach((value, index) => {
      expect(
        Math.abs(value - expected[index]),
        `${label} ${name} ${index}: got ${value}, splancs port ${expected[index]}`
      ).toBeLessThanOrEqual(tolerance);
    });
  };
  // Measured agreement is within 2e-6 of the largest K; the bounds leave a factor of five.
  expectClose(actual.spatialK, port.spatialK, 'ks', 1e-5 * Math.max(...port.spatialK));
  expectClose(actual.temporalK, port.temporalK, 'kt', 1e-5 * Math.max(...port.temporalK));
  expectClose(actual.k, port.k, 'kst', 1e-5 * scale);
  expectClose(actual.d, port.d, 'D', 2e-5 * scale);
  // The statistic of stmctest: the sum of D over the grid of classes.
  expect(
    Math.abs(actual.statistics[0] - port.statistic),
    `${label} statistic: got ${actual.statistics[0]}, splancs port ${port.statistic}`
  ).toBeLessThanOrEqual(1e-5 * scale * port.d.length);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.observed]).toBe(actual.statistics[0]);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.usedClasses]).toBe(port.d.length);
}

it('GPUSpaceTimeDiggleTest returns the stkhat functions and the stmctest statistic of the splancs port', async () => {
  const actual = await runDiggleTest({
    ...FIXTURE_SCENE,
    times: SPLANCS_FIXTURE.times,
    permutationCount: 3,
    seed: 5
  });
  expectMatchesPort(actual, SPLANCS_FIXTURE.observed, 'observed');
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.includedCount]).toBe(
    SPLANCS_FIXTURE.times.length
  );
});

it('GPUSpaceTimeDiggleTest matches the splancs port for six relabellings of the times', async () => {
  for (const [index, relabelling] of SPLANCS_FIXTURE.relabellings.entries()) {
    const actual = await runDiggleTest({
      ...FIXTURE_SCENE,
      times: relabelling.times,
      permutationCount: 1,
      seed: 1
    });
    expectMatchesPort(actual, relabelling, `relabelling ${index}`);
  }
});

it('GPUSpaceTimeDiggleTest draws a null distribution like the stmctest simulations of the port', async () => {
  const port = SPLANCS_FIXTURE.monteCarlo;
  const permutationCount = port.permutationCount;
  const actual = await runDiggleTest({
    ...FIXTURE_SCENE,
    times: SPLANCS_FIXTURE.times,
    permutationCount,
    seed: 2026
  });
  const permuted = actual.statistics.slice(1);
  const mean = permuted.reduce((sum, value) => sum + value, 0) / permutationCount;
  const standardDeviation = Math.sqrt(
    permuted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (permutationCount - 1)
  );
  // Two independent samples of 199 relabellings (numpy's and the kernel's shuffle): the standard
  // deviations agree to sampling error (about 7% at one sigma), the means to 0.4 deviations.
  expect(Math.abs(standardDeviation / port.standardDeviation - 1)).toBeLessThan(0.25);
  expect(Math.abs(mean - port.mean)).toBeLessThan(0.4 * port.standardDeviation);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.permutationMean]).toBeCloseTo(mean, 2);
  // The port ranks the observed statistic third of 200 (p = 0.02); the kernel's own sample of
  // relabellings must put it in the same tail.
  expect(port.pseudoPGreater).toBe(0.02);
  const greater = permuted.filter(value => value >= actual.statistics[0]).length;
  expect(greater).toBeLessThanOrEqual(10);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.greaterCount]).toBe(greater);
  expect(actual.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]).toBeCloseTo(
    (greater + 1) / (permutationCount + 1),
    6
  );
});

it('GPUSpaceTimeDiggleTest matches the splancs port on whole-day times, where the tie rules act on many pairs', async () => {
  const scene = SPLANCS_FIXTURE.integerTimes;
  for (const entry of scene.cases) {
    for (const [labellingIndex, port] of entry.labellings.entries()) {
      const actual = await runDiggleTest({
        positions: scene.positions,
        times: scene.labellingTimes[labellingIndex],
        timeLimits: scene.timeLimits,
        maximumDistance: scene.maximumDistance,
        maximumTimeLag: entry.maximumTimeLag,
        distanceClassCount: scene.distanceClassCount,
        timeClassCount: entry.timeClassCount,
        permutationCount: 1,
        seed: 1
      });
      expectMatchesPort(actual, port, `lag ${entry.maximumTimeLag} labelling ${labellingIndex}`);
    }
  }
});

it('GPUSpaceTimeDiggleTest applies the splancs tie rules to its own permutations of whole-day times', async () => {
  const scene = SPLANCS_FIXTURE.integerTimes;
  const permutationCount = 12;
  const seed = 31;
  const rows = scene.labellingTimes[0].length;
  const seedWords = getGPUSpaceTimeDiggleRandomParameterValues(seed);
  for (const [caseIndex, entry] of scene.cases.entries()) {
    const actual = await runDiggleTest({
      positions: scene.positions,
      times: scene.labellingTimes[0],
      timeLimits: scene.timeLimits,
      maximumDistance: scene.maximumDistance,
      maximumTimeLag: entry.maximumTimeLag,
      distanceClassCount: scene.distanceClassCount,
      timeClassCount: entry.timeClassCount,
      permutationCount,
      seed
    });
    // The binary64 reference with the splancs rules (equal to the port on this scene, see the
    // node spec) on the kernel's own shuffles: every event is included, so rank and row agree.
    const options = getSplancsIntegerTimeReferenceOptions(caseIndex, 0);
    const scale = Math.max(...entry.labellings[0].k.map(Math.abs));
    for (let permutation = 0; permutation < permutationCount; permutation++) {
      const order = new Uint32Array(rows);
      fillShufflePermutation(
        order,
        rows,
        [seedWords[0], seedWords[1]],
        permutation + 1,
        PHILOX_PURPOSES.spaceTimeDiggleShuffle.id
      );
      const times = Array.from(order, row => scene.labellingTimes[0][row]);
      const reference = getSpaceTimeKReference({
        ...options,
        scene: {positions: scene.positions, times}
      });
      const expected = reference.d.reduce((sum, value) => sum + value, 0);
      expect(
        Math.abs(actual.statistics[permutation + 1] - expected),
        `lag ${entry.maximumTimeLag} permutation ${permutation}: got ${actual.statistics[permutation + 1]}, reference ${expected}`
      ).toBeLessThanOrEqual(1e-5 * scale * reference.d.length);
    }
  }
});

/** Runs `GPUSpaceTimeKFunction` (isotropic) on one class of distance and lag. */
async function runKFunction(scene: {
  positions: number[];
  times: number[];
  timeLimits: readonly [number, number];
  maximumDistance: number;
  maximumTimeLag: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'space-time-k-tie-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const input = <Format extends 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    data: number[]
  ) =>
    importGraphBuffer(
      graph,
      name,
      track(createInputBuffer(device, Float32Array.from(data))),
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  const outputs = {
    k: track(createOutputBuffer(device, 1)),
    spatialK: track(createOutputBuffer(device, 1)),
    temporalK: track(createOutputBuffer(device, 1))
  };
  graph.add(
    new GPUSpaceTimeKFunction({
      positions: input('positions', 'float32x2', scene.positions),
      times: input('times', 'float32', scene.times),
      parameters: input(
        'parameters',
        'float32',
        Array.from(
          getGPUSpaceTimeKFunctionParameterValues({
            bounds: SPLANCS_FIXTURE.bounds,
            maximumDistance: scene.maximumDistance,
            maximumTimeLag: scene.maximumTimeLag,
            timeWindow: scene.timeLimits
          })
        )
      ),
      gridSize: [4, 4],
      distanceClassCount: 1,
      timeClassCount: 1,
      output: {
        k: importGraphBuffer(graph, 'k', outputs.k, 'float32', 1),
        spatialK: importGraphBuffer(graph, 'spatial-k', outputs.spatialK, 'float32', 1),
        temporalK: importGraphBuffer(graph, 'temporal-k', outputs.temporalK, 'float32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      k: await readFloat32(outputs.k, 1),
      spatialK: await readFloat32(outputs.spatialK, 1),
      temporalK: await readFloat32(outputs.temporalK, 1)
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

it('at exact ties GPUSpaceTimeDiggleTest follows splancs and GPUSpaceTimeKFunction follows stpp', async () => {
  const tie = SPLANCS_FIXTURE.temporalTie;
  const scene = {
    positions: tie.positions,
    times: tie.times,
    timeLimits: tie.timeLimits,
    maximumDistance: tie.distanceBreaks[0],
    maximumTimeLag: tie.timeBreaks[0]
  };
  const single = {distanceClassCount: 1, timeClassCount: 1, permutationCount: 1, seed: 1};
  // A lag that exactly reaches a time limit. Two events at t = 3 and 6 in [0, 10], lag 3.
  // splancs stkhat.f: (ti - tlow) .le. tj, so the event at t = 3 has weight 2:
  // kt = 10 / 2 * (2 + 1) = 15, kst = 80 * 10 / 2 * 3 = 1200 (the port, by hand as well).
  expect(tie.temporalK).toEqual([15]);
  const diggle = await runDiggleTest({...scene, ...single});
  expect(diggle.temporalK).toEqual([15]);
  expect(diggle.spatialK).toEqual([80]);
  expect(diggle.k).toEqual([1200]);
  expect(diggle.d).toEqual([0]);
  // stpp stikfunction.f: vij = 1 when binf .ge. binft and bsup .le. bsupt (closed window).
  const kFunction = await runKFunction(scene);
  expect(kFunction.temporalK).toEqual([10]);
  expect(kFunction.spatialK).toEqual([80]);
  expect(kFunction.k).toEqual([800]);

  // A lag exactly at the largest lag: dropped by splancs (tj .ge. tmax), counted by stpp
  // (tij .le. t). By hand: kt = 0 and kst = 0 against 10 and 800.
  const atLargestLag = {...scene, maximumTimeLag: 3};
  const droppedLag = await runDiggleTest({...atLargestLag, ...single});
  expect(droppedLag.temporalK).toEqual([0]);
  expect(droppedLag.k).toEqual([0]);
  expect(droppedLag.spatialK).toEqual([80]);
  expect((await runKFunction(atLargestLag)).temporalK).toEqual([10]);

  // A distance exactly at the largest distance (the events are 1 apart): dropped by splancs
  // (sj .ge. smax) from ks and kst, counted by stpp (hij .le. s).
  const atLargestDistance = {...scene, maximumDistance: 1};
  const droppedDistance = await runDiggleTest({...atLargestDistance, ...single});
  expect(droppedDistance.spatialK).toEqual([0]);
  expect(droppedDistance.k).toEqual([0]);
  expect(droppedDistance.temporalK).toEqual([15]);
  const countedDistance = await runKFunction(atLargestDistance);
  expect(countedDistance.spatialK).toEqual([80]);
  expect(countedDistance.k).toEqual([800]);
});
