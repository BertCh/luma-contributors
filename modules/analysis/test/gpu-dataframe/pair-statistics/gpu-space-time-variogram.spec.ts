// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getGPUSpaceTimeVariogramParameterValues,
  GPUSpaceTimeVariogram
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {getGPUSpaceTimeRelativeTimes} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';

const BOUNDS = [0, 0, 12, 6] as const;
const MAXIMUM_DISTANCE = 7;
const MAXIMUM_TIME_LAG = 55;
const LAG_COUNT = 4;
const TEMPORAL_LAG_COUNT = 5;

type Scene = {
  positions: number[];
  values: number[];
  secondValues: number[];
  /** Float64 times in any origin; the GPU receives them after origin subtraction. */
  times: number[];
  mask: number[];
};

/** 12 stations on a 4 x 3 lattice, five times each, plus a coincident duplicate and bad rows. */
function createScene(timeOrigin: number): Scene {
  const scene: Scene = {positions: [], values: [], secondValues: [], times: [], mask: []};
  let state = 12345;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  const addRow = (x: number, y: number, time: number) => {
    scene.positions.push(x, y);
    scene.values.push(Math.round(random() * 64) / 8);
    scene.secondValues.push(Math.round(random() * 64) / 8 - 2);
    scene.times.push(timeOrigin + time);
    scene.mask.push(1);
  };
  for (let station = 0; station < 12; station++)
    for (let step = 0; step < 5; step++)
      addRow((station % 4) * 3, Math.floor(station / 4) * 3, step * 10);
  // Coincident with row 0 at the same time: fills the (0, 0) zero-lag bin.
  addRow(0, 0, 0);
  // Excluded rows: masked, non-finite value, non-finite time, outside the bounds.
  addRow(3, 3, 20);
  scene.mask[scene.mask.length - 1] = 0;
  addRow(6, 3, 20);
  scene.values[scene.values.length - 1] = NaN;
  addRow(9, 3, 20);
  scene.times[scene.times.length - 1] = NaN;
  addRow(40, 40, 20);
  return scene;
}

/** Brute-force double-precision reference with the documented classes. */
function getReference(scene: Scene, timeOrigin: number, useSecondValues: boolean) {
  const bins = (LAG_COUNT + 1) * (TEMPORAL_LAG_COUNT + 1);
  const sums = new Array<number>(bins).fill(0);
  const counts = new Array<number>(bins).fill(0);
  const rows = scene.values.length;
  const included = (row: number) =>
    scene.mask[row] !== 0 &&
    Number.isFinite(scene.values[row]) &&
    Number.isFinite(scene.secondValues[row]) &&
    Number.isFinite(scene.times[row]) &&
    scene.positions[row * 2] >= BOUNDS[0] &&
    scene.positions[row * 2] <= BOUNDS[2] &&
    scene.positions[row * 2 + 1] >= BOUNDS[1] &&
    scene.positions[row * 2 + 1] <= BOUNDS[3];
  for (let first = 0; first < rows; first++) {
    for (let second = first + 1; second < rows; second++) {
      if (!included(first) || !included(second)) continue;
      const deltaX = scene.positions[first * 2] - scene.positions[second * 2];
      const deltaY = scene.positions[first * 2 + 1] - scene.positions[second * 2 + 1];
      const distance = Math.hypot(deltaX, deltaY);
      const timeLag = Math.abs(scene.times[first] - scene.times[second]);
      if (distance > MAXIMUM_DISTANCE || timeLag > MAXIMUM_TIME_LAG) continue;
      const spatialClass =
        deltaX === 0 && deltaY === 0
          ? 0
          : 1 + Math.min(Math.floor((distance / MAXIMUM_DISTANCE) * LAG_COUNT), LAG_COUNT - 1);
      const temporalClass =
        timeLag === 0
          ? 0
          : 1 +
            Math.min(
              Math.floor((timeLag / MAXIMUM_TIME_LAG) * TEMPORAL_LAG_COUNT),
              TEMPORAL_LAG_COUNT - 1
            );
      const bin = spatialClass * (TEMPORAL_LAG_COUNT + 1) + temporalClass;
      const other = useSecondValues ? scene.secondValues : scene.values;
      sums[bin] +=
        0.5 * (scene.values[first] - scene.values[second]) * (other[first] - other[second]);
      counts[bin]++;
    }
  }
  void timeOrigin;
  return {
    semivariances: sums.map((sum, bin) => (counts[bin] > 0 ? sum / counts[bin] : NaN)),
    counts
  };
}

const DEFAULT_SHAPE = {
  bounds: BOUNDS as readonly [number, number, number, number],
  maximumDistance: MAXIMUM_DISTANCE,
  maximumTimeLag: MAXIMUM_TIME_LAG,
  lagCount: LAG_COUNT,
  temporalLagCount: TEMPORAL_LAG_COUNT
};

async function runVariogram(
  scene: Scene,
  relativeTimes: Float32Array,
  useSecondValues: boolean,
  shape = DEFAULT_SHAPE
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
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
  const bins = (shape.lagCount + 1) * (shape.temporalLagCount + 1);
  const outputs = {
    semivariances: createOutputBuffer(device, bins),
    pairCounts: createOutputBuffer(device, bins),
    status: createOutputBuffer(device, bins)
  };
  buffers.push(...Object.values(outputs));
  graph.add(
    new GPUSpaceTimeVariogram({
      positions: input('positions', 'float32x2', scene.positions),
      values: input('values', 'float32', scene.values),
      secondValues: useSecondValues
        ? input('secondValues', 'float32', scene.secondValues)
        : undefined,
      times: input('times', 'float32', relativeTimes),
      mask: input('mask', 'uint32', scene.mask),
      parameters: input('parameters', 'float32', getGPUSpaceTimeVariogramParameterValues(shape)),
      gridSize: [4, 4],
      lagCount: shape.lagCount,
      temporalLagCount: shape.temporalLagCount,
      output: {
        semivariances: importGraphBuffer(
          graph,
          'out-semivariances',
          outputs.semivariances,
          'float32',
          bins
        ),
        pairCounts: importGraphBuffer(graph, 'out-counts', outputs.pairCounts, 'uint32', bins),
        status: importGraphBuffer(graph, 'out-status', outputs.status, 'uint32', bins)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    semivariances: Array.from(await readFloat32(outputs.semivariances, bins)),
    counts: Array.from(await readUint32(outputs.pairCounts, bins)),
    status: Array.from(await readUint32(outputs.status, bins))
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

function expectMatchesReference(
  actual: Awaited<ReturnType<typeof runVariogram>>,
  expected: ReturnType<typeof getReference>
) {
  expect(actual.counts).toEqual(expected.counts);
  expected.semivariances.forEach((value, bin) => {
    if (Number.isNaN(value)) {
      expect(actual.semivariances[bin], `bin ${bin}`).toBeNaN();
      expect(actual.status[bin]).toBe(1);
    } else {
      expect(Math.abs(actual.semivariances[bin] - value), `bin ${bin}`).toBeLessThan(2e-3);
      expect(actual.status[bin]).toBe(0);
    }
  });
}

it.each([
  false,
  true
])('matches the brute-force reference, including zero-lag classes (second values: %s)', async useSecondValues => {
  const scene = createScene(0);
  const expected = getReference(scene, 0, useSecondValues);
  const actual = await runVariogram(scene, Float32Array.from(scene.times), useSecondValues);
  expectMatchesReference(actual, expected);
  // Coincident points at the same time fill only the (0, 0) bin (time lags are 10 to 40, so the
  // last temporal class is empty); the first row is purely
  // temporal and the first column purely spatial.
  expect(actual.counts[0]).toBe(1);
  expect(actual.counts.slice(1, TEMPORAL_LAG_COUNT).every(count => count > 0)).toBe(true);
  expect(actual.counts[2 * (TEMPORAL_LAG_COUNT + 1)]).toBeGreaterThan(0);
});

it('isolates the purely temporal variogram for epoch-second times', async () => {
  const epochOrigin = 1_700_000_000;
  const scene = createScene(epochOrigin);
  const expected = getReference(scene, epochOrigin, false);
  // Raw epoch seconds in float32 are spaced 128 s apart: every 10 s step would collapse.
  expect(new Set(Float32Array.from(scene.times.filter(Number.isFinite))).size).toBeLessThan(3);
  const relative = getGPUSpaceTimeRelativeTimes(
    scene.times.map(time => (Number.isFinite(time) ? time : epochOrigin)),
    epochOrigin
  );
  scene.times.forEach((time, row) => {
    if (!Number.isFinite(time)) relative.times[row] = NaN;
  });
  const actual = await runVariogram(scene, relative.times, false);
  expectMatchesReference(actual, expected);
  const temporalRow = actual.counts.slice(0, TEMPORAL_LAG_COUNT + 1);
  expect(temporalRow[0]).toBe(1);
  expect(temporalRow.slice(1, TEMPORAL_LAG_COUNT).every(count => count > 0)).toBe(true);
});

it('keeps the previous three-point values, moved to the zero-lag class layout', async () => {
  // The former fixture: bins [4, 5, 8] of the old 3 x 3 grid held -4, -1 and -1. With a zero-lag
  // class on each axis the same pairs sit one class further in, in a 4 x 4 grid.
  const scene: Scene = {
    positions: [0, 0, 1, 0, 2, 0],
    values: [1, 3, 2],
    secondValues: [4, 0, 2],
    times: [0, 1, 3],
    mask: [1, 1, 1]
  };
  const shape = {
    bounds: [0, 0, 2, 0] as const,
    maximumDistance: 3,
    maximumTimeLag: 3,
    lagCount: 3,
    temporalLagCount: 3
  };
  const actual = await runVariogram(scene, Float32Array.from(scene.times), true, shape);
  const expectedValues = new Array<number>(16).fill(NaN);
  expectedValues[10] = -4;
  expectedValues[11] = -1;
  expectedValues[15] = -1;
  expectedValues.forEach((value, bin) =>
    Number.isNaN(value)
      ? expect(actual.semivariances[bin]).toBeNaN()
      : expect(actual.semivariances[bin]).toBeCloseTo(value, 3)
  );
  expect(actual.counts).toEqual(expectedValues.map(value => (Number.isNaN(value) ? 0 : 1)));
});
