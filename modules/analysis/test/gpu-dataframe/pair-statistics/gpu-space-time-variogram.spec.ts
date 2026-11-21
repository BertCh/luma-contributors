// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {AnalysisRig} from '../../gpu-spatial-analysis/catchment-accessibility/rig';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
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

it('excluded positional outliers cannot change retained semivariances (R19 scale eligibility)', async () => {
  // Six adjacent pairs have differences 1: each semivariance is exactly 1/2. The included
  // distant sentinel sets the variance floor but forms no pair. Eight outside rows must
  // enter neither the moments nor the robust IQR used for the fixed-point unit.
  for (const excludedCount of [0, 8]) {
    const values = [0, 1, 2, 3, 4, 5, 6, 1e6, ...Array<number>(excludedCount).fill(1e9)];
    const scene: Scene = {
      positions: [
        ...Array.from({length: 7}, (_, row) => [row, 0]).flat(),
        90,
        90,
        ...Array.from({length: excludedCount}, () => [200, 200]).flat()
      ],
      values,
      secondValues: values,
      times: values.map(() => 0),
      mask: values.map(() => 1)
    };
    const actual = await runVariogram(scene, Float32Array.from(scene.times), false, {
      bounds: [0, 0, 100, 100],
      maximumDistance: 1.1,
      maximumTimeLag: 1,
      lagCount: 1,
      temporalLagCount: 1
    });
    expect(actual.counts).toEqual([0, 0, 6, 0]);
    expect(actual.status[2]).toBe(0);
    expect(Number.isFinite(actual.semivariances[2])).toBe(true);
    expect(actual.semivariances[2]).toBeCloseTo(0.5, 5);
  }
});

async function runVariogram(
  scene: Scene,
  relativeTimes: Float32Array,
  useSecondValues: boolean,
  shape = DEFAULT_SHAPE,
  positionPrefix: readonly number[] = []
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
      format === 'uint32'
        ? Uint32Array.from(data)
        : Float32Array.from(name === 'positions' ? [...positionPrefix, ...Array.from(data)] : data)
    );
    buffers.push(buffer);
    const view = importGraphBuffer(graph, name, buffer, format);
    return graph.createDataView(view.buffer, {
      format,
      byteOffset: name === 'positions' ? positionPrefix.length * 4 : 0,
      length: data.length / (format === 'float32x2' ? 2 : 1)
    });
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

/** Shape of the complete-case fixtures: one positive spatial and temporal class, bin 3 = (1, 1). */
const COMPLETE_CASE_SHAPE = {
  bounds: [0, 0, 1, 1] as const,
  maximumDistance: 2,
  maximumTimeLag: 2,
  lagCount: 1,
  temporalLagCount: 1
};

function expectOnlyBinThree(
  actual: Awaited<ReturnType<typeof runVariogram>>,
  semivariance: number,
  count: number
) {
  expect(Number.isFinite(actual.semivariances[3])).toBe(true);
  expect(actual.semivariances[3]).toBeCloseTo(semivariance, 5);
  expect(actual.counts).toEqual([0, 0, 0, count]);
  expect(actual.status).toEqual([1, 1, 1, 0]);
}

it.each([
  false,
  true
])('centers values on the complete-case rows only (DPC-112, second values: %s)', async useSecondValues => {
  // Rows 0 and 1 form the only complete pair: (2 - 0)^2 / 2 = 2. Row 2 has a NaN time, so it is in
  // no pair; centering on all value-finite rows put the center at about 3.3e8, where float32
  // rounds both retained values to the same number and the semivariance to 0.
  const scene: Scene = {
    positions: [0, 0, 1, 0, 0, 1],
    values: [0, 2, 1e9],
    secondValues: [0, 2, 1e9],
    times: [0, 1, NaN],
    mask: [1, 1, 1]
  };
  const actual = await runVariogram(
    scene,
    Float32Array.from(scene.times),
    useSecondValues,
    COMPLETE_CASE_SHAPE
  );
  expectOnlyBinThree(actual, 2, 1);
  // Masking out the already ineligible row leaves every retained bin unchanged.
  const masked = await runVariogram(
    {...scene, mask: [1, 1, 0]},
    Float32Array.from(scene.times),
    useSecondValues,
    COMPLETE_CASE_SHAPE
  );
  expect(masked).toEqual(actual);
});

it('ignores an excluded high value near retained values of about 1 (DPC-112)', async () => {
  // Expectation from float64 NumPy over the retained rows only
  // (fixes/fix-pair-statistics/reference.py): values [1, 1.5, 1.25], pairs (0,1) (0,2) (1,2),
  // terms [0.125, 0.03125, 0.03125], mean 0.0625. All three pairs have distance in (0, 2] and time
  // lag in (0, 2], so they fall in bin 3.
  const positions = [0, 0, 1, 0, 0, 1, 1, 1];
  // Row 3 is excluded by a NaN time (first route) or a NaN second value (cross route).
  const byTime: Scene = {
    positions,
    values: [1, 1.5, 1.25, 1e9],
    secondValues: [1, 1.5, 1.25, 1e9],
    times: [0, 1, 0.5, NaN],
    mask: [1, 1, 1, 1]
  };
  expectOnlyBinThree(
    await runVariogram(byTime, Float32Array.from(byTime.times), false, COMPLETE_CASE_SHAPE),
    0.0625,
    3
  );
  const bySecondValue: Scene = {
    ...byTime,
    secondValues: [1, 1.5, 1.25, NaN],
    times: [0, 1, 0.5, 1.5]
  };
  expectOnlyBinThree(
    await runVariogram(
      bySecondValue,
      Float32Array.from(bySecondValue.times),
      true,
      COMPLETE_CASE_SHAPE
    ),
    0.0625,
    3
  );
});

it('reports a pair count above 2^32 - 1 as status 3 (DPC-113)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  recordWGSLKernelNodes(rig.graph);
  const rows = 3;
  const bins = 4;
  const semivariances = rig.output('float32', bins);
  const pairCounts = rig.output('uint32', bins);
  const status = rig.output('uint32', bins);
  try {
    const nodes = new GPUSpaceTimeVariogram({
      id: 'wide-count',
      positions: createTransientView(rig.graph, 'positions', 'float32x2', rows),
      values: createTransientView(rig.graph, 'values', 'float32', rows),
      times: createTransientView(rig.graph, 'times', 'float32', rows),
      parameters: rig.input(
        getGPUSpaceTimeVariogramParameterValues(COMPLETE_CASE_SHAPE),
        'float32'
      ),
      gridSize: [1, 1],
      lagCount: COMPLETE_CASE_SHAPE.lagCount,
      temporalLagCount: COMPLETE_CASE_SHAPE.temporalLagCount,
      output: {semivariances, pairCounts, status}
    }).getCommandNodes(rig.graph);
    const finishNodes = nodes.filter(node => node.workload?.variant === 'finish');
    expect(finishNodes).toHaveLength(1);
    const record = getWGSLKernelNodeRecord(rig.graph, finishNodes[0].id)!;
    const accumulators = record.bindings.find(binding => binding.name === 'accumulators')!.view;
    const scale = record.bindings.find(binding => binding.name === 'scaleView')!.view;
    // Five channels of two words per bin, then the accumulation-overflow word (left clear).
    expect(accumulators.length).toBe(bins * 5 * 2 + 1);
    // Bin 0: 92683 * 92682 / 2 = 4295022903 = 2^32 + 55607 unordered pairs (high word 1).
    // Bin 1: the exact control 0xffffffff (high word 0). Bin 2: one pair. Bin 3: empty.
    const inject = createWGSLKernelNode(rig.graph, {
      id: 'synthetic-wide-count',
      operation: 'PairCountOverflowTest',
      invocationCount: accumulators.length,
      bindings: [
        {name: 'words', view: accumulators, type: 'u32', access: 'read_write'},
        {name: 'scaleWord', view: scale, type: 'f32', access: 'read_write'}
      ],
      body: `var word = 0u;
if (index == 0u) { word = 55607u; }
if (index == 1u) { word = 1u; }
if (index == 10u) { word = 0xffffffffu; }
if (index == 20u) { word = 1u; }
words[wordsOffset + index] = word;
if (index == 0u) { scaleWord[scaleWordOffset] = 1.0; }`
    });
    rig.run({getCommandNodes: () => [inject, finishNodes[0]]});
    expect(await rig.readUint(pairCounts)).toEqual([0xffffffff, 0xffffffff, 1, 0]);
    expect(await rig.readUint(status)).toEqual([3, 0, 0, 1]);
    const values = await rig.readFloat(semivariances);
    expect(values.slice(0, 3)).toEqual([0, 0, 0]);
    expect(values[3]).toBeNaN();
  } finally {
    rig.destroy();
  }
});

it('PCK-01 retains semivariance two with an eight-byte positions prefix', async () => {
  // Only rows 0 and 1 form a complete pair: (2 - 0)^2 / 2 = 2.
  const scene: Scene = {
    positions: [0, 0, 1, 0, 200, 200],
    values: [0, 2, 1e9],
    secondValues: [0, 2, 1e9],
    times: [0, 1, NaN],
    mask: [1, 1, 1]
  };
  const actual = await runVariogram(
    scene,
    Float32Array.from(scene.times),
    false,
    {
      bounds: [0, 0, 10, 10],
      maximumDistance: 2,
      maximumTimeLag: 2,
      lagCount: 1,
      temporalLagCount: 1
    },
    [100, 100]
  );
  expect(actual.counts).toEqual([0, 0, 0, 1]);
  expect(actual.status[3]).toBe(0);
  expect(Number.isFinite(actual.semivariances[3])).toBe(true);
  expect(actual.semivariances[3]).toBeCloseTo(2, 5);
});
