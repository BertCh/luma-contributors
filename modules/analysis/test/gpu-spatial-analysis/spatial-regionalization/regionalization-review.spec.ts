// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';
import {createReviewR1} from './regionalization-review-fixtures';
import {createReviewR2} from './regionalization-review-fixtures';
import {createReviewR3} from './regionalization-review-fixtures';
import {createReviewR4} from './regionalization-review-fixtures';
import {CPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import type {Device} from '@luma.gl/core';
import {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPUAZPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {createAuditMaxPInput, createAuditMaxPProps} from './regionalization-audit-max-p-fixtures';
import {createAuditAZPInput, createAuditAZPProps} from './regionalization-audit-azp-fixtures';

function createRepeatedReviewGraph(
  device: Device,
  maxP: boolean,
  lossName?: 'overflow' | 'candidateOverflow'
) {
  const rig = new WeightsRig(device);
  const maxPInput = createAuditMaxPInput();
  const azpInput = createAuditAZPInput();
  const input = maxP ? maxPInput : azpInput;
  const rows = input.offsets.length - 1;
  const labels = rig.output('uint32', rows);
  const objective = rig.output('float32', 1);
  const status = rig.output('uint32', maxP ? 6 : 4);
  const flag = lossName ? rig.input(Uint32Array.of(1), 'uint32', 1) : undefined;
  const weights = {
    ...rig.uploadWeights({
      offsets: Array.from(input.offsets),
      neighbors: Array.from(input.neighbors),
      weights: Array.from(input.weights),
      distances: []
    }),
    ...(flag ? {status: {[lossName!]: flag}} : {})
  };
  const common = {
    weights,
    values: rig.input(input.values, 'float32', input.values.length),
    extensiveValues: rig.input(input.extensiveValues!, 'float32', rows),
    parameters: rig.input(input.parameters, 'uint32', input.parameters.length),
    output: {labels: labels.view, objective: objective.view, status: status.view}
  };
  rig.graph.add(
    maxP
      ? new GPUMaxPRegions({
          ...createAuditMaxPProps(),
          ...common,
          threshold: rig.input(maxPInput.threshold, 'float32', 1)
        })
      : new GPUAZPRegions({
          ...createAuditAZPProps(),
          ...common,
          initialLabels: rig.input(azpInput.initialLabels, 'uint32', rows),
          extensiveThreshold: rig.input(azpInput.extensiveThreshold!, 'float32', 1)
        })
  );
  const compiled = rig.graph.compile();
  return {
    rig,
    compiled,
    flag,
    rows,
    async read() {
      return {
        labels: await readUint32(labels.buffer, rows),
        objective: (await readFloat32(objective.buffer, 1))[0],
        status: await readUint32(status.buffer, maxP ? 6 : 4)
      };
    }
  };
}

async function evaluateReviewPartition(device: Device, values: number[], labels: number[]) {
  const rig = new WeightsRig(device);
  try {
    const summary = rig.output('float32', 9);
    const sizes = rig.output('uint32', 2);
    const means = rig.output('float32', 2);
    const within = rig.output('float32', 2);
    const invalid = rig.output('uint32', 1);
    rig.graph.add(
      new GPURegionPartitionEvaluation({
        values: rig.input(Float32Array.from(values), 'float32', values.length),
        labels: rig.input(Uint32Array.from(labels), 'uint32', labels.length),
        labelCapacity: 2,
        output: {
          summary: summary.view,
          regionSizes: sizes.view,
          regionMeans: means.view,
          regionWithinSumOfSquares: within.view,
          invalidRowCount: invalid.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      return {
        summary: await readFloat32(summary.buffer, 9),
        sizes: await readUint32(sizes.buffer, 2),
        means: await readFloat32(means.buffer, 2),
        within: await readFloat32(within.buffer, 2),
        invalid: await readUint32(invalid.buffer, 1)
      };
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
}

it('R1 publishes final feasibility and SSD on both Max-P backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createReviewR1();
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(fixture.input),
      operations: [new Operation(fixture.props)],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual(fixture.labels);
    expect(Array.from(result['status'])).toEqual(fixture.status);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    // Small raw inputs: float32 means/SSD permit 1e-5 absolute error.
    expect(Math.abs(result['objective'][0] - fixture.objective)).toBeLessThanOrEqual(1e-5);
  }
});

it('R3 keeps a connected zero-mass remainder together on both Max-P backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createReviewR3();
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(fixture.input),
      operations: [new Operation(fixture.props)],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual(fixture.labels);
    expect(Array.from(result['status'])).toEqual(fixture.status);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    expect(Math.abs(result['objective'][0] - 25.5)).toBeLessThanOrEqual(1e-5);
  }
});

it('R4 selects by every frozen grown-region score on both Max-P backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createReviewR4();
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(fixture.input),
      operations: [new Operation(fixture.props)],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual(fixture.labels);
    expect(Array.from(result['status'])).toEqual(fixture.status);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    expect(Math.abs(result['objective'][0] - 25.5)).toBeLessThanOrEqual(1e-5);
  }
});

it('R2 chooses the admissible AZP destination on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createReviewR2();
  for (const Operation of [GPUAZPRegionsOperation, CPUAZPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(fixture.input),
      operations: [new Operation(fixture.props)],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual(fixture.labels);
    expect(Array.from(result['status'])).toEqual(fixture.status);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    expect(Math.abs(result['objective'][0] - 0.5)).toBeLessThanOrEqual(1e-5);
  }
});

it('R5 preserves exact zero SSD and pseudo-F one for float32 constant regions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await evaluateReviewPartition(
    device,
    [0.9, 0.9, 0.9, 2, 2, 2],
    [0, 0, 0, 1, 1, 1]
  );
  expect(result.sizes).toEqual([3, 3]);
  expect(result.within).toEqual([0, 0]);
  expect(result.summary[1]).toBe(0);
  expect(result.summary[8]).toBe(1);
  expect(result.invalid).toEqual([0]);
  [Math.fround(0.9), 2].forEach((expected, index) => expect(result.means[index]).toBe(expected));
  // NumPy over the stored float32 input: total=between=1.8150000786781319.
  const expected = [2, 0, 1.8150000786781319, 1.8150000786781319, 3, 3, 0, 0, 1];
  expected.forEach((value, index) => {
    expect(Number.isFinite(result.summary[index])).toBe(true);
    expect(Math.abs(result.summary[index] - value)).toBeLessThanOrEqual(
      1e-5 * Math.max(1, Math.abs(value))
    );
  });
});

for (const exponent of [0, -24]) {
  it(`R12 preserves mixed-sign regional means and between SSD at scale 2^${exponent}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scale = 2 ** exponent;
    const result = await evaluateReviewPartition(
      device,
      [16777216, -16777215, -0.5, -0.5].map(value => value * scale),
      [0, 0, 1, 1]
    );
    expect(result.means).toEqual([0.5 * scale, -0.5 * scale]);
    expect(result.summary[2]).toBe(scale ** 2);
    expect(result.sizes).toEqual([2, 2]);
    expect(result.invalid).toEqual([0]);
    expect(result.within[1]).toBe(0);
    // Companion NumPy oracle over stored float32 inputs. Relative tolerance, with no absolute
    // floor that could hide an error in the scaled case.
    const within = 562949919866880.5 * scale ** 2;
    const total = 562949919866881.5 * scale ** 2;
    const expected = [2, within, scale ** 2, total, 2, 2, 0, 0, (2 * scale ** 2) / within];
    expected.forEach((value, index) => {
      expect(Number.isFinite(result.summary[index])).toBe(true);
      expect(Math.abs(result.summary[index] - value)).toBeLessThanOrEqual(1e-5 * Math.abs(value));
    });
    expect(Number.isFinite(result.within[0])).toBe(true);
    expect(Math.abs(result.within[0] - within)).toBeLessThanOrEqual(1e-5 * within);
  });
}

for (const values of [
  [1, 3, 2 ** 27, -(2 ** 27)],
  [2 ** 27, 1, -(2 ** 27), 3],
  [3, -(2 ** 27), 1, 2 ** 27]
]) {
  it(`W5-15 preserves a unit regional mean in order ${values}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await evaluateReviewPartition(device, values, [0, 0, 0, 0]);
    // Integer sum four divided by four, independently of association; label 1 is empty.
    expect(Array.from(result.sizes)).toEqual([4, 0]);
    expect(Number.isFinite(result.means[0])).toBe(true);
    expect(Number.isNaN(result.means[1])).toBe(true);
    expect(result.means[0]).toBe(1);
    expect(result.summary[2]).toBe(0);
  });
}

for (const magnitude of [1, 2 ** 50]) {
  it(`W5-16 gives permuted identical regions zero between SSD at ${magnitude}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await evaluateReviewPartition(
      device,
      [0, 0, magnitude, magnitude, 0, 0],
      [0, 0, 0, 1, 1, 1]
    );
    // Both regions and the table have mean magnitude/3, so N S_r - n_r S = 6M - 3 * 2M = 0.
    expect(Array.from(result.sizes)).toEqual([3, 3]);
    expect(result.means[0]).toBe(result.means[1]);
    expect(Number.isFinite(result.summary[2])).toBe(true);
    expect(result.summary[2]).toBe(0);
  });
}

for (const exponent of [-84, -80]) {
  it(`W5-15 retains a normal cancellation residual at scale 2^${exponent}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await evaluateReviewPartition(
      device,
      [2 ** exponent, 2 ** (exponent - 26), -(2 ** exponent)],
      [0, 0, 0]
    );
    // Exact sum 2^(exponent - 26), a normal float32, divided by three.
    const expected = Math.fround(2 ** (exponent - 26) / 3);
    expect(Number.isFinite(result.means[0])).toBe(true);
    expect(result.means[0]).toBe(expected);
  });
}

it('W5-16 reports zero between SSD for equal regional and grand means', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await evaluateReviewPartition(device, [16777216, 2, 8388609], [0, 0, 1]);
  // Both regional means and the grand mean are exactly 8388609.
  expect(result.means).toEqual([8388609, 8388609]);
  expect(result.summary[2]).toBe(0);
});

it('W5-15 preserves each constant regional mean across unrelated magnitude scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await evaluateReviewPartition(device, [2e38, 2e38, 1e-35, 1e-35], [0, 0, 1, 1]);
  expect(result.means).toEqual([Math.fround(2e38), Math.fround(1e-35)]);
  expect(result.within).toEqual([0, 0]);
});

it('R13 reports exactly zero within, between and total SSD for one constant region', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await evaluateReviewPartition(
    device,
    [1000000.125, 1000000.125, 1000000.125],
    [0, 0, 0]
  );
  expect(result.sizes).toEqual([3, 0]);
  expect(result.means[0]).toBe(1000000.125);
  expect(Number.isNaN(result.means[1])).toBe(true);
  expect(result.within).toEqual([0, 0]);
  expect(result.summary.slice(0, 8)).toEqual([1, 0, 0, 0, 3, 3, 0, 0]);
  expect(Number.isNaN(result.summary[8])).toBe(true);
  expect(result.invalid).toEqual([0]);
});

it('R15 keeps pseudo-F finite when an unscaled mean square would become subnormal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scale = 2 ** -63;
  const result = await evaluateReviewPartition(
    device,
    [scale, -scale, 0, 0, 4 * scale, 4 * scale, 4 * scale, 4 * scale],
    [0, 0, 0, 0, 1, 1, 1, 1]
  );
  expect(result.sizes).toEqual([4, 4]);
  expect(result.means).toEqual([0, 4 * scale]);
  expect(result.within).toEqual([2 * scale ** 2, 0]);
  expect(result.invalid).toEqual([0]);
  // All terms are exact binary multiples. The old unscaled division gives 95.99999 with
  // gradual underflow, or Infinity if its subnormal denominator is flushed to zero.
  expect(result.summary[8]).toBe(96);
  const expected = [2, 2 * scale ** 2, 32 * scale ** 2, 34 * scale ** 2, 4, 4, 0, 0, 96];
  expected.forEach((value, index) => {
    expect(Number.isFinite(result.summary[index])).toBe(true);
    expect(Math.abs(result.summary[index] - value)).toBeLessThanOrEqual(1e-5 * Math.abs(value));
  });
});

it('R8 repeats the same compiled Max-P graph with atomic region-presence flags', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createRepeatedReviewGraph(device, true);
  try {
    for (let submission = 0; submission < 3; submission++) {
      submitGraph(device, fixture.compiled, undefined);
      const result = await fixture.read();
      expect(result.labels).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
      expect(result.status).toEqual([4, 1, 0, 0, 0, 0]);
      expect(Number.isFinite(result.objective)).toBe(true);
      expect(Math.abs(result.objective - 28)).toBeLessThanOrEqual(1e-5);
    }
  } finally {
    fixture.compiled.destroy();
    fixture.rig.destroy();
  }
});

it('R9 keeps finite pseudo-F when degrees-of-freedom products would overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const amplitude = 3 * 2 ** 61;
  const result = await evaluateReviewPartition(
    device,
    [-amplitude, -amplitude / 2, amplitude, 1.5 * amplitude],
    [0, 0, 1, 1]
  );
  expect(result.sizes).toEqual([2, 2]);
  expect(result.invalid).toEqual([0]);
  expect(Number.isFinite(result.summary[8])).toBe(true);
  expect(Math.abs(result.summary[8] - 32)).toBeLessThanOrEqual(32e-5);
  [-0.75 * amplitude, 1.25 * amplitude].forEach((value, index) => {
    expect(result.means[index]).toBe(value);
    expect(Number.isFinite(result.within[index])).toBe(true);
    expect(Math.abs(result.within[index] - 5.981525981032121e36)).toBeLessThanOrEqual(
      5.981525981032121e31
    );
  });
  const expected = [
    2, 1.1963051962064243e37, 1.9140883139302789e38, 2.0337188335509213e38, 2, 2, 0, 0, 32
  ];
  expected.forEach((value, index) => {
    expect(Number.isFinite(result.summary[index])).toBe(true);
    expect(Math.abs(result.summary[index] - value)).toBeLessThanOrEqual(
      1e-5 * Math.max(1, Math.abs(value))
    );
  });
});

for (const lossName of ['overflow', 'candidateOverflow'] as const) {
  it(`R10 recovers and suppresses again on one compiled graph and one ${lossName} buffer`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const maxP of [false, true]) {
      const fixture = createRepeatedReviewGraph(device, maxP, lossName);
      try {
        for (const flagValue of [1, 0, 1]) {
          fixture.rig.bufferOf(fixture.flag!).write(Uint32Array.of(flagValue));
          submitGraph(device, fixture.compiled, undefined);
          const result = await fixture.read();
          if (flagValue) {
            expect(result.labels).toEqual(new Array(fixture.rows).fill(4294967295));
            expect(Number.isNaN(result.objective)).toBe(true);
            expect(result.status).toEqual(new Array(maxP ? 6 : 4).fill(0));
          } else {
            expect(result.labels).toEqual(
              maxP ? [0, 0, 1, 1, 2, 2, 3, 3] : [0, 0, 1, 1, 1, 1, 2, 2, 3]
            );
            expect(result.status).toEqual(maxP ? [4, 1, 0, 0, 0, 0] : [1, 1, 0, 0]);
            expect(Number.isFinite(result.objective)).toBe(true);
            expect(Math.abs(result.objective - (maxP ? 28 : 24))).toBeLessThanOrEqual(1e-5);
          }
        }
      } finally {
        fixture.compiled.destroy();
        fixture.rig.destroy();
      }
    }
  });
}
