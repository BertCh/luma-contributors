// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPURegionPartitionEvaluation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

// One non-finite value must not turn every statistic into NaN: the row is left out, as an
// unlabelled row is, and reported. Expected values are by hand for the six finite rows:
//   region 0 = {1, 2, 3}, region 1 = {10, 11, 12}; within 2 + 2 = 4; mean 6.5, total
//   2 (5.5^2 + 4.5^2 + 3.5^2) = 125.5; between 121.5; pseudo-F (121.5 / 1) / (4 / 4) = 121.5.

const LAYOUT = GPU_REGION_PARTITION_EVALUATION_LAYOUT;

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    expect(Math.abs(actual[index] - value), `entry ${index}`).toBeLessThanOrEqual(
      1e-6 * Math.abs(value)
    );
  });
}

async function evaluate(values: number[], labels: number[]) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const summary = rig.output('float32', LAYOUT.length);
  const regionSizes = rig.output('uint32', 2);
  const regionMeans = rig.output('float32', 2);
  const regionWithin = rig.output('float32', 2);
  const invalidRowCount = rig.output('uint32', 1);
  // A path over the rows, for the cross-link fraction.
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < labels.length; row++) {
    if (row > 0) neighbors.push(row - 1);
    if (row + 1 < labels.length) neighbors.push(row + 1);
    offsets.push(neighbors.length);
  }
  rig.graph.add(
    new GPURegionPartitionEvaluation({
      values: rig.input(new Float32Array(values), 'float32', values.length),
      labels: rig.input(new Uint32Array(labels), 'uint32', labels.length),
      labelCapacity: 2,
      weights: rig.uploadWeights(
        {offsets, neighbors, weights: neighbors.map(() => 1), distances: []},
        0
      ),
      output: {
        summary: summary.view,
        regionSizes: regionSizes.view,
        regionMeans: regionMeans.view,
        regionWithinSumOfSquares: regionWithin.view,
        invalidRowCount: invalidRowCount.view
      }
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    summary: Array.from(await readFloat32(summary.buffer, LAYOUT.length)),
    regionSizes: Array.from(await readUint32(regionSizes.buffer, 2)),
    regionMeans: Array.from(await readFloat32(regionMeans.buffer, 2)),
    regionWithin: Array.from(await readFloat32(regionWithin.buffer, 2)),
    invalidRowCount: (await readUint32(invalidRowCount.buffer, 1))[0]
  };
  compiled.destroy();
  rig.destroy();
  return result;
}

for (const [name, missing] of [
  ['NaN', Number.NaN],
  ['an infinity', Number.POSITIVE_INFINITY]
] as const) {
  it(`GPURegionPartitionEvaluation leaves out a row whose value is ${name} and reports it`, async () => {
    // Row 3 sits in the middle of the path, between the two regions.
    const result = await evaluate([1, 2, 3, missing, 10, 11, 12], [0, 0, 0, 1, 1, 1, 1]);
    expect(result.invalidRowCount).toBe(1);
    expect(result.regionSizes).toEqual([3, 3]);
    // Division on the GPU is not always correctly rounded (6 / 3 can come back one last place
    // below 2), so floats are compared to a relative 1e-6.
    expectClose(result.regionMeans, [2, 11]);
    expectClose(result.regionWithin, [2, 2]);
    expect(result.summary[LAYOUT.regionCount]).toBe(2);
    expectClose(
      [
        result.summary[LAYOUT.withinSumOfSquares],
        result.summary[LAYOUT.totalSumOfSquares],
        result.summary[LAYOUT.betweenSumOfSquares],
        result.summary[LAYOUT.pseudoF]
      ],
      [4, 125.5, 121.5, 121.5]
    );
    expect(result.summary[LAYOUT.minimumSize]).toBe(3);
    expect(result.summary[LAYOUT.maximumSize]).toBe(3);
    expect(result.summary[LAYOUT.ignoredRows]).toBe(1);
    // The left-out row breaks the path: the links between used rows are 0-1, 1-2, 4-5, 5-6 in
    // both directions, none of them across regions.
    expect(result.summary[LAYOUT.crossLinkFraction]).toBe(0);
  });
}

it('GPURegionPartitionEvaluation reports no invalid row for a finite table', async () => {
  const result = await evaluate([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1]);
  expect(result.invalidRowCount).toBe(0);
  expect(result.summary[LAYOUT.ignoredRows]).toBe(0);
  expectClose([result.summary[LAYOUT.pseudoF]], [121.5]);
  // Links 0-1, 1-2, 2-3, 3-4, 4-5 in both directions: one of the five crosses the regions.
  expect(result.summary[LAYOUT.crossLinkFraction]).toBeCloseTo(0.2, 6);
});
