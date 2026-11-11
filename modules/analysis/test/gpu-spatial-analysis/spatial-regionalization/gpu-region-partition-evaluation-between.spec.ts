// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPURegionPartitionEvaluation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

// Fix review follow-up to point-process-clustering: the between-region sum of squares of a
// partition that explains a millionth of the variance. As `total - within` it is the difference
// of two float32 sums of about 32,000 whose last place (0.002) is a sixteenth of the answer
// (0.032). The direct form `sum n_k |mean_k - mean|^2` has no such cancellation. Expected values
// are float64 evaluations of the definitions over the float32 inputs.

const LAYOUT = GPU_REGION_PARTITION_EVALUATION_LAYOUT;

it('GPURegionPartitionEvaluation keeps a between sum of squares that is 1e-6 of the total', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 65536;
  // Two labels by row parity. Inside each label the values alternate +0.7 and -0.7 in equal
  // numbers, and label 1 is shifted by 0.0014: the label means are 0 and 0.0014.
  const values = Float32Array.from(
    {length: rows},
    (_, row) => (((row >> 1) & 1) === 0 ? 0.7 : -0.7) + (row % 2) * 0.0014
  );
  const labels = Uint32Array.from({length: rows}, (_, row) => row % 2);

  const sums = [0, 0];
  for (let row = 0; row < rows; row++) sums[labels[row]] += values[row];
  const means = sums.map(sum => sum / (rows / 2));
  const mean = (sums[0] + sums[1]) / rows;
  let within = 0;
  let total = 0;
  for (let row = 0; row < rows; row++) {
    within += (values[row] - means[labels[row]]) ** 2;
    total += (values[row] - mean) ** 2;
  }
  const between = (rows / 2) * ((means[0] - mean) ** 2 + (means[1] - mean) ** 2);
  const pseudoF = between / (within / (rows - 2));
  // The scene is what the comment says: the partition explains about a millionth.
  expect(between / total).toBeGreaterThan(0.9e-6);
  expect(between / total).toBeLessThan(1.1e-6);

  const rig = new WeightsRig(device);
  const summary = rig.output('float32', LAYOUT.length);
  rig.graph.add(
    new GPURegionPartitionEvaluation({
      values: rig.input(values, 'float32', rows),
      labels: rig.input(labels, 'uint32', rows),
      labelCapacity: 2,
      output: {summary: summary.view}
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const result = Array.from(await readFloat32(summary.buffer, LAYOUT.length));
  compiled.destroy();
  rig.destroy();

  expect(result[LAYOUT.regionCount]).toBe(2);
  expect(Math.abs(result[LAYOUT.withinSumOfSquares] - within)).toBeLessThan(1e-5 * within);
  expect(Math.abs(result[LAYOUT.totalSumOfSquares] - total)).toBeLessThan(1e-5 * total);
  expect(Math.abs(result[LAYOUT.betweenSumOfSquares] - between)).toBeLessThan(1e-3 * between);
  expect(Math.abs(result[LAYOUT.pseudoF] - pseudoF)).toBeLessThan(1e-3 * pseudoF);
});
