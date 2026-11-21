// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNearRepeatKnox} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-near-repeat-knox';
import {
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  getGPUSpaceTimeParameterValues
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('near-repeat counts each tied-time table independently of the shuffle and clears unused draws', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'table-parameters',
    format: 'uint32',
    length: GPU_SPACE_TIME_PARAMETER_LENGTH
  });
  const counts = rig.output('uint32', 16);
  const summary = rig.output('float32', 32);
  const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
  const candidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
  const pairs = {
    offsets: rig.input(Uint32Array.of(0, 2, 3, 3), 'uint32', 4),
    neighbors: rig.input(Uint32Array.of(1, 2, 2), 'uint32', 3),
    weights: rig.input(Float32Array.of(1, 1, 1), 'float32', 3),
    distances: rig.input(Float32Array.of(1, 3, 2), 'float32', 3),
    status: {overflow, candidateOverflow}
  };
  const contributor = new GPUNearRepeatKnox({
    pairs,
    times: rig.input(Float32Array.of(7, 7, 7), 'float32', 3),
    parameters: parameters.importToGraph(rig.graph),
    distanceEdges: [0, 1, 3],
    timeEdges: [0, 1, 2],
    maximumPermutations: 3,
    output: {counts: counts.view, summary: summary.view}
  });
  // Removing the caller's fields after validation must not remove the original bindings.
  expect(Reflect.deleteProperty(pairs.status, 'overflow')).toBe(true);
  expect(Reflect.deleteProperty(pairs.status, 'candidateOverflow')).toBe(true);
  rig.graph.add(contributor);
  const compiled = rig.graph.compile();
  try {
    // Three unique unordered pairs have distances 1, 2, 3. Inclusive upper band boundaries
    // put one in distance band 0 and two in band 1. Every time gap is zero under EVERY
    // permutation, so only time band 0 is occupied. No production shuffle is an oracle.
    parameters.write(getGPUSpaceTimeParameterValues({seed: 17, permutationCount: 2}));
    for (const [finalLoss, candidateLoss] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      rig.bufferOf(overflow).write(Uint32Array.of(finalLoss));
      rig.bufferOf(candidateOverflow).write(Uint32Array.of(candidateLoss));
      submitGraph(device, compiled, undefined);
      const incomplete = finalLoss !== 0 || candidateLoss !== 0;
      const actualCounts = await readUint32(counts.buffer, 16);
      const actualSummary = await readFloat32(summary.buffer, 32);
      if (incomplete) {
        expect(actualCounts).toEqual(Array(16).fill(0));
        expect(actualSummary.every(Number.isNaN)).toBe(true);
      } else {
        expect(actualCounts).toEqual([1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0]);
        // Every draw ties: means equal observed counts, variance is 0, both exceedance
        // counts are 2, both p-values are 1; 0/0 Knox ratios are unavailable.
        expect(actualSummary).toEqual([
          1,
          1,
          0,
          2,
          2,
          1,
          1,
          1,
          0,
          0,
          0,
          2,
          2,
          1,
          1,
          NaN,
          2,
          2,
          0,
          2,
          2,
          1,
          1,
          1,
          0,
          0,
          0,
          2,
          2,
          1,
          1,
          NaN
        ]);
      }
    }
    parameters.write(getGPUSpaceTimeParameterValues({seed: 29, permutationCount: 1}));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(counts.buffer, 16)).toEqual([
      1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  } finally {
    compiled.destroy();
    parameters.destroy();
    rig.destroy();
  }
});
