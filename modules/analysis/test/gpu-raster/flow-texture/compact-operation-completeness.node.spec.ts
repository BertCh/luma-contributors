// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {CPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectCompactStatus} from './flow-output-assertions';
import {
  COMPACT_STATUS_OUTPUTS,
  getCompactPeaksFixture,
  getCompactStreamlinesFixture
} from './compact-completeness-fixtures';

it('RT-125 compact peak operation exposes undecided loss separately from capacity', async () => {
  const {props, inputs} = getCompactPeaksFixture();
  const rows = await runOperationGraph(createNullWebGPUDevice(), {
    inputs,
    operations: [new CPUProfilePeaksOperation(props)],
    outputs: COMPACT_STATUS_OUTPUTS
  });
  // [0,1,0] has one strict maximum, but zero suppression rounds decide none.
  expectCompactStatus(rows, [0, 0, 1]);
});

it('S3 convergence does not certify that a compact peak result fits its capacity', async () => {
  const {props, inputs} = getCompactPeaksFixture();
  const rows = await runOperationGraph(createNullWebGPUDevice(), {
    inputs: {
      ...inputs,
      values: {format: 'float32', values: Float32Array.of(0, 1, 0, 0, 1, 0)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 6)}
    },
    operations: [
      new CPUProfilePeaksOperation({
        ...props,
        sampleCount: 6,
        maximumSuppressionIterations: 16,
        output: {
          peaks: {
            ids: 'ids',
            capacity: 1,
            status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'}
          },
          status: {converged: 'converged'}
        }
      })
    ],
    outputs: [...COMPACT_STATUS_OUTPUTS, 'converged']
  });
  // Two strict maxima with radius 0 both finish suppression; one compact slot drops one.
  expectCompactStatus(rows, [1, 1, 0]);
  expect(Array.from(rows['converged'])).toEqual([1]);
});

for (const [budget, expected] of [
  [1, [2, 0, 1]],
  [32, [3, 0, 0]]
] as const) {
  it(`RT-125 compact streamline operation preserves candidate completeness at budget ${budget}`, async () => {
    const {props, inputs} = getCompactStreamlinesFixture(budget);
    const rows = await runOperationGraph(createNullWebGPUDevice(), {
      inputs,
      operations: [new CPUStreamlinesOperation(props)],
      outputs: COMPACT_STATUS_OUTPUTS
    });
    // Philox(seed=3,purpose=2) gives seed x ~[3.64262,6.71425,13.43090], priority
    // high words [50769,1502,45318]. Two unit steps each way occupy cells [1..5],
    // [4..8], [11..15]. Round 1 accepts seeds 0 and 2; seed 1 waits on seed 0.
    // Round 2 trims seed 1's backward span at occupied cell 5, keeping [6..8].
    // Thus all three fit 3 lines / 15 points; capacity overflow alone stays zero.
    expectCompactStatus(rows, expected);
  });
}
