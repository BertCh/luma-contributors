// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {GPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectCompactStatus} from './flow-output-assertions';
import {
  COMPACT_STATUS_OUTPUTS,
  getCompactPeaksFixture,
  getCompactStreamlinesFixture
} from './compact-completeness-fixtures';

it('RT-125 compact GPU peak operation exposes candidate loss at zero rounds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {props, inputs} = getCompactPeaksFixture();
  const rows = await runOperationGraph(device, {
    inputs,
    operations: [new GPUProfilePeaksOperation(props)],
    outputs: COMPACT_STATUS_OUTPUTS
  });
  expectCompactStatus(rows, [0, 0, 1]);
});

for (const [budget, expected] of [
  [1, [2, 0, 1]],
  [32, [3, 0, 0]]
] as const) {
  it(`RT-125 compact GPU streamline operation preserves completeness at budget ${budget}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {props, inputs} = getCompactStreamlinesFixture(budget);
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUStreamlinesOperation(props)],
      outputs: COMPACT_STATUS_OUTPUTS
    });
    expectCompactStatus(rows, expected);
  });
}
