// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUSpatialBalancedSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  BALANCED_PROPS,
  expectBalancedOracle,
  getBalancedInputs,
  FRACTIONAL_BALANCED_PROPS,
  FRACTIONAL_BALANCED_INPUTS,
  expectFractionalBalancedOracle
} from './balanced-sample-output-oracles-fixture';

it('CPU balanced sampling preserves certainty IDs, unit probabilities and frame invalidation', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const invalid of [false, true])
      expectBalancedOracle(
        await runOperationGraph(device, {
          inputs: getBalancedInputs(invalid),
          operations: [new CPUSpatialBalancedSampleOperation(BALANCED_PROPS)],
          outputs: Object.values(BALANCED_PROPS.output)
        }),
        invalid
      );
  } finally {
    device.destroy();
  }
});

it('CPU balanced sampling distinguishes probability from inverse design weight', async () => {
  const device = createNullWebGPUDevice();
  try {
    const result = await runOperationGraph(device, {
      inputs: FRACTIONAL_BALANCED_INPUTS,
      operations: [new CPUSpatialBalancedSampleOperation(FRACTIONAL_BALANCED_PROPS)],
      outputs: Object.values(FRACTIONAL_BALANCED_PROPS.output)
    });
    expectFractionalBalancedOracle(result);
    expect(() =>
      expectFractionalBalancedOracle({
        ...result,
        probabilities: result['weights'],
        weights: result['probabilities']
      })
    ).toThrow();
  } finally {
    device.destroy();
  }
});
