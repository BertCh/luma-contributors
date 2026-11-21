// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUSpatialBalancedSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runPoisonedOperation} from './output-oracle-harness';
import {
  BALANCED_PROPS,
  expectBalancedOracle,
  getBalancedInputs,
  FRACTIONAL_BALANCED_PROPS,
  FRACTIONAL_BALANCED_INPUTS,
  expectFractionalBalancedOracle
} from './balanced-sample-output-oracles-fixture';

it('GPU balanced sampling preserves certainty IDs, unit probabilities and frame invalidation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const invalid of [false, true])
    expectBalancedOracle(
      await runPoisonedOperation(
        device,
        new GPUSpatialBalancedSampleOperation(BALANCED_PROPS),
        getBalancedInputs(invalid),
        Object.values(BALANCED_PROPS.output)
      ),
      invalid
    );
});

it('GPU balanced sampling distinguishes probability from inverse design weight', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expectFractionalBalancedOracle(
    await runPoisonedOperation(
      device,
      new GPUSpatialBalancedSampleOperation(FRACTIONAL_BALANCED_PROPS),
      FRACTIONAL_BALANCED_INPUTS,
      Object.values(FRACTIONAL_BALANCED_PROPS.output)
    )
  );
});
