// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUContinuousSpatialSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runPoisonedOperation} from './output-oracle-harness';
import {
  CONTINUOUS_INPUTS,
  CONTINUOUS_PROPS,
  expectContinuousOracle
} from './continuous-sample-output-oracles-fixture';

it('GPU continuous sampling operation and contributor publish the independent Philox line oracle including unused tails', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expectContinuousOracle(
    await runPoisonedOperation(
      device,
      new GPUContinuousSpatialSampleOperation(CONTINUOUS_PROPS),
      CONTINUOUS_INPUTS,
      Object.values(CONTINUOUS_PROPS.output)
    )
  );
});
