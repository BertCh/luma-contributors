// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {
  expectResidualInference,
  getResidualInputs,
  getResidualProps
} from './mantel-residual-w5-fixture';

for (const [star, seed, stopping] of [
  [true, 0, true],
  [true, 84, false],
  [false, 8, false]
] as const) {
  it(`W5-06/W5-07 device Pearson residuals: star=${star}, seed=${seed}, stopping=${stopping}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runPoisonedOperation(
      device,
      new GPUMantelTestOperation(getResidualProps(star, stopping)),
      getResidualInputs(star, seed),
      ['statistics', 'summary', 'used', 'usable']
    );
    expectResidualInference(result, star, stopping);
  });
}
