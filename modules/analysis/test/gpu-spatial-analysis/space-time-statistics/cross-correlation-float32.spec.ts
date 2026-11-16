// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('correlates finite large observations without overflowing variance products or means', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1e-37, 1e10, 1e20, 8e37]) {
    const rig = new WeightsRig(device);
    try {
      const first = new Float32Array([1, 2, 3, 4].map(value => value * scale));
      const second = Float32Array.from(first, value => -value);
      const correlations = rig.output('float32', 1);
      const pValues = rig.output('float32', 1);
      rig.run(
        new GPUSpaceTimeCrossCorrelation({
          first: rig.input(first, 'float32', 4),
          second: rig.input(second, 'float32', 4),
          layout: 'cell-major',
          cellCount: 1,
          periodCount: 4,
          lags: [0],
          output: {correlations: correlations.view, pValues: pValues.view}
        })
      );
      expect((await readFloat32(correlations.buffer, 1))[0]).toBeCloseTo(-1, 6);
      expect((await readFloat32(pValues.buffer, 1))[0]).toBeLessThan(1e-10);
    } finally {
      rig.destroy();
    }
  }
});
