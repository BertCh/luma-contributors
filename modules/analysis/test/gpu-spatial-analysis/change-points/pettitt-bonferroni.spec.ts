// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import {
  GPURasterFalseDiscoveryRateMask,
  getGPURasterFalseDiscoveryRateMaskParameterValues
} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {AnalysisRig} from '../catchment-accessibility/rig';

it('DGN-01 Pettitt to Bonferroni excludes untested cells and retains tested constants', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const control of ['missing', 'single', 'constant'] as const) {
    const rig = new AnalysisRig(device);
    try {
      const second = new Float32Array(12).fill(control === 'constant' ? 3 : NaN);
      if (control === 'single') second[0] = 3;
      const pValues = rig.output('float32', 2);
      const adjustedPValues = rig.output('float32', 2);
      const significant = rig.output('uint32', 2);
      const effectiveFamilySize = rig.output('uint32', 1);
      rig.run(
        new GPUPettittTest({
          values: rig.input(
            Float32Array.of(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...second),
            'float32'
          ),
          cellCount: 2,
          sliceCount: 12,
          layout: 'cell-major',
          output: {
            pValues,
            statistic: rig.output('uint32', 2),
            changeSlice: rig.output('uint32', 2)
          }
        }),
        new GPURasterFalseDiscoveryRateMask({
          pValues,
          cellCount: 2,
          method: 'bonferroni',
          parameters: rig.input(
            getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05}),
            'float32'
          ),
          output: {adjustedPValues, significant, effectiveFamilySize}
        })
      );
      // K=36,n=12: 2 exp(-6*36²/(12³+12²)) = .03140780113025614 (Python math.exp).
      const constant = control === 'constant';
      expectNumericArrayClose(
        await rig.readFloat(pValues),
        [0.03140780113025614, constant ? 1 : NaN],
        {absoluteTolerance: 1e-7}
      );
      expectNumericArrayClose(
        await rig.readFloat(adjustedPValues),
        [constant ? 0.06281560226051228 : 0.03140780113025614, constant ? 1 : NaN],
        {absoluteTolerance: 1e-7}
      );
      expect(await rig.readUint(significant)).toEqual([constant ? 0 : 1, 0]);
      expect(await rig.readUint(effectiveFamilySize)).toEqual([constant ? 2 : 1]);
    } finally {
      rig.destroy();
    }
  }
});
