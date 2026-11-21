// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const adjusted of [false, true])
  it(`shared normal tail retains GPU output writers: adjusted=${adjusted}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const output = {
        correlations: rig.output('float32', 1),
        pValues: rig.output('float32', 1),
        adjustedPValues: adjusted ? rig.output('float32', 1) : undefined,
        testedCount: rig.output('uint32', 1)
      };
      rig.run(
        new GPUSpaceTimeCrossCorrelation({
          first: rig.input(Float32Array.of(1, 1, -1, -1, 1, 1, -1, -1), 'float32'),
          second: rig.input(
            Float32Array.of(
              1.366025447845459,
              -0.3660254180431366,
              0.3660254180431366,
              -1.366025447845459,
              1.366025447845459,
              -0.3660254180431366,
              0.3660254180431366,
              -1.366025447845459
            ),
            'float32'
          ),
          layout: 'cell-major',
          cellCount: 1,
          periodCount: 8,
          lags: [0],
          output
        })
      );
      // /usr/bin/python3 NumPy corrcoef of these float32 inputs; SciPy erfc(abs(atanh(r))*sqrt(5/2)).
      const correlation = (await rig.readFloat(output.correlations))[0];
      const probability = (await rig.readFloat(output.pValues))[0];
      expect(Number.isFinite(correlation)).toBe(true);
      expect(Number.isFinite(probability)).toBe(true);
      expect(Math.abs(correlation - 0.49999999854928157)).toBeLessThan(2e-6);
      expect(Math.abs(probability - 0.2193396715575464)).toBeLessThan(2e-6);
      expect(await rig.readUint(output.testedCount)).toEqual([1]);
      if (output.adjustedPValues) {
        const value = (await rig.readFloat(output.adjustedPValues))[0];
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - probability)).toBeLessThan(1e-7);
      }
    } finally {
      rig.destroy();
    }
  });
