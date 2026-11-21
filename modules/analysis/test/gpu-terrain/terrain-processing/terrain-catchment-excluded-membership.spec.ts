// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../../gpu-spatial-analysis/catchment-accessibility/rig';

it('review excluded rows outrank unknown stream membership in profile diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [validity, basin, expectedStatus] of [
    [0, 0, 0],
    [1, 2, 0],
    [0, 2, 0],
    [1, 0, 2]
  ]) {
    const rig = new AnalysisRig(device);
    try {
      const profileStatus = rig.output('uint32', 1);
      const converged = rig.output('uint32', 1);
      const invalidCount = rig.output('uint32', 1);
      const iterationLimitReached = rig.output('uint32', 1);
      rig.run(
        new GPUTerrainCatchmentSummary({
          grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          basinCount: 1,
          histogramBins: 1,
          elevation: rig.input(Float32Array.of(1), 'float32'),
          validity: rig.input(Uint32Array.of(validity), 'uint32'),
          basinIds: rig.input(Uint32Array.of(basin), 'uint32'),
          streams: rig.input(Uint32Array.of(0xffffffff), 'uint32'),
          flowDirections: rig.input(Uint32Array.of(0), 'uint32'),
          output: {
            statistics: rig.output('float32', 6),
            hypsometry: rig.output('float32', 1),
            profiles: rig.output('float32', 3),
            profileStatus,
            status: {converged, invalidCount, iterationLimitReached}
          }
        })
      );
      // An excluded row contributes status 0, so the empty eligible domain converges.
      // A valid in-basin row with unknown membership remains unresolved (status 2).
      expect(await rig.readUint(profileStatus)).toEqual([expectedStatus]);
      expect(await rig.readUint(converged)).toEqual([expectedStatus === 0 ? 1 : 0]);
      expect(await rig.readUint(invalidCount)).toEqual([expectedStatus === 0 ? 1 : 0]);
      expect(await rig.readUint(iterationLimitReached)).toEqual([0]);
    } finally {
      rig.destroy();
    }
  }
});
