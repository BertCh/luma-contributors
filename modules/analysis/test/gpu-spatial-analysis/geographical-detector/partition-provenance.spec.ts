// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUGeographicalDetector} from '../../../src/gpu-spatial-analysis/geographical-detector';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

it('CPL-14 retains incumbent statistics and per-submission partition evidence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const labelsComplete = rig.output('uint32', 1),
      complete = rig.output('uint32', 1),
      summary = rig.output('float32', 8);
    labelsComplete.buffer.write(Uint32Array.of(1));
    rig.run(
      new GPUGeographicalDetector({
        values: rig.input(Float32Array.of(0, 2, 4, 6), 'float32'),
        labels: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32'),
        labelCapacity: 2,
        labelsComplete: labelsComplete.view,
        output: {summary: summary.view, labelsComplete: complete.view}
      })
    );
    for (const word of [1, 0, 1]) {
      labelsComplete.buffer.write(Uint32Array.of(word));
      rig.resubmit();
      expect(await complete.readUint32()).toEqual([word]);
      // Means 1,5: SSB=16, SSW=4, SST=20; incumbent q=.8 regardless of certificate.
      const values = await summary.readFloat32();
      expect(Number.isFinite(values[0])).toBe(true);
      expect(Math.abs(values[0] - 0.8)).toBeLessThan(1e-6);
    }
  } finally {
    rig.destroy();
  }
});
