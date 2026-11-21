// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUClassAssignment} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {AnalysisRig} from '../catchment-accessibility/rig';

it('assigns an inner edge to the requested left or right closed class', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const intervalClosure of ['left', 'right'] as const) {
    const rig = new AnalysisRig(device);
    try {
      const output = rig.output('uint32', 1);
      rig.run(
        new GPUClassAssignment({
          values: rig.input(Float32Array.of(1), 'float32'),
          breaks: rig.input(Float32Array.of(0, 1, 2), 'float32'),
          classCount: rig.input(Uint32Array.of(2), 'uint32'),
          intervalClosure,
          output
        })
      );
      expect(await rig.readUint(output)).toEqual([intervalClosure === 'left' ? 1 : 0]);
    } finally {
      rig.destroy();
    }
  }
});
