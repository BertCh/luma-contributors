// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUUtilizationVolumeIsopleths} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('writes adjacent volume and total-mass fields without mixed storage usage in a dispatch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    // Import the physical buffer once, then address each output by its logical offset.
    const packed = rig.output('float32', 3);
    const volume = rig.graph.createDataView(packed.view.buffer, {format: 'float32', length: 2});
    const totalMass = rig.graph.createDataView(packed.view.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: 8
    });
    rig.run(
      new GPUUtilizationVolumeIsopleths({
        density: rig.input(new Float32Array([2, 1]), 'float32', 2),
        parameters: rig.input(new Float32Array([0.95, 0, 0, 0]), 'float32', 4),
        output: {volume, totalMass}
      })
    );
    const values = await readFloat32(packed.buffer, 3);
    expect(values[0]).toBeCloseTo(2 / 3, 6);
    expect(values.slice(1)).toEqual([1, 3]);
  } finally {
    rig.destroy();
  }
});
