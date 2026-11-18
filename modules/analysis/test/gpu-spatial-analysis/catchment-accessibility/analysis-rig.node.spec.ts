// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {AnalysisRig} from './rig';

it.each([
  {format: 'uint32', components: 1},
  {format: 'float32', components: 1},
  {format: 'float32x2', components: 2},
  {format: 'float32x3', components: 3},
  {format: 'float32x4', components: 4}
] as const)('allocates all components for 16 $format output rows', ({format, components}) => {
  const device = createNullWebGPUDevice();
  const rig = new AnalysisRig(device);
  try {
    const output = rig.output(format, 16);
    expect(output.length).toBe(16);
    expect(output.buffer.byteLength).toBe(16 * components * Float32Array.BYTES_PER_ELEMENT);
  } finally {
    rig.destroy();
    device.destroy();
  }
});
