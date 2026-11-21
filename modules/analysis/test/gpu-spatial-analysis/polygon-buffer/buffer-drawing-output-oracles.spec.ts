// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createBufferDrawingOracle} from './buffer-drawing-output-oracles';

it('draws a retained rectangle only after both completeness flags clear', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = await createBufferDrawingOracle(device);
  try {
    // One retained square always has count=1 and offsets [0,1]/[0,1]/[0,4]. Completeness
    // alone controls whether that one ring is drawable; this is an independent Boolean oracle.
    for (const [overflow, candidateOverflow, expected] of [
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0]
    ]) {
      fixture.overflow.write(new Uint32Array([overflow]));
      fixture.candidateOverflow.write(new Uint32Array([candidateOverflow]));
      expect((await fixture.runGPU())['drawing-ring-count']).toEqual([expected]);
    }
  } finally {
    fixture.destroy();
  }
});
