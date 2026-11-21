// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOffsetOutputOracle} from './offset-output-oracles';

it('publishes exact compact counts with each optional scalar alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of [undefined, 'vertexCount', 'requiredVertexCount'] as const) {
    const fixture = createOffsetOutputOracle(device, optional);
    try {
      expect(await fixture.runGPU()).toEqual(fixture.expected);
      const parameters = fixture.columns.find(entry => entry.view === fixture.parameters)!;
      parameters.buffer.write(new Float32Array([-2, 5, 0, 0]));
      expect(await fixture.runGPU()).toEqual({...fixture.expected, positions: [0, -2]});
    } finally {
      fixture.destroy();
    }
  }
});
