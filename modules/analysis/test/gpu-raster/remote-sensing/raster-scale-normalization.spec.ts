// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTexture} from '../../../src/raster-analysis/remote-sensing/gpu-raster-texture';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

for (const scale of [Math.fround(3.402823466e38), 2 ** -120]) {
  it(`texture range ${scale} retains distinct quantization bins and all seven GLCM metrics`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const features = harness.output('features', 'float32', 28);
    const pairCount = harness.output('pairCount', 'uint32', 4);
    (features.buffer.defaultBuffer as Buffer).write(new Float32Array(28).fill(-12345));
    (pairCount.buffer.defaultBuffer as Buffer).write(new Uint32Array(4).fill(0xa5a5a5a5));
    const result = await harness.run(
      new GPURasterTexture({
        grid: makeGPURasterMetadata({width: 2, height: 2}),
        // Bins 0, 3 on each row; two identical directed pairs in every clipped radius-one window.
        values: harness.input([scale / 8, scale * 0.875, scale / 8, scale * 0.875]),
        levels: 4,
        range: [0, scale],
        radius: 1,
        offset: [1, 0],
        features,
        pairCount
      })
    );
    expect(result['pairCount']).toEqual([2, 2, 2, 2]);
    // Closed-form point-mass GLCM at (0,3): contrast 9, distance 3, homogeneity 1/10,
    // ASM/energy 1, constant-window correlation 1, entropy 0.
    const expected = [9, 3, 0.1, 1, 1, 1, 0];
    for (let cell = 0; cell < 4; cell++)
      for (let metric = 0; metric < 7; metric++)
        expect(result['features'][cell * 7 + metric]).toBeCloseTo(expected[metric], 5);
  });
}
