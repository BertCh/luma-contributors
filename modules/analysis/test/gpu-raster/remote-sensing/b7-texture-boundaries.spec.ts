// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTexture} from '../../../src/raster-analysis/remote-sensing/gpu-raster-texture';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

it('B7-TEX-01 GLCM keeps the sample below the exact midpoint in the lower bin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const result = await harness.run(
    new GPURasterTexture({
      grid: makeGPURasterMetadata({width: 2, height: 1}),
      values: harness.input([0.5, 1]),
      levels: 2,
      range: [2 ** -56, 1],
      radius: 1,
      offset: [1, 0],
      features: harness.output('features', 'float32', 14),
      pairCount: harness.output('pairs', 'uint32', 2)
    })
  );
  // The exact midpoint is 1/2 + 2^-57. Samples 1/2 and 1 enter bins 0 and 1,
  // giving one directed pair of contrast (1 - 0)^2 in each clipped window.
  expect(result['pairs']).toEqual([1, 1]);
  expect(result['features']).toHaveLength(14);
  for (const cell of [0, 1]) {
    expect(Number.isFinite(result['features'][cell * 7])).toBe(true);
    expect(result['features'][cell * 7]).toBe(1);
  }
});
