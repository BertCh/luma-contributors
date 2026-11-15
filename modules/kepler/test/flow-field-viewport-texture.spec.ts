// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {Texture} from '@luma.gl/core';
import {getWebGLTestDevice} from '@luma.gl/core/test-utils';
import {GL, withDeviceParameters} from '@luma.gl/core/webgl';
import {isFlowParticleSimulationSupported} from '@luma.gl/graph/experimental/simulation';
import {FlowFieldViewportTexture} from '../../../examples/deck/kepler-compute/src/flow-field-viewport-texture';

// Run on an actual WebGL device. CPU-grid cases in the browser scripts still use WebGPU and
// cannot establish GLSL viewport/blending correctness. Alpha here is metadata, never opacity.
test.each([
  {name: 'constant-speed velocity', sample: [5, 0, 1, 0]},
  {name: 'altitude', sample: [120, 0, 1, 0]}
])('WebGL preserves every zero-alpha $name texel despite inherited source-alpha blending', async ({
  sample
}) => {
  const device = await getWebGLTestDevice();
  expect(isFlowParticleSimulationSupported(device)).toBe(true);
  const columns = 24;
  const rows = 16;
  const texels = new Float32Array(columns * rows * 4);
  for (let cell = 0; cell < columns * rows; cell++) texels.set(sample, cell * 4);
  const source = device.createTexture({
    width: columns,
    height: rows,
    format: 'rgba32float',
    data: texels,
    usage: Texture.SAMPLE,
    sampler: {minFilter: 'nearest', magFilter: 'nearest'}
  });
  const crop = new FlowFieldViewportTexture(device, columns, rows);
  try {
    let drawn = false;
    for (let attempt = 0; attempt < 120 && !drawn; attempt++) {
      drawn = withDeviceParameters(
        device,
        {
          blend: true,
          blendColorSrcFactor: 'src-alpha',
          blendColorDstFactor: 'one-minus-src-alpha',
          blendAlphaSrcFactor: 'src-alpha',
          blendAlphaDstFactor: 'one-minus-src-alpha'
        },
        () => crop.update(source, [0.2, 0.25, 0.8, 0.75])
      );
      if (!drawn) await new Promise(resolve => setTimeout(resolve, 16));
    }
    expect(drawn).toBe(true);
    const output = device.readPixelsToArrayWebGL(crop.texture, {
      sourceWidth: columns,
      sourceHeight: rows,
      sourceFormat: GL.RGBA,
      sourceType: GL.FLOAT,
      target: new Float32Array(texels.length)
    });
    let maximumError = 0;
    for (let index = 0; index < output.length; index++)
      maximumError = Math.max(maximumError, Math.abs(output[index] - sample[index % 4]));
    // Float32 bilinear interpolation can round the sum of four constant samples slightly.
    expect(maximumError).toBeLessThanOrEqual(0.0001);
  } finally {
    crop.destroy();
    source.destroy();
  }
});
