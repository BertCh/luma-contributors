// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUDatumGridTransform} from '../../../src/geospatial-analysis/datum-grid-transform/index';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('datum inverse status follows an exact unresolved affine residual and clears on a constant-field replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0.5, 0.5, 0]));
    const samples = fixture.view(
      'float32x3',
      new Float32Array([0.125, 0, 0, 0.625, 0, 0, 0.125, 0, 0, 0.625, 0, 0])
    );
    const mask = fixture.view('uint32', new Uint32Array([1, 1, 1, 1]));
    const transformed = fixture.view('float32x3', new Float32Array(3));
    const status = fixture.view('uint32', new Uint32Array([0xffffffff]));
    fixture.graph.add(
      new GPUDatumGridTransform({
        positions: positions.view,
        samples: samples.view,
        mask: mask.view,
        grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
        operation: 'horizontal',
        direction: 'inverse',
        maximumIterations: 1,
        tolerance: 1e-6,
        output: {positions: transformed.view, status: status.view}
      })
    );
    // Bilinear shifts are s(x)=1/8+x/2. Solve x+s(x)=1/2:
    // x=1/4. One fixed-point update from 1/2 gives 1/8, whose
    // unresolved residual is 1/8 + 3/16 - 1/2 = -3/16.
    // Every number is dyadic and exact in float32; 3/16 >> 1e-6.
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([8]);
    // Constant s(x)=1/8 instead solves exactly after one update: x=3/8.
    // Zero residual, finite coordinates, valid masks and full coverage
    // force the entire status word to zero, clearing the earlier bit 8.
    samples.buffer.write(new Float32Array([0.125, 0, 0, 0.125, 0, 0, 0.125, 0, 0, 0.125, 0, 0]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    mask.buffer.write(new Uint32Array([1, 0, 1, 1]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([4]);
    mask.buffer.write(new Uint32Array([1, 1, 1, 1]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
