// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUBivariateClassification} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import {getGPUBivariateClassificationParameterValues} from '../../../src/dataframe-analysis/column-classification/bivariate-classification-parameters';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('bivariate counts rewrite inactive poisoned slots and clear a previously occupied joint class', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const valuesX = fixture.view('float32', new Float32Array([0, 1, 2, 3, NaN, 1]));
    const valuesY = fixture.view('float32', new Float32Array([0, 1, 2, 3, 0, 2]));
    const mask = fixture.view('uint32', new Uint32Array([1, 1, 1, 1, 1, 0]));
    const breaksX = fixture.view('float32', new Float32Array([0, 2, 4, 4, 4]));
    const breaksY = fixture.view('float32', new Float32Array([0, 2, 4, 4, 4]));
    const palette = fixture.view('uint32', new Uint32Array(16));
    const parameters = fixture.view(
      'float32',
      getGPUBivariateClassificationParameterValues({classCountX: 2, classCountY: 2})
    );
    const classCounts = fixture.view('uint32', new Uint32Array(16).fill(0xffffffff));
    fixture.graph.add(
      new GPUBivariateClassification({
        valuesX: valuesX.view,
        valuesY: valuesY.view,
        mask: mask.view,
        breaksX: breaksX.view,
        breaksY: breaksY.view,
        palette: palette.view,
        parameters: parameters.view,
        maximumClassCount: 4,
        output: {classCounts: classCounts.view}
      })
    );
    // Inner edge 2 assigns (0,0),(1,1) to joint class 0, and (2,2),
    // (3,3) to joint class 1*2+1=3. NaN and masked rows contribute zero.
    // All twelve inactive slots must be overwritten as well.
    await fixture.run();
    expect(await readIntegers(classCounts.buffer)).toEqual([
      2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
    // With one class on each axis, every one of the same four valid rows
    // contributes to joint class 0. The previous class 3 must clear.
    parameters.buffer.write(
      getGPUBivariateClassificationParameterValues({classCountX: 1, classCountY: 1})
    );
    await fixture.run();
    expect(await readIntegers(classCounts.buffer)).toEqual([
      4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
    mask.buffer.write(new Uint32Array(6));
    await fixture.run();
    expect(await readIntegers(classCounts.buffer)).toEqual(new Array(16).fill(0));
  } finally {
    fixture.destroy();
  }
});
