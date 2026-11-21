// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('class-break counts include every class slot and clear poisoned padding and prior counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const values = fixture.view('float32', new Float32Array([0, 0, 2, 4, NaN]));
    const mask = fixture.view('uint32', new Uint32Array(5).fill(1));
    const parameters = fixture.view(
      'float32',
      getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 4)
    );
    const breaks = fixture.view('float32', new Float32Array(5));
    const classCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
    const classCounts = fixture.view('uint32', new Uint32Array(4).fill(0xffffffff));
    fixture.graph.add(
      new GPUClassBreaks({
        values: values.view,
        mask: mask.view,
        parameters: parameters.view,
        maximumClassCount: 4,
        methods: ['equal-interval'],
        output: {breaks: breaks.view, classCount: classCount.view, classCounts: classCounts.view}
      })
    );
    // Two equal intervals on [0,4] have edges [0,2,4]. The two zeros
    // belong to class 0; 2 and the inclusive upper endpoint 4 to class 1.
    // NaN is excluded. Slots 2 and 3 are inactive and must be zero.
    await fixture.run();
    expect(await readIntegers(classCount.buffer)).toEqual([2]);
    expect(await readIntegers(classCounts.buffer)).toEqual([2, 2, 0, 0]);
    mask.buffer.write(new Uint32Array(5));
    await fixture.run();
    expect(await readIntegers(classCount.buffer)).toEqual([0]);
    expect(await readIntegers(classCounts.buffer)).toEqual([0, 0, 0, 0]);
    // A constant finite column has exactly one class containing all five rows.
    values.buffer.write(new Float32Array(5).fill(7));
    mask.buffer.write(new Uint32Array(5).fill(1));
    await fixture.run();
    expect(await readIntegers(classCount.buffer)).toEqual([1]);
    expect(await readIntegers(classCounts.buffer)).toEqual([5, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
