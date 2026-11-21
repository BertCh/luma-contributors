// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCalendarBuckets,
  getGPUCalendarBucketsParameterValues
} from '../../../src/dataframe-analysis/calendar-buckets/index';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('calendar counts rewrite every poisoned hour-weekday bin and clear bins vacated by an offset change', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Int64 millisecond words: epoch, +1 hour, masked +2 hours, Int64 max.
    // 1970-01-01 is Thursday (weekday 3 when Monday is zero).
    const timestampWords = fixture.view(
      'uint32',
      new Uint32Array([0, 0, 3600000, 0, 7200000, 0, 0xffffffff, 0x7fffffff])
    );
    const parameterWords = fixture.view('uint32', new Uint32Array([0, 0]));
    const mask = fixture.view('uint32', new Uint32Array([1, 1, 0, 1]));
    const hourWeekdayCounts = fixture.view('uint32', new Uint32Array(168).fill(0xffffffff));
    fixture.graph.add(
      new GPUCalendarBuckets({
        unit: 'millisecond',
        timestamps: fixture.graph.createDataView(timestampWords.view.buffer, {
          format: 'uint32x2',
          length: 4
        }),
        mask: mask.view,
        parameters: fixture.graph.createDataView(parameterWords.view.buffer, {
          format: 'sint32',
          length: 2
        }),
        output: {hourWeekdayCounts: hourWeekdayCounts.view}
      })
    );
    const expected = new Array<number>(168).fill(0);
    expected[72] = 1;
    expected[73] = 1;
    await fixture.run();
    expect(await readIntegers(hourWeekdayCounts.buffer)).toEqual(expected);
    // +23 hours maps epoch to Thursday 23:00 and +1h to Friday 00:00.
    // The old Thursday 00:00/01:00 bins must become zero in this encoding.
    parameterWords.buffer.write(getGPUCalendarBucketsParameterValues(1380, 0));
    const shifted = new Array<number>(168).fill(0);
    shifted[95] = 1;
    shifted[96] = 1;
    await fixture.run();
    expect(await readIntegers(hourWeekdayCounts.buffer)).toEqual(shifted);
    mask.buffer.write(new Uint32Array(4));
    await fixture.run();
    expect(await readIntegers(hourWeekdayCounts.buffer)).toEqual(new Array(168).fill(0));
  } finally {
    fixture.destroy();
  }
});
