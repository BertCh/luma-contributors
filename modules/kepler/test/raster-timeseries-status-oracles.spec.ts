// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {runRasterTimeSeries} from '../src/analysis/raster-timeseries';

for (const fixture of [
  {
    name: 'two observed endpoints across a long millisecond gap',
    lastTime: 82800000,
    count: 2
  },
  {
    name: '419-day reciprocal-rounding millisecond endpoints',
    lastTime: 419 * 86400000,
    count: 2
  },
  {
    name: 'default 24-point inclusive millisecond grid',
    lastTime: 950400000,
    count: 24
  }
] as const) {
  test(`Kepler resample retains ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runRasterTimeSeries(
      device,
      {
        cellCount: 1,
        sliceCount: 2,
        bandCount: 1,
        times: [0, fixture.lastTime],
        values: Float32Array.of(7, 11)
      },
      'resample',
      fixture.count === 2 ? {gridCount: 2, maximumGap: 0.001} : undefined
    );
    const values = Array.from(result.columns['resampled'].values);
    expect(Array.from(result.columns['status'].values)).toEqual(new Array(fixture.count).fill(0));
    expect(values).toHaveLength(fixture.count);
    expect(values[0]).toBe(7);
    expect(values[fixture.count - 1]).toBe(11);
    // The caller promises an inclusive grid. The independent two-observation line is
    // 7+4*i/(count-1); 2e-6 absolute tolerance covers float32 interior arithmetic.
    values.forEach((value, point) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - (7 + (4 * point) / (fixture.count - 1)))).toBeLessThanOrEqual(2e-6);
    });
  });
}
