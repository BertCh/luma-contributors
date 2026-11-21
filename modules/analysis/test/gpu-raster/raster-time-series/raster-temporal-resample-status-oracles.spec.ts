// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterTemporalResample,
  getGPURasterTemporalResampleParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';
import {getGPURasterCentredFractionalYears} from '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis';

it('retains millisecond endpoints supplied by the public centred-year helper', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const times = BigInt64Array.of(0n, 82800000n);
  const years = getGPURasterCentredFractionalYears(times, 'millisecond');
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 2,
      values: harness.floats([7, 11]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({
          gridStart: years[0],
          gridStep: years[1] - years[0],
          maximumGap: 0.001
        })
      ),
      dateAxis: {
        acquisitionTimes: harness.words(times),
        temporalContext: {
          unit: 'millisecond',
          epoch: 0n,
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        }
      },
      output: {
        values: harness.output('values', 'float32', 2),
        status: harness.output('status', 'uint32', 2)
      }
    })
  );
  // Both grid points are original observations; neither needs interpolation across the long gap.
  expect(result['status']).toEqual([0, 0]);
  expect(result['values']).toEqual([7, 11]);
});

it('preserves both observed endpoints while rejecting gap interiors and extrapolation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 5,
      values: harness.floats([7, 11]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({gridStart: -2, gridStep: 1, maximumGap: 1})
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.of(0n, 63113904n)),
        temporalContext: {
          unit: 'second',
          epoch: 0n,
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        }
      },
      output: {
        values: harness.output('values', 'float32', 5),
        status: harness.output('status', 'uint32', 5)
      }
    })
  );
  // The two dates are exactly -1/+1 centred years. Endpoint observations are not interpolation;
  // only grid point zero lies inside the excessive span, while -2/+2 are outside the range.
  // Exact float32 integers need no numeric tolerance, including canonical missing results.
  expect(result['status']).toEqual([1, 0, 2, 0, 1]);
  expect(result['values']).toEqual([NaN, 7, NaN, 11, NaN]);
});

it('retains an explicit final observation on a 24-point millisecond grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const times = BigInt64Array.of(0n, 950400000n);
  const years = getGPURasterCentredFractionalYears(times, 'millisecond');
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 24,
      gridEnd: years[1],
      values: harness.floats([7, 11]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({
          gridStart: years[0],
          gridStep: (years[1] - years[0]) / 23,
          maximumGap: 1
        })
      ),
      dateAxis: {
        acquisitionTimes: harness.words(times),
        temporalContext: {
          unit: 'millisecond',
          epoch: 0n,
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        }
      },
      output: {
        values: harness.output('values', 'float32', 24),
        status: harness.output('status', 'uint32', 24)
      }
    })
  );
  expect(result['status']).toEqual(new Array(24).fill(0));
  expect(result['values'][0]).toBe(7);
  expect(result['values'][23]).toBe(11);
  // Independent interpolation of the two observations on an inclusive 24-point grid.
  // Absolute 2e-6 allows float32 date, step and interpolation arithmetic.
  result['values'].forEach((value, point) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - (7 + (4 * point) / 23))).toBeLessThanOrEqual(2e-6);
  });
});
