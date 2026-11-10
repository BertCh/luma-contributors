// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection,
  GPURasterSeasonalBreak,
  GPURasterTemporal,
  getGPURasterCentredFractionalYears,
  GPU_RASTER_SECONDS_PER_YEAR
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';

const CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};
const YEAR_MILLISECONDS = GPU_RASTER_SECONDS_PER_YEAR * 1000;
const DAY_MILLISECONDS = 86400000;

function makeIrregularTimes(count: number): BigInt64Array {
  // Irregular spacing from a fixed recurrence, independent of the code under test.
  const times = new BigInt64Array(count);
  let day = 0;
  for (let index = 0; index < count; index++) {
    times[index] = 1500000000000n + BigInt(day * DAY_MILLISECONDS);
    day += 5 + ((index * 7) % 11);
  }
  return times;
}

function getCentredYears(times: BigInt64Array): number[] {
  const midpoint = times[0] + (times[times.length - 1] - times[0]) / 2n;
  return Array.from(times, time => Number(time - midpoint) / YEAR_MILLISECONDS);
}

function requireDevice() {
  return getWebGPUTestDevice().then(device => {
    requireTestDevice(device);
    return device;
  });
}

describe('TC-07 irregular date axis', () => {
  it('centred fractional years match an exact Int64 computation at millisecond and nanosecond resolution', () => {
    const milliseconds = makeIrregularTimes(9);
    const fractionalYears = getGPURasterCentredFractionalYears(milliseconds, 'millisecond');
    const expected = getCentredYears(milliseconds);
    fractionalYears.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 5));
    expect(fractionalYears[0]).toBeCloseTo(-fractionalYears[fractionalYears.length - 1], 5);
    const nanoseconds = BigInt64Array.from(milliseconds, time => time * 1000000n);
    const fractionalYearsNs = getGPURasterCentredFractionalYears(nanoseconds, 'nanosecond');
    fractionalYearsNs.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 5));
  });

  it('Theil-Sen slope uses elapsed years and recovers a known trend per year', async () => {
    const device = await requireDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const times = makeIrregularTimes(20);
    const years = getCentredYears(times);
    const slope = -3.25;
    const values = years.map(year => 12 + slope * year);
    const senWithAxis = harness.output('axis', 'float32', 1);
    const senWithoutAxis = harness.output('slots', 'float32', 1);
    const parameters = harness.floats(
      getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1, splitSlice: 1})
    );
    const axis = {acquisitionTimes: harness.words(times), temporalContext: CONTEXT};
    const result = await harness.run([
      new GPUChangeDetection({
        id: 'with-axis',
        cellCount: 1,
        sliceCount: 20,
        slices: harness.floats(values),
        parameters,
        dateAxis: axis,
        output: {senSlope: senWithAxis}
      }),
      new GPUChangeDetection({
        id: 'no-axis',
        cellCount: 1,
        sliceCount: 20,
        slices: harness.floats(values),
        parameters,
        output: {senSlope: senWithoutAxis}
      })
    ]);
    expect(result['axis'][0]).toBeCloseTo(slope, 3);
    expect(Math.abs(result['slots'][0] - slope)).toBeGreaterThan(0.5);
  });

  it('Savitzky-Golay reproduces a polynomial exactly in real time, which slot indices do not', async () => {
    const device = await requireDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const sliceCount = 25;
    const times = makeIrregularTimes(sliceCount);
    const years = getCentredYears(times);
    const polynomial = (year: number) => 1.5 - 2 * year + 0.75 * year * year + 0.125 * year ** 3;
    const values = years.map(polynomial);
    const output = () => ({
      values: harness.output(`values-${serial}`, 'float32', sliceCount),
      counts: harness.output(`counts-${serial}`, 'uint32', sliceCount),
      chosenSlice: harness.output(`chosen-${serial}`, 'uint32', sliceCount),
      status: harness.output(`status-${serial++}`, 'uint32', sliceCount)
    });
    let serial = 0;
    const axis = {acquisitionTimes: harness.words(times), temporalContext: CONTEXT};
    const result = await harness.run([
      new GPURasterTemporal({
        id: 'axis',
        cellCount: 1,
        sliceCount,
        values: harness.floats(values),
        operation: 'savitzky-golay',
        windowLength: 7,
        polynomialOrder: 3,
        dateAxis: axis,
        output: output()
      }),
      new GPURasterTemporal({
        id: 'slots',
        cellCount: 1,
        sliceCount,
        values: harness.floats(values),
        operation: 'savitzky-golay',
        windowLength: 7,
        polynomialOrder: 3,
        output: output()
      })
    ]);
    let axisError = 0;
    let slotError = 0;
    for (let slice = 0; slice < sliceCount; slice++) {
      expect(result['status-0'][slice]).toBe(0);
      axisError = Math.max(axisError, Math.abs(result['values-0'][slice] - values[slice]));
      slotError = Math.max(slotError, Math.abs(result['values-1'][slice] - values[slice]));
    }
    // Real-time fitting is exact on a cubic up to float32 rounding; slot fitting is not.
    expect(axisError).toBeLessThan(2e-3);
    expect(slotError).toBeGreaterThan(5 * axisError);
  });

  const seasonalCoefficients = {intercept: 4, slope: 1.2, cosine: 2.5, sine: -1.5, jump: 3};

  function makeSeasonalSeries(sliceCount: number, breakSlice: number) {
    const times = makeIrregularTimes(sliceCount);
    const years = getCentredYears(times);
    const {intercept, slope, cosine, sine, jump} = seasonalCoefficients;
    const values = years.map(
      (year, slice) =>
        intercept +
        slope * year +
        cosine * Math.cos(2 * Math.PI * year) +
        sine * Math.sin(2 * Math.PI * year) +
        (slice >= breakSlice ? jump : 0)
    );
    return {times, years, values};
  }

  it('seasonal break recovers trend per year, seasonal amplitude, jump and break date on irregular acquisitions', async () => {
    const device = await requireDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const sliceCount = 80;
    const breakSlice = 47;
    const {times, values} = makeSeasonalSeries(sliceCount, breakSlice);
    const result = await harness.run(
      new GPURasterSeasonalBreak({
        cellCount: 1,
        sliceCount,
        period: 1,
        minimumSegmentLength: 6,
        dateAxis: {acquisitionTimes: harness.words(times), temporalContext: CONTEXT},
        values: harness.floats(values),
        output: {
          coefficients: harness.output('coefficients', 'float32', 5),
          components: harness.output('components', 'float32', sliceCount * 3),
          breakIndex: harness.output('breakIndex', 'uint32', 1),
          improvement: harness.output('improvement', 'float32', 1),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['status'][0]).toBe(0);
    expect(result['breakIndex'][0]).toBe(breakSlice);
    const {intercept, slope, cosine, sine, jump} = seasonalCoefficients;
    [intercept, slope, cosine, sine, jump].forEach((expected, index) =>
      expect(result['coefficients'][index]).toBeCloseTo(expected, 2)
    );
    for (let slice = 0; slice < sliceCount; slice++) {
      const combined = result['components'][slice * 3] + result['components'][slice * 3 + 1];
      expect(combined).toBeCloseTo(values[slice], 2);
    }
  });

  it('extended seasonal interpolation across a gap uses elapsed time, not slice fractions', async () => {
    const device = await requireDevice();
    const harness = createRasterTimeSeriesHarness(device);
    // A long acquisition gap of 200 days sits between slices 15 and 16; slices 12 to 18 are missing,
    // so interpolating by slice fraction would misplace the filled values by months.
    const sliceCount = 48;
    const times = new BigInt64Array(sliceCount);
    let day = 0;
    for (let slice = 0; slice < sliceCount; slice++) {
      times[slice] = 1500000000000n + BigInt(day * DAY_MILLISECONDS);
      day += slice === 15 ? 200 : 16;
    }
    const years = getCentredYears(times);
    const slope = 2.75;
    const values = years.map(year => 3 + slope * year);
    const withGaps = values.map((value, slice) =>
      slice >= 12 && slice <= 18 ? Number.NaN : value
    );
    const result = await harness.run(
      new GPURasterSeasonalBreak({
        cellCount: 1,
        sliceCount,
        period: 1,
        minimumSegmentLength: 8,
        minimumBICGain: 1000,
        missingPolicy: 'interpolate',
        dateAxis: {acquisitionTimes: harness.words(times), temporalContext: CONTEXT},
        values: harness.floats(withGaps),
        output: {
          coefficients: harness.output('coefficients', 'float32', 5),
          components: harness.output('components', 'float32', sliceCount * 3),
          breakIndex: harness.output('breakIndex', 'uint32', 1),
          improvement: harness.output('improvement', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          diagnostics: harness.output('diagnostics', 'float32', 6)
        }
      })
    );
    expect(result['status'][0]).toBe(0);
    expect(result['breakIndex'][0]).toBe(0xffffffff);
    expect(result['diagnostics'][2]).toBe(sliceCount);
    // A straight line is recovered exactly only if the fill follows elapsed time.
    expect(result['coefficients'][1]).toBeCloseTo(slope, 2);
    expect(Math.abs(result['coefficients'][2])).toBeLessThan(5e-3);
    expect(Math.abs(result['coefficients'][3])).toBeLessThan(5e-3);
  });

  it('rejects unsupported temporal declarations at construction', async () => {
    const device = await requireDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const times = makeIrregularTimes(8);
    const acquisitionTimes = harness.words(times);
    const build = (context: GPUTemporalContext, seasonality?: 'dummy') =>
      new GPURasterSeasonalBreak({
        cellCount: 1,
        sliceCount: 8,
        period: seasonality ? 3 : 1,
        seasonality,
        minimumSegmentLength: 3,
        dateAxis: {acquisitionTimes, temporalContext: context},
        values: harness.floats(new Float32Array(8)),
        output: {
          coefficients: harness.output('c', 'float32', seasonality ? 5 : 5),
          components: harness.output('d', 'float32', 24),
          breakIndex: harness.output('b', 'uint32', 1),
          improvement: harness.output('i', 'float32', 1),
          status: harness.output('s', 'uint32', 1)
        }
      });
    expect(() => build({...CONTEXT, axis: {kind: 'regular', step: 1000}})).toThrow(
      /does not support a regular temporal axis/
    );
    expect(() => build({...CONTEXT, encoding: 'float32-relative', epoch: 0n})).toThrow(
      /does not support float32-relative/
    );
    expect(() => build(CONTEXT, 'dummy')).toThrow(/dateAxis requires harmonic seasonality/);
  });
});
