// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterHANTS,
  GPURasterTemporalResample,
  GPURasterWhittakerSmoother,
  getGPURasterHANTSParameterValues,
  getGPURasterTemporalResampleParameterValues,
  getGPURasterWhittakerSmootherParameterValues,
  GPU_RASTER_SECONDS_PER_YEAR
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts02-whittaker-hants-scipy.json';

const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

const toFloat = (values: (number | null)[]) => values.map(value => value ?? Number.NaN);

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

// float32 banded QR on centred values against SciPy binary64. Before the math correctness pass
// (raster-imagery-02) the kernel factored the normal equations and needed 1e-3 to 2.5e-2 here.
const TOLERANCE_BY_SMOOTHING: Record<number, number> = {
  20: 2e-5,
  50: 2e-5,
  200: 2e-5,
  5000: 2e-5
};

for (const entry of reference.whittaker) {
  it(`RTS-02 Whittaker ${entry.name} matches the binary64 banded solve`, async () => {
    const device = await getDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const sliceCount = reference.sliceCount;
    const result = await harness.run(
      new GPURasterWhittakerSmoother({
        cellCount: 1,
        sliceCount,
        differenceOrder: entry.order,
        reweightPassCount: entry.passes,
        reweightFunction: entry.function as 'bisquare' | 'upper-envelope',
        values: harness.floats(toFloat(entry.values)),
        parameters: harness.floats(
          getGPURasterWhittakerSmootherParameterValues({smoothingParameter: entry.smoothing})
        ),
        dateAxis: entry.timesSeconds
          ? {
              acquisitionTimes: harness.words(BigInt64Array.from(entry.timesSeconds, BigInt)),
              temporalContext: CONTEXT
            }
          : undefined,
        output: {
          smoothed: harness.output('smoothed', 'float32', sliceCount),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['status'][0]).toBe(0);
    const difference = getMaximumAbsoluteDifference(result['smoothed'], entry.expected);
    expect(difference).toBeLessThan(TOLERANCE_BY_SMOOTHING[entry.smoothing]);
  });
}

it('RTS-02 Whittaker reports insufficient support and publishes NaN', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterWhittakerSmoother({
      cellCount: 1,
      sliceCount: 6,
      differenceOrder: 2,
      reweightPassCount: 0,
      values: harness.floats([1, Number.NaN, Number.NaN, Number.NaN, Number.NaN, Number.NaN]),
      parameters: harness.floats(
        getGPURasterWhittakerSmootherParameterValues({smoothingParameter: 5})
      ),
      output: {
        smoothed: harness.output('smoothed', 'float32', 6),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['status'][0]).toBe(1);
  expect(result['smoothed'].every(Number.isNaN)).toBe(true);
});

function runHants(
  harness: ReturnType<typeof createRasterTimeSeriesHarness>,
  entry: typeof reference.hants,
  passes: number
) {
  const sliceCount = reference.sliceCount;
  return harness.run(
    new GPURasterHANTS({
      cellCount: 1,
      sliceCount,
      harmonicCount: entry.harmonicCount,
      rejectionSide: entry.side as 'low',
      rejectionPassCount: passes,
      values: harness.floats(toFloat(entry.values)),
      parameters: harness.floats(
        getGPURasterHANTSParameterValues({
          period: entry.period,
          fitErrorTolerance: entry.tolerance,
          degreeOfOverdetermination: entry.overdetermination
        })
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.from(entry.timesSeconds, BigInt)),
        temporalContext: CONTEXT
      },
      output: {
        smoothed: harness.output('smoothed', 'float32', sliceCount),
        coefficients: harness.output('coefficients', 'float32', 2 * entry.harmonicCount + 1),
        status: harness.output('status', 'uint32', 1),
        diagnostics: harness.output('diagnostics', 'uint32', 2)
      }
    })
  );
}

it('RTS-02 HANTS matches the binary64 reject-and-refit reference', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.hants;
  const result = await runHants(harness, entry, entry.passes);
  expect(result['status'][0]).toBe(0);
  expect(result['diagnostics'][0]).toBe(entry.expectedRejected);
  expect(result['diagnostics'][1]).toBe(entry.expectedPasses);
  expect(
    getMaximumAbsoluteDifference(result['coefficients'], entry.expectedCoefficients)
  ).toBeLessThan(1e-3);
  expect(getMaximumAbsoluteDifference(result['smoothed'], entry.expectedSmoothed)).toBeLessThan(
    1e-3
  );
});

it('RTS-02 HANTS reports the rejection limit and publishes its partial fit', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.hants;
  const limited = reference.hantsLimited;
  expect(limited.expectedConverged).toBe(false);
  const result = await runHants(harness, entry, limited.passes);
  expect(result['status'][0]).toBe(8);
  expect(result['diagnostics'][1]).toBe(limited.expectedPasses);
  expect(
    getMaximumAbsoluteDifference(result['coefficients'], limited.expectedCoefficients)
  ).toBeLessThan(1e-3);
  expect(getMaximumAbsoluteDifference(result['smoothed'], limited.expectedSmoothed)).toBeLessThan(
    1e-3
  );
});

it('RTS-02 temporal resample fills a regular grid and flags extrapolation and oversized gaps', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const yearSeconds = GPU_RASTER_SECONDS_PER_YEAR;
  const times = BigInt64Array.from([0, 0.2, 0.5, 1.4, 1.8, 2.0], fraction =>
    BigInt(Math.round(fraction * yearSeconds))
  );
  const values = [0, 2, 5, Number.NaN, 18, 20];
  const midpointSeconds = Number((times[0] + times[5]) / 2n);
  const toCentred = (years: number) => years - midpointSeconds / yearSeconds;
  const gridCount = 6;
  const gridStep = 0.4;
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 6,
      gridCount,
      values: harness.floats(values),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({
          gridStart: toCentred(0),
          gridStep,
          maximumGap: 1.0
        })
      ),
      dateAxis: {acquisitionTimes: harness.words(times), temporalContext: CONTEXT},
      output: {
        values: harness.output('resampled', 'float32', gridCount),
        status: harness.output('resampleStatus', 'uint32', gridCount)
      }
    })
  );
  // Independent piecewise-linear interpolation of (0, 0), (0.2, 2), (0.5, 5), (1.8, 18), (2.0, 20).
  // The 1.3-year gap between 0.5 and 1.8 exceeds maximumGap, so grid points inside it are status 2.
  const expected = [0, 4, Number.NaN, Number.NaN, 16 + 0, 20];
  expect(result['resampleStatus']).toEqual([0, 0, 2, 2, 2, 0]);
  const finitePairs = expected.flatMap((value, index) =>
    Number.isNaN(value) || result['resampleStatus'][index] !== 0
      ? []
      : [[value, result['resampled'][index]]]
  );
  for (const [wanted, actual] of finitePairs) {
    expect(actual).toBeCloseTo(wanted, 3);
  }
});

it('RT-130 interpolates in the same rounded coordinates used to bracket the grid', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  // Midpoint = 315569520 seconds. Float32 centred years are [-10.000000953674316,
  // -9.999999046325684, 10.000000953674316] with separately rounded division.
  // The common-coordinate midpoint fraction is 1/2 (value 50); exact elapsed spans
  // instead give fraction about 30.095 and overshoot. Metal may use reciprocal division,
  // so pin the interval independently of that permitted rounding choice.
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 3,
      gridCount: 1,
      values: harness.floats([0, 100, 200]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({gridStart: -10, gridStep: 1, maximumGap: 100})
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.from([0, 1, 631139040], BigInt)),
        temporalContext: CONTEXT
      },
      output: {
        values: harness.output('values', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([0]);
  expect(Number.isFinite(result['values'][0])).toBe(true);
  expect(result['values'][0]).toBeGreaterThanOrEqual(0);
  expect(result['values'][0]).toBeLessThanOrEqual(100);
});

it('RT-131 returns an exact observation before checking its neighboring gap', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  // One exact year, midpoint half a year: grid [-.5, 0] is [exact first, interpolation].
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 2,
      values: harness.floats([7, 9]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({
          gridStart: -0.5,
          gridStep: 0.5,
          maximumGap: 0.1
        })
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.from([0, 31556952], BigInt)),
        temporalContext: CONTEXT
      },
      output: {
        values: harness.output('values', 'float32', 2),
        status: harness.output('status', 'uint32', 2)
      }
    })
  );
  expect(result['values'][0]).toBe(7);
  expect(result['status']).toEqual([0, 2]);
  expect(result['values'][1]).toBeNaN();
});

it('RT-130 B3 common-coordinate midpoint is 50 even when clamping would admit the wrong fraction', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  // Both direct float32 division and reciprocal multiplication give observations
  // [-10.000000953674316, -9.99999713897705, 10.000000953674316]. The grid is their
  // first midpoint: common-coordinate fraction .5 => 50. The exact 65-second divisor
  // instead gives about .9260017 => 92.60017, which clamping cannot repair.
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 3,
      gridCount: 1,
      values: harness.floats([0, 100, 200]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({
          gridStart: -9.999999046325684,
          gridStep: 1,
          maximumGap: 100
        })
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.from([0, 65, 631139040], BigInt)),
        temporalContext: CONTEXT
      },
      output: {
        values: harness.output('values', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([0]);
  expect(result['values']).toHaveLength(1);
  expect(Number.isFinite(result['values'][0])).toBe(true);
  expect(result['values'][0]).toBeCloseTo(50, 5);
});

it('FLT-13 preserves exact Int64 target acquisitions and returns a retained hit before gap gating', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  const first = 1700000000000n;
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 3,
      gridCount: 1,
      values: harness.floats([7, 11, 13]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({gridStart: -1, gridStep: 1, maximumGap: 1e-10})
      ),
      dateAxis: {
        acquisitionTimes: harness.words(
          BigInt64Array.of(first, first + 1000n, first + 2n * 31556952000n)
        ),
        temporalContext: {...CONTEXT, unit: 'millisecond'}
      },
      targetTimes: harness.words(BigInt64Array.of(first)),
      output: {
        values: harness.output('exact', 'float32', 1),
        status: harness.output('code', 'uint32', 1)
      }
    })
  );
  expect(result['exact']).toEqual([7]);
  expect(result['code']).toEqual([0]);
});

it('FLT-13 RT-130 interpolates inside rounded fractional-year brackets', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 3,
      gridCount: 1,
      values: harness.floats([0, 100, 200]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({gridStart: -9.999999046325684, gridStep: 1})
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.of(0n, 65n, 631139040n)),
        temporalContext: CONTEXT
      },
      output: {
        values: harness.output('half', 'float32', 1),
        status: harness.output('code', 'uint32', 1)
      }
    })
  );
  // The target is the midpoint of the first two represented fractional-year coordinates.
  expect(result['half']).toEqual([50]);
  expect(result['code']).toEqual([0]);
});

it('review B1 interpolates an interior Int64 target across a low-word rollover', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 1,
      values: harness.floats([0, 100]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({gridStart: 0, gridStep: 1})
      ),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.of(4294967294n, 4294967298n)),
        temporalContext: {...CONTEXT, unit: 'millisecond'}
      },
      targetTimes: harness.words(BigInt64Array.of(4294967296n)),
      output: {
        values: harness.output('interior', 'float32', 1),
        status: harness.output('code', 'uint32', 1)
      }
    })
  );
  // Exact elapsed numerator 2 / denominator 4; the target has low word 0, high word 1.
  expect(result['interior']).toEqual([50]);
  expect(result['code']).toEqual([0]);
});

it('review earlier-date note keeps both 419-day helper endpoints inside support', async () => {
  const {getGPURasterCentredFractionalYears} = await import(
    '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis'
  );
  const harness = createRasterTimeSeriesHarness(await getDevice());
  const times = BigInt64Array.of(0n, 36201600000n);
  const years = getGPURasterCentredFractionalYears(times, 'millisecond');
  const result = await harness.run(
    new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 2,
      gridEnd: years[1],
      values: harness.floats([7, 11]),
      parameters: harness.floats(
        getGPURasterTemporalResampleParameterValues({
          gridStart: years[0],
          gridStep: years[1] - years[0]
        })
      ),
      dateAxis: {
        acquisitionTimes: harness.words(times),
        temporalContext: {...CONTEXT, unit: 'millisecond'}
      },
      output: {
        values: harness.output('endpoints', 'float32', 2),
        status: harness.output('code', 'uint32', 2)
      }
    })
  );
  expect(result['endpoints']).toEqual([7, 11]);
  expect(result['code']).toEqual([0, 0]);
});
