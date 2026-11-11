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
import {requireTestDevice} from '../../utils/require-test-device';
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
