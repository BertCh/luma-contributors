// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterBestAvailablePixel,
  GPURasterGeometricMedianComposite,
  GPURasterMedoidComposite,
  GPURasterPhenologyMetrics,
  GPURasterWhittakerSmoother,
  GPU_RASTER_PHENOLOGY_FLAG,
  GPU_RASTER_PHENOLOGY_METRIC,
  GPU_RASTER_PHENOLOGY_METRIC_COUNT,
  getGPURasterBestAvailablePixelParameterValues,
  getGPURasterGeometricMedianCompositeParameterValues,
  getGPURasterPhenologyMetricsParameterValues,
  getGPURasterWhittakerSmootherParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts05-compositing-numpy.json';

const NO_SLICE = 0xffffffff;
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

it('RTS-05 medoid matches the nanmedoid definition, including NaN, QA, ties and empty cells', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const {sliceCount, bandCount, cells} = reference.medoid;
  const result = await harness.run(
    new GPURasterMedoidComposite({
      cellCount: cells.length,
      sliceCount,
      bandCount,
      values: harness.floats(cells.flatMap(cell => toFloat(cell.values))),
      validity: harness.integers(cells.flatMap(cell => cell.validity)),
      output: {
        composite: harness.output('composite', 'float32', cells.length * bandCount),
        sliceIndex: harness.output('sliceIndex', 'uint32', cells.length),
        status: harness.output('status', 'uint32', cells.length)
      }
    })
  );
  cells.forEach((cell, index) => {
    const expectedSlice = cell.expectedSlice < 0 ? NO_SLICE : cell.expectedSlice;
    expect(result['sliceIndex'][index], `cell ${index} slice`).toBe(expectedSlice);
    expect(result['status'][index]).toBe(cell.expectedSlice < 0 ? 1 : 0);
    const actual = result['composite'].slice(index * bandCount, (index + 1) * bandCount);
    if (cell.expectedSlice < 0) {
      expect(actual.every(Number.isNaN)).toBe(true);
    } else {
      const source = toFloat(cell.values)
        .slice(cell.expectedSlice * bandCount, (cell.expectedSlice + 1) * bandCount)
        .map(Math.fround);
      expect(getMaximumAbsoluteDifference(actual, source)).toBe(0);
    }
  });
});

function runGeometricMedian(
  harness: ReturnType<typeof createRasterTimeSeriesHarness>,
  iterationCount: number
) {
  const {sliceCount, bandCount, cells, distanceEpsilon} = reference.geometricMedian;
  return harness.run(
    new GPURasterGeometricMedianComposite({
      cellCount: cells.length,
      sliceCount,
      bandCount,
      iterationCount,
      values: harness.floats(cells.flatMap(cell => toFloat(cell.values))),
      validity: harness.integers(cells.flatMap(cell => cell.validity)),
      parameters: harness.floats(
        getGPURasterGeometricMedianCompositeParameterValues({distanceEpsilon})
      ),
      output: {
        composite: harness.output('composite', 'float32', cells.length * bandCount),
        status: harness.output('status', 'uint32', cells.length),
        lastStep: harness.output('lastStep', 'float32', cells.length)
      }
    })
  );
}

it('RTS-05 geometric median matches binary64 Weiszfeld and the true minimiser', async () => {
  const device = await getDevice();
  const {bandCount, cells, iterationCount} = reference.geometricMedian;
  const result = await runGeometricMedian(createRasterTimeSeriesHarness(device), iterationCount);
  cells.forEach((cell, index) => {
    const actual = result['composite'].slice(index * bandCount, (index + 1) * bandCount);
    if (!cell.expected) {
      expect(result['status'][index]).toBe(1);
      expect(actual.every(Number.isNaN)).toBe(true);
      return;
    }
    expect(result['status'][index]).toBe(0);
    expect(getMaximumAbsoluteDifference(actual, cell.expected), `cell ${index}`).toBeLessThan(2e-5);
    // The sum of distances at the float32 result is within 1e-5 of the SciPy minimum.
    const points = toFloat(cell.values);
    let total = 0;
    for (let slice = 0; slice < reference.geometricMedian.sliceCount; slice++) {
      const row = points.slice(slice * bandCount, (slice + 1) * bandCount);
      if (!cell.validity[slice] || row.some(Number.isNaN)) continue;
      total += Math.hypot(...row.map((value, band) => value - actual[band]));
    }
    expect(total).toBeLessThan((cell.optimumObjective as number) * (1 + 1e-5) + 1e-6);
    expect(result['lastStep'][index]).toBeLessThan(1e-4);
  });
});

it('RTS-05 geometric median runs exactly iterationCount iterations', async () => {
  // Three Weiszfeld iterations are far from converged on the outlier cell; the binary64 reference
  // after exactly three pins the iteration count.
  const device = await getDevice();
  const {bandCount, cells} = reference.geometricMedian;
  const result = await runGeometricMedian(createRasterTimeSeriesHarness(device), 3);
  cells.forEach((cell, index) => {
    if (!cell.expectedAfterThreeIterations) return;
    const actual = result['composite'].slice(index * bandCount, (index + 1) * bandCount);
    expect(
      getMaximumAbsoluteDifference(actual, cell.expectedAfterThreeIterations),
      `cell ${index}`
    ).toBeLessThan(2e-5);
    expect(
      Math.abs(result['lastStep'][index] - (cell.expectedLastStepAfterThreeIterations as number))
    ).toBeLessThan(2e-5);
  });
});

it('RTS-05 best-available-pixel matches the Griffiths score definition', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const {sliceCount, bandCount, cells, dayOfYear, year, parameters} = reference.bestAvailablePixel;
  const cloud = cells.flatMap(cell =>
    cell.cloudDistance.map(distance =>
      distance === 'inf' ? Number.POSITIVE_INFINITY : Number(distance ?? Number.NaN)
    )
  );
  const result = await harness.run(
    new GPURasterBestAvailablePixel({
      cellCount: cells.length,
      sliceCount,
      bandCount,
      values: harness.floats(cells.flatMap(cell => toFloat(cell.values))),
      validity: harness.integers(cells.flatMap(cell => cell.validity)),
      cloudDistance: harness.floats(cloud),
      sliceAttributes: harness.floats(
        dayOfYear.flatMap((day, slice) => [Number(day), Number(year[slice])])
      ),
      parameters: harness.floats(getGPURasterBestAvailablePixelParameterValues(parameters)),
      output: {
        bestSlice: harness.output('bestSlice', 'uint32', cells.length),
        bestScore: harness.output('bestScore', 'float32', cells.length),
        composite: harness.output('composite', 'float32', cells.length * bandCount)
      }
    })
  );
  cells.forEach((cell, index) => {
    expect(result['bestSlice'][index], `cell ${index}`).toBe(
      cell.expectedSlice < 0 ? NO_SLICE : cell.expectedSlice
    );
    if (cell.expectedSlice < 0) {
      expect(Number.isNaN(result['bestScore'][index])).toBe(true);
      return;
    }
    expect(Math.abs(result['bestScore'][index] - (cell.expectedScore as number))).toBeLessThan(
      1e-6
    );
    const source = toFloat(cell.values)
      .slice(cell.expectedSlice * bandCount, (cell.expectedSlice + 1) * bandCount)
      .map(Math.fround);
    expect(
      getMaximumAbsoluteDifference(
        result['composite'].slice(index * bandCount, (index + 1) * bandCount),
        source
      )
    ).toBe(0);
  });
});

const PHENOLOGY_TOLERANCE = 2e-4;

it('RTS-05 phenology metrics match the threshold-crossing reference on irregular dates', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const {sliceCount, cells, settings, timesSeconds} = reference.phenology;
  const seasonSlots = 2;
  const result = await harness.run(
    new GPURasterPhenologyMetrics({
      cellCount: cells.length,
      sliceCount,
      maximumSeasonCount: seasonSlots,
      smoothed: harness.floats(cells.flatMap(cell => toFloat(cell.curve))),
      parameters: harness.floats(getGPURasterPhenologyMetricsParameterValues(settings)),
      dateAxis: {
        acquisitionTimes: harness.words(BigInt64Array.from(timesSeconds, BigInt)),
        temporalContext: CONTEXT
      },
      output: {
        metrics: harness.output(
          'metrics',
          'float32',
          cells.length * seasonSlots * GPU_RASTER_PHENOLOGY_METRIC_COUNT
        ),
        flags: harness.output('flags', 'uint32', cells.length * seasonSlots),
        seasonCount: harness.output('seasonCount', 'uint32', cells.length),
        status: harness.output('status', 'uint32', cells.length)
      }
    })
  );
  cells.forEach((cell, index) => {
    expect(result['status'][index], `${cell.name} status`).toBe(cell.expectedStatus);
    expect(result['seasonCount'][index], `${cell.name} seasons`).toBe(cell.expectedSeasons.length);
    for (let season = 0; season < seasonSlots; season++) {
      const slot = index * seasonSlots + season;
      const expected = cell.expectedSeasons[season] as Record<string, number | null> | undefined;
      for (const [field, position] of Object.entries(GPU_RASTER_PHENOLOGY_METRIC)) {
        const actual = result['metrics'][slot * GPU_RASTER_PHENOLOGY_METRIC_COUNT + position];
        const wanted = expected ? (expected[field] ?? Number.NaN) : Number.NaN;
        if (Number.isNaN(wanted)) {
          expect(Number.isNaN(actual), `${cell.name} season ${season} ${field}`).toBe(true);
        } else {
          expect(Math.abs(actual - wanted), `${cell.name} season ${season} ${field}`).toBeLessThan(
            PHENOLOGY_TOLERANCE
          );
        }
      }
      expect(result['flags'][slot], `${cell.name} season ${season} flags`).toBe(
        expected ? (expected['flags'] as number) : 0
      );
    }
  });
  // Series-edge flags are exercised by the two clipped fixtures.
  const startClipped = cells.findIndex(cell => cell.name === 'high-at-series-start');
  expect(
    result['flags'][startClipped * seasonSlots] & GPU_RASTER_PHENOLOGY_FLAG.startAtSeriesStart
  ).toBe(1);
});

it('RTS-05 phenology metrics equal the hand-computed triangle season', async () => {
  // z = 0,1,2,3,4,3,2,1,0 on unit slots. Start level 1 is crossed at t = 1; end level 0.6 at
  // t = 7.4. Slopes +1 and -1; integral = 15 + 0.5 * (1 + 0.6) * 0.4 = 15.32.
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterPhenologyMetrics({
      cellCount: 1,
      sliceCount: 9,
      maximumSeasonCount: 1,
      smoothed: harness.floats([0, 1, 2, 3, 4, 3, 2, 1, 0]),
      parameters: harness.floats(
        getGPURasterPhenologyMetricsParameterValues({
          startFraction: 0.25,
          endFraction: 0.15,
          minimumAmplitude: 0.1
        })
      ),
      output: {
        metrics: harness.output('metrics', 'float32', GPU_RASTER_PHENOLOGY_METRIC_COUNT),
        flags: harness.output('flags', 'uint32', 1),
        seasonCount: harness.output('seasonCount', 'uint32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  const metric = (name: keyof typeof GPU_RASTER_PHENOLOGY_METRIC) =>
    result['metrics'][GPU_RASTER_PHENOLOGY_METRIC[name]];
  expect(result['seasonCount'][0]).toBe(1);
  expect(metric('startTime')).toBeCloseTo(1, 5);
  expect(metric('peakTime')).toBe(4);
  expect(metric('endTime')).toBeCloseTo(7.4, 5);
  expect(metric('length')).toBeCloseTo(6.4, 5);
  expect(metric('amplitude')).toBe(4);
  expect(metric('leftSlope')).toBeCloseTo(1, 5);
  expect(metric('rightSlope')).toBeCloseTo(-1, 5);
  expect(metric('integral')).toBeCloseTo(15.32, 4);
});

it('RTS-05 phenology metrics read a GPURasterWhittakerSmoother output in one graph', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const sliceCount = 48;
  const noisy = Array.from({length: sliceCount}, (_, slice) => {
    const phase = (slice - 24) / 6;
    return 0.1 + 0.7 * Math.exp(-0.5 * phase * phase) + (slice % 2 ? 0.01 : -0.01);
  });
  noisy[20] = Number.NaN;
  const smoothed = harness.output('smoothed', 'float32', sliceCount);
  const result = await harness.run([
    new GPURasterWhittakerSmoother({
      cellCount: 1,
      sliceCount,
      reweightPassCount: 0,
      values: harness.floats(noisy),
      parameters: harness.floats(
        getGPURasterWhittakerSmootherParameterValues({smoothingParameter: 5})
      ),
      output: {smoothed, status: harness.output('smootherStatus', 'uint32', 1)}
    }),
    new GPURasterPhenologyMetrics({
      cellCount: 1,
      sliceCount,
      maximumSeasonCount: 1,
      smoothed,
      parameters: harness.floats(
        getGPURasterPhenologyMetricsParameterValues({
          startFraction: 0.25,
          endFraction: 0.15,
          minimumAmplitude: 0.1
        })
      ),
      output: {
        metrics: harness.output('metrics', 'float32', GPU_RASTER_PHENOLOGY_METRIC_COUNT),
        flags: harness.output('flags', 'uint32', 1),
        seasonCount: harness.output('seasonCount', 'uint32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  ]);
  expect(result['smootherStatus'][0]).toBe(0);
  expect(result['status'][0]).toBe(0);
  expect(result['seasonCount'][0]).toBe(1);
  expect(result['metrics'][GPU_RASTER_PHENOLOGY_METRIC.peakTime]).toBeGreaterThan(22);
  expect(result['metrics'][GPU_RASTER_PHENOLOGY_METRIC.peakTime]).toBeLessThan(26);
});
