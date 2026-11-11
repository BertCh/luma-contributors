// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterBFASTMonitor,
  GPURasterCOLDMonitor,
  GPURasterHarmonicRegression,
  GPU_RASTER_BFAST_MONITOR_NO_BREAK,
  GPU_RASTER_COLD_MONITOR_NO_BREAK,
  getGPURasterBFASTMonitorParameterValues,
  getGPURasterCOLDMonitorParameterValues,
  getGPURasterHarmonicRegressionParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts04-change-monitoring-numpy.json';

const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

const toFloat = (values: (number | null)[]) => values.map(value => value ?? Number.NaN);
const flattenRows = (rows: (number | null)[][]) => rows.flatMap(row => toFloat(row));

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function createAxis(harness: ReturnType<typeof createRasterTimeSeriesHarness>, seconds: number[]) {
  return {
    acquisitionTimes: harness.words(BigInt64Array.from(seconds, BigInt)),
    temporalContext: CONTEXT
  };
}

for (const entry of reference.harmonicRegression.cases) {
  it(`RTS-04 harmonic regression matches the binary64 fit with ${entry.harmonicCount} harmonics`, async () => {
    const device = await getDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const sliceCount = reference.harmonicRegression.values.length;
    const columns = 2 + 2 * entry.harmonicCount;
    const result = await harness.run(
      new GPURasterHarmonicRegression({
        cellCount: 1,
        sliceCount,
        values: harness.floats(toFloat(reference.harmonicRegression.values)),
        parameters: harness.floats(getGPURasterHarmonicRegressionParameterValues({period: 1})),
        harmonicCount: entry.harmonicCount,
        dateAxis: createAxis(harness, reference.harmonicRegression.timesSeconds),
        output: {
          coefficients: harness.output('coefficients', 'float32', columns),
          summary: harness.output('summary', 'uint32', 2),
          fitted: harness.output('fitted', 'float32', sliceCount),
          rmse: harness.output('rmse', 'float32', 1)
        }
      })
    );
    expect(result['summary'][0]).toBe(0);
    expect(result['summary'][1]).toBe(entry.observationCount);
    expect(getMaximumAbsoluteDifference(result['coefficients'], entry.coefficients)).toBeLessThan(
      5e-4
    );
    expect(getMaximumAbsoluteDifference(result['fitted'], entry.fitted)).toBeLessThan(5e-4);
    expect(Math.abs(result['rmse'][0] - entry.rmse)).toBeLessThan(5e-4);
  });
}

it('RTS-04 harmonic regression reports insufficient support and publishes NaN', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const sliceCount = 8;
  const seconds = Array.from({length: sliceCount}, (_, slice) => 1_000_000 + slice * 1_000_000);
  const result = await harness.run(
    new GPURasterHarmonicRegression({
      cellCount: 1,
      sliceCount,
      values: harness.floats([1, Number.NaN, 2, Number.NaN, Number.NaN, 3, Number.NaN, 4]),
      parameters: harness.floats(getGPURasterHarmonicRegressionParameterValues({period: 1})),
      harmonicCount: 1,
      dateAxis: createAxis(harness, seconds),
      output: {
        coefficients: harness.output('coefficients', 'float32', 4),
        summary: harness.output('summary', 'uint32', 2)
      }
    })
  );
  // Four valid observations cannot fit four coefficients with residual degrees of freedom.
  expect(result['summary'][0]).toBe(1);
  expect(result['coefficients'].every(Number.isNaN)).toBe(true);
});

type ColdCase = (typeof reference.cold)[number];

function createColdMonitor(
  harness: ReturnType<typeof createRasterTimeSeriesHarness>,
  entry: Omit<ColdCase, 'values'> & {values: (number | null)[][]},
  overrides: {cellCount?: number; maximumFitCount?: number; maximumSegmentCount?: number} = {}
) {
  const {config} = entry;
  const cellCount = overrides.cellCount ?? 1;
  const sliceCount = entry.timesSeconds.length;
  const maximumSegmentCount = overrides.maximumSegmentCount ?? config.maximumSegmentCount;
  const flat = flattenRows(entry.values);
  const values = Array.from({length: cellCount}, () => flat).flat();
  const entries = cellCount * maximumSegmentCount;
  return new GPURasterCOLDMonitor({
    cellCount,
    sliceCount,
    bandCount: config.bandCount,
    values: harness.floats(values),
    mask: entry.mask
      ? harness.integers(Array.from({length: cellCount}, () => entry.mask as number[]).flat())
      : undefined,
    parameters: harness.floats(
      getGPURasterCOLDMonitorParameterValues({
        period: config.period,
        changeThreshold: config.changeThreshold,
        outlierThreshold: config.outlierThreshold,
        minimumMeanCosine: config.minimumMeanCosine,
        minimumSpanYears: config.minimumSpanYears,
        maximumInitializationGapYears: config.maximumInitializationGapYears,
        minimumRMSE: config.minimumRMSE
      })
    ),
    dateAxis: createAxis(harness, entry.timesSeconds),
    minimumObservations: config.minimumObservations,
    consecutiveCount: config.consecutive,
    maximumCoefficientCount: config.maximumCoefficientCount as 4 | 6 | 8,
    maximumSegmentCount,
    maximumFitCount: overrides.maximumFitCount ?? config.maximumFitCount,
    output: {
      segments: harness.output('segments', 'uint32', entries * 4),
      coefficients: harness.output('coefficients', 'float32', entries * config.bandCount * 8),
      segmentStatistics: harness.output('statistics', 'float32', entries * config.bandCount * 2),
      cellSummary: harness.output('summary', 'uint32', cellCount * 3)
    }
  });
}

function readSegments(
  result: Record<string, number[]>,
  segmentCount: number,
  config: ColdCase['config']
) {
  const segments = [];
  for (let segment = 0; segment < segmentCount; segment++) {
    const base = segment * 4;
    const bandBase = segment * config.bandCount;
    const breakSlice = result['segments'][base + 2];
    segments.push({
      startSlice: result['segments'][base],
      endSlice: result['segments'][base + 1],
      breakSlice: breakSlice === GPU_RASTER_COLD_MONITOR_NO_BREAK ? null : breakSlice,
      coefficientCount: result['segments'][base + 3],
      coefficients: Array.from({length: config.bandCount}, (_, band) =>
        Array.from(
          {length: 8},
          (__, column) => result['coefficients'][(bandBase + band) * 8 + column]
        )
      ),
      rmse: Array.from(
        {length: config.bandCount},
        (_, band) => result['statistics'][(bandBase + band) * 2]
      ),
      magnitude: Array.from(
        {length: config.bandCount},
        (_, band) => result['statistics'][(bandBase + band) * 2 + 1]
      )
    });
  }
  return segments;
}

// float32 QR over bands of magnitude 0.2-0.6 against binary64: coefficients agree to about 1e-3.
const COEFFICIENT_TOLERANCE = 2e-3;
const RMSE_TOLERANCE = 1e-3;

for (const entry of reference.cold) {
  it(`RTS-04 simplified COLD matches the binary64 reference: ${entry.name}`, async () => {
    const device = await getDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const result = await harness.run(createColdMonitor(harness, entry));
    expect(result['summary'][1]).toBe(entry.simplified.status);
    expect(result['summary'][0]).toBe(entry.simplified.segments.length);
    expect(result['summary'][2]).toBe(entry.simplified.fitCount);
    const segments = readSegments(result, entry.simplified.segments.length, entry.config);
    for (const [index, expected] of entry.simplified.segments.entries()) {
      const segment = segments[index];
      expect(segment.startSlice).toBe(expected.startSlice);
      expect(segment.endSlice).toBe(expected.endSlice);
      expect(segment.breakSlice).toBe(expected.breakSlice);
      expect(segment.coefficientCount).toBe(expected.coefficientCount);
      expect(
        getMaximumAbsoluteDifference(segment.coefficients.flat(), expected.coefficients.flat())
      ).toBeLessThan(COEFFICIENT_TOLERANCE);
      expect(getMaximumAbsoluteDifference(segment.rmse, expected.rmse)).toBeLessThan(
        RMSE_TOLERANCE
      );
      if (expected.changeMagnitude) {
        expect(
          getMaximumAbsoluteDifference(segment.magnitude, expected.changeMagnitude)
        ).toBeLessThan(RMSE_TOLERANCE);
      }
    }
  });
}

/** Each case records a published COLD element the simplified monitor omits. */
const omissionCases: {name: string; toggle: string}[] = [
  {name: 'tmask-cloud-in-window', toggle: 'tmask'},
  {name: 'lasso-shrinkage', toggle: 'lasso'},
  {name: 'refit-cadence-drift', toggle: 'every-observation-refit'},
  {name: 'temporal-rmse-season-noise', toggle: 'temporal-rmse'}
];

for (const {name, toggle} of omissionCases) {
  it(`RTS-04 simplified COLD differs from full COLD behaviour: ${toggle}`, async () => {
    const entry = reference.cold.find(item => item.name === name) as ColdCase;
    const variant = entry.variants.find(item => item.toggle === toggle)?.result;
    expect(variant).toBeDefined();
    const device = await getDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const result = await harness.run(createColdMonitor(harness, entry));
    const segments = readSegments(result, result['summary'][0], entry.config);
    if (toggle === 'tmask') {
      // Without Tmask the cloud makes the stability test shift the first segment later.
      expect(segments[0].startSlice).toBe(entry.simplified.segments[0].startSlice);
      expect(segments[0].startSlice).not.toBe(variant?.segments[0].startSlice);
    } else if (toggle === 'lasso') {
      // The output is the OLS fit, so harmonic and trend terms are not shrunk.
      const gap = getMaximumAbsoluteDifference(
        segments[0].coefficients.flatMap(row => row.slice(1)),
        (variant?.segments[0].coefficients as number[][]).flatMap(row => row.slice(1))
      );
      expect(gap).toBeGreaterThan(
        reference.cold.find(item => item.name === name)?.minimumCoefficientGap as number
      );
    } else {
      const breaks = segments.map(segment => segment.breakSlice);
      expect(breaks).toEqual(entry.simplified.segments.map(segment => segment.breakSlice));
      expect(breaks).not.toEqual(variant?.segments.map(segment => segment.breakSlice));
    }
  });
}

it('RTS-04 COLD keeps cells independent in one dispatch', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.cold[0];
  const result = await harness.run(createColdMonitor(harness, entry, {cellCount: 2}));
  const words = entry.config.maximumSegmentCount * 4;
  expect(result['segments'].slice(0, words)).toEqual(result['segments'].slice(words, words * 2));
});

it('RTS-04 COLD publishes partial output and flags the fit budget', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.cold[0];
  const result = await harness.run(createColdMonitor(harness, entry, {maximumFitCount: 1}));
  // One fit initializes the window; the model fit is refused: bits 8 (budget) and 1 (no segment).
  expect(result['summary'][1]).toBe(9);
  expect(result['summary'][0]).toBe(0);
  expect(result['summary'][2]).toBe(1);
});

it('RTS-04 COLD reports segment capacity overflow', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.cold.find(item => item.name === 'two-breaks') as ColdCase;
  const result = await harness.run(createColdMonitor(harness, entry, {maximumSegmentCount: 1}));
  expect(result['summary'][1]).toBe(16);
  expect(result['summary'][0]).toBe(1);
  expect(result['segments'].slice(2, 3)).toEqual([entry.simplified.segments[0].breakSlice]);
});

it('RTS-04 COLD reports a cell with no valid observations', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.cold[0];
  const empty = {...entry, values: entry.values.map(row => row.map(() => null))};
  const result = await harness.run(createColdMonitor(harness, empty));
  expect(result['summary'][1]).toBe(1);
  expect(result['summary'][0]).toBe(0);
  expect(result['statistics'].every(Number.isNaN)).toBe(true);
});

it('RTS-04 COLD refuses configurations above its documented limits', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entry = reference.cold[0];
  const base = {
    cellCount: 1,
    sliceCount: 256,
    bandCount: 5,
    values: harness.floats(new Float32Array(256 * 5)),
    parameters: harness.floats(new Float32Array(11)),
    dateAxis: createAxis(
      harness,
      entry.timesSeconds.concat(entry.timesSeconds.map(value => value + 1e9))
    ),
    output: {
      segments: harness.output('s1', 'uint32', 32),
      coefficients: harness.output('c1', 'float32', 8 * 5 * 8),
      segmentStatistics: harness.output('r1', 'float32', 8 * 5 * 2),
      cellSummary: harness.output('t1', 'uint32', 3)
    }
  };
  expect(() => new GPURasterCOLDMonitor({...base, maximumCoefficientCount: 8})).toThrow(
    /raster-cold-monitor per-thread working memory .*must fit 2048 floats/
  );
  expect(() => new GPURasterCOLDMonitor({...base, sliceCount: 257})).toThrow(
    /raster-cold-monitor sliceCount must be an integer in \[1, 256\]/
  );
});

for (const entry of reference.bfastMonitor) {
  it(`RTS-04 BFAST Monitor matches the definition in binary64: ${entry.name}`, async () => {
    const device = await getDevice();
    const harness = createRasterTimeSeriesHarness(device);
    const sliceCount = entry.values.length;
    const columns = (entry.includeTrend ? 2 : 1) + 2 * entry.harmonicCount;
    const monitoringSlices = sliceCount - entry.historyCount;
    const result = await harness.run(
      new GPURasterBFASTMonitor({
        cellCount: 1,
        sliceCount,
        values: harness.floats(toFloat(entry.values)),
        parameters: harness.floats(
          getGPURasterBFASTMonitorParameterValues({
            period: entry.period,
            bandwidth: entry.bandwidth,
            criticalValue: entry.criticalValue
          })
        ),
        dateAxis: createAxis(harness, entry.timesSeconds),
        historySliceCount: entry.historyCount,
        harmonicCount: entry.harmonicCount,
        includeTrend: entry.includeTrend,
        output: {
          coefficients: harness.output('coefficients', 'float32', columns),
          scores: harness.output('scores', 'float32', 2),
          cellSummary: harness.output('summary', 'uint32', 3),
          process: harness.output('process', 'float32', monitoringSlices)
        }
      })
    );
    const {expected} = entry;
    expect(result['summary'][1]).toBe(0);
    expect(result['summary'][2]).toBe(expected.historyObservationCount);
    expect(result['summary'][0]).toBe(expected.breakSlice ?? GPU_RASTER_BFAST_MONITOR_NO_BREAK);
    expect(
      getMaximumAbsoluteDifference(result['coefficients'], expected.coefficients)
    ).toBeLessThan(1e-3);
    expect(Math.abs(result['scores'][0] - (expected.magnitude as number))).toBeLessThan(1e-3);
    const processTolerance = Math.max(0.02, 1e-3 * Math.abs(expected.maximumBoundaryRatio));
    expect(Math.abs(result['scores'][1] - expected.maximumBoundaryRatio)).toBeLessThan(
      processTolerance
    );
    const process = result['process'].filter(value => !Number.isNaN(value));
    expect(process).toHaveLength(expected.process.length);
    expect(getMaximumAbsoluteDifference(process, expected.process)).toBeLessThan(processTolerance);
  });
}

it('RTS-04 BFAST Monitor reports a short history', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const sliceCount = 12;
  const seconds = Array.from({length: sliceCount}, (_, slice) => 1_000_000 + slice * 2_000_000);
  const result = await harness.run(
    new GPURasterBFASTMonitor({
      cellCount: 1,
      sliceCount,
      values: harness.floats(new Float32Array(sliceCount).fill(1)),
      parameters: harness.floats(getGPURasterBFASTMonitorParameterValues({criticalValue: 1.5})),
      dateAxis: createAxis(harness, seconds),
      historySliceCount: 4,
      harmonicCount: 1,
      output: {
        coefficients: harness.output('coefficients', 'float32', 4),
        scores: harness.output('scores', 'float32', 2),
        cellSummary: harness.output('summary', 'uint32', 3)
      }
    })
  );
  // Four history observations cannot fit four coefficients with residual degrees of freedom.
  expect(result['summary'][1]).toBe(1);
  expect(result['summary'][0]).toBe(GPU_RASTER_BFAST_MONITOR_NO_BREAK);
  expect(result['scores'][0]).toBeNaN();
});
