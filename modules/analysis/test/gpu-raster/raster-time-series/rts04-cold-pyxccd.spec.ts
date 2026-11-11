// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterCOLDMonitor against pyxccd 1.1.0 `cold_detect`, the reference implementation of COLD
// (fixture and conventions: test/oracles/rts04-cold-pyxccd.py). The monitor runs on the five bands
// pyxccd scores (green to SWIR2) with pyxccd's thresholds, its variogram floor as `minimumRMSE` and
// its year of 365.25 days. Values are Landsat-scaled integers; the noise is 40 units per band.

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterCOLDMonitor,
  GPU_RASTER_COLD_MONITOR_NO_BREAK,
  getGPURasterCOLDMonitorParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts04-cold-pyxccd.json';
import study from '../../oracles/fixtures/rts04-cold-pyxccd-break-sizes.json';

const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};
const SECONDS_PER_DAY = 86400;
const SEGMENT_CAPACITY = 8;
const BAND_COUNT = reference.detectionBands.length;

type Case = (typeof reference.cases)[number];
type ReferenceSegment = Case['pyxccd']['leastSquares'][number];

function getCase(name: string): Case {
  return reference.cases.find(entry => entry.name === name) as Case;
}

function createMonitor(harness: ReturnType<typeof createRasterTimeSeriesHarness>, entry: Case) {
  return new GPURasterCOLDMonitor({
    cellCount: 1,
    sliceCount: entry.dates.length,
    bandCount: BAND_COUNT,
    values: harness.floats(
      entry.values.flatMap(row => reference.detectionBands.map(band => row[band]))
    ),
    // pyxccd keeps cfmask clear (0) and water (1).
    mask: harness.integers(entry.qa.map(flag => (flag < 2 ? 1 : 0))),
    parameters: harness.floats(
      getGPURasterCOLDMonitorParameterValues({
        period: reference.period,
        changeThreshold: reference.changeThreshold,
        outlierThreshold: reference.outlierThreshold,
        minimumSpanYears: reference.period,
        maximumInitializationGapYears: reference.period,
        minimumRMSE: entry.minimumRMSE
      })
    ),
    dateAxis: {
      acquisitionTimes: harness.words(
        BigInt64Array.from(entry.dates, date => BigInt(date) * BigInt(SECONDS_PER_DAY))
      ),
      temporalContext: CONTEXT
    },
    consecutiveCount: entry.consecutive,
    maximumSegmentCount: SEGMENT_CAPACITY,
    output: {
      segments: harness.output('segments', 'uint32', SEGMENT_CAPACITY * 4),
      coefficients: harness.output('coefficients', 'float32', SEGMENT_CAPACITY * BAND_COUNT * 8),
      segmentStatistics: harness.output('statistics', 'float32', SEGMENT_CAPACITY * BAND_COUNT * 2),
      cellSummary: harness.output('summary', 'uint32', 3)
    }
  });
}

async function runMonitor(entry: Case) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(createMonitor(harness, entry));
  const segments = [];
  for (let segment = 0; segment < result['summary'][0]; segment++) {
    const breakSlice = result['segments'][segment * 4 + 2];
    const bandBase = segment * BAND_COUNT;
    segments.push({
      startSlice: result['segments'][segment * 4],
      endSlice: result['segments'][segment * 4 + 1],
      breakSlice: breakSlice === GPU_RASTER_COLD_MONITOR_NO_BREAK ? null : breakSlice,
      coefficients: result['coefficients'].slice(bandBase * 8, (bandBase + BAND_COUNT) * 8),
      rmse: Array.from(
        {length: BAND_COUNT},
        (_, band) => result['statistics'][(bandBase + band) * 2]
      ),
      magnitude: Array.from(
        {length: BAND_COUNT},
        (_, band) => result['statistics'][(bandBase + band) * 2 + 1]
      )
    });
  }
  return {segments, status: result['summary'][1]};
}

const getConfirmedBreaks = (segments: ReferenceSegment[]) =>
  segments.filter(segment => segment.changeProbability === 100).map(segment => segment.breakSlice);

function getClearCount(entry: Case, startSlice: number, endSlice: number): number {
  return entry.qa.slice(startSlice, endSlice + 1).filter(flag => flag < 2).length;
}

// Tolerances in band units, measured on the test GPU (largest observed in brackets).
// A segment pyxccd last refitted on exactly the observations the monitor fits [0.07, 0.02].
const SAME_FIT_COEFFICIENT_TOLERANCE = 0.2;
const SAME_FIT_RMSE_TOLERANCE = 0.1;
// pyxccd refits only when the count has grown by 3%, so its record can lag by a few observations [2.73, 0.35].
const LAGGING_FIT_COEFFICIENT_TOLERANCE = 4;
const LAGGING_FIT_RMSE_TOLERANCE = 0.6;
// The last segment: the monitor also fits the last `consecutiveCount` observations [20.24, 2.31].
const LAST_SEGMENT_COEFFICIENT_TOLERANCE = 25;
const LAST_SEGMENT_RMSE_TOLERANCE = 3;
// The monitor's model at the break is its last 1.33 refit, pyxccd's is at most 3% old [19.9].
const MAGNITUDE_TOLERANCE = 25;

// Segment starts where the monitor is known to differ: pyxccd looks backward from the first accepted
// window (slice 2 here) to the start of the series; the monitor has no lookback.
const MONITOR_STARTS_WITHOUT_LOOKBACK: Record<string, number[]> = {'two-breaks': [2, 55, 110]};

for (const entry of reference.cases.filter(item => item.name !== 'gradual-change')) {
  it(`RTS-04 COLD agrees with pyxccd cold_detect: ${entry.name}`, async () => {
    const {leastSquares, lasso20} = entry.pyxccd;
    const result = await runMonitor(entry);
    expect(result.status).toBe(0);

    // Break dates: equal to the known breaks and to pyxccd with and without LASSO.
    const breaks = result.segments
      .map(segment => segment.breakSlice)
      .filter(breakSlice => breakSlice !== null);
    expect(breaks).toEqual(entry.knownBreakSlices);
    expect(breaks).toEqual(getConfirmedBreaks(lasso20));
    expect(breaks).toEqual(getConfirmedBreaks(leastSquares));

    // Segment bounds.
    expect(result.segments).toHaveLength(lasso20.length);
    expect(result.segments.map(segment => segment.startSlice)).toEqual(
      MONITOR_STARTS_WITHOUT_LOOKBACK[entry.name] ?? lasso20.map(segment => segment.startSlice)
    );
    expect(result.segments.map(segment => segment.endSlice)).toEqual(
      lasso20.map(segment => segment.endSlice)
    );

    for (const [index, segment] of result.segments.entries()) {
      const expected = leastSquares[index];
      // An unflagged cloud stays in the monitor's fit (no Tmask); measured in its own test below.
      const cloudInFit = entry.name === 'unflagged-cloud-first-window' && index === 0;
      if (segment.startSlice !== expected.startSlice || cloudInFit) continue;
      const coefficientGap = getMaximumAbsoluteDifference(
        segment.coefficients,
        expected.coefficients.flat()
      );
      const rmseGap = getMaximumAbsoluteDifference(segment.rmse, expected.rmse);
      if (segment.breakSlice === null) {
        expect(coefficientGap).toBeLessThan(LAST_SEGMENT_COEFFICIENT_TOLERANCE);
        // A partial break in pyxccd's tail (change probability above 0) is not comparable.
        if (expected.changeProbability === 0) {
          expect(rmseGap).toBeLessThan(LAST_SEGMENT_RMSE_TOLERANCE);
        }
        continue;
      }
      const sameFit =
        expected.observationCount === getClearCount(entry, segment.startSlice, segment.endSlice);
      expect(coefficientGap).toBeLessThan(
        sameFit ? SAME_FIT_COEFFICIENT_TOLERANCE : LAGGING_FIT_COEFFICIENT_TOLERANCE
      );
      expect(rmseGap).toBeLessThan(sameFit ? SAME_FIT_RMSE_TOLERANCE : LAGGING_FIT_RMSE_TOLERANCE);
      expect(getMaximumAbsoluteDifference(segment.magnitude, expected.magnitude)).toBeLessThan(
        MAGNITUDE_TOLERANCE
      );
    }
  });
}

it('RTS-04 COLD fits the same model as pyxccd where both fit the same observations', async () => {
  // Guards the tight tolerance: these two segments are the ones pyxccd refitted on its last observation.
  for (const name of ['one-break', 'flagged-clouds']) {
    const entry = getCase(name);
    const expected = entry.pyxccd.leastSquares[0];
    const result = await runMonitor(entry);
    const segment = result.segments[0];
    expect(expected.observationCount).toBe(
      getClearCount(entry, segment.startSlice, segment.endSlice)
    );
    expect(
      getMaximumAbsoluteDifference(segment.coefficients, expected.coefficients.flat())
    ).toBeLessThan(SAME_FIT_COEFFICIENT_TOLERANCE);
    expect(getMaximumAbsoluteDifference(segment.rmse, expected.rmse)).toBeLessThan(
      SAME_FIT_RMSE_TOLERANCE
    );
  }
});

it('RTS-04 COLD omits LASSO: pyxccd at its default penalty has a larger RMSE', async () => {
  const entry = getCase('one-break');
  const result = await runMonitor(entry);
  for (const [index, segment] of result.segments.entries()) {
    const penalized = entry.pyxccd.lasso20[index];
    // Measured: the RMSE at lam 20 is 10.2 to 12.6 above ordinary least squares on every band.
    for (let band = 0; band < BAND_COUNT; band++) {
      expect(penalized.rmse[band] - segment.rmse[band]).toBeGreaterThan(8);
      expect(penalized.rmse[band] - segment.rmse[band]).toBeLessThan(16);
    }
  }
});

it('RTS-04 COLD omits Tmask: an unflagged cloud stays in the fit and inflates the RMSE', async () => {
  const entry = getCase('unflagged-cloud-first-window');
  const expected = entry.pyxccd.leastSquares[0];
  const result = await runMonitor(entry);
  const segment = result.segments[0];
  // Same segment and break as pyxccd, which removed slice 6 before fitting.
  expect([segment.startSlice, segment.endSlice, segment.breakSlice]).toEqual([0, 84, 85]);
  expect(expected.observationCount).toBeLessThan(getClearCount(entry, 0, 84));
  // Measured: green 274 against 39.5 in pyxccd; every band is inflated by more than its noise.
  const inflation = segment.rmse.map((value, band) => value - expected.rmse[band]);
  expect(inflation[0]).toBeGreaterThan(200);
  expect(Math.min(...inflation)).toBeGreaterThan(40);
});

it('RTS-04 COLD dates a gradual change earlier than pyxccd', async () => {
  // A 30-slice ramp from slice 70. Refit schedule, temporally adjusted RMSE and LASSO all move the
  // confirmed date: pyxccd confirms at slice 77 without LASSO and 80 with it, the monitor at 75,
  // and the next segment starts at slice 95, 90 and 100.
  const entry = getCase('gradual-change');
  const result = await runMonitor(entry);
  expect(result.status).toBe(0);
  expect(getConfirmedBreaks(entry.pyxccd.leastSquares)).toEqual([77]);
  expect(getConfirmedBreaks(entry.pyxccd.lasso20)).toEqual([80]);
  expect(entry.pyxccd.leastSquares.map(segment => segment.startSlice)).toEqual([0, 95]);
  expect(entry.pyxccd.lasso20.map(segment => segment.startSlice)).toEqual([0, 90]);
  expect(
    result.segments.map(segment => [segment.startSlice, segment.endSlice, segment.breakSlice])
  ).toEqual([
    [0, 74, 75],
    [100, 159, null]
  ]);
});

it('RTS-04 COLD does not report the partial break pyxccd flags in the last observations', async () => {
  const entry = getCase('small-break');
  const tail = entry.pyxccd.leastSquares[1];
  // pyxccd: one anomalous observation at the end of the series, change probability 1 in 6.
  expect([tail.endSlice, tail.breakSlice, tail.changeProbability]).toEqual([148, 149, 16]);
  const result = await runMonitor(entry);
  expect(result.segments[1].endSlice).toBe(149);
  expect(result.segments[1].breakSlice).toBeNull();
  expect(result.segments[1].magnitude.every(Number.isNaN)).toBe(true);
});

it('RTS-04 COLD stability test takes the larger end residual, so segments start where pyxccd starts them', async () => {
  // Regression guard for raster-imagery-11. Until 2026-11-11 the stability test added the first
  // and the last absolute residual of the window; pyxccd takes the larger. The sum rejected
  // windows pyxccd accepts, and on these series the monitor then started its segments at
  // [1, 91], [2, 79] and [0, 103] (measured on the test GPU with the sum).
  const pyxccdStarts: Record<string, number[]> = {
    'small-break': [0, 90],
    'flagged-clouds': [0, 70],
    'four-consecutive': [0, 100]
  };
  for (const [name, starts] of Object.entries(pyxccdStarts)) {
    const entry = getCase(name);
    expect(entry.pyxccd.leastSquares.map(segment => segment.startSlice)).toEqual(starts);
    expect(entry.pyxccd.lasso20.map(segment => segment.startSlice)).toEqual(starts);
    const result = await runMonitor(entry);
    expect(result.segments.map(segment => segment.startSlice)).toEqual(starts);
  }
});

type StudySeries = (typeof study.series)[number];
type StudyBounds = [number, number | null][];

const STUDY_RECORD_WORDS = SEGMENT_CAPACITY * 4 + 3;

/**
 * Runs every series of the break-size study through one compiled monitor of one cell. Each run
 * copies its segments and summary into one buffer, read back once at the end.
 */
async function runBreakSizeStudy(): Promise<StudyBounds[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sliceCount = study.series[0].dates.length;
  const graph = new GPUCommandGraph(device, {id: 'cold-break-size-study'});
  const valuesBuffer = createInputBuffer(device, new Float32Array(sliceCount * BAND_COUNT));
  const parametersBuffer = createInputBuffer(device, new Float32Array(6 + BAND_COUNT));
  const datesBuffer = createInputBuffer(device, new Uint32Array(sliceCount * 2));
  const segmentsBuffer = createOutputBuffer(device, SEGMENT_CAPACITY * 4);
  const coefficientsBuffer = createOutputBuffer(device, SEGMENT_CAPACITY * BAND_COUNT * 8);
  const statisticsBuffer = createOutputBuffer(device, SEGMENT_CAPACITY * BAND_COUNT * 2);
  const summaryBuffer = createOutputBuffer(device, 3);
  const recordsBuffer = createOutputBuffer(device, study.series.length * STUDY_RECORD_WORDS);
  const buffers = [
    valuesBuffer,
    parametersBuffer,
    datesBuffer,
    segmentsBuffer,
    coefficientsBuffer,
    statisticsBuffer,
    summaryBuffer,
    recordsBuffer
  ];
  graph.add(
    new GPURasterCOLDMonitor({
      cellCount: 1,
      sliceCount,
      bandCount: BAND_COUNT,
      values: importGraphBuffer(graph, 'study-values', valuesBuffer, 'float32'),
      parameters: importGraphBuffer(graph, 'study-parameters', parametersBuffer, 'float32'),
      dateAxis: {
        acquisitionTimes: importGraphBuffer(graph, 'study-dates', datesBuffer, 'uint32x2'),
        temporalContext: CONTEXT
      },
      consecutiveCount: study.consecutive,
      maximumSegmentCount: SEGMENT_CAPACITY,
      output: {
        segments: importGraphBuffer(graph, 'study-segments', segmentsBuffer, 'uint32'),
        coefficients: importGraphBuffer(graph, 'study-coefficients', coefficientsBuffer, 'float32'),
        segmentStatistics: importGraphBuffer(
          graph,
          'study-statistics',
          statisticsBuffer,
          'float32'
        ),
        cellSummary: importGraphBuffer(graph, 'study-summary', summaryBuffer, 'uint32')
      }
    })
  );
  const compiled = graph.compile();
  try {
    study.series.forEach((series, index) => {
      expect(series.dates).toHaveLength(sliceCount);
      valuesBuffer.write(Float32Array.from(series.values.flat()));
      parametersBuffer.write(
        getGPURasterCOLDMonitorParameterValues({
          period: study.period,
          changeThreshold: study.changeThreshold,
          outlierThreshold: study.outlierThreshold,
          minimumSpanYears: study.period,
          maximumInitializationGapYears: study.period,
          minimumRMSE: series.minimumRMSE
        })
      );
      const words = new Uint32Array(sliceCount * 2);
      series.dates.forEach((date, slice) => {
        const seconds = BigInt(date) * BigInt(SECONDS_PER_DAY);
        words[slice * 2] = Number(seconds & 0xffffffffn);
        words[slice * 2 + 1] = Number(seconds >> 32n);
      });
      datesBuffer.write(words);
      const commandEncoder = device.createCommandEncoder({id: `study-encoder-${index}`});
      compiled.encode(commandEncoder, {parameters: undefined});
      const recordOffset = index * STUDY_RECORD_WORDS * Uint32Array.BYTES_PER_ELEMENT;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: segmentsBuffer,
        destinationBuffer: recordsBuffer,
        destinationOffset: recordOffset,
        size: SEGMENT_CAPACITY * 4 * Uint32Array.BYTES_PER_ELEMENT
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: summaryBuffer,
        destinationBuffer: recordsBuffer,
        destinationOffset: recordOffset + SEGMENT_CAPACITY * 4 * Uint32Array.BYTES_PER_ELEMENT,
        size: 3 * Uint32Array.BYTES_PER_ELEMENT
      });
      device.submit(commandEncoder.finish());
    });
    const records = await readUint32(recordsBuffer, study.series.length * STUDY_RECORD_WORDS);
    return study.series.map((_, index) => {
      const record = records.slice(index * STUDY_RECORD_WORDS, (index + 1) * STUDY_RECORD_WORDS);
      const [segmentCount, status] = record.slice(SEGMENT_CAPACITY * 4);
      expect(status).toBe(0);
      return Array.from({length: segmentCount}, (__, segment): [number, number | null] => [
        record[segment * 4],
        record[segment * 4 + 2] === GPU_RASTER_COLD_MONITOR_NO_BREAK
          ? null
          : record[segment * 4 + 2]
      ]);
    });
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

const getBreaks = (bounds: (number | null)[][]) =>
  bounds.map(segment => segment[1]).filter(breakSlice => breakSlice !== null);

type BreakSizeTally = {
  seriesCount: number;
  /** Break lists equal to pyxccd at `lam=0`. */
  breaksEqual: number;
  /** Break lists equal, some segment start different. */
  startsDiffer: number;
  /** Break lists equal to pyxccd at its default `lam=20`. */
  breaksEqualWithLasso: number;
  /** pyxccd (`lam=0`) confirms no break, the monitor does. */
  monitorOnly: number;
  /** pyxccd (`lam=0`) confirms a break, the monitor none. */
  pyxccdOnly: number;
  /** Both confirm, at different slices. */
  otherDate: number;
  /** Exactly the one true break, by pyxccd (`lam=0`) and by the monitor. */
  pyxccdExact: number;
  monitorExact: number;
};

function tallyBreakSizeStudy(results: StudyBounds[]): Map<number, BreakSizeTally> {
  const tallies = new Map<number, BreakSizeTally>();
  study.series.forEach((series: StudySeries, index) => {
    const tally = tallies.get(series.size) ?? {
      seriesCount: 0,
      breaksEqual: 0,
      startsDiffer: 0,
      breaksEqualWithLasso: 0,
      monitorOnly: 0,
      pyxccdOnly: 0,
      otherDate: 0,
      pyxccdExact: 0,
      monitorExact: 0
    };
    tallies.set(series.size, tally);
    const monitor = results[index];
    const pyxccd = series.pyxccd.leastSquares;
    const monitorBreaks = getBreaks(monitor);
    const pyxccdBreaks = getBreaks(pyxccd);
    const isEqual = (left: unknown, right: unknown) =>
      JSON.stringify(left) === JSON.stringify(right);
    tally.seriesCount++;
    if (isEqual(monitorBreaks, pyxccdBreaks)) {
      tally.breaksEqual++;
      const monitorStarts = monitor.map(segment => segment[0]);
      const pyxccdStarts = pyxccd.map(segment => segment[0]);
      if (!isEqual(monitorStarts, pyxccdStarts)) tally.startsDiffer++;
    } else if (pyxccdBreaks.length === 0) tally.monitorOnly++;
    else if (monitorBreaks.length === 0) tally.pyxccdOnly++;
    else tally.otherDate++;
    if (isEqual(monitorBreaks, getBreaks(series.pyxccd.lasso20))) tally.breaksEqualWithLasso++;
    if (isEqual(pyxccdBreaks, [series.breakSlice])) tally.pyxccdExact++;
    if (isEqual(monitorBreaks, [series.breakSlice])) tally.monitorExact++;
  });
  return tallies;
}

it('RTS-04 COLD break dates equal pyxccd for breaks well above the change threshold, not near it', async () => {
  // 160 series of 150 dates with one break (test/oracles/rts04-cold-pyxccd.py, the break-size
  // study): the full disturbance of the cases above (40 series), a half (40) and a quarter (80).
  // The noise-free change score of the break, the sum over bands of (shift / residual scale)^2,
  // is 230 to 355, 60 to 89 and 16 to 22; the change threshold is 15.09.
  const tallies = tallyBreakSizeStudy(await runBreakSizeStudy());

  // Full and half size (4 to 24 times the threshold): every break date equals pyxccd's, with and
  // without LASSO, and is the true one. Starts differ in 4 and 10 series on the test GPU
  // (pyxccd's lookback, and its window advance).
  for (const size of [1, 0.5]) {
    const tally = tallies.get(size) as BreakSizeTally;
    expect(tally.seriesCount).toBe(40);
    expect(tally.breaksEqual).toBe(40);
    expect(tally.breaksEqualWithLasso).toBe(40);
    expect(tally.pyxccdExact).toBe(40);
    expect(tally.monitorExact).toBe(40);
    expect(tally.startsDiffer).toBeGreaterThan(0);
    expect(tally.startsDiffer).toBeLessThan(16);
  }

  // Quarter size (1.05 to 1.45 times the threshold): no parity. Measured on the test GPU: the
  // break lists differ in 31 of 80 series. In 23 pyxccd confirms nothing and the monitor confirms
  // a break (usually some observations late), in 8 both confirm at different slices, in none does
  // only pyxccd confirm. pyxccd confirms exactly the true break in 16 series, the monitor in 26;
  // against pyxccd's default LASSO run the break lists are equal in 30. The monitor refits when
  // the count has grown by a third, pyxccd at 3%: pyxccd's model absorbs a small step before six
  // observations in a row exceed the threshold, the monitor's older model keeps scoring against
  // the level before the break. The bands allow a few borderline series to fall differently on
  // another adapter.
  const quarter = tallies.get(0.25) as BreakSizeTally;
  expect(quarter.seriesCount).toBe(80);
  expect(quarter.pyxccdExact).toBe(16);
  expect(quarter.breaksEqual).toBeGreaterThanOrEqual(45);
  expect(quarter.breaksEqual).toBeLessThanOrEqual(53);
  expect(quarter.monitorOnly).toBeGreaterThanOrEqual(19);
  expect(quarter.monitorOnly).toBeLessThanOrEqual(27);
  expect(quarter.pyxccdOnly).toBeLessThanOrEqual(3);
  expect(quarter.monitorExact).toBeGreaterThanOrEqual(22);
  expect(quarter.monitorExact).toBeLessThanOrEqual(30);
  expect(quarter.breaksEqualWithLasso).toBeLessThanOrEqual(36);
});
