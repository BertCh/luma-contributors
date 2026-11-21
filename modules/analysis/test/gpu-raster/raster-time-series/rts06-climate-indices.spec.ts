// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterClimateAnomaly,
  GPURasterClimateRunAccumulator,
  GPURasterRollingMean,
  GPURasterStandardizedIndex,
  GPURasterStandardizedIndexCalibration,
  GPURasterStreamingAccumulator,
  getGPURasterClimateRunAccumulatorParameterValues,
  getGPURasterFixedPointValues,
  getGPURasterStandardizedIndexCalibrationParameterValues,
  getGPURasterStandardizedIndexParameterValues,
  getGPURasterStreamingAccumulatorParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts06-climate-indices-xclim.json';

const CELLS = reference.cellCount;
const MONTHS = reference.monthCount;
const CHUNK = 210;
const CALIBRATION_MONTHS = reference.calibrationMonthCount;
const SLOT_COUNT = 12;
const MAXIMUM_VALUE = 2000;

const toFloat = (values: (number | null)[]) => values.map(value => value ?? Number.NaN);

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

/** Slices `source` (cell-major, length cells * months) into one `CHUNK`-wide window. */
function getChunk(source: number[], chunk: number): number[] {
  const values: number[] = [];
  for (let cell = 0; cell < CELLS; cell++) {
    for (let slice = 0; slice < CHUNK; slice++) {
      values.push(source[cell * MONTHS + chunk * CHUNK + slice]);
    }
  }
  return values;
}

function getChunkSlots(chunk: number): number[] {
  return reference.monthlySlots.slice(chunk * CHUNK, (chunk + 1) * CHUNK);
}

async function runStandardizedIndex(options: {
  series: number[];
  window: number;
  distribution: 'gamma' | 'fisk';
  zeroAtom: boolean;
  location: number;
  expected: (number | null)[];
  tolerance: number;
}) {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const contributors = [];
  const entries = CELLS * SLOT_COUNT;
  const counts = harness.output('state-counts', 'uint32', entries * 3);
  const moments = harness.output('state-moments', 'uint32', entries * 9);
  const fit = harness.output('fit', 'float32', entries * 3);
  const rollingOutputs = [0, 1].map(chunk =>
    options.window > 1 ? harness.output(`rolled-${chunk}`, 'float32', CELLS * CHUNK) : null
  );
  const ring =
    options.window > 1 ? harness.output('ring', 'float32', CELLS * options.window) : null;
  const seenCount = options.window > 1 ? harness.output('seen', 'uint32', CELLS) : null;
  const chunkValues = [0, 1].map(chunk => harness.floats(getChunk(options.series, chunk)));
  const slotViews = [0, 1].map(chunk => harness.integers(getChunkSlots(chunk)));
  const indexOutputs = [0, 1].map(chunk => ({
    index: harness.output(`index-${chunk}`, 'float32', CELLS * CHUNK),
    status: harness.output(`status-${chunk}`, 'uint32', CELLS * CHUNK)
  }));
  for (let chunk = 0; chunk < 2; chunk++) {
    const seriesView = options.window > 1 ? rollingOutputs[chunk]! : chunkValues[chunk];
    if (options.window > 1) {
      contributors.push(
        new GPURasterRollingMean({
          id: `rolling-${chunk}`,
          cellCount: CELLS,
          chunkSliceCount: CHUNK,
          windowLength: options.window,
          values: chunkValues[chunk],
          parameters: harness.floats(
            getGPURasterStreamingAccumulatorParameterValues(
              {reset: chunk === 0, maximumValueMagnitude: MAXIMUM_VALUE},
              CHUNK
            )
          ),
          ring: ring!,
          seenCount: seenCount!,
          output: rollingOutputs[chunk]!
        })
      );
    }
    const calibrationActive = Math.max(0, Math.min(CHUNK, CALIBRATION_MONTHS - chunk * CHUNK));
    if (calibrationActive > 0) {
      contributors.push(
        new GPURasterStandardizedIndexCalibration({
          id: `calibration-${chunk}`,
          cellCount: CELLS,
          chunkSliceCount: CHUNK,
          slotCount: SLOT_COUNT,
          distribution: options.distribution,
          zeroAtom: options.zeroAtom,
          values: seriesView,
          slotIndices: slotViews[chunk],
          parameters: harness.floats(
            getGPURasterStandardizedIndexCalibrationParameterValues(
              {
                reset: chunk === 0,
                activeSliceCount: calibrationActive,
                location: options.location,
                maximumValueMagnitude: MAXIMUM_VALUE
              },
              CHUNK
            )
          ),
          state: {counts, moments},
          fit
        })
      );
    }
  }
  for (let chunk = 0; chunk < 2; chunk++) {
    const seriesView = options.window > 1 ? rollingOutputs[chunk]! : chunkValues[chunk];
    contributors.push(
      new GPURasterStandardizedIndex({
        id: `index-${chunk}`,
        cellCount: CELLS,
        chunkSliceCount: CHUNK,
        slotCount: SLOT_COUNT,
        distribution: options.distribution,
        zeroAtom: options.zeroAtom,
        values: seriesView,
        slotIndices: slotViews[chunk],
        fit,
        parameters: harness.floats(
          getGPURasterStandardizedIndexParameterValues({location: options.location}, CHUNK)
        ),
        output: indexOutputs[chunk]
      })
    );
  }
  const result = await harness.run(contributors);
  const combined = new Float32Array(CELLS * MONTHS);
  for (let chunk = 0; chunk < 2; chunk++) {
    for (let cell = 0; cell < CELLS; cell++) {
      for (let slice = 0; slice < CHUNK; slice++) {
        combined[cell * MONTHS + chunk * CHUNK + slice] =
          result[`index-${chunk}`][cell * CHUNK + slice];
      }
    }
  }
  const expected = toFloat(options.expected);
  const difference = getMaximumAbsoluteDifference(combined, expected);
  expect(difference).toBeLessThan(options.tolerance);
  // Missing-window months must come back missing, not as an index.
  for (let index = 0; index < expected.length; index++) {
    expect(Number.isNaN(combined[index])).toBe(Number.isNaN(expected[index]));
  }
  return difference;
}

// RTS-06 target (research report): |dSPI| < 0.01 against xclim APP in float32 across 35 years.
const SPI_TOLERANCE = 0.01;

for (const windowCase of reference.windows) {
  it(`RTS-06 SPI-${windowCase.window} matches xclim 1.0.0 APP gamma across 420 months streamed in chunks`, async () => {
    await runStandardizedIndex({
      series: toFloat(reference.precipitation),
      window: windowCase.window,
      distribution: 'gamma',
      zeroAtom: true,
      location: 0,
      expected: windowCase.index,
      tolerance: SPI_TOLERANCE
    });
  });
}

it('RTS-06 SPEI matches xclim 1.0.0 APP log-logistic', async () => {
  await runStandardizedIndex({
    series: toFloat(reference.waterBalance),
    window: 1,
    distribution: 'fisk',
    zeroAtom: false,
    location: reference.waterBalanceLocation,
    expected: reference.spei,
    tolerance: SPI_TOLERANCE
  });
});

// Expected values from SciPy 1.18.1 in binary64: gamma.cdf/gamma.sf and fisk probabilities with
// norm.ppf/-norm.ppf applied to the smaller tail, computed independently of this code.
const GAMMA_TAIL_CASES = [
  {value: 1e-4, expected: -5.730740175408875},
  {value: 0.5, expected: -1.3394997958471566},
  {value: 30, expected: 6.884435183145828}
];
const FISK_TAIL_CASES = [
  {value: 0.001, expected: -5.99780701517012},
  {value: 10000, expected: 7.034483825301272}
];

it('RTS-06 standardized index keeps both tails of the probability', async () => {
  const device = await getDevice();
  const gammaHarness = createRasterTimeSeriesHarness(device);
  const gammaResult = await gammaHarness.run(
    new GPURasterStandardizedIndex({
      cellCount: 1,
      chunkSliceCount: 3,
      distribution: 'gamma',
      values: gammaHarness.floats(GAMMA_TAIL_CASES.map(entry => entry.value)),
      fit: gammaHarness.floats([2, 1, 0]),
      parameters: gammaHarness.floats(getGPURasterStandardizedIndexParameterValues({}, 3)),
      output: {
        index: gammaHarness.output('index', 'float32', 3),
        status: gammaHarness.output('status', 'uint32', 3)
      }
    })
  );
  GAMMA_TAIL_CASES.forEach((entry, slice) => {
    expect(gammaResult['status'][slice]).toBe(0);
    expect(gammaResult['index'][slice]).toBeCloseTo(entry.expected, 2);
  });
  const fiskHarness = createRasterTimeSeriesHarness(device);
  const fiskResult = await fiskHarness.run(
    new GPURasterStandardizedIndex({
      cellCount: 1,
      chunkSliceCount: 2,
      distribution: 'fisk',
      values: fiskHarness.floats(FISK_TAIL_CASES.map(entry => entry.value)),
      fit: fiskHarness.floats([3, 1, 0]),
      parameters: fiskHarness.floats(getGPURasterStandardizedIndexParameterValues({}, 2)),
      output: {
        index: fiskHarness.output('index', 'float32', 2),
        status: fiskHarness.output('status', 'uint32', 2)
      }
    })
  );
  FISK_TAIL_CASES.forEach((entry, slice) =>
    expect(fiskResult['index'][slice]).toBeCloseTo(entry.expected, 2)
  );
});

it('RTS-06 spells and degree days match xclim per calendar year', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const cellDays = reference.dayCount;
  const yearOffsets = [0];
  for (const length of reference.yearLengths)
    yearOffsets.push(yearOffsets[yearOffsets.length - 1] + length);
  const yearCount = reference.yearLengths.length;
  const yearChunk = 183;
  const makeChunk = (source: number[], year: number, chunk: number) => {
    const values = new Array<number>(CELLS * yearChunk).fill(Number.NaN);
    for (let cell = 0; cell < CELLS; cell++) {
      for (let day = 0; day < yearChunk; day++) {
        const absolute = chunk * yearChunk + day;
        if (absolute < reference.yearLengths[year]) {
          values[cell * yearChunk + day] = source[cell * cellDays + yearOffsets[year] + absolute];
        }
      }
    }
    return values;
  };
  const precipitation = toFloat(reference.precipitationDaily);
  const temperature = toFloat(reference.temperatureDaily);
  const cases = [
    {
      name: 'wet',
      data: precipitation,
      threshold: reference.wetThreshold,
      condition: 'at-or-above' as const,
      run: true
    },
    {
      name: 'dry',
      data: precipitation,
      threshold: reference.wetThreshold,
      condition: 'below' as const,
      run: true
    },
    {
      name: 'growing',
      data: temperature,
      threshold: reference.growingThreshold,
      condition: 'above' as const,
      run: false
    },
    {
      name: 'heating',
      data: temperature,
      threshold: reference.heatingThreshold,
      condition: 'below' as const,
      run: false
    }
  ];
  const contributors = [];
  for (const entry of cases) {
    for (let year = 0; year < yearCount; year++) {
      const state = entry.run
        ? {runs: harness.output(`${entry.name}-${year}-runs`, 'uint32', CELLS * 6)}
        : {degreeDays: harness.output(`${entry.name}-${year}-degree`, 'uint32', CELLS * 3)};
      for (let chunk = 0; chunk < 2; chunk++) {
        const remaining = Math.max(0, reference.yearLengths[year] - chunk * yearChunk);
        contributors.push(
          new GPURasterClimateRunAccumulator({
            id: `${entry.name}-${year}-${chunk}`,
            cellCount: CELLS,
            chunkSliceCount: yearChunk,
            condition: entry.condition,
            values: harness.floats(makeChunk(entry.data, year, chunk)),
            parameters: harness.floats(
              getGPURasterClimateRunAccumulatorParameterValues(
                {
                  reset: chunk === 0,
                  activeSliceCount: Math.min(yearChunk, remaining),
                  threshold: entry.threshold,
                  maximumIncrement: 200
                },
                yearChunk
              )
            ),
            state
          })
        );
      }
    }
  }
  const result = await harness.run(contributors);
  const degreeDayUnit = getGPURasterClimateRunAccumulatorParameterValues(
    {reset: true, threshold: 0, maximumIncrement: 200},
    yearChunk
  )[3];
  for (let cell = 0; cell < CELLS; cell++) {
    for (let year = 0; year < yearCount; year++) {
      expect(result[`wet-${year}-runs`][cell * 6 + 1]).toBe(
        reference.maximumConsecutiveWetDays[cell * yearCount + year]
      );
      expect(result[`dry-${year}-runs`][cell * 6 + 1]).toBe(
        reference.maximumConsecutiveDryDays[cell * yearCount + year]
      );
      const growing = getGPURasterFixedPointValues(result[`growing-${year}-degree`], degreeDayUnit)[
        cell
      ];
      expect(growing).toBeCloseTo(reference.growingDegreeDays[cell * yearCount + year]!, 2);
      const heating = getGPURasterFixedPointValues(result[`heating-${year}-degree`], degreeDayUnit)[
        cell
      ];
      expect(heating).toBeCloseTo(reference.heatingDegreeDays[cell * yearCount + year]!, 2);
    }
  }
});

it('RTS-06 climatology from a 420-month stream matches the binary64 reference', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const entries = CELLS * SLOT_COUNT;
  const precipitation = toFloat(reference.precipitation);
  const counts = harness.output('counts', 'uint32', entries);
  const sums = harness.output('sums', 'uint32', entries * 3);
  const moments = harness.output('moments', 'uint32', entries * 3);
  const climatologyMean = harness.output('mean', 'float32', entries);
  const standardDeviation = harness.output('deviation', 'float32', entries);
  const accumulatorParameters = harness.floats(
    getGPURasterStreamingAccumulatorParameterValues(
      {reset: false, maximumValueMagnitude: MAXIMUM_VALUE},
      CHUNK
    )
  );
  const firstParameters = harness.floats(
    getGPURasterStreamingAccumulatorParameterValues(
      {reset: true, maximumValueMagnitude: MAXIMUM_VALUE},
      CHUNK
    )
  );
  const contributors = [0, 1].map(
    chunk =>
      new GPURasterStreamingAccumulator({
        id: `chunk-${chunk}`,
        cellCount: CELLS,
        chunkSliceCount: CHUNK,
        slotCount: SLOT_COUNT,
        values: harness.floats(getChunk(precipitation, chunk)),
        slotIndices: harness.integers(getChunkSlots(chunk)),
        parameters: chunk === 0 ? firstParameters : accumulatorParameters,
        state: {counts, sums, moments}
      })
  );
  contributors.push(
    new GPURasterClimateAnomaly({
      id: 'climatology',
      cellCount: CELLS,
      slotCount: SLOT_COUNT,
      state: {counts, sums, moments},
      accumulatorParameters,
      output: {climatologyMean, standardDeviation}
    }) as unknown as GPURasterStreamingAccumulator
  );
  const result = await harness.run(contributors);
  for (const entry of reference.climatology) {
    const slotEntry = entry.cell * SLOT_COUNT + entry.slot;
    expect(result['counts'][slotEntry]).toBe(entry.count);
    expect(result['mean'][slotEntry]).toBeCloseTo(entry.mean, 3);
    expect(result['deviation'][slotEntry] ** 2).toBeCloseTo(entry.sampleVariance, 1);
  }
});
