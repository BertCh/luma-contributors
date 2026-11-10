// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterClimateAnomaly,
  GPURasterStreamingAccumulator,
  getGPURasterStreamingAccumulatorParameterValues,
  getGPURasterFixedPointValues
} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';

const CHUNK = 128;
const CELL_COUNT = 2;
const CHUNK_COUNT = 4;
const TOTAL = CHUNK * CHUNK_COUNT;
const MAXIMUM_VALUE_MAGNITUDE = 2000;

/** Deterministic series well past the 256-slice in-memory limit: large offset plus small variation. */
function makeSeries(): number[][] {
  const series = [new Array<number>(TOTAL), new Array<number>(TOTAL)];
  for (let slice = 0; slice < TOTAL; slice++) {
    series[0][slice] = 1000 + 5 * Math.sin(slice * 0.37) + ((slice * 2654435761) % 1000) / 1e5;
    series[1][slice] = -0.3 + 0.002 * ((slice * 40503) % 977);
  }
  return series;
}

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

it('TC-07 streaming accumulator matches binary64 statistics over 512 slices in four chunks with reset, slots and missing samples', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const series = makeSeries();
  const slotCount = 3;
  const slots = Array.from({length: TOTAL}, (_, slice) => slice % slotCount);
  const entryCount = CELL_COUNT * slotCount;
  const counts = harness.output('counts', 'uint32', entryCount);
  const sums = harness.output('sums', 'uint32', entryCount * 3);
  const moments = harness.output('moments', 'uint32', entryCount * 3);
  const extrema = harness.output('extrema', 'float32', entryCount * 2);
  const expectedCounts = new Array<number>(entryCount).fill(0);
  const binary64Sum = new Array<number>(entryCount).fill(0);
  const binary64Squares = new Array<number>(entryCount).fill(0);
  const minimum = new Array<number>(entryCount).fill(Infinity);
  const maximum = new Array<number>(entryCount).fill(-Infinity);
  const contributors = [];
  for (let chunk = 0; chunk < CHUNK_COUNT; chunk++) {
    const values = new Float32Array(CELL_COUNT * CHUNK);
    const validity = new Uint32Array(CELL_COUNT * CHUNK).fill(1);
    const chunkSlots = slots.slice(chunk * CHUNK, (chunk + 1) * CHUNK);
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      for (let slice = 0; slice < CHUNK; slice++) {
        const absolute = chunk * CHUNK + slice;
        const value = series[cell][absolute];
        const stored = Math.fround(value);
        values[cell * CHUNK + slice] = stored;
        const missing = absolute % 53 === 11;
        if (missing) {
          values[cell * CHUNK + slice] = Number.NaN;
          continue;
        }
        if (absolute % 97 === 41) {
          validity[cell * CHUNK + slice] = 0;
          continue;
        }
        const entry = cell * slotCount + chunkSlots[slice];
        expectedCounts[entry]++;
        binary64Sum[entry] += stored;
        minimum[entry] = Math.min(minimum[entry], stored);
        maximum[entry] = Math.max(maximum[entry], stored);
      }
    }
    contributors.push(
      new GPURasterStreamingAccumulator({
        id: `chunk-${chunk}`,
        cellCount: CELL_COUNT,
        chunkSliceCount: CHUNK,
        slotCount,
        values: harness.floats(values),
        validity: harness.integers(validity),
        slotIndices: harness.integers(chunkSlots),
        parameters: harness.floats(
          getGPURasterStreamingAccumulatorParameterValues(
            {reset: chunk === 0, maximumValueMagnitude: MAXIMUM_VALUE_MAGNITUDE},
            CHUNK
          )
        ),
        state: {counts, sums, moments, extrema}
      })
    );
  }
  // Reference: recompute the variance in binary64 over the same masked, float32-rounded samples.
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    for (let slot = 0; slot < slotCount; slot++) {
      const entry = cell * slotCount + slot;
      const selected = [];
      for (let slice = 0; slice < TOTAL; slice++) {
        if (slice % 53 !== 11 && slice % 97 !== 41 && slice % slotCount === slot) {
          selected.push(Math.fround(series[cell][slice]));
        }
      }
      const mean = selected.reduce((total, value) => total + value, 0) / selected.length;
      binary64Squares[entry] = selected.reduce((total, value) => total + (value - mean) ** 2, 0);
    }
  }
  const result = await harness.run(contributors);
  expect(result['counts']).toEqual(expectedCounts);
  const parameterValues = getGPURasterStreamingAccumulatorParameterValues(
    {reset: false, maximumValueMagnitude: MAXIMUM_VALUE_MAGNITUDE},
    CHUNK
  );
  const decodedSums = getGPURasterFixedPointValues(result['sums'], parameterValues[2]);
  const decodedMoments = getGPURasterFixedPointValues(result['moments'], parameterValues[3]);
  let largestNaiveError = 0;
  let largestCompensatedError = 0;
  for (let entry = 0; entry < entryCount; entry++) {
    const compensated = decodedSums[entry];
    largestCompensatedError = Math.max(
      largestCompensatedError,
      Math.abs(compensated - binary64Sum[entry]) / Math.max(1, Math.abs(binary64Sum[entry]))
    );
    // A naive float32 running sum over the same samples, for comparison.
    let naive = 0;
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      for (let slice = 0; slice < TOTAL; slice++) {
        if (slice % 53 === 11 || slice % 97 === 41) continue;
        if (cell * slotCount + (slice % slotCount) !== entry) continue;
        naive = Math.fround(naive + Math.fround(series[cell][slice]));
      }
    }
    largestNaiveError = Math.max(
      largestNaiveError,
      Math.abs(naive - binary64Sum[entry]) / Math.max(1, Math.abs(binary64Sum[entry]))
    );
    const moment = decodedMoments[entry];
    expect(
      Math.abs(moment - binary64Squares[entry]) / Math.max(1, binary64Squares[entry])
    ).toBeLessThan(1e-3);
    expect(result['extrema'][entry * 2]).toBe(minimum[entry]);
    expect(result['extrema'][entry * 2 + 1]).toBe(maximum[entry]);
  }
  // The fixed-point sums are exact to their unit, so they beat a float32 running sum outright.
  expect(largestCompensatedError).toBeLessThan(1e-10);
  expect(largestNaiveError).toBeGreaterThan(1e-8);
});

it('TC-07 reset discards earlier chunks, and short chunks stop at activeSliceCount', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const counts = harness.output('counts', 'uint32', 1);
  const sums = harness.output('sums', 'uint32', 3);
  const make = (values: number[], settings: {reset: boolean; activeSliceCount?: number}) =>
    new GPURasterStreamingAccumulator({
      cellCount: 1,
      chunkSliceCount: 4,
      values: harness.floats(values),
      parameters: harness.floats(
        getGPURasterStreamingAccumulatorParameterValues(
          {...settings, maximumValueMagnitude: 1000},
          4
        )
      ),
      state: {counts, sums},
      id: `case-${settings.reset}-${settings.activeSliceCount ?? 4}-${values[0]}`
    });
  const result = await harness.run([
    make([100, 200, 300, 400], {reset: true}),
    make([1, 2, 3, 900], {reset: true, activeSliceCount: 3}),
    make([4, 5, 6, 7], {reset: false, activeSliceCount: 2})
  ]);
  expect(result['counts']).toEqual([5]);
  expect(
    getGPURasterFixedPointValues(
      result['sums'],
      getGPURasterStreamingAccumulatorParameterValues(
        {reset: true, maximumValueMagnitude: 1000},
        4
      )[2]
    )[0]
  ).toBe(15);
});

it('RTS-06 climatology means and anomalies come straight from the accumulator state', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const counts = harness.output('counts', 'uint32', 4);
  const sums = harness.output('sums', 'uint32', 12);
  const moments = harness.output('moments', 'uint32', 12);
  const firstCell = [10, 11, 12, 13];
  const secondCell = [2, 4, 6, 12];
  const values = [...firstCell, ...secondCell];
  const slotIndices = harness.integers([0, 0, 1, 1]);
  const climatologyMean = harness.output('mean', 'float32', 4);
  const standardDeviation = harness.output('deviation', 'float32', 4);
  const anomaly = harness.output('anomaly', 'float32', 8);
  const accumulatorParameters = harness.floats(
    getGPURasterStreamingAccumulatorParameterValues({reset: true, maximumValueMagnitude: 100}, 4)
  );
  const result = await harness.run([
    new GPURasterStreamingAccumulator({
      id: 'accumulate',
      cellCount: 2,
      chunkSliceCount: 4,
      slotCount: 2,
      values: harness.floats(values),
      slotIndices,
      parameters: accumulatorParameters,
      state: {counts, sums, moments}
    }),
    new GPURasterClimateAnomaly({
      id: 'anomaly',
      cellCount: 2,
      slotCount: 2,
      chunkSliceCount: 4,
      state: {counts, sums, moments},
      accumulatorParameters,
      values: harness.floats(values),
      slotIndices,
      output: {climatologyMean, standardDeviation, anomaly}
    })
  ]);
  expect(result['mean']).toEqual([10.5, 12.5, 3, 9]);
  expect(result['deviation'][0]).toBeCloseTo(Math.SQRT1_2, 5);
  expect(result['deviation'][3]).toBeCloseTo(4.242641, 5);
  expect(result['anomaly']).toEqual([-0.5, 0.5, -0.5, 0.5, -1, 1, -3, 3]);
});
