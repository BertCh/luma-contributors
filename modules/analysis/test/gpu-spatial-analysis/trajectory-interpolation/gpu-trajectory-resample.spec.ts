// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  createRandomTracks,
  EDGE_CASE_TRACKS,
  packTracks
} from './trajectory-interpolation-fixtures';
import {expectClose, runResample} from './trajectory-interpolation-harness';
import {computeResampleOracle, type ResampleOracleResult} from './trajectory-interpolation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import numpyChecks from '../../oracles/fixtures/trajectories-lines-numpy-checks.json';

function expectResampleParity(
  actual: ResampleOracleResult,
  expected: ResampleOracleResult,
  label: string
): void {
  expectClose(actual.samples, expected.samples, 1e-3, 1e-5, `${label} samples`);
  expectClose(actual.sampleElevations, expected.sampleElevations, 1e-4, 1e-5, `${label} z`);
  expectClose(actual.sampleTimes, expected.sampleTimes, 1e-3, 1e-5, `${label} times`);
}

it('GPUTrajectoryResample matches the oracle on edge cases in time and arc-length spacing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tracks = packTracks(EDGE_CASE_TRACKS);
  for (const spacing of ['time', 'arc-length'] as const) {
    for (const sampleCount of [1, 2, 5, 17]) {
      const expected = computeResampleOracle(tracks, sampleCount, spacing);
      const actual = await runResample(device, tracks, sampleCount, spacing);
      expectResampleParity(actual, expected, `${spacing} n=${sampleCount}`);
    }
  }
  // Teeth: track 0 at 5 samples hits every stored vertex in time spacing, and the arc-length
  // spacing moves the middle sample to the midpoint of the long edge.
  const time = await runResample(device, tracks, 5, 'time');
  expect(time.samples.slice(0, 10)).toEqual([0, 0, 10, 0, 10, 20, 5, 20, 0, 20]);
  const arc = await runResample(device, tracks, 5, 'arc-length');
  expect(arc.samples.slice(0, 10)).toEqual([0, 0, 10, 0, 10, 10, 10, 20, 0, 20]);
  // Single point repeats, empty tracks are zero, duplicates end at the last duplicate.
  expect(time.samples.slice(20, 30)).toEqual([3, 4, 3, 4, 3, 4, 3, 4, 3, 4]);
  expect(time.samples.slice(30, 40)).toEqual(new Array(10).fill(0));
  expect(time.samples.slice(18, 20)).toEqual([22, 9]);
  device.destroy?.();
});

it('GPUTrajectoryResample Int64 words match the relative float32 result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const relative = packTracks(EDGE_CASE_TRACKS);
  const words = packTracks(EDGE_CASE_TRACKS, 397n * 2n ** 32n - 20n);
  for (const spacing of ['time', 'arc-length'] as const) {
    const expected = computeResampleOracle(relative, 9, spacing);
    expect(computeResampleOracle(words, 9, spacing)).toEqual(expected);
    expectResampleParity(
      await runResample(device, words, 9, spacing),
      expected,
      `words ${spacing}`
    );
  }
  device.destroy?.();
});

/**
 * Element-wise parity that also accepts the oracle evaluated with interior targets shifted by up
 * to 4 f32 ulps, the GPU division error that matters at discontinuities of the resampled path.
 */
function expectResampleParityWithinUlps(
  actual: ResampleOracleResult,
  tracks: ReturnType<typeof packTracks>,
  sampleCount: number,
  spacing: 'time' | 'arc-length',
  label: string
): number {
  const candidates = [0, -1, 1, -2, 2, -3, 3, -4, 4].map(offset =>
    computeResampleOracle(tracks, sampleCount, spacing, offset)
  );
  const isClose = (value: number, expected: number, absolute: number) =>
    Math.abs(value - expected) <= absolute + 1e-5 * Math.abs(expected);
  let shiftedCount = 0;
  for (let index = 0; index < actual.sampleTimes.length; index++) {
    const matches = candidates.findIndex(
      candidate =>
        isClose(actual.samples[2 * index], candidate.samples[2 * index], 1e-3) &&
        isClose(actual.samples[2 * index + 1], candidate.samples[2 * index + 1], 1e-3) &&
        isClose(actual.sampleElevations[index], candidate.sampleElevations[index], 1e-4) &&
        isClose(actual.sampleTimes[index], candidate.sampleTimes[index], 1e-3)
    );
    if (matches < 0) {
      expect.fail(
        `${label} sample ${index}: (${actual.samples[2 * index]}, ${actual.samples[2 * index + 1]}, t=${actual.sampleTimes[index]}) matches no oracle target within 4 ulps (expected (${candidates[0].samples[2 * index]}, ${candidates[0].samples[2 * index + 1]}, t=${candidates[0].sampleTimes[index]}))`
      );
    }
    shiftedCount += matches > 0 ? 1 : 0;
  }
  return shiftedCount;
}

it('GPUTrajectoryResample arc-length spacing stays uniform on a long track of equal steps (trajectories-lines-03)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 86,400 steps of the same length, back and forth between two points, one row per time unit.
  // Every step is equally long, so arc length is proportional to the row number and sample k of 8
  // lies at row 86400 k / 7 exactly: a closed form with no summation. A plain f32 running sum of
  // this track is 1,051 units short at the end and places the interior samples 11 to 44 rows off.
  const stepCount = 86_400;
  const stepLength = Math.fround(13.9);
  const sampleCount = 8;
  const positions = new Float32Array(2 * (stepCount + 1));
  for (let row = 1; row <= stepCount; row += 2) {
    positions[2 * row] = stepLength;
  }
  const actual = await runResample(
    device,
    {
      positions,
      times: {
        kind: 'float32',
        values: Float32Array.from({length: stepCount + 1}, (_, row) => row)
      },
      trackOffsets: [0, stepCount + 1]
    },
    sampleCount,
    'arc-length'
  );
  for (let sample = 0; sample < sampleCount; sample++) {
    const expectedRow = (stepCount * sample) / (sampleCount - 1);
    const segment = Math.floor(expectedRow);
    const fraction = expectedRow - segment;
    const expectedX = (segment % 2 === 0 ? fraction : 1 - fraction) * stepLength;
    // 0.1 rows is 1.39 units of 1,200,960: the compensated f32 sum is within 0.03 rows.
    expect(Math.abs(actual.sampleTimes[sample] - expectedRow), `time ${sample}`).toBeLessThan(0.1);
    expect(Math.abs(actual.samples[2 * sample] - expectedX), `x ${sample}`).toBeLessThan(1.5);
    expect(actual.samples[2 * sample + 1]).toBe(0);
  }
  // Second check: NumPy cumsum and interp over the same float32 positions, summed in binary64
  // (oracles/trajectories-lines-numpy-checks.py). It agrees with the closed form above.
  const reference = numpyChecks.arcLengthResample;
  expect([reference.stepCount, reference.sampleCount, reference.stepLength]).toEqual([
    stepCount,
    sampleCount,
    stepLength
  ]);
  for (let sample = 0; sample < sampleCount; sample++) {
    expect(reference.sampleTimes[sample]).toBeCloseTo((stepCount * sample) / (sampleCount - 1), 6);
    expect(
      Math.abs(actual.sampleTimes[sample] - reference.sampleTimes[sample]),
      `NumPy time ${sample}`
    ).toBeLessThan(0.1);
    expect(
      Math.abs(actual.samples[2 * sample] - reference.sampleX[sample]),
      `NumPy x ${sample}`
    ).toBeLessThan(1.5);
  }
  device.destroy?.();
});

it('GPUTrajectoryResample matches the oracle on random tracks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seed of [4, 5]) {
    const tracks = packTracks(createRandomTracks(seed, 400));
    for (const spacing of ['time', 'arc-length'] as const) {
      const actual = await runResample(device, tracks, 32, spacing);
      const shiftedCount = expectResampleParityWithinUlps(
        actual,
        tracks,
        32,
        spacing,
        `seed ${seed} ${spacing}`
      );
      // Discontinuity flips are rare: well under 1% of the 12,800 samples.
      expect(shiftedCount).toBeLessThan(128);
    }
  }
  device.destroy?.();
});
