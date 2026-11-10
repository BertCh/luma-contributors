// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {prepareGPUTrajectoryObservations} from '../../../src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations';
import {splitTimeWords} from '../../../src/dataframe-analysis/time-window-filter/time-words';
import fixture from '../../oracles/fixtures/network-movement-coverage.json';

it('GPUTrackSimilaritySearch LCSS preserves unequal-length band matches and prunes impossible cardinalities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    similarity: harness.output('float32', 2),
    upperBound: harness.output('float32', 2),
    matched: harness.output('uint32', 2),
    status: harness.output('uint32', 2)
  };
  const result = await harness.run(
    new GPUTrackSimilaritySearch({
      metric: 'longest-common-subsequence',
      lcssTolerance: 0,
      lcssBandRadius: 0,
      linesA: {
        kind: 'lines',
        positions: harness.input('float32x2', Float32Array.of(0, 0, 1, 0, 2, 0)),
        lineOffsets: harness.input('uint32', Uint32Array.of(0, 3))
      },
      linesB: {
        kind: 'lines',
        positions: harness.input('float32x2', Float32Array.of(2, 0, 10, 0)),
        lineOffsets: harness.input('uint32', Uint32Array.of(0, 1, 2))
      },
      pairA: harness.input('uint32', Uint32Array.of(0, 0)),
      pairB: harness.input('uint32', Uint32Array.of(0, 1)),
      threshold: harness.input('float32', Float32Array.of(1)),
      maximumFrechetVertexCount: 8,
      output
    })
  );
  expect(result(output.upperBound)).toEqual([1, 0]);
  expect(result(output.matched)).toEqual([1, 0]);
  expect(result(output.status)).toEqual([0, 1]);
  expect(result(output.similarity)[0]).toBe(1);
  expect(result(output.similarity)[1]).toBeNaN();
});

it('GPUTrajectoryKalman matches an independent NumPy 2.2.6 irregular-time forward and backward solve', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const reference = fixture.kalman;
  const count = reference.times.length;
  const output = {
    filteredState: harness.output('float32x4', count),
    filteredCovariance: harness.output('float32x4', count),
    status: harness.output('uint32', count),
    smoothedState: harness.output('float32x4', count),
    smoothedCovariance: harness.output('float32x4', count)
  };
  const result = await harness.run(
    new GPUTrajectoryKalman({
      positions: harness.input('float32x2', Float32Array.from(reference.positions.flat())),
      times: harness.input('float32', Float32Array.from(reference.times)),
      offsets: harness.input('uint32', new Uint32Array([0, count])),
      parameters: harness.input('float32', Float32Array.from(reference.parameters)),
      output
    })
  );
  for (const key of [
    'filteredState',
    'filteredCovariance',
    'smoothedState',
    'smoothedCovariance'
  ] as const)
    result(output[key]).forEach((value, index) =>
      expect(value, `${key}[${index}]`).toBeCloseTo(reference[key][index], 5)
    );
});

it('GPUTrajectoryKalman matches an independent two-observation Gaussian calculation and preserves filtered outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    filteredState: harness.output('float32x4', 2),
    filteredCovariance: harness.output('float32x4', 2),
    status: harness.output('uint32', 2),
    smoothedState: harness.output('float32x4', 2),
    smoothedCovariance: harness.output('float32x4', 2)
  };
  const result = await harness.run(
    new GPUTrajectoryKalman({
      positions: harness.input('float32x2', new Float32Array([0, 0, 2, 0])),
      times: harness.input('float32', new Float32Array([0, 1])),
      offsets: harness.input('uint32', new Uint32Array([0, 2])),
      parameters: harness.input('float32', new Float32Array([0, 1, 1, 10])),
      output
    })
  );
  const state = result(output.filteredState);
  expect(state[4]).toBeCloseTo(4 / 3, 6);
  expect(state[6]).toBeCloseTo(2 / 3, 6);
  expect(state[0]).toBe(0);
  const smoothed = result(output.smoothedState);
  expect(smoothed[0]).toBeCloseTo(2 / 3, 6);
  expect(smoothed[2]).toBeCloseTo(2 / 3, 6);
  const covariance = result(output.smoothedCovariance);
  expect(covariance[0]).toBeCloseTo(2 / 3, 6);
  expect(covariance[1]).toBeCloseTo(-1 / 3, 6);
  expect(covariance[1]).toBe(covariance[2]);
  expect(covariance[3]).toBeCloseTo(2 / 3, 6);
  expect(covariance[0] * covariance[3] - covariance[1] ** 2).toBeGreaterThan(0);
  expect(result(output.status)).toEqual([1, 0]);
});

it('GPUTrajectoryKalman predicts missing measurements and resets after gaps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    filteredState: harness.output('float32x4', 4),
    filteredCovariance: harness.output('float32x4', 4),
    status: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUTrajectoryKalman({
      positions: harness.input('float32x2', new Float32Array([0, 0, NaN, NaN, 20, 0, 30, 0])),
      times: harness.input('float32', new Float32Array([0, 1, 20, 19])),
      offsets: harness.input('uint32', new Uint32Array([0, 4])),
      parameters: harness.input('float32', new Float32Array([1, 1, 1, 5])),
      output
    })
  );
  expect(result(output.status)).toEqual([1, 2, 1, 4]);
  expect(result(output.filteredState).slice(8, 12)).toEqual([20, 0, 0, 0]);
  expect(result(output.filteredState)[12]).toBeNaN();
});

it('GPUTrajectoryRestriction clips a crossing segment and returns unknown for zero temporal support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 3),
    positions: harness.output('float32x4', 3),
    status: harness.output('uint32', 3),
    relationships: harness.output('uint32x4', 2)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', new Float32Array([-2, 0, 2, 0, 0, 0])),
      times: harness.input('float32', new Float32Array([0, 4, 20])),
      offsets: harness.input('uint32', new Uint32Array([0, 2, 3])),
      parameters: harness.input('float32', new Float32Array([0, 4, -1, -1, 1, 1, 5])),
      output
    })
  );
  expect(result(output.intervals).slice(0, 2)).toEqual([1, 3]);
  expect(result(output.positions).slice(0, 4)).toEqual([-1, 0, 1, 0]);
  expect(result(output.relationships).slice(0, 2)).toEqual([1, 0]);
  expect(result(output.relationships).slice(4, 6)).toEqual([0xffffffff, 0xffffffff]);
});

it('GPUTrajectoryRestriction does not interpolate through unsupported gaps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 2),
    positions: harness.output('float32x4', 2),
    status: harness.output('uint32', 2),
    relationships: harness.output('uint32x4', 1)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', new Float32Array([-2, 0, 2, 0])),
      times: harness.input('float32', new Float32Array([0, 10])),
      offsets: harness.input('uint32', new Uint32Array([0, 2])),
      parameters: harness.input('float32', new Float32Array([2, 8, -1, -1, 1, 1, 5])),
      output
    })
  );
  expect(result(output.status)[0]).toBe(8);
  expect(result(output.relationships).slice(0, 2)).toEqual([0xffffffff, 0xffffffff]);
});

it('GPUTrajectoryRestriction retains a closed measured endpoint beside an unsupported gap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 2),
    positions: harness.output('float32x4', 2),
    status: harness.output('uint32', 2),
    relationships: harness.output('uint32x4', 1)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', Float32Array.of(0, 0, 2, 0)),
      times: harness.input('float32', Float32Array.of(0, 10)),
      offsets: harness.input('uint32', Uint32Array.of(0, 2)),
      parameters: harness.input('float32', Float32Array.of(0, 0, -1, -1, 1, 1, 5)),
      output
    })
  );
  expect(result(output.intervals).slice(0, 2)).toEqual([0, 0]);
  expect(result(output.relationships)).toEqual([1, 1, 1, 0]);
});

for (const metric of ['dynamic-time-warping', 'frechet'] as const) {
  it(`GPUTrackSimilaritySearch ${metric} preserves threshold ties and prunes only impossible pairs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const output = {
      distance: harness.output('float32', 3),
      lowerBound: harness.output('float32', 3),
      matched: harness.output('uint32', 3),
      status: harness.output('uint32', 3)
    };
    const result = await harness.run(
      new GPUTrackSimilaritySearch({
        metric,
        linesA: {
          kind: 'lines',
          positions: harness.input('float32x2', new Float32Array([0, 0, 1, 0])),
          lineOffsets: harness.input('uint32', new Uint32Array([0, 2]))
        },
        linesB: {
          kind: 'lines',
          positions: harness.input('float32x2', new Float32Array([0, 0, 1, 0, 10, 0, 11, 0])),
          lineOffsets: harness.input('uint32', new Uint32Array([0, 2, 4]))
        },
        pairA: harness.input('uint32', new Uint32Array([0, 0, 9])),
        pairB: harness.input('uint32', new Uint32Array([0, 1, 0])),
        threshold: harness.input('float32', new Float32Array([0])),
        maximumFrechetVertexCount: 8,
        output
      })
    );
    expect(result(output.status)).toEqual([0, 1, 2]);
    expect(result(output.matched)).toEqual([1, 0, 0]);
    expect(result(output.distance)[0]).toBe(0);
    expect(result(output.distance)[1]).toBeNaN();
    expect(result(output.lowerBound)[1]).toBe(10);
  });
}

/** Epoch milliseconds a little more than a year after 2023-01-01 UTC. */
const YEAR_MS = 1_672_531_200_000n + 400n * 86_400_000n;

it('prepareGPUTrajectoryObservations returns exact words beside the collapsing f32 seconds', () => {
  const count = 30;
  const timestamps = Array.from({length: count}, (_, row) => YEAR_MS + BigInt(row * 100));
  const prepared = prepareGPUTrajectoryObservations({
    longitudeLatitude: new Float64Array(count * 2),
    timestamps,
    epoch: 1_672_531_200_000n,
    unitsPerSecond: 1000n,
    origin: [0, 0]
  });
  expect(prepared.timeWords.length).toBe(count * 2);
  for (let row = 0; row < count; row++) {
    const words = splitTimeWords(timestamps[row]);
    expect([prepared.timeWords[row * 2], prepared.timeWords[row * 2 + 1]]).toEqual([
      words.low,
      words.high
    ]);
  }
  // The f32 seconds of samples 100 ms apart a year from the epoch cannot resolve the step.
  expect(new Set(prepared.times).size).toBeLessThan(count);
});

it('GPUTrajectoryKalman keeps 10 Hz steps with word timestamps more than a year from the epoch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 30;
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) positions[row * 2] = row * 0.1;
  const run = async (times: Float32Array | Uint32Array, wordMode: boolean, maximumGap: number) => {
    const harness = createOptimizationHarness(device);
    const output = {
      filteredState: harness.output('float32x4', count),
      filteredCovariance: harness.output('float32x4', count),
      status: harness.output('uint32', count),
      smoothedState: harness.output('float32x4', count),
      smoothedCovariance: harness.output('float32x4', count)
    };
    const result = await harness.run(
      new GPUTrajectoryKalman({
        positions: harness.input('float32x2', positions),
        times: wordMode
          ? harness.input('uint32x2', times as Uint32Array)
          : harness.input('float32', times as Float32Array),
        timeUnitsPerSecond: wordMode ? 1000 : undefined,
        offsets: harness.input('uint32', new Uint32Array([0, count])),
        parameters: harness.input('float32', new Float32Array([0.01, 0.01, 1, maximumGap])),
        output
      })
    );
    return {
      state: result(output.filteredState),
      smoothed: result(output.smoothedState),
      status: result(output.status)
    };
  };
  const words = new Uint32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const split = splitTimeWords(YEAR_MS + BigInt(row * 100));
    words[row * 2] = split.low;
    words[row * 2 + 1] = split.high;
  }
  // maximumGap is in the timestamps' unit: 10 s is 10000 ms for words and 10 for f32 seconds.
  const wordResult = await run(words, true, 10000);
  const reference = await run(
    Float32Array.from({length: count}, (_, row) => row * 0.1),
    false,
    10
  );
  wordResult.state.forEach((value, index) =>
    expect(value, `filtered[${index}]`).toBeCloseTo(reference.state[index], 3)
  );
  wordResult.smoothed.forEach((value, index) =>
    expect(value, `smoothed[${index}]`).toBeCloseTo(reference.smoothed[index], 3)
  );
  expect(wordResult.status).toEqual(reference.status);
  expect(wordResult.state[4 * (count - 1) + 2]).toBeCloseTo(1, 1);
  // The same samples as f32 seconds a year from the epoch collapse to zero-length steps.
  const collapsed = await run(
    Float32Array.from({length: count}, (_, row) => 34_560_000 + row * 0.1),
    false,
    10
  );
  expect(Math.abs(collapsed.state[4 * (count - 1) + 2])).toBeLessThan(0.5);
});

it('GPUTrajectoryRestriction word timestamps match near-origin f32 results for every variant', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 12;
  const positions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) positions[row * 2] = row * 0.5 + 0.25;
  const scenarios = [
    {name: 'rectangle', parameters: [0, 1100, 1.1, -1, 2.4, 1, 500], extra: {}},
    {
      name: 'polygon',
      parameters: [0, 1100, 0, 0, 0, 0, 500, 1.1, -1, 2.4, -1, 2.4, 1, 1.1, 1],
      extra: {polygonRingVertexCounts: [4]}
    },
    {
      name: 'pairwise',
      parameters: [0, 1100, 0.6, 0, 0, 0, 500],
      extra: {pairwiseTracks: [[0, 1] as const], intervalsPerPair: 4},
      offsets: [0, 6, 12]
    }
  ];
  for (const scenario of scenarios) {
    const offsets = scenario.offsets ?? [0, rowCount];
    const run = async (wordMode: boolean) => {
      const harness = createOptimizationHarness(device);
      const intervalRows = scenario.extra.pairwiseTracks ? 4 : rowCount;
      const output = {
        intervals: harness.output('float32x2', intervalRows),
        positions: harness.output('float32x4', intervalRows),
        status: harness.output('uint32', intervalRows),
        relationships: harness.output(
          'uint32x4',
          scenario.extra.pairwiseTracks ? 1 : offsets.length - 1
        )
      };
      const words = new Uint32Array(rowCount * 2);
      for (let row = 0; row < rowCount; row++) {
        const split = splitTimeWords(YEAR_MS + BigInt(row * 100));
        words[row * 2] = split.low;
        words[row * 2 + 1] = split.high;
      }
      const origin = splitTimeWords(YEAR_MS);
      const result = await harness.run(
        new GPUTrajectoryRestriction({
          ...scenario.extra,
          positions: harness.input('float32x2', positions),
          times: wordMode
            ? harness.input('uint32x2', words)
            : harness.input(
                'float32',
                Float32Array.from({length: rowCount}, (_, row) => row * 100)
              ),
          timeOrigin: wordMode
            ? harness.input('uint32', Uint32Array.of(origin.low, origin.high))
            : undefined,
          offsets: harness.input('uint32', Uint32Array.from(offsets)),
          parameters: harness.input('float32', Float32Array.from(scenario.parameters)),
          output
        })
      );
      return {
        intervals: result(output.intervals),
        positions: result(output.positions),
        status: result(output.status),
        relationships: result(output.relationships)
      };
    };
    const reference = await run(false);
    expect(
      reference.relationships.some(value => value !== 0),
      scenario.name
    ).toBe(true);
    expect(await run(true), scenario.name).toEqual(reference);
  }
});

it('GPUTrajectoryKalman reads the trajectory port and takes its unit from the temporal context', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 12;
  const positions = new Float32Array(count * 2);
  const words = new Uint32Array(count * 2);
  for (let row = 0; row < count; row++) {
    positions[row * 2] = row * 0.1;
    const split = splitTimeWords(YEAR_MS + BigInt(row * 100));
    words[row * 2] = split.low;
    words[row * 2 + 1] = split.high;
  }
  const run = async (
    build: (harness: ReturnType<typeof createOptimizationHarness>) => Record<string, unknown>
  ) => {
    const harness = createOptimizationHarness(device);
    const output = {
      filteredState: harness.output('float32x4', count),
      filteredCovariance: harness.output('float32x4', count),
      status: harness.output('uint32', count)
    };
    const result = await harness.run(
      new GPUTrajectoryKalman({
        positions: harness.input('float32x2', positions),
        parameters: harness.input('float32', new Float32Array([0.01, 0.01, 1, 10000])),
        ...build(harness),
        output
      } as ConstructorParameters<typeof GPUTrajectoryKalman>[0])
    );
    return {state: result(output.filteredState), status: result(output.status)};
  };
  const legacy = await run(harness => ({
    times: harness.input('uint32x2', words),
    offsets: harness.input('uint32', new Uint32Array([0, count])),
    timeUnitsPerSecond: 1000
  }));
  const port = await run(harness => ({
    timestamps: harness.input('uint32x2', words),
    trackOffsets: harness.input('uint32', new Uint32Array([0, count])),
    temporalContext: {
      unit: 'millisecond',
      epoch: 'unix',
      encoding: 'int64-words',
      axis: {kind: 'irregular'}
    }
  }));
  expect(port).toEqual(legacy);
  expect(port.status.every(flags => (flags & 4) === 0)).toBe(true);
});

it('GPUTrajectoryRestriction reads the trajectory port shape', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 6;
  const positions = Float32Array.from({length: rows * 2}, (_, index) =>
    index % 2 ? 0 : index * 0.25
  );
  const run = async (port: boolean) => {
    const harness = createOptimizationHarness(device);
    const output = {
      intervals: harness.output('float32x2', rows),
      positions: harness.output('float32x4', rows),
      status: harness.output('uint32', rows),
      relationships: harness.output('uint32x4', 1)
    };
    const times = harness.input(
      'float32',
      Float32Array.from({length: rows}, (_, row) => row * 10)
    );
    const offsets = harness.input('uint32', Uint32Array.of(0, rows));
    const result = await harness.run(
      new GPUTrajectoryRestriction({
        positions: harness.input('float32x2', positions),
        ...(port ? {timestamps: times, trackOffsets: offsets} : {times, offsets}),
        parameters: harness.input('float32', Float32Array.of(0, 50, 0.3, -1, 1.2, 1, 20)),
        output
      } as ConstructorParameters<typeof GPUTrajectoryRestriction>[0])
    );
    return {
      intervals: result(output.intervals),
      status: result(output.status),
      relationships: result(output.relationships)
    };
  };
  const legacy = await run(false);
  expect(legacy.relationships[0]).toBe(1);
  expect(await run(true)).toEqual(legacy);
});

it('GPUTrajectoryKalman maximumGap 0 means no limit in both covariance modes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 3;
  for (const covarianceMode of ['isotropic', 'full'] as const) {
    const run = async (maximumGap: number) => {
      const harness = createOptimizationHarness(device);
      const output = {
        filteredState: harness.output('float32x4', count),
        filteredCovariance: harness.output(
          'float32x4',
          covarianceMode === 'full' ? count * 4 : count
        ),
        status: harness.output('uint32', count)
      };
      const parameters =
        covarianceMode === 'full'
          ? [0.1, 0, 0.1, 1, 0, 1, 1, 0, 1, maximumGap]
          : [0.1, 1, 1, maximumGap];
      const result = await harness.run(
        new GPUTrajectoryKalman({
          covarianceMode,
          positions: harness.input('float32x2', Float32Array.of(0, 0, 1, 0, 2, 0)),
          times: harness.input('float32', Float32Array.of(0, 1, 1000)),
          offsets: harness.input('uint32', Uint32Array.of(0, count)),
          parameters: harness.input('float32', Float32Array.from(parameters)),
          output
        })
      );
      return {state: result(output.filteredState), status: result(output.status)};
    };
    const unlimited = await run(1e30);
    // The 999-unit step is bridged without a reset.
    expect(unlimited.status).toEqual([1, 0, 0]);
    expect(await run(0), covarianceMode).toEqual(unlimited);
    // A positive limit still resets at that step.
    expect((await run(10)).status).toEqual([1, 0, 1]);
  }
});

it('GPUTrajectoryRestriction maximumGap 0 means no limit in every variant', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 4;
  const positions = Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0);
  const times = Float32Array.of(0, 10, 1000, 1010);
  const scenarios = [
    {name: 'rectangle', head: [0, 1010, 0.5, -1, 2.5, 1], tail: [], extra: {}, rows: rowCount},
    {
      name: 'polygon',
      head: [0, 1010, 0, 0, 0, 0],
      tail: [0.5, -1, 2.5, -1, 2.5, 1, 0.5, 1],
      extra: {polygonRingVertexCounts: [4]},
      rows: rowCount
    },
    {
      name: 'pairwise',
      head: [0, 1010, 5, 0, 0, 0],
      tail: [],
      extra: {pairwiseTracks: [[0, 1] as const], intervalsPerPair: 8},
      rows: 8
    }
  ];
  for (const scenario of scenarios) {
    const pairwise = scenario.name === 'pairwise';
    const run = async (maximumGap: number) => {
      const harness = createOptimizationHarness(device);
      const output = {
        intervals: harness.output('float32x2', scenario.rows),
        positions: harness.output('float32x4', scenario.rows),
        status: harness.output('uint32', scenario.rows),
        relationships: harness.output('uint32x4', 1)
      };
      const result = await harness.run(
        new GPUTrajectoryRestriction({
          ...scenario.extra,
          positions: harness.input(
            'float32x2',
            pairwise ? Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1, 1, 2, 1, 3, 1) : positions
          ),
          times: harness.input('float32', pairwise ? Float32Array.of(...times, ...times) : times),
          offsets: harness.input(
            'uint32',
            pairwise ? Uint32Array.of(0, rowCount, 2 * rowCount) : Uint32Array.of(0, rowCount)
          ),
          parameters: harness.input(
            'float32',
            Float32Array.from([...scenario.head, maximumGap, ...scenario.tail])
          ),
          output
        } as ConstructorParameters<typeof GPUTrajectoryRestriction>[0])
      );
      return {
        intervals: result(output.intervals),
        status: result(output.status),
        relationships: result(output.relationships)
      };
    };
    const unlimited = await run(1e30);
    expect(
      unlimited.status.some(flags => (flags & 4) !== 0),
      scenario.name
    ).toBe(false);
    expect(await run(0), scenario.name).toEqual(unlimited);
    // A positive limit still reports the 990-unit step as an unsupported gap.
    expect(
      (await run(100)).status.some(flags => (flags & 8) !== 0),
      scenario.name
    ).toBe(true);
  }
});
