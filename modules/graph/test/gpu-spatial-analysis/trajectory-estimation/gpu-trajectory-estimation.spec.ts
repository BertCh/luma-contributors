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
