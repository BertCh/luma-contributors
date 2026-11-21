// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {prepareGPUTrajectoryObservations} from '../../../src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations';
import {computeTrajectoryKalmanOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation';
import {
  resolveGPUSpatialContext,
  getGPUSpatialContextCompatibility,
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
} from '../../../src/analysis-core/contracts/spatial-context';

it('B5 equal tangent origins compose and different origins are incompatible', () => {
  const prepare = (origin: [number, number]) =>
    prepareGPUTrajectoryObservations({
      longitudeLatitude: [30, 0],
      timestamps: [0n],
      epoch: 0n,
      unitsPerSecond: 1n,
      origin
    });
  const first = prepare([30, 0]);
  const same = prepare([30, 0]);
  const other = prepare([31, 0]);
  expect(getGPUSpatialContextCompatibility(first.spatialContext, same.spatialContext)).toEqual({
    compatible: true
  });
  expect(
    getGPUSpatialContextCompatibility(first.spatialContext, other.spatialContext).compatible
  ).toBe(false);
});

it('UNI-15 ING-07 captures the projection origin and separates seconds from absolute ticks', () => {
  const origin: [number, number] = [0, 0];
  const result = prepareGPUTrajectoryObservations({
    longitudeLatitude: [0, 0, 0.000135, 0],
    timestamps: [1000500n, 1001500n],
    epoch: 1000500n,
    unitsPerSecond: 1000n,
    origin
  });
  origin[0] = 90;
  expect(result.projection.origin).toEqual([0, 0]);
  expect(Object.isFrozen(result.projection.origin)).toBe(true);
  expect(Array.from(result.times)).toEqual([0, 1]);
  expect(Array.from(result.timeWords)).toEqual([1000500, 0, 1001500, 0]);
  expect(result.timesMetadata).toEqual({
    encoding: 'float32-relative',
    unit: 'second',
    epochTicks: 1000500n,
    epochUnitsPerSecond: 1000n
  });
  expect(result.timeWordsTemporalContext).toEqual({
    encoding: 'int64-words',
    unit: 'millisecond',
    epoch: 'unix',
    axis: {kind: 'irregular'}
  });
  // A fractional-second epoch cannot be represented by GPUTemporalContext's integer-second origin.
  expect(result.timesTemporalContext).toBeUndefined();
  expect(
    resolveGPUSpatialContext('prepared', result.spatialContext, GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT)
      .units
  ).toBe('meters');
  const input = {
    positions: result.positions,
    trackOffsets: Uint32Array.of(0, 2),
    parameters: Float32Array.of(0, 1, 1, 0)
  };
  const seconds = computeTrajectoryKalmanOnCPU({...input, timestamps: result.times});
  const words = computeTrajectoryKalmanOnCPU(
    {...input, timestamps: result.timeWords},
    {temporalContext: result.timeWordsTemporalContext}
  );
  // WGS84 tangent displacement is ~15.028131m; K_velocity=1/3 at dt=1s.
  for (const filtered of [seconds.filteredState, words.filteredState]) {
    expect(Number.isFinite(filtered[6])).toBe(true);
    expect(filtered[6]).toBeCloseTo(5.009377, 5);
  }
  expect(words.filteredState).toEqual(seconds.filteredState);
});

it('UNI-15 exposes a seconds context only for an exactly integral seconds epoch', () => {
  const result = prepareGPUTrajectoryObservations({
    longitudeLatitude: [],
    timestamps: [],
    epoch: 2000n,
    unitsPerSecond: 1000n,
    origin: [0, 0]
  });
  expect(result.timesTemporalContext).toEqual({
    encoding: 'float32-relative',
    unit: 'second',
    epoch: 2n,
    axis: {kind: 'irregular'}
  });
  const custom = prepareGPUTrajectoryObservations({
    longitudeLatitude: [],
    timestamps: [],
    epoch: 1n,
    unitsPerSecond: 3n,
    origin: [0, 0]
  });
  expect(custom.timeWordsTemporalContext).toBeUndefined();
  expect(custom.timesMetadata.epochUnitsPerSecond).toBe(3n);
});
