// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectoryKalmanOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation';

const observations = {
  positions: Float32Array.of(2, 4, 6, 8),
  timestamps: Float32Array.of(0, 0),
  parameters: Float32Array.of(0, 2, 0, 0)
};

it('requires exact offsets for compact host Kalman observations', () => {
  const run = (trackOffsets: Uint32Array) =>
    computeTrajectoryKalmanOnCPU({...observations, trackOffsets, storage: 'compact'});
  expect(() => run(Uint32Array.of(0, 99))).toThrow(/trackOffsets.*2.*99/);
  expect(() => run(Uint32Array.of(0, 1))).toThrow(/trackOffsets.*terminal.*2/);
  expect(() => run(Uint32Array.of(1, 2))).toThrow(/trackOffsets.*initial.*0/);
  expect(() => run(Uint32Array.of(0))).toThrow(/trackOffsets.*terminal.*2/);

  const result = run(Uint32Array.of(0, 2));
  // Same timestamp, no process noise, and equal measurement variance give the mean.
  expect(Array.from(result.filteredState)).toEqual([2, 4, 0, 0, 4, 6, 0, 0]);

  const empty = computeTrajectoryKalmanOnCPU({
    positions: new Float32Array(),
    timestamps: new Float32Array(),
    trackOffsets: Uint32Array.of(0),
    parameters: observations.parameters,
    storage: 'compact'
  });
  expect(empty.filteredState).toHaveLength(0);
});

it('clamps resident Kalman offsets by default and when requested explicitly', () => {
  const input = {...observations, trackOffsets: Uint32Array.of(0, 99)};
  const explicit = computeTrajectoryKalmanOnCPU({...input, storage: 'resident-capacity'});
  expect(Array.from(explicit.filteredState)).toEqual([2, 4, 0, 0, 4, 6, 0, 0]);
  expect(computeTrajectoryKalmanOnCPU(input)).toEqual(explicit);
});
